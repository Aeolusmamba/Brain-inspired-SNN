"""Low-dimensional BDM-SNN baseline for robosuite's Panda Lift task.

This is deliberately separate from the Flappy Bird examples.  It uses the
simulator's low-dimensional state in the first stage: no camera pixels, CNN,
communication compression, or continuous torque policy is used here.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# EGL enables off-screen MuJoCo rendering on the shared Linux machine.  A user
# can override it before launching the script for a different render backend.
os.environ.setdefault("MUJOCO_GL", "egl")

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

from braincog.model_zoo.bdmsnn import BDMSNN


BASE_NUM_STATE = 64
NUM_PHASE_CONTEXT = 4
NUM_ACTION_DURATION_BINS = 3
# Physical actions remain eight-wide because the deployment FSM still issues
# descend / close / lift / recover primitives.  The align SNN can use either
# all eight output channels (legacy baseline) or only the four lateral ones.
NUM_ACTION = 8
NUM_ALIGN_ACTION = 4
TEMPORAL_ACTION_SYMBOLS = NUM_ACTION + 1
OPTION_NAMES = ("align", "descend", "close", "lift", "recover")
NUM_OPTION_CONTEXT = len(OPTION_NAMES)
ALIGN_GRID_BINS = 5
ALIGN_GRID_STATES = ALIGN_GRID_BINS * ALIGN_GRID_BINS
# The legacy default has ten 10-mm in-range bins plus two overflow bins.
ALIGN_RESIDUAL_BINS = 12
ALIGN_MEMORY_SYMBOLS = 5  # none, +x, -x, +y, -y
OPTION_MEMORY_BASE_STATES = 128
FACTORIZED_ALIGN_STATE_COUNT = ALIGN_GRID_STATES + ALIGN_MEMORY_SYMBOLS + 4 * BASE_NUM_STATE
ACTION_NAMES = (
    "+x", "-x", "+y", "-y", "+z", "-z", "gripper_open", "gripper_close",
)
ALIGN_ACTION_NAMES = ACTION_NAMES[:NUM_ALIGN_ACTION]
REGION_NAMES = ("StrD1", "StrD2", "STN", "GPe", "GPi", "thalamus", "PM", "DLPFC")


def set_seed(seed):
    """Set all random generators used by the online control loop."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LiftStateEncoder:
    """Map low-dimensional Lift state to 8 direction x 4 range x 2 grip bins."""

    def __init__(self, distance_edges=(0.04, 0.08, 0.14), use_phase_context=False,
                 use_action_duration=False, use_option_context=False,
                 use_align_grid=False, use_progress_memory_context=False,
                 use_factorized_progress_memory=False, use_align_residual_grid=False,
                 use_align_residual_axis=False, align_residual_bins=ALIGN_RESIDUAL_BINS,
                 snn_align_only=False):
        self.distance_edges = np.asarray(distance_edges, dtype=np.float64)
        self.use_phase_context = use_phase_context
        self.use_action_duration = use_action_duration
        self.use_option_context = use_option_context
        self.use_align_grid = use_align_grid
        self.use_progress_memory_context = use_progress_memory_context
        self.use_factorized_progress_memory = use_factorized_progress_memory
        self.use_align_residual_grid = use_align_residual_grid
        self.use_align_residual_axis = use_align_residual_axis
        if align_residual_bins < 4:
            raise ValueError("align_residual_bins must retain at least two in-range bins and two overflow bins")
        self.align_residual_bins = align_residual_bins
        self.snn_align_only = snn_align_only

    @property
    def align_residual_states(self):
        return self.align_residual_bins * self.align_residual_bins

    @property
    def align_residual_axis_states(self):
        return 2 * self.align_residual_states

    @property
    def num_state(self):
        if self.use_align_residual_axis:
            if not self.use_option_context or self.use_action_duration:
                raise ValueError("align residual-axis code requires option context without duration context")
            return (self.align_residual_axis_states if self.snn_align_only else
                    self.align_residual_axis_states + (NUM_OPTION_CONTEXT - 1) * BASE_NUM_STATE)
        if self.use_align_residual_grid:
            if not self.use_option_context or self.use_action_duration:
                raise ValueError("align residual grid requires option context without duration context")
            return (self.align_residual_states if self.snn_align_only else
                    self.align_residual_states + (NUM_OPTION_CONTEXT - 1) * BASE_NUM_STATE)
        if self.use_factorized_progress_memory:
            return FACTORIZED_ALIGN_STATE_COUNT
        if self.use_progress_memory_context:
            # Align uses 25 xy cells x five memory symbols. The other four
            # options retain their 64 base states in separate 128-state blocks.
            return NUM_OPTION_CONTEXT * OPTION_MEMORY_BASE_STATES
        size = BASE_NUM_STATE * (NUM_PHASE_CONTEXT if self.use_phase_context else 1)
        if self.use_option_context:
            size *= NUM_OPTION_CONTEXT
        if self.use_action_duration:
            size *= TEMPORAL_ACTION_SYMBOLS * NUM_ACTION_DURATION_BINS
        return size

    @staticmethod
    def direction_sector(relative_position):
        # Three sign bits encode the cube's octant relative to the end effector.
        bits = (np.asarray(relative_position) >= 0.0).astype(np.int64)
        return int(bits[0] | (bits[1] << 1) | (bits[2] << 2))

    def residual_bin(self, error):
        """Use two overflow bins and evenly partition the +/-50-mm operating range."""
        if error < -0.050:
            return 0
        if error >= 0.050:
            return self.align_residual_bins - 1
        return 1 + int(np.floor((error + 0.050) /
                                (0.100 / (self.align_residual_bins - 2))))

    def encode(self, observation, grasped, previous_action=None, action_duration=0,
               option_index=0, progress_memory_action=None):
        relative_position = observation["cube_pos"] - observation["robot0_eef_pos"]
        direction = self.direction_sector(relative_position)
        distance_bin = int(np.digitize(np.linalg.norm(relative_position), self.distance_edges))
        # The single binary mode represents a closed gripper or an actual grasp.
        gripper_closed = bool(grasped or np.max(np.abs(observation["robot0_gripper_qpos"])) < 0.012)
        base_state = ((direction * 4) + distance_bin) * 2 + int(gripper_closed)
        if self.use_align_residual_axis and option_index == 0:
            xy_error = relative_position[:2] - np.array((-0.020, 0.0))
            grid = np.asarray([self.residual_bin(value) for value in xy_error], dtype=np.int64)
            dominant_axis = int(abs(xy_error[1]) > abs(xy_error[0]))
            base_state = int(grid[0] + self.align_residual_bins * grid[1] +
                             dominant_axis * self.align_residual_states)
        elif self.use_align_residual_grid and option_index == 0:
            xy_error = relative_position[:2] - np.array((-0.020, 0.0))
            grid = np.asarray([self.residual_bin(value) for value in xy_error], dtype=np.int64)
            base_state = int(grid[0] + self.align_residual_bins * grid[1])
        elif self.use_align_grid and option_index == 0:
            # Align only needs lateral error.  Five bins per axis preserve
            # direction and error magnitude without expanding the 320-state
            # option input population.
            xy_error = relative_position[:2] - np.array((-0.020, 0.0))
            grid = np.clip(np.floor((xy_error + 0.050) / 0.020),
                           0, ALIGN_GRID_BINS - 1).astype(np.int64)
            base_state = int(grid[0] + ALIGN_GRID_BINS * grid[1])
        # This context comes only from observable geometry and gripper state,
        # never from the teacher's hidden phase.  It splits aliased states such
        # as "near and above the cube" before closing versus while lifting.
        xy_distance = float(np.linalg.norm(relative_position[:2]))
        if gripper_closed:
            phase_context = 3  # lift / carry
        elif xy_distance > 0.035:
            phase_context = 0  # reach laterally
        elif relative_position[2] < -0.02:
            phase_context = 1  # descend
        else:
            phase_context = 2  # close at contact
        if self.use_align_residual_grid or self.use_align_residual_axis:
            if self.use_progress_memory_context or self.use_factorized_progress_memory:
                raise ValueError("align residual encoders cannot be combined with progress-memory encoders")
            align_states = (self.align_residual_axis_states if self.use_align_residual_axis
                            else self.align_residual_states)
            if option_index != 0:
                if self.snn_align_only:
                    # This value is never sent through the SNN: fixed-action
                    # options are handled by the deployment FSM alone.
                    return None, {"option_index": option_index, "snn_active": False}
                # Residual cells occupy the first block; remaining
                # options retain their original 64-state local encodings.
                state = align_states + (option_index - 1) * BASE_NUM_STATE + base_state
            else:
                state = base_state
            base_size = self.num_state
        elif self.use_factorized_progress_memory:
            if not self.use_option_context or not self.use_align_grid or self.use_action_duration:
                raise ValueError("factorized progress memory requires option/xy-grid contexts without duration context")
            if option_index == 0:
                memory_symbol = 0 if progress_memory_action is None else int(progress_memory_action) + 1
                if not 0 <= memory_symbol < ALIGN_MEMORY_SYMBOLS:
                    raise ValueError("progress memory action must be one of the four align actions")
                # Two simultaneous sensory populations: one xy cell and one
                # direction-memory cell. Their action weights are shared
                # across all pairings instead of forming a 25 x 5 lookup table.
                state = (base_state, ALIGN_GRID_STATES + memory_symbol)
            else:
                state = (ALIGN_GRID_STATES + ALIGN_MEMORY_SYMBOLS +
                         (option_index - 1) * BASE_NUM_STATE + base_state,)
            base_size = self.num_state
        elif self.use_progress_memory_context:
            if not self.use_option_context or not self.use_align_grid:
                raise ValueError("progress memory context requires option and xy-grid contexts")
            if option_index == 0:
                memory_symbol = 0 if progress_memory_action is None else int(progress_memory_action) + 1
                if not 0 <= memory_symbol < ALIGN_MEMORY_SYMBOLS:
                    raise ValueError("progress memory action must be one of the four align actions")
                state = memory_symbol * ALIGN_GRID_STATES + base_state
            else:
                state = option_index * OPTION_MEMORY_BASE_STATES + base_state
            base_size = self.num_state
        else:
            state = (phase_context * BASE_NUM_STATE + base_state
                     if self.use_phase_context else base_state)
            base_size = BASE_NUM_STATE * (NUM_PHASE_CONTEXT if self.use_phase_context else 1)
            if self.use_option_context:
                if not 0 <= option_index < NUM_OPTION_CONTEXT:
                    raise ValueError("option index is outside the observable option alphabet")
                state = option_index * base_size + state
                base_size *= NUM_OPTION_CONTEXT
        if self.use_action_duration:
            action_symbol = NUM_ACTION if previous_action is None else previous_action
            if not 0 <= action_symbol <= NUM_ACTION:
                raise ValueError("previous action is outside the temporal encoder alphabet")
            duration_bin = 0 if action_duration <= 1 else 1 if action_duration <= 5 else 2
            state = ((action_symbol * NUM_ACTION_DURATION_BINS + duration_bin) *
                     base_size + state)
        return state, {
            "relative_position": relative_position,
            "direction_sector": direction,
            "distance_bin": distance_bin,
            "gripper_closed_or_grasped": gripper_closed,
            "phase_context": phase_context,
            "previous_action": previous_action,
            "action_duration": action_duration,
            "option_index": option_index,
            "align_grid": (base_state if (self.use_align_grid or self.use_align_residual_grid or
                           self.use_align_residual_axis)
                           and option_index == 0 else None),
            "progress_memory_action": progress_memory_action,
            "snn_active": True,
        }


class LiftActionPrimitives:
    """Translate the physical eight-action alphabet into Panda OSC commands."""

    def __init__(self, magnitude=0.25):
        self.magnitude = magnitude

    def command(self, action, keep_gripper_closed=False):
        if not 0 <= action < NUM_ACTION:
            raise ValueError("action must be in [0, 7]")
        command = np.zeros(7, dtype=np.float64)
        if action < 6:
            axis = action // 2
            command[axis] = self.magnitude if action % 2 == 0 else -self.magnitude
        elif action == 6:
            command[6] = -1.0  # PandaGripper: -1 opens, +1 closes.
        else:
            command[6] = 1.0
        if keep_gripper_closed:
            # The gripper integrator needs a close command while a Cartesian
            # primitive is being executed; otherwise it continues closing or
            # opening according to a stale action state.
            command[6] = 1.0
        return command


class LiftOptionController:
    """Observable deployment-time task sequencer, not a teacher policy.

    It exposes the manipulation preconditions as five finite options and only
    gates infeasible primitives.  It uses geometry, grasp detection, and
    bounded counters available at deployment; it never queries teacher state
    or provides an action label to the SNN.
    """

    def __init__(self, contact_offset=(-0.020, 0.0), xy_tolerance=0.002,
                 descend_target=-0.01, close_steps=2, failed_lift_steps=4,
                 recover_open_steps=2):
        self.contact_offset = np.asarray(contact_offset, dtype=np.float64)
        self.xy_tolerance = xy_tolerance
        self.descend_target = descend_target
        self.close_steps = close_steps
        self.failed_lift_steps = failed_lift_steps
        self.recover_open_steps = recover_open_steps
        self.reset()

    def reset(self):
        self.option = 0
        self.close_count = 0
        self.lift_count = 0
        self.recover_count = 0

    def update_from_observation(self, observation, grasped):
        relative = observation["cube_pos"] - observation["robot0_eef_pos"]
        if self.option == 0:
            xy_error = relative[:2] - self.contact_offset
            if np.linalg.norm(xy_error) <= self.xy_tolerance:
                self.option = 1
        elif self.option == 1 and relative[2] >= self.descend_target:
            self.option = 2
            self.close_count = 0
        elif self.option == 3 and not grasped and self.lift_count >= self.failed_lift_steps:
            self.option = 4
            self.recover_count = 0
        return self.option

    def allowed_actions(self):
        return ((0, 1, 2, 3), (5,), (7,), (4,), (6,))[self.option]

    def keep_gripper_closed(self):
        return self.option == 3

    def after_action(self, action):
        if self.option == 2 and action == 7:
            self.close_count += 1
            if self.close_count >= self.close_steps:
                self.option = 3
                self.lift_count = 0
        elif self.option == 3 and action == 4:
            self.lift_count += 1
        elif self.option == 4 and action == 6:
            self.recover_count += 1
            if self.recover_count >= self.recover_open_steps:
                self.option = 0

    def teacher_action(self, observation):
        """Return a label only while a teacher curriculum is explicitly active."""
        if self.option == 0:
            relative = observation["cube_pos"] - observation["robot0_eef_pos"]
            xy_error = relative[:2] - self.contact_offset
            if abs(xy_error[0]) >= abs(xy_error[1]):
                return 0 if xy_error[0] > 0 else 1
            return 2 if xy_error[1] > 0 else 3
        return self.allowed_actions()[0]


class LiftTeacher:
    """A discrete, contact-aware curriculum demonstrator using the same 8 actions.

    It is not the evaluated autonomous policy.  It establishes that the state
    encoding and action primitives admit a successful Lift trajectory, and
    supplies sparse reward during the initial online R-STDP curriculum.
    """

    def __init__(self, contact_offset=(-0.020, 0.0), xy_tolerance=0.002,
                 descend_target=-0.01, close_decisions=1):
        self.contact_offset = np.asarray(contact_offset, dtype=np.float64)
        self.xy_tolerance = xy_tolerance
        self.descend_target = descend_target
        self.close_decisions = close_decisions
        self.phase = "approach"
        self.close_count = 0
        self.lift_count = 0
        self.recovery_count = 0

    def reset(self):
        self.phase = "approach"
        self.close_count = 0
        self.lift_count = 0
        self.recovery_count = 0

    def action(self, observation, grasped):
        relative = observation["cube_pos"] - observation["robot0_eef_pos"]
        if self.phase == "approach":
            xy_error = relative[:2] - self.contact_offset
            if np.linalg.norm(xy_error) <= self.xy_tolerance:
                self.phase = "descend"
            else:
                if abs(xy_error[0]) >= abs(xy_error[1]):
                    return (0 if xy_error[0] > 0 else 1), False
                return (2 if xy_error[1] > 0 else 3), False
        if self.phase == "descend":
            if relative[2] < self.descend_target:
                return 5, False
            self.phase = "close"
        if self.phase == "close":
            # A one-step close followed by an immediate upward motion is not
            # contact-stable for every MuJoCo reset.  Hold the same discrete
            # close primitive before lifting; this does not add an action type.
            self.close_count += 1
            if self.close_count < self.close_decisions:
                return 7, False
            self.phase = "lift"
            self.lift_count = 0
            return 7, False
        if self.phase == "lift":
            self.lift_count += 1
            # A failed close often pushes the cube sideways.  Reopen and
            # reacquire instead of continuing an unproductive upward motion.
            if not grasped and self.lift_count >= 4:
                self.phase = "recover_open"
                self.recovery_count = 0
                return 6, False
            return 4, True
        if self.phase == "recover_open":
            self.recovery_count += 1
            if self.recovery_count >= 2:
                self.phase = "approach"
            return 6, False
        raise RuntimeError("unknown teacher phase: " + self.phase)


class ActionDurationMemory:
    """Episode-local executed-action history for temporal state encoding."""

    def __init__(self):
        self.previous_action = None
        self.duration = 0

    def update(self, action):
        if action == self.previous_action:
            self.duration += 1
        else:
            self.previous_action = int(action)
            self.duration = 1


class CoverageAudit:
    """Observe teacher support and student shift without affecting control."""

    def __init__(self, enabled, distance_threshold):
        self.enabled = enabled
        self.distance_threshold = distance_threshold
        self.teacher_labels = defaultdict(Counter)
        self.teacher_features = defaultdict(list)
        self.teacher_histograms = defaultdict(list)
        self.student_records = defaultdict(list)
        self.initial_teacher_labels = None
        self.initial_teacher_features = None

    def freeze_initial_teacher_support(self):
        """Snapshot pre-DAgger demonstrations for an honest shift measure."""
        if not self.enabled or self.initial_teacher_labels is not None:
            return
        self.initial_teacher_labels = {key: Counter(value) for key, value in self.teacher_labels.items()}
        self.initial_teacher_features = {key: list(value) for key, value in self.teacher_features.items()}

    @staticmethod
    def state_key(option_index, state):
        return (int(option_index), tuple(int(item) for item in active_states(state)))

    @staticmethod
    def feature(observation, grasped):
        relative = observation["cube_pos"] - observation["robot0_eef_pos"]
        xy_error = relative[:2] - np.array((-0.020, 0.0))
        return np.asarray((xy_error[0], xy_error[1], relative[2], float(grasped)), dtype=np.float64)

    def observe_teacher(self, option_index, state, action, observation, grasped):
        if not self.enabled:
            return
        key = self.state_key(option_index, state)
        self.teacher_labels[key][int(action)] += 1
        feature = self.feature(observation, grasped)
        self.teacher_features[int(option_index)].append(feature)
        self.teacher_histograms[int(option_index)].append(feature)

    def observe_student(self, phase, option_index, state, teacher_action, network_action,
                        executed_action, observation, grasped):
        if not self.enabled or teacher_action is None:
            return
        self.student_records[phase].append({
            "option": int(option_index),
            "state": self.state_key(option_index, state),
            "teacher_action": int(teacher_action),
            "network_action": int(network_action),
            "executed_action": int(executed_action),
            "feature": self.feature(observation, grasped),
        })

    def _student_summary(self, records, initial_support=False):
        if not records:
            return {"decisions": 0}
        state_out = continuous_out = both_out = 0
        agreement_network = agreement_executed = 0
        by_option = Counter()
        distances = []
        labels = (self.initial_teacher_labels if initial_support and self.initial_teacher_labels is not None
                  else self.teacher_labels)
        features = (self.initial_teacher_features if initial_support and self.initial_teacher_features is not None
                    else self.teacher_features)
        for record in records:
            state_missing = record["state"] not in labels
            support = features.get(record["option"], [])
            if support:
                values = np.asarray(support)
                # Grasp mode is part of support; a mismatch cannot be called a
                # nearby demonstrated contact state merely from position.
                delta = values[:, :3] - record["feature"][:3]
                distance = np.linalg.norm(delta, axis=1)
                distance[values[:, 3] != record["feature"][3]] = np.inf
                minimum = float(distance.min())
            else:
                minimum = float("inf")
            continuous_missing = minimum > self.distance_threshold
            state_out += int(state_missing)
            continuous_out += int(continuous_missing)
            both_out += int(state_missing and continuous_missing)
            agreement_network += int(record["network_action"] == record["teacher_action"])
            agreement_executed += int(record["executed_action"] == record["teacher_action"])
            by_option[OPTION_NAMES[record["option"]]] += 1
            distances.append(minimum)
        count = len(records)
        finite = [value for value in distances if np.isfinite(value)]
        return {
            "decisions": count,
            "state_out_of_demo_rate": state_out / count,
            "continuous_out_of_demo_rate": continuous_out / count,
            "both_out_of_demo_rate": both_out / count,
            "mean_nearest_teacher_distance_m": float(np.mean(finite)) if finite else None,
            "network_teacher_agreement_rate": agreement_network / count,
            "executed_teacher_agreement_rate": agreement_executed / count,
            "option_decisions": {name: by_option[name] for name in OPTION_NAMES},
        }

    def summary(self):
        if not self.enabled:
            return None
        entropies, conflicts = [], 0
        for labels in self.teacher_labels.values():
            probabilities = np.asarray(list(labels.values()), dtype=np.float64)
            probabilities /= probabilities.sum()
            entropies.append(float(-(probabilities * np.log2(probabilities)).sum()))
            conflicts += int(len(labels) > 1)
        histograms = {}
        evaluation_records = (self.student_records["evaluation"] or
                              self.student_records["deployment"])
        for option_index in range(NUM_OPTION_CONTEXT):
            teacher = np.asarray(self.teacher_histograms.get(option_index, []))
            deployment = np.asarray([record["feature"] for record in
                                     evaluation_records
                                     if record["option"] == option_index])
            histograms[OPTION_NAMES[option_index]] = {
                "teacher_count": int(len(teacher)),
                "deployment_count": int(len(deployment)),
                "xy_error_histogram_teacher": (np.histogram2d(teacher[:, 0], teacher[:, 1], bins=8,
                    range=((-0.08, 0.08), (-0.08, 0.08)))[0].tolist() if len(teacher) else []),
                "xy_error_histogram_deployment": (np.histogram2d(deployment[:, 0], deployment[:, 1], bins=8,
                    range=((-0.08, 0.08), (-0.08, 0.08)))[0].tolist() if len(deployment) else []),
            }
        return {
            "continuous_distance_threshold_m": self.distance_threshold,
            "teacher_labeled_state_count": len(self.teacher_labels),
            "teacher_conflicting_state_count": conflicts,
            "teacher_mean_label_entropy_bits": float(np.mean(entropies)) if entropies else 0.0,
            "student_during_curriculum": self._student_summary(self.student_records["curriculum"]),
            "student_dagger_against_initial_teacher": self._student_summary(
                self.student_records["dagger"], initial_support=True),
            "student_evaluation": self._student_summary(evaluation_records),
            "student_deployment": self._student_summary(evaluation_records),
            "per_option_xy_error_histograms": histograms,
        }


class AlignActionPersistence:
    """Retain an ambiguous SNN align decision until the xy grid changes.

    This is deployment-time control hysteresis, not a teacher or a policy:
    PM can immediately replace the action whenever it has one unique winner.
    The stored action is used only after a silent/tied PM readout and only for
    a bounded number of decisions in the same observable align state.
    """

    def __init__(self, maximum_hold):
        self.maximum_hold = maximum_hold
        self.reset()

    def reset(self):
        self.state = None
        self.action = None
        self.count = 0

    def choose(self, state, proposed_action, pm_confident, allowed_actions):
        if pm_confident:
            self.state = state
            self.action = int(proposed_action)
            self.count = 1
            return int(proposed_action), False
        if (self.state == state and self.action in allowed_actions and
                self.count < self.maximum_hold):
            self.count += 1
            return self.action, True
        self.state = state
        self.action = int(proposed_action)
        self.count = 1
        return int(proposed_action), False


class AlignProgressMemory:
    """Reuse only an align direction that measurably reduced xy error."""

    def __init__(self, maximum_hold, minimum_improvement=1e-4):
        self.maximum_hold = maximum_hold
        self.minimum_improvement = minimum_improvement
        self.reset()

    def reset(self):
        self.good_action = None
        self.remaining = 0

    def context_action(self):
        """Expose a valid short-term direction token to DLPFC, if available."""
        return self.good_action if self.remaining > 0 else None

    def choose(self, proposed_action, pm_confident, allowed_actions):
        if (not pm_confident and self.good_action in allowed_actions and
                self.remaining > 0):
            self.remaining -= 1
            return self.good_action, True
        return int(proposed_action), False

    def observe(self, action, previous_error, next_error):
        if action < 4 and next_error < previous_error - self.minimum_improvement:
            self.good_action = int(action)
            self.remaining = self.maximum_hold
        elif action == self.good_action and next_error >= previous_error:
            self.remaining = 0


class ProgressDirectionBias:
    """Local, non-teacher bias learned only from measured xy progress."""

    def __init__(self, learning_rate=0.15, maximum=0.50):
        self.learning_rate = learning_rate
        self.maximum = maximum
        self.values = np.zeros(4, dtype=np.float32)

    def update(self, action, previous_error, next_error):
        if action >= 4:
            return
        progress = previous_error - next_error
        self.values[action] = np.clip(
            self.values[action] + self.learning_rate * np.clip(progress / 0.01, -1.0, 1.0),
            -self.maximum, self.maximum)

    def scores(self, token_action):
        scores = np.zeros(NUM_ACTION, dtype=np.float32)
        if token_action is not None:
            scores[token_action] = max(0.0, self.values[token_action])
        return scores


class AlignProgressValue:
    """Persistent local action values from executed xy-error reductions.

    This pathway is deliberately weaker than a policy: it only resolves an
    ambiguous PM readout, and it learns from the physical error reduction of
    the action actually executed in the current observable align state.
    """

    def __init__(self, num_state, learning_rate=0.15, maximum=0.50, minimum_visits=1):
        self.values = np.zeros((num_state, 4), dtype=np.float32)
        self.visits = np.zeros((num_state, 4), dtype=np.int32)
        self.learning_rate = learning_rate
        self.maximum = maximum
        self.minimum_visits = minimum_visits

    def scores(self, state):
        scores = np.zeros(NUM_ACTION, dtype=np.float32)
        # Align uses one discrete state in the current baseline; retaining the
        # active-state convention also keeps this valid for future factorized
        # encoders without turning it into a teacher lookup table.
        for active_state in active_states(state):
            eligible = self.visits[active_state] >= self.minimum_visits
            scores[:4] += self.values[active_state] * eligible
        return scores

    def update(self, state, action, previous_error, next_error):
        if action >= 4 or previous_error is None or next_error is None:
            return
        progress = float(np.clip((previous_error - next_error) / 0.01, -1.0, 1.0))
        for active_state in active_states(state):
            value = self.values[active_state, action]
            self.values[active_state, action] = np.clip(
                value + self.learning_rate * (progress - value),
                0.0, self.maximum,
            )
            self.visits[active_state, action] += 1


class TeacherPolicyMemory:
    """Persistent state-action vote table for a behavior-cloning diagnostic."""

    def __init__(self):
        self.counts = {}

    def observe(self, state, action):
        votes = self.counts.setdefault(state, np.zeros(NUM_ACTION, dtype=np.int64))
        votes[action] += 1

    def predict(self, state):
        votes = self.counts.get(state)
        if votes is None or votes.sum() == 0:
            return None
        return int(np.argmax(votes))

    def summary(self):
        labeled = len(self.counts)
        conflicts = sum(int(np.count_nonzero(votes) > 1) for votes in self.counts.values())
        return {"labeled_states": labeled, "conflicting_states": conflicts}


class TeacherCloneMLP:
    """Small supervised policy diagnostic over the same low-dimensional inputs.

    It is deliberately separate from BDM-SNN: its purpose is to decide whether
    the teacher trajectory is learnable from continuous observation features
    before attributing a failure to the basal-ganglia readout.
    """

    def __init__(self, device):
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, NUM_ACTION),
        ).to(device)
        self.features = []
        self.actions = []
        self.last_loss = float("nan")
        self.last_accuracy = float("nan")

    @staticmethod
    def features_from(observation, decision_index, max_decisions):
        relative = observation["cube_pos"] - observation["robot0_eef_pos"]
        gripper = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)
        # Fixed physical scales avoid test-set statistics and keep the inputs
        # approximately comparable without leaking any teacher state.
        time_context = np.asarray((decision_index / max(1, max_decisions - 1),), dtype=np.float32)
        return np.concatenate((relative / np.array((0.15, 0.15, 0.20)),
                               gripper / 0.04, time_context)).astype(np.float32)

    def observe(self, observation, action, decision_index, max_decisions):
        self.features.append(self.features_from(observation, decision_index, max_decisions))
        self.actions.append(int(action))

    def fit(self, epochs, batch_size, rng):
        if not self.actions:
            return
        features = torch.as_tensor(np.asarray(self.features), device=self.device)
        labels = torch.as_tensor(self.actions, dtype=torch.long, device=self.device)
        counts = torch.bincount(labels, minlength=NUM_ACTION).float()
        # Later grasp / lift actions are rare but essential to the task.
        class_weights = torch.where(counts > 0, 1.0 / counts, torch.zeros_like(counts))
        class_weights /= class_weights[class_weights > 0].mean()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3)
        self.model.train()
        for _ in range(epochs):
            for indices in torch.as_tensor(rng.permutation(len(labels)), device=self.device).split(batch_size):
                logits = self.model(features[indices])
                loss = F.cross_entropy(logits, labels[indices], weight=class_weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features)
            self.last_loss = float(F.cross_entropy(logits, labels, weight=class_weights).item())
            self.last_accuracy = float((logits.argmax(dim=1) == labels).float().mean().item())

    def predict(self, observation, decision_index, max_decisions):
        if not self.actions:
            return None
        feature = torch.as_tensor(self.features_from(observation, decision_index, max_decisions),
                                  device=self.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            return int(self.model(feature).argmax(dim=1).item())

    def summary(self):
        return {"samples": len(self.actions), "train_loss": self.last_loss,
                "train_accuracy": self.last_accuracy}


class TeacherCloneGRU:
    """Recurrent behavior-cloning diagnostic with episode-local hidden state."""

    def __init__(self, device):
        self.device = device
        self.model = nn.GRU(5, 32, batch_first=True).to(device)
        self.readout = nn.Linear(32, NUM_ACTION).to(device)
        self.episodes = []
        self.current_features = []
        self.current_actions = []
        self.hidden = None
        self.last_loss = float("nan")
        self.last_accuracy = float("nan")

    def reset_episode(self):
        self.hidden = None
        self.current_features = []
        self.current_actions = []

    def observe(self, observation, action):
        self.current_features.append(TeacherCloneMLP.features_from(observation, 0, 1)[:5])
        self.current_actions.append(int(action))

    def finish_teacher_episode(self):
        if self.current_actions:
            self.episodes.append((np.asarray(self.current_features, dtype=np.float32),
                                  np.asarray(self.current_actions, dtype=np.int64)))

    def fit(self, epochs, rng):
        if not self.episodes:
            return
        labels_all = np.concatenate([labels for _, labels in self.episodes])
        counts = torch.bincount(torch.as_tensor(labels_all, device=self.device), minlength=NUM_ACTION).float()
        class_weights = torch.where(counts > 0, 1.0 / counts, torch.zeros_like(counts))
        class_weights /= class_weights[class_weights > 0].mean()
        optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.readout.parameters()), lr=2e-3)
        self.model.train()
        self.readout.train()
        for _ in range(epochs):
            for index in rng.permutation(len(self.episodes)):
                features_np, labels_np = self.episodes[index]
                features = torch.as_tensor(features_np, device=self.device).unsqueeze(0)
                labels = torch.as_tensor(labels_np, dtype=torch.long, device=self.device)
                hidden, _ = self.model(features)
                loss = F.cross_entropy(self.readout(hidden[0]), labels, weight=class_weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.model.eval()
        self.readout.eval()
        correct = total = 0
        losses = []
        with torch.no_grad():
            for features_np, labels_np in self.episodes:
                features = torch.as_tensor(features_np, device=self.device).unsqueeze(0)
                labels = torch.as_tensor(labels_np, dtype=torch.long, device=self.device)
                logits = self.readout(self.model(features)[0][0])
                losses.append(F.cross_entropy(logits, labels, weight=class_weights).item())
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                total += len(labels)
        self.last_loss = float(np.mean(losses))
        self.last_accuracy = correct / total

    def predict(self, observation):
        feature = torch.as_tensor(TeacherCloneMLP.features_from(observation, 0, 1)[:5],
                                  device=self.device).view(1, 1, -1)
        self.model.eval()
        self.readout.eval()
        with torch.no_grad():
            output, self.hidden = self.model(feature, self.hidden)
            return int(self.readout(output[0, 0]).argmax().item())

    def summary(self):
        return {"episodes": len(self.episodes),
                "samples": int(sum(len(labels) for _, labels in self.episodes)),
                "train_loss": self.last_loss, "train_accuracy": self.last_accuracy}


class ProgressReward:
    """Configurable R-STDP reward, logged separately from Lift's sparse success."""

    def __init__(self, timeout_penalty=-1.0, stagnant_penalty=-0.10, unsafe_penalty=-2.0):
        self.timeout_penalty = timeout_penalty
        self.stagnant_penalty = stagnant_penalty
        self.unsafe_penalty = unsafe_penalty
        self.previous_distance = None
        self.previous_grasped = False
        self.previous_success = False
        self.stagnant_steps = 0

    def reset(self, observation):
        self.previous_distance = float(np.linalg.norm(
            observation["cube_pos"] - observation["robot0_eef_pos"]))
        self.previous_grasped = False
        self.previous_success = False
        self.stagnant_steps = 0

    def compute(self, observation, grasped, success, timeout, unsafe):
        distance = float(np.linalg.norm(observation["cube_pos"] - observation["robot0_eef_pos"]))
        # Clipping keeps a single large contact transient from dominating STDP.
        distance_progress = float(np.clip(self.previous_distance - distance, -0.03, 0.03))
        reward = 15.0 * distance_progress
        components = {"approach": 15.0 * distance_progress, "grasp": 0.0,
                      "lift": 0.0, "stagnation": 0.0, "timeout": 0.0, "unsafe": 0.0}

        if grasped and not self.previous_grasped:
            components["grasp"] = 1.0
            reward += components["grasp"]
        if success and not self.previous_success:
            components["lift"] = 10.0
            reward += components["lift"]

        if distance_progress <= 1e-4 and not grasped:
            self.stagnant_steps += 1
        else:
            self.stagnant_steps = 0
        if self.stagnant_steps >= 8:
            components["stagnation"] = self.stagnant_penalty
            reward += components["stagnation"]
        if timeout:
            components["timeout"] = self.timeout_penalty
            reward += components["timeout"]
        if unsafe:
            components["unsafe"] = self.unsafe_penalty
            reward += components["unsafe"]

        self.previous_distance = distance
        self.previous_grasped = grasped
        self.previous_success = success
        return reward, components


class TabularTDCritic:
    """Small state-value baseline for a local three-factor plasticity ablation."""

    def __init__(self, num_state, learning_rate=0.10, discount=0.98):
        self.values = {}
        self.learning_rate = learning_rate
        self.discount = discount

    def update(self, state, reward, next_state, terminal):
        state_key = tuple(active_states(state))
        next_key = tuple(active_states(next_state))
        value = self.values.get(state_key, 0.0)
        bootstrap = 0.0 if terminal else self.discount * self.values.get(next_key, 0.0)
        td_error = float(reward + bootstrap - value)
        self.values[state_key] = value + self.learning_rate * td_error
        return td_error


def make_env(seed, horizon, camera_observations=False, deterministic_robot_start=False):
    controller = load_composite_controller_config(controller="BASIC")
    kwargs = dict(
        robots="Panda",
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=camera_observations,
        use_camera_obs=camera_observations,
        reward_shaping=True,
        horizon=horizon,
        control_freq=20,
        seed=seed,
        initialization_noise=None if deterministic_robot_start else "default",
    )
    if camera_observations:
        kwargs.update(camera_names="frontview", camera_heights=64, camera_widths=64)
    return suite.make("Lift", **kwargs)


def set_cube_position(env, xy_position):
    """Fix the cube pose and refresh robosuite's cached observations.

    ``_get_observations()`` returns cached sensor values unless it is explicitly
    forced to update.  A stale pre-reset cube position made the first teacher
    action depend on a different state from the physical scene.
    """
    cube_qpos = env.sim.data.get_joint_qpos("cube_joint0").copy()
    cube_qpos[:3] = (float(xy_position[0]), float(xy_position[1]), 0.8202)
    # Lift's cube is square in the table plane, but fixing its yaw removes an
    # avoidable reset variable from the contact-sensitive curriculum.
    cube_qpos[3:] = (1.0, 0.0, 0.0, 0.0)
    env.sim.data.set_joint_qpos("cube_joint0", cube_qpos)
    env.sim.data.set_joint_qvel("cube_joint0", np.zeros(6, dtype=np.float64))
    env.sim.forward()
    return env._get_observations(force_update=True)


def is_unsafe(observation):
    eef_position = observation["robot0_eef_pos"]
    return bool(eef_position[2] < 0.76 or eef_position[2] > 1.45 or
                np.linalg.norm(eef_position[:2]) > 0.65)


def make_input(state, num_state, device):
    encoded = torch.zeros(num_state, dtype=torch.float32, device=device)
    states = active_states(state)
    # Keep the total sensory current equal to the one-hot baseline even when
    # a factorized representation activates several cells.
    encoded[list(states)] = 2.0 / len(states)
    return encoded


def active_states(state):
    """Normalize one-hot and factorized DLPFC state representations."""
    return state if isinstance(state, tuple) else (state,)


def learning_states(state, action):
    """Restrict population supervision to the sensory axis that chose action."""
    del action
    return active_states(state)


def choose_action(network, state, eligibility_d1, eligibility_d2, rng, epsilon,
                  internal_steps, max_internal_steps, trace_decay, num_state, device,
                  allowed_actions=None, fixed_pm_window=False, action_bias=None,
                  decision_readout="pm"):
    """Decode either PM or thalamic spikes without changing BDM-SNN dynamics."""
    input_current = make_input(state, num_state, device)
    pm_spikes = torch.zeros(network.num_action, dtype=torch.float32, device=device)
    thalamus_spikes = torch.zeros(network.num_action, dtype=torch.float32, device=device)
    region_spikes = torch.zeros(len(REGION_NAMES), dtype=torch.float32, device=device)
    executed_internal_steps = 0
    for _ in range(max_internal_steps):
        pm_output, dw = network(input_current)
        executed_internal_steps += 1
        pm_spikes += pm_output.detach()
        thalamus_spikes += network.out[5].detach()
        region_spikes += torch.stack([out.detach().sum() for out in network.out])
        eligibility_d1.mul_(trace_decay).add_(dw[0][0].detach())
        eligibility_d2.mul_(trace_decay).add_(dw[1][0].detach())
        # Fixed windows give spike counts an unambiguous physical/control
        # horizon. The legacy path stops after the first PM response.
        if (not fixed_pm_window and executed_internal_steps >= internal_steps and
                pm_spikes.sum().item() > 0):
            break

    pm_silent = bool(pm_spikes.sum().item() == 0)
    allowed_actions = np.asarray(range(network.num_action) if allowed_actions is None else allowed_actions,
                               dtype=np.int64)
    if allowed_actions.size == 0 or np.any(allowed_actions < 0) or np.any(allowed_actions >= network.num_action):
        raise ValueError("an option must allow at least one action")
    allowed_spikes = pm_spikes[torch.as_tensor(allowed_actions, device=device)]
    max_spike_count = torch.max(allowed_spikes)
    tied_actions = allowed_actions[torch.nonzero(allowed_spikes == max_spike_count,
                                                  as_tuple=False).flatten().detach().cpu().numpy()]
    pm_action = int(rng.choice(tied_actions))
    pm_confident = not pm_silent and tied_actions.size == 1
    if decision_readout == "pm":
        readout_spikes = pm_spikes
    elif decision_readout == "thalamus":
        # The preceding probe establishes whether this upstream selection
        # signal is more reliable; no synapse, neuron, or teacher label is
        # added here. PM activity is still logged for a fair ablation.
        readout_spikes = thalamus_spikes
    else:
        raise ValueError("decision_readout must be 'pm' or 'thalamus'")
    allowed_readout = readout_spikes[torch.as_tensor(allowed_actions, device=device)]
    max_readout = torch.max(allowed_readout)
    readout_ties = allowed_actions[torch.nonzero(allowed_readout == max_readout,
                                                  as_tuple=False).flatten().detach().cpu().numpy()]
    readout_silent = bool(readout_spikes.sum().item() == 0)
    # Random tie resolution remains necessary, but its rate is measured rather
    # than treated as evidence of a successful autonomous decision.
    network_action = int(rng.choice(readout_ties))
    readout_confident = not readout_silent and readout_ties.size == 1
    bias_intervened = False
    selected_action = network_action
    if not readout_confident and action_bias is not None:
        allowed_bias = np.asarray(action_bias, dtype=np.float32)[allowed_actions]
        if np.any(allowed_bias > 0.0):
            biased_spikes = allowed_readout + torch.as_tensor(
                allowed_bias, device=device, dtype=allowed_readout.dtype)
            biased_max = torch.max(biased_spikes)
            biased_ties = allowed_actions[torch.nonzero(biased_spikes == biased_max,
                                                         as_tuple=False).flatten().detach().cpu().numpy()]
            selected_action = int(rng.choice(biased_ties))
            # With one positive direction bias, the fallback resolves an
            # otherwise silent/tied PM decision using measured physical progress.
            bias_intervened = selected_action != network_action or biased_ties.size < tied_actions.size
    exploratory = bool(rng.random() < epsilon)
    # A silent PM population must not abort the physical episode.  Its fallback
    # is recorded independently from intentional epsilon exploration.
    action = int(rng.choice(allowed_actions)) if exploratory else selected_action
    return action, eligibility_d1, eligibility_d2, {
        "pm_spikes": pm_spikes,
        "thalamus_spikes": thalamus_spikes,
        "region_spikes": region_spikes,
        "pm_silent": pm_silent,
        "readout_silent": readout_silent,
        "exploratory": exploratory,
        "network_action": network_action,
        "pm_confident": pm_confident,
        "readout_confident": readout_confident,
        "pm_tie_count": int(tied_actions.size),
        "readout_tie_count": int(readout_ties.size),
        "progress_bias_intervened": bias_intervened and not exploratory,
        "internal_steps": executed_internal_steps,
    }


def striatum_weight_index(network, state, action):
    """Locate a state-action synapse in expanded or population-level Str."""
    return int(action if network.compact_striatum else state * network.num_action + action)


def striatum_action_slice(network, state):
    return (slice(0, network.num_action) if network.compact_striatum else
            slice(state * network.num_action, (state + 1) * network.num_action))


def reward_modulated_update(network, state, action, reward, eligibility_d1, eligibility_d2):
    """Apply the original D1 / D2 reward modulation to an arbitrary action slice."""
    reward_mask = torch.ones_like(eligibility_d1)
    for active_state in active_states(state):
        reward_mask[active_state, striatum_weight_index(network, active_state, action)] = reward
    before_d1 = network.connection[0].weight.detach().clone()
    before_d2 = network.connection[1].weight.detach().clone()
    for active_state in active_states(state):
        network.UpdateWeight(0, active_state, network.num_action, reward_mask * eligibility_d1)
        network.UpdateWeight(1, active_state, network.num_action, -reward_mask * eligibility_d2)
    return {
        "d1_weight_change_l1": float((network.connection[0].weight - before_d1).abs().sum().item()),
        "d2_weight_change_l1": float((network.connection[1].weight - before_d2).abs().sum().item()),
    }


def three_factor_update(network, state, action, td_error, eligibility_d1, eligibility_d2,
                        learning_rate):
    """Apply a local reward-modulated update without row-wise normalization.

    The legacy ``UpdateWeight`` z-scores all eight actions and rescales the
    row after every decision.  This diagnostic instead updates only the action
    that the environment executed, preserving the magnitude and sign of the
    critic's TD error.  D1 and D2 use opposite dopamine modulation.
    """
    changes = [0.0, 0.0]
    with torch.no_grad():
        for active_state in learning_states(state, action):
            index = striatum_weight_index(network, active_state, action)
            before_d1 = network.connection[0].weight[active_state, index].item()
            before_d2 = network.connection[1].weight[active_state, index].item()
            trace_d1 = float(torch.nan_to_num(eligibility_d1[active_state, index]).item())
            trace_d2 = float(torch.nan_to_num(eligibility_d2[active_state, index]).item())
            d1 = network.connection[0].weight[active_state, index]
            d2 = network.connection[1].weight[active_state, index]
            d1.add_(learning_rate * td_error * trace_d1).clamp_(0.05, network.weight_exc)
            d2.add_(-learning_rate * td_error * trace_d2).clamp_(0.05, network.weight_exc)
            changes[0] += abs(d1.item() - before_d1)
            changes[1] += abs(d2.item() - before_d2)
    return {
        "d1_weight_change_l1": changes[0],
        "d2_weight_change_l1": changes[1],
    }


def initialize_three_factor_weights(network):
    """Start the two plastic pathways away from saturation for signed updates."""
    with torch.no_grad():
        for connection in network.connection[:2]:
            connection.weight.copy_(0.5 * connection.mask)


def executed_action_eligibility(network, eligibility, state, action):
    """Assign local eligibility to the action actually sent to the robot.

    The PM can be silent or tied while an option-safe fallback still sends one
    primitive to the robot.  Its measured reward must update that executed
    state-action synapse, rather than an unrelated raw PM candidate.  The
    accumulated natural trace remains intact for subsequent decisions.
    """
    clamped = torch.zeros_like(eligibility)
    for active_state in learning_states(state, action):
        clamped[active_state, striatum_weight_index(network, active_state, action)] = 1.0
    return clamped


def behavior_clone_update(network, state, action, off_weight):
    """Make the teacher's discrete action the direct-path preference for one state.

    This is an explicit teacher signal, not an R-STDP approximation: D1 makes
    the demonstrated action the strongest direct-path synapse, while D2 keeps
    that action's indirect-path synapse weak.  It is applied only during a
    teacher-controlled decision, after the ordinary reward-modulated update.
    """
    with torch.no_grad():
        for active_state in learning_states(state, action):
            action_slice = striatum_action_slice(network, active_state)
            d1 = network.connection[0].weight[active_state, action_slice]
            d2 = network.connection[1].weight[active_state, action_slice]
            d1.fill_(off_weight)
            d2.fill_(network.weight_exc)
            d1[action] = network.weight_exc
            d2[action] = off_weight


def teacher_probability(args, episode_index):
    """Linearly fade teacher actions while keeping the same SNN reward update."""
    if args.teacher_episodes <= 0 or episode_index >= args.teacher_episodes:
        return 0.0
    progress = min(1.0, episode_index / max(1, args.teacher_episodes - 1))
    return args.teacher_start + progress * (args.teacher_end - args.teacher_start)


def validate_environment(seed, control_steps):
    """Minimal robosuite integration test, including optional fixed camera output."""
    rng = np.random.default_rng(seed)
    env = make_env(seed, horizon=control_steps * 8, camera_observations=True)
    observation = env.reset()
    print("Lift validation")
    print("  action_dim:", env.action_dim)
    print("  observation keys:", sorted(observation.keys()))
    print("  frontview shape:", tuple(observation["frontview_image"].shape))
    rewards = []
    for _ in range(4):
        command = rng.uniform(env.action_spec[0], env.action_spec[1])
        observation, reward, _, _ = env.step(command)
        rewards.append(float(reward))
    print("  random-step rewards:", rewards)
    env.close()


def create_plot(metrics, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = np.arange(1, len(metrics["episodes"]) + 1)
    returns = [episode["learning_return"] for episode in metrics["episodes"]]
    successes = np.asarray([episode["sparse_success"] for episode in metrics["episodes"]], dtype=float)
    action_counts = np.asarray([episode["action_counts"] for episode in metrics["episodes"]]).sum(axis=0)
    spike_counts = np.asarray([episode["region_spikes"] for episode in metrics["episodes"]]).mean(axis=0)
    rolling_success = np.cumsum(successes) / episodes

    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(episodes, returns, marker="o", linewidth=1)
    axes[0, 0].set(title="R-STDP learning reward", xlabel="Episode", ylabel="Return")
    axes[0, 1].plot(episodes, rolling_success, marker="o", linewidth=1)
    axes[0, 1].set(title="Sparse Lift success rate", xlabel="Episode", ylabel="Rate", ylim=(-0.05, 1.05))
    axes[1, 0].bar(ACTION_NAMES, action_counts)
    axes[1, 0].tick_params(axis="x", rotation=35)
    axes[1, 0].set(title="Executed action distribution", ylabel="Decision count")
    axes[1, 1].bar(REGION_NAMES, spike_counts)
    axes[1, 1].tick_params(axis="x", rotation=35)
    axes[1, 1].set(title="Mean spikes per episode", ylabel="Spike count")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    rng = np.random.default_rng(args.seed)
    # Rank 0 retains full currents; the monitor only counts baseline source
    # spikes and does not alter the BDM-SNN forward path.
    encoder = LiftStateEncoder(use_phase_context=args.phase_context,
                               use_action_duration=args.action_duration_context,
                               use_option_context=args.option_context,
                               use_align_grid=args.align_grid_context,
                               use_progress_memory_context=args.progress_memory_context,
                               use_factorized_progress_memory=args.factorized_progress_memory,
                               use_align_residual_grid=args.align_residual_grid_context,
                               use_align_residual_axis=args.align_residual_axis_context,
                               align_residual_bins=args.align_residual_bins,
                               snn_align_only=args.snn_align_only)
    snn_action_count = args.align_action_count
    network = BDMSNN(encoder.num_state, snn_action_count, 1.0, -0.5, "lif",
                     communication_mode="all_cross_core", pm_threshold=args.pm_threshold,
                     pm_lateral_gain=args.pm_lateral_gain,
                     compact_striatum=args.compact_striatum).to(device)
    if args.plasticity_rule == "three_factor":
        initialize_three_factor_weights(network)
    critic = (TabularTDCritic(encoder.num_state, args.critic_learning_rate,
                              args.critic_discount)
              if args.plasticity_rule == "three_factor" else None)
    primitives = LiftActionPrimitives(args.action_magnitude)
    teacher = LiftTeacher()
    option_controller = (LiftOptionController(close_steps=args.close_steps,
                                              failed_lift_steps=args.failed_lift_steps)
                         if args.option_context else None)
    align_persistence = (AlignActionPersistence(args.align_hold_decisions)
                         if args.align_action_persistence else None)
    align_progress_memory = (AlignProgressMemory(args.align_hold_decisions)
                             if (args.align_progress_persistence or args.progress_memory_context or
                                 args.progress_direction_bias) else None)
    direction_bias = (ProgressDirectionBias(args.progress_bias_learning_rate,
                                            args.progress_bias_maximum)
                      if args.progress_direction_bias else None)
    align_progress_value = (AlignProgressValue(encoder.num_state,
                                               args.progress_value_learning_rate,
                                               args.progress_value_maximum,
                                               args.progress_value_minimum_visits)
                            if args.align_progress_value else None)
    coverage_audit = CoverageAudit(args.coverage_audit, args.coverage_distance_threshold)
    policy_memory = TeacherPolicyMemory()
    clone_mlp = TeacherCloneMLP(device)
    clone_gru = TeacherCloneGRU(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "configuration": vars(args),
        "state_space": (
            ("4 observable phase contexts x " if args.phase_context else "") +
            "8 relative-position octants x 4 distance bins x 2 closed-or-grasped modes" +
            (" x 5 observable option contexts" if args.option_context else "") +
            ("; align option replaces its base code with a 5x5 xy-error grid"
             if args.align_grid_context else "") +
            ("; align option instead uses a 12x12 xy residual grid with explicit overflow bins"
             if args.align_residual_grid_context else "") +
            ("; align residual grid includes an observable dominant-error-axis bit"
             if args.align_residual_axis_context else "") +
            (f"; residual grid uses {args.align_residual_bins} bins per axis"
             if args.align_residual_grid_context or args.align_residual_axis_context else "") +
            ("; fixed-action options bypass DLPFC/BG/PM and do not receive SNN plasticity"
             if args.snn_align_only else "") +
            ("; StrD1/StrD2 use one population LIF channel per action"
             if args.compact_striatum else "") +
            (f"; align SNN exposes {snn_action_count} lateral action channels while "
             "the FSM retains the four non-align physical primitives"
             if snn_action_count != NUM_ACTION else "") +
            ("; valid progress direction is an explicit five-symbol DLPFC context"
             if args.progress_memory_context else "") +
            ("; align xy and progress-token neurons are factorized and co-active"
             if args.factorized_progress_memory else "") +
            (" x 9 previous-action symbols x 3 action-duration bins"
             if args.action_duration_context else "")),
        "action_space": list(ACTION_NAMES),
        "snn_align_action_space": list(ALIGN_ACTION_NAMES[:snn_action_count]),
        "communication": "full, uncompressed BDM-SNN baseline",
        "pm_dynamics": {
            "threshold": network.pm_threshold,
            "lateral_gain": network.pm_lateral_gain,
        },
        "episodes": [],
    }

    for episode_index in range(args.episodes):
        episode_phase = ("teacher" if episode_index < args.teacher_episodes else
                         "dagger" if episode_index < args.teacher_episodes + args.dagger_episodes else
                         "evaluation")
        frozen_evaluation = episode_phase == "evaluation" and args.freeze_evaluation
        if episode_phase == "dagger":
            coverage_audit.freeze_initial_teacher_support()
        env = make_env(
            args.seed + episode_index, args.max_decisions * args.control_steps,
            deterministic_robot_start=args.deterministic_robot_start)
        observation = env.reset()
        if args.fixed_cube:
            observation = set_cube_position(env, args.fixed_cube_xy)
        network.reset()  # Reset membrane and STDP traces, retain learned weights.
        teacher.reset()
        if option_controller is not None:
            option_controller.reset()
        if align_persistence is not None:
            align_persistence.reset()
        if align_progress_memory is not None:
            align_progress_memory.reset()
        clone_gru.reset_episode()
        action_memory = ActionDurationMemory()
        eligibility_d1 = torch.zeros_like(network.connection[0].weight)
        eligibility_d2 = torch.zeros_like(network.connection[1].weight)
        reward_model = ProgressReward(args.timeout_penalty, args.stagnant_penalty, args.unsafe_penalty)
        reward_model.reset(observation)
        epsilon = max(args.epsilon_end, args.epsilon_start *
                      (args.epsilon_decay ** episode_index))
        if args.autonomous_epsilon is not None and episode_phase != "teacher":
            epsilon = args.autonomous_epsilon
        current_teacher_probability = teacher_probability(args, episode_index)
        action_counts = Counter()
        region_spikes = np.zeros(len(REGION_NAMES), dtype=np.float64)
        learning_return = 0.0
        env_return = 0.0
        update_l1 = np.zeros(2, dtype=np.float64)
        td_errors = []
        pm_silent_decisions = 0
        readout_silent_decisions = 0
        internal_snn_steps = 0
        exploratory_decisions = 0
        teacher_decisions = 0
        action_clamp_decisions = 0
        executed_action_credit_decisions = 0
        teacher_pm_agreements = 0
        behavior_clone_decisions = 0
        dagger_label_decisions = 0
        teacher_action_labels = {}
        behavior_clone_label_conflicts = 0
        clone_readout_decisions = 0
        clone_readout_unknown_states = 0
        clone_mlp_readout_decisions = 0
        pm_tie_counts = []
        option_decisions = Counter()
        align_persistence_decisions = 0
        align_progress_persistence_decisions = 0
        progress_bias_decisions = 0
        progress_value_decisions = 0
        align_source_stats = defaultdict(lambda: {
            "decisions": 0, "positive_progress": 0, "xy_progress_sum_m": 0.0,
            "entered_descend": 0, "grasp_onset": 0, "lift_success": 0,
        })
        # These are observability-only milestones.  They make the first
        # broken transition in align -> descend -> close -> lift explicit
        # without changing either the option controller or the SNN action.
        stage_first_decision = {name: None for name in OPTION_NAMES}
        stage_teacher_labels = Counter()
        stage_network_teacher_agreements = Counter()
        stage_executed_teacher_agreements = Counter()
        initial_align_xy_error = None
        minimum_align_xy_error = float("inf")
        align_exit_xy_error = None
        first_grasp_decision = None
        first_success_decision = None
        success = False
        grasped_any = False

        for decision_index in range(args.max_decisions):
            grasped = bool(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cube))
            observation_before_action = observation
            option_index = (option_controller.update_from_observation(observation, grasped)
                            if option_controller is not None else 0)
            if stage_first_decision[OPTION_NAMES[option_index]] is None:
                stage_first_decision[OPTION_NAMES[option_index]] = decision_index + 1
            if option_index == 0:
                xy_error = float(np.linalg.norm(
                    (observation["cube_pos"] - observation["robot0_eef_pos"])[:2] -
                    np.array((-0.020, 0.0))))
                if initial_align_xy_error is None:
                    initial_align_xy_error = xy_error
                minimum_align_xy_error = min(minimum_align_xy_error, xy_error)
            elif option_index == 1 and align_exit_xy_error is None:
                # The option can only enter descend after passing the fixed
                # deployment-visible xy tolerance.
                align_exit_xy_error = float(np.linalg.norm(
                    (observation["cube_pos"] - observation["robot0_eef_pos"])[:2] -
                    np.array((-0.020, 0.0))))
            progress_memory_action = (align_progress_memory.context_action()
                                      if (align_progress_memory is not None and option_index == 0) else None)
            action_bias = (direction_bias.scores(progress_memory_action)
                           if direction_bias is not None else None)
            state, state_info = encoder.encode(observation, grasped, action_memory.previous_action,
                                               action_memory.duration, option_index, progress_memory_action)
            snn_active = bool(state_info.get("snn_active", True))
            if align_progress_value is not None and option_index == 0:
                # This is used only when PM has no unique winner inside
                # choose_action; a unique SNN decision remains untouched.
                action_bias = align_progress_value.scores(state)
            # The audit asks the deterministic, deployment-visible option
            # controller for a counterfactual label but never uses it to pick
            # an action or update a weight.
            audit_teacher_action = (option_controller.teacher_action(observation)
                                    if option_controller is not None else None)
            if snn_active:
                action, eligibility_d1, eligibility_d2, decision = choose_action(
                    network, state, eligibility_d1, eligibility_d2, rng, epsilon,
                    args.internal_steps, args.max_internal_steps, args.trace_decay,
                    encoder.num_state, device,
                    (option_controller.allowed_actions() if option_controller is not None else None),
                    args.fixed_pm_window, action_bias, args.decision_readout)
            else:
                # Non-align options expose one safe primitive.  Bypassing the
                # SNN prevents fixed FSM actions from allocating DLPFC/Str
                # states or producing irrelevant plasticity updates.
                action = int(option_controller.allowed_actions()[0])
                decision = {
                    "region_spikes": torch.zeros(len(REGION_NAMES), device=device),
                    "pm_silent": False, "readout_silent": False, "exploratory": False,
                    "network_action": action, "pm_confident": False,
                    "readout_confident": False, "pm_tie_count": 0,
                    "progress_bias_intervened": False, "internal_steps": 0,
                }
            teacher_used = episode_phase == "teacher" and bool(rng.random() < current_teacher_probability)
            action_source = ("pm_unique" if decision["readout_confident"]
                             else "ambiguous_fallback")
            if teacher_used:
                if option_controller is not None:
                    teacher_action = option_controller.teacher_action(observation)
                    keep_gripper_closed = option_controller.keep_gripper_closed()
                else:
                    teacher_action, keep_gripper_closed = teacher.action(observation, grasped)
                action = teacher_action
                teacher_decisions += 1
                teacher_pm_agreements += int(snn_active and decision["network_action"] == teacher_action)
                if snn_active:
                    label_key = tuple(active_states(state))
                    previous_labels = teacher_action_labels.setdefault(label_key, set())
                    behavior_clone_label_conflicts += int(bool(previous_labels and action not in previous_labels))
                    previous_labels.add(action)
                    policy_memory.observe(state, action)
                clone_mlp.observe(observation, action, decision_index, args.max_decisions)
                clone_gru.observe(observation, action)
                if snn_active:
                    coverage_audit.observe_teacher(option_index, state, action, observation, grasped)
                    coverage_audit.observe_student("curriculum", option_index, state,
                                                   audit_teacher_action, decision["network_action"], action,
                                                   observation, grasped)
            else:
                # Cartesian actions must preserve a completed close command.
                # Previously this hold signal was discarded for all autonomous
                # policies, making a PM-selected lift physically unable to
                # retain the cube after the preceding close action.
                keep_gripper_closed = (option_controller.keep_gripper_closed()
                                       if option_controller is not None else
                                       bool(np.max(np.abs(observation["robot0_gripper_qpos"])) < 0.012)
                                       and action != 6)
                if snn_active and args.autonomous_readout == "clone_table":
                    cloned_action = policy_memory.predict(state)
                    if cloned_action is None:
                        clone_readout_unknown_states += 1
                    else:
                        action = cloned_action
                        clone_readout_decisions += 1
                elif snn_active and args.autonomous_readout == "clone_mlp":
                    action = clone_mlp.predict(observation, decision_index, args.max_decisions)
                    clone_mlp_readout_decisions += 1
                    # Retain grasp during a Cartesian primitive after the
                    # learned policy has issued a close command.
                    keep_gripper_closed = bool(
                        np.max(np.abs(observation["robot0_gripper_qpos"])) < 0.012
                    ) and action != 6
                elif snn_active and args.autonomous_readout == "clone_gru":
                    action = clone_gru.predict(observation)
                    clone_mlp_readout_decisions += 1
                    keep_gripper_closed = bool(
                        np.max(np.abs(observation["robot0_gripper_qpos"])) < 0.012
                    ) and action != 6
                if align_persistence is not None and option_index == 0:
                    action, retained = align_persistence.choose(
                        state, action, decision["readout_confident"],
                        option_controller.allowed_actions())
                    align_persistence_decisions += int(retained)
                if (align_progress_memory is not None and option_index == 0 and
                        args.align_progress_persistence):
                    action, retained = align_progress_memory.choose(
                        action, decision["readout_confident"], option_controller.allowed_actions())
                    align_progress_persistence_decisions += int(retained)
                    if retained:
                        action_source = "progress_memory"
                if direction_bias is not None and option_index == 0:
                    progress_bias_decisions += int(decision["progress_bias_intervened"])
                if align_progress_value is not None and option_index == 0:
                    progress_value_decisions += int(decision["progress_bias_intervened"])
                    if decision["progress_bias_intervened"]:
                        action_source = "progress_value"
                # A validated short-term direction is applied after the
                # value tie-break, so it is the actual source if retained.
                if (align_progress_memory is not None and option_index == 0 and
                        args.align_progress_persistence and retained):
                    action_source = "progress_memory"
                if snn_active:
                    coverage_audit.observe_student(episode_phase, option_index, state,
                                                   audit_teacher_action, decision["network_action"], action,
                                                   observation, grasped)
            if snn_active and not teacher_used and audit_teacher_action is not None:
                option_name = OPTION_NAMES[option_index]
                stage_teacher_labels[option_name] += 1
                stage_network_teacher_agreements[option_name] += int(
                    decision["network_action"] == audit_teacher_action)
                stage_executed_teacher_agreements[option_name] += int(
                    action == audit_teacher_action)
            action_counts[action] += 1
            if option_controller is not None:
                option_decisions[OPTION_NAMES[option_index]] += 1
            action_memory.update(action)
            region_spikes += decision["region_spikes"].detach().cpu().numpy()
            pm_silent_decisions += int(decision["pm_silent"])
            readout_silent_decisions += int(decision["readout_silent"])
            internal_snn_steps += decision["internal_steps"]
            exploratory_decisions += int(decision["exploratory"])
            pm_tie_counts.append(decision["pm_tie_count"])

            command = primitives.command(action, keep_gripper_closed=keep_gripper_closed)
            previous_xy_error = (float(np.linalg.norm(
                (observation["cube_pos"] - observation["robot0_eef_pos"])[:2] -
                np.array((-0.020, 0.0)))) if option_index == 0 else None)
            decision_env_reward = 0.0
            for _ in range(args.control_steps):
                observation, env_reward, _, _ = env.step(command)
                decision_env_reward += float(env_reward)
                if env._check_success():
                    break

            grasped = bool(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cube))
            # Progress is an observability metric for every align decision;
            # optional memory/value modules consume the same measurement.
            next_xy_error = (float(np.linalg.norm(
                (observation["cube_pos"] - observation["robot0_eef_pos"])[:2] -
                np.array((-0.020, 0.0)))) if option_index == 0 else None)
            if align_progress_memory is not None and option_index == 0:
                align_progress_memory.observe(action, previous_xy_error, next_xy_error)
            if direction_bias is not None and option_index == 0:
                direction_bias.update(action, previous_xy_error, next_xy_error)
            if align_progress_value is not None and option_index == 0 and not teacher_used:
                align_progress_value.update(state, action, previous_xy_error, next_xy_error)
            if option_controller is not None:
                option_controller.after_action(action)
            grasped_any |= grasped
            if grasped and first_grasp_decision is None:
                first_grasp_decision = decision_index + 1
            success = bool(env._check_success())
            if success and first_success_decision is None:
                first_success_decision = decision_index + 1
            timeout = decision_index + 1 == args.max_decisions and not success
            unsafe = is_unsafe(observation)
            learning_reward, _ = reward_model.compute(observation, grasped, success, timeout, unsafe)
            next_option_index = (option_controller.update_from_observation(observation, grasped)
                                 if option_controller is not None else 0)
            if not teacher_used and option_index == 0:
                source_stats = align_source_stats[action_source]
                source_stats["decisions"] += 1
                xy_progress = previous_xy_error - next_xy_error
                source_stats["xy_progress_sum_m"] += xy_progress
                source_stats["positive_progress"] += int(xy_progress > 1e-4)
                source_stats["entered_descend"] += int(next_option_index == 1)
                source_stats["grasp_onset"] += int(first_grasp_decision == decision_index + 1)
                source_stats["lift_success"] += int(first_success_decision == decision_index + 1)
            if stage_first_decision[OPTION_NAMES[next_option_index]] is None:
                stage_first_decision[OPTION_NAMES[next_option_index]] = decision_index + 1
            next_state, next_info = encoder.encode(observation, grasped, action_memory.previous_action,
                                                    action_memory.duration, next_option_index,
                                                    (align_progress_memory.context_action()
                                                     if align_progress_memory is not None else None))
            # Keep the current align state as a zero-progress bootstrap when
            # its physical action enters a fixed FSM stage.
            critic_next_state = next_state if next_info.get("snn_active", True) else state
            teacher_action_clamped = (snn_active and teacher_used and
                                      args.teacher_credit_mode == "action_clamp")
            executed_action_credited = snn_active and not teacher_used and args.executed_action_credit
            if teacher_action_clamped or executed_action_credited:
                update_eligibility_d1 = executed_action_eligibility(network, eligibility_d1, state, action)
                update_eligibility_d2 = executed_action_eligibility(network, eligibility_d2, state, action)
                action_clamp_decisions += int(teacher_action_clamped)
                executed_action_credit_decisions += int(executed_action_credited)
            else:
                update_eligibility_d1 = eligibility_d1
                update_eligibility_d2 = eligibility_d2
            changes = {"d1_weight_change_l1": 0.0, "d2_weight_change_l1": 0.0}
            if snn_active and not frozen_evaluation and args.plasticity_rule == "three_factor":
                td_error = critic.update(next_state=critic_next_state, state=state, reward=learning_reward,
                                         terminal=bool(success or unsafe or timeout))
                changes = three_factor_update(
                    network, state, action, td_error, update_eligibility_d1,
                    update_eligibility_d2, args.three_factor_learning_rate)
                td_errors.append(td_error)
            elif snn_active and not frozen_evaluation:
                changes = reward_modulated_update(
                    network, state, action, learning_reward, update_eligibility_d1, update_eligibility_d2)
            if snn_active and teacher_used and args.teacher_learning_mode == "behavior_clone":
                behavior_clone_update(network, state, action, args.clone_off_weight)
                behavior_clone_decisions += 1
            elif snn_active and episode_phase == "dagger" and audit_teacher_action is not None:
                # DAgger labels the state the student actually reached; the
                # student action remains in the environment and TD update.
                behavior_clone_update(network, state, audit_teacher_action, args.clone_off_weight)
                coverage_audit.observe_teacher(option_index, state, audit_teacher_action,
                                               observation_before_action, grasped)
                dagger_label_decisions += 1
            update_l1 += [changes["d1_weight_change_l1"], changes["d2_weight_change_l1"]]
            learning_return += learning_reward
            env_return += decision_env_reward
            if success or unsafe:
                break

        env.close()
        if teacher_decisions:
            clone_gru.finish_teacher_episode()
        if teacher_decisions and args.autonomous_readout == "clone_mlp":
            clone_mlp.fit(args.clone_train_epochs, args.clone_train_batch_size, rng)
        if teacher_decisions and args.autonomous_readout == "clone_gru":
            clone_gru.fit(args.clone_train_epochs, rng)
        episode_metrics = {
            "episode": episode_index,
            "phase": episode_phase,
            "frozen_evaluation": frozen_evaluation,
            "epsilon": epsilon,
            "teacher_probability": current_teacher_probability,
            "teacher_decisions": teacher_decisions,
            "action_clamp_decisions": action_clamp_decisions,
            "executed_action_credit_decisions": executed_action_credit_decisions,
            "teacher_pm_agreement_rate": (teacher_pm_agreements / teacher_decisions
                                          if teacher_decisions else float("nan")),
            "behavior_clone_decisions": behavior_clone_decisions,
            "dagger_label_decisions": dagger_label_decisions,
            "behavior_clone_label_conflicts": behavior_clone_label_conflicts,
            "teacher_labeled_states": len(teacher_action_labels),
            "clone_readout_decisions": clone_readout_decisions,
            "clone_readout_unknown_states": clone_readout_unknown_states,
            "clone_mlp_readout_decisions": clone_mlp_readout_decisions,
            "clone_mlp_train_loss": clone_mlp.last_loss,
            "clone_mlp_train_accuracy": clone_mlp.last_accuracy,
            "clone_gru_train_loss": clone_gru.last_loss,
            "clone_gru_train_accuracy": clone_gru.last_accuracy,
            "decisions": decision_index + 1,
            "control_steps": (decision_index + 1) * args.control_steps,
            "learning_return": learning_return,
            "robosuite_shaping_return": env_return,
            "sparse_success": success,
            "grasped_any": grasped_any,
            "unsafe_termination": unsafe,
            "pm_silent_decisions": pm_silent_decisions,
            "readout_silent_decisions": readout_silent_decisions,
            "internal_snn_steps": internal_snn_steps,
            "exploratory_decisions": exploratory_decisions,
            "mean_pm_tie_count": float(np.mean(pm_tie_counts)),
            # Counts refer to physical commands, including the FSM-only
            # descend / close / lift / recover primitives.
            "action_counts": [action_counts[index] for index in range(NUM_ACTION)],
            "region_spikes": region_spikes.tolist(),
            "d1_weight_change_l1": float(update_l1[0]),
            "d2_weight_change_l1": float(update_l1[1]),
            "mean_td_error": float(np.mean(td_errors)) if td_errors else float("nan"),
            "option_decisions": {name: option_decisions[name] for name in OPTION_NAMES},
            "align_persistence_decisions": align_persistence_decisions,
            "align_progress_persistence_decisions": align_progress_persistence_decisions,
            "progress_bias_decisions": progress_bias_decisions,
            "progress_value_decisions": progress_value_decisions,
            "align_action_sources": dict(align_source_stats),
            "stage_milestones": {
                "first_reached_decision": stage_first_decision,
                "initial_align_xy_error_m": initial_align_xy_error,
                "minimum_align_xy_error_m": (minimum_align_xy_error
                                               if np.isfinite(minimum_align_xy_error) else None),
                "align_exit_xy_error_m": align_exit_xy_error,
                "first_grasp_decision": first_grasp_decision,
                "first_success_decision": first_success_decision,
                "counterfactual_teacher_agreement": {
                    name: {
                        "decisions": stage_teacher_labels[name],
                        "network": (stage_network_teacher_agreements[name] / stage_teacher_labels[name]
                                    if stage_teacher_labels[name] else None),
                        "executed": (stage_executed_teacher_agreements[name] / stage_teacher_labels[name]
                                     if stage_teacher_labels[name] else None),
                    }
                    for name in OPTION_NAMES
                },
            },
        }
        metrics["episodes"].append(episode_metrics)
        print("episode={episode} reward={learning_return:.3f} success={sparse_success} "
              "grasped={grasped_any} decisions={decisions} teacher={teacher_decisions} "
              "epsilon={epsilon:.3f}".format(**episode_metrics))

    metrics["summary"] = {
        "mean_learning_return": float(np.mean([x["learning_return"] for x in metrics["episodes"]])),
        "sparse_success_rate": float(np.mean([x["sparse_success"] for x in metrics["episodes"]])),
        "grasp_any_rate": float(np.mean([x["grasped_any"] for x in metrics["episodes"]])),
        "mean_decisions": float(np.mean([x["decisions"] for x in metrics["episodes"]])),
        "teacher_assisted_success_rate": float(np.mean([
            x["sparse_success"] for x in metrics["episodes"] if x["teacher_decisions"] > 0]))
        if any(x["teacher_decisions"] > 0 for x in metrics["episodes"]) else float("nan"),
        "clone_policy_memory": policy_memory.summary(),
        "clone_mlp": clone_mlp.summary(),
        "clone_gru": clone_gru.summary(),
        "full_communication_traffic": network.communication_metrics(),
        "coverage_audit": coverage_audit.summary(),
        "phase_success": {
            phase: {
                "episodes": sum(x["phase"] == phase for x in metrics["episodes"]),
                "lift_successes": sum(x["sparse_success"] for x in metrics["episodes"]
                                      if x["phase"] == phase),
            }
            for phase in ("teacher", "dagger", "evaluation")
        },
    }
    with (output_dir / "lift_bdm_snn_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    np.savez_compressed(
        output_dir / "lift_bdm_snn_metrics.npz",
        learning_return=np.asarray([x["learning_return"] for x in metrics["episodes"]]),
        sparse_success=np.asarray([x["sparse_success"] for x in metrics["episodes"]], dtype=np.int8),
        action_counts=np.asarray([x["action_counts"] for x in metrics["episodes"]]),
        region_spikes=np.asarray([x["region_spikes"] for x in metrics["episodes"]]),
    )
    create_plot(metrics, output_dir / "lift_bdm_snn_baseline.png")
    print("summary:", json.dumps(metrics["summary"], indent=2))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-env", action="store_true", help="Run reset, random action, reward, and camera smoke test only")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-decisions", type=int, default=80)
    parser.add_argument("--control-steps", type=int, default=2)
    parser.add_argument("--internal-steps", type=int, default=6)
    parser.add_argument("--max-internal-steps", type=int, default=None,
                        help="Maximum SNN integration steps per decision; defaults to --internal-steps")
    parser.add_argument("--action-magnitude", type=float, default=0.25)
    parser.add_argument("--trace-decay", type=float, default=0.80)
    parser.add_argument("--epsilon-start", type=float, default=0.70)
    parser.add_argument("--epsilon-end", type=float, default=0.10)
    parser.add_argument("--epsilon-decay", type=float, default=0.92)
    parser.add_argument("--teacher-episodes", type=int, default=0,
                        help="Number of curriculum episodes with a fading discrete teacher")
    parser.add_argument("--teacher-start", type=float, default=1.0,
                        help="Teacher action probability in the first curriculum episode")
    parser.add_argument("--teacher-end", type=float, default=0.0,
                        help="Teacher action probability in the final curriculum episode")
    parser.add_argument("--dagger-episodes", type=int, default=0,
                        help="Student-executed training episodes with counterfactual teacher labels")
    parser.add_argument("--freeze-evaluation", action="store_true",
                        help="Freeze all synaptic and critic updates after teacher and DAgger episodes")
    parser.add_argument("--teacher-credit-mode", choices=("natural", "action_clamp"),
                        default="natural",
                        help="Use natural STDP traces or one-hot state-teacher-action credit during teacher decisions")
    parser.add_argument("--executed-action-credit", action="store_true",
                        help="During autonomous online learning, credit the primitive actually executed after PM ties/fallbacks")
    parser.add_argument("--teacher-learning-mode", choices=("rstdp", "behavior_clone"),
                        default="rstdp",
                        help="Keep reward-only learning or apply explicit D1/D2 teacher-action readout supervision")
    parser.add_argument("--plasticity-rule", choices=("legacy", "three_factor"), default="legacy",
                        help="Use legacy row-normalized update or local TD-modulated three-factor update")
    parser.add_argument("--three-factor-learning-rate", type=float, default=0.08,
                        help="Synaptic step size for --plasticity-rule three_factor")
    parser.add_argument("--critic-learning-rate", type=float, default=0.10,
                        help="Tabular TD critic step size for --plasticity-rule three_factor")
    parser.add_argument("--critic-discount", type=float, default=0.98,
                        help="TD discount for --plasticity-rule three_factor")
    parser.add_argument("--clone-off-weight", type=float, default=0.05,
                        help="Non-target D1 / target D2 weight used by --teacher-learning-mode behavior_clone")
    parser.add_argument("--autonomous-epsilon", type=float, default=None,
                        help="Override epsilon after teacher episodes; use 0 for deterministic PM evaluation")
    parser.add_argument("--autonomous-readout", choices=("pm", "clone_table", "clone_mlp", "clone_gru"), default="pm",
                        help="Use PM or a separately reported teacher-label lookup readout after curriculum")
    parser.add_argument("--clone-train-epochs", type=int, default=30,
                        help="Supervised epochs after each teacher episode for --autonomous-readout clone_mlp")
    parser.add_argument("--clone-train-batch-size", type=int, default=256)
    parser.add_argument("--phase-context", action="store_true",
                        help="Append a four-bin geometry/gripper task context to the 64-state encoder")
    parser.add_argument("--action-duration-context", action="store_true",
                        help="Append previous action and its 1 / 2-5 / 6+ decision duration bins to the SNN state")
    parser.add_argument("--option-context", action="store_true",
                        help="Deploy observable Lift options that gate infeasible actions; teacher remains training-only")
    parser.add_argument("--failed-lift-steps", type=int, default=4,
                        help="Consecutive lift primitives allowed before a non-grasped object enters recovery")
    parser.add_argument("--close-steps", type=int, default=2,
                        help="Consecutive close primitives before the option controller permits lifting")
    parser.add_argument("--align-grid-context", action="store_true",
                        help="Use a 5x5 observable xy-error code while the deployed option is align")
    parser.add_argument("--align-residual-grid-context", action="store_true",
                        help="Use a 12x12 10-mm xy residual code with explicit overflow bins for align")
    parser.add_argument("--align-residual-axis-context", action="store_true",
                        help="Add the teacher-relevant observable dominant xy-error axis to the residual grid")
    parser.add_argument("--align-residual-bins", type=int, default=ALIGN_RESIDUAL_BINS,
                        help="Residual bins per xy axis, including lower/upper overflow bins")
    parser.add_argument("--snn-align-only", action="store_true",
                        help="Route fixed descend/close/lift/recover actions through the FSM without SNN states or updates")
    parser.add_argument("--compact-striatum", action="store_true",
                        help="Use one D1 and one D2 LIF action-channel population instead of S x A expanded Str neurons")
    parser.add_argument("--align-action-count", type=int, choices=(4, 8), default=NUM_ACTION,
                        help="Number of SNN output actions during align; 4 keeps only +x/-x/+y/-y and leaves other primitives to the FSM")
    parser.add_argument("--fixed-pm-window", action="store_true",
                        help="Always integrate --max-internal-steps and select from PM spike counts")
    parser.add_argument("--decision-readout", choices=("pm", "thalamus"), default="pm",
                        help="Decode PM spikes or the upstream thalamic spike count; use only after the planar ablation")
    parser.add_argument("--pm-threshold", type=float, default=0.5,
                        help="PM IF threshold; default preserves the original network")
    parser.add_argument("--pm-lateral-gain", type=float, default=-2.5,
                        help="PM off-diagonal lateral inhibition; default preserves the original network")
    parser.add_argument("--coverage-audit", action="store_true",
                        help="Record teacher support and autonomous distribution shift without changing control")
    parser.add_argument("--coverage-distance-threshold", type=float, default=0.006,
                        help="Metres for matching autonomous continuous xy/z geometry to a teacher sample")
    parser.add_argument("--align-action-persistence", action="store_true",
                        help="Retain a non-confident PM align action until its xy grid state changes")
    parser.add_argument("--align-hold-decisions", type=int, default=4,
                        help="Maximum same-grid decisions retained by --align-action-persistence")
    parser.add_argument("--align-progress-persistence", action="store_true",
                        help="Reuse only a previous align direction proven to reduce xy error")
    parser.add_argument("--progress-memory-context", action="store_true",
                        help="Encode the validated align direction as DLPFC state; never override PM action")
    parser.add_argument("--factorized-progress-memory", action="store_true",
                        help="Co-activate shared xy-grid and progress-token DLPFC neurons instead of lookup pairs")
    parser.add_argument("--progress-direction-bias", action="store_true",
                        help="Add a measured-progress direction bias only to ambiguous PM align readouts")
    parser.add_argument("--progress-bias-learning-rate", type=float, default=0.15)
    parser.add_argument("--progress-bias-maximum", type=float, default=0.50)
    parser.add_argument("--align-progress-value", action="store_true",
                        help="Use persistent state-action xy-progress values only to resolve ambiguous PM align readouts")
    parser.add_argument("--progress-value-learning-rate", type=float, default=0.15)
    parser.add_argument("--progress-value-maximum", type=float, default=0.50)
    parser.add_argument("--progress-value-minimum-visits", type=int, default=1,
                        help="Physical state-action progress samples required before an ambiguous-PM value is used")
    parser.add_argument("--fixed-cube", action="store_true",
                        help="Use the reproducible curriculum cube start position")
    parser.add_argument("--deterministic-robot-start", action="store_true",
                        help="Disable Panda initial-joint noise for the curriculum")
    parser.add_argument("--fixed-cube-xy", nargs=2, type=float, default=(0.0088, 0.0069),
                        metavar=("X", "Y"))
    parser.add_argument("--timeout-penalty", type=float, default=-1.0)
    parser.add_argument("--stagnant-penalty", type=float, default=-0.10)
    parser.add_argument("--unsafe-penalty", type=float, default=-2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="results/lift_bdm_snn_baseline")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if sum((arguments.align_grid_context, arguments.align_residual_grid_context,
            arguments.align_residual_axis_context)) > 1:
        raise ValueError("select one align state encoder")
    if arguments.max_internal_steps is None:
        arguments.max_internal_steps = arguments.internal_steps
    if arguments.max_internal_steps < arguments.internal_steps:
        raise ValueError("--max-internal-steps must be at least --internal-steps")
    if arguments.align_action_count != NUM_ACTION and not (
            arguments.option_context and arguments.snn_align_only):
        raise ValueError("--align-action-count 4 requires --option-context and --snn-align-only")
    if arguments.validate_env:
        validate_environment(arguments.seed, arguments.control_steps)
    else:
        train(arguments)
