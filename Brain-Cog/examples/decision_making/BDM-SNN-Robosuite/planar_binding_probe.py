"""Minimal 2-D binding probe for BDM-SNN state representations.

The task is deliberately simpler than Lift: a point on a 5x5 grid must reach
the origin with four cardinal actions.  The teacher always chooses the axis
with larger absolute error.  This isolates whether a representation can bind
x and y evidence before the BDM-SNN action readout.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from braincog.model_zoo.bdmsnn import BDMSNN


GRID_VALUES = (-2, -1, 0, 1, 2)
NUM_ACTION = 4
ACTION_NAMES = ("+x", "-x", "+y", "-y")
AXIS_CELLS = 10  # 2 signs + 3 magnitude bins per axis
CONJUNCTION_CELLS = 25


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def teacher_action(position):
    x, y = position
    if abs(x) >= abs(y) and x:
        return 0 if x < 0 else 1
    if y:
        return 2 if y < 0 else 3
    return None


def apply_action(position, action):
    x, y = position
    if action == 0:
        x = min(2, x + 1)
    elif action == 1:
        x = max(-2, x - 1)
    elif action == 2:
        y = min(2, y + 1)
    else:
        y = max(-2, y - 1)
    return x, y


def axis_cells(value, offset):
    sign = offset + int(value >= 0)
    magnitude = offset + 2 + (0 if abs(value) == 0 else 1 if abs(value) == 1 else 2)
    return sign, magnitude


def encode(position, representation):
    x, y = position
    if representation == "lookup":
        return (GRID_VALUES.index(x) + 5 * GRID_VALUES.index(y),), 25
    x_cells = axis_cells(x, 0)
    y_cells = axis_cells(y, 5)
    if representation == "axis":
        return x_cells + y_cells, AXIS_CELLS
    conjunction = GRID_VALUES.index(x) + 5 * GRID_VALUES.index(y)
    # The conjunction cell is an explicit local AND of x/y bins.  We learn
    # only this cell, rather than incorrectly assigning whole actions to each
    # separately reusable axis feature.
    return (AXIS_CELLS + conjunction,), AXIS_CELLS + CONJUNCTION_CELLS


def input_current(states, num_state, device):
    current = torch.zeros(num_state, dtype=torch.float32, device=device)
    current[list(states)] = 2.0 / len(states)
    return current


def decision(network, states, num_state, device, rng):
    current = input_current(states, num_state, device)
    spikes = torch.zeros(NUM_ACTION, dtype=torch.float32, device=device)
    for _ in range(12):
        output, _ = network(current)
        spikes += output.detach()
    ties = np.flatnonzero(spikes.detach().cpu().numpy() == spikes.max().item())
    return int(rng.choice(ties)), int(ties.size), int(spikes.sum().item() == 0)


def clone_update(network, states, action, representation, off_weight=0.05):
    # Axis rows are deliberately independent and hence expose label conflicts.
    # The conjunctive representation updates its unique binding row only.
    learned_states = states if representation != "conjunctive" else states[-1:]
    with torch.no_grad():
        for state in learned_states:
            action_slice = slice(state * NUM_ACTION, (state + 1) * NUM_ACTION)
            d1 = network.connection[0].weight[state, action_slice]
            d2 = network.connection[1].weight[state, action_slice]
            d1.fill_(off_weight)
            d2.fill_(network.weight_exc)
            d1[action] = network.weight_exc
            d2[action] = off_weight


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    _, num_state = encode((0, 0), args.representation)
    network = BDMSNN(num_state, NUM_ACTION, 1.0, -0.5, "lif",
                     communication_mode="all_cross_core").to(device)
    conflicts = Counter()
    labels = {}
    teacher_steps = 0
    starts = [(x, y) for x in GRID_VALUES for y in GRID_VALUES]
    for _ in range(args.teacher_epochs):
        for position in starts:
            network.reset()
            for _ in range(args.max_steps):
                action = teacher_action(position)
                if action is None:
                    break
                states, _ = encode(position, args.representation)
                learned_states = states if args.representation != "conjunctive" else states[-1:]
                for state in learned_states:
                    previous = labels.setdefault(state, set())
                    conflicts[state] += int(bool(previous and action not in previous))
                    previous.add(action)
                # Preserve the BDM-SNN forward path during teacher training.
                decision(network, states, num_state, device, rng)
                clone_update(network, states, action, args.representation)
                position = apply_action(position, action)
                teacher_steps += 1

    evaluations = []
    for start_x in GRID_VALUES:
        for start_y in GRID_VALUES:
            start = (start_x, start_y)
            position = start
            network.reset()
            trace = []
            silent = ties = 0
            for _ in range(args.max_steps):
                if position == (0, 0):
                    break
                states, _ = encode(position, args.representation)
                action, tie_count, is_silent = decision(network, states, num_state, device, rng)
                trace.append(action)
                ties += tie_count
                silent += is_silent
                position = apply_action(position, action)
            evaluations.append({
                "start": start, "success": position == (0, 0), "steps": len(trace),
                "silent": silent, "mean_tie_count": ties / max(1, len(trace)),
                "trace": trace,
            })
    result = {
        "representation": args.representation,
        "num_state": num_state,
        "teacher_steps": teacher_steps,
        "teacher_labeled_rows": len(labels),
        "teacher_conflicting_updates": int(sum(conflicts.values())),
        "evaluation_success": int(sum(item["success"] for item in evaluations)),
        "evaluation_total": len(evaluations),
        "mean_evaluation_steps": float(np.mean([item["steps"] for item in evaluations])),
        "mean_pm_silent": float(np.mean([item["silent"] for item in evaluations])),
        "mean_pm_tie_count": float(np.mean([item["mean_tie_count"] for item in evaluations])),
        "evaluations": evaluations,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=("lookup", "axis", "conjunctive"), required=True)
    parser.add_argument("--teacher-epochs", type=int, default=1,
                        help="Complete teacher sweeps over all 25 grid starts")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
