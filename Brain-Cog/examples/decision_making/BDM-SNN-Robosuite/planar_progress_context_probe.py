"""Test an online progress-memory input population before adding it to Lift.

The 5x5 planar task retains the BDM-SNN basal-ganglia/PM forward path.  The
teacher writes only the position rows.  During deployment, a small context
population encodes a direction that just reduced Manhattan distance, the
remaining-distance bin, and a finite validity counter.  It is updated only
from physical progress and never receives a teacher action label.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from braincog.model_zoo.bdmsnn import BDMSNN


VALUES = (-2, -1, 0, 1, 2)
NUM_POSITION = 25
NUM_ACTION = 4
MEMORY_TTL = 3
NUM_MEMORY = 1 + NUM_ACTION * 2 * MEMORY_TTL  # none + direction x range x age


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def position_id(position):
    return VALUES.index(position[0]) + 5 * VALUES.index(position[1])


def teacher_action(position):
    x, y = position
    if abs(x) >= abs(y) and x:
        return 0 if x < 0 else 1
    if y:
        return 2 if y < 0 else 3
    return None


def step(position, action):
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


def distance(position):
    return abs(position[0]) + abs(position[1])


def memory_id(direction, remaining_distance, remaining_ttl):
    if direction is None or remaining_ttl <= 0:
        return 0
    range_bin = int(remaining_distance > 1)
    age = min(MEMORY_TTL, int(remaining_ttl)) - 1
    return 1 + direction * (2 * MEMORY_TTL) + range_bin * MEMORY_TTL + age


def input_current(position, memory, use_context, context_current, device):
    num_state = NUM_POSITION + (NUM_MEMORY if use_context else 0)
    active = [position_id(position)]
    if use_context and memory["direction"] is not None and memory["ttl"] > 0:
        active.append(NUM_POSITION + memory_id(memory["direction"], distance(position), memory["ttl"]))
    current = torch.zeros(num_state, dtype=torch.float32, device=device)
    # Preserve the trained position population's original drive.  A context
    # group is a modulatory population, not a reason to halve position
    # evidence merely because an additional sensory cell is active.
    current[active[0]] = 2.0
    if len(active) == 2:
        current[active[1]] = context_current
    return current, active


def clone_position(network, position, action):
    row = position_id(position)
    with torch.no_grad():
        segment = slice(row * NUM_ACTION, (row + 1) * NUM_ACTION)
        d1 = network.connection[0].weight[row, segment]
        d2 = network.connection[1].weight[row, segment]
        d1.fill_(0.05)
        d2.fill_(network.weight_exc)
        d1[action] = network.weight_exc
        d2[action] = 0.05


def decide(network, position, memory, args, device, rng):
    current, active = input_current(position, memory, args.use_context,
                                    args.context_current, device)
    counts = torch.zeros(NUM_ACTION, dtype=torch.float32, device=device)
    for _ in range(args.window):
        output, _ = network(current)
        counts += output.detach()
    values = counts.cpu().numpy()
    winners = np.flatnonzero(np.isclose(values, values.max()))
    action = int(rng.choice(winners))
    return action, active, {
        "pm_silent": bool(counts.sum().item() == 0),
        "pm_unique": bool(counts.sum().item() > 0 and winners.size == 1),
    }


def online_context_update(network, memory_row, action, progress, learning_rate):
    """Use local progress as a three-factor scalar for the active memory row."""
    if memory_row is None:
        return
    with torch.no_grad():
        index = memory_row * NUM_ACTION + action
        d1 = network.connection[0].weight[memory_row, index]
        d2 = network.connection[1].weight[memory_row, index]
        d1.add_(learning_rate * progress).clamp_(0.05, network.weight_exc)
        d2.add_(-learning_rate * progress).clamp_(0.05, network.weight_exc)


def run(args):
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    num_state = NUM_POSITION + (NUM_MEMORY if args.use_context else 0)
    network = BDMSNN(num_state, NUM_ACTION, 1.0, -0.5, "lif",
                     communication_mode="all_cross_core", pm_threshold=args.pm_threshold,
                     pm_lateral_gain=args.pm_lateral_gain).to(device)
    starts = [(x, y) for x in VALUES for y in VALUES]

    # Teacher labels only the position population, never the progress context.
    for _ in range(args.teacher_epochs):
        for start in starts:
            position = start
            network.reset()
            for _ in range(args.max_steps):
                action = teacher_action(position)
                if action is None:
                    break
                clone_position(network, position, action)
                position = step(position, action)

    episodes = []
    source = {"decisions": 0, "positive_progress": 0, "pm_unique": 0}
    for episode in range(args.episodes):
        position = starts[int(rng.integers(len(starts)))]
        memory = {"direction": None, "ttl": 0}
        network.reset()
        trace = []
        for _ in range(args.max_steps):
            if position == (0, 0):
                break
            action, active, detail = decide(network, position, memory, args, device, rng)
            old_distance = distance(position)
            next_position = step(position, action)
            progress = float(np.clip(old_distance - distance(next_position), -1, 1))
            memory_row = active[1] if len(active) == 2 else None
            if args.use_context:
                online_context_update(network, memory_row, action, progress,
                                      args.context_learning_rate)
            if progress > 0:
                memory = {"direction": action, "ttl": MEMORY_TTL}
            elif memory["ttl"] > 0:
                memory["ttl"] -= 1
                if memory["ttl"] == 0:
                    memory["direction"] = None
            source["decisions"] += 1
            source["positive_progress"] += int(progress > 0)
            source["pm_unique"] += int(detail["pm_unique"])
            trace.append({"position": position, "action": action, "progress": progress,
                          "memory_direction": memory["direction"], **detail})
            position = next_position
        episodes.append({"success": position == (0, 0), "steps": len(trace), "trace": trace})

    result = {
        "use_context": args.use_context,
        "num_state": num_state,
        "teacher_deployment_actions": 0,
        "episodes": args.episodes,
        "context_current": args.context_current if args.use_context else 0.0,
        "success": int(sum(item["success"] for item in episodes)),
        "mean_steps": float(np.mean([item["steps"] for item in episodes])),
        "positive_progress_rate": source["positive_progress"] / max(1, source["decisions"]),
        "pm_unique_rate": source["pm_unique"] / max(1, source["decisions"]),
        "decision_count": source["decisions"],
        "traces": episodes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "traces"}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-context", action="store_true")
    parser.add_argument("--teacher-epochs", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--context-learning-rate", type=float, default=0.12)
    parser.add_argument("--context-current", type=float, default=0.25,
                        help="Modulatory current added by the active progress-memory neuron")
    parser.add_argument("--pm-threshold", type=float, default=0.5)
    parser.add_argument("--pm-lateral-gain", type=float, default=-2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
