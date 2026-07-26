"""Continuous 2-D probe comparing hard bins against bilinear conjunctions."""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from braincog.model_zoo.bdmsnn import BDMSNN


GRID = np.linspace(-2.0, 2.0, 5)
NUM_ACTION = 4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def teacher_action(position):
    x, y = position
    if abs(x) >= abs(y) and abs(x) > 0.05:
        return 0 if x < 0 else 1
    if abs(y) > 0.05:
        return 2 if y < 0 else 3
    return None


def step(position, action):
    delta = np.zeros(2, dtype=np.float64)
    delta[action // 2] = 0.50 if action % 2 == 0 else -0.50
    return np.clip(np.asarray(position) + delta, -2.0, 2.0)


def bilinear(position):
    coordinate = np.clip(np.asarray(position) + 2.0, 0.0, 4.0)
    lower = np.floor(coordinate).astype(int)
    upper = np.minimum(lower + 1, 4)
    fraction = coordinate - lower
    result = Counter()
    for ix, wx in ((lower[0], 1.0 - fraction[0]), (upper[0], fraction[0])):
        for iy, wy in ((lower[1], 1.0 - fraction[1]), (upper[1], fraction[1])):
            result[int(ix + 5 * iy)] += wx * wy
    return tuple(result), np.asarray([result[key] for key in result], dtype=np.float32)


def encode(position, representation):
    if representation == "bilinear":
        return bilinear(position)
    nearest = np.abs(GRID[:, None] - np.asarray(position)).argmin(axis=0)
    return (int(nearest[0] + 5 * nearest[1]),), np.ones(1, dtype=np.float32)


def current(states, weights, device):
    value = torch.zeros(25, dtype=torch.float32, device=device)
    value[list(states)] = torch.as_tensor(2.0 * weights, device=device)
    return value


def pm_action(network, states, weights, device, rng):
    spikes = torch.zeros(NUM_ACTION, dtype=torch.float32, device=device)
    for _ in range(12):
        output, _ = network(current(states, weights, device))
        spikes += output.detach()
    ties = np.flatnonzero(spikes.detach().cpu().numpy() == spikes.max().item())
    return int(rng.choice(ties)), int(ties.size), int(spikes.sum().item() == 0)


def clone(network, states, weights, action, learning_mode):
    with torch.no_grad():
        for state, weight in zip(states, weights):
            action_slice = slice(state * NUM_ACTION, (state + 1) * NUM_ACTION)
            d1 = network.connection[0].weight[state, action_slice]
            d2 = network.connection[1].weight[state, action_slice]
            if learning_mode == "overwrite":
                d1.fill_(0.05)
                d2.fill_(network.weight_exc)
                d1[action] = network.weight_exc
                d2[action] = 0.05
            else:
                # A local teacher signal is weighted by this cell's bilinear
                # activation. Unlike row overwrite, shared neighbours retain
                # evidence from nearby states with different valid actions.
                rate = 0.08 * float(weight)
                target_d1 = torch.full_like(d1, 0.05)
                target_d2 = torch.full_like(d2, network.weight_exc)
                target_d1[action] = network.weight_exc
                target_d2[action] = 0.05
                d1.add_(rate * (target_d1 - d1))
                d2.add_(rate * (target_d2 - d2))


def run(args):
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    network = BDMSNN(25, NUM_ACTION, 1.0, -0.5, "lif",
                     communication_mode="all_cross_core").to(device)
    labels, conflicts = {}, 0
    for _ in range(args.teacher_episodes):
        position = rng.uniform(-2.0, 2.0, size=2)
        network.reset()
        for _ in range(args.max_steps):
            action = teacher_action(position)
            if action is None:
                break
            states, weights = encode(position, args.representation)
            for state in states:
                seen = labels.setdefault(state, set())
                conflicts += int(bool(seen and action not in seen))
                seen.add(action)
            pm_action(network, states, weights, device, rng)
            clone(network, states, weights, action, args.learning_mode)
            position = step(position, action)

    starts = [(x, y) for x in (-1.6, -0.8, 0.0, 0.8, 1.6)
              for y in (-1.6, -0.8, 0.0, 0.8, 1.6)]
    outcomes = []
    for start in starts:
        position = np.asarray(start, dtype=np.float64)
        network.reset()
        ties = silent = 0
        for steps in range(args.max_steps):
            if teacher_action(position) is None:
                break
            states, weights = encode(position, args.representation)
            action, tie_count, is_silent = pm_action(network, states, weights, device, rng)
            ties += tie_count
            silent += is_silent
            position = step(position, action)
        outcomes.append({"success": teacher_action(position) is None, "steps": steps,
                         "silent": silent, "mean_ties": ties / max(1, steps)})
    result = {
        "representation": args.representation, "learning_mode": args.learning_mode,
        "teacher_labeled_rows": len(labels),
        "teacher_conflicting_updates": conflicts,
        "success": int(sum(item["success"] for item in outcomes)), "total": len(outcomes),
        "mean_silent": float(np.mean([item["silent"] for item in outcomes])),
        "mean_ties": float(np.mean([item["mean_ties"] for item in outcomes])),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=("lookup", "bilinear"), required=True)
    parser.add_argument("--learning-mode", choices=("overwrite", "incremental"), default="overwrite")
    parser.add_argument("--teacher-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
