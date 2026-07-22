"""Summarize and plot the first RRR communication-subspace ablation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = RESULT_DIR / "rrr_striatum_to_output_seed0_600.json"
OUTPUT_FIGURE = RESULT_DIR / "rrr_striatum_to_output_seed0_600.png"


def load_result(rank):
    rewards = np.load(RESULT_DIR / f"lif_reward_striatum_rrr_delayed_rank{rank}_seed0.npy")
    scores = np.load(RESULT_DIR / f"lif_score_striatum_rrr_delayed_rank{rank}_seed0.npy")
    return {
        "rank": rank,
        "frames": int(scores.size),
        "updates": int(rewards.size),
        "score": int(scores[-1]),
        "reward_sum": float(rewards.sum()),
        "collision_count": int((rewards == -100).sum()),
        "pass_frames": (np.flatnonzero(np.diff(scores) > 0) + 2).astype(int).tolist(),
    }, rewards, scores


def main():
    results = []
    traces = {}
    for rank in (1, 2):
        result, rewards, scores = load_result(rank)
        results.append(result)
        traces[rank] = (rewards, scores)

    # The rank-0 full path comes from the identical seed-0 baseline run.
    baseline_rewards = np.load(RESULT_DIR / "lif_reward_rrr_rank0_seed0.npy")
    baseline_scores = np.load(RESULT_DIR / "lif_score_rrr_rank0_seed0.npy")
    baseline = {
        "rank": 0,
        "frames": int(baseline_scores.size),
        "updates": int(baseline_rewards.size),
        "score": int(baseline_scores[-1]),
        "reward_sum": float(baseline_rewards.sum()),
        "collision_count": int((baseline_rewards == -100).sum()),
        "pass_frames": (np.flatnonzero(np.diff(baseline_scores) > 0) + 2).astype(int).tolist(),
    }
    results.insert(0, baseline)
    OUTPUT_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    labels = ["full", "RRR k=1", "RRR k=2"]
    scores = [entry["score"] for entry in results]
    rewards = [entry["reward_sum"] for entry in results]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    axes[0].bar(labels, scores, color=colors)
    axes[0].set_ylabel("Score at 600-frame cap")
    axes[0].set_ylim(0, max(scores) + 1)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, rewards, color=colors)
    axes[1].set_ylabel("Cumulative reward")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Striatum-to-output RRR ablation (seed 0)")
    fig.savefig(OUTPUT_FIGURE, dpi=180)
    print(f"saved {OUTPUT_JSON}")
    print(f"saved {OUTPUT_FIGURE}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
