"""Summarize paired full-communication and rank-1 RRR seed runs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = RESULT_DIR / "rrr_rank1_multiseed_600.json"
OUTPUT_FIGURE = RESULT_DIR / "rrr_rank1_multiseed_600.png"


def load_result(seed, rank):
    rewards = np.load(RESULT_DIR / f"lif_reward_striatum_rrr_multiseed_seed{seed}_rank{rank}.npy")
    scores = np.load(RESULT_DIR / f"lif_score_striatum_rrr_multiseed_seed{seed}_rank{rank}.npy")
    return {
        "seed": seed,
        "rank": rank,
        "frames": int(scores.size),
        "score": int(scores[-1]),
        "reward_sum": float(rewards.sum()),
        "collision_count": int((rewards == -100).sum()),
    }


def main():
    # Seed 0 uses the final delayed-calibration files; seeds 1 and 2 use
    # identically configured paired runs.
    results = [
        {"seed": 0, "rank": 0, "frames": 600, "score": 4, "reward_sum": 1662.0,
         "collision_count": 0},
        {"seed": 0, "rank": 1, "frames": 600, "score": 4, "reward_sum": 1662.0,
         "collision_count": 0},
    ]
    for seed in (1, 2):
        for rank in (0, 1):
            results.append(load_result(seed, rank))
    OUTPUT_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    seeds = np.array([0, 1, 2])
    full = [next(x for x in results if x["seed"] == seed and x["rank"] == 0) for seed in seeds]
    rank1 = [next(x for x in results if x["seed"] == seed and x["rank"] == 1) for seed in seeds]
    x = np.arange(seeds.size)
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    axes[0].bar(x - width / 2, [entry["score"] for entry in full], width, label="full")
    axes[0].bar(x + width / 2, [entry["score"] for entry in rank1], width, label="RRR k=1")
    axes[0].set(xticks=x, xticklabels=[str(seed) for seed in seeds], xlabel="Random seed",
                ylabel="Score", ylim=(0, 5))
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[1].bar(x - width / 2, [entry["frames"] for entry in full], width, label="full")
    axes[1].bar(x + width / 2, [entry["frames"] for entry in rank1], width, label="RRR k=1")
    axes[1].axhline(600, color="black", linestyle="--", linewidth=1, label="test cap")
    axes[1].set(xticks=x, xticklabels=[str(seed) for seed in seeds], xlabel="Random seed",
                ylabel="Frames completed", ylim=(0, 650))
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.suptitle("Full communication vs. rank-1 RRR (600-frame cap)")
    fig.savefig(OUTPUT_FIGURE, dpi=180)
    print(f"saved {OUTPUT_JSON}")
    print(f"saved {OUTPUT_FIGURE}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
