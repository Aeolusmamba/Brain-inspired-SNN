"""Plot the saved Flappy Bird baseline metrics without rerunning the model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_DIR = Path(__file__).resolve().parent
REWARD_PATH = RESULT_DIR / "lif_reward_l.npy"
SCORE_PATH = RESULT_DIR / "lif_score_l.npy"
OUTPUT_PATH = RESULT_DIR / "baseline_metrics_seed0.png"


def main():
    rewards = np.load(REWARD_PATH)
    scores = np.load(SCORE_PATH)
    frames = np.arange(1, scores.size + 1)
    updates = np.arange(1, rewards.size + 1)
    pass_frames = frames[1:][np.diff(scores) > 0]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    fig.suptitle("BDM-SNN Flappy Bird Baseline (seed 0, 600-frame cap)", fontsize=14)

    axes[0].step(frames, scores, where="post", color="#1f77b4", linewidth=2)
    axes[0].scatter(pass_frames, scores[pass_frames - 1], color="#d62728", zorder=3,
                    label="pipe pair passed")
    axes[0].set(xlabel="Game frame", ylabel="Score", xlim=(1, scores.size),
                ylim=(-0.2, scores.max() + 0.8))
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left")

    colors = np.where(rewards > 0, "#2ca02c", "#d62728")
    axes[1].bar(updates, rewards, color=colors, width=1.0)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Decision / reward update", ylabel="Immediate reward",
                xlim=(1, rewards.size))
    axes[1].grid(axis="y", alpha=0.25)

    cumulative_reward = np.cumsum(rewards)
    axes[2].plot(updates, cumulative_reward, color="#9467bd", linewidth=2)
    axes[2].set(xlabel="Decision / reward update", ylabel="Cumulative reward",
                xlim=(1, rewards.size))
    axes[2].grid(alpha=0.25)
    axes[2].annotate(f"final = {cumulative_reward[-1]:.0f}",
                     xy=(updates[-1], cumulative_reward[-1]), xytext=(-84, 12),
                     textcoords="offset points", arrowprops={"arrowstyle": "->"})

    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"saved {OUTPUT_PATH}")
    print(f"frames={scores.size} passed={int(scores[-1])} reward_sum={rewards.sum():.0f}")


if __name__ == "__main__":
    main()
