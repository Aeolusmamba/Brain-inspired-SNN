"""Create PPT-ready summary plots from completed no-teacher Lift logs.

The figure deliberately reports only the 50 evaluation episodes per seed, so
the teacher-controlled curriculum episodes cannot inflate the displayed rate.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "results"


def load_evaluation(pattern):
    """Load the autonomous evaluation records from one result family."""
    records_by_seed = {}
    for path in sorted(RESULTS.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["configuration"]["seed"])
        records_by_seed[seed] = [
            episode for episode in payload["episodes"] if episode["phase"] == "evaluation"
        ]
    if sorted(records_by_seed) != [0, 1, 2]:
        raise RuntimeError("expected completed seed 0, 1, 2 logs for " + pattern)
    return records_by_seed


def success_rate(records):
    return 100.0 * np.mean([item["sparse_success"] for item in records])


def stage_rate(records, stage):
    return 100.0 * np.mean([
        item["stage_milestones"]["first_reached_decision"][stage] is not None
        for item in records
    ])


def style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)


def label_bars(axis, bars, decimals=0):
    for bar in bars:
        height = bar.get_height()
        axis.text(bar.get_x() + bar.get_width() / 2, height + 1.8,
                  f"{height:.{decimals}f}%", ha="center", va="bottom", fontsize=10)


def create_figure(original, weak_pm_180, best_240, output):
    original_all = [item for records in original.values() for item in records]
    weak_180_all = [item for records in weak_pm_180.values() for item in records]
    best_all = [item for records in best_240.values() for item in records]

    colors = {"original": "#A7A7A7", "weak": "#4C78A8", "best": "#2E8B57"}
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    figure.suptitle("BDM-SNN Panda Lift: autonomous online-learning result", fontsize=19,
                    fontweight="bold")

    # Panel A: causal sequence of tested configurations.
    axis = axes[0, 0]
    labels = ["Original PM\n180 decisions", "Weak PM inhibition\n180 decisions",
              "Weak PM inhibition\n240 decisions"]
    values = [success_rate(original_all), success_rate(weak_180_all), success_rate(best_all)]
    bars = axis.bar(labels, values, color=[colors["original"], colors["weak"], colors["best"]], width=0.62)
    label_bars(axis, bars, decimals=1)
    axis.set(title="A. Lift success after the teacher is removed", ylabel="Success rate (%)", ylim=(0, 100))
    style_axis(axis)
    axis.text(1, 8, "PM lateral inhibition: -2.5 -> -0.1", ha="center", fontsize=9,
              color="#1E4E79")

    # Panel B: task-chain reachability for the best configuration.
    axis = axes[0, 1]
    stages = ["Descend", "Close", "Lift option", "Lift success"]
    values = [stage_rate(best_all, "descend"), stage_rate(best_all, "close"),
              stage_rate(best_all, "lift"), success_rate(best_all)]
    bars = axis.bar(stages, values, color=["#73A7D8", "#73A7D8", "#73A7D8", colors["best"]])
    label_bars(axis, bars, decimals=1)
    axis.set(title="B. Best configuration: stage completion", ylabel="Episodes reaching stage (%)",
             ylim=(0, 110))
    style_axis(axis)

    # Panel C: independent-seed reproducibility.
    axis = axes[1, 0]
    seed_rates = [success_rate(best_240[seed]) for seed in range(3)]
    bars = axis.bar(["Seed 0", "Seed 1", "Seed 2"], seed_rates, color=colors["best"], width=0.55)
    label_bars(axis, bars, decimals=0)
    axis.axhline(np.mean(seed_rates), color="#222222", linestyle="--", linewidth=1.2,
                 label=f"Mean = {np.mean(seed_rates):.1f}%")
    axis.set(title="C. Reproducibility across random seeds", ylabel="Success rate (%)", ylim=(0, 100))
    axis.legend(frameon=False, loc="lower left")
    style_axis(axis)

    # Panel D: five non-overlapping 10-episode windows within the no-teacher phase.
    axis = axes[1, 1]
    windows = np.arange(1, 6)
    for seed, records in best_240.items():
        rates = [success_rate(records[index:index + 10]) for index in range(0, 50, 10)]
        axis.plot(windows, rates, marker="o", linewidth=2, label=f"Seed {seed}")
    axis.set_xticks(windows, ["1-10", "11-20", "21-30", "31-40", "41-50"])
    axis.set(title="D. No-teacher success across evaluation windows",
             xlabel="Autonomous episode window", ylabel="Success rate (%)", ylim=(0, 105))
    axis.legend(frameon=False, ncol=3, loc="lower left")
    style_axis(axis)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=RESULTS / "lift_bdm_snn_ppt" / "lift_best_autonomous_summary")
    args = parser.parse_args()
    original = load_evaluation("lift_bdm_snn_alignaxis_online_progressvalue3_sourceaudit_seed*/lift_bdm_snn_metrics.json")
    weak_pm_180 = load_evaluation("lift_bdm_snn_alignaxis_online_pmrecon_gm01_seed*/lift_bdm_snn_metrics.json")
    best_240 = load_evaluation("lift_bdm_snn_alignaxis_online_pmrecon_budget240_seed*/lift_bdm_snn_metrics.json")
    create_figure(original, weak_pm_180, best_240, args.output)
    print("saved:", args.output.with_suffix(".png"))
    print("saved:", args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
