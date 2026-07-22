"""Summarize extended all-cross-core RRR experiments for the BDM-SNN task."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = RESULT_DIR / "all_cross_core_extended_seed0_summary.json"
OUTPUT_FIGURE = RESULT_DIR / "all_cross_core_extended_seed0_summary.png"

RUNS = (
    ("Full", "extended_full_seed0_3000.json"),
    ("All k=1", "extended_all_cross_core_rank1_seed0_3000.json"),
    ("All k=2", "extended_all_cross_core_rank2_seed0_3000.json"),
    ("D1+D2 k=1", "extended_selected_striatum_rank1_seed0_3000.json"),
    ("Small loops", "extended_safe_small_rank1_seed0_3000.json"),
    ("Loops+Th", "extended_safe_plus_thalamus_rank1_seed0_3000.json"),
    ("Full seed 1", "extended_full_seed1_3000.json"),
    ("Loops+Th seed 1", "extended_safe_plus_thalamus_rank1_seed1_3000.json"),
)


def load_run(label, filename):
    data = json.loads((RESULT_DIR / filename).read_text(encoding="utf-8"))
    return {
        "label": label,
        "file": filename,
        "frames": data["frames"],
        "score": data["score"],
        "reward_sum": data["reward_sum"],
        "links": data["communication"]["links"],
    }


def traffic_proxy(run):
    """Summarize post-warm-up full source events versus latent scalar values."""
    full_events = 0
    latent_values = 0
    for link in run["links"].values():
        transport = link.get("transport")
        if transport is None:
            continue
        full_events += (transport["observed_source_logical_events"] -
                        transport["full_fallback_source_logical_events"])
        latent_values += transport["continuous_latent_scalar_values"]
    if full_events == 0:
        return None
    return {
        "post_warmup_source_logical_events": full_events,
        "post_warmup_continuous_latent_values": latent_values,
        "address_value_proxy_reduction": 1.0 - latent_values / full_events,
    }


def make_report(runs):
    report = {"runs": []}
    for run in runs:
        entry = {key: run[key] for key in ("label", "file", "frames", "score", "reward_sum")}
        entry["traffic_proxy"] = traffic_proxy(run)
        entry["links"] = {
            name: {
                "source_dim": values["source_dim"],
                "target_dim": values["target_dim"],
                "retained_variance": values.get("retained_variance"),
                "fitted": values.get("fitted"),
                "last_refit_status": values.get("transport", {}).get("last_refit_status"),
            }
            for name, values in run["links"].items()
        }
        report["runs"].append(entry)
    report["notes"] = [
        "The latent count is a continuous-value/address proxy, not AER event, packet, bit, or energy savings.",
        "All k=1/k=2 and D1+D2 runs are intentionally lossy online RRR tests; early collision is a control failure, not a measurement error.",
        "Small loops selects STN->GPe, STN->GPi, and GPe->STN; these rank-1 links retained 1.0 in the final fitting window and matched the full baseline for this seed.",
    ]
    return report


def plot(runs):
    labels = [run["label"] for run in runs]
    x = np.arange(len(runs))
    colors = ["#4c78a8"] + ["#e45756"] * 3 + ["#54a24b"] * 2 + ["#4c78a8", "#54a24b"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    axes[0].bar(x, [run["frames"] for run in runs], color=colors)
    axes[0].axhline(3001, color="black", lw=1, ls="--", label="3000-frame cap")
    axes[0].set_ylabel("Frames completed")
    axes[0].set_ylim(0, 3250)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].bar(x, [run["score"] for run in runs], color=colors)
    axes[1].set_ylabel("Final score")
    axes[1].set_ylim(0, 26)
    axes[1].grid(axis="y", alpha=0.25)

    proxy_runs = [run for run in runs if traffic_proxy(run) is not None]
    proxy_x = np.arange(len(proxy_runs))
    full = [traffic_proxy(run)["post_warmup_source_logical_events"] for run in proxy_runs]
    latent = [traffic_proxy(run)["post_warmup_continuous_latent_values"] for run in proxy_runs]
    width = 0.36
    axes[2].bar(proxy_x - width / 2, full, width, label="full source events")
    axes[2].bar(proxy_x + width / 2, latent, width, label="latent scalars")
    axes[2].set_ylabel("Post-warm-up count")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(fontsize=8)
    axes[2].set_xticks(proxy_x, [run["label"] for run in proxy_runs], rotation=20, ha="right")

    for axis in axes[:2]:
        axis.set_xticks(x, labels, rotation=20, ha="right")
    fig.suptitle("Extended all-cross-core RRR ablation (seed 0)")
    fig.savefig(OUTPUT_FIGURE, dpi=180)


def main():
    runs = [load_run(label, filename) for label, filename in RUNS]
    report = make_report(runs)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot(runs)
    print(f"saved {OUTPUT_JSON}")
    print(f"saved {OUTPUT_FIGURE}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
