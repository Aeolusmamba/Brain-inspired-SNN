"""Plot held-out target-current reconstruction for the RRR prototype.

This is an offline, decision-level diagnostic.  It deliberately uses the
first 200 reward-update samples to fit and the final 100 samples to evaluate,
which makes its explained-variance number distinct from the online RRR
module's latest internal-SNN-step calibration-window diagnostic.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_DIR = Path(__file__).resolve().parent
TELEMETRY_PATH = RESULT_DIR / "bdm_baseline_seed0_600.npz"
OUTPUT_JSON = RESULT_DIR / "rrr_target_current_heldout_seed0.json"
OUTPUT_FIGURE = RESULT_DIR / "rrr_target_current_heldout_seed0.png"
TRAIN_SAMPLES = 200


def structural_current(source_spikes):
    """Return the isolated StrD1/2-to-target current under the fixed map."""
    # Every state has two action-selective striatal neurons.  Each projects
    # only to its matching output neuron with the fixed inhibitory weight -0.5.
    weights = np.zeros((source_spikes.shape[1], 2), dtype=np.float64)
    weights[np.arange(source_spikes.shape[1]), np.arange(source_spikes.shape[1]) % 2] = -0.5
    return np.dot(source_spikes.astype(np.float64), weights)


def fit_rrr(source, target, rank):
    """Fit affine RRR in the same OLS-plus-target-SVD form as the module."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    ols = np.dot(np.linalg.pinv(centered_source), centered_target)
    predicted_target = np.dot(centered_source, ols)
    _, _, right = np.linalg.svd(predicted_target, full_matrices=False)
    target_vectors = right[:rank].T
    projection = np.dot(ols, target_vectors)
    decoder = target_vectors.T
    return source_mean, target_mean, projection, decoder


def predict_rrr(source, model):
    source_mean, target_mean, projection, decoder = model
    return np.dot(np.dot(source - source_mean, projection), decoder) + target_mean


def explained_variance(target, prediction):
    """Return channel-wise and pooled 1 - SSE/SST on a held-out set."""
    residual_sum = np.square(target - prediction).sum(axis=0)
    total_sum = np.square(target - target.mean(axis=0)).sum(axis=0)
    channel = np.where(total_sum > 1e-12, 1.0 - residual_sum / total_sum, np.nan)
    pooled = 1.0 - residual_sum.sum() / total_sum.sum()
    return channel, float(pooled)


def evaluate_link(source_spikes):
    target = structural_current(source_spikes)
    source_train, source_test = source_spikes[:TRAIN_SAMPLES], source_spikes[TRAIN_SAMPLES:]
    target_train, target_test = target[:TRAIN_SAMPLES], target[TRAIN_SAMPLES:]
    result = {"target_test": target_test, "prediction": {}, "channel_ev": {}, "pooled_ev": {}}
    for rank in (1, 2):
        prediction = predict_rrr(source_test, fit_rrr(source_train, target_train, rank))
        channel_ev, pooled_ev = explained_variance(target_test, prediction)
        result["prediction"][rank] = prediction
        result["channel_ev"][rank] = channel_ev
        result["pooled_ev"][rank] = pooled_ev
    return result


def plot_trace(axis, result, title):
    target = result["target_test"]
    prediction = result["prediction"][1]
    time = np.arange(TRAIN_SAMPLES + 1, TRAIN_SAMPLES + target.shape[0] + 1)
    colors = ("#4c78a8", "#f58518")
    for channel, color in enumerate(colors):
        axis.step(time, target[:, channel], where="mid", color=color, lw=1.8,
                  label=f"target ch{channel}")
        axis.step(time, prediction[:, channel], where="mid", color=color, lw=1.2,
                  ls="--", alpha=0.9, label=f"RRR k=1 ch{channel}")
    axis.set_title(title)
    axis.set_xlabel("Held-out reward update")
    axis.set_ylabel("Isolated target current")
    axis.set_ylim(-0.6, 0.1)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8, loc="lower left")


def plot_ev(axis, result, title):
    labels = ("ch0", "ch1", "pooled")
    k1 = np.r_[result["channel_ev"][1], result["pooled_ev"][1]]
    k2 = np.r_[result["channel_ev"][2], result["pooled_ev"][2]]
    x = np.arange(len(labels))
    width = 0.34
    axis.bar(x - width / 2, k1, width, color="#f58518", label="RRR k=1")
    axis.bar(x + width / 2, k2, width, color="#54a24b", label="RRR k=2")
    for offset, values in ((-width / 2, k1), (width / 2, k2)):
        for index, value in enumerate(values):
            axis.text(index + offset, value + 0.025, f"{value:.3f}", ha="center",
                      va="bottom", fontsize=8, rotation=90)
    axis.set_title(title)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Held-out explained variance")
    axis.set_ylim(0, 1.15)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, loc="lower right")


def serializable(link_result):
    return {
        "k1_channel_explained_variance": link_result["channel_ev"][1].tolist(),
        "k1_pooled_explained_variance": link_result["pooled_ev"][1],
        "k2_channel_explained_variance": link_result["channel_ev"][2].tolist(),
        "k2_pooled_explained_variance": link_result["pooled_ev"][2],
    }


def main():
    telemetry = np.load(TELEMETRY_PATH, allow_pickle=False)
    if telemetry["strd1_spike"].shape[0] <= TRAIN_SAMPLES:
        raise ValueError("telemetry must contain more samples than TRAIN_SAMPLES")

    direct = evaluate_link(telemetry["strd1_spike"])
    indirect = evaluate_link(telemetry["strd2_spike"])
    report = {
        "protocol": {
            "source": TELEMETRY_PATH.name,
            "fit_samples": [1, TRAIN_SAMPLES],
            "heldout_samples": [TRAIN_SAMPLES + 1, int(telemetry["strd1_spike"].shape[0])],
            "sample_unit": "one environment decision/reward update",
            "target_current": "striatum spike vector x fixed 18x2 structural weight",
            "explained_variance": "1 - SSE / SST on held-out samples; pooled sums over both target channels",
        },
        "StrD1_to_GPi": serializable(direct),
        "StrD2_to_GPe": serializable(indirect),
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), constrained_layout=True)
    plot_trace(axes[0, 0], direct, "StrD1 -> GPi: held-out target currents")
    plot_ev(axes[0, 1], direct, "StrD1 -> GPi: explained variance")
    plot_trace(axes[1, 0], indirect, "StrD2 -> GPe: held-out target currents")
    plot_ev(axes[1, 1], indirect, "StrD2 -> GPe: explained variance")
    fig.suptitle("Offline RRR target-current reconstruction (seed 0 baseline)", fontsize=14)
    fig.savefig(OUTPUT_FIGURE, dpi=180)

    print(f"saved {OUTPUT_JSON}")
    print(f"saved {OUTPUT_FIGURE}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
