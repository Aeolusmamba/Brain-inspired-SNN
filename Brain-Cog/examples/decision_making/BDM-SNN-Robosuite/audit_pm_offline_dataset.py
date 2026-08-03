"""Audit a saved binary-latent PM teacher dataset and its deployed checkpoint.

This is deliberately a diagnostic, not a training entry point.  It reports
class/state support and replays every saved one-hot state through the saved
hard-spike model from reset membrane state, so a low average PM loss cannot
hide a collapsed deployed action readout.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from braincog.model_zoo.bdmsnn import BDMSNN  # noqa: E402


def build_network(checkpoint, device):
    """Recreate exactly the saved communication path before loading weights."""
    config = checkpoint["configuration"]
    network = BDMSNN(
        checkpoint["num_state"], checkpoint["num_action"], 1.0, -0.5, "lif",
        communication_rank=int(config["rrr_state_to_striatum_rank"]),
        communication_mode=("state_to_striatum" if config["rrr_state_to_striatum_rank"]
                            else "all_cross_core"),
        pm_threshold=config["pm_threshold"], pm_lateral_gain=config["pm_lateral_gain"],
        compact_striatum=checkpoint["compact_striatum"],
        striatum_latent_rank=int(config["nmf_striatum_latent_rank"]),
        striatum_latent_delta_transport=config["nmf_latent_delta_transport"],
        striatum_latent_quant_bits=config["nmf_latent_quant_bits"],
        striatum_latent_quant_scale=config["nmf_latent_quant_scale"],
        striatum_latent_spike_transport=config["nmf_latent_spike_transport"],
        striatum_latent_spike_threshold=config["nmf_latent_spike_threshold"],
        striatum_latent_spike_gain=config["nmf_latent_spike_gain"],
        striatum_latent_spike_decoder_gain=config["nmf_latent_spike_decoder_gain"],
        striatum_latent_spike_soft_reset=config["nmf_latent_spike_soft_reset"],
        striatum_latent_spike_normalize_factors=config["nmf_latent_spike_normalize_factors"],
        striatum_latent_spike_contrast_encode=config.get("nmf_latent_spike_contrast_encode", False),
        communication_window=config["rrr_window"],
        communication_warmup=config["rrr_warmup_decisions"],
        communication_refit_interval=config["rrr_refit_interval"],
    ).to(device)
    # NMF factors are buffers initialized empty until the first factorization.
    # Populate their saved shapes before strict checkpoint restoration.
    state = checkpoint["model_state_dict"]
    for pathway in ("d1", "d2"):
        setattr(network, f"_striatum_nmf_{pathway}_left",
                state[f"_striatum_nmf_{pathway}_left"].to(device).clone())
        setattr(network, f"_striatum_nmf_{pathway}_right",
                state[f"_striatum_nmf_{pathway}_right"].to(device).clone())
    network.load_state_dict(state)
    network.eval()
    return network


def reset_readout(network, source, internal_steps):
    """Run the hard deployed forward path from a clean episode membrane state."""
    network.reset()
    pm_count = torch.zeros(network.num_action, device=source.device)
    # BDMSNN also computes STDP eligibility with autograd during forward;
    # retain grad mode while detaching only the diagnostic spike count.
    for _ in range(internal_steps):
        pm_output, _ = network(source)
        pm_count += pm_output.detach()
    # Deterministic smallest-index tie break exposes an actual mode collapse.
    return int(pm_count.argmax().item()), pm_count.cpu().numpy().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--internal-steps", type=int, default=None)
    args = parser.parse_args()
    result_dir = args.result_dir
    checkpoint = torch.load(result_dir / "bdmsnn_final_state.pt", map_location="cpu")
    data = np.load(result_dir / "pm_offline_teacher_dataset.npz")
    sources, labels = data["sources"], data["labels"].astype(np.int64)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    network = build_network(checkpoint, device)
    internal_steps = args.internal_steps or checkpoint["configuration"]["max_internal_steps"]

    state_ids = sources.argmax(axis=1).astype(np.int64)
    label_counts = np.bincount(labels, minlength=checkpoint["num_action"])
    state_labels = defaultdict(Counter)
    for state, label in zip(state_ids, labels):
        state_labels[int(state)][int(label)] += 1
    predictions, pm_counts = [], []
    for source in sources:
        prediction, pm_count = reset_readout(
            network, torch.as_tensor(source, device=device), internal_steps)
        predictions.append(prediction)
        pm_counts.append(pm_count)
    predictions = np.asarray(predictions, dtype=np.int64)
    confusion = np.zeros((network.num_action, network.num_action), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    recalls = np.divide(np.diag(confusion), confusion.sum(axis=1),
                        out=np.full(network.num_action, np.nan), where=confusion.sum(axis=1) > 0)
    precision = np.divide(np.diag(confusion), confusion.sum(axis=0),
                          out=np.full(network.num_action, np.nan), where=confusion.sum(axis=0) > 0)
    f1 = np.divide(2 * precision * recalls, precision + recalls,
                   out=np.zeros(network.num_action), where=(precision + recalls) > 0)
    # A random sample split is leaky if each held-out state is also seen in train.
    rng = np.random.default_rng(checkpoint["configuration"]["seed"])
    validation = set(rng.permutation(len(labels))[:max(1, round(0.2 * len(labels)))].tolist())
    train_states = {int(state_ids[i]) for i in range(len(labels)) if i not in validation}
    validation_states = {int(state_ids[i]) for i in validation}
    report = {
        "samples": int(len(labels)),
        "unique_one_hot_states": int(len(state_labels)),
        "label_counts": label_counts.tolist(),
        "label_fractions": (label_counts / len(labels)).tolist(),
        "state_label_counts": {str(state): dict(counts) for state, counts in sorted(state_labels.items())},
        "states_with_conflicting_labels": int(sum(len(counts) > 1 for counts in state_labels.values())),
        "random_sample_validation": {
            "validation_state_count": len(validation_states),
            "states_also_in_training": len(validation_states.intersection(train_states)),
        },
        "reset_state_deployed_pm_readout": {
            "top1_accuracy": float((predictions == labels).mean()),
            "prediction_counts": np.bincount(predictions, minlength=network.num_action).tolist(),
            "confusion_rows_teacher_columns_pm": confusion.tolist(),
            "per_class_precision": precision.tolist(),
            "per_class_recall": recalls.tolist(),
            "macro_f1_present_classes": float(np.nanmean(f1[label_counts > 0])),
            "pm_counts_per_sample": pm_counts,
            "interpretation": (
                "Each saved state is replayed from reset neuron/residual state using the hard deployed "
                "binary latent path. This is a statewise diagnostic, not a multi-step control evaluation."),
        },
    }
    output = result_dir / "pm_offline_dataset_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "samples", "unique_one_hot_states", "label_counts", "states_with_conflicting_labels",
        "random_sample_validation", "reset_state_deployed_pm_readout")}, indent=2))
    print("saved:", output)


if __name__ == "__main__":
    main()
