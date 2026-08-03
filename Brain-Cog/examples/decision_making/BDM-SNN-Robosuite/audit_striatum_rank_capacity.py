"""Audit whether a DLPFC--Str NMF rank can encode four lateral directions.

The report is static: it never alters a checkpoint or runs the robot.  It
repeats the deployed source-side normalization and contrast reference, then
checks both teacher-action prototype rows and the best bounded nonnegative
latent code for each action.  This separates a factor-capacity limit from a
closed-loop control failure.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT))


def nmf_factorize(target, rank, iterations):
    """Use the same deterministic multiplicative NMF initialization as BDMSNN."""
    target = target.clamp_min(0)
    left_svd, singular_values, right_svd = torch.linalg.svd(target, full_matrices=False)
    root = singular_values[:rank].sqrt()
    left = left_svd[:, :rank].abs() * root
    right = root[:, None] * right_svd[:rank].abs()
    scale = max(float(target.mean().item()), 1e-4) ** 0.5 * 1e-3
    left.clamp_min_(scale)
    right.clamp_min_(scale)
    epsilon = torch.finfo(target.dtype).eps
    for _ in range(iterations):
        right.mul_((left.T @ target) / ((left.T @ left) @ right + epsilon))
        left.mul_((target @ right.T) / (left @ (right @ right.T) + epsilon))
        left.clamp_min_(epsilon)
        right.clamp_min_(epsilon)
    return left, right


def deployed_factors(left, right):
    """Apply exactly the source normalization and contrast encoding in forward()."""
    eps = torch.finfo(left.dtype).eps
    scale = left.amax(dim=0).clamp_min(eps)
    encoded_left = left / scale
    decoder = right * scale[:, None]
    floor = encoded_left.amin(dim=0)
    encoded_left = (encoded_left - floor) / (encoded_left.amax(dim=0) - floor).clamp_min(eps)
    return encoded_left, decoder


def best_bounded_margin(decoder, action):
    """Maximize the target current minus every competing current for z in [0,1]^k."""
    decoder = decoder.detach().cpu().numpy()
    rank, actions = decoder.shape
    inequalities = []
    for competitor in range(actions):
        if competitor == action:
            continue
        # -(v_action-v_competitor) z + margin <= 0
        inequalities.append(np.r_[-(decoder[:, action] - decoder[:, competitor]), 1.0])
    result = linprog(np.r_[np.zeros(rank), -1.0], A_ub=np.asarray(inequalities),
                     b_ub=np.zeros(len(inequalities)),
                     bounds=[(0.0, 1.0)] * rank + [(None, None)], method="highs")
    if not result.success:
        return {"success": False, "message": result.message}
    return {
        "success": True,
        "margin": float(result.x[-1]),
        "latent_code": result.x[:rank].tolist(),
        "decoded_current": (result.x[:rank] @ decoder).tolist(),
    }


def analyse_pathway(left, right, labels, state_ids):
    encoded_left, decoder = deployed_factors(left, right)
    prototypes = []
    for action in range(4):
        rows = state_ids[labels == action]
        prototype = encoded_left[torch.as_tensor(rows)].mean(dim=0)
        current = prototype @ decoder
        prototypes.append({
            "samples": int(len(rows)),
            "latent": prototype.tolist(),
            "decoded_current": current.tolist(),
            "argmax_action": int(current.argmax().item()),
        })
    currents = torch.stack([torch.tensor(item["decoded_current"]) for item in prototypes])
    cosine = torch.nn.functional.cosine_similarity(
        currents[:, None, :], currents[None, :, :], dim=-1).tolist()
    return {
        "prototype_tokens": prototypes,
        "prototype_current_cosine": cosine,
        "best_bounded_nonnegative_margin": [best_bounded_margin(decoder, action)
                                               for action in range(4)],
        "residual_state_decoded_argmax_counts": torch.bincount(
            (encoded_left @ decoder).argmax(dim=1), minlength=decoder.shape[1]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--ranks", nargs="+", type=int, default=(3, 4, 5, 6, 8))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    data = np.load(args.dataset)
    labels = data["labels"].astype(np.int64)
    state_ids = data["sources"].argmax(axis=1).astype(np.int64)
    report = {
        "checkpoint": str(args.checkpoint), "dataset": str(args.dataset),
        "note": ("All results are static current-code diagnostics. Bounded latent codes are "
                 "continuous z in [0,1]^k, so a nonpositive margin rules out even the more "
                 "permissive continuous code; it does not claim binary spike feasibility."),
        "label_counts": np.bincount(labels, minlength=4).tolist(), "ranks": {},
    }
    for rank in args.ranks:
        rank_report = {}
        for pathway, connection_index in (("d1", 0), ("d2", 1)):
            target = state[f"connection.{connection_index}.weight"].float()
            if rank == state[f"_striatum_nmf_{pathway}_left"].shape[1]:
                left = state[f"_striatum_nmf_{pathway}_left"].float().clone()
                right = state[f"_striatum_nmf_{pathway}_right"].float().clone()
                source = "saved_checkpoint_factors"
            else:
                left, right = nmf_factorize(target, rank, args.iterations)
                source = f"fresh_nmf_{args.iterations}_iterations_from_effective_weight"
            reconstructed = left @ right
            energy = target.square().sum().clamp_min(torch.finfo(target.dtype).eps)
            rank_report[pathway] = {
                "factor_source": source,
                "effective_weight_matrix_rank": int(torch.linalg.matrix_rank(target).item()),
                "projection_nrmse": float(((target - reconstructed).square().sum() / energy).sqrt().item()),
                **analyse_pathway(left, right, labels, state_ids),
            }
        report["ranks"][str(rank)] = rank_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for rank, value in report["ranks"].items():
        for pathway, diagnostics in value.items():
            margins = [round(item.get("margin", float("nan")), 6)
                       for item in diagnostics["best_bounded_nonnegative_margin"]]
            argmaxes = [item["argmax_action"] for item in diagnostics["prototype_tokens"]]
            print(f"rank={rank} {pathway}: nrmse={diagnostics['projection_nrmse']:.6f} "
                  f"token_argmax={argmaxes} margins={margins}")
    print("saved:", args.output)


if __name__ == "__main__":
    main()
