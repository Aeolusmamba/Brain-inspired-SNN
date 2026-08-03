"""Audit first errors and PM readout ambiguity in saved autonomous Lift traces."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def summarize(records):
    """Summarize agreement and PM ties for a homogeneous record group."""
    if not records:
        return {"decisions": 0}
    pm = np.asarray([record["pm_spike_count"] for record in records], dtype=float)
    maxima = pm.max(axis=1)
    ties = (pm == maxima[:, None]).sum(axis=1)
    agreement = [record["network_action"] == record["teacher_action"] for record in records]
    return {
        "decisions": len(records),
        "teacher_agreement_rate": float(np.mean(agreement)),
        "unique_pm_winner_rate": float(np.mean(ties == 1)),
        "mean_pm_tie_count": float(np.mean(ties)),
        "zero_pm_window_rate": float(np.mean(maxima == 0.0)),
    }


def episode_successes(metrics):
    """Recover evaluation success labels, which are stored outside decision traces."""
    return {
        item["episode"]: bool(item.get("sparse_success", False))
        for item in metrics["episodes"]
        if item.get("phase") == "evaluation"
    }


def first_error(records):
    for record in records:
        if record["network_action"] != record["teacher_action"]:
            pm = np.asarray(record["pm_spike_count"], dtype=float)
            tie_count = int(np.sum(pm == pm.max()))
            return {
                "decision": record["decision"],
                "state": record["state"],
                "teacher_action": record["teacher_action"],
                "network_action": record["network_action"],
                "pm_tie_count": tie_count,
                "xy_error_before_m": record.get("xy_error_before_m"),
                "xy_error_after_m": record.get("xy_error_after_m"),
            }
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--phase", default="evaluation", choices=("teacher", "dagger", "evaluation"))
    args = parser.parse_args()

    metrics = json.loads((args.result_dir / "lift_bdm_snn_metrics.json").read_text(encoding="utf-8"))
    traces = json.loads((args.result_dir / "lift_bdm_snn_decision_traces.json").read_text(encoding="utf-8"))
    success_by_episode = episode_successes(metrics)
    episodes = [item for item in traces if item["phase"] == args.phase]
    labeled = []
    first_errors, by_outcome = [], {"success": [], "failure": []}
    by_state_action = defaultdict(list)

    for episode in episodes:
        records = [
            item for item in episode["trace"]
            if item.get("option") == "align" and item.get("teacher_action") is not None
        ]
        labeled.extend(records)
        outcome = "success" if success_by_episode.get(episode["episode"], False) else "failure"
        by_outcome[outcome].extend(records)
        error = first_error(records)
        first_errors.append({"episode": episode["episode"], "outcome": outcome, "first_error": error})
        for record in records:
            by_state_action[(tuple(record["state"]), int(record["teacher_action"]))].append(record)

    state_action = {}
    for (state, action), records in sorted(by_state_action.items()):
        state_action[f"{state}|{action}"] = summarize(records)

    report = {
        "source": str(args.result_dir),
        "phase": args.phase,
        "all_align_decisions": summarize(labeled),
        "by_final_outcome": {key: summarize(value) for key, value in by_outcome.items()},
        "first_errors": first_errors,
        "first_error_state_action_counts": Counter(
            f"{tuple(error['first_error']['state'])}|{error['first_error']['teacher_action']}"
            for error in first_errors if error["first_error"] is not None),
        "by_state_teacher_action": state_action,
        "interpretation": (
            "Teacher actions are counterfactual diagnostic labels during autonomous evaluation. "
            "This report distinguishes first policy disagreements and PM tie readout ambiguity; "
            "it does not establish causality or measure latent-code separation."),
    }
    output = args.result_dir / "closed_loop_failure_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "source": report["source"], "phase": report["phase"],
        "all_align_decisions": report["all_align_decisions"],
        "by_final_outcome": report["by_final_outcome"],
        "first_errors": report["first_errors"],
        "first_error_state_action_counts": report["first_error_state_action_counts"],
    }, indent=2))
    print("saved:", output)


if __name__ == "__main__":
    main()
