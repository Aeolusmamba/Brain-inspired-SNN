"""Audit binary DLPFC--Str latent patterns from saved Lift decision traces."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def bit_pattern(record):
    """Flatten ordered D1/D2 binary slots, falling back to aggregate counts."""
    slots = record.get("latent_binary_pattern")
    if slots:
        values = []
        for pathway in ("d1", "d2"):
            for slot in slots.get(pathway, []):
                values.extend(int(float(value) > 0.0) for value in slot)
        if values:
            return tuple(values)
    events = record.get("latent_event_count", {})
    values = []
    for pathway in ("d1", "d2"):
        counts = events.get(pathway)
        values.extend([] if counts is None else [int(float(value) > 0.0) for value in counts])
    return tuple(values)


def hamming(left, right):
    if len(left) != len(right):
        raise ValueError(
            "Cannot compare latent patterns with different time widths; "
            "use a fixed decision window or pad the traces explicitly."
        )
    return sum(a != b for a, b in zip(left, right))


def mean_pairwise_distance(groups, same_group):
    distances = []
    for left_index, left in enumerate(groups):
        for right_index, right in enumerate(groups):
            if same_group and right_index <= left_index:
                continue
            if not same_group and right_index == left_index:
                continue
            distances.append(hamming(left, right))
    return float(np.mean(distances)) if distances else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--phase", default="evaluation", choices=("teacher", "dagger", "evaluation", "all"))
    args = parser.parse_args()
    trace_path = args.result_dir / "lift_bdm_snn_decision_traces.json"
    trajectories = json.loads(trace_path.read_text(encoding="utf-8"))
    records = [
        item for trajectory in trajectories
        if args.phase == "all" or trajectory["phase"] == args.phase
        for item in trajectory["trace"]
        if item.get("option") == "align" and item.get("teacher_action") is not None
    ]
    by_label, by_state_action = defaultdict(list), defaultdict(list)
    patterns = []
    for record in records:
        pattern = bit_pattern(record)
        if not pattern:
            continue
        label = int(record["teacher_action"])
        state = tuple(record["state"])
        patterns.append(pattern)
        by_label[label].append(pattern)
        by_state_action[(state, label)].append(pattern)

    unique_patterns = set(patterns)
    pattern_labels = defaultdict(set)
    for label, values in by_label.items():
        for pattern in values:
            pattern_labels[pattern].add(label)
    collisions = {pattern: labels for pattern, labels in pattern_labels.items() if len(labels) > 1}
    class_distances = {
        str(label): mean_pairwise_distance(values, same_group=True)
        for label, values in sorted(by_label.items())
    }
    cross_distances = {}
    labels = sorted(by_label)
    for offset, left_label in enumerate(labels):
        for right_label in labels[offset + 1:]:
            values = [hamming(left, right) for left in by_label[left_label]
                      for right in by_label[right_label]]
            cross_distances[f"{left_label}_{right_label}"] = float(np.mean(values)) if values else None
    per_state_action = {}
    for (state, label), values in sorted(by_state_action.items()):
        counts = Counter(values)
        per_state_action[f"{state}|{label}"] = {
            "samples": len(values), "unique_patterns": len(counts),
            "dominant_pattern_fraction": max(counts.values()) / len(values),
            "mean_hamming_distance": mean_pairwise_distance(values, same_group=True),
        }
    report = {
        "source": str(trace_path), "phase": args.phase, "align_labeled_records": len(records),
        "binary_pattern_width": len(patterns[0]) if patterns else 0,
        "unique_patterns": len(unique_patterns),
        "cross_label_pattern_collisions": len(collisions),
        "cross_label_collision_fraction": len(collisions) / len(unique_patterns) if unique_patterns else None,
        "mean_hamming_distance_within_teacher_action": class_distances,
        "mean_hamming_distance_between_teacher_actions": cross_distances,
        "state_action_pattern_stability": per_state_action,
        "interpretation": (
            "Patterns preserve the ordered binary D1/D2 latent events over one fixed decision window. "
            "This audit diagnoses code collisions and residual-phase sensitivity; it does not measure "
            "packet bits or establish a nonlinear manifold."),
    }
    output = args.result_dir / "latent_pattern_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "state_action_pattern_stability"}, indent=2))
    print("saved:", output)


if __name__ == "__main__":
    main()
