"""Train-only diagnostic for TEST_007 final role constraint failures."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.test_007 import finalize_test_007_specialization as finalizer
from src.test_007 import run_test_007_evidence_fast as fast


def diagnose(artifact_root: Path, config_path: Path, output: Path) -> Path:
    root = artifact_root.resolve()
    goal = finalizer.load_config(config_path.resolve())
    data = goal["data"]
    raw = (finalizer.REPO_ROOT / data["raw_dir"]).resolve()
    forbidden = finalizer._trusted_forbidden()
    train_path = finalizer._guard_not_forbidden(
        raw / data["train_file"], forbidden, "Train"
    )
    train = pd.read_csv(train_path)
    features = train.drop(columns=[data["target_column"], data["id_column"]])
    fast.GROUP_TRAIN_HASH = finalizer._sha256(train_path)
    fast.initialize_group_context(features)
    finalizer.nested._fold_safe_groups = fast.checkpointed_fold_safe_groups
    finalizer._masked_paired_bootstrap = fast.fast_masked_paired_bootstrap
    finalizer.paired_bootstrap = fast.fast_paired_bootstrap

    classes = finalizer._guarded_json(root / "class_names.json", forbidden, "classes")
    assignments = finalizer._normalize_alignment(
        pd.read_csv(root / "fold_assignments.csv")
    )
    _, banks = finalizer._load_base_manifest(
        root, classes, assignments, forbidden
    )
    records = finalizer._selection_records(root)
    with np.load(root / "nested_strategy_probability.npz", allow_pickle=False) as archive:
        stored = {
            lane: finalizer._probability(
                archive[lane], (3, len(train), len(classes))
            )
            for lane in finalizer.LANES
        }
    labels = finalizer._encode_labels(train, data["target_column"], classes)
    replay = finalizer._replay_nested(
        banks, records, assignments, stored, labels, []
    )
    verified = finalizer._verify_nested_metrics(
        root, stored, replay, records, assignments, labels
    )
    _, _, constraints, collapse = finalizer._slice_evidence(
        train,
        data["target_column"],
        data["id_column"],
        labels,
        classes,
        assignments,
        replay,
    )

    stats = {}
    collapse_summary = {}
    for lane in finalizer.LANES:
        summary = verified["summaries"][lane]
        collision, net_rescue, collision_details = finalizer._collision_evidence(
            records, lane, banks, replay[lane], assignments, labels
        )
        variants = [
            records[(lane, seed, fold)]["selected_variant"]
            for seed in finalizer.SEEDS
            for fold in finalizer.FOLDS
        ]
        complexity = sum(
            (100 if variant["kind"] == "collision_gate" else 0)
            + len(
                variant.get(
                    "models", variant.get("backbone_variant", {}).get("models", [])
                )
            )
            for variant in variants
        )
        stats[lane] = {
            "mean": float(summary["mean_macro_f1"]),
            "std": float(summary["std_macro_f1"]),
            "se": float(summary["std_macro_f1"]) / math.sqrt(3),
            "gap": float(np.mean(verified["gaps"][lane])),
            "complexity": complexity,
            "worst_floor": collapse[lane]["worst_floor"],
            "no_collapse": collapse[lane]["no_collapse"],
            "collision_evidence": collision,
            "net_rescue": net_rescue,
            "best_collision_pair": collision_details.get("ordered_pair"),
            "collision_qualifies": collision_details.get("qualifies"),
        }
        repeated_evidence = {}
        for dimension, dimension_data in collapse[lane]["by_dimension"].items():
            repeated_evidence[dimension] = {
                name: dimension_data["evidence"][name]
                for name in dimension_data["repeated"]
            }
        collapse_summary[lane] = {
            "no_collapse": collapse[lane]["no_collapse"],
            "repeated_slices": collapse[lane]["repeated_slices"],
            "repeated_evidence": repeated_evidence,
        }
    payload = {
        "schema_version": 1,
        "selection_uses_test": False,
        "constraint_summary": constraints,
        "no_collapse_lanes": [
            lane for lane in finalizer.LANES if collapse[lane]["no_collapse"]
        ],
        "candidate_statistics": stats,
        "collapse": collapse_summary,
    }
    finalizer._atomic_json(output.resolve(), payload)
    return output.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(diagnose(Path(args.artifact_root), Path(args.config), Path(args.output)))


if __name__ == "__main__":
    main()
