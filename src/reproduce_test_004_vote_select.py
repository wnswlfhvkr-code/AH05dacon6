"""Reproduce TEST_004_8/9 from a compact, versioned OOF probability bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import yaml

from src.models.baseline.TEST_004_8 import SPEC as SPEC_8
from src.models.baseline.TEST_004_9 import SPEC as SPEC_9
from src.pipelines.preprocessing_registry import create_preprocessing_pipeline


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def learned_mapping(matrix, y, mask, class_count, min_count, shrinkage):
    counts = np.bincount(y[mask], minlength=class_count)
    weights = mask.sum() / (class_count * np.maximum(counts, 1))
    groups: dict[tuple[int, ...], list] = {}
    for index in np.where(mask)[0]:
        key = tuple(map(int, matrix[index]))
        group = groups.setdefault(key, [0, {}])
        group[0] += 1
        for label in set(key):
            group[1][label] = group[1].get(label, 0.0) + (
                (y[index] == label) * weights[y[index]]
            )
    mapping = {}
    for key, (count, utility) in groups.items():
        if count < min_count:
            continue
        label = max(utility, key=utility.get)
        if label != key[0] and utility[label] > utility.get(key[0], 0.0) + shrinkage:
            mapping[key] = int(label)
    return mapping


def apply_mapping(base, matrix, mapping, mask=None):
    result = base.copy()
    indices = np.arange(len(base)) if mask is None else np.where(mask)[0]
    for index in indices:
        result[index] = mapping.get(tuple(map(int, matrix[index])), int(base[index]))
    return result


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def id_hash(frame: pd.DataFrame) -> str:
    payload = "\n".join(frame["ID"].astype(str)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    experiment = config["model"]["spec"]
    spec = {"TEST_004_8": SPEC_8, "TEST_004_9": SPEC_9}.get(experiment)
    if spec is None:
        raise ValueError("This runner supports only TEST_004_8 and TEST_004_9")

    pipeline = create_preprocessing_pipeline(dict(config["preprocessing"]))
    if pipeline.name != spec["pipeline_name"]:
        raise ValueError("config, model spec, and pipeline name do not match")
    if args.check_only:
        print(f"{experiment}: config/model/{spec['pipeline_file']} linkage OK")
        return

    bundle_path = ROOT / config["artifacts"]["probability_bundle"]
    manifest_path = bundle_path.with_suffix(".json")
    if not bundle_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "Probability bundle is missing. Run: python -m src.build_test_004_vote_bundle"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if file_hash(bundle_path) != manifest["sha256"]:
        raise ValueError("Probability bundle SHA-256 mismatch")

    data = config["data"]
    raw = ROOT / data["raw_dir"]
    train = pd.read_csv(raw / data["train_file"], usecols=[data["id_column"]])
    test = pd.read_csv(raw / data["test_file"], usecols=[data["id_column"]])
    sample = pd.read_csv(raw / data["submission_file"])
    if id_hash(train) != manifest["train_id_sha256"] or id_hash(test) != manifest["test_id_sha256"]:
        raise ValueError("Raw data row IDs do not match the probability bundle")
    if not sample[data["id_column"]].astype(str).equals(test[data["id_column"]].astype(str)):
        raise ValueError("sample_submission and test ID order do not match")

    bundle = np.load(bundle_path)
    y = bundle["y"].astype(np.int16)
    folds = bundle["fold_ids"].astype(np.int8)
    classes = bundle["classes"].astype(str)
    prefix = spec["selector_version"]
    base_oof = bundle[f"{prefix}_base_oof"].astype(np.int16)
    base_test = bundle[f"{prefix}_base_test"].astype(np.int16)
    full_version = spec["full_meta_version"]
    oof_matrix = np.column_stack([
        base_oof,
        bundle[f"full_{full_version}_oof"].argmax(axis=1),
        bundle["lgb_oof"].argmax(axis=1),
        bundle["corrected_oof"].argmax(axis=1),
        bundle["cpem_oof"].argmax(axis=1),
    ])
    test_matrix = np.column_stack([
        base_test,
        bundle[f"full_{full_version}_test"].argmax(axis=1),
        bundle["lgb_test"].argmax(axis=1),
        bundle["corrected_test"].argmax(axis=1),
        bundle["cpem_test"].argmax(axis=1),
    ])

    min_count = int(spec["selector"]["min_count"])
    shrinkage = float(spec["selector"]["shrinkage"])
    oof_prediction = base_oof.copy()
    for fold in sorted(np.unique(folds)):
        mapping = learned_mapping(
            oof_matrix, y, folds != fold, len(classes), min_count, shrinkage
        )
        oof_prediction = apply_mapping(
            oof_prediction, oof_matrix, mapping, mask=folds == fold
        )
    full_mapping = learned_mapping(
        oof_matrix, y, np.ones(len(y), dtype=bool), len(classes), min_count, shrinkage
    )
    test_prediction = apply_mapping(base_test, test_matrix, full_mapping)

    output = ROOT / data["processed_dir"] / config["record"]["submission_file"]
    output.parent.mkdir(parents=True, exist_ok=True)
    sample[data["target_column"]] = classes[test_prediction]
    sample.to_csv(output, index=False, encoding="utf-8-sig")
    digest = file_hash(output)
    expected = spec["historical_submission_sha256"]
    if digest != expected:
        raise ValueError(f"Submission hash mismatch: expected {expected}, got {digest}")

    score = float(f1_score(y, oof_prediction, average="macro"))
    fold_scores = [
        float(f1_score(y[folds == fold], oof_prediction[folds == fold], average="macro"))
        for fold in sorted(np.unique(folds))
    ]
    result = {
        "experiment": experiment,
        "author": config["project"]["author"],
        "pipeline": spec["pipeline_file"],
        "oof_macro_f1": score,
        "fold_macro_f1": fold_scores,
        "public_macro_f1": spec["historical_public_macro_f1"],
        "mapping_count": len(full_mapping),
        "changed_oof": int(np.sum(oof_prediction != base_oof)),
        "changed_test": int(np.sum(test_prediction != base_test)),
        "submission": str(output),
        "submission_sha256": digest,
        "exact_historical_reproduction": True,
        "external_data_used": False,
        "test_used_for_fit": False,
        "test_labels_used": False,
    }
    metrics_path = output.parent / f"{experiment}_metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
