"""Checkpointed Train-only Linear SVM search for TEST_007."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import yaml

from src.models.linear_svc_model import create_model
from src.pipelines.preprocessing_registry import create_preprocessing_pipeline
from src.test_007.run_test_007_elasticnet_set import (
    ControlledStop,
    _atomic_json,
    _atomic_npz,
    _canonical_hash,
    _file_hash,
    _resolve,
    _validate_fold_assignments,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1


def _validate_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not trials:
        raise ValueError("Linear SVM trial list is empty.")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in trials:
        name = str(raw.get("name", ""))
        regularization = float(raw["C"])
        if not name or not name.replace("_", "").isalnum() or name in names:
            raise ValueError(f"Invalid or duplicate trial name: {name!r}")
        if regularization <= 0:
            raise ValueError(f"Linear SVM C must be positive: {raw}")
        names.add(name)
        normalized.append({"name": name, "C": regularization})
    return normalized


def _load_contract(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config.get("data", {})
    if {"test_file", "submission_file", "test", "submission"}.intersection(data_config):
        raise ValueError("Linear SVM search is Train-only; Test/submission keys are forbidden.")
    train_path = _resolve(Path(data_config["raw_dir"]) / data_config["train_file"])
    fold_path = _resolve(config["validation"]["fold_assignments"])
    trials = _validate_trials(config["search"]["trials"])
    source_paths = [
        Path(__file__).resolve(),
        (REPO_ROOT / "src/models/linear_svc_model.py").resolve(),
        (REPO_ROOT / "src/test_007/run_test_007_elasticnet_set.py").resolve(),
        (REPO_ROOT / "src/pipelines/jyp_preprocessing/pipeline_pipe_comb_v3.py").resolve(),
        (REPO_ROOT / "src/pipelines/preprocessing_registry.py").resolve(),
    ]
    contract = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": _file_hash(config_path),
        "train_hash": _file_hash(train_path),
        "fold_assignments_hash": _file_hash(fold_path),
        "source_hashes": {str(path): _file_hash(path) for path in source_paths},
        "seed": int(config["project"]["seed"]),
        "folds": int(config["validation"]["folds"]),
        "trials": trials,
        "preprocessing": config["preprocessing"],
        "test_accessed": False,
    }
    return config, contract


def _fit_fold(
    config: dict[str, Any],
    trials: list[dict[str, Any]],
    features: pd.DataFrame,
    labels: pd.Series,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    global_encoder: LabelEncoder,
) -> dict[str, np.ndarray]:
    preprocessor = create_preprocessing_pipeline(config["preprocessing"])
    train_x = preprocessor.fit_transform(features.iloc[train_index], labels.iloc[train_index])
    valid_x = preprocessor.transform(features.iloc[valid_index])
    train_y = preprocessor.encode_labels(labels.iloc[train_index])
    valid_y = preprocessor.encode_labels(labels.iloc[valid_index])

    predictions = np.empty((len(trials), len(valid_index)), dtype=np.int32)
    train_scores = np.empty(len(trials), dtype=np.float64)
    valid_scores = np.empty(len(trials), dtype=np.float64)
    iterations = np.empty(len(trials), dtype=np.int32)
    convergence_warnings = np.empty(len(trials), dtype=np.int32)
    for trial_index, trial in enumerate(trials):
        model = create_model({**config["model"], "C": trial["C"]}, int(config["project"]["seed"]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_x, train_y)
        train_prediction = model.predict(train_x)
        valid_prediction = model.predict(valid_x)
        decoded = preprocessor.decode_labels(valid_prediction)
        predictions[trial_index] = global_encoder.transform(decoded).astype(np.int32)
        train_scores[trial_index] = f1_score(train_y, train_prediction, average="macro")
        valid_scores[trial_index] = f1_score(valid_y, valid_prediction, average="macro")
        iterations[trial_index] = int(np.max(model.n_iter_))
        convergence_warnings[trial_index] = sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
    return {
        "valid_index": valid_index.astype(np.int32),
        "predictions": predictions,
        "train_scores": train_scores,
        "valid_scores": valid_scores,
        "iterations": iterations,
        "convergence_warnings": convergence_warnings,
    }


def _load_checkpoint(path: Path, receipt: dict[str, Any]) -> dict[str, np.ndarray]:
    if set(receipt) != {"sha256", "row_count"} or _file_hash(path) != receipt["sha256"]:
        raise RuntimeError(f"Checkpoint receipt mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as payload:
        result = {name: payload[name] for name in payload.files}
    if len(result["valid_index"]) != int(receipt["row_count"]):
        raise RuntimeError(f"Checkpoint row count mismatch: {path.name}")
    return result


def run_search(
    config_path: Path,
    *,
    fold_executor: Callable[..., dict[str, np.ndarray]] = _fit_fold,
    max_new_folds: int | None = None,
) -> dict[str, Any]:
    config, contract = _load_contract(config_path)
    checkpoint_dir = _resolve(config["runtime"]["checkpoint_dir"])
    result_dir = _resolve(config["runtime"]["result_dir"])
    report_path = _resolve(config["runtime"]["report_file"])
    manifest_path = checkpoint_dir / "manifest.json"
    contract_hash = _canonical_hash(contract)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if set(manifest) != {"schema_version", "contract", "contract_hash", "completed_folds"}:
            raise RuntimeError("Checkpoint manifest schema mismatch.")
        if manifest["contract"] != contract or manifest["contract_hash"] != contract_hash:
            raise RuntimeError("Checkpoint contract changed; refusing stale resume.")
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "contract": contract,
            "contract_hash": contract_hash,
            "completed_folds": {},
        }
        _atomic_json(manifest_path, manifest)

    data_config = config["data"]
    train = pd.read_csv(_resolve(Path(data_config["raw_dir"]) / data_config["train_file"]))
    assignments = pd.read_csv(_resolve(config["validation"]["fold_assignments"]))
    target = data_config["target_column"]
    identifier = data_config["id_column"]
    features = train.drop(columns=[target, identifier])
    labels = train[target].astype(str)
    seed = int(config["project"]["seed"])
    folds = int(config["validation"]["folds"])
    fold_values = _validate_fold_assignments(assignments, train, identifier, seed, folds)
    trials = contract["trials"]
    global_encoder = LabelEncoder().fit(labels)

    completed: dict[int, dict[str, np.ndarray]] = {}
    new_folds = 0
    expected_keys = {
        "valid_index", "predictions", "train_scores", "valid_scores",
        "iterations", "convergence_warnings",
    }
    for fold in range(folds):
        checkpoint_path = checkpoint_dir / f"fold_{fold}.npz"
        receipt = manifest["completed_folds"].get(str(fold))
        if receipt is not None:
            completed[fold] = _load_checkpoint(checkpoint_path, receipt)
            continue
        train_index = np.flatnonzero(fold_values != fold)
        valid_index = np.flatnonzero(fold_values == fold)
        result = fold_executor(
            config, trials, features, labels, train_index, valid_index, global_encoder
        )
        if set(result) != expected_keys or result["predictions"].shape != (len(trials), len(valid_index)):
            raise RuntimeError(f"Fold executor result schema mismatch for fold {fold}.")
        _atomic_npz(checkpoint_path, **result)
        manifest["completed_folds"][str(fold)] = {
            "sha256": _file_hash(checkpoint_path), "row_count": int(len(valid_index)),
        }
        _atomic_json(manifest_path, manifest)
        completed[fold] = result
        new_folds += 1
        if max_new_folds is not None and new_folds >= max_new_folds:
            raise ControlledStop(f"Stopped after {new_folds} new fold checkpoint(s).")

    true_labels = global_encoder.transform(labels)
    oof = np.full((len(trials), len(train)), -1, dtype=np.int32)
    fold_train = np.empty((len(trials), folds), dtype=np.float64)
    fold_valid = np.empty((len(trials), folds), dtype=np.float64)
    fold_iterations = np.empty((len(trials), folds), dtype=np.int32)
    fold_warnings = np.empty((len(trials), folds), dtype=np.int32)
    for fold, result in completed.items():
        valid_index = result["valid_index"].astype(int)
        oof[:, valid_index] = result["predictions"]
        fold_train[:, fold] = result["train_scores"]
        fold_valid[:, fold] = result["valid_scores"]
        fold_iterations[:, fold] = result["iterations"]
        fold_warnings[:, fold] = result["convergence_warnings"]
    if np.any(oof < 0):
        raise RuntimeError("OOF predictions are incomplete.")

    rows: list[dict[str, Any]] = []
    for index, trial in enumerate(trials):
        rows.append({
            **trial,
            "oof_macro_f1": float(f1_score(true_labels, oof[index], average="macro")),
            "fold_mean_macro_f1": float(fold_valid[index].mean()),
            "fold_std_macro_f1": float(fold_valid[index].std(ddof=0)),
            "mean_train_macro_f1": float(fold_train[index].mean()),
            "mean_gap": float((fold_train[index] - fold_valid[index]).mean()),
            "fold_scores": fold_valid[index].tolist(),
            "fold_train_scores": fold_train[index].tolist(),
            "max_iterations": int(fold_iterations[index].max()),
            "convergence_warning_count": int(fold_warnings[index].sum()),
        })
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["oof_macro_f1"], row["fold_std_macro_f1"],
            abs(row["mean_gap"]), row["name"],
        ),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": config["project"]["experiment_name"],
        "objective": "train_only_oof_macro_f1",
        "selection_uses_test": False,
        "test_accessed": False,
        "gap_used_in_objective": False,
        "gap_usage": "tie_break_only_after_oof_and_fold_stability",
        "seed": seed,
        "folds": folds,
        "contract_hash": contract_hash,
        "best_trial": ranked[0],
        "trials": ranked,
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(result_dir / "metrics.json", payload)
    _atomic_npz(result_dir / "oof_predictions.npz", predictions=oof, true_labels=true_labels)

    lines = [
        "# TEST_007 Linear SVM 세트 — Train-only", "",
        "> Test와 submission은 열지 않았으며 모델 선택에 사용하지 않았다.", "",
        "| 순위 | 후보 | C | OOF Macro F1 | fold σ | 평균 gap | 수렴 경고 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | `{row['name']}` | {row['C']:.4g} | {row['oof_macro_f1']:.6f} | "
            f"{row['fold_std_macro_f1']:.6f} | {row['mean_gap']:.6f} | "
            f"{row['convergence_warning_count']} |"
        )
    lines.extend([
        "", "## 선택 규칙", "",
        "OOF Macro F1을 최대화한다. fold 표준편차와 gap은 점수가 동률일 때만 사용한다.", "",
        f"- 최종 후보: `{ranked[0]['name']}`",
        f"- 최종 OOF Macro F1: `{ranked[0]['oof_macro_f1']:.6f}`",
        f"- checkpoint contract: `{contract_hash}`",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_search(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
