from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.models.logistic_regression_elasticnet_model import create_model
from src.test_007 import run_test_007_elasticnet_set as elastic


def _fixture(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    train = pd.DataFrame({
        "ID": [f"T{i}" for i in range(8)],
        "SUBCLASS": ["A", "B"] * 4,
        "GENE": list(range(8)),
    })
    train.to_csv(raw / "train.csv", index=False)
    folds = pd.DataFrame({
        "ID": train["ID"],
        "row_index": list(range(8)),
        "seed": [42] * 8,
        "fold": [0, 0, 1, 1, 0, 0, 1, 1],
    })
    folds.to_csv(tmp_path / "folds.csv", index=False)
    config = {
        "project": {"experiment_name": "synthetic_enet", "seed": 42},
        "data": {
            "raw_dir": str(raw), "train_file": "train.csv",
            "target_column": "SUBCLASS", "id_column": "ID",
        },
        "validation": {"fold_assignments": str(tmp_path / "folds.csv"), "folds": 2},
        "preprocessing": {"name": "pipeComb_v3"},
        "model": {"penalty": "elasticnet", "solver": "saga", "max_iter": 10},
        "search": {"trials": [
            {"name": "a", "C": 0.1, "l1_ratio": 0.25},
            {"name": "b", "C": 1.0, "l1_ratio": 0.75},
        ]},
        "runtime": {
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "result_dir": str(tmp_path / "results"),
            "report_file": str(tmp_path / "report.md"),
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _executor(calls: list[int]):
    def execute(config, trials, features, labels, train_index, valid_index, global_encoder):
        calls.append(int(valid_index[0]))
        true = global_encoder.transform(labels.iloc[valid_index])
        predictions = np.tile(true, (len(trials), 1)).astype(np.int32)
        return {
            "valid_index": valid_index.astype(np.int32),
            "predictions": predictions,
            "train_scores": np.full(len(trials), 0.9),
            "valid_scores": np.full(len(trials), 0.8),
            "iterations": np.ones(len(trials), dtype=np.int32),
            "convergence_warnings": np.zeros(len(trials), dtype=np.int32),
        }
    return execute


def test_factory_builds_elasticnet_logistic_model():
    model = create_model({"C": 0.5, "l1_ratio": 0.75}, seed=42)
    params = model.get_params()
    assert params["penalty"] == "elasticnet"
    assert params["solver"] == "saga"
    assert params["l1_ratio"] == 0.75
    assert params["class_weight"] == "balanced"


@pytest.mark.parametrize("field,value", [("C", 0), ("l1_ratio", -0.1), ("l1_ratio", 1.1)])
def test_factory_rejects_invalid_parameters(field, value):
    config = {"C": 1.0, "l1_ratio": 0.5, field: value}
    with pytest.raises(ValueError):
        create_model(config, seed=42)


def test_fold_checkpoint_resume_skips_completed_fold(tmp_path):
    config = _fixture(tmp_path)
    first_calls: list[int] = []
    with pytest.raises(elastic.ControlledStop):
        elastic.run_search(config, fold_executor=_executor(first_calls), max_new_folds=1)
    assert len(first_calls) == 1

    resumed_calls: list[int] = []
    result = elastic.run_search(config, fold_executor=_executor(resumed_calls))
    assert len(resumed_calls) == 1
    assert result["test_accessed"] is False
    assert result["selection_uses_test"] is False
    assert result["best_trial"]["oof_macro_f1"] == 1.0


def test_stale_checkpoint_contract_is_rejected(tmp_path):
    config = _fixture(tmp_path)
    elastic.run_search(config, fold_executor=_executor([]))
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["search"]["trials"][0]["C"] = 9.0
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(RuntimeError, match="contract changed"):
        elastic.run_search(config, fold_executor=_executor([]))


def test_train_only_config_rejects_test_key(tmp_path):
    config = _fixture(tmp_path)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["data"]["test_file"] = "test.csv"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="Train-only"):
        elastic.run_search(config, fold_executor=_executor([]))


def test_manifest_receipt_tamper_is_rejected(tmp_path):
    config = _fixture(tmp_path)
    elastic.run_search(config, fold_executor=_executor([]))
    manifest_path = tmp_path / "checkpoints" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["completed_folds"]["0"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="receipt mismatch"):
        elastic.run_search(config, fold_executor=_executor([]))
