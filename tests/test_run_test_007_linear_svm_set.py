from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.models.linear_svc_model import create_model
from src.test_007 import run_test_007_linear_svm_set as svm


def _fixture(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    train = pd.DataFrame({
        "ID": [f"T{i}" for i in range(8)], "SUBCLASS": ["A", "B"] * 4,
        "GENE": list(range(8)),
    })
    train.to_csv(raw / "train.csv", index=False)
    pd.DataFrame({
        "ID": train["ID"], "row_index": list(range(8)), "seed": [42] * 8,
        "fold": [0, 0, 1, 1, 0, 0, 1, 1],
    }).to_csv(tmp_path / "folds.csv", index=False)
    config = {
        "project": {"experiment_name": "synthetic_svm", "seed": 42},
        "data": {"raw_dir": str(raw), "train_file": "train.csv", "target_column": "SUBCLASS", "id_column": "ID"},
        "validation": {"fold_assignments": str(tmp_path / "folds.csv"), "folds": 2},
        "preprocessing": {"name": "pipeComb_v3"},
        "model": {"class_weight": "balanced", "max_iter": 10, "dual": "auto"},
        "search": {"trials": [{"name": "a", "C": 0.1}, {"name": "b", "C": 0.2}]},
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
        return {
            "valid_index": valid_index.astype(np.int32),
            "predictions": np.tile(true, (len(trials), 1)).astype(np.int32),
            "train_scores": np.full(len(trials), 0.9),
            "valid_scores": np.full(len(trials), 0.8),
            "iterations": np.ones(len(trials), dtype=np.int32),
            "convergence_warnings": np.zeros(len(trials), dtype=np.int32),
        }
    return execute


def test_existing_factory_builds_linear_svc():
    model = create_model({"C": 0.2, "class_weight": "balanced"}, seed=42)
    assert model.get_params()["C"] == 0.2
    assert model.get_params()["class_weight"] == "balanced"


def test_checkpoint_resume_skips_completed_fold(tmp_path):
    config = _fixture(tmp_path)
    first: list[int] = []
    with pytest.raises(svm.ControlledStop):
        svm.run_search(config, fold_executor=_executor(first), max_new_folds=1)
    assert len(first) == 1
    resumed: list[int] = []
    result = svm.run_search(config, fold_executor=_executor(resumed))
    assert len(resumed) == 1
    assert result["test_accessed"] is False
    assert result["best_trial"]["oof_macro_f1"] == 1.0


def test_stale_contract_is_rejected(tmp_path):
    config = _fixture(tmp_path)
    svm.run_search(config, fold_executor=_executor([]))
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["search"]["trials"][0]["C"] = 3.0
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(RuntimeError, match="contract changed"):
        svm.run_search(config, fold_executor=_executor([]))


def test_test_key_is_forbidden(tmp_path):
    config = _fixture(tmp_path)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["data"]["test_file"] = "test.csv"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="Train-only"):
        svm.run_search(config, fold_executor=_executor([]))


def test_receipt_tamper_is_rejected(tmp_path):
    config = _fixture(tmp_path)
    svm.run_search(config, fold_executor=_executor([]))
    manifest_path = tmp_path / "checkpoints" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["completed_folds"]["0"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="receipt mismatch"):
        svm.run_search(config, fold_executor=_executor([]))
