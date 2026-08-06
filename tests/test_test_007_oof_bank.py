from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import yaml

import src.test_007.run_test_007_oof_bank as bank


CLASSES = [f"C{i:02d}" for i in range(26)]
PROTECTED = [bank._repo_root() / path for path in bank.PROTECTED_FILES]


class FakeEncoder:
    classes_ = np.asarray(CLASSES)


class FakePipeline:
    def __init__(self):
        self.label_encoder = FakeEncoder()

    def fit_transform(self, features, labels):
        return features.reset_index(drop=True)

    def transform(self, features):
        return features.reset_index(drop=True)

    def encode_labels(self, labels):
        lookup = {name: index for index, name in enumerate(CLASSES)}
        return np.asarray([lookup[value] for value in labels], dtype=np.int32)


class FakeModel:
    # Deliberately reversed to exercise fixed-order probability alignment.
    classes_ = np.arange(25, -1, -1)

    def fit(self, features, labels):
        return self

    def predict_proba(self, features):
        result = np.full((len(features), 26), 0.1 / 25, dtype=np.float64)
        encoded = features["signal"].to_numpy(dtype=int)
        for row, target in enumerate(encoded):
            source_column = 25 - target
            result[row, source_column] = 0.9
        return result

    def predict(self, features):
        return features["signal"].to_numpy(dtype=np.int32)


class FakeLazyModel(FakeModel):
    lazy_fit_count = 0
    test_loaded = False

    def __init__(self):
        self.ready = False

    def fit(self, features, labels):
        self.ready = True
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state["ready"] = False
        return state

    def _ensure_ready(self):
        if not self.ready:
            assert not type(self).test_loaded, "lazy fit occurred after Test load"
            type(self).lazy_fit_count += 1
            self.ready = True

    def predict_proba(self, features):
        self._ensure_ready()
        return super().predict_proba(features)

    def predict(self, features):
        self._ensure_ready()
        return super().predict(features)


class FakeLazyExpert:
    lazy_fit_count = 0
    test_loaded = False

    def _ensure_classifier(self):
        if not hasattr(self, "classifier_"):
            assert not type(self).test_loaded, "expert lazy fit occurred after Test load"
            type(self).lazy_fit_count += 1
            self.classifier_ = object()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("classifier_", None)
        return state


class FakeCollisionLazyModel(FakeModel):
    def fit(self, features, labels):
        self.experts_ = {("C06", "C11"): FakeLazyExpert()}
        return self

    # Deliberately never routes the warm-up rows to the expert. The runner must
    # initialize experts directly instead of relying only on top-1 routing.
    def predict_proba(self, features):
        return super().predict_proba(features)


@pytest.fixture()
def fixture_run(tmp_path, monkeypatch):
    raw = bank._repo_root() / "data" / "raw"
    config_path = tmp_path / "config.yaml"
    config = {
        "project": {"experiment_name": "fake_oof", "seed": 42},
        "data": {
            "raw_dir": str(raw),
            "train_file": "train.csv",
            "test_file": "test.csv",
            "submission_file": "sample_submission.csv",
            "target_column": "SUBCLASS",
            "id_column": "ID",
        },
        "model": {"name": "fake", "class_names": CLASSES},
        "preprocessing": {"name": "pipeComb_v3"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    rows = 26 * 5
    frame = pd.DataFrame(
        {
            "ID": [f"R{i}" for i in range(rows)],
            "row_id": np.arange(rows),
            "signal": np.tile(np.arange(26), 5),
            "SUBCLASS": np.tile(CLASSES, 5),
        }
    )
    real_read_csv = pd.read_csv
    reads = []

    def guarded_read_csv(path, *args, **kwargs):
        reads.append(Path(path).name)
        if Path(path).name == "test.csv":
            raise AssertionError("OOF phase accessed test.csv")
        if Path(path).name == "train.csv":
            return frame.copy()
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(bank.pd, "read_csv", guarded_read_csv)
    monkeypatch.setattr(bank, "create_preprocessing_pipeline", lambda config: FakePipeline())
    monkeypatch.setattr(bank, "build_model", lambda config: FakeModel())
    monkeypatch.setattr(bank, "fit_model", lambda model, x, y, vx=None, vy=None: model.fit(x, y))
    monkeypatch.setattr(bank, "_source_hashes", lambda config: {"src/fake.py": "abc"})
    return config_path, tmp_path / "artifacts", reads, frame


def test_oof_is_train_only_complete_aligned_and_reusable(fixture_run):
    config_path, artifact_root, reads, frame = fixture_run
    output = bank.run_oof(config_path, 42, None, artifact_root)

    assert "test.csv" not in reads
    assert reads.count("train.csv") == 1
    probability = np.load(output / "oof_probability_by_seed.npy")
    prediction = np.load(output / "oof_prediction_by_seed.npy")
    folds = pd.read_csv(output / "fold_assignments.csv")
    assert probability.shape == (1, len(frame), 26)
    assert prediction.shape == (1, len(frame))
    np.testing.assert_allclose(probability.sum(axis=2), 1.0, atol=1e-5)
    np.testing.assert_array_equal(prediction[0], frame["signal"].to_numpy())
    assert folds["row_index"].nunique() == len(frame)
    assert set(folds["fold"]) == set(range(5))
    assert (folds.groupby("row_index").size() == 1).all()
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["test_accessed"] is False
    assert manifest["config_hash"] and manifest["source_hash"] and manifest["train_data_hash"]
    audit = json.loads((output / "io_audit.json").read_text(encoding="utf-8"))
    assert audit["test_accessed"] is False
    assert audit["inputs"]["test"]["accessed"] is False

    before_mtime = (output / "run_manifest.json").stat().st_mtime_ns
    reused = bank.run_oof(config_path, 42, None, artifact_root)
    assert reused == output
    assert (output / "run_manifest.json").stat().st_mtime_ns == before_mtime


def test_different_existing_manifest_requires_overwrite(fixture_run):
    config_path, artifact_root, _, _ = fixture_run
    output = bank.run_oof(config_path, 42, None, artifact_root)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_identity_hash"] = "different"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileExistsError):
        bank.run_oof(config_path, 42, None, artifact_root)
    assert bank.run_oof(config_path, 42, None, artifact_root, overwrite=True) == output


def test_partial_output_without_manifest_requires_explicit_overwrite(fixture_run):
    config_path, artifact_root, _, _ = fixture_run
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_hash = bank._sha256_file(config_path.resolve())
    partial = bank._run_paths(config, config_hash, 42, artifact_root.resolve()).run_dir
    partial.mkdir(parents=True)
    (partial / "partial.tmp").write_text("interrupted", encoding="utf-8")

    with pytest.raises(FileExistsError, match="different manifest"):
        bank.run_oof(config_path, 42, None, artifact_root)
    assert (partial / "partial.tmp").exists()

    output = bank.run_oof(config_path, 42, None, artifact_root, overwrite=True)
    assert output == partial
    assert not (output / "partial.tmp").exists()
    assert (output / "run_manifest.json").is_file()


def test_predict_labels_accepts_csr_matrix():
    features = sparse.csr_matrix(np.eye(4, dtype=np.float64))

    class SparseModel:
        def predict(self, matrix):
            assert sparse.isspmatrix_csr(matrix)
            return np.arange(matrix.shape[0], dtype=np.int32)

    np.testing.assert_array_equal(
        bank._predict_labels(SparseModel(), features), np.arange(4, dtype=np.int32)
    )


def test_infer_guard_rejects_before_test_access(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project: {experiment_name: x}\n", encoding="utf-8")
    calls = []

    def forbidden_read(path, *args, **kwargs):
        calls.append(path)
        raise AssertionError("data must not be read before frozen manifest validation")

    monkeypatch.setattr(bank.pd, "read_csv", forbidden_read)
    with pytest.raises(RuntimeError, match="frozen_manifest.json is missing"):
        bank.run_infer(config_path, 42, tmp_path / "artifacts")
    assert calls == []


@pytest.mark.parametrize(
    "manifest",
    [
        {},
        {
            "schema_version": 1,
            "frozen": False,
            "selection_uses_test": False,
            "thresholds_frozen": True,
            "candidates": [],
        },
        {
            "schema_version": 1,
            "frozen": True,
            "selection_uses_test": True,
            "thresholds_frozen": True,
            "candidates": [{"config_hash": "x", "seeds": [42]}],
        },
    ],
)
def test_infer_guard_rejects_invalid_freeze_before_data_access(
    tmp_path, monkeypatch, manifest
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project: {experiment_name: x}\n", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "frozen_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    calls = []

    def forbidden_read(path, *args, **kwargs):
        calls.append(path)
        raise AssertionError("data must not be read before frozen manifest validation")

    monkeypatch.setattr(bank.pd, "read_csv", forbidden_read)
    with pytest.raises(RuntimeError, match="infer is blocked"):
        bank.run_infer(config_path, 42, artifact_root)
    assert calls == []


def test_custom_fold_assignments_are_preserved(fixture_run, tmp_path):
    config_path, artifact_root, _, frame = fixture_run
    assignment_path = tmp_path / "folds.csv"
    expected = np.tile(np.arange(5), 26)
    pd.DataFrame({"row_index": np.arange(len(frame)), "seed": 42, "fold": expected}).to_csv(
        assignment_path, index=False
    )
    output = bank.run_oof(config_path, 42, assignment_path, artifact_root)
    actual = pd.read_csv(output / "fold_assignments.csv")["fold"].to_numpy()
    np.testing.assert_array_equal(actual, expected)


def test_class_order_is_derived_from_train_when_config_omits_it(fixture_run):
    config_path, artifact_root, _, frame = fixture_run
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del config["model"]["class_names"]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = bank.run_oof(config_path, 42, None, artifact_root)
    assert json.loads((output / "class_names.json").read_text(encoding="utf-8")) == CLASSES
    assert np.load(output / "oof_probability_by_seed.npy").shape == (1, len(frame), 26)


def test_outer_validation_is_never_used_for_early_stopping(fixture_run, monkeypatch):
    config_path, artifact_root, _, _ = fixture_run
    fit_calls = []

    class EarlyModel(FakeModel):
        early_stopping_rounds = 3
        best_iteration = 1

    monkeypatch.setattr(bank, "build_model", lambda config: EarlyModel())
    monkeypatch.setattr(bank.pickle, "dump", lambda value, file, protocol=None: file.write(b"fake"))

    def record_fit(model, train_x, train_y, valid_x=None, valid_y=None):
        fit_calls.append(
            (
                set(train_x["row_id"].tolist()),
                None if valid_x is None else set(valid_x["row_id"].tolist()),
            )
        )
        model.fit(train_x, train_y)

    monkeypatch.setattr(bank, "fit_model", record_fit)
    bank.run_oof(config_path, 42, None, artifact_root)
    inner_calls = [(train_ids, valid_ids) for train_ids, valid_ids in fit_calls if valid_ids]
    refit_calls = [(train_ids, valid_ids) for train_ids, valid_ids in fit_calls if valid_ids is None]
    assert len(inner_calls) == 5 and len(refit_calls) == 5
    for train_ids, valid_ids in inner_calls:
        assert train_ids.isdisjoint(valid_ids)
        assert len(train_ids | valid_ids) == 104  # exactly the outer-train rows
    assert all(len(train_ids) == 104 for train_ids, _ in refit_calls)


def test_decision_calibration_fits_preprocessing_inside_raw_outer_train(
    fixture_run, monkeypatch
):
    config_path, artifact_root, _, _ = fixture_run
    preprocessing_fit_rows = []

    class RecordingPipeline(FakePipeline):
        def fit_transform(self, features, labels):
            preprocessing_fit_rows.append(set(features["row_id"].tolist()))
            return super().fit_transform(features, labels)

    class DecisionModel:
        classes_ = np.arange(26)

        def fit(self, features, labels):
            return self

        def decision_function(self, features):
            scores = np.full((len(features), 26), -2.0)
            scores[np.arange(len(features)), features["signal"].to_numpy(dtype=int)] = 2.0
            return scores

        def predict(self, features):
            return features["signal"].to_numpy(dtype=np.int32)

    monkeypatch.setattr(bank, "create_preprocessing_pipeline", lambda config: RecordingPipeline())
    monkeypatch.setattr(bank, "build_model", lambda config: DecisionModel())
    monkeypatch.setattr(bank, "fit_model", lambda model, x, y, vx=None, vy=None: model.fit(x, y))
    monkeypatch.setattr(bank.pickle, "dump", lambda value, file, protocol=None: file.write(b"fake"))
    output = bank.run_oof(config_path, 42, None, artifact_root)
    assert len(preprocessing_fit_rows) == 10
    for calibration_fit, outer_full_fit in zip(
        preprocessing_fit_rows[0::2], preprocessing_fit_rows[1::2]
    ):
        assert calibration_fit < outer_full_fit
        assert len(calibration_fit) == 78
        assert len(outer_full_fit) == 104
    passport = json.loads((output / "model_passport.json").read_text(encoding="utf-8"))
    assert passport["probability_mode"] == "calibrated_decision"
    np.testing.assert_allclose(
        np.load(output / "oof_probability_by_seed.npy").sum(axis=2), 1.0, atol=1e-5
    )


def test_protected_files_remain_byte_identical(fixture_run):
    config_path, artifact_root, _, _ = fixture_run
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in PROTECTED}
    bank.run_oof(config_path, 42, None, artifact_root)
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in PROTECTED}
    assert after == before


def test_oof_rejects_configured_test_as_fold_input_before_hash_or_read(
    fixture_run, monkeypatch
):
    config_path, artifact_root, reads, _ = fixture_run
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    test_path = (bank._repo_root() / config["data"]["raw_dir"] / config["data"]["test_file"]).resolve()
    real_hash = bank._sha256_file
    hashed = []

    def recording_hash(path):
        hashed.append(Path(path).resolve())
        return real_hash(Path(path))

    monkeypatch.setattr(bank, "_sha256_file", recording_hash)
    with pytest.raises(ValueError, match="configured test/submission"):
        bank.run_oof(config_path, 42, test_path, artifact_root)
    assert test_path not in hashed
    assert "test.csv" not in reads


def test_oof_rejects_hardlink_alias_of_configured_test_before_hash(
    fixture_run, tmp_path, monkeypatch
):
    config_path, artifact_root, reads, _ = fixture_run
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw = tmp_path / "hardlink_raw"
    raw.mkdir()
    configured_test = raw / "test.csv"
    configured_test.write_bytes(b"configured test")
    alias = tmp_path / "fold_alias.csv"
    try:
        os.link(configured_test, alias)
    except OSError as error:
        pytest.skip(f"hardlinks unavailable on this filesystem: {error}")
    (raw / "train.csv").write_bytes(b"train")
    (raw / "sample_submission.csv").write_bytes(b"submission")
    config["data"]["raw_dir"] = str(raw)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    real_hash = bank._sha256_file
    hashed = []

    def recording_hash(path):
        hashed.append(Path(path).resolve())
        return real_hash(Path(path))

    monkeypatch.setattr(bank, "_sha256_file", recording_hash)
    with pytest.raises(ValueError, match="configured test/submission"):
        bank.run_oof(config_path, 42, alias, artifact_root)
    assert alias.resolve() not in hashed
    assert configured_test.resolve() not in hashed
    assert "test.csv" not in reads


def test_experiment_name_cannot_escape_artifact_root(fixture_run, tmp_path):
    config_path, artifact_root, _, _ = fixture_run
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["project"]["experiment_name"] = "../../victim"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    victim = tmp_path.parent / "victim"
    with pytest.raises(ValueError, match="unsafe experiment_name"):
        bank.run_oof(config_path, 42, None, artifact_root, overwrite=True)
    assert not victim.exists()


@pytest.mark.parametrize("corruption", ["missing", "shape", "hash"])
def test_incomplete_or_corrupt_artifacts_are_never_reused(fixture_run, corruption):
    config_path, artifact_root, _, _ = fixture_run
    output = bank.run_oof(config_path, 42, None, artifact_root)
    probability_path = output / "oof_probability_by_seed.npy"
    if corruption == "missing":
        probability_path.unlink()
    elif corruption == "shape":
        np.save(probability_path, np.zeros((1, 1, 26), dtype=np.float64))
    else:
        payload = bytearray(probability_path.read_bytes())
        payload[-1] ^= 1
        probability_path.write_bytes(payload)
    with pytest.raises(FileExistsError, match="different manifest"):
        bank.run_oof(config_path, 42, None, artifact_root)


@pytest.mark.parametrize(
    "fold_values,error",
    [
        ([0.9] + [0] * 129, "integer-valued"),
        ([float("nan")] + [0] * 129, "finite"),
        ([5] + [0] * 129, "range"),
    ],
)
def test_fold_values_are_validated_before_integer_cast(
    fixture_run, tmp_path, fold_values, error
):
    _, _, _, frame = fixture_run
    path = tmp_path / "invalid_folds.csv"
    pd.DataFrame(
        {"row_index": np.arange(len(frame)), "seed": 42, "fold": fold_values}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match=error):
        bank._load_assignments(path, frame, frame["SUBCLASS"], "ID", 42)


def test_fold_assignment_duplicate_and_wrong_seed_are_rejected(fixture_run, tmp_path):
    _, _, _, frame = fixture_run
    duplicate = tmp_path / "duplicate.csv"
    pd.DataFrame(
        {
            "row_index": [0, 0, *range(2, len(frame))],
            "seed": 42,
            "fold": np.tile(np.arange(5), 26),
        }
    ).to_csv(duplicate, index=False)
    with pytest.raises(ValueError, match="duplicate row_index"):
        bank._load_assignments(duplicate, frame, frame["SUBCLASS"], "ID", 42)

    wrong_seed = tmp_path / "wrong_seed.csv"
    pd.DataFrame(
        {
            "row_index": np.arange(len(frame)),
            "seed": 777,
            "fold": np.tile(np.arange(5), 26),
        }
    ).to_csv(wrong_seed, index=False)
    with pytest.raises(ValueError, match="no rows for seed 42"):
        bank._load_assignments(wrong_seed, frame, frame["SUBCLASS"], "ID", 42)


def test_protected_hash_change_aborts_without_manifest(fixture_run, monkeypatch):
    config_path, artifact_root, _, _ = fixture_run
    protected_target = PROTECTED[0].resolve()
    real_hash = bank._sha256_file
    protected_calls = 0

    def changing_hash(path):
        nonlocal protected_calls
        resolved = Path(path).resolve()
        if resolved == protected_target:
            protected_calls += 1
            if protected_calls >= 2:
                return "0" * 64
        return real_hash(Path(path))

    monkeypatch.setattr(bank, "_sha256_file", changing_hash)
    with pytest.raises(RuntimeError, match="protected source files changed"):
        bank.run_oof(config_path, 42, None, artifact_root)
    assert list(artifact_root.rglob("run_manifest.json")) == []


def _write_frozen_manifest(config_path: Path, artifact_root: Path, output: Path) -> dict:
    run_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    frozen_fields = {
        key: run_manifest[key]
        for key in (
            "train_data_hash",
            "source_hash",
            "fold_assignments_hash",
            "class_names_hash",
            "artifact_bundle_hash",
            "bundle_safe_for_infer",
            "library_versions",
            "external_runtime_snapshot",
        )
    }
    frozen_fields["run_manifest_hash"] = bank._sha256_file(output / "run_manifest.json")
    frozen = {
        "schema_version": 1,
        "frozen": True,
        "selection_uses_test": False,
        "thresholds_frozen": True,
        "class_names": CLASSES,
        "candidates": [
            {
                "config_hash": bank._sha256_file(config_path),
                "seeds": [42],
                "runs": {"42": frozen_fields},
            }
        ],
    }
    (artifact_root / "frozen_manifest.json").write_text(
        json.dumps(frozen), encoding="utf-8"
    )
    return frozen


def test_infer_uses_frozen_bundles_and_reads_test_once_after_all_validation(
    fixture_run, monkeypatch
):
    config_path, artifact_root, _, frame = fixture_run
    output = bank.run_oof(config_path, 42, None, artifact_root)
    _write_frozen_manifest(config_path, artifact_root, output)
    real_read_csv = pd.read_csv
    reads = []
    test = frame.drop(columns=["SUBCLASS"]).iloc[:7].copy()

    def infer_read_csv(path, *args, **kwargs):
        name = Path(path).name
        reads.append(name)
        if name == "train.csv":
            return frame.copy()
        if name == "test.csv":
            return test.copy()
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(bank.pd, "read_csv", infer_read_csv)
    monkeypatch.setattr(
        bank,
        "_fit_fold",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("infer must not fit")),
    )
    infer_output = bank.run_infer(config_path, 42, artifact_root)
    assert reads.count("test.csv") == 1
    assert (infer_output / "test_probability_by_seed_fold.npy").is_file()
    assert np.load(infer_output / "test_probability_by_seed_fold.npy").shape == (1, 5, 7, 26)


def test_infer_warms_lazy_models_before_first_test_read(fixture_run, monkeypatch):
    config_path, artifact_root, _, frame = fixture_run
    FakeLazyModel.lazy_fit_count = 0
    FakeLazyModel.test_loaded = False
    monkeypatch.setattr(bank, "build_model", lambda config: FakeLazyModel())
    output = bank.run_oof(config_path, 42, None, artifact_root)
    assert FakeLazyModel.lazy_fit_count == 0
    _write_frozen_manifest(config_path, artifact_root, output)
    real_read_csv = pd.read_csv
    test = frame.drop(columns=["SUBCLASS"]).iloc[:5].copy()

    def infer_read_csv(path, *args, **kwargs):
        name = Path(path).name
        if name == "train.csv":
            return frame.copy()
        if name == "test.csv":
            FakeLazyModel.test_loaded = True
            return test.copy()
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(bank.pd, "read_csv", infer_read_csv)
    monkeypatch.setattr(
        bank,
        "_fit_fold",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("infer must not fit")),
    )
    bank.run_infer(config_path, 42, artifact_root)
    assert FakeLazyModel.test_loaded is True
    assert FakeLazyModel.lazy_fit_count == 5


def test_infer_eagerly_initializes_realtabr_like_unrouted_experts(
    fixture_run, monkeypatch
):
    config_path, artifact_root, _, frame = fixture_run
    FakeLazyExpert.lazy_fit_count = 0
    FakeLazyExpert.test_loaded = False
    monkeypatch.setattr(bank, "build_model", lambda config: FakeCollisionLazyModel())
    output = bank.run_oof(config_path, 42, None, artifact_root)
    assert FakeLazyExpert.lazy_fit_count == 0
    _write_frozen_manifest(config_path, artifact_root, output)
    real_read_csv = pd.read_csv
    test = frame.drop(columns=["SUBCLASS"]).iloc[:5].copy()

    def infer_read_csv(path, *args, **kwargs):
        name = Path(path).name
        if name == "train.csv":
            return frame.copy()
        if name == "test.csv":
            FakeLazyExpert.test_loaded = True
            return test.copy()
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(bank.pd, "read_csv", infer_read_csv)
    bank.run_infer(config_path, 42, artifact_root)
    assert FakeLazyExpert.test_loaded is True
    assert FakeLazyExpert.lazy_fit_count == 5


def test_infer_rejects_tampered_frozen_identity_before_test_read(
    fixture_run, monkeypatch
):
    config_path, artifact_root, _, _ = fixture_run
    output = bank.run_oof(config_path, 42, None, artifact_root)
    frozen = _write_frozen_manifest(config_path, artifact_root, output)
    frozen["candidates"][0]["runs"]["42"]["artifact_bundle_hash"] = "tampered"
    (artifact_root / "frozen_manifest.json").write_text(json.dumps(frozen), encoding="utf-8")
    test_reads = []
    real_read_csv = pd.read_csv

    def guarded(path, *args, **kwargs):
        if Path(path).name == "test.csv":
            test_reads.append(path)
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(bank.pd, "read_csv", guarded)
    with pytest.raises(RuntimeError, match="artifact_bundle_hash"):
        bank.run_infer(config_path, 42, artifact_root)
    assert test_reads == []


def test_frozen_class_order_requires_26_unique_names(fixture_run):
    config_path, artifact_root, _, _ = fixture_run
    artifact_root.mkdir()
    frozen = {
        "schema_version": 1,
        "frozen": True,
        "selection_uses_test": False,
        "thresholds_frozen": True,
        "class_names": ["same"] * 26,
        "candidates": [
            {
                "config_hash": bank._sha256_file(config_path),
                "seeds": [42],
                "runs": {"42": {}},
            }
        ],
    }
    (artifact_root / "frozen_manifest.json").write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(RuntimeError, match="class order is invalid"):
        bank._load_frozen_manifest(artifact_root, config_path, 42)


def test_library_snapshot_records_only_installed_runtime_distributions(monkeypatch):
    installed = {"numpy": "2.4.2", "torch": "2.13.0"}

    def fake_version(name):
        if name in installed:
            return installed[name]
        raise bank.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(bank.metadata, "version", fake_version)
    versions = bank._library_versions({"model": {"name": "not-a-distribution"}})
    assert versions["numpy"] == "2.4.2"
    assert versions["torch"] == "2.13.0"
    assert "catboost" not in versions
    assert all(value is not None for value in versions.values())


def test_library_snapshot_includes_faiss_cpu_when_installed(monkeypatch):
    installed = {
        "numpy": "2.4.2",
        "pytabkit": "1.7.3",
        "faiss-cpu": "1.14.3",
    }

    def fake_version(name):
        if name in installed:
            return installed[name]
        raise bank.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(bank.metadata, "version", fake_version)
    versions = bank._library_versions(
        {"model": {"name": "realtabr_collision_expert"}}
    )
    assert versions["pytabkit"] == "1.7.3"
    assert versions["faiss-cpu"] == "1.14.3"


@pytest.mark.parametrize("missing", ["pytabkit", "faiss-cpu"])
def test_realtabr_missing_required_runtime_fails(missing, monkeypatch):
    installed = {"pytabkit": "1.7.3", "faiss-cpu": "1.14.3"}
    del installed[missing]

    def fake_version(name):
        if name in installed:
            return installed[name]
        raise bank.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(bank.metadata, "version", fake_version)
    with pytest.raises(RuntimeError, match=missing):
        bank._library_versions({"model": {"name": "realtabr_collision_expert"}})

