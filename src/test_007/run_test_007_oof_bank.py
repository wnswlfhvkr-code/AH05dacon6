"""Train-only TEST_007 OOF probability bank and frozen inference runner."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import pickle
import platform
import re
import shutil
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.pipelines.preprocessing_registry import create_preprocessing_pipeline
from src.train import build_model, fit_model, load_config, used_tree_count


SCHEMA_VERSION = 1
N_SPLITS = 5
PROTECTED_FILES = (
    "src/train.py",
    "src/pipelines/jyp_preprocessing/pipeline_pipe_comb_v3.py",
    "src/pipelines/preprocessing_registry.py",
)
SAFE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
RUNTIME_DISTRIBUTIONS = {
    "numpy": ("numpy",),
    "pandas": ("pandas",),
    "scipy": ("scipy",),
    "scikit-learn": ("scikit-learn",),
    "torch": ("torch",),
    "xgboost": ("xgboost",),
    "lightgbm": ("lightgbm",),
    "catboost": ("catboost",),
    "tabm": ("tabm",),
    "tabpfn": ("tabpfn",),
    "tabicl": ("tabicl",),
    "tabfm": ("tabfm",),
    "pytabkit": ("pytabkit",),
    "faiss-cpu": ("faiss-cpu",),
    "xrfm": ("xrfm",),
    "TALENT": ("TALENT", "talent"),
    "autogluon.tabular": ("autogluon.tabular",),
}


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    manifest: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_slug(value: Any, field: str) -> str:
    slug = str(value)
    if slug in {".", ".."} or SAFE_SLUG.fullmatch(slug) is None:
        raise ValueError(f"unsafe {field}: {slug!r}")
    return slug


def _contained_path(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"artifact path escapes root: {resolved_candidate}") from error
    if resolved_candidate == resolved_root:
        raise ValueError("artifact run path cannot equal artifact root")
    return resolved_candidate


def _safe_remove_tree(root: Path, candidate: Path) -> None:
    shutil.rmtree(_contained_path(root, candidate))


def _source_hashes(config: dict) -> dict[str, str]:
    root = _repo_root()
    candidates = [Path(__file__).resolve(), *(root / item for item in PROTECTED_FILES)]
    model_name = str(config["model"]["name"])
    model_path = root / "src" / "models" / f"{model_name}_model.py"
    if model_path.exists():
        candidates.append(model_path)
    return {
        str(path.relative_to(root)).replace("\\", "/"): _sha256_file(path)
        for path in candidates
    }


def _validate_config(config: dict) -> tuple[list[str] | None, dict]:
    classes = [str(value) for value in config.get("model", {}).get("class_names", [])]
    if classes and (len(classes) != 26 or len(set(classes)) != 26):
        raise ValueError("model.class_names must contain exactly 26 unique classes")
    preprocessing = deepcopy(config.get("preprocessing", {}))
    if preprocessing.get("name") != "pipeComb_v3":
        raise ValueError("TEST_007 OOF bank requires preprocessing.name=pipeComb_v3")
    return classes or None, preprocessing


def _run_paths(config: dict, config_hash: str, seed: int, artifact_root: Path) -> RunPaths:
    experiment = _safe_slug(config["project"]["experiment_name"], "experiment_name")
    seed_slug = _safe_slug(seed, "seed")
    run_dir = _contained_path(
        artifact_root, artifact_root.resolve() / experiment / config_hash / seed_slug
    )
    return RunPaths(run_dir=run_dir, manifest=run_dir / "run_manifest.json")


def _audit_oof_inputs(
    config_path: Path,
    train_path: Path,
    fold_path: Path | None,
    test_path: Path,
    submission_path: Path,
) -> dict:
    resolved = {
        "config": config_path.resolve(),
        "train": train_path.resolve(),
        "test": test_path.resolve(),
        "submission": submission_path.resolve(),
    }
    if fold_path is not None:
        resolved["fold_assignments"] = fold_path.resolve()
        if any(
            _same_file_identity(resolved["fold_assignments"], blocked)
            for blocked in (resolved["test"], resolved["submission"])
        ):
            raise ValueError("OOF fold assignments cannot reference configured test/submission data")
    if any(
        _same_file_identity(resolved["train"], blocked)
        for blocked in (resolved["test"], resolved["submission"])
    ):
        raise ValueError("OOF train path aliases configured test/submission data")
    return {
        "phase": "oof",
        "allowed_roles": ["config", "train", "fold_assignments", "checkpoint"],
        "inputs": {
            role: {
                "resolved_path": str(path),
                "allowed": role in {"config", "train", "fold_assignments"},
                "accessed": role in {"config", "train"} or (role == "fold_assignments" and fold_path is not None),
            }
            for role, path in resolved.items()
        },
        "test_accessed": False,
    }


def _same_file_identity(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def _library_versions(config: dict) -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for key, distributions in RUNTIME_DISTRIBUTIONS.items():
        for distribution in distributions:
            try:
                versions[key] = metadata.version(distribution)
                break
            except metadata.PackageNotFoundError:
                continue
    if config["model"]["name"] == "realtabr_collision_expert":
        missing = sorted({"pytabkit", "faiss-cpu"} - versions.keys())
        if missing:
            raise RuntimeError(
                f"RealTabR runtime snapshot is incomplete; missing: {', '.join(missing)}"
            )
    return versions


def _load_assignments(
    path: Path | None,
    train: pd.DataFrame,
    labels: pd.Series,
    id_column: str,
    seed: int,
) -> np.ndarray:
    if path is None:
        folds = np.full(len(train), -1, dtype=np.int16)
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (_, valid_index) in enumerate(splitter.split(train, labels)):
            folds[valid_index] = fold
        return folds

    assignments = pd.read_csv(path)
    if "fold" not in assignments:
        raise ValueError("fold assignment file must contain a 'fold' column")
    if "seed" in assignments:
        seed_values = pd.to_numeric(assignments["seed"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(seed_values).all() or not np.equal(seed_values, np.floor(seed_values)).all():
            raise ValueError("fold assignment seeds must be finite integers")
        assignments = assignments.loc[seed_values.astype(np.int64) == seed]
        if assignments.empty:
            raise ValueError(f"fold assignments contain no rows for seed {seed}")
    fold_values = pd.to_numeric(assignments["fold"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(fold_values).all():
        raise ValueError("fold values must be finite")
    if not np.equal(fold_values, np.floor(fold_values)).all():
        raise ValueError("fold values must be integer-valued")
    if ((fold_values < 0) | (fold_values >= N_SPLITS)).any():
        raise ValueError("fold values must be in the range 0 through 4")
    assignments = assignments.copy()
    assignments["fold"] = fold_values.astype(np.int16)
    if "row_index" in assignments:
        row_values = pd.to_numeric(assignments["row_index"], errors="coerce").to_numpy(dtype=float)
        if (
            not np.isfinite(row_values).all()
            or not np.equal(row_values, np.floor(row_values)).all()
        ):
            raise ValueError("row_index values must be finite integers")
        assignments["row_index"] = row_values.astype(np.int64)
        if assignments["row_index"].duplicated().any():
            raise ValueError("duplicate row_index in fold assignments")
        if len(assignments) != len(train):
            raise ValueError("fold assignments must contain exactly one row per training row")
        mapped = assignments.set_index("row_index")["fold"].reindex(np.arange(len(train)))
    elif id_column in assignments:
        if assignments[id_column].duplicated().any() or train[id_column].duplicated().any():
            raise ValueError("IDs must be unique for ID-based fold assignments")
        if len(assignments) != len(train):
            raise ValueError("fold assignments must contain exactly one row per training ID")
        mapped = assignments.set_index(id_column)["fold"].reindex(train[id_column])
    else:
        raise ValueError(f"fold assignments need row_index or {id_column}")
    if mapped.isna().any():
        raise ValueError("fold assignments do not cover every training row")
    folds = mapped.to_numpy(dtype=np.int16)
    if set(folds.tolist()) != set(range(N_SPLITS)):
        raise ValueError("fold assignments must use every fold from 0 through 4")
    return folds


def _local_class_names(preprocessor: Any, fixed_classes: list[str]) -> list[str]:
    encoder = getattr(preprocessor, "label_encoder", None)
    classes = getattr(encoder, "classes_", None)
    return [str(value) for value in classes] if classes is not None else fixed_classes


def _align_probabilities(
    probabilities: Any,
    model: Any,
    local_classes: list[str],
    fixed_classes: list[str],
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"predict_proba must return 2 dimensions, got {values.shape}")
    model_classes = getattr(model, "classes_", None)
    if model_classes is None:
        if values.shape[1] != len(local_classes):
            raise ValueError("probability columns cannot be mapped without model.classes_")
        names = local_classes
    else:
        names = []
        for value in np.asarray(model_classes).tolist():
            if isinstance(value, (int, np.integer)):
                index = int(value)
                if not 0 <= index < len(local_classes):
                    raise ValueError(f"encoded class index out of range: {index}")
                names.append(local_classes[index])
            else:
                names.append(str(value))
    if len(names) != values.shape[1] or len(set(names)) != len(names):
        raise ValueError("model probability class metadata is invalid")
    fixed_index = {name: index for index, name in enumerate(fixed_classes)}
    aligned = np.zeros((len(values), len(fixed_classes)), dtype=np.float64)
    for source, name in enumerate(names):
        if name not in fixed_index:
            raise ValueError(f"unknown probability class: {name}")
        aligned[:, fixed_index[name]] = values[:, source]
    _validate_probabilities(aligned)
    return aligned


def _validate_probabilities(values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError("probabilities contain NaN or infinity")
    if (values < -1e-8).any():
        raise ValueError("probabilities contain negative values")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("probability rows must sum to one")


def _predict_labels(model: Any, features: Any) -> np.ndarray:
    prediction = np.asarray(model.predict(features))
    row_count = int(features.shape[0])
    if prediction.ndim != 1 or prediction.shape[0] != row_count:
        raise ValueError("predict must return one value per row")
    return prediction.astype(np.int32)


def _prediction_to_fixed(
    prediction: np.ndarray,
    local_classes: list[str],
    fixed_classes: list[str],
) -> np.ndarray:
    fixed_index = {name: index for index, name in enumerate(fixed_classes)}
    converted: list[int] = []
    for value in np.asarray(prediction).tolist():
        if isinstance(value, (int, np.integer)):
            index = int(value)
            if not 0 <= index < len(local_classes):
                raise ValueError(f"predicted class index out of range: {index}")
            name = local_classes[index]
        else:
            name = str(value)
        if name not in fixed_index:
            raise ValueError(f"unknown predicted class: {name}")
        converted.append(fixed_index[name])
    return np.asarray(converted, dtype=np.int32)


def _stratified_inner_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    _, counts = np.unique(labels, return_counts=True)
    active_classes = len(counts)
    if active_classes < 2 or (counts < 2).any():
        raise ValueError("inner split requires at least two samples per outer-train class")
    validation_rows = max(active_classes, int(np.ceil(0.2 * len(labels))))
    if len(labels) - validation_rows < active_classes:
        validation_rows = len(labels) - active_classes
    return train_test_split(
        indices,
        test_size=validation_rows,
        random_state=seed,
        stratify=labels,
    )


def _fit_without_outer_validation(
    config: dict,
    train_x: Any,
    train_y: np.ndarray,
) -> Any:
    model = build_model(config)
    if getattr(model, "early_stopping_rounds", None) is None:
        fit_model(model, train_x, train_y)
        return model
    fit_index, validation_index = _stratified_inner_indices(
        train_y, int(config["project"]["seed"])
    )
    inner_train_x = train_x.iloc[fit_index] if hasattr(train_x, "iloc") else train_x[fit_index]
    inner_valid_x = (
        train_x.iloc[validation_index] if hasattr(train_x, "iloc") else train_x[validation_index]
    )
    fit_model(
        model,
        inner_train_x,
        train_y[fit_index],
        inner_valid_x,
        train_y[validation_index],
    )
    tree_count = used_tree_count(model)
    refit_config = deepcopy(config)
    refit_config["model"].pop("early_stopping_rounds", None)
    if tree_count is not None:
        refit_config["model"]["n_estimators"] = tree_count
    refit_model = build_model(refit_config)
    fit_model(refit_model, train_x, train_y)
    return refit_model


def _fit_fold(
    config: dict,
    preprocessing_config: dict,
    raw_train_x: pd.DataFrame,
    raw_train_labels: pd.Series,
) -> tuple[Any, Any, str, Any | None, Any, np.ndarray]:
    """Fit calibration and the final base using outer-train rows only."""
    probe = build_model(config)
    has_native = hasattr(probe, "predict_proba")
    has_decision = hasattr(probe, "decision_function")
    calibrator = None
    if not has_native and has_decision:
        fit_index, calibration_index = _stratified_inner_indices(
            raw_train_labels.to_numpy(), int(config["project"]["seed"])
        )
        calibration_preprocessor = create_preprocessing_pipeline(preprocessing_config)
        calibration_train_x = calibration_preprocessor.fit_transform(
            raw_train_x.iloc[fit_index], raw_train_labels.iloc[fit_index]
        )
        calibration_x = calibration_preprocessor.transform(raw_train_x.iloc[calibration_index])
        calibration_train_y = np.asarray(
            calibration_preprocessor.encode_labels(raw_train_labels.iloc[fit_index]),
            dtype=np.int32,
        )
        calibration_y = np.asarray(
            calibration_preprocessor.encode_labels(raw_train_labels.iloc[calibration_index]),
            dtype=np.int32,
        )
        calibration_model = _fit_without_outer_validation(
            config, calibration_train_x, calibration_train_y
        )
        calibration_scores = np.asarray(calibration_model.decision_function(calibration_x))
        if calibration_scores.ndim == 1:
            calibration_scores = np.column_stack((-calibration_scores, calibration_scores))
        calibrator = LogisticRegression(
            max_iter=1000, random_state=int(config["project"]["seed"])
        )
        calibrator.fit(calibration_scores, calibration_y)
        mode = "calibrated_decision"
    elif has_native:
        mode = "native_proba"
    else:
        mode = "hard_only"

    preprocessor = create_preprocessing_pipeline(preprocessing_config)
    encoded_train_x = preprocessor.fit_transform(raw_train_x, raw_train_labels)
    encoded_train_y = np.asarray(preprocessor.encode_labels(raw_train_labels), dtype=np.int32)
    model = _fit_without_outer_validation(config, encoded_train_x, encoded_train_y)
    return preprocessor, model, mode, calibrator, encoded_train_x, encoded_train_y


def _predict_probability(
    model: Any,
    mode: str,
    calibrator: Any | None,
    features: Any,
    local_classes: list[str],
    fixed_classes: list[str],
) -> np.ndarray | None:
    if mode == "native_proba":
        return _align_probabilities(
            model.predict_proba(features), model, local_classes, fixed_classes
        )
    if mode == "calibrated_decision":
        scores = np.asarray(model.decision_function(features))
        if scores.ndim == 1:
            scores = np.column_stack((-scores, scores))
        return _align_probabilities(
            calibrator.predict_proba(scores), calibrator, local_classes, fixed_classes
        )
    return None


def _peak_memory() -> dict[str, int | None]:
    rss = None
    try:
        import psutil

        rss = int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        pass
    cuda = None
    try:
        import torch

        if torch.cuda.is_available():
            cuda = int(torch.cuda.max_memory_allocated())
    except ImportError:
        pass
    return {"peak_rss_bytes": rss, "peak_cuda_bytes": cuda}


def _manifest_identity(
    config_path: Path,
    config: dict,
    train_path: Path,
    fold_path: Path | None,
    seed: int,
    library_versions: dict[str, str],
    external_runtime_snapshot: dict[str, Any],
) -> dict:
    source_hashes = _source_hashes(config)
    protected_before = {
        name: _sha256_file(_repo_root() / name) for name in PROTECTED_FILES
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "phase": "oof",
        "config_path": str(config_path.resolve()),
        "config_hash": _sha256_file(config_path),
        "source_hashes": source_hashes,
        "source_hash": _canonical_hash(source_hashes),
        "protected_hashes_before": protected_before,
        "train_data_hash": _sha256_file(train_path),
        "fold_assignments_input_hash": _sha256_file(fold_path) if fold_path else None,
        "library_versions": library_versions,
        "external_runtime_snapshot": external_runtime_snapshot,
        "seed": seed,
        "folds": N_SPLITS,
        "test_accessed": False,
    }
    identity["manifest_identity_hash"] = _canonical_hash(identity)
    return identity


def _artifact_integrity(run_dir: Path, names: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        path = run_dir / name
        entry: dict[str, Any] = {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
        if path.suffix == ".npy":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            entry.update({"shape": list(array.shape), "dtype": str(array.dtype)})
        result[name] = entry
    return result


def _validate_artifacts(run_dir: Path, manifest: dict) -> bool:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    required = {
        "oof_prediction_by_seed.npy",
        "fold_assignments.csv",
        "fold_metrics.csv",
        "class_names.json",
        "runtime_metrics.json",
        "model_passport.json",
        "io_audit.json",
        "fold_bundles.pkl",
    }
    if manifest.get("probability_mode") != "hard_only":
        required.add("oof_probability_by_seed.npy")
    if not required.issubset(artifacts):
        return False
    for name, expected in artifacts.items():
        path = run_dir / name
        if not path.is_file() or path.stat().st_size != expected.get("size_bytes"):
            return False
        if _sha256_file(path) != expected.get("sha256"):
            return False
        if path.suffix == ".npy":
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError):
                return False
            if list(array.shape) != expected.get("shape") or str(array.dtype) != expected.get("dtype"):
                return False
    train_rows = manifest.get("train_rows")
    prediction = artifacts.get("oof_prediction_by_seed.npy", {})
    if prediction.get("shape") != [1, train_rows]:
        return False
    if manifest.get("probability_mode") != "hard_only":
        probability = artifacts.get("oof_probability_by_seed.npy", {})
        if probability.get("shape") != [1, train_rows, 26]:
            return False
    return True


def _same_manifest(path: Path, identity: dict) -> bool:
    if not path.exists():
        return False
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        return (
            existing.get("manifest_identity_hash") == identity["manifest_identity_hash"]
            and _validate_artifacts(path.parent, existing)
        )
    except (OSError, ValueError, TypeError):
        return False


def run_oof(
    config_path: Path,
    seed: int,
    fold_assignments_path: Path | None,
    artifact_root: Path,
    overwrite: bool = False,
) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    configured_classes, preprocessing_config = _validate_config(config)
    config = deepcopy(config)
    config["project"]["seed"] = seed
    data = config["data"]
    raw_dir = (_repo_root() / data["raw_dir"]).resolve()
    train_path = raw_dir / data["train_file"]
    io_audit = _audit_oof_inputs(
        config_path,
        train_path,
        fold_assignments_path,
        raw_dir / data["test_file"],
        raw_dir / data["submission_file"],
    )
    library_versions = _library_versions(config)
    external_runtime_snapshot: dict[str, Any] = {}
    identity = _manifest_identity(
        config_path,
        config,
        train_path,
        fold_assignments_path,
        seed,
        library_versions,
        external_runtime_snapshot,
    )
    paths = _run_paths(config, identity["config_hash"], seed, artifact_root.resolve())
    if paths.run_dir.exists():
        if _same_manifest(paths.manifest, identity) and not overwrite:
            return paths.run_dir
        if not overwrite:
            raise FileExistsError(f"OOF output already exists with a different manifest: {paths.run_dir}")
        _safe_remove_tree(artifact_root.resolve(), paths.run_dir)

    started = time.perf_counter()
    train = pd.read_csv(train_path)
    target, identifier = data["target_column"], data["id_column"]
    labels = train[target].astype(str)
    fixed_classes = configured_classes or sorted(labels.unique().tolist())
    if len(fixed_classes) != 26:
        raise ValueError("training labels must contain exactly 26 classes")
    unknown = sorted(set(labels) - set(fixed_classes))
    if unknown:
        raise ValueError(f"training data contains classes absent from model.class_names: {unknown}")
    features = train.drop(columns=[target, identifier])
    folds = _load_assignments(fold_assignments_path, train, labels, identifier, seed)
    if (folds < 0).any() or np.bincount(folds, minlength=N_SPLITS).sum() != len(train):
        raise ValueError("every training row must receive exactly one OOF fold")

    probabilities = np.full((len(train), len(fixed_classes)), np.nan, dtype=np.float64)
    predictions = np.full(len(train), -1, dtype=np.int32)
    coverage = np.zeros(len(train), dtype=np.int8)
    fold_rows: list[dict] = []
    fold_bundles: list[dict[str, Any]] = []
    probability_mode: str | None = None
    fit_seconds = inference_seconds = 0.0

    for fold in range(N_SPLITS):
        train_index = np.flatnonzero(folds != fold)
        valid_index = np.flatnonzero(folds == fold)
        if not len(valid_index):
            raise ValueError(f"fold {fold} has no validation rows")
        fold_train_x, fold_valid_x = features.iloc[train_index], features.iloc[valid_index]
        fold_train_labels, fold_valid_labels = labels.iloc[train_index], labels.iloc[valid_index]
        fit_started = time.perf_counter()
        preprocessor, model, fold_mode, calibrator, encoded_train_x, encoded_train_y = _fit_fold(
            config, preprocessing_config, fold_train_x, fold_train_labels
        )
        fit_seconds += time.perf_counter() - fit_started
        encoded_valid_x = preprocessor.transform(fold_valid_x)
        encoded_valid_y = np.asarray(preprocessor.encode_labels(fold_valid_labels), dtype=np.int32)
        if probability_mode is None:
            probability_mode = fold_mode
        elif probability_mode != fold_mode:
            raise RuntimeError("model probability interface changed between folds")

        infer_started = time.perf_counter()
        local_classes = _local_class_names(preprocessor, fixed_classes)
        fold_probability = _predict_probability(
            model, fold_mode, calibrator, encoded_valid_x, local_classes, fixed_classes
        )
        if fold_probability is None:
            fold_prediction = _prediction_to_fixed(
                _predict_labels(model, encoded_valid_x), local_classes, fixed_classes
            )
        else:
            fold_prediction = fold_probability.argmax(axis=1).astype(np.int32)
            probabilities[valid_index] = fold_probability
        train_prediction = _predict_labels(model, encoded_train_x)
        inference_seconds += time.perf_counter() - infer_started

        predictions[valid_index] = fold_prediction
        coverage[valid_index] += 1
        fold_rows.append(
            {
                "seed": seed,
                "fold": fold,
                "train_rows": len(train_index),
                "valid_rows": len(valid_index),
                "train_f1": float(f1_score(encoded_train_y, train_prediction, average="macro")),
                "valid_f1": float(
                    f1_score(
                        fold_valid_labels.to_numpy(),
                        np.asarray(fixed_classes)[fold_prediction],
                        average="macro",
                        labels=fixed_classes,
                        zero_division=0,
                    )
                ),
            }
        )
        fold_bundles.append(
            {
                "fold": fold,
                "preprocessor": preprocessor,
                "model": model,
                "calibrator": calibrator,
                "probability_mode": fold_mode,
                "class_names": fixed_classes,
            }
        )

    if not np.all(coverage == 1) or (predictions < 0).any():
        raise RuntimeError("OOF coverage invariant failed")
    if probability_mode != "hard_only":
        _validate_probabilities(probabilities)

    paths.run_dir.mkdir(parents=True, exist_ok=False)
    np.save(paths.run_dir / "oof_prediction_by_seed.npy", predictions[None, :])
    if probability_mode != "hard_only":
        np.save(paths.run_dir / "oof_probability_by_seed.npy", probabilities[None, :, :])
    pd.DataFrame(
        {
            "row_index": np.arange(len(train)),
            identifier: train[identifier].to_numpy(),
            "seed": seed,
            "fold": folds,
        }
    ).to_csv(paths.run_dir / "fold_assignments.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(paths.run_dir / "fold_metrics.csv", index=False)
    (paths.run_dir / "class_names.json").write_text(
        json.dumps(fixed_classes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    runtime = {
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - started,
        **_peak_memory(),
    }
    (paths.run_dir / "runtime_metrics.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    passport = {
        "experiment_name": config["project"]["experiment_name"],
        "model_name": config["model"]["name"],
        "probability_mode": probability_mode,
        "probability_ensemble_eligible": probability_mode != "hard_only",
        "bundle_safe_for_infer": True,
        "oof_macro_f1": float(
            f1_score(
                labels.to_numpy(),
                np.asarray(fixed_classes)[predictions],
                average="macro",
                labels=fixed_classes,
                zero_division=0,
            )
        ),
    }
    (paths.run_dir / "model_passport.json").write_text(
        json.dumps(passport, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (paths.run_dir / "io_audit.json").write_text(
        json.dumps(io_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (paths.run_dir / "fold_bundles.pkl").open("wb") as file:
        pickle.dump(fold_bundles, file, protocol=pickle.HIGHEST_PROTOCOL)
    protected_after = {
        name: _sha256_file(_repo_root() / name) for name in PROTECTED_FILES
    }
    if protected_after != identity["protected_hashes_before"]:
        raise RuntimeError("protected source files changed during OOF execution")
    artifact_names = sorted(path.name for path in paths.run_dir.iterdir())
    artifacts = _artifact_integrity(paths.run_dir, artifact_names)
    manifest = {
        **identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "class_names": fixed_classes,
        "class_names_hash": _sha256_file(paths.run_dir / "class_names.json"),
        "fold_assignments_hash": _sha256_file(paths.run_dir / "fold_assignments.csv"),
        "artifact_bundle_hash": _sha256_file(paths.run_dir / "fold_bundles.pkl"),
        "bundle_safe_for_infer": True,
        "library_versions": library_versions,
        "external_runtime_snapshot": external_runtime_snapshot,
        "train_rows": len(train),
        "probability_mode": probability_mode,
        "score_source": "train_only_outer_oof",
        "protected_hashes_after": protected_after,
        "outputs": artifact_names,
        "artifacts": artifacts,
    }
    paths.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths.run_dir


def _load_frozen_manifest(
    artifact_root: Path, config_path: Path, seed: int
) -> tuple[dict, dict, dict]:
    path = artifact_root / "frozen_manifest.json"
    if not path.is_file():
        raise RuntimeError("infer is blocked: frozen_manifest.json is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": SCHEMA_VERSION,
        "frozen": True,
        "selection_uses_test": False,
        "thresholds_frozen": True,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise RuntimeError("infer is blocked: frozen manifest safety fields are invalid")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("infer is blocked: frozen manifest has no candidates")
    config_hash = _sha256_file(config_path)
    match = next(
        (
            item
            for item in candidates
            if item.get("config_hash") == config_hash and seed in item.get("seeds", [])
        ),
        None,
    )
    if match is None:
        raise RuntimeError("infer is blocked: requested config/seed is not frozen")
    class_names = manifest.get("class_names")
    if not isinstance(class_names, list) or len(class_names) != 26 or len(set(class_names)) != 26:
        raise RuntimeError("infer is blocked: frozen class order is invalid")
    runs = match.get("runs")
    expected = runs.get(str(seed)) if isinstance(runs, dict) else None
    required_run_fields = {
        "run_manifest_hash",
        "train_data_hash",
        "source_hash",
        "fold_assignments_hash",
        "class_names_hash",
        "artifact_bundle_hash",
        "bundle_safe_for_infer",
        "library_versions",
        "external_runtime_snapshot",
    }
    if not isinstance(expected, dict) or not required_run_fields.issubset(expected):
        raise RuntimeError("infer is blocked: frozen OOF run identity is incomplete")
    return manifest, match, expected


def _validate_frozen_oof(
    config: dict,
    train_path: Path,
    run_paths: RunPaths,
    frozen: dict,
    expected: dict,
) -> dict:
    # Check environment identity before hashing Train bytes or opening the OOF
    # artifact bundle.
    live_environment = {
        "library_versions": _library_versions(config),
        "external_runtime_snapshot": {},
    }
    for field, actual in live_environment.items():
        if actual != expected[field]:
            raise RuntimeError(f"infer is blocked: frozen {field} does not match")
    if not run_paths.manifest.is_file():
        raise RuntimeError("infer is blocked: frozen OOF run manifest is missing")
    if _sha256_file(run_paths.manifest) != expected["run_manifest_hash"]:
        raise RuntimeError("infer is blocked: OOF run manifest hash changed")
    run_manifest = json.loads(run_paths.manifest.read_text(encoding="utf-8"))
    comparisons = {
        "train_data_hash": _sha256_file(train_path),
        "source_hash": _canonical_hash(_source_hashes(config)),
        "fold_assignments_hash": _sha256_file(run_paths.run_dir / "fold_assignments.csv"),
        "class_names_hash": _sha256_file(run_paths.run_dir / "class_names.json"),
        "artifact_bundle_hash": _sha256_file(run_paths.run_dir / "fold_bundles.pkl"),
    }
    for field, actual in comparisons.items():
        if actual != expected[field] or run_manifest.get(field) != expected[field]:
            raise RuntimeError(f"infer is blocked: frozen {field} does not match")
    for field, actual in live_environment.items():
        if run_manifest.get(field) != actual:
            raise RuntimeError(f"infer is blocked: OOF {field} does not match")
    if expected["bundle_safe_for_infer"] is not True or run_manifest.get(
        "bundle_safe_for_infer"
    ) is not True:
        raise RuntimeError("infer is blocked: bundle_safe_for_infer=false")
    if run_manifest.get("class_names") != frozen["class_names"]:
        raise RuntimeError("infer is blocked: OOF and frozen class orders differ")
    if run_manifest.get("test_accessed") is not False or not _validate_artifacts(
        run_paths.run_dir, run_manifest
    ):
        raise RuntimeError("infer is blocked: OOF artifact integrity validation failed")
    return run_manifest


def _warm_up_bundles(
    bundles: list[dict], train_features: pd.DataFrame, fixed_classes: list[str]
) -> None:
    """Trigger any post-unpickle lazy initialization before Test is opened."""
    raw_warmup = train_features.iloc[: min(64, len(train_features))]
    if raw_warmup.empty:
        raise RuntimeError("infer is blocked: bundle warm-up needs Train rows")
    try:
        for bundle in sorted(bundles, key=lambda item: item["fold"]):
            preprocessor = bundle["preprocessor"]
            model = bundle["model"]
            mode = bundle["probability_mode"]
            lazy_components = [model]
            experts = getattr(model, "experts_", None)
            if isinstance(experts, dict):
                lazy_components.extend(experts.values())
            for component in lazy_components:
                ensure = getattr(component, "_ensure_classifier", None)
                if callable(ensure):
                    ensure()
                    if not hasattr(component, "classifier_"):
                        raise RuntimeError("lazy classifier did not initialize")
            transformed = preprocessor.transform(raw_warmup)
            probability = _predict_probability(
                model,
                mode,
                bundle["calibrator"],
                transformed,
                _local_class_names(preprocessor, fixed_classes),
                fixed_classes,
            )
            if probability is None:
                _predict_labels(model, transformed)
            else:
                _validate_probabilities(probability)
            bundle["warmup_completed"] = True
    except Exception as error:
        raise RuntimeError(
            "infer is blocked: bundle_safe_for_infer=false; Train-only warm-up failed"
        ) from error


def run_infer(
    config_path: Path,
    seed: int,
    artifact_root: Path,
    overwrite: bool = False,
) -> Path:
    config_path = config_path.resolve()
    artifact_root = artifact_root.resolve()
    frozen, _, expected = _load_frozen_manifest(artifact_root, config_path, seed)
    config = load_config(config_path)
    configured_classes, preprocessing_config = _validate_config(config)
    fixed_classes = configured_classes or [str(value) for value in frozen["class_names"]]
    if fixed_classes != frozen["class_names"]:
        raise RuntimeError("infer is blocked: config and frozen class orders differ")
    config = deepcopy(config)
    config["project"]["seed"] = seed
    config_hash = _sha256_file(config_path)
    oof_paths = _run_paths(config, config_hash, seed, artifact_root)
    oof_dir = oof_paths.run_dir
    assignment_path = oof_dir / "fold_assignments.csv"
    if not assignment_path.is_file():
        raise RuntimeError("infer is blocked: frozen OOF fold assignments are missing")

    output_dir = _contained_path(
        artifact_root,
        artifact_root
        / "inference"
        / _safe_slug(config["project"]["experiment_name"], "experiment_name")
        / config_hash
        / _safe_slug(seed, "seed"),
    )
    data = config["data"]
    raw_dir = (_repo_root() / data["raw_dir"]).resolve()
    train_path = raw_dir / data["train_file"]
    oof_manifest = _validate_frozen_oof(config, train_path, oof_paths, frozen, expected)
    train = pd.read_csv(train_path)
    target, identifier = data["target_column"], data["id_column"]
    labels = train[target].astype(str)
    train_features = train.drop(columns=[target, identifier])
    _load_assignments(assignment_path, train, labels, identifier, seed)
    with (oof_dir / "fold_bundles.pkl").open("rb") as file:
        bundles = pickle.load(file)
    if (
        not isinstance(bundles, list)
        or len(bundles) != N_SPLITS
        or {bundle.get("fold") for bundle in bundles} != set(range(N_SPLITS))
        or any(bundle.get("class_names") != fixed_classes for bundle in bundles)
        or {bundle.get("probability_mode") for bundle in bundles}
        != {oof_manifest["probability_mode"]}
    ):
        raise RuntimeError("infer is blocked: frozen fold bundles are invalid")
    _warm_up_bundles(bundles, train_features, fixed_classes)

    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"inference output already exists: {output_dir}")
    if output_dir.exists():
        _safe_remove_tree(artifact_root, output_dir)

    # This is the first and only Test read. All fitting was completed during OOF.
    test = pd.read_csv(raw_dir / data["test_file"])
    test_features = test.drop(columns=[identifier])
    if list(train_features.columns) != list(test_features.columns):
        raise RuntimeError("frozen inference requires an identical train/test feature schema")
    test_probabilities = np.full(
        (1, N_SPLITS, len(test), len(fixed_classes)), np.nan, dtype=np.float64
    )
    hard_predictions = np.full((1, N_SPLITS, len(test)), -1, dtype=np.int32)
    modes: list[str] = []
    for bundle in sorted(bundles, key=lambda item: item["fold"]):
        fold = int(bundle["fold"])
        preprocessor = bundle["preprocessor"]
        model = bundle["model"]
        mode = bundle["probability_mode"]
        calibrator = bundle["calibrator"]
        encoded_test_x = preprocessor.transform(test_features)
        modes.append(mode)
        probability = _predict_probability(
            model,
            mode,
            calibrator,
            encoded_test_x,
            _local_class_names(preprocessor, fixed_classes),
            fixed_classes,
        )
        if probability is None:
            hard_predictions[0, fold] = _prediction_to_fixed(
                _predict_labels(model, encoded_test_x),
                _local_class_names(preprocessor, fixed_classes),
                fixed_classes,
            )
        else:
            test_probabilities[0, fold] = probability
            hard_predictions[0, fold] = probability.argmax(axis=1)
    output_dir.mkdir(parents=True, exist_ok=False)
    np.save(output_dir / "test_prediction_by_seed_fold.npy", hard_predictions)
    if set(modes) != {"hard_only"}:
        _validate_probabilities(test_probabilities.reshape(-1, len(fixed_classes)))
        np.save(output_dir / "test_probability_by_seed_fold.npy", test_probabilities)
    (output_dir / "inference_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "frozen_manifest_hash": _sha256_file(artifact_root / "frozen_manifest.json"),
                "config_hash": config_hash,
                "seed": seed,
                "test_accessed": True,
                "selection_or_threshold_changes": False,
                "probability_mode": modes[0],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("oof", "infer"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--fold-assignments", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=Path("data/processed/oof_bank"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.phase == "oof":
        output = run_oof(
            args.config, args.seed, args.fold_assignments, args.artifact_root, args.overwrite
        )
    else:
        if args.fold_assignments is not None:
            raise ValueError("--fold-assignments is only valid for --phase oof")
        output = run_infer(args.config, args.seed, args.artifact_root, args.overwrite)
    print(json.dumps({"phase": args.phase, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
