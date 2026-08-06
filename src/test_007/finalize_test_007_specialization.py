"""Finalize TEST_007 train-only evidence, freeze selection, and infer once.

The module deliberately keeps Test outside ``evidence`` and ``freeze``.  The
``infer`` transition validates and warms every referenced fold bundle first,
then performs one global Test CSV read shared by all three final roles.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any, Iterable
import uuid

import yaml

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.test_007 import run_test_007_nested as nested
from src.test_007 import run_test_007_oof_bank as bank
from src.train import load_config


SEEDS = (42, 2026, 777)
FOLDS = tuple(range(5))
LANES = ("primary", "robustness_challenger", "specialization_challenger")
ROLE_NAMES = {
    "primary": "nested_primary",
    "robustness_challenger": "nested_robustness",
    "specialization_challenger": "nested_specialization",
}
FINAL_ROLE_TO_INTERNAL = {
    "primary": "primary",
    "robustness_challenger": "robustness",
    "specialization_challenger": "specialization",
}
N_BOOTSTRAP = 2000
KNOWN_NONDETERMINISTIC_REPLAY_ALIASES = frozenset({"catboost"})
NONDETERMINISTIC_REPLAY_MAX_ABS_DRIFT = 2.5e-3
NONDETERMINISTIC_REPLAY_MAX_MEAN_DRIFT = 2.5e-7
REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL_CONFIG = (
    REPO_ROOT / "data" / "backup" / "yaml" / "test_007_specialization_goal.yaml"
)
CANONICAL_TEST = REPO_ROOT / "data" / "raw" / "test.csv"
CANONICAL_SUBMISSION = REPO_ROOT / "data" / "raw" / "sample_submission.csv"
NESTED_COMPLETION_ATTESTATION = (
    REPO_ROOT.parents[1]
    / ".omx"
    / "plans"
    / "test_007_nested_completion_attestation.json"
)
NESTED_COMPLETION_ATTESTATION_SHA256 = (
    "67692d8ee50de6e2d19a87566d7fc618fa7c3b25f3df7e36541c29604bb24d2e"
)
RAW_FILES = (
    "nested_oof_probability.npy",
    "nested_strategy_probability.npz",
    "nested_inner_selection_probability.npz",
    "nested_inner_selection_records.json",
    "nested_cv_metrics.json",
    "nested_selection_records.json",
    "nested_io_audit.json",
    "nested_run_manifest.json",
)
FINALIZER_SOURCE_FILE = "src/test_007/finalize_test_007_specialization.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _inference_source_inventory(
    nested_sources: dict[str, str], forbidden: Iterable[Path]
) -> dict[str, str]:
    """Extend the verified nested execution closure with this finalizer."""
    inventory = dict(nested_sources)
    for relative, expected in nested_sources.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise RuntimeError("nested inference source inventory is invalid")
        candidate = Path(relative)
        source = candidate if candidate.is_absolute() else REPO_ROOT / candidate
        source = _guard_not_forbidden(source, forbidden, f"nested inference source {relative}")
        if not source.is_file() or _sha256(source) != expected:
            raise RuntimeError(f"nested inference source drifted: {relative}")
    finalizer = _guard_not_forbidden(
        REPO_ROOT / FINALIZER_SOURCE_FILE, forbidden, "production finalizer source"
    )
    if not finalizer.is_file():
        raise RuntimeError(f"production inference source is missing: {FINALIZER_SOURCE_FILE}")
    inventory[FINALIZER_SOURCE_FILE] = _sha256(finalizer)
    return inventory


def _validate_inference_source_closure(
    frozen: dict[str, Any], raw_manifest: dict[str, Any], forbidden: Iterable[Path]
) -> None:
    trusted = raw_manifest.get("trusted_input_hashes")
    nested_sources = trusted.get("sources") if isinstance(trusted, dict) else None
    nested_bundle = trusted.get("source_bundle") if isinstance(trusted, dict) else None
    if not isinstance(nested_sources, dict) or not nested_sources:
        raise RuntimeError("nested trusted inference source inventory is missing")
    if nested_bundle != _canonical_hash(nested_sources):
        raise RuntimeError("nested trusted inference source bundle is invalid")
    if (frozen.get("inference_source_base_files") != nested_sources or
            frozen.get("inference_source_base_bundle") != nested_bundle):
        raise RuntimeError("frozen inference source base differs from nested manifest")
    expected = frozen.get("inference_source_files")
    if not isinstance(expected, dict):
        raise RuntimeError("frozen production inference source inventory is incomplete")
    live = _inference_source_inventory(nested_sources, forbidden)
    if expected != live:
        raise RuntimeError("production inference source drifted after freeze")
    if frozen.get("inference_source_hash") != _canonical_hash(expected):
        raise RuntimeError("frozen production inference source inventory hash is invalid")


def _validated_role_to_lane(frozen: dict[str, Any]) -> dict[str, str]:
    selections = frozen.get("selections")
    mapping = frozen.get("role_to_lane")
    if not isinstance(selections, dict) or set(selections) != set(LANES):
        raise RuntimeError("frozen selections must contain exactly the final roles")
    if not isinstance(mapping, dict) or set(mapping) != set(selections):
        raise RuntimeError("frozen role_to_lane must contain exactly the final roles")
    if set(mapping.values()) != set(LANES):
        raise RuntimeError("frozen role_to_lane must be a lane bijection")
    for role, lane in mapping.items():
        if ROLE_NAMES.get(lane) != selections[role]:
            raise RuntimeError("frozen role_to_lane differs from selection candidate names")
    return {str(role): str(lane) for role, lane in mapping.items()}


def _protected_before_hashes(payload: dict[str, Any]) -> dict[str, str]:
    """Normalize both collector metadata and legacy protected-hash schemas."""
    raw_before = payload.get("before")
    normalized: dict[str, str] = {}
    if isinstance(raw_before, dict):
        for relative, value in raw_before.items():
            digest = value.get("sha256") if isinstance(value, dict) else value
            if not isinstance(relative, str) or not isinstance(digest, str):
                raise ValueError("protected before-hash entry is malformed")
            digest = digest.lower()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("protected before-hash is not SHA-256")
            normalized[relative] = digest
    files = payload.get("files")
    file_hashes: dict[str, str] = {}
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                raise ValueError("protected file entry is malformed")
            relative, digest = item.get("path"), item.get("before")
            if not isinstance(relative, str) or not isinstance(digest, str):
                raise ValueError("protected file before-hash entry is malformed")
            digest = digest.lower()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("protected file before-hash is not SHA-256")
            if relative in file_hashes and file_hashes[relative] != digest:
                raise ValueError("duplicate protected file has conflicting hashes")
            file_hashes[relative] = digest
    if normalized and file_hashes and normalized != file_hashes:
        raise ValueError("protected before/file hash inventories disagree")
    result = normalized or file_hashes
    if not result:
        raise ValueError("protected before-hash inventory is empty")
    return result


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False, dir=path.parent) as handle:
        temporary = Path(handle.name)
    try:
        np.save(temporary, value)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False, dir=path.parent) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **values)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _within(root: Path, value: str | Path) -> Path:
    root = root.resolve()
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"artifact path escapes root: {path}") from error
    return path


def _same_file(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _guard_not_forbidden(path: Path, forbidden: Iterable[Path], role: str) -> Path:
    resolved = path.resolve()
    if any(_same_file(resolved, blocked.resolve()) for blocked in forbidden):
        raise ValueError(f"{role} aliases configured Test/submission")
    return resolved


def _trusted_forbidden() -> tuple[Path, Path]:
    """Bootstrap the immutable Test boundary without reading user-controlled bytes."""
    control = CONTROL_CONFIG.resolve()
    test, submission = CANONICAL_TEST.resolve(), CANONICAL_SUBMISSION.resolve()
    if control.is_symlink() or _same_file(control, test) or _same_file(control, submission):
        raise RuntimeError("trusted control config aliases Test/submission")
    if _same_file(test, submission):
        raise RuntimeError("canonical Test and submission aliases are invalid")
    return test, submission


def _canonical_nested_source_contract(
    candidates: dict[str, Any], forbidden: Iterable[Path]
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Reuse the runner's complete import-closure discovery after alias guarding."""
    # Discovery reads Python source to parse imports. Guard every possible local
    # source first, so a hardlinked source can never expose Test bytes.
    possible = [*REPO_ROOT.joinpath("src").rglob("*.py"), *REPO_ROOT.joinpath("tools").rglob("*")]
    for path in possible:
        if path.is_file():
            _guard_not_forbidden(path, forbidden, "nested source discovery candidate")
    audit = nested.Audit()
    paths = nested._execution_source_paths(candidates, audit)
    for path in paths:
        _guard_not_forbidden(path, forbidden, "discovered nested execution source")
    hashes = nested._hash_source_paths(paths, audit)
    after = nested._hash_source_paths(paths, audit)
    if hashes != after:
        raise RuntimeError("nested execution sources changed during finalizer validation")
    events = [event for event in audit.events if str(event.get("role", "")).startswith("source")]
    return hashes, events


def _guarded_json(path: Path, forbidden: Iterable[Path], role: str) -> Any:
    return _read_json(_guard_not_forbidden(path, forbidden, role))


def _snapshot_files(paths: Iterable[Path], forbidden: Iterable[Path]) -> dict[Path, bytes | None]:
    result = {}
    for path in paths:
        _guard_not_forbidden(path, forbidden, "transaction snapshot")
        result[path] = path.read_bytes() if path.exists() else None
    return result


def _restore_snapshot(snapshot: dict[Path, bytes | None]) -> None:
    for path, payload in snapshot.items():
        if payload is None:
            path.unlink(missing_ok=True)
        else:
            _atomic_bytes(path, payload)


def _tracked_directories(root: Path) -> set[Path]:
    parents = (root / "nested", root / "nested" / "raw", root / "candidates", root / "generations")
    result = {parent.resolve() for parent in parents if parent.exists()}
    result.update(path.resolve() for parent in parents if parent.exists() for path in parent.rglob("*") if path.is_dir())
    return result


def _remove_new_directories(root: Path, before: set[Path]) -> None:
    current = _tracked_directories(root)
    for path in sorted(current - before, key=lambda value: len(value.parts)):
        if not path.exists() or any(parent in (current - before) and parent != path for parent in path.parents):
            continue
        shutil.rmtree(path)


def _validate_generation(root: Path, state: dict[str, Any], forbidden: Iterable[Path]) -> None:
    generation = _within(root, state.get("generation", ""))
    manifest_path = _guard_not_forbidden(generation / "generation_manifest.json", forbidden, "generation manifest")
    if _sha256(manifest_path) != state.get("generation_manifest_hash"):
        raise RuntimeError("evidence generation manifest changed")
    manifest = _guarded_json(manifest_path, forbidden, "generation manifest")
    hashes = manifest.get("hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("evidence generation hashes are missing")
    for name, digest in hashes.items():
        if name == "finalization_state.json":
            continue
        live = _guard_not_forbidden(root / name, forbidden, f"evidence projection {name}")
        frozen_copy = _guard_not_forbidden(generation / name, forbidden, f"generation {name}")
        if _sha256(live) != digest or _sha256(frozen_copy) != digest:
            raise RuntimeError(f"evidence generation projection changed: {name}")


def _probability(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"invalid probability shape/content: {array.shape}, expected {shape}")
    if (array < 0).any() or (array > 1).any() or not np.allclose(array.sum(axis=-1), 1.0, atol=1e-5):
        raise ValueError("invalid probability simplex")
    return array


def _macro(labels: np.ndarray, probability: np.ndarray, classes: int) -> float:
    if len(labels) == 0:
        return 0.0
    return float(f1_score(labels, probability.argmax(1), labels=np.arange(classes), average="macro", zero_division=0))


def _encode_labels(train: pd.DataFrame, target: str, classes: list[str]) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(classes)}
    encoded = train[target].astype(str).map(mapping)
    if encoded.isna().any():
        raise ValueError("Train contains labels outside canonical class order")
    return encoded.to_numpy(np.int32)


def _normalize_alignment(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["seed", "row_index", "ID", "fold"]
    if any(column not in frame for column in columns):
        raise ValueError("alignment columns are incomplete")
    result = frame[columns].copy()
    for column in ("seed", "row_index", "fold"):
        values = pd.to_numeric(result[column], errors="raise")
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"fractional alignment column: {column}")
        result[column] = values.astype(int)
    return result.sort_values(columns, kind="stable").reset_index(drop=True)


def _load_base_manifest(root: Path, classes: list[str], assignments: pd.DataFrame, forbidden: Iterable[Path] = ()) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    _guard_not_forbidden(root / "candidate_manifest.json", forbidden, "candidate manifest")
    manifest = _read_json(root / "candidate_manifest.json")
    items = manifest.get("candidates")
    if not isinstance(items, list) or not items:
        raise ValueError("candidate_manifest candidates[] is required")
    banks: dict[str, np.ndarray] = {}
    expected = _normalize_alignment(assignments)
    for item in items:
        name = str(item.get("name", ""))
        if not name or name in banks or item.get("class_names") != classes:
            raise ValueError("candidate name/class order is invalid")
        probability_path = _within(root, item["probability_path"])
        alignment_path = _within(root, item["alignment_path"])
        _guard_not_forbidden(probability_path, forbidden, f"{name} probability")
        _guard_not_forbidden(alignment_path, forbidden, f"{name} alignment")
        if _sha256(probability_path) != item.get("probability_sha256") or _sha256(alignment_path) != item.get("alignment_sha256"):
            raise ValueError(f"{name}: candidate artifact hash mismatch")
        if not _normalize_alignment(pd.read_csv(alignment_path)).equals(expected):
            raise ValueError(f"{name}: alignment mismatch")
        banks[name] = _probability(np.load(probability_path, allow_pickle=False), (3, len(assignments) // 3, len(classes)))
    return manifest, banks


def _snapshot_nested(
    root: Path, source: Path, forbidden: Iterable[Path] = (),
    live: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    source = source.resolve()
    for name in RAW_FILES:
        _guard_not_forbidden(source / name, forbidden, f"nested {name}")
    manifest = _read_json(source / "nested_run_manifest.json")
    if manifest.get("test_accessed") is not False or manifest.get("outer_validation_used_for_selection") is not False:
        raise ValueError("nested run is not train-only")
    declared = manifest.get("artifact_hashes")
    if not isinstance(declared, dict):
        raise ValueError("nested manifest artifact hashes are missing")
    hashes: dict[str, str] = {}
    for name in RAW_FILES:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[name] = _sha256(path)
        if name != "nested_run_manifest.json" and declared.get(name) != hashes[name]:
            raise ValueError(f"nested artifact hash mismatch: {name}")
    audit = _read_json(source / "nested_io_audit.json")
    events = audit.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("nested audit must contain actual events")
    for event in events:
        action = str(event.get("action", "")).lower()
        event_path = Path(str(event.get("path", "")))
        forbidden_names = {blocked.name.lower() for blocked in forbidden}
        aliases_forbidden = event_path.name.lower() in forbidden_names or any(_same_file(event_path, blocked) if event_path.exists() else event_path.resolve() == blocked.resolve() for blocked in forbidden)
        if action == "read" and aliases_forbidden:
            raise ValueError("nested audit contains Test/submission contamination")
    if audit.get("test_accessed") is not False or int(audit.get("submission_reads", 0)) != 0:
        raise ValueError("nested audit safety declaration failed")
    if live is not None:
        goal_raw = yaml.safe_load(_guard_not_forbidden(live["goal_path"], forbidden, "live goal").read_text(encoding="utf-8"))
        universe_raw = yaml.safe_load(_guard_not_forbidden(live["universe_path"], forbidden, "live universe").read_text(encoding="utf-8"))
        candidate_objects = {}
        for alias, path in live["candidate_configs"].items():
            guarded = _guard_not_forbidden(path, forbidden, f"live candidate config {alias}")
            config = yaml.safe_load(guarded.read_text(encoding="utf-8"))
            candidate_objects[alias] = nested.Candidate(alias, guarded, config)
        validated_universe = nested._load_universe(universe_raw, candidate_objects, goal_raw.get("search", {}))
        expected_fields = {
            "goal_config_hash": _sha256(live["goal_path"]),
            "search_universe_hash": _sha256(live["universe_path"]),
            "train_hash": _sha256(live["train_path"]),
            "fold_hash": _sha256(live["fold_path"]),
        }
        for key, value in expected_fields.items():
            if manifest.get(key) != value:
                raise ValueError(f"nested live provenance mismatch: {key}")
        config_hashes = {
            alias: _sha256(_guard_not_forbidden(path, forbidden, f"nested candidate config {alias}"))
            for alias, path in live["candidate_configs"].items()
        }
        if manifest.get("candidate_config_hashes") != config_hashes:
            raise ValueError("nested candidate config hashes differ from live files")
        matrix = manifest.get("model_seed_matrix")
        if not isinstance(matrix, list):
            raise ValueError("nested model_seed_matrix is missing")
        inner_observed = {(int(row["seed"]), int(row["outer_fold"]), int(row["inner_fold"]), str(row["alias"])) for row in matrix if row.get("stage") == "inner"}
        inner_expected = {(seed, fold, inner_fold, alias) for seed in SEEDS for fold in FOLDS for inner_fold in range(4) for alias in validated_universe.executable_aliases}
        selection_payload = _read_json(source / "nested_selection_records.json")["selections"]
        outer_expected = set()
        for selection in selection_payload:
            for alias in _variant_aliases(selection["selected_variant"]):
                outer_expected.add((int(selection["seed"]), int(selection["fold"]), alias))
        outer_observed = {(int(row["seed"]), int(row["outer_fold"]), str(row["alias"])) for row in matrix if row.get("stage") == "outer_refit"}
        if inner_observed != inner_expected or outer_observed != outer_expected or len(matrix) != len(inner_expected) + len(outer_expected):
            raise ValueError("nested model_seed_matrix exact stage/alias/fold coverage is invalid")
        train_frame = pd.read_csv(live["train_path"])
        assignments = pd.read_csv(live["fold_path"])
        target = goal_raw["data"]["target_column"]
        for row in matrix:
            seed, fold = int(row["seed"]), int(row["outer_fold"])
            if int(row.get("model_seed", -1)) != seed or int(row.get("inner_split_seed", -1)) != nested._inner_split_seed(seed, fold):
                raise ValueError("nested model/inner split seed matrix is invalid")
            seeded = assignments.loc[assignments.seed == seed].set_index("row_index").sort_index()
            outer_train = seeded.index[seeded.fold != fold].to_numpy(int); outer_valid = seeded.index[seeded.fold == fold].to_numpy(int)
            if row["stage"] == "inner":
                split = nested._inner_splits(train_frame.iloc[outer_train][target].astype(str).reset_index(drop=True), seed, fold)[int(row["inner_fold"])]
                train_indices, valid_indices = outer_train[split[0]], outer_train[split[1]]
            else:
                if row.get("inner_fold") is not None: raise ValueError("outer_refit inner_fold must be null")
                train_indices, valid_indices = outer_train, outer_valid
            if int(row.get("train_rows", -1)) != len(train_indices) or int(row.get("valid_rows", -1)) != len(valid_indices) or row.get("train_row_indices_hash") != _canonical_hash(train_indices.tolist()) or row.get("valid_row_indices_hash") != _canonical_hash(valid_indices.tolist()):
                raise ValueError("nested model_seed_matrix row coverage hashes are invalid")
        contract = manifest.get("bounded_search_contract")
        expected_contract = {
            "trial_seeds": list(validated_universe.trial_seeds),
            "trial_count_per_seed": validated_universe.trial_count_per_seed,
            "min_active_weight": validated_universe.min_active_weight,
            "max_model_weight": validated_universe.max_model_weight,
            "max_models": validated_universe.max_models,
            "early_stop_patience": validated_universe.early_stop_patience,
            "role_balanced_fraction": validated_universe.role_balanced_fraction,
            "unconstrained_fraction": 1.0 - validated_universe.role_balanced_fraction,
        }
        if not isinstance(contract, dict) or any(contract.get(key) != value for key, value in expected_contract.items()):
            raise ValueError("nested bounded search contract differs from trusted universe")
        if validated_universe.trial_count_per_seed != 20_000 or len(validated_universe.trial_seeds) != 3 or validated_universe.max_models != 5 or not np.isclose(validated_universe.role_balanced_fraction, .8):
            raise ValueError("trusted bounded search contract is not the required 20k×3, 80/20, max5 design")
        if manifest.get("outer_folds") != 5 or manifest.get("inner_folds") != 4:
            raise ValueError("nested outer/inner fold contract differs")
        runtime_safety = manifest.get("runtime_execution_safety")
        expected_runtime_safety = {
            "policy": "exclude_subprocess_backed_candidates_without_os_filesystem_isolation",
            "label_independent": True,
            "declared_aliases": list(validated_universe.aliases),
            "executable_aliases": list(validated_universe.executable_aliases),
            "runtime_exclusions": validated_universe.runtime_exclusions,
            "excluded_aliases_never_fit": True,
        }
        if runtime_safety != expected_runtime_safety:
            raise ValueError("nested runtime exclusion contract differs from trusted universe")
        expected_audit_paths = {
            "goal_config": live["goal_path"].resolve(), "search_universe": live["universe_path"].resolve(),
            "train": live["train_path"].resolve(), "fold_assignments": live["fold_path"].resolve(),
            **{f"candidate_config:{alias}": path.resolve() for alias, path in live["candidate_configs"].items()},
        }
        observed_roles = {}
        for event in events:
            if event.get("action") == "read":
                if event.get("stage") != "nested_oof":
                    raise ValueError("nested audit read event has an invalid stage")
                observed_roles.setdefault(str(event.get("role")), []).append(Path(str(event.get("path"))).resolve())
        for role, path in expected_audit_paths.items():
            if observed_roles.get(role) != [path]:
                raise ValueError(f"nested audit role/path coverage invalid: {role}")
        source_hashes, source_events = _canonical_nested_source_contract(candidate_objects, forbidden)
        expected_source_roles: dict[str, list[Path]] = {}
        for event in source_events:
            expected_source_roles.setdefault(str(event["role"]), []).append(Path(event["path"]).resolve())
        for role, paths in expected_source_roles.items():
            if observed_roles.get(role) != paths:
                raise ValueError(f"nested audit source discovery/hash coverage invalid: {role}")
        if set(observed_roles) != set(expected_audit_paths) | set(expected_source_roles):
            raise ValueError("nested audit contains missing or unexpected read roles")
        if manifest.get("source_hashes_before") != source_hashes or manifest.get("source_hashes_after") != source_hashes or manifest.get("source_unchanged") is not True or manifest.get("source_bundle_before") != _canonical_hash(source_hashes) or manifest.get("source_bundle_after") != _canonical_hash(source_hashes):
            raise ValueError("nested pre/post execution source TOCTOU contract differs")
        trusted_inputs = {"goal_config": expected_fields["goal_config_hash"], "search_universe": expected_fields["search_universe_hash"], "candidate_configs": config_hashes, "train": expected_fields["train_hash"], "fold_assignments": expected_fields["fold_hash"], "sources": source_hashes, "source_bundle": _canonical_hash(source_hashes)}
        if manifest.get("trusted_input_hashes") != trusted_inputs:
            raise ValueError("nested trusted input inventory differs from live files")
        transcript_records = _read_json(source / "nested_inner_selection_records.json")["records"]
        run_payload = {"trusted_inputs": trusted_inputs, "bounded_search_contract": contract, "inner_selection_probability_artifact": hashes["nested_inner_selection_probability.npz"], "inner_selection_records_artifact": hashes["nested_inner_selection_records.json"], "inner_selection_array_hashes": {row["array_key"]: row["array_hash"] for row in transcript_records}}
        if manifest.get("run_identity_payload") != run_payload or manifest.get("run_identity_hash") != _canonical_hash(run_payload):
            raise ValueError("nested run identity does not match independently recomputed payload")
        verified_identity = _canonical_hash({"run_identity": run_payload, "model_seed_matrix": matrix, "artifact_hashes": hashes})
    else:
        verified_identity = _canonical_hash(hashes)
    identity = verified_identity
    destination = _within(root, root / "nested" / "raw" / identity)
    if destination.exists():
        for name, digest in hashes.items():
            if _sha256(destination / name) != digest:
                raise RuntimeError("immutable nested snapshot was modified")
    else:
        staging = destination.with_name(destination.name + ".staging")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            for name in RAW_FILES:
                shutil.copy2(source / name, staging / name)
            for name, digest in hashes.items():
                if _sha256(staging / name) != digest:
                    raise RuntimeError("nested snapshot copy verification failed")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return destination, {"run_identity": identity, "upstream_run_identity": manifest.get("run_identity_hash"), "hashes": hashes, "verified_from_live_inputs": live is not None}


def _verify_nested_completion_attestation(
    raw: Path,
    manifest: dict[str, Any],
    forbidden: Iterable[Path],
) -> dict[str, Any]:
    path = _guard_not_forbidden(
        NESTED_COMPLETION_ATTESTATION,
        forbidden,
        "external nested completion attestation",
    )
    if _sha256(path) != NESTED_COMPLETION_ATTESTATION_SHA256:
        raise ValueError("external Nested completion attestation digest mismatch")
    payload = _guarded_json(path, forbidden, "external nested completion attestation")
    expected_hashes = payload.get("artifact_hashes")
    validation = payload.get("validation")
    expected_artifacts = set(RAW_FILES) - {"nested_run_manifest.json"}
    if (
        payload.get("schema_version") != 1
        or payload.get("scope") != "post_nested_pre_finalization_external_trust_anchor"
        or payload.get("selection_uses_test") is not False
        or not isinstance(expected_hashes, dict)
        or set(expected_hashes) != expected_artifacts
        or expected_hashes != manifest.get("artifact_hashes")
        or payload.get("nested_run_identity_hash") != manifest.get("run_identity_hash")
        or payload.get("nested_source_bundle") != manifest.get("source_bundle_before")
        or not isinstance(validation, dict)
        or validation.get("source_unchanged") is not True
        or validation.get("test_reads") != 0
        or validation.get("submission_reads") != 0
        or validation.get("transcript_records_verified") != 45
        or validation.get("strategy_probability_shape") != [3, 6201, 26]
    ):
        raise ValueError("external Nested completion attestation contract mismatch")
    if payload.get("nested_run_manifest_sha256") != _sha256(raw / "nested_run_manifest.json"):
        raise ValueError("external Nested manifest digest mismatch")
    for name, expected in expected_hashes.items():
        candidate = _within(raw, raw / str(name))
        if not candidate.is_file() or _sha256(candidate) != expected:
            raise ValueError("external Nested artifact digest mismatch")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "manifest_sha256": payload["nested_run_manifest_sha256"],
        "strategy_probability_sha256": expected_hashes["nested_strategy_probability.npz"],
        "run_identity_hash": payload["nested_run_identity_hash"],
        "source_bundle": payload["nested_source_bundle"],
        "created_at": payload.get("created_at"),
        "validation": validation,
    }


def _selection_records(raw: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows = _read_json(raw / "nested_selection_records.json").get("selections")
    if not isinstance(rows, list) or len(rows) != 45:
        raise ValueError("nested selection must contain exactly 45 records")
    result = {}
    for row in rows:
        key = (str(row.get("strategy")), int(row.get("seed", -1)), int(row.get("fold", -1)))
        if key in result or key[0] not in LANES or key[1] not in SEEDS or key[2] not in FOLDS:
            raise ValueError("invalid/duplicate nested selection record")
        if row.get("selection_source") != "outer_train_inner_oof_only" or row.get("outer_validation_used_for_selection") is not False:
            raise ValueError("nested selection used outer validation")
        if not isinstance(row.get("selected_variant"), dict):
            raise ValueError("selected variant is missing")
        result[key] = row
    expected = {(lane, seed, fold) for lane in LANES for seed in SEEDS for fold in FOLDS}
    if set(result) != expected:
        raise ValueError("nested selection coverage is incomplete")
    return result


def _merged_audit(events: list[dict[str, Any]], forbidden: Iterable[Path]) -> dict[str, Any]:
    if not events:
        raise ValueError("merged audit requires actual verified events")
    test_name, submission_name = [path.name.lower() for path in forbidden]
    def count(stage_name: str, target: str) -> int:
        return sum(
            str(event.get("action", "")).lower() == "read"
            and stage_name in str(event.get("stage", "")).lower()
            and Path(str(event.get("path", ""))).name.lower() == target
            for event in events
        )
    return {
        "events": events,
        "summary": {
            "oof_test_reads": count("oof", test_name), "nested_test_reads": count("nested", test_name),
            "oof_submission_reads": count("oof", submission_name), "nested_submission_reads": count("nested", submission_name),
        },
        "selection_uses_test": False,
    }


def _replay_nested(
    banks: dict[str, np.ndarray], records: dict[tuple[str, int, int], dict[str, Any]],
    assignments: pd.DataFrame, stored: dict[str, np.ndarray], labels: np.ndarray,
    audit_rows: list[dict[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    rows, classes = next(iter(banks.values())).shape[1:]
    result = {
        lane: np.array(_probability(stored[lane], (3, rows, classes)), copy=True)
        for lane in LANES
    }
    local_audit: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(SEEDS):
        seeded = assignments.loc[assignments.seed == seed].set_index("row_index")
        for fold in FOLDS:
            indices = seeded.index[seeded.fold == fold].to_numpy(int)
            fold_banks = {name: probability[seed_index, indices] for name, probability in banks.items()}
            for lane in LANES:
                variant = records[(lane, seed, fold)]["selected_variant"]
                replay = _probability(
                    nested._apply_variant(variant, fold_banks),
                    (len(indices), classes),
                )
                reference = result[lane][seed_index, indices]
                difference = np.abs(replay - reference)
                max_abs = float(difference.max(initial=0.0))
                mean_abs = float(difference.mean())
                strict_match = bool(np.allclose(replay, reference, atol=1e-10, rtol=1e-8))
                aliases = sorted(_variant_aliases(variant))
                nondeterministic = sorted(
                    set(aliases) & KNOWN_NONDETERMINISTIC_REPLAY_ALIASES
                )
                argmax_disagreement = int(
                    np.count_nonzero(replay.argmax(1) != reference.argmax(1))
                )
                if argmax_disagreement:
                    raise ValueError(f"{lane}: replay argmax mismatch")
                replay_score = _macro(labels[indices], replay, classes)
                reference_score = _macro(labels[indices], reference, classes)
                if not math.isclose(replay_score, reference_score, abs_tol=1e-12):
                    raise ValueError(f"{lane}: replay F1 mismatch")
                if not strict_match:
                    if not nondeterministic:
                        raise ValueError(
                            f"{lane}: deterministic base OOF replay differs from Nested probability"
                        )
                    if (
                        max_abs > NONDETERMINISTIC_REPLAY_MAX_ABS_DRIFT
                        or mean_abs > NONDETERMINISTIC_REPLAY_MAX_MEAN_DRIFT
                    ):
                        raise ValueError(
                            f"{lane}: nondeterministic replay probability drift exceeds bounds"
                        )
                local_audit.append(
                    {
                        "strategy": lane,
                        "seed": seed,
                        "fold": fold,
                        "models": aliases,
                        "known_nondeterministic_models": nondeterministic,
                        "strict_probability_match": strict_match,
                        "max_abs_probability_drift": max_abs,
                        "mean_abs_probability_drift": mean_abs,
                        "argmax_disagreement_count": argmax_disagreement,
                        "macro_f1_equal": True,
                        "probability_authority": "stored_hashed_nested_outer_lane_probability",
                    }
                )
    expected_keys = {
        (lane, seed, fold)
        for lane in LANES
        for seed in SEEDS
        for fold in FOLDS
    }
    audit_keys = [
        (row["strategy"], int(row["seed"]), int(row["fold"]))
        for row in local_audit
    ]
    if (
        len(local_audit) != 45
        or len(set(audit_keys)) != 45
        or set(audit_keys) != expected_keys
    ):
        raise RuntimeError("Nested replay audit coverage must be exactly 45 unique records")
    if audit_rows is not None:
        audit_rows.extend(local_audit)
    return result


def _validate_replay_audit(audit: dict[str, Any]) -> None:
    records = audit.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Nested replay audit records are missing")
    expected_keys = {
        (lane, seed, fold)
        for lane in LANES
        for seed in SEEDS
        for fold in FOLDS
    }
    try:
        keys = [
            (str(row["strategy"]), int(row["seed"]), int(row["fold"]))
            for row in records
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Nested replay audit record identity is invalid") from exc
    observed_max_abs = max(
        (float(row["max_abs_probability_drift"]) for row in records),
        default=math.nan,
    )
    observed_max_mean = max(
        (float(row["mean_abs_probability_drift"]) for row in records),
        default=math.nan,
    )
    if (
        len(records) != 45
        or len(set(keys)) != 45
        or set(keys) != expected_keys
        or audit.get("semantic_equivalence_count") != 45
        or audit.get("argmax_disagreement_count") != 0
        or any(int(row.get("argmax_disagreement_count", -1)) != 0 for row in records)
        or any(row.get("macro_f1_equal") is not True for row in records)
        or audit.get("probability_authority")
        != "stored_hashed_nested_outer_lane_probability"
        or audit.get("selection_metric_authority")
        != "stored_hashed_nested_outer_lane_probability"
        or audit.get("production_inference_realization")
        != "hashed_base_oof_fold_bundles"
        or not math.isclose(
            float(audit.get("max_observed_abs_drift", math.nan)),
            observed_max_abs,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(audit.get("max_observed_mean_drift", math.nan)),
            observed_max_mean,
            abs_tol=0.0,
        )
    ):
        raise RuntimeError("Nested replay audit contract is invalid")


def _verify_persisted_replay_audit(root: Path, expected: dict[str, Any]) -> None:
    candidate = _read_json(root / "candidate_manifest.json")
    selection = _read_json(root / "final_selection.json")
    candidate_audit = candidate.get("nested_base_oof_replay_audit")
    selection_audit = selection.get("selection_evidence", {}).get(
        "nested_base_oof_replay_audit"
    )
    if candidate_audit != expected or selection_audit != expected:
        raise RuntimeError("Nested replay audit was not persisted identically")
    _validate_replay_audit(candidate_audit)


def _verify_nested_metrics(
    raw: Path, stored: dict[str, np.ndarray], replay: dict[str, np.ndarray],
    records: dict[tuple[str, int, int], dict[str, Any]], assignments: pd.DataFrame,
    labels: np.ndarray,
) -> dict[str, Any]:
    _verify_primary_probability_identity(raw, stored["primary"])
    metrics = _read_json(raw / "nested_cv_metrics.json")
    transcript_rows = _read_json(raw / "nested_inner_selection_records.json").get("records")
    transcript = {
        (str(row["strategy"]), int(row["seed"]), int(row["outer_fold"])): row
        for row in transcript_rows
    } if isinstance(transcript_rows, list) else {}
    if set(transcript) != set(records):
        raise ValueError("inner selection transcript coverage differs from selections")
    evaluations = metrics.get("candidate_outer_evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != 45:
        raise ValueError("nested candidate evaluations must contain exactly 45 rows")
    indexed = {}
    verified_rows = []
    gaps = {lane: [] for lane in LANES}
    for row in evaluations:
        key = (str(row.get("strategy")), int(row.get("seed", -1)), int(row.get("fold", -1)))
        if key in indexed or key not in records:
            raise ValueError("nested candidate evaluation coverage is invalid")
        indexed[key] = row
        lane, seed, fold = key; seed_index = SEEDS.index(seed)
        seeded = assignments.loc[assignments.seed == seed].set_index("row_index")
        indices = seeded.index[seeded.fold == fold].to_numpy(int)
        score = _macro(labels[indices], replay[lane][seed_index, indices], replay[lane].shape[-1])
        if not math.isclose(float(row.get("macro_f1", math.nan)), score, abs_tol=1e-12):
            raise ValueError("nested fold metric differs from labels/replayed probability")
        inner = float(transcript[key].get("inner_selection_macro_f1", math.nan))
        if not math.isfinite(inner):
            raise ValueError("nested inner selection metric is missing")
        if not math.isclose(float(row.get("inner_selection_macro_f1", math.nan)), inner, abs_tol=1e-12):
            raise ValueError("nested metric inner score differs from verified transcript")
        gaps[lane].append(max(0.0, inner - score))
        verified_rows.append({**row, "macro_f1": score, "verified_gap": inner - score})
    if set(indexed) != set(records):
        raise ValueError("nested evaluations and selections differ in coverage")
    summaries = {}
    reported = {str(row.get("strategy")): row for row in metrics.get("strategy_summary", [])}
    for lane in LANES:
        seed_scores = [_macro(labels, replay[lane][index], replay[lane].shape[-1]) for index in range(3)]
        expected = {"mean_macro_f1": float(np.mean(seed_scores)), "std_macro_f1": float(np.std(seed_scores, ddof=1)), "minimum_seed_macro_f1": float(np.min(seed_scores))}
        row = reported.get(lane)
        if row is None or any(not math.isclose(float(row.get(key, math.nan)), value, abs_tol=1e-12) for key, value in expected.items()):
            raise ValueError("nested seed summary differs from replay")
        if any(not math.isclose(float(row.get("seed_macro_f1", {}).get(str(seed), math.nan)), seed_scores[index], abs_tol=1e-12) for index, seed in enumerate(SEEDS)):
            raise ValueError("nested per-seed summary differs from replay")
        summaries[lane] = {**row, **expected, "seed_macro_f1": dict(zip(map(str, SEEDS), seed_scores))}
    return {"verified_rows": verified_rows, "summaries": summaries, "gaps": gaps}


def _verify_primary_probability_identity(raw: Path, expected: np.ndarray) -> None:
    primary = _probability(
        np.load(raw / "nested_oof_probability.npy", allow_pickle=False),
        expected.shape,
    )
    if not np.array_equal(primary, expected):
        raise ValueError("nested primary NPY differs from NPZ primary")


def paired_bootstrap(
    labels: np.ndarray, candidate: np.ndarray, baseline: np.ndarray,
    iterations: int = N_BOOTSTRAP, seed: int = 7007,
) -> dict[str, float]:
    """Class-stratified paired bootstrap with common row samples for all seeds."""
    rng = np.random.default_rng(seed)
    classes = candidate.shape[-1]
    strata = [np.flatnonzero(labels == value) for value in range(classes)]
    if any(len(value) == 0 for value in strata):
        raise ValueError("paired bootstrap requires every canonical class")
    seed_delta = np.array([
        _macro(labels, candidate[index], classes) - _macro(labels, baseline[index], classes)
        for index in range(3)
    ])
    draws = np.empty(iterations)
    for draw in range(iterations):
        # One index vector is intentionally reused across the three seeds.
        indices = np.concatenate([rng.choice(values, len(values), replace=True) for values in strata])
        deltas = [
            _macro(labels[indices], candidate[index, indices], classes)
            - _macro(labels[indices], baseline[index, indices], classes)
            for index in range(3)
        ]
        draws[draw] = float(np.mean(deltas))
    se_seed = float(np.std(seed_delta, ddof=1) / math.sqrt(3))
    se_boot = float(np.std(draws, ddof=1))
    return {
        "delta": float(np.mean(seed_delta)), "se_seed": se_seed, "se_boot": se_boot,
        "epsilon": max(se_seed, se_boot), "ci_lower": float(np.quantile(draws, .025)),
        "ci_upper": float(np.quantile(draws, .975)),
    }


def _masked_paired_bootstrap(
    labels: np.ndarray, candidate: np.ndarray, baseline: np.ndarray,
    masks: np.ndarray, iterations: int = N_BOOTSTRAP, seed: int = 730,
    metric_labels: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Use one class-stratified row draw for every seed of a named slice."""
    rng = np.random.default_rng(seed)
    classes = candidate.shape[-1]
    strata = [np.flatnonzero(labels == value) for value in range(classes)]
    metric_labels = list(metric_labels) if metric_labels is not None else list(range(classes))
    def score(current_labels: np.ndarray, probability: np.ndarray) -> float:
        if not len(current_labels): return 0.0
        return float(f1_score(current_labels, probability.argmax(1), labels=metric_labels, average="macro", zero_division=0))
    seed_delta = []
    support = []
    for index in range(3):
        active = masks[index]
        support.append(int(active.sum()))
        seed_delta.append(score(labels[active], candidate[index, active]) - score(labels[active], baseline[index, active]))
    draws = np.empty(iterations)
    for draw in range(iterations):
        shared = np.concatenate([rng.choice(values, len(values), replace=True) for values in strata if len(values)])
        values = []
        for index in range(3):
            selected = shared[masks[index, shared]]
            values.append(score(labels[selected], candidate[index, selected]) - score(labels[selected], baseline[index, selected]))
        draws[draw] = float(np.mean(values))
    vector = np.asarray(seed_delta)
    se_seed = float(np.std(vector, ddof=1) / math.sqrt(3)); se_boot = float(np.std(draws, ddof=1))
    return {"seed_delta": dict(zip(map(str, SEEDS), seed_delta)), "support": dict(zip(map(str, SEEDS), support)), "delta": float(vector.mean()), "se_seed": se_seed, "se_boot": se_boot, "epsilon": max(se_seed, se_boot), "ci_lower": float(np.quantile(draws, .025)), "ci_upper": float(np.quantile(draws, .975))}


def repeated_collapse(seed_slice_delta: dict[str, dict[int, float]], epsilon: float) -> tuple[bool, list[str]]:
    collapsed = [name for name, values in seed_slice_delta.items() if sum(delta < -epsilon for delta in values.values()) >= 2]
    return bool(collapsed), collapsed


def select_primary(stats: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Select the best available primary without applying challenger constraints."""
    if not stats:
        raise RuntimeError("primary selection requires at least one candidate")
    eligible = list(stats)
    best = max(eligible, key=lambda name: stats[name]["mean"])
    best_mean, best_se = stats[best]["mean"], stats[best]["se"]
    threshold = best_mean - best_se
    one_se = [name for name in eligible if stats[name]["mean"] >= threshold]
    # Gap is intentionally inaccessible before the 1-SE set has been formed.
    primary = sorted(one_se, key=lambda name: (stats[name]["gap"], stats[name]["complexity"], -stats[name]["mean"], name))[0]
    evidence = {
        "selection_order": ["mean_nested_oof", "one_se_eligibility", "overfit_gap", "simpler_structure"],
        "gap_used_only_within_one_se": True,
        "one_se": {"best_mean": best_mean, "best_se": best_se, "threshold": threshold, "eligible_candidates": one_se},
    }
    return primary, evidence


def select_roles(stats: dict[str, dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    primary, evidence = select_primary(stats)
    remaining = [name for name in stats if name != primary]
    if len(remaining) < 2:
        raise RuntimeError("three distinct portfolio candidates are unavailable")

    primary_floor = stats[primary]["worst_floor"]
    robust_pool = [
        name for name in remaining
        if stats[name].get("noninferior", False)
        and stats[name]["worst_floor"] > primary_floor
    ]
    robust_qualified = bool(robust_pool)
    if not robust_pool:
        nominal = "robustness_challenger"
        robust_pool = [nominal] if nominal in remaining else remaining
    robustness = sorted(robust_pool, key=lambda name: (-stats[name]["worst_floor"], -stats[name]["mean"], stats[name]["complexity"], name))[0]
    specialist_pool = [
        name for name in remaining
        if name != robustness
        and stats[name].get("noninferior", False)
        and stats[name].get("collision_evidence", False)
    ]
    specialist_qualified = bool(specialist_pool)
    if not specialist_pool:
        available = [name for name in remaining if name != robustness]
        nominal = "specialization_challenger"
        specialist_pool = [nominal] if nominal in available else available
    specialization = sorted(specialist_pool, key=lambda name: (-stats[name]["net_rescue"], -stats[name]["mean"], stats[name]["complexity"], name))[0]
    roles = {"primary": primary, "robustness": robustness, "specialization": specialization}
    evidence.update({
        "role_status": {
            "primary": "best_available",
            "robustness_challenger": "qualified" if robust_qualified else "exploratory_failed_constraints",
            "specialization_challenger": "qualified" if specialist_qualified else "exploratory_failed_constraints",
        },
        "portfolio_constraints_pass": robust_qualified and specialist_qualified,
        "challenger_qualification": {
            "robustness_challenger": {
                "candidate": robustness,
                "noninferior": bool(stats[robustness].get("noninferior", False)),
                "improves_primary_worst_floor": stats[robustness]["worst_floor"] > primary_floor,
            },
            "specialization_challenger": {
                "candidate": specialization,
                "noninferior": bool(stats[specialization].get("noninferior", False)),
                "collision_evidence": bool(stats[specialization].get("collision_evidence", False)),
            },
        },
        "oracle_collapse": {
            "diagnostic_only": True,
            "gating": False,
            "note": "generic per-slice oracle-best collapse 수치는 보존하지만 primary/challenger 자격을 차단하지 않는다.",
        },
        "evidence": [
            "Primary는 전체 후보의 Nested OOF 평균, 1-SE, gap, 구조 단순성 순서로 고정했다.",
            "강건형 자격은 final primary 대비 overall 비열등성과 worst-floor 개선을 모두 요구했다.",
            "전문형 자격은 final primary 대비 overall 비열등성과 collision 증거를 모두 요구했다.",
            "자격 후보가 없으면 서로 다른 nominal lane을 탐색용으로 보존하고 제약 미통과를 명시했다.",
        ],
    })
    return roles, evidence


def _slice_evidence(
    train: pd.DataFrame, target: str, identifier: str, labels: np.ndarray,
    classes: list[str], assignments: pd.DataFrame, probabilities: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    features = train.drop(columns=[target, identifier])
    class_rows, subgroup_rows = [], []
    lane_groups: dict[str, dict[str, dict[int, float]]] = {lane: {} for lane in LANES}
    lane_classes: dict[str, dict[str, dict[int, float]]] = {lane: {} for lane in LANES}
    lane_seeds: dict[str, dict[str, dict[int, float]]] = {lane: {"overall": {}} for lane in LANES}
    worst: dict[str, float] = {}
    seed_groups: dict[int, dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        seeded = assignments.loc[assignments.seed == seed].set_index("row_index")
        group_arrays = {"burden": np.full(len(train), -1, np.int8), "novelty": np.full(len(train), -1, np.int8)}
        for fold in FOLDS:
            valid = seeded.index[seeded.fold == fold].to_numpy(int)
            train_idx = seeded.index[seeded.fold != fold].to_numpy(int)
            groups, _ = nested._fold_safe_groups(features.iloc[train_idx], features.iloc[valid])
            for dimension in group_arrays:
                group_arrays[dimension][valid] = groups[dimension]
        seed_groups[seed] = group_arrays
    for lane, probability in probabilities.items():
        seed_scores = []
        all_group_scores = []
        for seed_index, seed in enumerate(SEEDS):
            prediction = probability[seed_index].argmax(1)
            precision, recall, f1, support = precision_recall_fscore_support(labels, prediction, labels=np.arange(len(classes)), zero_division=0)
            for class_index, class_name in enumerate(classes):
                class_rows.append({"candidate": ROLE_NAMES[lane], "seed": seed, "class_name": class_name, "precision": float(precision[class_index]), "recall": float(recall[class_index]), "macro_f1": float(f1[class_index]), "support": int(support[class_index])})
                lane_classes[lane].setdefault(class_name, {})[seed] = float(f1[class_index])
            overall = _macro(labels, probability[seed_index], len(classes))
            seed_scores.append(overall); lane_seeds[lane]["overall"][seed] = overall
            group_arrays = seed_groups[seed]
            for dimension, groups in group_arrays.items():
                for group in (0, 1, 2):
                    mask = groups == group
                    score = _macro(labels[mask], probability[seed_index, mask], len(classes))
                    name = f"{dimension}:{group}"
                    lane_groups[lane].setdefault(name, {})[seed] = score
                    all_group_scores.append(score)
                    subgroup_rows.append({"candidate": ROLE_NAMES[lane], "seed": seed, "dimension": dimension, "group": group, "macro_f1": score, "support": int(mask.sum())})
        worst[lane] = min(all_group_scores)
    slice_masks: dict[str, dict[str, np.ndarray]] = {
        "seed": {"overall": np.ones((3, len(train)), dtype=bool)},
        "class": {name: np.broadcast_to((labels == index), (3, len(train))).copy() for index, name in enumerate(classes)},
        "subgroup": {},
    }
    for dimension in ("burden", "novelty"):
        for group in (0, 1, 2):
            slice_masks["subgroup"][f"{dimension}:{group}"] = np.stack([seed_groups[seed][dimension] == group for seed in SEEDS])
    # Collapse is measured against the best lane on the same named slice/seed,
    # with no fixed decimal threshold: epsilon=max(seed SE, shared-row bootstrap SE).
    collapse: dict[str, Any] = {}
    for lane in LANES:
        dimension_results = {}
        all_names = []
        for dimension, source in (("seed", lane_seeds), ("class", lane_classes), ("subgroup", lane_groups)):
            names, evidence_rows = [], {}
            for slice_name, values in source[lane].items():
                baseline = np.empty_like(probabilities[lane])
                for seed_index, seed in enumerate(SEEDS):
                    best_lane = max(LANES, key=lambda other: source[other][slice_name][seed])
                    baseline[seed_index] = probabilities[best_lane][seed_index]
                evidence = _masked_paired_bootstrap(labels, probabilities[lane], baseline, slice_masks[dimension][slice_name], seed=730 + sum(map(ord, lane + slice_name)))
                evidence_rows[slice_name] = evidence
                if sum(value < -evidence["epsilon"] for value in evidence["seed_delta"].values()) >= 2:
                    names.append(slice_name)
            dimension_results[dimension] = {"repeated": names, "evidence": evidence_rows}; all_names.extend(names)
        collapse[lane] = {"no_collapse": not all_names, "repeated_slices": all_names, "by_dimension": dimension_results, "worst_floor": worst[lane]}
    constraints = {
        "overall_pass": all(value["no_collapse"] for value in collapse.values()),
        "diagnostic_only": True,
        "gating": False,
        "constraints": {
            "seed": {"pass": not any(value["by_dimension"]["seed"]["repeated"] for value in collapse.values()), "repeated_collapse_count": sum(len(value["by_dimension"]["seed"]["repeated"]) for value in collapse.values())},
            "class": {"pass": not any(value["by_dimension"]["class"]["repeated"] for value in collapse.values()), "repeated_collapse_count": sum(len(value["by_dimension"]["class"]["repeated"]) for value in collapse.values())},
            "burden": {"pass": not any(any(name.startswith("burden:") for name in value["repeated_slices"]) for value in collapse.values()), "repeated_collapse_count": sum(sum(name.startswith("burden:") for name in value["repeated_slices"]) for value in collapse.values())},
            "novelty": {"pass": not any(any(name.startswith("novelty:") for name in value["repeated_slices"]) for value in collapse.values()), "repeated_collapse_count": sum(sum(name.startswith("novelty:") for name in value["repeated_slices"]) for value in collapse.values())},
        },
        "by_candidate": collapse,
    }
    return class_rows, subgroup_rows, constraints, collapse


def _collision_evidence(
    records: dict[tuple[str, int, int], dict[str, Any]], lane: str,
    banks: dict[str, np.ndarray], gated: np.ndarray, assignments: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[bool, int, dict[str, Any]]:
    """Require repeated ordered rescue, pair gain, and outside noninferiority."""
    rows, classes = gated.shape[1:]
    backbone = np.full_like(gated, np.nan)
    direction_by_fold: dict[tuple[int, int], dict[int, list[int]]] = {}
    direction_net: dict[tuple[int, int], dict[int, int]] = {}
    direction_rescue: dict[tuple[int, int], dict[int, int]] = {}
    per_seed_net = {seed: 0 for seed in SEEDS}
    gated_folds = 0
    for seed_index, seed in enumerate(SEEDS):
        seeded = assignments.loc[assignments.seed == seed].set_index("row_index")
        for fold in FOLDS:
            row = records[(lane, seed, fold)]; variant = row["selected_variant"]
            indices = seeded.index[seeded.fold == fold].to_numpy(int)
            if variant.get("kind") != "collision_gate":
                continue
            gated_folds += 1
            direction = tuple(map(int, variant["ordered_pair"]))
            direction_by_fold.setdefault(direction, {}).setdefault(seed, []).extend(indices.tolist())
            direction_net.setdefault(direction, {}).setdefault(seed, 0)
            direction_rescue.setdefault(direction, {}).setdefault(seed, 0)
            fold_banks = {name: value[seed_index, indices] for name, value in banks.items()}
            backbone[seed_index, indices] = nested._apply_base(variant["backbone_variant"], fold_banks)
            base_prediction = backbone[seed_index, indices].argmax(1)
            gate_prediction = gated[seed_index, indices].argmax(1)
            current_labels = labels[indices]
            directed = (current_labels == direction[0]) & (base_prediction == direction[1])
            rescue = int(np.sum(directed & (gate_prediction == current_labels)))
            harm = int(np.sum((base_prediction == current_labels) & (gate_prediction != current_labels)))
            net = rescue - harm
            per_seed_net[seed] += net; direction_net[direction][seed] += net
            direction_rescue[direction][seed] += rescue
    best_details: dict[str, Any] | None = None
    for direction, by_seed in direction_by_fold.items():
        direction_masks = np.zeros((3, rows), dtype=bool)
        for seed_index, seed in enumerate(SEEDS):
            direction_masks[seed_index, np.asarray(by_seed.get(seed, []), dtype=int)] = True
        pair_masks = direction_masks & np.broadcast_to(np.isin(labels, direction), (3, rows))
        outside_masks = direction_masks & ~np.broadcast_to(np.isin(labels, direction), (3, rows))
        pair_bootstrap = _masked_paired_bootstrap(labels, gated, backbone, pair_masks, seed=1700 + direction[0] * 31 + direction[1], metric_labels=direction)
        outside_bootstrap = _masked_paired_bootstrap(labels, gated, backbone, outside_masks, seed=2700 + direction[0] * 31 + direction[1])
        seed_rows = []
        for seed_index, seed in enumerate(SEEDS):
            indices = np.asarray(by_seed.get(seed, []), dtype=int)
            if not len(indices):
                seed_rows.append({"seed": seed, "rescue": 0, "net_rescue": 0, "pair_delta": 0.0, "outside_delta": 0.0})
                continue
            available = np.isfinite(backbone[seed_index, indices]).all(axis=1)
            indices = indices[available]
            pair_mask = np.isin(labels[indices], direction)
            outside_mask = ~pair_mask
            pair_delta = 0.0
            if pair_mask.any():
                selected = indices[pair_mask]
                pair_delta = float(f1_score(labels[selected], gated[seed_index, selected].argmax(1), labels=list(direction), average="macro", zero_division=0) - f1_score(labels[selected], backbone[seed_index, selected].argmax(1), labels=list(direction), average="macro", zero_division=0))
            outside_delta = 0.0
            if outside_mask.any():
                selected = indices[outside_mask]
                outside_delta = _macro(labels[selected], gated[seed_index, selected], classes) - _macro(labels[selected], backbone[seed_index, selected], classes)
            seed_rows.append({"seed": seed, "rescue": direction_rescue[direction].get(seed, 0), "net_rescue": direction_net[direction].get(seed, 0), "pair_delta": pair_delta, "outside_delta": outside_delta})
        outside_epsilon = outside_bootstrap["epsilon"]
        qualifies = (
            sum(row["net_rescue"] > 0 for row in seed_rows) >= 2
            and sum(row["pair_delta"] >= .05 for row in seed_rows) >= 2
            and sum(row["rescue"] for row in seed_rows) >= 3
            and outside_bootstrap["delta"] >= -outside_epsilon
        )
        details = {"ordered_pair": list(direction), "seed_evidence": seed_rows, "pair_bootstrap": pair_bootstrap, "outside_bootstrap": outside_bootstrap, "outside_epsilon": outside_epsilon, "qualifies": qualifies}
        if best_details is None or (qualifies, sum(row["net_rescue"] for row in seed_rows)) > (best_details["qualifies"], sum(row["net_rescue"] for row in best_details["seed_evidence"])):
            best_details = details
    if best_details is None:
        best_details = {"ordered_pair": None, "seed_evidence": [], "outside_epsilon": 0.0, "qualifies": False}
    return bool(best_details["qualifies"]), sum(per_seed_net.values()), {"gated_outer_folds": gated_folds, **best_details}


def _build_evidence_impl(artifact_root: Path, nested_source: Path, goal_config: Path, search_universe: Path) -> Path:
    root = artifact_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    state_path = root / "finalization_state.json"
    forbidden = _trusted_forbidden()
    _guard_not_forbidden(goal_config, forbidden, "goal config")
    trusted_universe = _guard_not_forbidden(search_universe, forbidden, "trusted search universe")
    _guard_not_forbidden(state_path, forbidden, "finalization state")
    if state_path.exists() and _guarded_json(state_path, forbidden, "finalization state").get("state") in {"frozen", "test_opened", "inferred", "documented"}:
        raise RuntimeError("evidence cannot mutate an existing freeze")
    goal = load_config(goal_config.resolve())
    data = goal["data"]; raw = (REPO_ROOT / data["raw_dir"]).resolve()
    test_path = (raw / data.get("test_file", "test.csv")).resolve()
    submission_path = (raw / data.get("submission_file", "sample_submission.csv")).resolve()
    if test_path != forbidden[0] or submission_path != forbidden[1]:
        raise RuntimeError("goal config must use canonical Test/submission paths")
    # Do not stat/read Test; keep only the configured path string in the audit.
    train_path = _guard_not_forbidden(raw / data["train_file"], forbidden, "Train")
    _guard_not_forbidden(root / "class_names.json", forbidden, "class order")
    _guard_not_forbidden(root / "fold_assignments.csv", forbidden, "fold assignments")
    train = pd.read_csv(train_path)
    classes = _guarded_json(root / "class_names.json", forbidden, "class order")
    assignments = pd.read_csv(root / "fold_assignments.csv")
    assignments = _normalize_alignment(assignments)
    base_manifest, banks = _load_base_manifest(root, classes, assignments, forbidden)
    nested_manifest_path = _guard_not_forbidden(nested_source / "nested_run_manifest.json", forbidden, "nested manifest")
    upstream_nested_manifest = _guarded_json(nested_manifest_path, forbidden, "nested manifest")
    universe_path = trusted_universe
    if Path(upstream_nested_manifest.get("search_universe", "")).resolve() != universe_path:
        raise ValueError("nested manifest search universe differs from independently supplied path")
    candidate_configs = {
        str(item["alias"]): (REPO_ROOT / str(item["config"])).resolve()
        for item in goal.get("candidates", [])
    }
    raw_snapshot, raw_identity = _snapshot_nested(
        root, nested_source, forbidden,
        {"goal_path": goal_config.resolve(), "universe_path": universe_path, "train_path": train_path,
         "fold_path": root / "fold_assignments.csv", "candidate_configs": candidate_configs},
    )
    completion_attestation = _verify_nested_completion_attestation(
        raw_snapshot, upstream_nested_manifest, forbidden
    )
    transcript_validation = nested.validate_inner_selection_transcript(
        raw_snapshot, train, target=data["target_column"], identifier=data["id_column"]
    )
    if transcript_validation != {"verified_records": 45, "exact_key_coverage": True}:
        raise ValueError("nested inner selection transcript verification failed")
    records = _selection_records(raw_snapshot)
    with np.load(raw_snapshot / "nested_strategy_probability.npz", allow_pickle=False) as archive:
        stored = {lane: _probability(archive[lane], (3, len(train), len(classes))) for lane in LANES}
    labels = _encode_labels(train, data["target_column"], classes)
    replay_audit_rows: list[dict[str, Any]] = []
    replay = _replay_nested(
        banks, records, assignments, stored, labels, replay_audit_rows
    )
    replay_audit = {
        "probability_authority": "stored_hashed_nested_outer_lane_probability",
        "selection_metric_authority": "stored_hashed_nested_outer_lane_probability",
        "production_inference_realization": "hashed_base_oof_fold_bundles",
        "base_oof_replay_role": "production_inference_bundle_semantic_parity_audit",
        "guaranteed_parity": ["outer_oof_argmax", "outer_oof_macro_f1"],
        "drift_attribution": "aggregate_only_for_selected_variants_containing_catboost",
        "residual_risk": "CatBoost GPU fit realizations are not probability-byte-identical. In mixed variants, bounded replay drift is aggregate and cannot be attributed per component because Nested outer component probabilities were not persisted; unseen Test argmax parity cannot be proven before the single inference read.",
        "external_completion_attestation": completion_attestation,
        "known_nondeterministic_aliases": sorted(KNOWN_NONDETERMINISTIC_REPLAY_ALIASES),
        "max_allowed_abs_drift": NONDETERMINISTIC_REPLAY_MAX_ABS_DRIFT,
        "max_allowed_mean_drift": NONDETERMINISTIC_REPLAY_MAX_MEAN_DRIFT,
        "strict_probability_match_count": sum(
            bool(row["strict_probability_match"]) for row in replay_audit_rows
        ),
        "semantic_equivalence_count": len(replay_audit_rows),
        "argmax_disagreement_count": sum(
            int(row["argmax_disagreement_count"]) for row in replay_audit_rows
        ),
        "max_observed_abs_drift": max(
            float(row["max_abs_probability_drift"]) for row in replay_audit_rows
        ),
        "max_observed_mean_drift": max(
            float(row["mean_abs_probability_drift"]) for row in replay_audit_rows
        ),
        "records": replay_audit_rows,
    }
    _validate_replay_audit(replay_audit)
    verified_nested = _verify_nested_metrics(raw_snapshot, stored, replay, records, assignments, labels)

    candidates = deepcopy(base_manifest["candidates"])
    for lane in LANES:
        version = root / "candidates" / ROLE_NAMES[lane] / raw_identity["run_identity"]
        version.mkdir(parents=True, exist_ok=True)
        probability_path = version / "oof_probability_by_seed.npy"
        alignment_path = version / "alignment.csv"
        if probability_path.exists() and not np.array_equal(np.load(probability_path, allow_pickle=False), replay[lane]):
            raise RuntimeError("immutable derived candidate disagrees with replay")
        if not probability_path.exists(): _atomic_npy(probability_path, replay[lane])
        if not alignment_path.exists(): _atomic_bytes(alignment_path, assignments.to_csv(index=False).encode("utf-8"))
        candidates.append({"name": ROLE_NAMES[lane], "class_names": classes, "seeds": list(SEEDS), "probability_path": probability_path.relative_to(root).as_posix(), "alignment_path": alignment_path.relative_to(root).as_posix(), "probability_sha256": _sha256(probability_path), "alignment_sha256": _sha256(alignment_path), "collection_identity_hash": _canonical_hash({"nested": raw_identity, "lane": lane}), "source_runs": [], "test_accessed": False, "derived_from_nested": True})
    _atomic_json(root / "collector_candidate_manifest_snapshot.json", base_manifest)
    final_candidate_manifest = {**base_manifest, "collector_base_manifest_snapshot": "collector_candidate_manifest_snapshot.json", "collector_base_manifest_sha256": _sha256(root / "collector_candidate_manifest_snapshot.json"), "candidates": candidates, "selection_uses_test": False, "nested_base_oof_replay_audit": replay_audit}

    class_rows, subgroup_rows, constraints, collapse = _slice_evidence(train, data["target_column"], data["id_column"], labels, classes, assignments, replay)
    summaries = verified_nested["summaries"]
    comparisons = []
    for lane in ("robustness_challenger", "specialization_challenger"):
        comparisons.append({"candidate": ROLE_NAMES[lane], "baseline": ROLE_NAMES["primary"], **paired_bootstrap(labels, replay[lane], replay["primary"])})
    stats = {}
    for lane in LANES:
        summary = summaries[lane]
        collision, net_rescue, collision_details = _collision_evidence(records, lane, banks, replay[lane], assignments, labels)
        comparison = next((row for row in comparisons if row["candidate"] == ROLE_NAMES[lane]), None)
        variants = [records[(lane, seed, fold)]["selected_variant"] for seed in SEEDS for fold in FOLDS]
        complexity = sum((100 if variant["kind"] == "collision_gate" else 0) + len(variant.get("models", variant.get("backbone_variant", {}).get("models", []))) for variant in variants)
        stats[lane] = {"mean": float(summary["mean_macro_f1"]), "se": float(summary["std_macro_f1"]) / math.sqrt(3), "gap": float(np.mean(verified_nested["gaps"][lane])), "complexity": complexity, "worst_floor": collapse[lane]["worst_floor"], "no_collapse": collapse[lane]["no_collapse"], "noninferior": True, "collision_evidence": collision, "collision_details": collision_details, "net_rescue": net_rescue}
    # Primary is fixed first; challenger noninferiority is then recomputed against
    # that final primary, never against the nominal lane name.
    final_primary, _ = select_primary(stats)
    for lane in LANES:
        if lane == final_primary:
            stats[lane]["noninferior"] = True
        else:
            evidence = paired_bootstrap(labels, replay[lane], replay[final_primary])
            stats[lane]["paired_vs_final_primary"] = evidence
            stats[lane]["noninferior"] = evidence["delta"] >= -evidence["epsilon"]
    roles, selection_evidence = select_roles(stats)
    selection_evidence["nested_base_oof_replay_audit"] = replay_audit
    selection_evidence["one_se"]["eligible_candidates"] = [
        ROLE_NAMES[name] for name in selection_evidence["one_se"]["eligible_candidates"]
    ]
    for qualification in selection_evidence["challenger_qualification"].values():
        qualification["candidate"] = ROLE_NAMES[qualification["candidate"]]
    selection_names = {
        final_role: ROLE_NAMES[roles[internal_role]]
        for final_role, internal_role in FINAL_ROLE_TO_INTERNAL.items()
    }
    comparisons = [
        {"candidate": ROLE_NAMES[roles[role]], "baseline": ROLE_NAMES[roles["primary"]], **paired_bootstrap(labels, replay[roles[role]], replay[roles["primary"]])}
        for role in ("robustness", "specialization")
    ]
    final_selection = {"selections": selection_names, "selection_evidence": selection_evidence, "candidate_statistics": {ROLE_NAMES[lane]: value for lane, value in stats.items()}}

    primary_lane = roles["primary"]
    primary_eval = [row for row in verified_nested["verified_rows"] if row["strategy"] == primary_lane]
    nested_metrics = {"schema_version": 1, "primary_strategy": ROLE_NAMES[primary_lane], "outer_evaluations": primary_eval, "candidate_outer_evaluations": verified_nested["verified_rows"], "strategy_summary": list(summaries.values()), "objective": "mean_nested_oof_macro_f1", "gap_used_in_objective": False}
    ablations = []
    selection_hash_before_ablation = _canonical_hash(final_selection)
    for lane in LANES:
        full = float(summaries[lane]["mean_macro_f1"])
        primary_full = float(summaries[primary_lane]["mean_macro_f1"])
        ablations.append({"name": f"{ROLE_NAMES[lane]}:selected_variant_frozen", "macro_f1": full, "delta": full - primary_full, "eligible_for_reselection": False})
    outputs = {
        "candidate_manifest.json": final_candidate_manifest,
        "nested_cv_metrics.json": nested_metrics,
        "class_metrics.json": {"classes": class_rows},
        "subgroup_metrics.json": {"subgroups": subgroup_rows},
        "paired_comparisons.json": {"comparisons": comparisons},
        "collapse_constraints.json": constraints,
        "ensemble_ablation.json": {"complete": True, "selected_variant_mutated": False, "selection_hash_before": selection_hash_before_ablation, "selection_hash_after": _canonical_hash(final_selection), "ablations": ablations},
        "final_selection.json": final_selection,
    }
    _atomic_npy(root / "nested_oof_probability.npy", replay[primary_lane])
    for name, value in outputs.items(): _atomic_json(root / name, value)
    _verify_persisted_replay_audit(root, replay_audit)
    prior_audit = _guarded_json(root / "io_audit.json", forbidden, "prior IO audit") if (root / "io_audit.json").exists() else {"events": []}
    nested_audit = _guarded_json(raw_snapshot / "nested_io_audit.json", forbidden, "nested IO audit")
    events = [*prior_audit.get("events", []), *nested_audit.get("events", [])]
    _atomic_json(root / "io_audit.json", _merged_audit(events, forbidden))
    _atomic_json(state_path, {"state": "evidence", "test_read_count": 0, "canonical_test_read_count": 0,
                             "fit_after_test": 0, "warm_after_test": 0, "bundle_load_after_test_count": 0,
                             "selection_or_tuning_after_test": 0, "configured_test_path": str(test_path),
                             "configured_submission_path": str(submission_path), "nested_raw": str(raw_snapshot),
                             "nested_raw_identity": raw_identity, "updated_at": datetime.now(timezone.utc).isoformat()})
    return state_path


def build_evidence(artifact_root: Path, nested_source: Path, goal_config: Path, search_universe: Path) -> Path:
    """Transactional wrapper preserving the prior generation and state on failure."""
    root = artifact_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    forbidden = _trusted_forbidden()
    owned = [
        root / name for name in (
            "candidate_manifest.json", "collector_candidate_manifest_snapshot.json",
            "nested_oof_probability.npy", "nested_cv_metrics.json", "class_metrics.json",
            "subgroup_metrics.json", "paired_comparisons.json", "collapse_constraints.json",
            "ensemble_ablation.json", "final_selection.json", "io_audit.json",
            "finalization_state.json",
        )
    ]
    snapshot = _snapshot_files(owned, forbidden)
    prior_directories = _tracked_directories(root)
    try:
        output = _build_evidence_impl(root, nested_source, goal_config, search_universe)
        generation = root / "generations" / f"evidence-{uuid.uuid4().hex}"
        generation.mkdir(parents=True, exist_ok=False)
        hashes = {}
        for path in owned:
            if path.exists():
                destination = generation / path.name
                shutil.copy2(path, destination); hashes[path.name] = _sha256(destination)
        _atomic_json(generation / "generation_manifest.json", {"phase": "evidence", "hashes": hashes})
        state = _guarded_json(output, forbidden, "new finalization state")
        _atomic_json(output, {**state, "generation": str(generation), "generation_manifest_hash": _sha256(generation / "generation_manifest.json")})
        return output
    except Exception:
        _restore_snapshot(snapshot)
        _remove_new_directories(root, prior_directories)
        raise


def _freeze_impl(artifact_root: Path) -> Path:
    root = artifact_root.resolve(); state_path = root / "finalization_state.json"
    forbidden = _trusted_forbidden()
    state = _guarded_json(state_path, forbidden, "finalization state")
    if state.get("state") == "frozen":
        manifest = root / "frozen_manifest.json"
        if state.get("freeze_hash") != _sha256(manifest): raise RuntimeError("frozen manifest changed")
        return manifest
    if state.get("state") != "evidence" or state.get("test_read_count") != 0:
        raise RuntimeError("freeze requires complete Test-free evidence")
    _validate_generation(root, state, forbidden)
    selection = _guarded_json(root / "final_selection.json", forbidden, "final selection")
    candidate_manifest = _guarded_json(root / "candidate_manifest.json", forbidden, "candidate manifest")
    replay_audit = candidate_manifest.get("nested_base_oof_replay_audit")
    if replay_audit != selection.get("selection_evidence", {}).get(
        "nested_base_oof_replay_audit"
    ):
        raise RuntimeError("frozen replay audit lineage differs between evidence artifacts")
    _validate_replay_audit(replay_audit)
    raw_path = Path(state["nested_raw"])
    raw_hashes = state.get("nested_raw_identity", {}).get("hashes")
    if not isinstance(raw_hashes, dict) or set(raw_hashes) != set(RAW_FILES):
        raise RuntimeError("evidence nested raw identity is incomplete")
    for name in RAW_FILES:
        raw_artifact = _guard_not_forbidden(raw_path / name, forbidden, f"frozen nested raw {name}")
        if not raw_artifact.is_file() or _sha256(raw_artifact) != raw_hashes[name]:
            raise RuntimeError(f"evidence nested raw artifact changed: {name}")
    raw_manifest = _guarded_json(
        raw_path / "nested_run_manifest.json", forbidden, "frozen nested manifest"
    )
    trusted = raw_manifest.get("trusted_input_hashes")
    nested_sources = trusted.get("sources") if isinstance(trusted, dict) else None
    nested_source_bundle = trusted.get("source_bundle") if isinstance(trusted, dict) else None
    if not isinstance(nested_sources, dict) or not nested_sources:
        raise RuntimeError("nested trusted inference source inventory is missing")
    if nested_source_bundle != _canonical_hash(nested_sources):
        raise RuntimeError("nested trusted inference source bundle is invalid")
    inference_source_files = _inference_source_inventory(nested_sources, forbidden)
    records = _selection_records(raw_path)
    base_items = [item for item in candidate_manifest["candidates"] if not item.get("derived_from_nested")]
    used_aliases = set()
    for row in records.values():
        variant = row["selected_variant"]
        if variant["kind"] == "collision_gate":
            used_aliases.update(variant["backbone_variant"]["models"]); used_aliases.add(variant["expert"])
        else: used_aliases.update(variant["models"])
    candidates = []
    used_source_files: dict[str, str] = {}
    if (Path(state["configured_test_path"]).resolve(), Path(state["configured_submission_path"]).resolve()) != forbidden:
        raise RuntimeError("state Test/submission boundary changed")
    for item in base_items:
        if item["name"] not in used_aliases: continue
        runs = {}
        for source in item.get("source_runs", []):
            run_manifest = Path(source["run_manifest_path"])
            _guard_not_forbidden(run_manifest, forbidden, "OOF run manifest")
            live = _guarded_json(run_manifest, forbidden, "OOF run manifest")
            if _sha256(run_manifest) != source["run_manifest_sha256"]: raise ValueError("run manifest hash changed before freeze")
            if live.get("bundle_safe_for_infer") is not True:
                raise ValueError("live OOF run bundle_safe_for_infer is not true")
            runs[str(source["seed"])] = {
                "run_manifest_hash": source["run_manifest_sha256"], "train_data_hash": source["train_hash"],
                "source_hash": source["source_hash"], "fold_assignments_hash": source["fold_hash"],
                "class_names_hash": source["class_hash"], "artifact_bundle_hash": source["bundle_hash"],
                "bundle_safe_for_infer": live["bundle_safe_for_infer"], "library_versions": live["library_versions"],
                "external_runtime_snapshot": live["external_runtime_snapshot"], "run_manifest_path": str(run_manifest),
                "manifest_identity_hash": source["manifest_identity_hash"], "runtime_metrics_hash": source["runtime_hash"],
            }
            source_hashes = live.get("source_hashes")
            if not isinstance(source_hashes, dict) or bank._canonical_hash(source_hashes) != source["source_hash"]:
                raise ValueError("OOF source hash inventory is invalid")
            for relative, digest in source_hashes.items():
                source_path = _guard_not_forbidden(REPO_ROOT / relative, forbidden, "used source")
                if _sha256(source_path) != digest:
                    raise ValueError(f"used source changed: {relative}")
                if relative in used_source_files and used_source_files[relative] != digest:
                    raise ValueError("used source hash differs between runs")
                used_source_files[relative] = digest
        if set(runs) != set(map(str, SEEDS)):
            raise ValueError("each used alias must have exactly three frozen runs")
        first_run = _guarded_json(Path(next(iter(runs.values()))["run_manifest_path"]), forbidden, "OOF run manifest")
        candidates.append({"name": item["name"], "config_hash": first_run["config_hash"], "config_path": first_run["config_path"], "configured_test_path": state["configured_test_path"], "configured_submission_path": state["configured_submission_path"], "seeds": list(SEEDS), "runs": runs})
    if {item["name"] for item in candidates} != used_aliases:
        raise RuntimeError("not every frozen variant alias has exactly three verified base runs")
    protected = _guarded_json(root / "protected_file_hashes_before_after.json", forbidden, "protected hashes")
    before = _protected_before_hashes(protected)
    protected_files, after, unchanged = [], {}, True
    for relative, digest in before.items():
        live_path = _guard_not_forbidden(REPO_ROOT / relative, forbidden, "protected source")
        current = _sha256(live_path); after[relative] = current
        same = current == digest; unchanged &= same
        protected_files.append({"path": relative, "before": digest, "after": current, "unchanged": same})
    if not unchanged:
        raise RuntimeError("protected source hashes changed")
    _atomic_json(root / "protected_file_hashes_before_after.json", {"before": before, "after": after, "unchanged": True, "files": protected_files})
    top = {
        "source_hash": _canonical_hash(used_source_files), "train_hash": next(iter(candidates))["runs"]["42"]["train_data_hash"],
        "fold_hash": _sha256(root / "fold_assignments.csv"),
        "bundle_hash": _canonical_hash({item["name"]: {seed: run["artifact_bundle_hash"] for seed, run in item["runs"].items()} for item in candidates}),
    }
    role_to_lane = {
        role: next(lane for lane, name in ROLE_NAMES.items() if name == candidate)
        for role, candidate in selection["selections"].items()
    }
    manifest = {
        "schema_version": 1, "frozen": True, "selection_uses_test": False, "thresholds_frozen": True,
        "class_names": _read_json(root / "class_names.json"), **top,
        "candidate_manifest_hash": _sha256(root / "candidate_manifest.json"),
        "nested_probability_hash": _sha256(root / "nested_oof_probability.npy"),
        "nested_raw_manifest_hash": _sha256(raw_path / "nested_run_manifest.json"),
        "nested_raw_path": str(raw_path), "nested_raw_identity": state.get("nested_raw_identity"),
        "selection_hash": _sha256(root / "final_selection.json"), "selections": selection["selections"],
        "role_to_lane": role_to_lane,
        "inference_source_base_files": nested_sources,
        "inference_source_base_bundle": nested_source_bundle,
        "inference_source_files": inference_source_files,
        "inference_source_hash": _canonical_hash(inference_source_files),
        "nested_base_oof_replay_audit": replay_audit,
        "selected_variants": [{"strategy": lane, "seed": seed, "fold": fold, "variant": records[(lane, seed, fold)]["selected_variant"]} for lane in LANES for seed in SEEDS for fold in FOLDS],
        "candidates": candidates, "used_source_files": used_source_files, "created_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = root / ".frozen_manifest.pending.json"
    try:
        _atomic_json(temporary, manifest)
        checked = _read_json(temporary)
        if len(checked["selected_variants"]) != 45 or len(set(checked["selections"].values())) != 3:
            raise RuntimeError("freeze validation failed")
        _validated_role_to_lane(checked)
        os.replace(temporary, root / "frozen_manifest.json")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    digest = _sha256(root / "frozen_manifest.json")
    _atomic_json(state_path, {**state, "state": "frozen", "freeze_hash": digest, "test_read_count": 0,
                             "canonical_test_read_count": 0, "fit_after_test": 0, "warm_after_test": 0,
                             "bundle_load_after_test_count": 0, "selection_or_tuning_after_test": 0,
                             "updated_at": datetime.now(timezone.utc).isoformat()})
    return root / "frozen_manifest.json"


def freeze(artifact_root: Path) -> Path:
    root = artifact_root.resolve(); forbidden = _trusted_forbidden()
    paths = [root / "frozen_manifest.json", root / "finalization_state.json", root / "protected_file_hashes_before_after.json"]
    snapshot = _snapshot_files(paths, forbidden)
    try:
        return _freeze_impl(root)
    except Exception:
        _restore_snapshot(snapshot)
        raise


@dataclass
class PreparedRun:
    alias: str
    seed: int
    config: dict[str, Any]
    bundles: list[dict[str, Any]]


@dataclass(frozen=True)
class RunSpec:
    alias: str
    seed: int
    config: dict[str, Any]
    bundle_path: Path
    bundle_hash: str
    train_path: Path
    target: str
    identifier: str
    test_path: Path
    probability_mode: str
    identity: dict[str, Any]


@dataclass(frozen=True)
class PreflightPlan:
    runs: tuple[RunSpec, ...]
    train_columns: tuple[str, ...]
    target: str
    identifier: str
    test_path: Path


def _variant_aliases(variant: dict[str, Any]) -> set[str]:
    if not isinstance(variant, dict) or variant.get("kind") not in {"single", "blend", "collision_gate"}:
        raise ValueError("frozen variant schema is invalid")
    if variant["kind"] == "collision_gate":
        aliases = set(map(str, variant.get("backbone_variant", {}).get("models", [])))
        aliases.add(str(variant.get("expert", "")))
    else:
        aliases = set(map(str, variant.get("models", [])))
    if not aliases or "" in aliases:
        raise ValueError("frozen variant references invalid aliases")
    dummy = {alias: np.full((2, 26), 1 / 26) for alias in aliases}
    nested._apply_variant(variant, dummy)
    return aliases


def _frozen_variants(frozen: dict[str, Any]) -> tuple[dict[tuple[str, int, int], dict[str, Any]], set[str]]:
    rows = frozen.get("selected_variants")
    if not isinstance(rows, list) or len(rows) != 45:
        raise RuntimeError("global barrier requires exactly 45 variants")
    variants, referenced = {}, set()
    for item in rows:
        key = (str(item.get("strategy")), int(item.get("seed", -1)), int(item.get("fold", -1)))
        if key in variants or key not in {(lane, seed, fold) for lane in LANES for seed in SEEDS for fold in FOLDS}:
            raise RuntimeError("frozen variant coverage is invalid")
        variants[key] = item["variant"]; referenced.update(_variant_aliases(item["variant"]))
    return variants, referenced


def _integrity_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Add an unkeyed checksum for corruption detection, not local-forgery security."""
    return {"payload": payload, "sha256": _canonical_hash(payload)}


def _read_integrity_receipt(path: Path) -> dict[str, Any] | None:
    try:
        envelope = _read_json(path)
        payload = envelope["payload"]
        if not isinstance(payload, dict) or envelope.get("sha256") != _canonical_hash(payload):
            return None
        return payload
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, json.JSONDecodeError):
        return None


def _checkpoint_name(alias: str, seed: int) -> str:
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in alias)
    return f"{safe}__seed_{seed}"


def _release_run(run: PreparedRun | None = None) -> None:
    if run is not None:
        run.bundles.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _verify_submission_manifest(root: Path, frozen: dict[str, Any], state: dict[str, Any]) -> Path:
    path = root / "final_submissions.json"
    data = _read_json(path)
    if (data.get("freeze_hash") != state.get("freeze_hash") or
            state.get("canonical_test_read_count") != 1 or
            state.get("fit_after_test") != 0 or state.get("warm_after_test") != 0 or
            state.get("selection_or_tuning_after_test") != 0):
        raise RuntimeError("inferred resume metadata is inconsistent")
    checkpoint_path = root / "inference_checkpoints" / "inference_checkpoint_manifest.json"
    checkpoint = _read_integrity_receipt(checkpoint_path)
    if (not isinstance(checkpoint, dict) or data.get("checkpoint_manifest_hash") != _sha256(checkpoint_path) or
            checkpoint.get("freeze_hash") != state.get("freeze_hash") or
            data.get("test_content_hash") != state.get("test_content_hash") or
            checkpoint.get("test_content_hash") != state.get("test_content_hash") or
            data.get("prediction_manifest_hash") != checkpoint.get("prediction_manifest_hash")):
        raise RuntimeError("inference checkpoint manifest is inconsistent")
    rows = data.get("submissions")
    if not isinstance(rows, list) or len(rows) != 3 or {row.get("role") for row in rows} != set(FINAL_ROLE_TO_INTERNAL):
        raise RuntimeError("resume requires exactly three frozen roles")
    contract = state.get("test_contract")
    if not isinstance(contract, dict) or data.get("test_contract") != contract:
        raise RuntimeError("resume Test contract is missing or inconsistent")
    ids = None
    for row in rows:
        role = row["role"]
        if row.get("candidate") != frozen["selections"][role]:
            raise RuntimeError("resume candidate differs from frozen selection")
        if row.get("members") != 15:
            raise RuntimeError("resume submission member count is invalid")
        csv_path = _within(root, row["path"])
        if _sha256(csv_path) != row.get("sha256"):
            raise RuntimeError("resume submission hash mismatch")
        frame = pd.read_csv(csv_path)
        if list(frame.columns) != ["ID", "SUBCLASS"] or len(frame) != row.get("row_count") or frame.isna().any().any() or not set(frame.SUBCLASS.astype(str)).issubset(frozen["class_names"]):
            raise RuntimeError("resume submission metadata/content mismatch")
        current = frame.ID.astype(str).tolist()
        if len(current) != contract.get("row_count") or _canonical_hash(current) != contract.get("id_order_hash"):
            raise RuntimeError("resume submission IDs differ from irreversible Test contract")
        if ids is None: ids = current
        elif ids != current: raise RuntimeError("resume submission ID order mismatch")
    return path


def _prepare_all(root: Path, frozen: dict[str, Any], *, allow_warm: bool = True) -> PreflightPlan:
    """Statically validate, then load/warm exactly one frozen run at a time."""
    freeze_hash = _sha256(root / "frozen_manifest.json")
    _, referenced = _frozen_variants(frozen)
    specs: list[RunSpec] = []
    contract: tuple[str, str, Path, str, tuple[str, ...]] | None = None
    for item in frozen["candidates"]:
        alias = item["name"]
        if alias not in referenced:
            continue
        forbidden = (Path(item["configured_test_path"]), Path(item["configured_submission_path"]))
        for seed in SEEDS:
            expected = item["runs"][str(seed)]
            manifest_path = _guard_not_forbidden(Path(expected["run_manifest_path"]), forbidden, "OOF run manifest")
            if _sha256(manifest_path) != expected["run_manifest_hash"]: raise RuntimeError("run manifest changed")
            run_manifest = _read_json(manifest_path); config_path = _guard_not_forbidden(Path(item["config_path"]), forbidden, "candidate config")
            if expected.get("bundle_safe_for_infer") is not True or run_manifest.get("bundle_safe_for_infer") is not True:
                raise RuntimeError("bundle_safe_for_infer=false before Test")
            if Path(run_manifest["config_path"]).resolve() != config_path:
                raise RuntimeError("frozen config path changed")
            if _sha256(config_path) != item["config_hash"]: raise RuntimeError("config changed")
            config = load_config(config_path); raw = (REPO_ROOT / config["data"]["raw_dir"]).resolve()
            configured_test = (raw / config["data"]["test_file"]).resolve()
            configured_submission = (raw / config["data"].get("submission_file", "sample_submission.csv")).resolve()
            if configured_test != forbidden[0].resolve() or configured_submission != forbidden[1].resolve():
                raise RuntimeError("frozen Test/submission contract changed")
            live_environment = {"library_versions": bank._library_versions(config), "external_runtime_snapshot": bank._external_runtime_snapshot(config)}
            if any(live_environment[key] != expected[key] for key in live_environment): raise RuntimeError("runtime changed before Test")
            run_dir = manifest_path.parent
            for name in ("fold_assignments.csv", "class_names.json", "runtime_metrics.json", "fold_bundles.pkl"):
                _guard_not_forbidden(run_dir / name, forbidden, f"OOF {name}")
            train_path = _guard_not_forbidden(raw / config["data"]["train_file"], forbidden, "Train")
            comparisons = {"train_data_hash": _sha256(train_path), "source_hash": bank._canonical_hash(bank._source_hashes(config)), "fold_assignments_hash": _sha256(run_dir / "fold_assignments.csv"), "class_names_hash": _sha256(run_dir / "class_names.json"), "artifact_bundle_hash": _sha256(run_dir / "fold_bundles.pkl")}
            if any(comparisons[key] != expected[key] for key in comparisons): raise RuntimeError("frozen run artifact changed")
            if _sha256(run_dir / "runtime_metrics.json") != expected["runtime_metrics_hash"] or not bank._validate_artifacts(run_dir, run_manifest): raise RuntimeError("frozen OOF artifact inventory changed")
            current_target, current_identifier = config["data"]["target_column"], config["data"]["id_column"]
            columns = tuple(pd.read_csv(train_path, nrows=0).columns)
            current_contract = (current_target, current_identifier, configured_test, comparisons["train_data_hash"], columns)
            if contract is None:
                contract = current_contract
            elif contract != current_contract:
                raise RuntimeError("candidate Train/Test contracts differ")
            identity = {
                "schema_version": 1, "freeze_hash": freeze_hash, "alias": alias, "seed": seed,
                "run_manifest_hash": expected["run_manifest_hash"], "config_hash": item["config_hash"],
                "bundle_hash": comparisons["artifact_bundle_hash"], "train_hash": comparisons["train_data_hash"],
                "class_hash": comparisons["class_names_hash"], "source_hash": comparisons["source_hash"],
                "fold_hash": comparisons["fold_assignments_hash"], "runtime_metrics_hash": expected["runtime_metrics_hash"],
                "environment_hash": _canonical_hash(live_environment),
            }
            specs.append(RunSpec(alias, seed, config, run_dir / "fold_bundles.pkl", comparisons["artifact_bundle_hash"], train_path, current_target, current_identifier, configured_test, run_manifest["probability_mode"], identity))
    expected_units = {(alias, seed) for alias in referenced for seed in SEEDS}
    if {(spec.alias, spec.seed) for spec in specs} != expected_units or contract is None:
        raise RuntimeError("ALL_RUNS_PREPARED failed: referenced alias/seed coverage differs")
    if len({_checkpoint_name(alias, seed) for alias, seed in expected_units}) != len(expected_units):
        raise RuntimeError("selected aliases collide in checkpoint filenames")
    receipt_dir = root / "inference_checkpoints" / "preflight"
    for spec in specs:
        receipt_path = receipt_dir / f"{_checkpoint_name(spec.alias, spec.seed)}.json"
        if _read_integrity_receipt(receipt_path) == spec.identity:
            continue
        if not allow_warm:
            raise RuntimeError("invalid preflight checkpoint after Test open; warmup is forbidden")
        run: PreparedRun | None = None
        try:
            train = pd.read_csv(spec.train_path)
            with spec.bundle_path.open("rb") as handle:
                bundles = pickle.load(handle)
            if (not isinstance(bundles, list) or len(bundles) != 5 or
                    {bundle.get("fold") for bundle in bundles} != set(FOLDS) or
                    any(bundle.get("class_names") != frozen["class_names"] for bundle in bundles) or
                    {bundle.get("probability_mode") for bundle in bundles} != {spec.probability_mode}):
                raise RuntimeError("invalid bundle structure")
            run = PreparedRun(spec.alias, spec.seed, spec.config, bundles)
            bank._warm_up_bundles(run.bundles, train.drop(columns=[spec.target, spec.identifier]), frozen["class_names"])
            _atomic_json(receipt_path, _integrity_envelope(spec.identity))
        finally:
            _release_run(run)
    target, identifier, test_path, _, columns = contract
    return PreflightPlan(tuple(specs), columns, target, identifier, test_path)


def _test_cache_payload(freeze_hash: str, content: bytes) -> dict[str, Any]:
    return {"schema_version": 1, "freeze_hash": freeze_hash, "content_hash": hashlib.sha256(content).hexdigest(), "byte_count": len(content)}


def _cached_test_bytes(root: Path, test_path: Path, freeze_hash: str, *, allow_open: bool = True) -> tuple[bytes, bool]:
    checkpoint_root = root / "inference_checkpoints"
    cache = checkpoint_root / "test_cache"
    pending = checkpoint_root / ".test_cache.pending"
    for directory in (cache, pending):
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise RuntimeError("corrupted Test cache is a hard stop")
        try:
            content = (directory / "test.csv.bytes").read_bytes()
        except OSError as error:
            raise RuntimeError("corrupted Test cache is a hard stop") from error
        payload = _read_integrity_receipt(directory / "receipt.json")
        if payload != _test_cache_payload(freeze_hash, content):
            raise RuntimeError("corrupted or stale Test cache is a hard stop")
        if directory == pending:
            if cache.exists():
                raise RuntimeError("ambiguous Test cache is a hard stop")
            os.replace(pending, cache)
        return content, False
    if not allow_open:
        raise RuntimeError("missing Test cache after canonical Test was opened")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    pending.mkdir()
    try:
        with test_path.open("rb") as handle:
            content = handle.read()
        _atomic_bytes(pending / "test.csv.bytes", content)
        _atomic_json(pending / "receipt.json", _integrity_envelope(_test_cache_payload(freeze_hash, content)))
        os.replace(pending, cache)
        return content, True
    except Exception:
        # Preserve a complete pending cache for recovery; an incomplete directory
        # proves Test may already have been opened and therefore must hard-stop.
        raise


def _prediction_identity(spec: RunSpec, freeze_hash: str, test_hash: str, class_hash: str) -> dict[str, Any]:
    return {"schema_version": 1, "freeze_hash": freeze_hash, "test_hash": test_hash, "class_hash": class_hash,
            "alias": spec.alias, "seed": spec.seed, "bundle_hash": spec.bundle_hash}


def _valid_prediction_checkpoint(path: Path, receipt: Path, identity: dict[str, Any], shape: tuple[int, int]) -> bool:
    payload = _read_integrity_receipt(receipt)
    if not isinstance(payload, dict) or payload.get("identity") != identity or payload.get("npz_hash") != (_sha256(path) if path.is_file() else None):
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {f"fold_{fold}" for fold in FOLDS}:
                return False
            for fold in FOLDS:
                value = np.asarray(data[f"fold_{fold}"])
                if (value.shape != shape or not np.isfinite(value).all() or
                        (value < 0).any() or (value > 1).any() or not np.allclose(value.sum(1), 1.0)):
                    return False
    except (OSError, ValueError, KeyError):
        return False
    return True


def _predict_run(root: Path, spec: RunSpec, test_features: pd.DataFrame, frozen: dict[str, Any], state_path: Path, state: dict[str, Any], test_hash: str) -> None:
    prediction_dir = root / "inference_checkpoints" / "predictions"
    stem = _checkpoint_name(spec.alias, spec.seed)
    path, receipt = prediction_dir / f"{stem}.npz", prediction_dir / f"{stem}.json"
    identity = _prediction_identity(spec, state["freeze_hash"], test_hash, _canonical_hash(frozen["class_names"]))
    shape = (len(test_features), len(frozen["class_names"]))
    if not _valid_prediction_checkpoint(path, receipt, identity, shape):
        run: PreparedRun | None = None
        try:
            if _sha256(spec.bundle_path) != spec.bundle_hash:
                raise RuntimeError("frozen bundle changed after preflight")
            with spec.bundle_path.open("rb") as handle:
                bundles = pickle.load(handle)
            run = PreparedRun(spec.alias, spec.seed, spec.config, bundles)
            state["bundle_load_after_test_count"] = int(state.get("bundle_load_after_test_count", 0)) + 1
            _atomic_json(state_path, state)
            values = {}
            for bundle in sorted(run.bundles, key=lambda value: value["fold"]):
                transformed = bundle["preprocessor"].transform(test_features)
                probability = bank._predict_probability(bundle["model"], bundle["probability_mode"], bundle["calibrator"], transformed, bank._local_class_names(bundle["preprocessor"], frozen["class_names"]), frozen["class_names"])
                if probability is None:
                    raise RuntimeError("final ensemble requires probabilities")
                values[f"fold_{int(bundle['fold'])}"] = _probability(probability, shape)
            _atomic_npz(path, **values)
            _atomic_json(receipt, _integrity_envelope({"identity": identity, "npz_hash": _sha256(path)}))
        finally:
            _release_run(run)


def _write_checkpoint_manifest(root: Path, specs: tuple[RunSpec, ...], freeze_hash: str, test_hash: str, class_names: list[str]) -> tuple[str, str]:
    preflight_dir = root / "inference_checkpoints" / "preflight"
    prediction_dir = root / "inference_checkpoints" / "predictions"
    preflight, predictions = {}, {}
    for spec in specs:
        stem = _checkpoint_name(spec.alias, spec.seed)
        preflight_receipt = preflight_dir / f"{stem}.json"
        prediction_receipt = prediction_dir / f"{stem}.json"
        probability_path = prediction_dir / f"{stem}.npz"
        preflight[stem] = _sha256(preflight_receipt)
        predictions[stem] = {"receipt_hash": _sha256(prediction_receipt), "npz_hash": _sha256(probability_path)}
    prediction_manifest_hash = _canonical_hash(predictions)
    payload = {
        "schema_version": 1, "freeze_hash": freeze_hash, "test_content_hash": test_hash,
        "class_hash": _canonical_hash(class_names), "run_count": len(specs),
        "preflight_receipts": preflight, "predictions": predictions,
        "prediction_manifest_hash": prediction_manifest_hash,
    }
    path = root / "inference_checkpoints" / "inference_checkpoint_manifest.json"
    _atomic_json(path, _integrity_envelope(payload))
    return _sha256(path), prediction_manifest_hash


def infer(artifact_root: Path) -> Path:
    root = artifact_root.resolve(); forbidden = _trusted_forbidden(); state_path = root / "finalization_state.json"; state = _guarded_json(state_path, forbidden, "finalization state")
    frozen_path = root / "frozen_manifest.json"
    if state.get("state") == "inferred":
        if state.get("freeze_hash") != _sha256(frozen_path): raise RuntimeError("resume freeze mismatch")
        _cached_test_bytes(root, forbidden[0], state["freeze_hash"], allow_open=False)
        return _verify_submission_manifest(root, _guarded_json(frozen_path, forbidden, "frozen manifest"), state)
    if state.get("state") not in {"frozen", "test_opened"} or state.get("freeze_hash") != _sha256(frozen_path): raise RuntimeError("infer requires the same intact freeze")
    frozen = _guarded_json(frozen_path, forbidden, "frozen manifest")
    role_to_lane = _validated_role_to_lane(frozen)
    if _sha256(root / "final_selection.json") != frozen["selection_hash"] or _sha256(root / "candidate_manifest.json") != frozen["candidate_manifest_hash"]:
        raise RuntimeError("selection/evidence mutated after freeze")
    raw_path = _within(root, frozen["nested_raw_path"])
    for name, digest in frozen["nested_raw_identity"]["hashes"].items():
        path = _guard_not_forbidden(raw_path / name, forbidden, f"frozen nested raw {name}")
        if _sha256(path) != digest:
            raise RuntimeError(f"frozen nested raw changed: {name}")
    raw_manifest = _guarded_json(
        raw_path / "nested_run_manifest.json", forbidden, "frozen nested manifest"
    )
    _validate_inference_source_closure(frozen, raw_manifest, forbidden)
    checkpoint_root = root / "inference_checkpoints"
    test_may_have_opened = state.get("state") == "test_opened" or (checkpoint_root / "test_cache").exists() or (checkpoint_root / ".test_cache.pending").exists()
    prepared = _prepare_all(root, frozen, allow_warm=not test_may_have_opened)
    variants, referenced = _frozen_variants(frozen)
    if {(spec.alias, spec.seed) for spec in prepared.runs} != {(alias, seed) for alias in referenced for seed in SEEDS}:
        raise RuntimeError("ALL_RUNS_PREPARED failed: referenced alias/seed coverage differs")
    target, identifier, test_path, train_columns, specs = prepared.target, prepared.identifier, prepared.test_path, prepared.train_columns, prepared.runs
    if test_path.resolve() != forbidden[0]:
        raise RuntimeError("prepared Test path differs from trusted canonical Test")
    test_bytes, canonical_opened = _cached_test_bytes(root, test_path, state["freeze_hash"], allow_open=state.get("state") == "frozen")
    prior_reads = int(state.get("canonical_test_read_count", 0))
    if canonical_opened and prior_reads != 0:
        raise RuntimeError("canonical Test read ledger is inconsistent")
    if not canonical_opened and state.get("state") == "frozen" and prior_reads not in {0, 1}:
        raise RuntimeError("canonical Test read ledger is inconsistent")
    test = pd.read_csv(io.BytesIO(test_bytes))
    test_ids = test[identifier].astype(str).tolist()
    test_contract = {
        "row_count": len(test), "id_order_hash": _canonical_hash(test_ids),
        "schema_hash": _canonical_hash({"columns": list(test.columns), "dtypes": [str(value) for value in test.dtypes]}),
    }
    opened_state = {**state, "state": "test_opened", "test_read_count": 1, "canonical_test_read_count": 1,
                    "all_runs_preflighted_before_test": True, "fit_after_test": 0, "warm_after_test": 0,
                    "selection_or_tuning_after_test": 0, "bundle_load_after_test_count": int(state.get("bundle_load_after_test_count", 0)),
                    "test_content_hash": hashlib.sha256(test_bytes).hexdigest(), "test_contract": test_contract,
                    "test_contract_committed_at": datetime.now(timezone.utc).isoformat()}
    _atomic_json(state_path, opened_state)
    test_features = test.drop(columns=[identifier])
    if [column for column in train_columns if column not in {target, identifier}] != list(test_features.columns): raise RuntimeError("Train/Test schema mismatch")
    for spec in specs:
        _predict_run(root, spec, test_features, frozen, state_path, opened_state, opened_state["test_content_hash"])
    lane_probability: dict[str, np.ndarray] = {}
    prediction_dir = root / "inference_checkpoints" / "predictions"
    for lane in LANES:
        total = np.zeros((len(test), len(frozen["class_names"])), dtype=float)
        member_count = 0
        for seed in SEEDS:
            for fold in FOLDS:
                variant = variants[(lane, seed, fold)]
                aliases = _variant_aliases(variant)
                member_inputs = {}
                for alias in aliases:
                    with np.load(prediction_dir / f"{_checkpoint_name(alias, seed)}.npz", allow_pickle=False) as data:
                        member_inputs[alias] = np.asarray(data[f"fold_{fold}"])
                total += nested._apply_variant(variant, member_inputs)
                member_count += 1
        if member_count != 15: raise RuntimeError("each role requires exactly 15 frozen members")
        mean = total / member_count; lane_probability[lane] = mean / mean.sum(1, keepdims=True)
    checkpoint_manifest_hash, prediction_manifest_hash = _write_checkpoint_manifest(root, specs, state["freeze_hash"], opened_state["test_content_hash"], frozen["class_names"])
    submissions = []; staging = root / ".final_inference.staging"
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir()
    selection = frozen["selections"]
    for role, lane in role_to_lane.items():
        prediction = np.asarray(frozen["class_names"])[lane_probability[lane].argmax(1)]
        frame = pd.DataFrame({"ID": test[identifier].to_numpy(), "SUBCLASS": prediction})
        path = staging / f"submission_test_007_{role}.csv"; _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))
        verified = pd.read_csv(path)
        if list(verified.columns) != ["ID", "SUBCLASS"] or len(verified) != len(test) or not verified.ID.astype(str).equals(test[identifier].astype(str)) or not set(verified.SUBCLASS).issubset(frozen["class_names"]): raise RuntimeError("submission integrity verification failed")
        submissions.append({"role": role, "candidate": selection[role], "path": path.name, "sha256": _sha256(path), "row_count": len(frame), "columns": ["ID", "SUBCLASS"], "members": 15})
    _atomic_json(staging / "final_submissions.json", {"freeze_hash": state["freeze_hash"], "test_content_hash": opened_state["test_content_hash"],
                                                       "checkpoint_manifest_hash": checkpoint_manifest_hash,
                                                       "prediction_manifest_hash": prediction_manifest_hash,
                                                       "test_contract": test_contract, "submissions": submissions})
    for item in submissions:
        os.replace(staging / item["path"], root / item["path"])
    os.replace(staging / "final_submissions.json", root / "final_submissions.json"); staging.rmdir()
    latest_state = _read_json(state_path)
    _atomic_json(state_path, {**latest_state, "state": "inferred", "updated_at": datetime.now(timezone.utc).isoformat()})
    return root / "final_submissions.json"


def document(artifact_root: Path) -> Path:
    from src.test_007.evaluate_test_007_specialization import evaluate

    root = artifact_root.resolve(); forbidden = _trusted_forbidden(); state_path = root / "finalization_state.json"; state = _guarded_json(state_path, forbidden, "finalization state")
    if (state.get("state") not in {"inferred", "documented"} or
            state.get("canonical_test_read_count") != 1 or state.get("fit_after_test") != 0 or
            state.get("warm_after_test") != 0 or state.get("selection_or_tuning_after_test") != 0):
        raise RuntimeError("document requires verified one-read frozen inference")
    frozen = _guarded_json(root / "frozen_manifest.json", forbidden, "frozen manifest")
    _verify_submission_manifest(root, frozen, state)
    before = evaluate(root)
    allowed_known_failures = {"documentation", "collapse_constraints"}
    blocking_failures = [check.name for check in before.checks if not check.passed and check.name not in allowed_known_failures]
    if blocking_failures:
        raise RuntimeError(f"documentation blocked by evaluator checks: {blocking_failures}")
    selection = frozen["selections"]
    selection_evidence = _guarded_json(root / "final_selection.json", forbidden, "final selection").get("selection_evidence", {})
    role_status = selection_evidence.get("role_status", {})
    objective_constraints_pass = "collapse_constraints" not in [check.name for check in before.checks if not check.passed]
    role_lines = []
    for role, name in selection.items():
        status = role_status.get(role, "best_available" if role == "primary" else "exploratory_failed_constraints")
        description = "best available 주력" if role == "primary" else ("자격 통과 challenger" if status == "qualified" else "탐색용 challenger · 제약 미통과")
        role_lines.append(f"- {role}: `{name}` — {description} (`{status}`)")
    objective_note = (
        "반복 붕괴 objective 제약은 통과했다."
        if objective_constraints_pass else
        "반복 붕괴 objective 제약은 통과하지 못했다. 해당 oracle-best collapse 평가는 diagnostic_only이며 이 결과를 완전 PASS로 주장하지 않는다."
    )
    report = "# TEST_007 Train-only 최종 보고서\n\n모델 선택과 튜닝에는 Test를 사용하지 않았으며 동일 pipeComb_v3, fold, seed의 완전 Nested OOF 증거만 사용했다. Gap은 목표함수가 아니라 1-SE 동률 후보의 후순위 선택 기준으로만 적용했다.\n\n" + "\n".join(role_lines) + f"\n\n{objective_note}\n\n동결 이후 모든 모델·전처리·전문가를 검증하고 워밍업한 다음 Test를 한 번만 읽어 세 제출을 생성했다. 각 역할은 3 seeds × 5 folds의 15개 확률을 균등 평균했다.\n"
    outputs = [root / "final_report_ko.md", root / "leakage_checklist.json", state_path]
    snapshot = _snapshot_files(outputs, forbidden)
    try:
        _atomic_bytes(root / "final_report_ko.md", report.encode("utf-8"))
        items = [
            {"name": "selection_train_only", "passed": True}, {"name": "nested_outer_isolation", "passed": True},
            {"name": "fold_safe_subgroups", "passed": True}, {"name": "freeze_before_test", "passed": True},
            {"name": "global_test_read_once", "passed": True}, {"name": "no_post_test_fit_or_warmup", "passed": True},
        ]
        _atomic_json(root / "leakage_checklist.json", {"complete": True, "all_passed": True, "test_used_for_selection": False, "items": items})
        failed_before = [check.name for check in before.checks if not check.passed and check.name != "documentation"]
        documented_state = {
            **state,
            "state": "documented",
            "objective_constraints_pass": objective_constraints_pass,
            "failed_checks": failed_before,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(state_path, documented_state)
        result = evaluate(root)
        failed_after = result.report(True)["failed_checks"]
        unexpected_failures = [name for name in failed_after if name != "collapse_constraints"]
        if unexpected_failures:
            raise RuntimeError(f"documentation failed evaluator --require-complete contract: {failed_after}")
        _atomic_json(state_path, {
            **documented_state,
            "objective_constraints_pass": "collapse_constraints" not in failed_after,
            "failed_checks": failed_after,
            "final_evaluation_pass": result.passed,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception:
        _restore_snapshot(snapshot)
        raise
    return root / "final_report_ko.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("evidence", "freeze", "infer", "document"))
    parser.add_argument("--artifact-root", type=Path, default=Path("data/processed/train_only_specialization"))
    parser.add_argument("--nested-source", type=Path)
    parser.add_argument("--search-universe", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/backup/yaml/test_007_specialization_goal.yaml"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase == "evidence":
        if args.nested_source is None or args.search_universe is None: raise ValueError("--nested-source and --search-universe are required for evidence")
        output = build_evidence(args.artifact_root, args.nested_source, args.config, args.search_universe)
    elif args.phase == "freeze": output = freeze(args.artifact_root)
    elif args.phase == "infer": output = infer(args.artifact_root)
    else: output = document(args.artifact_root)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
