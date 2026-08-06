"""Complete train-only Nested OOF selection for TEST_007.

The input universe is label-independent: it declares models, architectural roles,
and bounded search grids, but never a previously selected model combination.  For
each of 3 seeds x 5 outer folds, four-fold inner OOF predictions select the model
subset, temperature, weights, and ordered collision gate.  Every selected base
learner is then refit on all outer-train rows before outer-valid is predicted once.
"""

from __future__ import annotations

import argparse
import ast
import builtins
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import heapq
import io
from itertools import product
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import yaml

from src.test_007 import run_test_007_oof_bank as bank
from src.models import MODEL_BUILDERS
from src.train import build_model


SCHEMA_VERSION = 2
EXPECTED_SEEDS = (42, 2026, 777)
EXPECTED_OUTER_FOLDS = 5
EXPECTED_INNER_FOLDS = 4
LANES = ("primary", "robustness_challenger", "specialization_challenger")
FORBIDDEN_SELECTION_KEYS = {
    "strategies",
    "primary_strategy",
    "top_candidates",
    "selected_candidates",
    "selected_models",
    "frozen_top3",
}
OWNED_OUTPUTS = (
    "nested_oof_probability.npy",
    "nested_strategy_probability.npz",
    "nested_inner_selection_probability.npz",
    "nested_inner_selection_records.json",
    "nested_cv_metrics.json",
    "nested_selection_records.json",
    "nested_run_manifest.json",
    "nested_io_audit.json",
)


@dataclass(frozen=True)
class Candidate:
    alias: str
    config_path: Path
    config: dict[str, Any]


@dataclass(frozen=True)
class SearchUniverse:
    aliases: tuple[str, ...]
    executable_aliases: tuple[str, ...]
    runtime_exclusions: dict[str, str]
    roles: dict[str, tuple[str, ...]]
    allowed_kinds: tuple[str, ...]
    max_models: int
    temperatures: tuple[float, ...]
    trial_seeds: tuple[int, ...]
    trial_count_per_seed: int
    min_active_weight: float
    max_model_weight: float
    early_stop_patience: int
    role_balanced_fraction: float
    robustness_candidate_cap: int
    collision_candidate_cap: int
    margin_thresholds: tuple[float, ...]
    expert_confidence_thresholds: tuple[float, ...]
    expert_weights: tuple[float, ...]
    minimum_directed_errors: int
    minimum_rescues: int
    collapse_tolerance: float
    maximum_repeated_collapse_folds: int
    provenance: dict[str, Any]


@dataclass(frozen=True)
class DataPaths:
    raw_dir: Path
    train: Path
    test: Path
    submission: Path


@dataclass
class Audit:
    events: list[dict[str, Any]] = field(default_factory=list)

    def read(self, path: Path, role: str) -> None:
        self.events.append(
            {
                "stage": "nested_oof",
                "action": "read",
                "role": role,
                "path": str(path.resolve()),
            }
        )

    def payload(self) -> dict[str, Any]:
        def count(role: str) -> int:
            return sum(
                event["role"] == role and event.get("action") == "read"
                for event in self.events
            )

        denied = sum(str(event.get("action", "")).startswith("denied_") for event in self.events)

        return {
            "phase": "nested_oof",
            "events": self.events,
            "train_reads": count("train"),
            "test_reads": count("test"),
            "submission_reads": count("submission"),
            "nested_test_reads": count("test"),
            "nested_submission_reads": count("submission"),
            "denied_runtime_protected_reads": denied,
            "test_accessed": count("test") > 0,
        }


Predictor = Callable[
    [Candidate, pd.DataFrame, pd.Series, pd.DataFrame, tuple[str, ...], int, dict[str, Any]],
    np.ndarray,
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(path: str | Path, base: Path | None = None) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else ((base or _repo_root()) / value).resolve()


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_hash(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _variant_hash(variant: dict[str, Any]) -> str:
    return hashlib.sha256(_variant_fingerprint(variant).encode("utf-8")).hexdigest()


def _read_bytes(path: Path, audit: Audit, role: str) -> bytes:
    payload = path.read_bytes()
    audit.read(path, role)
    return payload


def _read_yaml(path: Path, audit: Audit, role: str) -> tuple[dict[str, Any], str]:
    payload = _read_bytes(path, audit, role)
    value = yaml.safe_load(payload.decode("utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{role} must contain a mapping")
    return value, _sha_bytes(payload)


def _read_csv(path: Path, audit: Audit, role: str) -> tuple[pd.DataFrame, str]:
    payload = _read_bytes(path, audit, role)
    return pd.read_csv(io.BytesIO(payload)), _sha_bytes(payload)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _same_file(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def _guard_inputs(
    goal: Path,
    train: Path,
    test: Path,
    submission: Path,
    fold: Path,
    universe: Path,
    candidate_configs: Iterable[Path],
) -> None:
    protected = {"test": test, "submission": submission}
    allowed = {
        "goal_config": goal,
        "train": train,
        "fold_assignments": fold,
        "search_universe": universe,
        **{f"candidate_config:{index}": path for index, path in enumerate(candidate_configs)},
    }
    for allowed_role, allowed_path in allowed.items():
        for protected_role, protected_path in protected.items():
            if _same_file(allowed_path, protected_path):
                raise ValueError(
                    f"nested input alias blocked before data read: {allowed_role} aliases {protected_role}"
                )
    if _same_file(fold, train) or _same_file(universe, train) or _same_file(fold, universe):
        raise ValueError("train, fold assignments, and search universe must be distinct files")


def _guard_goal_before_read(goal_path: Path) -> None:
    """Block bootstrap aliases without opening goal bytes first."""
    root = _repo_root()
    forbidden = {
        "canonical_test": root / "data" / "raw" / "test.csv",
        "canonical_submission": root / "data" / "raw" / "sample_submission.csv",
        "trusted_fold_manifest": root
        / "data"
        / "processed"
        / "train_only_specialization"
        / "fold_assignments.csv",
    }
    for role, path in forbidden.items():
        if _same_file(goal_path, path):
            raise ValueError(f"goal config alias blocked before config read: {role}")


@contextmanager
def _runtime_read_guard(
    test_path: Path,
    submission_path: Path,
    audit: Audit,
    stage: str,
    alias: str,
):
    """Deny protected reads performed anywhere inside model/preprocessing execution."""
    protected = (test_path.resolve(), submission_path.resolve())
    original_builtin_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open
    original_popen = subprocess.Popen

    def inspect_path(value: Any, action: str) -> Path | None:
        if isinstance(value, int):
            return None
        try:
            path = Path(os.fspath(value)).resolve()
        except (TypeError, ValueError, OSError):
            return None
        if any(_same_file(path, blocked) for blocked in protected):
            audit.events.append(
                {
                    "stage": stage,
                    "alias": alias,
                    "action": f"denied_{action}",
                    "role": "test" if _same_file(path, protected[0]) else "submission",
                    "path": str(path),
                }
            )
            raise RuntimeError(f"runtime protected read blocked for {path.name}")
        try:
            path.relative_to(_repo_root())
        except ValueError:
            return path
        audit.events.append(
            {
                "stage": stage,
                "alias": alias,
                "action": action,
                "role": "runtime_file",
                "path": str(path),
            }
        )
        return path

    def guarded_open(file, mode="r", *args, **kwargs):
        if "r" in str(mode) or "+" in str(mode):
            inspect_path(file, "read")
        return original_builtin_open(file, mode, *args, **kwargs)

    def guarded_io_open(file, mode="r", *args, **kwargs):
        if "r" in str(mode) or "+" in str(mode):
            inspect_path(file, "read")
        return original_io_open(file, mode, *args, **kwargs)

    def guarded_os_open(path, flags, *args, **kwargs):
        access_mask = getattr(os, "O_ACCMODE", os.O_WRONLY | os.O_RDWR)
        access_mode = int(flags) & access_mask
        if access_mode != os.O_WRONLY:
            inspect_path(path, "read")
        return original_os_open(path, flags, *args, **kwargs)

    class GuardedPopen(original_popen):
        def __init__(self, args, *popen_args, **popen_kwargs):
            tokens = args if isinstance(args, (list, tuple)) else [args]
            protected_strings = tuple(str(path).lower() for path in protected)
            if any(
                protected_value in str(token).lower()
                for token in tokens
                for protected_value in protected_strings
            ):
                audit.events.append(
                    {
                        "stage": stage,
                        "alias": alias,
                        "action": "denied_subprocess_read",
                        "role": "test_or_submission",
                        "path": " ".join(map(str, tokens)),
                    }
                )
                raise RuntimeError("runtime subprocess protected read blocked")
            super().__init__(args, *popen_args, **popen_kwargs)

    builtins.open = guarded_open
    io.open = guarded_io_open
    os.open = guarded_os_open
    subprocess.Popen = GuardedPopen
    try:
        yield
    finally:
        subprocess.Popen = original_popen
        os.open = original_os_open
        io.open = original_io_open
        builtins.open = original_builtin_open


def _validate_probability(probability: np.ndarray, rows: int, classes: int) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.shape != (rows, classes):
        raise ValueError(f"probability shape must be {(rows, classes)}, got {values.shape}")
    bank._validate_probabilities(values)
    return values


def _normalise_weights(values: Iterable[float], size: int) -> tuple[float, ...]:
    weights = np.asarray(tuple(values), dtype=np.float64)
    if weights.shape != (size,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError(f"weight candidate for size {size} is invalid")
    if float(weights.sum()) <= 0:
        raise ValueError("weights must have a positive sum")
    weights /= weights.sum()
    return tuple(float(value) for value in weights)


def _candidate_paths(goal: dict[str, Any]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in goal.get("candidates", []):
        alias = str(item.get("alias", "")).strip()
        if not alias or alias in result:
            raise ValueError("candidate aliases must be non-empty and unique")
        result[alias] = _resolve(item["config"])
    if not result:
        raise ValueError("goal config contains no candidates")
    return result


def _load_candidates(
    paths: dict[str, Path], audit: Audit
) -> tuple[dict[str, Candidate], dict[str, str]]:
    candidates: dict[str, Candidate] = {}
    hashes: dict[str, str] = {}
    for alias, path in paths.items():
        config, digest = _read_yaml(path, audit, f"candidate_config:{alias}")
        _, preprocessing = bank._validate_config(config)
        if preprocessing.get("name") != "pipeComb_v3":
            raise ValueError(f"{alias}: nested runner requires pipeComb_v3")
        candidates[alias] = Candidate(alias, path, config)
        hashes[alias] = digest
    return candidates, hashes


def _common_data_paths(goal: dict[str, Any], candidates: dict[str, Candidate]) -> DataPaths:
    configured: list[DataPaths] = []
    for alias, candidate in candidates.items():
        data = candidate.config.get("data", {})
        required = ("raw_dir", "train_file", "test_file", "submission_file")
        if any(not data.get(key) for key in required):
            raise ValueError(f"{alias}: candidate config data paths are incomplete")
        raw = _resolve(data["raw_dir"])
        configured.append(
            DataPaths(
                raw_dir=raw,
                train=(raw / data["train_file"]).resolve(),
                test=(raw / data["test_file"]).resolve(),
                submission=(raw / data["submission_file"]).resolve(),
            )
        )
    common = configured[0]
    if any(paths != common for paths in configured[1:]):
        raise ValueError("candidate configs must share identical Train/Test/submission paths")
    goal_data = goal.get("data", {})
    goal_raw = _resolve(goal_data.get("raw_dir", common.raw_dir))
    goal_train = (goal_raw / goal_data.get("train_file", common.train.name)).resolve()
    if goal_raw != common.raw_dir or goal_train != common.train:
        raise ValueError("goal Train path must match the common candidate-config Train path")
    goal_test = (goal_raw / goal_data.get("test_file", common.test.name)).resolve()
    goal_submission = (
        goal_raw / goal_data.get("submission_file", common.submission.name)
    ).resolve()
    if goal_test != common.test or goal_submission != common.submission:
        raise ValueError("goal Test/submission paths must match candidate-config common paths")
    return common


def dry_validate_goal(config_path: Path) -> dict[str, Any]:
    """Validate the real goal and all candidate builders without reading any data CSV."""
    audit = Audit()
    goal_path = _resolve(config_path)
    _guard_goal_before_read(goal_path)
    goal, goal_hash = _read_yaml(goal_path, audit, "goal_config")
    paths = _candidate_paths(goal)
    canonical = DataPaths(
        raw_dir=_repo_root() / "data" / "raw",
        train=_repo_root() / "data" / "raw" / "train.csv",
        test=_repo_root() / "data" / "raw" / "test.csv",
        submission=_repo_root() / "data" / "raw" / "sample_submission.csv",
    )
    for index, path in enumerate(paths.values()):
        if _same_file(path, canonical.test) or _same_file(path, canonical.submission):
            raise ValueError(f"candidate config {index} aliases protected data")
    candidates, hashes = _load_candidates(paths, audit)
    data_paths = _common_data_paths(goal, candidates)
    missing = sorted(
        {
            str(candidate.config.get("model", {}).get("name", ""))
            for candidate in candidates.values()
        }
        - set(MODEL_BUILDERS)
    )
    if missing:
        raise ValueError(f"candidate model builders are not registered: {missing}")
    built_models: list[str] = []
    for alias, candidate in candidates.items():
        try:
            build_model(deepcopy(candidate.config))
        except Exception as error:
            raise ValueError(f"{alias}: model builder dry validation failed") from error
        built_models.append(str(candidate.config["model"]["name"]))
    return {
        "goal_hash": goal_hash,
        "candidate_count": len(candidates),
        "candidate_config_hashes": hashes,
        "common_data_paths": {
            "raw_dir": str(data_paths.raw_dir),
            "train": str(data_paths.train),
            "test": str(data_paths.test),
            "submission": str(data_paths.submission),
        },
        "registered_model_builders": sorted(built_models),
        "builders_instantiated": len(built_models),
        "data_csv_reads": 0,
        "io_events": audit.events,
    }


def _local_import_paths(path: Path, audit: Audit | None = None) -> set[Path]:
    root = _repo_root()
    try:
        payload = (
            _read_bytes(path, audit, f"source_discovery:{path.relative_to(root).as_posix()}")
            if audit is not None
            else path.read_bytes()
        )
        tree = ast.parse(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names if alias.name.startswith("src"))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src"):
            modules.add(node.module)
    result: set[Path] = set()
    for module in modules:
        relative = Path(*module.split("."))
        file_path = root / relative.with_suffix(".py")
        init_path = root / relative / "__init__.py"
        if file_path.is_file():
            result.add(file_path.resolve())
        elif init_path.is_file():
            result.add(init_path.resolve())
    return result


def _execution_source_paths(
    candidates: dict[str, Candidate], audit: Audit | None = None
) -> set[Path]:
    root = _repo_root()
    seeds = {
        Path(__file__).resolve(),
        Path(bank.__file__).resolve(),
        root / "src" / "train.py",
        root / "src" / "models" / "__init__.py",
        root / "src" / "models" / "torch_tabular_base.py",
        root / "src" / "pipelines" / "base.py",
        root / "src" / "pipelines" / "preprocessing_registry.py",
        root / "src" / "ensembles" / "common.py",
        root
        / "src"
        / "pipelines"
        / "jyp_preprocessing"
        / "pipeline_pipe_comb_v3.py",
    }
    for candidate in candidates.values():
        model_path = root / "src" / "models" / f"{candidate.config['model']['name']}_model.py"
        if model_path.is_file():
            seeds.add(model_path)
    closure = {path.resolve() for path in seeds if path.is_file()}
    pending = list(closure)
    while pending:
        current = pending.pop()
        for dependency in _local_import_paths(current, audit):
            if dependency not in closure:
                closure.add(dependency)
                pending.append(dependency)
    return closure


def _hash_source_paths(paths: Iterable[Path], audit: Audit) -> dict[str, str]:
    root = _repo_root()
    result: dict[str, str] = {}
    for path in sorted(paths, key=str):
        try:
            name = path.relative_to(root).as_posix()
        except ValueError:
            name = str(path.resolve())
        payload = _read_bytes(path, audit, f"source:{name}")
        result[name] = _sha_bytes(payload)
    return result


def _trusted_source_hashes(
    candidates: dict[str, Candidate], audit: Audit
) -> dict[str, str]:
    return _hash_source_paths(_execution_source_paths(candidates, audit), audit)


def _assert_source_unchanged(
    source_paths: Iterable[Path], before: dict[str, str], audit: Audit
) -> dict[str, str]:
    after = _hash_source_paths(source_paths, audit)
    if after != before:
        changed = sorted(set(before).union(after) - {key for key in before if before.get(key) == after.get(key)})
        raise RuntimeError(f"execution source drift detected during Nested run: {changed}")
    return after


def _fixed_class_names(candidates: dict[str, Candidate], labels: pd.Series) -> tuple[str, ...]:
    orders = [
        tuple(map(str, candidate.config.get("model", {}).get("class_names", [])))
        for candidate in candidates.values()
        if candidate.config.get("model", {}).get("class_names")
    ]
    if orders and any(order != orders[0] for order in orders[1:]):
        raise ValueError("candidate model.class_names orders must be identical")
    names = orders[0] if orders else tuple(sorted(labels.unique()))
    if len(names) != 26 or len(set(names)) != 26 or set(names) != set(labels.unique()):
        raise ValueError("fixed 26-class order must exactly match training labels")
    return names


def _load_universe(
    raw: dict[str, Any], candidates: dict[str, Candidate], goal_search: dict[str, Any]
) -> SearchUniverse:
    def forbidden_keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return {
                *(str(key) for key in value if str(key) in FORBIDDEN_SELECTION_KEYS),
                *(key for item in value.values() for key in forbidden_keys(item)),
            }
        if isinstance(value, list):
            return {key for item in value for key in forbidden_keys(item)}
        return set()

    forbidden = sorted(forbidden_keys(raw))
    if forbidden:
        raise ValueError(
            "preselected/top-3 strategy input is forbidden; declare a label-independent "
            f"search_universe instead: {forbidden}"
        )
    provenance = raw.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("search universe requires provenance")
    if (
        provenance.get("label_independent") is not True
        or provenance.get("global_oof_selection_used") is not False
        or provenance.get("kind") != "predeclared_search_universe"
        or provenance.get("declared_before_outer_labels") is not True
    ):
        raise ValueError("search universe provenance must prove label-independent predeclaration")
    search = raw.get("search_universe")
    if not isinstance(search, dict):
        raise ValueError("search_universe mapping is required")
    aliases = tuple(map(str, search.get("models", [])))
    if not aliases or len(set(aliases)) != len(aliases) or any(alias not in candidates for alias in aliases):
        raise ValueError("search_universe.models must be unique declared candidate aliases")
    if set(aliases) != set(candidates):
        raise ValueError("search_universe.models must include every goal candidate without score-based pruning")
    expected_exclusions: dict[str, str] = {}
    runtime_exclusions = {
        str(alias): str(reason)
        for alias, reason in (search.get("runtime_exclusions") or {}).items()
    }
    if runtime_exclusions != expected_exclusions:
        raise ValueError(
            "runtime_exclusions must be empty for supported candidates"
        )
    executable_aliases = tuple(alias for alias in aliases if alias not in runtime_exclusions)
    if len(executable_aliases) < 2:
        raise ValueError("runtime safety exclusions leave too few executable candidates")
    roles_raw = search.get("roles", {})
    roles = {
        alias: tuple(map(str, roles_raw.get(alias, [])))
        for alias in aliases
    }
    robustness = search.get("robustness_constraints")
    if not isinstance(robustness, dict):
        raise ValueError("search universe must declare robustness_constraints")
    if (
        robustness.get("anchor_role") != "ROBUSTNESS_ANCHOR"
        or set(map(str, robustness.get("dimensions", []))) != {"burden", "novelty"}
        or robustness.get("evidence_source") != "inner_fold_train_only"
    ):
        raise ValueError(
            "robustness constraints require an anchor and fold-train burden/novelty evidence"
        )
    if not any("ROBUSTNESS_ANCHOR" in roles[alias] for alias in executable_aliases):
        raise ValueError("at least one model must declare ROBUSTNESS_ANCHOR")
    if not any("GLOBAL_BACKBONE" in roles[alias] for alias in executable_aliases):
        raise ValueError("at least one model must declare GLOBAL_BACKBONE")
    if not any("COLLISION_EXPERT" in roles[alias] for alias in executable_aliases):
        raise ValueError("at least one model must declare COLLISION_EXPERT")
    allowed = tuple(map(str, search.get("allowed_kinds", ["single", "blend", "collision_gate"])))
    if not set(allowed).issubset({"single", "blend", "collision_gate"}):
        raise ValueError("allowed_kinds contains an unsupported structure")
    if "blend" not in allowed:
        raise ValueError("bounded random search requires blend in allowed_kinds")
    max_models = int(search.get("max_models", 3))
    if not 1 <= max_models <= min(5, len(executable_aliases)):
        raise ValueError("max_models must be between 1 and min(5, model count)")
    temperatures = tuple(float(value) for value in search.get("temperature_values", [0.85, 1.0, 1.15]))
    if not temperatures or any(not np.isfinite(value) or value <= 0 for value in temperatures):
        raise ValueError("temperature_values must be finite and positive")
    trial_seeds = tuple(int(value) for value in search.get("trial_seeds", []))
    trial_count = int(search.get("trial_count_per_seed", 0))
    min_weight = float(search.get("min_active_weight", -1))
    max_weight = float(search.get("max_model_weight", -1))
    patience = int(search.get("early_stop_patience", 0))
    balanced_fraction = float(search.get("role_balanced_fraction", -1))
    expected_seeds = tuple(int(value) for value in goal_search.get("random_seeds", []))
    expected_trials = int(goal_search.get("random_trials", 0))
    expected_min = float(goal_search.get("min_active_weight", -1))
    expected_max = float(goal_search.get("max_model_weight", -1))
    expected_patience = int(goal_search.get("early_stop_patience", 0))
    expected_models = int(goal_search.get("max_models", 0))
    if (
        not trial_seeds
        or trial_seeds != expected_seeds
        or trial_count != expected_trials
        or trial_count < 20_000
        or not np.isclose(min_weight, expected_min)
        or not np.isclose(max_weight, expected_max)
        or patience != expected_patience
        or max_models != expected_models
    ):
        raise ValueError("search universe trial contract must exactly match goal search settings")
    if not 0 < min_weight <= max_weight <= 0.80:
        raise ValueError("active weight bounds must satisfy 0 < min <= max <= 0.80")
    if patience <= 0 or patience > trial_count:
        raise ValueError("early_stop_patience must be within the per-seed trial budget")
    if not np.isclose(balanced_fraction, 0.80):
        raise ValueError("role_balanced_fraction must be exactly 0.80")
    robust_cap = int(search.get("robustness_candidate_cap", min(512, trial_count)))
    collision_cap = int(search.get("collision_candidate_cap", min(5_000, trial_count)))
    if robust_cap <= 0 or collision_cap <= 0:
        raise ValueError("robustness/collision candidate caps must be positive")
    margins = tuple(float(value) for value in search.get("margin_thresholds", [0.05, 0.10, 0.15]))
    confidences = tuple(
        float(value) for value in search.get("expert_confidence_thresholds", [0.5, 0.6])
    )
    expert_weights = tuple(float(value) for value in search.get("expert_weights", [0.1, 0.2, 0.3]))
    if any(not 0 <= value <= 1 for value in (*margins, *confidences)):
        raise ValueError("gate thresholds must be within [0, 1]")
    if any(not 0.10 <= value <= 0.30 for value in expert_weights):
        raise ValueError("expert weights must be within [0.10, 0.30]")
    collapse_tolerance = float(robustness.get("collapse_tolerance", 0.05))
    maximum_repeated = int(robustness.get("maximum_repeated_collapse_folds", 1))
    if not 0 <= collapse_tolerance <= 1:
        raise ValueError("robustness collapse_tolerance must be within [0, 1]")
    if not 0 <= maximum_repeated < EXPECTED_INNER_FOLDS:
        raise ValueError("maximum_repeated_collapse_folds must be between 0 and 3")
    return SearchUniverse(
        aliases=aliases,
        executable_aliases=executable_aliases,
        runtime_exclusions=runtime_exclusions,
        roles=roles,
        allowed_kinds=allowed,
        max_models=max_models,
        temperatures=temperatures,
        trial_seeds=trial_seeds,
        trial_count_per_seed=trial_count,
        min_active_weight=min_weight,
        max_model_weight=max_weight,
        early_stop_patience=patience,
        role_balanced_fraction=balanced_fraction,
        robustness_candidate_cap=robust_cap,
        collision_candidate_cap=collision_cap,
        margin_thresholds=margins,
        expert_confidence_thresholds=confidences,
        expert_weights=expert_weights,
        minimum_directed_errors=int(search.get("minimum_directed_errors", 1)),
        minimum_rescues=int(search.get("minimum_rescues", 1)),
        collapse_tolerance=collapse_tolerance,
        maximum_repeated_collapse_folds=maximum_repeated,
        provenance=deepcopy(provenance),
    )


def _load_fold_matrix(
    assignments: pd.DataFrame,
    train: pd.DataFrame,
    identifier: str,
    seeds: tuple[int, ...],
) -> dict[int, np.ndarray]:
    required = {"row_index", identifier, "seed", "fold"}
    if not required.issubset(assignments):
        raise ValueError(f"fold assignments require columns {sorted(required)}")
    frame = assignments.copy()
    for column in ("row_index", "seed", "fold"):
        numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"fold assignment {column} must contain finite integers")
        frame[column] = numeric.astype(np.int64)
    if set(frame.seed.tolist()) != set(seeds) or len(frame) != len(train) * len(seeds):
        raise ValueError("fold assignments must contain exactly the fixed seed matrix")
    if train[identifier].astype(str).duplicated().any():
        raise ValueError("Train IDs must be unique for fold alignment")
    matrix: dict[int, np.ndarray] = {}
    expected_rows = np.arange(len(train), dtype=np.int64)
    expected_ids = train[identifier].astype(str).to_numpy()
    for seed in seeds:
        selected = frame.loc[frame.seed == seed].sort_values("row_index", kind="stable")
        if len(selected) != len(train) or selected.row_index.duplicated().any():
            raise ValueError(f"seed {seed}: fold assignments require one row per train row")
        if not np.array_equal(selected.row_index.to_numpy(), expected_rows):
            raise ValueError(f"seed {seed}: row_index does not exactly cover Train order")
        if not np.array_equal(selected[identifier].astype(str).to_numpy(), expected_ids):
            raise ValueError(f"seed {seed}: ID does not match Train after row_index ordering")
        if selected[identifier].astype(str).duplicated().any():
            raise ValueError(f"seed {seed}: fold assignment IDs must be unique")
        folds = selected.fold.to_numpy(dtype=np.int16)
        if set(folds.tolist()) != set(range(EXPECTED_OUTER_FOLDS)):
            raise ValueError(f"seed {seed}: folds must cover 0 through 4")
        matrix[seed] = folds
    return matrix


def _temperature(probability: np.ndarray, value: float) -> np.ndarray:
    logits = np.log(np.clip(probability, 1e-12, 1.0)) / value
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def _role_balanced_subset(
    universe: SearchUniverse, size: int, rng: np.random.Generator
) -> tuple[str, ...]:
    global_models = [
        alias for alias in universe.executable_aliases if "GLOBAL_BACKBONE" in universe.roles[alias]
    ]
    anchors = [
        alias
        for alias in universe.executable_aliases
        if set(universe.roles[alias]).intersection(
            {"STABILITY_ANCHOR", "ROBUSTNESS_ANCHOR", "DIVERSITY_MODEL"}
        )
    ]
    experts = [
        alias
        for alias in universe.executable_aliases
        if any(
            role in {"COLLISION_EXPERT", "RARE_CLASS_RESCUER", "NOVELTY_EXPERT"}
            for role in universe.roles[alias]
        )
    ]
    selected = [str(rng.choice(global_models))]
    if (
        anchors
        and not any(alias in anchors for alias in selected)
        and len(selected) < size
        and bool(rng.integers(0, 2))
    ):
        choice = str(rng.choice(anchors))
        if choice not in selected:
            selected.append(choice)
    expert_slots = min(2, size - len(selected), len(experts))
    if expert_slots:
        expert_count = int(rng.integers(0, expert_slots + 1))
        if expert_count:
            selected.extend(
                map(str, rng.choice(experts, size=expert_count, replace=False).tolist())
            )
    special = set(anchors).union(experts)
    remaining = [
        alias for alias in universe.executable_aliases if alias not in selected and alias not in special
    ]
    if len(selected) < size:
        needed = size - len(selected)
        if len(remaining) < needed:
            anchor_count = sum(alias in anchors for alias in selected)
            expert_count = sum(alias in experts for alias in selected)
            fallback = [
                alias
                for alias in universe.executable_aliases
                if alias not in selected
                and (
                    (alias in anchors and anchor_count < 1)
                    or (alias in experts and expert_count < 2)
                )
            ]
            for alias in map(str, rng.permutation(fallback).tolist()):
                if len(remaining) >= needed:
                    break
                if alias in anchors and anchor_count >= 1:
                    continue
                if alias in experts and expert_count >= 2:
                    continue
                remaining.append(alias)
                anchor_count += int(alias in anchors)
                expert_count += int(alias in experts)
        if len(remaining) < needed:
            raise RuntimeError(
                "role-balanced universe cannot satisfy GLOBAL + <=1 anchor + <=2 expert constraints"
            )
        selected.extend(
            map(str, rng.choice(remaining, size=needed, replace=False).tolist())
        )
    result = tuple(sorted(selected))
    anchor_count = sum(alias in anchors for alias in result)
    expert_count = sum(alias in experts for alias in result)
    if anchor_count > 1 or expert_count > 2 or not any(alias in global_models for alias in result):
        raise RuntimeError("role-balanced subset invariant failed")
    return result


def _sample_weights(
    universe: SearchUniverse, size: int, rng: np.random.Generator
) -> tuple[float, ...]:
    for _ in range(128):
        weights = rng.dirichlet(np.ones(size))
        if (
            np.all(weights >= universe.min_active_weight)
            and np.all(weights <= universe.max_model_weight)
        ):
            return tuple(map(float, weights))
    equal = np.full(size, 1.0 / size)
    if (
        np.all(equal >= universe.min_active_weight)
        and np.all(equal <= universe.max_model_weight)
    ):
        return tuple(map(float, equal))
    raise RuntimeError("weight bounds cannot produce an eligible convex blend")


def _random_base_variant(
    universe: SearchUniverse,
    rng: np.random.Generator,
    role_balanced: bool,
) -> dict[str, Any]:
    size = int(rng.integers(2, universe.max_models + 1))
    if role_balanced:
        aliases = _role_balanced_subset(universe, size, rng)
    else:
        aliases = tuple(sorted(map(str, rng.choice(universe.executable_aliases, size=size, replace=False))))
    return {
        "kind": "blend",
        "models": aliases,
        "weights": _sample_weights(universe, size, rng),
        "temperatures": tuple(
            float(rng.choice(universe.temperatures)) for _ in range(size)
        ),
    }


def _stream_random_variants(
    universe: SearchUniverse, trial_seed: int
) -> Iterable[tuple[int, bool, dict[str, Any]]]:
    """Yield a fixed label-independent trial stream without retaining prior variants."""
    if trial_seed not in universe.trial_seeds:
        raise ValueError("trial seed is outside the declared fixed search contract")
    rng = np.random.default_rng(trial_seed)
    for trial in range(universe.trial_count_per_seed):
        role_balanced = (trial % 5) != 0
        yield trial, role_balanced, _random_base_variant(universe, rng, role_balanced)


def _single_variants(universe: SearchUniverse) -> Iterable[dict[str, Any]]:
    if "single" not in universe.allowed_kinds:
        return
    for alias in universe.executable_aliases:
        for temperature in universe.temperatures:
            yield {
                "kind": "single",
                "models": (alias,),
                "weights": (1.0,),
                "temperatures": (temperature,),
            }


def _exhaustive_subset_temperature_upper_bound(universe: SearchUniverse) -> int:
    minimum = 1 if "single" in universe.allowed_kinds else 2
    return sum(
        math.comb(len(universe.executable_aliases), size) * len(universe.temperatures) ** size
        for size in range(minimum, universe.max_models + 1)
    )


def _apply_base(
    variant: dict[str, Any],
    probabilities: dict[str, np.ndarray],
    calibrated_cache: dict[tuple[str, float], np.ndarray] | None = None,
) -> np.ndarray:
    cache = calibrated_cache if calibrated_cache is not None else {}
    result = None
    for alias, weight, temperature in zip(
        variant["models"], variant["weights"], variant["temperatures"]
    ):
        key = (alias, float(temperature))
        calibrated = cache.get(key)
        if calibrated is None:
            calibrated = _temperature(probabilities[alias], float(temperature))
            cache[key] = calibrated
        result = float(weight) * calibrated if result is None else result + float(weight) * calibrated
    assert result is not None
    return result / result.sum(axis=1, keepdims=True)


def _apply_gate(
    variant: dict[str, Any],
    probabilities: dict[str, np.ndarray],
    calibrated_cache: dict[tuple[str, float], np.ndarray] | None = None,
) -> np.ndarray:
    cache = calibrated_cache if calibrated_cache is not None else {}
    backbone = _apply_base(variant["backbone_variant"], probabilities, cache)
    expert_key = (variant["expert"], float(variant["expert_temperature"]))
    expert = cache.get(expert_key)
    if expert is None:
        expert = _temperature(probabilities[variant["expert"]], variant["expert_temperature"])
        cache[expert_key] = expert
    source, target = variant["ordered_pair"]  # true source A -> predicted target B
    order = np.argsort(backbone, axis=1)[:, -2:][:, ::-1]
    margin = backbone[np.arange(len(backbone)), order[:, 0]] - backbone[
        np.arange(len(backbone)), order[:, 1]
    ]
    gate = (
        (order[:, 0] == target)
        & (order[:, 1] == source)
        & (margin <= variant["margin_threshold"])
        & (expert.max(axis=1) >= variant["expert_confidence_threshold"])
    )
    result = backbone.copy()
    weight = float(variant["expert_weight"])
    global_weight = float(variant["global_weight"])
    if (
        not 0.10 <= weight <= 0.30
        or not 0.70 <= global_weight <= 0.90
        or not np.isclose(global_weight + weight, 1.0, atol=1e-12)
    ):
        raise ValueError("collision gate must use 70-90% global blend and 10-30% expert")
    result[gate] = (1.0 - weight) * backbone[gate] + weight * expert[gate]
    return result / result.sum(axis=1, keepdims=True)


def _apply_variant(variant: dict[str, Any], probabilities: dict[str, np.ndarray]) -> np.ndarray:
    return _apply_gate(variant, probabilities) if variant["kind"] == "collision_gate" else _apply_base(variant, probabilities)


def _macro_f1(labels: np.ndarray, probability: np.ndarray, classes: int) -> float:
    return float(
        f1_score(
            labels,
            probability.argmax(axis=1),
            labels=np.arange(classes),
            average="macro",
            zero_division=0,
        )
    )


def _variant_fingerprint(variant: dict[str, Any]) -> str:
    return json.dumps(variant, sort_keys=True, separators=(",", ":"))


def _discover_ordered_pairs(
    labels: np.ndarray, backbone: np.ndarray, minimum_errors: int
) -> list[tuple[int, int]]:
    prediction = backbone.argmax(axis=1)
    counts: dict[tuple[int, int], int] = {}
    for source, target in zip(labels.tolist(), prediction.tolist()):
        if source != target:
            counts[(int(source), int(target))] = counts.get((int(source), int(target)), 0) + 1
    return sorted(pair for pair, count in counts.items() if count >= minimum_errors)


def _gate_evidence(
    labels: np.ndarray,
    base_probability: np.ndarray,
    gate_probability: np.ndarray,
    ordered_pair: tuple[int, int],
) -> dict[str, int]:
    source, target = ordered_pair
    base_prediction = base_probability.argmax(axis=1)
    gate_prediction = gate_probability.argmax(axis=1)
    base_correct = base_prediction == labels
    gate_correct = gate_prediction == labels
    directed_error = (labels == source) & (base_prediction == target)
    rescue = int(np.sum(directed_error & gate_correct))
    harm = int(np.sum(base_correct & ~gate_correct))
    return {
        "directed_error_support": int(np.sum(directed_error)),
        "rescue": rescue,
        "harm": harm,
        "net_rescue": rescue - harm,
    }


def _mutation_mask(frame: pd.DataFrame) -> np.ndarray:
    values = frame.fillna("WT").astype(str).to_numpy()
    normalized = np.char.upper(np.char.strip(values.astype(str)))
    return ~np.isin(normalized, ("", "WT", "WILDTYPE", "NAN", "NONE"))


def _assign_quantile_groups(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.where(values <= lower, 0, np.where(values >= upper, 2, 1)).astype(np.int8)


def _fold_safe_groups(
    train_x: pd.DataFrame, valid_x: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    train_active = _mutation_mask(train_x)
    valid_active = _mutation_mask(valid_x)
    train_burden = train_active.sum(axis=1).astype(float)
    valid_burden = valid_active.sum(axis=1).astype(float)
    burden_thresholds = np.quantile(train_burden, [0.25, 0.75]).astype(float)

    train_values = train_x.fillna("WT").astype(str).to_numpy()
    valid_values = valid_x.fillna("WT").astype(str).to_numpy()
    train_rare = np.zeros(len(train_x), dtype=float)
    valid_unseen = np.zeros(len(valid_x), dtype=float)
    train_denominator = np.maximum(train_active.sum(axis=1), 1)
    valid_denominator = np.maximum(valid_active.sum(axis=1), 1)
    for column in range(train_values.shape[1]):
        active_tokens = train_values[train_active[:, column], column]
        unique, counts = np.unique(active_tokens, return_counts=True)
        frequency = dict(zip(unique.tolist(), counts.tolist()))
        for row in np.flatnonzero(train_active[:, column]):
            train_rare[row] += frequency.get(train_values[row, column], 0) <= 1
        for row in np.flatnonzero(valid_active[:, column]):
            valid_unseen[row] += valid_values[row, column] not in frequency
    train_novelty = train_rare / train_denominator
    valid_novelty = valid_unseen / valid_denominator
    novelty_thresholds = np.quantile(train_novelty, [0.25, 0.75]).astype(float)
    return (
        {
            "burden": _assign_quantile_groups(
                valid_burden, burden_thresholds[0], burden_thresholds[1]
            ),
            "novelty": _assign_quantile_groups(
                valid_novelty, novelty_thresholds[0], novelty_thresholds[1]
            ),
        },
        {
            "burden": burden_thresholds.tolist(),
            "novelty": novelty_thresholds.tolist(),
        },
    )


def _worst_group_floor(
    labels: np.ndarray,
    probability: np.ndarray,
    groups: dict[str, np.ndarray],
    classes: int,
    mask: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    active = np.ones(len(labels), dtype=bool) if mask is None else mask
    scores: dict[str, float] = {}
    for dimension, values in groups.items():
        for group in (0, 1, 2):
            selected = active & (values == group)
            if selected.any():
                scores[f"{dimension}:{group}"] = _macro_f1(
                    labels[selected], probability[selected], classes
                )
    return (min(scores.values()) if scores else 0.0), scores


def _robustness_evidence(
    labels: np.ndarray,
    candidate: np.ndarray,
    primary: np.ndarray,
    groups: dict[str, np.ndarray],
    inner_fold_ids: np.ndarray,
    classes: int,
    tolerance: float,
) -> dict[str, Any]:
    floor, scores = _worst_group_floor(labels, candidate, groups, classes)
    repeated = 0
    fold_deltas: dict[str, float] = {}
    for fold in range(EXPECTED_INNER_FOLDS):
        mask = inner_fold_ids == fold
        candidate_floor, _ = _worst_group_floor(labels, candidate, groups, classes, mask)
        primary_floor, _ = _worst_group_floor(labels, primary, groups, classes, mask)
        delta = candidate_floor - primary_floor
        fold_deltas[str(fold)] = delta
        repeated += int(delta < -tolerance)
    return {
        "worst_group_floor": floor,
        "group_scores": scores,
        "fold_worst_group_delta_vs_primary": fold_deltas,
        "repeated_collapse_folds": repeated,
    }


def _collision_best(
    universe: SearchUniverse,
    probabilities: dict[str, np.ndarray],
    labels: np.ndarray,
    backbone_variant: dict[str, Any],
    calibrated_cache: dict[tuple[str, float], np.ndarray],
) -> tuple[tuple[float, str, dict[str, Any], dict[str, int]] | None, dict[str, Any]]:
    if "collision_gate" not in universe.allowed_kinds:
        return None, {"enabled": False, "theoretical_candidates": 0, "evaluated": 0}
    experts = [
        alias for alias in universe.executable_aliases if "COLLISION_EXPERT" in universe.roles.get(alias, ())
    ]
    base_probability = _apply_base(backbone_variant, probabilities, calibrated_cache)
    ordered_pairs = _discover_ordered_pairs(
        labels, base_probability, universe.minimum_directed_errors
    )
    experts = [alias for alias in experts if alias not in backbone_variant["models"]]
    axes: tuple[tuple[Any, ...], ...] = (
        tuple(experts),
        tuple(universe.temperatures),
        tuple(ordered_pairs),
        tuple(universe.margin_thresholds),
        tuple(universe.expert_confidence_thresholds),
        tuple(universe.expert_weights),
    )
    theoretical = math.prod(len(axis) for axis in axes)
    limit = min(theoretical, universe.collision_candidate_cap)
    sampled = theoretical > limit

    def indices() -> Iterable[int]:
        if not sampled:
            yield from range(theoretical)
            return
        rng = np.random.default_rng(universe.trial_seeds[0] + 730)
        start = int(rng.integers(0, theoretical))
        step = int(rng.integers(1, theoretical))
        while math.gcd(step, theoretical) != 1:
            step = step % theoretical + 1
        for offset in range(limit):
            yield (start + offset * step) % theoretical

    best: tuple[float, str, dict[str, Any], dict[str, int]] | None = None
    evaluated = eligible_count = 0
    for flat_index in indices():
        remainder = flat_index
        decoded: list[Any] = []
        for axis in reversed(axes):
            decoded.append(axis[remainder % len(axis)])
            remainder //= len(axis)
        expert_alias, expert_temperature, pair, margin, confidence, weight = reversed(decoded)
        variant = {
            "kind": "collision_gate",
            "backbone_variant": backbone_variant,
            "expert": expert_alias,
            "ordered_pair": pair,
            "expert_temperature": expert_temperature,
            "margin_threshold": margin,
            "expert_confidence_threshold": confidence,
            "expert_weight": weight,
            "global_weight": 1.0 - weight,
        }
        gated = _apply_gate(variant, probabilities, calibrated_cache)
        evidence = _gate_evidence(labels, base_probability, gated, pair)
        evaluated += 1
        if (
            evidence["directed_error_support"] >= universe.minimum_directed_errors
            and evidence["rescue"] >= universe.minimum_rescues
            and evidence["net_rescue"] > 0
        ):
            eligible_count += 1
            score = _macro_f1(labels, gated, base_probability.shape[1])
            row = (score, _variant_fingerprint(variant), variant, evidence)
            if best is None or (-row[0], row[1]) < (-best[0], best[1]):
                best = row
    return best, {
        "enabled": True,
        "theoretical_candidates": theoretical,
        "candidate_cap": universe.collision_candidate_cap,
        "bounded_sampling_applied": sampled,
        "sampling_seed": universe.trial_seeds[0] + 730,
        "evaluated": evaluated,
        "eligible_positive_net_rescue": eligible_count,
        "peak_materialized_variants": 1,
    }


def _select_inner_lanes(
    universe: SearchUniverse,
    probabilities: dict[str, np.ndarray],
    labels: np.ndarray,
    classes: int,
    groups: dict[str, np.ndarray],
    inner_fold_ids: np.ndarray,
    group_thresholds: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    calibrated_cache: dict[tuple[str, float], np.ndarray] = {}
    anchors = {
        alias for alias, roles in universe.roles.items() if "ROBUSTNESS_ANCHOR" in roles
    }
    primary: tuple[float, str, dict[str, Any]] | None = None
    backbone: tuple[float, str, dict[str, Any]] | None = None
    robust_heap: list[tuple[float, float, int, str, dict[str, Any]]] = []
    evaluated_by_seed: dict[str, int] = {str(seed): 0 for seed in universe.trial_seeds}
    stopped_by_patience: dict[str, bool] = {str(seed): False for seed in universe.trial_seeds}
    counter = 0

    def evaluate(variant: dict[str, Any]) -> bool:
        nonlocal primary, backbone, counter
        probability = _apply_base(variant, probabilities, calibrated_cache)
        score = _macro_f1(labels, probability, classes)
        fingerprint = _variant_fingerprint(variant)
        improved = False
        row = (score, fingerprint, variant)
        if primary is None or (-score, fingerprint) < (-primary[0], primary[1]):
            primary = row
            improved = True
        is_backbone = (
            variant["kind"] == "blend"
            and len(variant["models"]) >= 2
            and all(float(weight) > 0 for weight in variant["weights"])
            and any("GLOBAL_BACKBONE" in universe.roles[alias] for alias in variant["models"])
            and all("COLLISION_EXPERT" not in universe.roles[alias] for alias in variant["models"])
        )
        if is_backbone and (
            backbone is None or (-score, fingerprint) < (-backbone[0], backbone[1])
        ):
            backbone = row
            improved = True
        if anchors.intersection(variant["models"]):
            floor, _ = _worst_group_floor(labels, probability, groups, classes)
            heap_row = (floor, score, counter, fingerprint, variant)
            counter += 1
            if len(robust_heap) < universe.robustness_candidate_cap:
                heapq.heappush(robust_heap, heap_row)
                improved = True
            elif heap_row[:2] > robust_heap[0][:2]:
                heapq.heapreplace(robust_heap, heap_row)
                improved = True
        return improved

    for variant in _single_variants(universe):
        evaluate(variant)
    random_evaluated = 0
    balanced_evaluated = unconstrained_evaluated = 0
    for trial_seed in universe.trial_seeds:
        no_improvement = 0
        for trial, role_balanced, variant in _stream_random_variants(universe, trial_seed):
            improved = evaluate(variant)
            random_evaluated += 1
            evaluated_by_seed[str(trial_seed)] += 1
            balanced_evaluated += int(role_balanced)
            unconstrained_evaluated += int(not role_balanced)
            no_improvement = 0 if improved else no_improvement + 1
            if no_improvement >= universe.early_stop_patience:
                stopped_by_patience[str(trial_seed)] = True
                break
    if primary is None:
        raise RuntimeError("streaming search produced no primary variant")
    if backbone is None:
        raise RuntimeError("streaming search produced no global convex backbone blend")
    primary_probability = _apply_base(primary[2], probabilities, calibrated_cache)
    if not robust_heap:
        raise RuntimeError("no streamed variant contains the declared robustness anchor")
    robust_heap.sort(key=lambda row: (-row[0], -row[1], row[3]))
    robust_selected: tuple[float, float, int, str, dict[str, Any], dict[str, Any]] | None = None
    for floor, mean_score, index, fingerprint, variant in robust_heap:
        if fingerprint == primary[1]:
            continue
        evidence = _robustness_evidence(
            labels,
            _apply_base(variant, probabilities, calibrated_cache),
            primary_probability,
            groups,
            inner_fold_ids,
            classes,
            universe.collapse_tolerance,
        )
        if evidence["repeated_collapse_folds"] <= universe.maximum_repeated_collapse_folds:
            robust_selected = (floor, mean_score, index, fingerprint, variant, evidence)
            break
    if robust_selected is None:
        raise RuntimeError("no robustness-anchor variant passes repeated-collapse constraints")
    gate_selected, collision_diagnostics = _collision_best(
        universe, probabilities, labels, backbone[2], calibrated_cache
    )
    if gate_selected is None:
        raise RuntimeError("no ordered collision gate has positive inner net rescue")
    requested_total = len(universe.trial_seeds) * universe.trial_count_per_seed
    single_upper = len(universe.executable_aliases) * len(universe.temperatures)
    robustness_theoretical_upper = requested_total + single_upper
    base_diagnostics = {
        "trial_seeds": list(universe.trial_seeds),
        "trial_count_per_seed": universe.trial_count_per_seed,
        "requested_random_trials_total": requested_total,
        "exhaustive_subset_temperature_upper_bound": (
            _exhaustive_subset_temperature_upper_bound(universe)
        ),
        "evaluated_random_trials_total": random_evaluated,
        "evaluated_by_seed": evaluated_by_seed,
        "early_stop_patience": universe.early_stop_patience,
        "stopped_by_patience": stopped_by_patience,
        "role_balanced_evaluated": balanced_evaluated,
        "unconstrained_evaluated": unconstrained_evaluated,
        "role_balanced_target_fraction": universe.role_balanced_fraction,
        "robustness_theoretical_candidates": robustness_theoretical_upper,
        "robustness_candidate_cap": universe.robustness_candidate_cap,
        "robustness_bounded_pool_applied": (
            robustness_theoretical_upper > universe.robustness_candidate_cap
        ),
        "peak_materialized_base_variant": 1,
        "peak_retained_robustness_variants": len(robust_heap),
        "calibrated_probability_cache_entries": len(calibrated_cache),
        "calibrated_probability_cache_upper_bound": len(universe.executable_aliases)
        * len(universe.temperatures),
    }
    return {
        "primary": {"variant": primary[2], "inner_macro_f1": primary[0]},
        "robustness_challenger": {
            "variant": robust_selected[4],
            "inner_macro_f1": robust_selected[1],
            "robustness_evidence": {
                **robust_selected[5],
                "thresholds_by_inner_fold": group_thresholds,
                "dimensions": ["burden", "novelty"],
                "threshold_source": "each_inner_fold_train_only",
            },
        },
        "specialization_challenger": {
            "variant": gate_selected[2],
            "inner_macro_f1": gate_selected[0],
            "gate_evidence": gate_selected[3],
        },
        "_search_diagnostics": {
            "base": base_diagnostics,
            "collision": collision_diagnostics,
        },
    }


def _default_predictor(
    candidate: Candidate,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    class_names: tuple[str, ...],
    model_seed: int,
    context: dict[str, Any],
) -> np.ndarray:
    del context
    config = deepcopy(candidate.config)
    config["project"]["seed"] = int(model_seed)
    _, preprocessing = bank._validate_config(config)
    preprocessor, model, mode, calibrator, _, _ = bank._fit_fold(
        config, preprocessing, train_x, train_y
    )
    if mode == "hard_only":
        raise ValueError(f"{candidate.alias}: hard-only model cannot enter probability ensemble")
    encoded_valid = preprocessor.transform(valid_x)
    local_classes = bank._local_class_names(preprocessor, list(class_names))
    probability = bank._predict_probability(
        model, mode, calibrator, encoded_valid, local_classes, list(class_names)
    )
    if probability is None:
        raise RuntimeError(f"{candidate.alias}: probability prediction unexpectedly unavailable")
    return _validate_probability(probability, len(valid_x), len(class_names))


def _inner_split_seed(seed: int, outer_fold: int) -> int:
    return int(seed * 1009 + outer_fold * 9173) % (2**31 - 1)


def _inner_splits(labels: pd.Series, seed: int, outer_fold: int) -> list[tuple[np.ndarray, np.ndarray]]:
    counts = labels.value_counts()
    if len(counts) < 2 or int(counts.min()) < EXPECTED_INNER_FOLDS:
        raise ValueError("each outer-train class needs at least four rows for inner StratifiedKFold")
    splitter = StratifiedKFold(
        n_splits=EXPECTED_INNER_FOLDS,
        shuffle=True,
        random_state=_inner_split_seed(seed, outer_fold),
    )
    return list(splitter.split(np.zeros(len(labels)), labels.to_numpy()))


def _prepare_outputs(root: Path, overwrite: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    existing = [root / name for name in OWNED_OUTPUTS if (root / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"nested output exists; use --overwrite: {existing[0]}")
    for path in existing:
        shutil.rmtree(path) if path.is_dir() else path.unlink()


def _inner_selection_key(lane: str, seed: int, outer_fold: int) -> str:
    return f"{lane}__seed_{seed}__fold_{outer_fold}"


def validate_inner_selection_transcript(
    artifact_root: Path,
    train: pd.DataFrame,
    target: str = "SUBCLASS",
    identifier: str = "ID",
    fold_assignments_path: Path | None = None,
) -> dict[str, Any]:
    """Recompute all 45 Inner-selection scores and reject transcript tampering."""
    root = _resolve(artifact_root)
    manifest = json.loads((root / "nested_run_manifest.json").read_text(encoding="utf-8"))
    live_fold_path = _resolve(
        fold_assignments_path or manifest.get("fold_assignments_path", "")
    )
    if not live_fold_path.is_file():
        raise ValueError("live fold assignment file is unavailable")
    live_fold_payload = live_fold_path.read_bytes()
    live_fold_hash = _sha_bytes(live_fold_payload)
    trusted_fold_hash = manifest.get("trusted_input_hashes", {}).get("fold_assignments")
    if live_fold_hash != trusted_fold_hash or live_fold_hash != manifest.get("fold_hash"):
        raise ValueError("live fold assignment hash does not match Nested manifest")
    live_assignments = pd.read_csv(io.BytesIO(live_fold_payload))
    fold_matrix = _load_fold_matrix(live_assignments, train, identifier, EXPECTED_SEEDS)
    transcript = json.loads(
        (root / "nested_inner_selection_records.json").read_text(encoding="utf-8")
    )
    selections = json.loads(
        (root / "nested_selection_records.json").read_text(encoding="utf-8")
    )["selections"]
    records = transcript.get("records")
    if not isinstance(records, list) or len(records) != len(LANES) * len(EXPECTED_SEEDS) * 5:
        raise ValueError("inner selection transcript must contain exactly 45 records")
    expected = {
        _inner_selection_key(lane, seed, fold)
        for lane in LANES
        for seed in EXPECTED_SEEDS
        for fold in range(EXPECTED_OUTER_FOLDS)
    }
    observed = {str(record.get("array_key")) for record in records}
    if observed != expected or len(observed) != len(records):
        raise ValueError("inner selection transcript key coverage is incomplete or duplicated")
    selection_lookup = {
        _inner_selection_key(row["strategy"], int(row["seed"]), int(row["fold"])): row
        for row in selections
    }
    if set(selection_lookup) != expected:
        raise ValueError("selected variant transcript does not cover the same 45 keys")
    class_names = tuple(map(str, manifest.get("class_names", [])))
    if len(class_names) != 26 or len(set(class_names)) != 26:
        raise ValueError("manifest class order is invalid")
    class_lookup = {name: index for index, name in enumerate(class_names)}
    labels = train[target].astype(str).map(class_lookup)
    if labels.isna().any():
        raise ValueError("Train labels do not match manifest class order")
    with np.load(root / "nested_inner_selection_probability.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != expected:
            raise ValueError("inner selection probability keys are incomplete or unexpected")
        verified = 0
        for record in records:
            key = str(record["array_key"])
            lane = str(record.get("strategy"))
            seed = int(record.get("seed"))
            outer_fold = int(record.get("outer_fold"))
            if (
                lane not in LANES
                or seed not in EXPECTED_SEEDS
                or outer_fold not in range(EXPECTED_OUTER_FOLDS)
                or key != _inner_selection_key(lane, seed, outer_fold)
            ):
                raise ValueError(f"{key}: lane/seed/outer-fold identity mismatch")
            probability = np.asarray(arrays[key], dtype=np.float64)
            row_indices = np.asarray(record.get("row_indices", []), dtype=np.int64)
            row_ids = list(map(str, record.get("row_ids", [])))
            coverage = np.asarray(record.get("inner_oof_coverage", []), dtype=np.int8)
            fold_ids = np.asarray(record.get("inner_fold_ids", []), dtype=np.int8)
            expected_rows = np.flatnonzero(fold_matrix[seed] != outer_fold)
            if probability.shape != (len(expected_rows), len(class_names)) or not np.array_equal(
                row_indices, expected_rows
            ):
                raise ValueError(f"{key}: rows must equal the exact ordered outer-train complement")
            _validate_probability(probability, len(row_indices), len(class_names))
            if row_ids != train.iloc[row_indices][identifier].astype(str).tolist():
                raise ValueError(f"{key}: row IDs do not match current Train")
            if coverage.shape != (len(row_indices),) or not np.all(coverage == 1):
                raise ValueError(f"{key}: inner OOF coverage must equal one for every row")
            expected_inner_fold_ids = np.full(len(expected_rows), -1, dtype=np.int8)
            outer_train_labels = train.iloc[expected_rows][target].astype(str).reset_index(drop=True)
            for expected_fold, (_, inner_valid_local) in enumerate(
                _inner_splits(outer_train_labels, seed, outer_fold)
            ):
                expected_inner_fold_ids[inner_valid_local] = expected_fold
            if fold_ids.shape != (len(expected_rows),) or not np.array_equal(
                fold_ids, expected_inner_fold_ids
            ):
                raise ValueError(f"{key}: inner fold IDs differ from deterministic live split")
            if _array_hash(probability) != record.get("array_hash"):
                raise ValueError(f"{key}: probability array hash mismatch")
            selection = selection_lookup[key]
            if (
                selection.get("strategy") != lane
                or int(selection.get("seed")) != seed
                or int(selection.get("fold")) != outer_fold
            ):
                raise ValueError(f"{key}: selection identity mismatch")
            variant = selection["selected_variant"]
            fingerprint = _variant_fingerprint(variant)
            if (
                fingerprint != record.get("selected_variant_fingerprint")
                or _variant_hash(variant) != record.get("selected_variant_hash")
            ):
                raise ValueError(f"{key}: selected variant fingerprint/hash mismatch")
            score = _macro_f1(
                labels.iloc[row_indices].to_numpy(dtype=np.int32),
                probability,
                len(class_names),
            )
            recorded_score = float(record.get("inner_selection_macro_f1"))
            selection_score = float(selection.get("inner_selection_macro_f1"))
            if not np.isclose(score, recorded_score, atol=1e-12, rtol=0.0) or not np.isclose(
                score, selection_score, atol=1e-12, rtol=0.0
            ):
                raise ValueError(f"{key}: inner selection Macro F1 transcript mismatch")
            verified += 1
    return {"verified_records": verified, "exact_key_coverage": True}


def run_nested(
    config_path: Path,
    strategy_path: Path,
    fold_assignments_path: Path,
    artifact_root: Path,
    overwrite: bool = False,
    predictor: Predictor | None = None,
) -> Path:
    """Run fully nested selection without opening Test or submission bytes."""
    audit = Audit()
    goal_path = _resolve(config_path)
    universe_path = _resolve(strategy_path)
    fold_path = _resolve(fold_assignments_path)
    _guard_goal_before_read(goal_path)
    goal, goal_hash = _read_yaml(goal_path, audit, "goal_config")
    validation = goal.get("validation", {})
    seeds = tuple(int(value) for value in validation.get("final_seeds", EXPECTED_SEEDS))
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"nested runner requires fixed seeds {EXPECTED_SEEDS}")
    if int(validation.get("outer_folds", EXPECTED_OUTER_FOLDS)) != EXPECTED_OUTER_FOLDS:
        raise ValueError("nested runner requires five outer folds")
    if int(validation.get("inner_folds", EXPECTED_INNER_FOLDS)) != EXPECTED_INNER_FOLDS:
        raise ValueError("nested runner requires four inner folds")

    data = goal["data"]
    raw_dir = _resolve(data.get("raw_dir", "data/raw"))
    preliminary_train = (raw_dir / data.get("train_file", "train.csv")).resolve()
    preliminary_test = (raw_dir / data.get("test_file", "test.csv")).resolve()
    preliminary_submission = (
        raw_dir / data.get("submission_file", "sample_submission.csv")
    ).resolve()
    candidate_paths = _candidate_paths(goal)
    _guard_inputs(
        goal_path,
        preliminary_train,
        preliminary_test,
        preliminary_submission,
        fold_path,
        universe_path,
        candidate_paths.values(),
    )

    candidates, candidate_hashes = _load_candidates(candidate_paths, audit)
    common_data = _common_data_paths(goal, candidates)
    _guard_inputs(
        goal_path,
        common_data.train,
        common_data.test,
        common_data.submission,
        fold_path,
        universe_path,
        candidate_paths.values(),
    )
    universe_raw, universe_hash = _read_yaml(universe_path, audit, "search_universe")
    universe = _load_universe(universe_raw, candidates, goal.get("search", {}))
    for alias, reason in sorted(universe.runtime_exclusions.items()):
        audit.events.append(
            {
                "stage": "nested_setup",
                "action": "excluded",
                "role": "runtime_exclusion",
                "alias": alias,
                "model_name": candidates[alias].config["model"]["name"],
                "reason": reason,
                "path": str(candidates[alias].config_path),
            }
        )
    source_paths = _execution_source_paths(candidates, audit)
    source_hashes_before = _hash_source_paths(source_paths, audit)
    train, train_hash = _read_csv(common_data.train, audit, "train")
    assignments, fold_hash = _read_csv(fold_path, audit, "fold_assignments")
    target, identifier = str(data["target_column"]), str(data["id_column"])
    labels = train[target].astype(str)
    class_names = _fixed_class_names(candidates, labels)
    encoded_labels = labels.map({name: index for index, name in enumerate(class_names)}).to_numpy(np.int32)
    features = train.drop(columns=[target, identifier])
    fold_matrix = _load_fold_matrix(assignments, train, identifier, seeds)
    predictor = predictor or _default_predictor
    root = _resolve(artifact_root)
    _prepare_outputs(root, overwrite)

    probabilities = {
        lane: np.full((len(seeds), len(train), len(class_names)), np.nan, dtype=np.float64)
        for lane in LANES
    }
    coverage = {lane: np.zeros((len(seeds), len(train)), dtype=np.int8) for lane in LANES}
    evaluations: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    seed_matrix: list[dict[str, Any]] = []
    search_diagnostics: list[dict[str, Any]] = []
    inner_selection_arrays: dict[str, np.ndarray] = {}
    inner_selection_records: list[dict[str, Any]] = []

    for seed_index, seed in enumerate(seeds):
        outer_folds = fold_matrix[seed]
        for outer_fold in range(EXPECTED_OUTER_FOLDS):
            outer_train_index = np.flatnonzero(outer_folds != outer_fold)
            outer_valid_index = np.flatnonzero(outer_folds == outer_fold)
            outer_train_x, outer_valid_x = features.iloc[outer_train_index], features.iloc[outer_valid_index]
            outer_train_y = labels.iloc[outer_train_index]
            inner_probability = {
                alias: np.full((len(outer_train_index), len(class_names)), np.nan)
                for alias in universe.executable_aliases
            }
            inner_coverage = np.zeros(len(outer_train_index), dtype=np.int8)
            inner_fold_ids = np.full(len(outer_train_index), -1, dtype=np.int8)
            inner_groups = {
                "burden": np.full(len(outer_train_index), -1, dtype=np.int8),
                "novelty": np.full(len(outer_train_index), -1, dtype=np.int8),
            }
            group_thresholds: list[dict[str, Any]] = []
            split_seed = _inner_split_seed(seed, outer_fold)
            for inner_fold, (inner_train_local, inner_valid_local) in enumerate(
                _inner_splits(outer_train_y.reset_index(drop=True), seed, outer_fold)
            ):
                inner_coverage[inner_valid_local] += 1
                inner_fold_ids[inner_valid_local] = inner_fold
                fold_groups, thresholds = _fold_safe_groups(
                    outer_train_x.iloc[inner_train_local],
                    outer_train_x.iloc[inner_valid_local],
                )
                for dimension in inner_groups:
                    inner_groups[dimension][inner_valid_local] = fold_groups[dimension]
                group_thresholds.append({"inner_fold": inner_fold, **thresholds})
                for alias in universe.executable_aliases:
                    context = {
                        "stage": "inner",
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "train_row_indices": outer_train_index[inner_train_local].tolist(),
                        "valid_row_indices": outer_train_index[inner_valid_local].tolist(),
                    }
                    with _runtime_read_guard(
                        common_data.test, common_data.submission, audit, "inner", alias
                    ):
                        predicted = predictor(
                            candidates[alias],
                            outer_train_x.iloc[inner_train_local],
                            outer_train_y.iloc[inner_train_local],
                            outer_train_x.iloc[inner_valid_local],
                            class_names,
                            seed,
                            context,
                        )
                    inner_probability[alias][inner_valid_local] = _validate_probability(
                        predicted, len(inner_valid_local), len(class_names)
                    )
                    seed_matrix.append(
                        {
                            "stage": "inner",
                            "alias": alias,
                            "seed": seed,
                            "model_seed": seed,
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "inner_split_seed": split_seed,
                            "train_rows": len(inner_train_local),
                            "valid_rows": len(inner_valid_local),
                            "train_row_indices_hash": _canonical_hash(
                                outer_train_index[inner_train_local].tolist()
                            ),
                            "valid_row_indices_hash": _canonical_hash(
                                outer_train_index[inner_valid_local].tolist()
                            ),
                            "coverage_contract": "each outer-train row valid exactly once across inner folds",
                        }
                    )
            if not np.all(inner_coverage == 1):
                raise RuntimeError("inner OOF coverage invariant failed")
            if (inner_fold_ids < 0).any() or any((values < 0).any() for values in inner_groups.values()):
                raise RuntimeError("inner fold-safe subgroup coverage invariant failed")
            for value in inner_probability.values():
                _validate_probability(value, len(outer_train_index), len(class_names))
            selected = _select_inner_lanes(
                universe,
                inner_probability,
                encoded_labels[outer_train_index],
                len(class_names),
                inner_groups,
                inner_fold_ids,
                group_thresholds,
            )
            search_diagnostics.append(
                {
                    "seed": seed,
                    "outer_fold": outer_fold,
                    **selected["_search_diagnostics"],
                }
            )
            for lane in LANES:
                inner_combined = _apply_variant(selected[lane]["variant"], inner_probability)
                _validate_probability(
                    inner_combined, len(outer_train_index), len(class_names)
                )
                inner_score = _macro_f1(
                    encoded_labels[outer_train_index], inner_combined, len(class_names)
                )
                if not np.isclose(
                    inner_score,
                    float(selected[lane]["inner_macro_f1"]),
                    atol=1e-12,
                    rtol=0.0,
                ):
                    raise RuntimeError("selected Inner probability score transcript mismatch")
                key = _inner_selection_key(lane, seed, outer_fold)
                if key in inner_selection_arrays:
                    raise RuntimeError(f"duplicate Inner selection transcript key: {key}")
                immutable_probability = np.asarray(inner_combined, dtype=np.float64).copy()
                immutable_probability.flags.writeable = False
                inner_selection_arrays[key] = immutable_probability
                variant = selected[lane]["variant"]
                inner_selection_records.append(
                    {
                        "array_key": key,
                        "strategy": lane,
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "row_indices": outer_train_index.tolist(),
                        "row_ids": train.iloc[outer_train_index][identifier].astype(str).tolist(),
                        "inner_oof_coverage": inner_coverage.astype(int).tolist(),
                        "inner_fold_ids": inner_fold_ids.astype(int).tolist(),
                        "selected_variant_fingerprint": _variant_fingerprint(variant),
                        "selected_variant_hash": _variant_hash(variant),
                        "array_hash": _array_hash(immutable_probability),
                        "array_shape": list(immutable_probability.shape),
                        "array_dtype": str(immutable_probability.dtype),
                        "inner_selection_macro_f1": inner_score,
                        "outer_validation_used": False,
                    }
                )

            selected_aliases: set[str] = set()
            for lane in LANES:
                selection = selected[lane]
                variant = selection["variant"]
                if variant["kind"] == "collision_gate":
                    selected_aliases.update(variant["backbone_variant"]["models"])
                    selected_aliases.add(variant["expert"])
                else:
                    selected_aliases.update(variant["models"])
            outer_probability: dict[str, np.ndarray] = {}
            for alias in sorted(selected_aliases):
                context = {
                    "stage": "outer_refit",
                    "seed": seed,
                    "outer_fold": outer_fold,
                    "inner_fold": None,
                    "train_row_indices": outer_train_index.tolist(),
                    "valid_row_indices": outer_valid_index.tolist(),
                }
                with _runtime_read_guard(
                    common_data.test, common_data.submission, audit, "outer_refit", alias
                ):
                    predicted = predictor(
                        candidates[alias],
                        outer_train_x,
                        outer_train_y,
                        outer_valid_x,
                        class_names,
                        seed,
                        context,
                    )
                outer_probability[alias] = _validate_probability(
                    predicted, len(outer_valid_index), len(class_names)
                )
                seed_matrix.append(
                    {
                        "stage": "outer_refit",
                        "alias": alias,
                        "seed": seed,
                        "model_seed": seed,
                        "outer_fold": outer_fold,
                        "inner_fold": None,
                        "inner_split_seed": split_seed,
                        "train_rows": len(outer_train_index),
                        "valid_rows": len(outer_valid_index),
                        "train_row_indices_hash": _canonical_hash(outer_train_index.tolist()),
                        "valid_row_indices_hash": _canonical_hash(outer_valid_index.tolist()),
                        "coverage_contract": "selected alias refit on full outer-train and predicts outer-valid once",
                    }
                )
            outer_y = encoded_labels[outer_valid_index]
            for lane in LANES:
                probability = _apply_variant(selected[lane]["variant"], outer_probability)
                _validate_probability(probability, len(outer_valid_index), len(class_names))
                probabilities[lane][seed_index, outer_valid_index] = probability
                coverage[lane][seed_index, outer_valid_index] += 1
                evaluations.append(
                    {
                        "strategy": lane,
                        "seed": seed,
                        "fold": outer_fold,
                        "inner_selection_macro_f1": selected[lane]["inner_macro_f1"],
                        "macro_f1": _macro_f1(outer_y, probability, len(class_names)),
                        "outer_train_rows": len(outer_train_index),
                        "outer_valid_rows": len(outer_valid_index),
                    }
                )
                selections.append(
                    {
                        "strategy": lane,
                        "seed": seed,
                        "fold": outer_fold,
                        "selected_variant": selected[lane]["variant"],
                        "inner_selection_macro_f1": selected[lane]["inner_macro_f1"],
                        "gate_evidence": selected[lane].get("gate_evidence"),
                        "robustness_evidence": selected[lane].get("robustness_evidence"),
                        "selection_source": "outer_train_inner_oof_only",
                        "outer_validation_used_for_selection": False,
                    }
                )

    for lane, probability in probabilities.items():
        if not np.all(coverage[lane] == 1):
            raise RuntimeError(f"{lane}: outer OOF coverage invariant failed")
        for seed_index in range(len(seeds)):
            _validate_probability(probability[seed_index], len(train), len(class_names))

    np.save(root / "nested_oof_probability.npy", probabilities["primary"])
    np.savez_compressed(root / "nested_strategy_probability.npz", **probabilities)
    expected_inner_keys = {
        _inner_selection_key(lane, seed, fold)
        for lane in LANES
        for seed in seeds
        for fold in range(EXPECTED_OUTER_FOLDS)
    }
    if (
        set(inner_selection_arrays) != expected_inner_keys
        or len(inner_selection_records) != len(expected_inner_keys)
    ):
        raise RuntimeError("Inner selection transcript does not cover exactly 3 lanes x 3 seeds x 5 folds")
    np.savez_compressed(
        root / "nested_inner_selection_probability.npz", **inner_selection_arrays
    )
    _atomic_json(
        root / "nested_inner_selection_records.json",
        {
            "schema_version": SCHEMA_VERSION,
            "immutable": True,
            "selection_uses_outer_validation": False,
            "expected_key_count": len(expected_inner_keys),
            "records": inner_selection_records,
        },
    )
    summaries = []
    for lane in LANES:
        seed_scores = [
            _macro_f1(encoded_labels, probabilities[lane][index], len(class_names))
            for index in range(len(seeds))
        ]
        summaries.append(
            {
                "strategy": lane,
                "seed_macro_f1": dict(zip(map(str, seeds), seed_scores)),
                "mean_macro_f1": float(np.mean(seed_scores)),
                "std_macro_f1": float(np.std(seed_scores, ddof=1)),
                "minimum_seed_macro_f1": float(np.min(seed_scores)),
            }
        )
    _atomic_json(
        root / "nested_cv_metrics.json",
        {
            "schema_version": SCHEMA_VERSION,
            "primary_strategy": "primary",
            "outer_evaluations": [row for row in evaluations if row["strategy"] == "primary"],
            "candidate_outer_evaluations": evaluations,
            "strategy_summary": summaries,
            "objective": "mean_nested_oof_macro_f1",
            "gap_used_in_objective": False,
        },
    )
    _atomic_json(root / "nested_selection_records.json", {"selections": selections})
    source_hashes_after = _assert_source_unchanged(
        source_paths, source_hashes_before, audit
    )
    audit_payload = audit.payload()
    if audit_payload["test_accessed"] or audit_payload["submission_reads"]:
        raise RuntimeError("nested Test/submission isolation invariant failed")
    _atomic_json(root / "nested_io_audit.json", audit_payload)
    bounded_contract = {
        "trial_seeds": list(universe.trial_seeds),
        "trial_seed_generation_is_label_independent": True,
        "trial_seed_source": "search_universe_exact_match_to_goal_search",
        "trial_count_per_seed": universe.trial_count_per_seed,
        "requested_trial_count_total": len(universe.trial_seeds)
        * universe.trial_count_per_seed,
        "min_active_weight": universe.min_active_weight,
        "max_model_weight": universe.max_model_weight,
        "max_models": universe.max_models,
        "early_stop_patience": universe.early_stop_patience,
        "patience_uses_inner_selection_rows_only": True,
        "role_balanced_fraction": universe.role_balanced_fraction,
        "unconstrained_fraction": 1.0 - universe.role_balanced_fraction,
        "robustness_candidate_cap": universe.robustness_candidate_cap,
        "collision_candidate_cap": universe.collision_candidate_cap,
        "peak_materialized_base_variants": 1,
        "exhaustive_subset_temperature_upper_bound": (
            _exhaustive_subset_temperature_upper_bound(universe)
        ),
    }
    if any(row["model_seed"] != row["seed"] for row in seed_matrix):
        raise RuntimeError("model seed matrix contains a non-outer-seed model seed")
    expected_inner_seed_rows = {
        (seed, fold, inner_fold, alias)
        for seed in seeds
        for fold in range(EXPECTED_OUTER_FOLDS)
        for inner_fold in range(EXPECTED_INNER_FOLDS)
        for alias in universe.executable_aliases
    }
    observed_inner_seed_rows = {
        (row["seed"], row["outer_fold"], row["inner_fold"], row["alias"])
        for row in seed_matrix
        if row["stage"] == "inner"
    }
    if observed_inner_seed_rows != expected_inner_seed_rows:
        raise RuntimeError("model seed matrix inner stage/alias/fold coverage is incomplete")
    artifact_hashes = {
        name: _sha256(root / name)
        for name in OWNED_OUTPUTS
        if name != "nested_run_manifest.json" and (root / name).is_file()
    }
    trusted_input_hashes = {
        "goal_config": goal_hash,
        "search_universe": universe_hash,
        "candidate_configs": candidate_hashes,
        "train": train_hash,
        "fold_assignments": fold_hash,
        "sources": source_hashes_before,
        "source_bundle": _canonical_hash(source_hashes_before),
    }
    run_identity_payload = {
        "trusted_inputs": trusted_input_hashes,
        "bounded_search_contract": bounded_contract,
        "inner_selection_probability_artifact": artifact_hashes[
            "nested_inner_selection_probability.npz"
        ],
        "inner_selection_records_artifact": artifact_hashes[
            "nested_inner_selection_records.json"
        ],
        "inner_selection_array_hashes": {
            record["array_key"]: record["array_hash"] for record in inner_selection_records
        },
    }
    _atomic_json(
        root / "nested_run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "goal_config": str(goal_path),
            "goal_config_hash": goal_hash,
            "search_universe": str(universe_path),
            "search_universe_hash": universe_hash,
            "search_universe_provenance": universe.provenance,
            "preselected_structure_input_rejected": True,
            "class_names": list(class_names),
            "train_hash": train_hash,
            "fold_hash": fold_hash,
            "fold_assignments_path": str(fold_path),
            "seeds": list(seeds),
            "model_seed_matrix": seed_matrix,
            "model_seed_matrix_contract": {
                "model_seed_equals_outer_seed": True,
                "inner_split_seed_formula": "(outer_seed * 1009 + outer_fold * 9173) mod (2^31-1)",
                "expected_inner_stage_rows": len(expected_inner_seed_rows),
                "observed_inner_stage_rows": len(observed_inner_seed_rows),
                "stage_alias_outer_inner_coverage_complete": True,
            },
            "bounded_search_contract": bounded_contract,
            "search_diagnostics_by_outer_fold": search_diagnostics,
            "outer_folds": EXPECTED_OUTER_FOLDS,
            "inner_folds": EXPECTED_INNER_FOLDS,
            "candidate_config_hashes": candidate_hashes,
            "runtime_execution_safety": {
                "policy": "exclude_subprocess_backed_candidates_without_os_filesystem_isolation",
                "label_independent": True,
                "declared_aliases": list(universe.aliases),
                "executable_aliases": list(universe.executable_aliases),
                "runtime_exclusions": universe.runtime_exclusions,
                "excluded_aliases_never_fit": all(
                    row["alias"] not in universe.runtime_exclusions for row in seed_matrix
                ),
            },
            "source_hashes_before": source_hashes_before,
            "source_hashes_after": source_hashes_after,
            "source_bundle_before": _canonical_hash(source_hashes_before),
            "source_bundle_after": _canonical_hash(source_hashes_after),
            "source_unchanged": True,
            "trusted_input_hashes": trusted_input_hashes,
            "artifact_hashes": artifact_hashes,
            "inner_selection_transcript": {
                "key_count": len(inner_selection_records),
                "expected_key_count": 45,
                "immutable": True,
                "outer_validation_used": False,
                "probability_artifact_hash": artifact_hashes[
                    "nested_inner_selection_probability.npz"
                ],
                "records_artifact_hash": artifact_hashes[
                    "nested_inner_selection_records.json"
                ],
            },
            "io_audit": {
                "artifact_path": "nested_io_audit.json",
                "artifact_hash": artifact_hashes["nested_io_audit.json"],
                "events": audit_payload["events"],
                "test_reads": audit_payload["test_reads"],
                "submission_reads": audit_payload["submission_reads"],
            },
            "selection_source": "outer_train_inner_oof_only",
            "outer_validation_used_for_selection": False,
            "test_accessed": False,
            "run_identity_payload": run_identity_payload,
            "run_identity_hash": _canonical_hash(run_identity_payload),
        },
    )
    return root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="TEST_007 goal YAML")
    parser.add_argument(
        "--search-universe",
        "--strategies",
        dest="strategies",
        type=Path,
        required=True,
        help="Label-independent model/role/search-grid YAML (preselected top-3 is rejected)",
    )
    parser.add_argument("--fold-assignments", type=Path, required=True)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("data/processed/train_only_specialization")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = run_nested(
        args.config,
        args.strategies,
        args.fold_assignments,
        args.artifact_root,
        args.overwrite,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
