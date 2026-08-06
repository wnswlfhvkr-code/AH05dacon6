"""Evaluate completion of the train-only TEST_007 specialization workflow.

The evaluator never opens canonical Test/sample-submission data. It independently
validates frozen evidence, inference checkpoints, and the three generated CSVs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_TEST = REPO_ROOT / "data" / "raw" / "test.csv"
CANONICAL_SUBMISSION = REPO_ROOT / "data" / "raw" / "sample_submission.csv"
CANONICAL_TRAIN = REPO_ROOT / "data" / "raw" / "train.csv"


EXPECTED_SEEDS = (42, 2026, 777)
EXPECTED_FOLDS = tuple(range(5))
EXPECTED_ROWS = 6201
EXPECTED_CLASSES = 26
MIN_PROTECTED_FILES = 69
FINAL_ROLES = {
    "primary",
    "robustness_challenger",
    "specialization_challenger",
}
LANE_CANDIDATE_NAMES = {
    "primary": "nested_primary",
    "robustness_challenger": "nested_robustness",
    "specialization_challenger": "nested_specialization",
}
FINALIZER_SOURCE_FILE = "src/test_007/finalize_test_007_specialization.py"
EXPECTED_NESTED_RAW = {
    "nested_oof_probability.npy", "nested_strategy_probability.npz",
    "nested_inner_selection_probability.npz", "nested_inner_selection_records.json",
    "nested_cv_metrics.json", "nested_selection_records.json", "nested_io_audit.json",
    "nested_run_manifest.json",
}


def _same_file(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def _trusted_forbidden() -> tuple[Path, Path]:
    return CANONICAL_TEST.resolve(), CANONICAL_SUBMISSION.resolve()


def _guard_not_forbidden(path: Path, forbidden: tuple[Path, ...], role: str) -> Path:
    resolved = path.resolve()
    if any(_same_file(resolved, blocked.resolve()) for blocked in forbidden):
        raise ValueError(f"{role} aliases canonical Test/sample_submission")
    return resolved


def _guard_artifact_path(path: Path) -> Path:
    return _guard_not_forbidden(path, _trusted_forbidden(), "artifact path")


def _sha256(path: Path) -> str:
    path = _guard_artifact_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validated_inference_sources(
    frozen: dict[str, Any], raw_manifest: dict[str, Any]
) -> dict[str, str]:
    trusted = raw_manifest.get("trusted_input_hashes")
    nested_sources = trusted.get("sources") if isinstance(trusted, dict) else None
    nested_bundle = trusted.get("source_bundle") if isinstance(trusted, dict) else None
    if not isinstance(nested_sources, dict) or not nested_sources:
        raise ValueError("nested trusted inference source inventory is missing")
    if nested_bundle != _canonical_hash(nested_sources):
        raise ValueError("nested trusted inference source bundle differs")
    if (frozen.get("inference_source_base_files") != nested_sources or
            frozen.get("inference_source_base_bundle") != nested_bundle):
        raise ValueError("frozen inference source base differs from nested manifest")
    inference_sources = frozen.get("inference_source_files")
    if not isinstance(inference_sources, dict):
        raise ValueError("frozen production inference source inventory is incomplete")
    live_sources = dict(nested_sources)
    forbidden = _trusted_forbidden()
    for relative, digest in nested_sources.items():
        candidate = Path(relative)
        source = candidate if candidate.is_absolute() else REPO_ROOT / candidate
        source = _guard_not_forbidden(source, forbidden, f"nested inference source {relative}")
        if not source.is_file() or _sha256(source) != digest:
            raise ValueError(f"nested trusted inference source differs: {relative}")
    finalizer_source = _guard_not_forbidden(
        REPO_ROOT / FINALIZER_SOURCE_FILE, forbidden, "production finalizer source"
    )
    live_sources[FINALIZER_SOURCE_FILE] = _sha256(finalizer_source)
    if inference_sources != live_sources:
        raise ValueError("frozen production inference source inventory differs")
    if frozen.get("inference_source_hash") != _canonical_hash(inference_sources):
        raise ValueError("frozen production inference source inventory hash differs")
    return inference_sources


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class Evaluation:
    artifact_root: Path
    checks: list[Check] = field(default_factory=list)

    def record(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(Check(name=name, passed=bool(passed), detail=str(detail)))

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    def report(self, require_complete: bool) -> dict[str, Any]:
        failures = [check.name for check in self.checks if not check.passed]
        return {
            "schema_version": 1,
            "status": "PASS" if self.passed else "FAIL",
            "complete": self.passed,
            "require_complete": require_complete,
            "artifact_root": str(self.artifact_root),
            "checks": [check.__dict__ for check in self.checks],
            "failed_checks": failures,
            "summary": {
                "passed": sum(check.passed for check in self.checks),
                "failed": len(failures),
                "total": len(self.checks),
            },
        }


def _read_json(path: Path) -> Any:
    path = _guard_artifact_path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_text(path: Path) -> str:
    return _guard_artifact_path(path).read_text(encoding="utf-8")


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(_guard_artifact_path(path), **kwargs)


def _load_array(path: Path) -> np.ndarray:
    return np.load(_guard_artifact_path(path), allow_pickle=False)


def _within_root(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"artifact path escapes artifact root: {relative}") from error
    return _guard_artifact_path(candidate)


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-fA-F]{64}", value))


def _integrity_payload(path: Path) -> dict[str, Any]:
    """Validate an unkeyed corruption checksum; this is not forgery protection."""
    envelope = _read_json(path)
    payload = envelope.get("payload") if isinstance(envelope, dict) else None
    if not isinstance(payload, dict) or envelope.get("sha256") != _canonical_hash(payload):
        raise ValueError(f"integrity receipt checksum mismatch: {path.name}")
    return payload


def _checkpoint_name(alias: str, seed: int) -> str:
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in alias)
    return f"{safe}__seed_{seed}"


def _variant_aliases(variant: dict[str, Any]) -> set[str]:
    if not isinstance(variant, dict) or variant.get("kind") not in {"single", "blend", "collision_gate"}:
        raise ValueError("frozen selected variant schema is invalid")
    if variant["kind"] == "collision_gate":
        aliases = set(map(str, variant.get("backbone_variant", {}).get("models", [])))
        aliases.add(str(variant.get("expert", "")))
    else:
        aliases = set(map(str, variant.get("models", [])))
    if not aliases or "" in aliases:
        raise ValueError("frozen selected variant aliases are invalid")
    return aliases


def _temperature(probability: np.ndarray, value: float) -> np.ndarray:
    if not _finite_number(value) or value <= 0:
        raise ValueError("frozen temperature must be positive and finite")
    logits = np.log(np.clip(probability, 1e-12, 1.0)) / float(value)
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def _apply_base_variant(variant: dict[str, Any], probabilities: dict[str, np.ndarray]) -> np.ndarray:
    models, weights, temperatures = variant.get("models"), variant.get("weights"), variant.get("temperatures")
    if not isinstance(models, list) or not models or not isinstance(weights, list) or not isinstance(temperatures, list) or not (len(models) == len(weights) == len(temperatures)):
        raise ValueError("frozen base variant vectors are invalid")
    result = None
    for alias, weight, temperature in zip(models, weights, temperatures):
        if alias not in probabilities or not _finite_number(weight) or weight < 0:
            raise ValueError("frozen base variant alias/weight is invalid")
        calibrated = _temperature(probabilities[alias], float(temperature))
        result = float(weight) * calibrated if result is None else result + float(weight) * calibrated
    if result is None or (result.sum(axis=1) <= 0).any():
        raise ValueError("frozen base variant has zero probability mass")
    return result / result.sum(axis=1, keepdims=True)


def _apply_frozen_variant(variant: dict[str, Any], probabilities: dict[str, np.ndarray]) -> np.ndarray:
    if variant.get("kind") != "collision_gate":
        return _apply_base_variant(variant, probabilities)
    backbone = _apply_base_variant(variant.get("backbone_variant", {}), probabilities)
    expert_alias = str(variant.get("expert", ""))
    if expert_alias not in probabilities:
        raise ValueError("frozen collision expert is missing")
    expert = _temperature(probabilities[expert_alias], float(variant.get("expert_temperature")))
    source, target = variant.get("ordered_pair", (None, None))
    order = np.argsort(backbone, axis=1)[:, -2:][:, ::-1]
    margin = backbone[np.arange(len(backbone)), order[:, 0]] - backbone[np.arange(len(backbone)), order[:, 1]]
    weight, global_weight = float(variant.get("expert_weight")), float(variant.get("global_weight"))
    if not 0.10 <= weight <= 0.30 or not 0.70 <= global_weight <= 0.90 or not np.isclose(weight + global_weight, 1.0, atol=1e-12):
        raise ValueError("frozen collision gate weights are invalid")
    gate = ((order[:, 0] == target) & (order[:, 1] == source) &
            (margin <= float(variant.get("margin_threshold"))) &
            (expert.max(axis=1) >= float(variant.get("expert_confidence_threshold"))))
    result = backbone.copy(); result[gate] = (1.0 - weight) * backbone[gate] + weight * expert[gate]
    return result / result.sum(axis=1, keepdims=True)


def _selected_aliases(frozen: dict[str, Any]) -> set[str]:
    rows = frozen.get("selected_variants")
    expected = {(role, seed, fold) for role in FINAL_ROLES for seed in EXPECTED_SEEDS for fold in EXPECTED_FOLDS}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("frozen selected variants must cover 3 roles x 3 seeds x 5 folds")
    seen, aliases = set(), set()
    for row in rows:
        key = (str(row.get("strategy")), int(row.get("seed", -1)), int(row.get("fold", -1)))
        if key in seen or key not in expected:
            raise ValueError("frozen selected variant coverage is invalid")
        seen.add(key); aliases.update(_variant_aliases(row.get("variant")))
    return aliases


def _role_to_lane(frozen: dict[str, Any]) -> dict[str, str]:
    selections, mapping = frozen.get("selections"), frozen.get("role_to_lane")
    if not isinstance(selections, dict) or set(selections) != FINAL_ROLES:
        raise ValueError("frozen selections must contain exactly the final roles")
    if not isinstance(mapping, dict) or set(mapping) != FINAL_ROLES:
        raise ValueError("frozen role_to_lane must contain exactly the final roles")
    if set(mapping.values()) != FINAL_ROLES:
        raise ValueError("frozen role_to_lane must be a lane bijection")
    for role, lane in mapping.items():
        if selections[role] != LANE_CANDIDATE_NAMES.get(str(lane)):
            raise ValueError("frozen role_to_lane differs from selection candidate names")
    return {str(role): str(lane) for role, lane in mapping.items()}


def _run_identity(freeze_hash: str, candidate: dict[str, Any], seed: int) -> dict[str, Any]:
    run = candidate["runs"][str(seed)]
    environment = {"library_versions": run.get("library_versions"), "external_runtime_snapshot": run.get("external_runtime_snapshot")}
    required = {
        "run_manifest_hash": run.get("run_manifest_hash"), "config_hash": candidate.get("config_hash"),
        "bundle_hash": run.get("artifact_bundle_hash"), "train_hash": run.get("train_data_hash"),
        "class_hash": run.get("class_names_hash"), "source_hash": run.get("source_hash"),
        "fold_hash": run.get("fold_assignments_hash"), "runtime_metrics_hash": run.get("runtime_metrics_hash"),
    }
    if any(not _valid_hash(value) for value in required.values()) or any(value is None for value in environment.values()):
        raise ValueError("frozen run identity fields are incomplete")
    return {"schema_version": 1, "freeze_hash": freeze_hash, "alias": str(candidate["name"]), "seed": seed,
            **required, "environment_hash": _canonical_hash(environment)}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_probability(array: np.ndarray, expected_shape: tuple[int, ...]) -> None:
    if array.shape != expected_shape:
        raise ValueError(f"expected probability shape {expected_shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("probability contains non-finite or non-numeric values")
    if (array < 0).any() or (array > 1.0).any():
        raise ValueError("probability is outside [0, 1]")
    if not np.allclose(array.sum(axis=-1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("probability rows do not sum to one")


def _capture(evaluation: Evaluation, name: str, validator: Callable[[], str]) -> None:
    try:
        detail = validator()
    except Exception as error:  # The report must include every failed contract surface.
        evaluation.record(name, False, f"{type(error).__name__}: {error}")
    else:
        evaluation.record(name, True, detail)


def _protected_hashes(root: Path) -> str:
    data = _read_json(root / "protected_file_hashes_before_after.json")
    files = data.get("files") if isinstance(data, dict) else None
    if data.get("unchanged") is not True or not isinstance(files, list):
        raise ValueError("unchanged=true and files[] are required")
    if len(files) < MIN_PROTECTED_FILES:
        raise ValueError(f"at least {MIN_PROTECTED_FILES} protected files are required")
    paths: set[str] = set()
    for item in files:
        path = item.get("path")
        before, after = item.get("before"), item.get("after")
        if not isinstance(path, str) or not path or path in paths:
            raise ValueError("protected paths must be non-empty and unique")
        paths.add(path)
        if item.get("unchanged") is not True or before != after:
            raise ValueError(f"protected file changed: {path}")
        if not _valid_hash(before) or not _valid_hash(after):
            raise ValueError(f"invalid before/after hash: {path}")
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
            if not candidate.is_file():
                candidate = root / path
        if not candidate.is_file():
            raise ValueError(f"protected file is missing: {path}")
        if _sha256(candidate) != after:
            raise ValueError(f"protected current hash differs: {path}")
    return f"{len(files)} protected files are byte-identical"


def _class_names(root: Path) -> tuple[list[str], str]:
    names = _read_json(root / "class_names.json")
    if not isinstance(names, list) or len(names) != EXPECTED_CLASSES:
        raise ValueError(f"class_names must contain {EXPECTED_CLASSES} entries")
    names = [str(value) for value in names]
    if len(set(names)) != EXPECTED_CLASSES or any(not value for value in names):
        raise ValueError("class_names must be unique non-empty strings")
    canonical = json.dumps(names, ensure_ascii=False, separators=(",", ":"))
    return names, canonical


def _fold_assignments(root: Path) -> pd.DataFrame:
    frame = _read_csv(root / "fold_assignments.csv")
    required = ["seed", "row_index", "ID", "fold"]
    if any(column not in frame for column in required):
        raise ValueError(f"fold_assignments.csv requires columns {required}")
    if len(frame) != len(EXPECTED_SEEDS) * EXPECTED_ROWS:
        raise ValueError("fold assignment row coverage is incomplete")
    if frame[required].isna().any().any():
        raise ValueError("fold assignments contain nulls")
    frame = frame[required].copy()
    frame["seed"] = frame["seed"].astype(int)
    frame["row_index"] = frame["row_index"].astype(int)
    frame["fold"] = frame["fold"].astype(int)
    if set(frame["seed"]) != set(EXPECTED_SEEDS):
        raise ValueError("fold assignments must cover seeds 42, 2026, 777")
    for seed in EXPECTED_SEEDS:
        seeded = frame.loc[frame["seed"] == seed]
        if len(seeded) != EXPECTED_ROWS:
            raise ValueError(f"seed {seed} does not cover {EXPECTED_ROWS} rows")
        if set(seeded["fold"]) != set(EXPECTED_FOLDS):
            raise ValueError(f"seed {seed} does not cover folds 0..4")
        if seeded["row_index"].nunique() != EXPECTED_ROWS:
            raise ValueError(f"seed {seed} row_index is not one-to-one")
        if seeded["ID"].astype(str).nunique() != EXPECTED_ROWS:
            raise ValueError(f"seed {seed} ID is not one-to-one")
        if not np.array_equal(np.sort(seeded["row_index"]), np.arange(EXPECTED_ROWS)):
            raise ValueError(f"seed {seed} row_index order domain is invalid")
    return frame.sort_values(required, kind="stable").reset_index(drop=True)


def _io_audit(root: Path) -> str:
    data = _read_json(root / "io_audit.json")
    events = data.get("events")
    summary = data.get("summary")
    if not isinstance(events, list) or not events or not isinstance(summary, dict):
        raise ValueError("io_audit requires actual events[] and summary")
    zero_fields = (
        "oof_test_reads",
        "nested_test_reads",
        "oof_submission_reads",
        "nested_submission_reads",
    )
    for key in zero_fields:
        if summary.get(key) != 0:
            raise ValueError(f"{key} must be zero")
    contaminated: list[str] = []
    for event in events:
        stage = str(event.get("stage", "")).lower()
        action = str(event.get("action", "")).lower()
        path = Path(str(event.get("path", ""))).name.lower()
        if action == "read" and ("oof" in stage or "nested" in stage):
            if path == "test.csv" or "submission" in path:
                contaminated.append(f"{stage}:{path}")
    if contaminated:
        raise ValueError(f"Test/submission read during OOF: {contaminated}")
    return f"{len(events)} audited IO events; OOF/Nested Test reads=0"


def _candidate_probabilities(
    root: Path,
    names: list[str],
    canonical_assignments: pd.DataFrame,
) -> tuple[set[str], str]:
    manifest = _read_json(root / "candidate_manifest.json")
    candidates = manifest.get("candidates") if isinstance(manifest, dict) else None
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate_manifest.json requires non-empty candidates[]")
    candidate_names: set[str] = set()
    required_alignment = ["seed", "row_index", "ID", "fold"]
    for item in candidates:
        candidate = str(item.get("name", ""))
        if not candidate or candidate in candidate_names:
            raise ValueError("candidate names must be non-empty and unique")
        candidate_names.add(candidate)
        if item.get("class_names") != names:
            raise ValueError(f"{candidate}: class order differs from class_names.json")
        probability_path = _within_root(root, str(item.get("probability_path", "")))
        alignment_path = _within_root(root, str(item.get("alignment_path", "")))
        if item.get("probability_sha256") != _sha256(probability_path):
            raise ValueError(f"{candidate}: probability SHA-256 mismatch")
        if item.get("alignment_sha256") != _sha256(alignment_path):
            raise ValueError(f"{candidate}: alignment SHA-256 mismatch")
        probability = _load_array(probability_path)
        _validate_probability(
            probability, (len(EXPECTED_SEEDS), EXPECTED_ROWS, EXPECTED_CLASSES)
        )
        alignment = _read_csv(alignment_path)
        if any(column not in alignment for column in required_alignment):
            raise ValueError(f"{candidate}: alignment columns are incomplete")
        alignment = alignment[required_alignment].copy()
        for column in ("seed", "row_index", "fold"):
            alignment[column] = alignment[column].astype(int)
        alignment = alignment.sort_values(required_alignment, kind="stable").reset_index(drop=True)
        if not alignment.equals(canonical_assignments):
            raise ValueError(f"{candidate}: ID/seed/fold alignment differs")
    return candidate_names, f"{len(candidates)} candidate probability banks are aligned"


def _nested_probability(root: Path) -> str:
    probability = _load_array(root / "nested_oof_probability.npy")
    _validate_probability(
        probability, (len(EXPECTED_SEEDS), EXPECTED_ROWS, EXPECTED_CLASSES)
    )
    return f"nested_oof_probability shape={list(probability.shape)}"


def _nested_metrics(root: Path) -> str:
    data = _read_json(root / "nested_cv_metrics.json")
    rows = data.get("outer_evaluations") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) < 15:
        raise ValueError("nested_cv_metrics requires at least 15 outer evaluations")
    coverage: set[tuple[int, int]] = set()
    for row in rows:
        seed, fold = row.get("seed"), row.get("fold")
        if seed not in EXPECTED_SEEDS or fold not in EXPECTED_FOLDS:
            raise ValueError("nested evaluation seed/fold is invalid")
        if not _finite_number(row.get("macro_f1")):
            raise ValueError("nested evaluation macro_f1 must be finite")
        coverage.add((int(seed), int(fold)))
    expected = {(seed, fold) for seed in EXPECTED_SEEDS for fold in EXPECTED_FOLDS}
    if not expected.issubset(coverage):
        raise ValueError("nested metrics do not cover all 3 seeds x 5 folds")
    return f"{len(rows)} nested outer evaluations cover 3 seeds x 5 folds"


def _slice_metrics(root: Path, class_names: list[str]) -> str:
    class_data = _read_json(root / "class_metrics.json")
    class_rows = class_data.get("classes") if isinstance(class_data, dict) else None
    if not isinstance(class_rows, list):
        raise ValueError("class_metrics.json requires classes[]")
    observed = {str(row.get("class_name")) for row in class_rows}
    if not set(class_names).issubset(observed):
        raise ValueError("class metrics do not cover all 26 classes")
    if any(not _finite_number(row.get("macro_f1")) for row in class_rows):
        raise ValueError("class macro_f1 values must be finite")

    subgroup_data = _read_json(root / "subgroup_metrics.json")
    subgroup_rows = subgroup_data.get("subgroups") if isinstance(subgroup_data, dict) else None
    if not isinstance(subgroup_rows, list) or not subgroup_rows:
        raise ValueError("subgroup_metrics.json requires non-empty subgroups[]")
    dimensions = {str(row.get("dimension")) for row in subgroup_rows}
    if not {"burden", "novelty"}.issubset(dimensions):
        raise ValueError("subgroup metrics require burden and novelty dimensions")
    if any(not _finite_number(row.get("macro_f1")) for row in subgroup_rows):
        raise ValueError("subgroup macro_f1 values must be finite")
    return f"class coverage={len(observed)}; subgroup dimensions={sorted(dimensions)}"


def _paired_comparisons(root: Path) -> str:
    data = _read_json(root / "paired_comparisons.json")
    rows = data.get("comparisons") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("paired_comparisons.json requires comparisons[]")
    numeric = ("delta", "se_seed", "se_boot", "epsilon", "ci_lower", "ci_upper")
    for row in rows:
        if not row.get("candidate") or not row.get("baseline"):
            raise ValueError("paired comparison requires candidate and baseline")
        if any(not _finite_number(row.get(key)) for key in numeric):
            raise ValueError("paired comparison statistics must be finite")
        if row["se_seed"] < 0 or row["se_boot"] < 0 or row["epsilon"] < 0:
            raise ValueError("standard errors and epsilon must be non-negative")
        if not math.isclose(row["epsilon"], max(row["se_seed"], row["se_boot"]), abs_tol=1e-12):
            raise ValueError("paired comparison epsilon must equal max(se_seed, se_boot)")
        if row["ci_lower"] > row["ci_upper"]:
            raise ValueError("paired comparison CI is reversed")
    return f"{len(rows)} paired comparisons include delta/SE/epsilon/CI"


def _collapse_constraints(root: Path) -> str:
    data = _read_json(root / "collapse_constraints.json")
    if data.get("overall_pass") is not True:
        raise ValueError("overall repeated-collapse constraint failed")
    constraints = data.get("constraints")
    if not isinstance(constraints, dict):
        raise ValueError("collapse constraints mapping is missing")
    required = {"seed", "class", "burden", "novelty"}
    if not required.issubset(constraints):
        raise ValueError("seed/class/burden/novelty constraints are required")
    for name in required:
        item = constraints[name]
        if not isinstance(item, dict) or item.get("pass") is not True:
            raise ValueError(f"{name} repeated-collapse constraint failed")
        count = item.get("repeated_collapse_count")
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"{name} repeated_collapse_count is invalid")
    candidates = data.get("by_candidate")
    if not isinstance(candidates, dict) or len(candidates) < 3:
        raise ValueError("collapse constraints require per-candidate bootstrap evidence")
    for candidate, payload in candidates.items():
        dimensions = payload.get("by_dimension") if isinstance(payload, dict) else None
        if not isinstance(dimensions, dict) or not {"seed", "class", "subgroup"}.issubset(dimensions):
            raise ValueError(f"{candidate}: collapse dimensions are incomplete")
        for dimension, detail in dimensions.items():
            evidence = detail.get("evidence") if isinstance(detail, dict) else None
            if not isinstance(evidence, dict) or not evidence:
                raise ValueError(f"{candidate}/{dimension}: bootstrap evidence is missing")
            for slice_name, row in evidence.items():
                for key in ("se_seed", "se_boot", "epsilon", "ci_lower", "ci_upper"):
                    if not _finite_number(row.get(key)):
                        raise ValueError(f"{candidate}/{slice_name}: {key} is invalid")
                if not math.isclose(row["epsilon"], max(row["se_seed"], row["se_boot"]), abs_tol=1e-12):
                    raise ValueError(f"{candidate}/{slice_name}: epsilon is not max(SE)")
                support = row.get("support")
                if not isinstance(support, dict) or set(map(int, support)) != set(EXPECTED_SEEDS) or any(not isinstance(value, int) or value < 0 for value in support.values()):
                    raise ValueError(f"{candidate}/{slice_name}: seed support is invalid")
    return "seed/class/burden/novelty repeated-collapse constraints pass"


def _ensemble_ablation(root: Path) -> str:
    data = _read_json(root / "ensemble_ablation.json")
    rows = data.get("ablations") if isinstance(data, dict) else None
    if data.get("complete") is not True or not isinstance(rows, list) or not rows:
        raise ValueError("complete ensemble ablations are required")
    for row in rows:
        if not row.get("name"):
            raise ValueError("ablation name is required")
        if not _finite_number(row.get("macro_f1")) or not _finite_number(row.get("delta")):
            raise ValueError("ablation macro_f1/delta must be finite")
    return f"{len(rows)} ensemble ablations are complete"


def _final_selection(root: Path, candidates: set[str]) -> str:
    data = _read_json(root / "final_selection.json")
    selections = data.get("selections") if isinstance(data, dict) else None
    if not isinstance(selections, dict) or set(selections) != FINAL_ROLES:
        raise ValueError("final selections must contain exactly the three required roles")
    selected = [str(selections[role]) for role in sorted(FINAL_ROLES)]
    if len(set(selected)) != 3 or any(name not in candidates for name in selected):
        raise ValueError("final selections must be distinct names in candidate_manifest")
    evidence = data.get("selection_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("selection_evidence is missing")
    expected_order = [
        "mean_nested_oof",
        "one_se_eligibility",
        "overfit_gap",
        "simpler_structure",
    ]
    if evidence.get("selection_order") != expected_order:
        raise ValueError("selection order must apply gap only after 1-SE eligibility")
    if evidence.get("gap_used_only_within_one_se") is not True:
        raise ValueError("gap use is not proven to be restricted to 1-SE ties")
    one_se = evidence.get("one_se")
    if not isinstance(one_se, dict):
        raise ValueError("one_se evidence is missing")
    for key in ("best_mean", "best_se", "threshold"):
        if not _finite_number(one_se.get(key)):
            raise ValueError(f"one_se.{key} must be finite")
    if one_se["best_se"] < 0:
        raise ValueError("one_se.best_se must be non-negative")
    if not math.isclose(
        one_se["threshold"], one_se["best_mean"] - one_se["best_se"], abs_tol=1e-12
    ):
        raise ValueError("one_se threshold does not equal best_mean - best_se")
    eligible = one_se.get("eligible_candidates")
    if not isinstance(eligible, list) or not eligible or any(name not in candidates for name in eligible):
        raise ValueError("one_se eligible_candidates are invalid")
    if not isinstance(evidence.get("evidence"), list) or not evidence["evidence"]:
        raise ValueError("human-readable 1-SE/gap evidence is required")
    return "three roles selected with documented OOF -> 1-SE -> gap -> simplicity order"


def _frozen_manifest(root: Path) -> str:
    data = _read_json(root / "frozen_manifest.json")
    required_bools = {
        "frozen": True,
        "selection_uses_test": False,
        "thresholds_frozen": True,
    }
    for key, expected in required_bools.items():
        if data.get(key) is not expected:
            raise ValueError(f"frozen_manifest.{key} must be {expected}")
    for key in ("source_hash", "train_hash", "fold_hash", "bundle_hash"):
        if not _valid_hash(data.get(key)):
            raise ValueError(f"frozen_manifest.{key} must be a SHA-256 hash")
    if data["fold_hash"] != _sha256(root / "fold_assignments.csv"):
        raise ValueError("frozen fold_hash differs from canonical root folds")
    if data["train_hash"] != _sha256(CANONICAL_TRAIN):
        raise ValueError("frozen train_hash differs from canonical Train")
    used_sources = data.get("used_source_files")
    if not isinstance(used_sources, dict) or not used_sources:
        raise ValueError("frozen used source inventory is missing")
    for relative, digest in used_sources.items():
        source = Path(relative)
        source = source if source.is_absolute() else REPO_ROOT / source
        source = _guard_not_forbidden(source, _trusted_forbidden(), f"used source {relative}")
        if not source.is_file() or _sha256(source) != digest:
            raise ValueError(f"frozen used source differs: {relative}")
    if data["source_hash"] != _canonical_hash(used_sources):
        raise ValueError("frozen source_hash differs from used source inventory")
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("frozen bundle inventory is missing")
    bundle_inventory = {}
    for candidate in candidates:
        name, runs = str(candidate.get("name", "")), candidate.get("runs")
        if not name or not isinstance(runs, dict) or set(map(int, runs)) != set(EXPECTED_SEEDS):
            raise ValueError("frozen candidate requires exactly three runs")
        bundle_inventory[name] = {}
        for seed, run in runs.items():
            manifest_path = Path(str(run.get("run_manifest_path", "")))
            manifest = _read_json(manifest_path)
            bundle = manifest_path.parent / "fold_bundles.pkl"
            if manifest.get("bundle_safe_for_infer") is not True or run.get("bundle_safe_for_infer") is not True:
                raise ValueError("frozen bundle_safe_for_infer is not true")
            if _sha256(bundle) != run.get("artifact_bundle_hash") or manifest.get("artifact_bundle_hash") != run.get("artifact_bundle_hash"):
                raise ValueError("frozen bundle inventory hash mismatch")
            bundle_inventory[name][str(seed)] = run["artifact_bundle_hash"]
    if data["bundle_hash"] != _canonical_hash(bundle_inventory):
        raise ValueError("frozen bundle_hash differs from run bundle inventory")
    relationships = {
        "selection_hash": root / "final_selection.json",
        "candidate_manifest_hash": root / "candidate_manifest.json",
        "nested_probability_hash": root / "nested_oof_probability.npy",
    }
    for key, path in relationships.items():
        if data.get(key) != _sha256(path):
            raise ValueError(f"frozen_manifest.{key} provenance mismatch")
    selected = _read_json(root / "final_selection.json").get("selections")
    if data.get("selections") != selected:
        raise ValueError("frozen selections differ from final_selection.json")
    _role_to_lane(data)
    raw_path_value = data.get("nested_raw_path")
    raw_identity = data.get("nested_raw_identity")
    if not isinstance(raw_path_value, str) or not isinstance(raw_identity, dict):
        raise ValueError("frozen nested raw provenance is missing")
    raw_path = Path(raw_path_value).resolve()
    try:
        raw_path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("nested raw path escapes artifact root") from error
    raw_manifest = raw_path / "nested_run_manifest.json"
    if data.get("nested_raw_manifest_hash") != _sha256(raw_manifest):
        raise ValueError("nested raw manifest provenance mismatch")
    if raw_identity.get("hashes", {}).get("nested_run_manifest.json") != data["nested_raw_manifest_hash"]:
        raise ValueError("nested raw identity hash inventory mismatch")
    raw_hashes = raw_identity.get("hashes")
    if not isinstance(raw_hashes, dict) or set(raw_hashes) != EXPECTED_NESTED_RAW:
        raise ValueError("nested raw identity is incomplete")
    for name, digest in raw_hashes.items():
        path = raw_path / name
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"nested raw artifact provenance mismatch: {name}")
    raw_data = _read_json(raw_manifest)
    _validated_inference_sources(data, raw_data)
    state = _read_json(root / "finalization_state.json")
    if (state.get("state") not in {"inferred", "documented"} or
            state.get("freeze_hash") != _sha256(root / "frozen_manifest.json") or
            state.get("canonical_test_read_count") != 1 or state.get("fit_after_test") != 0 or
            state.get("warm_after_test") != 0 or state.get("selection_or_tuning_after_test") != 0 or
            not isinstance(state.get("bundle_load_after_test_count"), int) or state["bundle_load_after_test_count"] < 0):
        raise ValueError("finalization state differs from frozen inference contract")
    generation = Path(str(state.get("generation", "")))
    generation_manifest = generation / "generation_manifest.json"
    if not generation_manifest.is_file() or state.get("generation_manifest_hash") != _sha256(generation_manifest):
        raise ValueError("finalization state generation provenance is invalid")
    generation_data = _read_json(generation_manifest)
    for name, digest in generation_data.get("hashes", {}).items():
        if name == "finalization_state.json":
            continue
        projected, immutable = root / name, generation / name
        if not projected.is_file() or not immutable.is_file() or _sha256(projected) != digest or _sha256(immutable) != digest:
            raise ValueError(f"generation projection provenance mismatch: {name}")
    return "selection excludes Test; thresholds and source/train/fold/bundle hashes are frozen"


def _inference_checkpoints(root: Path) -> str:
    frozen_path = root / "frozen_manifest.json"
    frozen = _read_json(frozen_path); freeze_hash = _sha256(frozen_path)
    aliases = _selected_aliases(frozen)
    role_to_lane = _role_to_lane(frozen)
    candidates = {str(candidate.get("name")): candidate for candidate in frozen.get("candidates", [])}
    if not aliases or not aliases.issubset(candidates):
        raise ValueError("selected aliases are missing from the frozen candidate inventory")
    expected_units = {(alias, seed) for alias in aliases for seed in EXPECTED_SEEDS}
    expected_stems = {_checkpoint_name(alias, seed) for alias, seed in expected_units}
    if len(expected_stems) != len(expected_units):
        raise ValueError("selected aliases collide in checkpoint filenames")
    state = _read_json(root / "finalization_state.json")
    if not isinstance(state.get("bundle_load_after_test_count"), int) or state["bundle_load_after_test_count"] < len(expected_units):
        raise ValueError("post-Test bundle load count is below completed selected-run coverage")
    test_hash, contract, class_names = state.get("test_content_hash"), state.get("test_contract"), frozen.get("class_names")
    if not _valid_hash(test_hash) or not isinstance(contract, dict) or not isinstance(contract.get("row_count"), int) or contract["row_count"] <= 0:
        raise ValueError("state Test hash/contract is invalid")
    if not isinstance(class_names, list) or class_names != _read_json(root / "class_names.json"):
        raise ValueError("frozen class order differs from canonical class_names artifact")
    class_hash = _canonical_hash(class_names)
    checkpoint_root = _guard_artifact_path(root / "inference_checkpoints")
    preflight_dir, prediction_dir = _guard_artifact_path(checkpoint_root / "preflight"), _guard_artifact_path(checkpoint_root / "predictions")
    if not preflight_dir.is_dir() or {path.name for path in preflight_dir.iterdir()} != {f"{stem}.json" for stem in expected_stems}:
        raise ValueError("preflight receipt coverage is not exact")
    if not prediction_dir.is_dir() or {path.name for path in prediction_dir.iterdir()} != {name for stem in expected_stems for name in (f"{stem}.json", f"{stem}.npz")}:
        raise ValueError("prediction checkpoint coverage is not exact")
    preflight_hashes, prediction_hashes = {}, {}
    expected_shape = (contract["row_count"], len(class_names))
    for alias, seed in sorted(expected_units):
        stem = _checkpoint_name(alias, seed)
        preflight_receipt = preflight_dir / f"{stem}.json"
        if _integrity_payload(preflight_receipt) != _run_identity(freeze_hash, candidates[alias], seed):
            raise ValueError(f"preflight run identity mismatch: {stem}")
        preflight_hashes[stem] = _sha256(preflight_receipt)
        probability_path, prediction_receipt = prediction_dir / f"{stem}.npz", prediction_dir / f"{stem}.json"
        expected_prediction_identity = {"schema_version": 1, "freeze_hash": freeze_hash, "test_hash": test_hash,
                                        "class_hash": class_hash, "alias": alias, "seed": seed,
                                        "bundle_hash": candidates[alias]["runs"][str(seed)]["artifact_bundle_hash"]}
        actual_npz_hash = _sha256(probability_path)
        if _integrity_payload(prediction_receipt) != {"identity": expected_prediction_identity, "npz_hash": actual_npz_hash}:
            raise ValueError(f"prediction receipt identity/hash mismatch: {stem}")
        with np.load(_guard_artifact_path(probability_path), allow_pickle=False) as probabilities:
            if set(probabilities.files) != {f"fold_{fold}" for fold in EXPECTED_FOLDS}:
                raise ValueError(f"prediction fold coverage mismatch: {stem}")
            for fold in EXPECTED_FOLDS:
                _validate_probability(np.asarray(probabilities[f"fold_{fold}"]), expected_shape)
        prediction_hashes[stem] = {"receipt_hash": _sha256(prediction_receipt), "npz_hash": actual_npz_hash}
    test_cache = _guard_artifact_path(checkpoint_root / "test_cache")
    if not test_cache.is_dir() or {path.name for path in test_cache.iterdir()} != {"test.csv.bytes", "receipt.json"}:
        raise ValueError("Test cache inventory is incomplete")
    test_bytes_path = _guard_artifact_path(test_cache / "test.csv.bytes")
    content = test_bytes_path.read_bytes(); content_hash = hashlib.sha256(content).hexdigest()
    expected_cache = {"schema_version": 1, "freeze_hash": freeze_hash, "content_hash": content_hash, "byte_count": len(content)}
    if content_hash != test_hash or _integrity_payload(test_cache / "receipt.json") != expected_cache:
        raise ValueError("Test cache content/receipt differs from frozen inference state")
    cached_test = pd.read_csv(io.BytesIO(content))
    if ("ID" not in cached_test.columns or len(cached_test) != contract["row_count"] or
            _canonical_hash(cached_test["ID"].astype(str).tolist()) != contract.get("id_order_hash") or
            _canonical_hash({"columns": list(cached_test.columns), "dtypes": [str(value) for value in cached_test.dtypes]}) != contract.get("schema_hash")):
        raise ValueError("cached Test bytes differ from the irreversible Test contract")
    prediction_manifest_hash = _canonical_hash(prediction_hashes)
    expected_checkpoint_payload = {
        "schema_version": 1, "freeze_hash": freeze_hash, "test_content_hash": test_hash,
        "class_hash": class_hash, "run_count": len(expected_units), "preflight_receipts": preflight_hashes,
        "predictions": prediction_hashes, "prediction_manifest_hash": prediction_manifest_hash,
    }
    checkpoint_manifest_path = checkpoint_root / "inference_checkpoint_manifest.json"
    if _integrity_payload(checkpoint_manifest_path) != expected_checkpoint_payload:
        raise ValueError("checkpoint manifest does not match independently re-derived inventory")
    final = _read_json(root / "final_submissions.json")
    if (final.get("freeze_hash") != freeze_hash or final.get("test_content_hash") != test_hash or
            final.get("checkpoint_manifest_hash") != _sha256(checkpoint_manifest_path) or
            final.get("prediction_manifest_hash") != prediction_manifest_hash):
        raise ValueError("final submission manifest checkpoint provenance mismatch")
    role_totals = {role: np.zeros(expected_shape, dtype=np.float64) for role in FINAL_ROLES}
    role_counts = {role: 0 for role in FINAL_ROLES}
    for row in frozen["selected_variants"]:
        role, seed, fold, variant = str(row["strategy"]), int(row["seed"]), int(row["fold"]), row["variant"]
        member_inputs = {}
        for alias in _variant_aliases(variant):
            probability_path = prediction_dir / f"{_checkpoint_name(alias, seed)}.npz"
            with np.load(_guard_artifact_path(probability_path), allow_pickle=False) as probabilities:
                member_inputs[alias] = np.asarray(probabilities[f"fold_{fold}"], dtype=np.float64)
        role_totals[role] += _apply_frozen_variant(variant, member_inputs); role_counts[role] += 1
    if set(role_counts.values()) != {15}:
        raise ValueError("frozen replay requires exactly 15 members per role")
    expected_labels = {
        role: np.asarray(class_names)[
            (role_totals[lane] / role_counts[lane]).argmax(axis=1)
        ].tolist()
        for role, lane in role_to_lane.items()
    }
    rows = final.get("submissions")
    if not isinstance(rows, list) or {str(row.get("role")) for row in rows} != FINAL_ROLES:
        raise ValueError("final submissions are missing replay roles")
    expected_ids = cached_test["ID"].astype(str).tolist()
    for row in rows:
        role = str(row["role"]); frame = _read_csv(_within_root(root, str(row.get("path", ""))))
        if frame["ID"].astype(str).tolist() != expected_ids or frame["SUBCLASS"].astype(str).tolist() != expected_labels[role]:
            raise ValueError(f"submission labels do not match frozen checkpoint replay: {role}")
    return f"{len(expected_units)} checkpoints replay to three exact 15-member submissions"


def _final_submissions(root: Path) -> str:
    data = _read_json(root / "final_submissions.json")
    rows = data.get("submissions") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("exactly three final submission manifest entries are required")
    if {str(row.get("role")) for row in rows} != FINAL_ROLES:
        raise ValueError("submission roles do not match final selection roles")
    expected_selections = _read_json(root / "final_selection.json")["selections"]
    frozen_hash = _sha256(root / "frozen_manifest.json")
    if data.get("freeze_hash") != frozen_hash:
        raise ValueError("submission manifest freeze hash mismatch")
    state = _read_json(root / "finalization_state.json")
    expected_contract = state.get("test_contract")
    if not isinstance(expected_contract, dict) or data.get("test_contract") != expected_contract:
        raise ValueError("submission manifest differs from irreversible first-read Test contract")
    if not isinstance(expected_contract.get("row_count"), int) or expected_contract["row_count"] <= 0 or not _valid_hash(expected_contract.get("id_order_hash")) or not _valid_hash(expected_contract.get("schema_hash")):
        raise ValueError("irreversible first-read Test contract is invalid")
    id_order: list[str] | None = None
    for row in rows:
        path = _within_root(root, str(row.get("path", "")))
        if not path.is_file():
            raise ValueError(f"submission file is missing: {path.name}")
        if not _valid_hash(row.get("sha256")):
            raise ValueError(f"submission hash is invalid: {path.name}")
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"submission content hash mismatch: {path.name}")
        if not isinstance(row.get("row_count"), int) or row["row_count"] <= 0:
            raise ValueError(f"submission row_count is invalid: {path.name}")
        if row.get("columns") != ["ID", "SUBCLASS"]:
            raise ValueError(f"submission columns are invalid: {path.name}")
        frame = _read_csv(path)
        if list(frame.columns) != ["ID", "SUBCLASS"] or len(frame) != row["row_count"]:
            raise ValueError(f"submission metadata differs from CSV: {path.name}")
        if frame.isna().any().any() or not set(frame["SUBCLASS"].astype(str)).issubset(set(_read_json(root / "class_names.json"))):
            raise ValueError(f"submission contains invalid IDs/classes: {path.name}")
        current_ids = frame["ID"].astype(str).tolist()
        if len(current_ids) != expected_contract["row_count"] or _canonical_hash(current_ids) != expected_contract["id_order_hash"]:
            raise ValueError(f"submission ID order differs from first-read Test contract: {path.name}")
        if len(set(current_ids)) != len(current_ids):
            raise ValueError(f"submission IDs are not unique: {path.name}")
        if id_order is None:
            id_order = current_ids
        elif current_ids != id_order:
            raise ValueError("submission ID order differs across roles")
        role = str(row["role"])
        if row.get("candidate") != expected_selections[role]:
            raise ValueError(f"submission candidate differs from frozen selection: {role}")
        if row.get("members") != 15:
            raise ValueError(f"submission member count must equal 15: {role}")
    return "three submission CSVs match hashes, metadata, class domain, and ID order"


def _documentation(root: Path) -> str:
    report_path = root / "final_report_ko.md"
    report = _read_text(report_path)
    if len(report.strip()) < 100 or re.search(r"[가-힣]", report) is None:
        raise ValueError("final_report_ko.md must be a substantive Korean report")
    checklist = _read_json(root / "leakage_checklist.json")
    items = checklist.get("items") if isinstance(checklist, dict) else None
    if checklist.get("complete") is not True or checklist.get("all_passed") is not True:
        raise ValueError("leakage checklist is incomplete or failed")
    if not isinstance(items, list) or not items or any(item.get("passed") is not True for item in items):
        raise ValueError("every leakage checklist item must pass")
    if checklist.get("test_used_for_selection") is not False:
        raise ValueError("leakage checklist must state test_used_for_selection=false")
    return f"Korean report and {len(items)} leakage checks are complete"


def evaluate(artifact_root: Path) -> Evaluation:
    root = artifact_root.resolve()
    result = Evaluation(artifact_root=root)
    shared: dict[str, Any] = {}

    _capture(result, "protected_file_hashes", lambda: _protected_hashes(root))

    def classes() -> str:
        shared["class_names"], canonical = _class_names(root)
        shared["class_names_canonical"] = canonical
        return "26 unique class names with fixed order"

    _capture(result, "class_names", classes)

    def assignments() -> str:
        shared["assignments"] = _fold_assignments(root)
        return "fixed outer coverage=3 seeds x 5 folds x 6201 rows"

    _capture(result, "outer_fold_coverage", assignments)
    _capture(result, "io_audit", lambda: _io_audit(root))

    def candidate_banks() -> str:
        if "class_names" not in shared or "assignments" not in shared:
            raise ValueError("class_names and fold assignments must pass first")
        shared["candidates"], detail = _candidate_probabilities(
            root, shared["class_names"], shared["assignments"]
        )
        return detail

    _capture(result, "candidate_oof_probabilities", candidate_banks)
    _capture(result, "nested_oof_probability", lambda: _nested_probability(root))
    _capture(result, "nested_cv_metrics", lambda: _nested_metrics(root))

    def slices() -> str:
        if "class_names" not in shared:
            raise ValueError("class_names must pass first")
        return _slice_metrics(root, shared["class_names"])

    _capture(result, "class_and_subgroup_metrics", slices)
    _capture(result, "paired_comparisons", lambda: _paired_comparisons(root))
    _capture(result, "collapse_constraints", lambda: _collapse_constraints(root))
    _capture(result, "ensemble_ablation", lambda: _ensemble_ablation(root))

    def selection() -> str:
        if "candidates" not in shared:
            raise ValueError("candidate probabilities must pass first")
        return _final_selection(root, shared["candidates"])

    _capture(result, "final_selection", selection)
    _capture(result, "frozen_manifest", lambda: _frozen_manifest(root))
    _capture(result, "inference_checkpoints", lambda: _inference_checkpoints(root))
    _capture(result, "final_submissions", lambda: _final_submissions(root))
    _capture(result, "documentation", lambda: _documentation(root))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("data/processed/train_only_specialization"),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero unless every required artifact and invariant passes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate(args.artifact_root)
    print(json.dumps(result.report(args.require_complete), ensure_ascii=False, indent=2))
    return 1 if args.require_complete and not result.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
