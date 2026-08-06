"""Transactionally validate and consolidate strict three-seed TEST_007 OOF evidence."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SEEDS = (42, 2026, 777)
EXPECTED_ROWS = 6201
EXPECTED_CLASSES = 26
EXPECTED_FOLDS = tuple(range(5))
PROTECTED_FILES = (
    "src/train.py",
    "src/pipelines/jyp_preprocessing/pipeline_pipe_comb_v3.py",
    "src/pipelines/preprocessing_registry.py",
)
REQUIRED_ARTIFACTS = {
    "oof_prediction_by_seed.npy",
    "oof_probability_by_seed.npy",
    "fold_assignments.csv",
    "fold_metrics.csv",
    "class_names.json",
    "runtime_metrics.json",
    "model_passport.json",
    "io_audit.json",
    "fold_bundles.pkl",
}
RUNTIME_DISTRIBUTIONS = {
    "numpy": ("numpy",), "pandas": ("pandas",), "scipy": ("scipy",),
    "scikit-learn": ("scikit-learn",), "torch": ("torch",),
    "xgboost": ("xgboost",), "lightgbm": ("lightgbm",),
    "catboost": ("catboost",), "tabm": ("tabm",), "tabpfn": ("tabpfn",),
    "tabicl": ("tabicl",), "tabfm": ("tabfm",), "pytabkit": ("pytabkit",),
    "faiss-cpu": ("faiss-cpu",), "xrfm": ("xrfm",),
    "TALENT": ("TALENT", "talent"), "autogluon.tabular": ("autogluon.tabular",),
}
ANALYSIS_STAGES = ("screen", "three-seed", "search", "nested")
SAFE_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Kept as a small public test/helper surface; production reads use EvidenceReader.
def _sha256(path: Path) -> str:
    return _raw_sha256(path)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _safe_slug(value: Any, field: str) -> str:
    value = str(value)
    if value in {".", ".."} or SAFE_SLUG.fullmatch(value) is None:
        raise ValueError(f"unsafe {field}: {value!r}")
    return value


def _within(root: Path, path: Path) -> Path:
    root, path = root.resolve(), path.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes allowed root: {path}") from error
    return path


def _same_file(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _control_config(path: Path, repo_root: Path) -> Path:
    """Reject direct/symlink/hardlink data aliases before parsing a control YAML."""
    resolved = _within(repo_root / "configs", path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        if resolved.stat().st_nlink > 1:
            raise RuntimeError(f"hardlinked control config is forbidden: {resolved}")
    except OSError as error:
        raise RuntimeError(f"cannot establish control-config identity: {resolved}") from error
    return resolved


def _configured_paths(repo_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    data = config.get("data", {})
    raw = _within(repo_root, repo_root / str(data.get("raw_dir", "data/raw")))
    return {
        "train": _within(repo_root, raw / str(data.get("train_file", "train.csv"))),
        "test": _within(repo_root, raw / str(data.get("test_file", "test.csv"))),
        "submission": _within(repo_root, raw / str(data.get("submission_file", "sample_submission.csv"))),
    }


class EvidenceReader:
    """Every production byte read passes forbidden identity checks and is logged."""

    def __init__(self, repo_root: Path, forbidden: Iterable[Path]) -> None:
        self.repo_root = repo_root.resolve()
        self.forbidden = tuple(path.resolve() for path in forbidden)
        self.events: list[dict[str, Any]] = []

    def guard(self, path: Path, allowed_root: Path, role: str, stage: str) -> Path:
        path = _within(allowed_root, path)
        if any(_same_file(path, blocked) for blocked in self.forbidden):
            raise RuntimeError(f"forbidden Test/submission alias blocked before read: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _record(self, path: Path, role: str, stage: str, operation: str) -> None:
        self.events.append({
            "stage": stage,
            "action": "read",
            "operation": operation,
            "role": role,
            "path": str(path),
            "evidence_kind": "verified_collector_read",
        })

    def bytes(self, path: Path, root: Path, role: str, stage: str) -> bytes:
        path = self.guard(path, root, role, stage)
        self._record(path, role, stage, "bytes")
        return path.read_bytes()

    def text(self, path: Path, root: Path, role: str, stage: str) -> str:
        return self.bytes(path, root, role, stage).decode("utf-8")

    def json(self, path: Path, root: Path, role: str, stage: str) -> dict[str, Any]:
        value = json.loads(self.text(path, root, role, stage))
        if not isinstance(value, dict):
            raise ValueError(f"JSON root must be a mapping: {path}")
        return value

    def yaml(self, path: Path, root: Path, role: str, stage: str) -> dict[str, Any]:
        value = yaml.safe_load(self.text(path, root, role, stage))
        if not isinstance(value, dict):
            raise ValueError(f"YAML root must be a mapping: {path}")
        return value

    def csv(self, path: Path, root: Path, role: str, stage: str) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.bytes(path, root, role, stage)))

    def npy(self, path: Path, root: Path, role: str, stage: str) -> np.ndarray:
        return np.load(io.BytesIO(self.bytes(path, root, role, stage)), allow_pickle=False)

    def sha256(self, path: Path, root: Path, role: str, stage: str) -> str:
        return hashlib.sha256(self.bytes(path, root, role, stage)).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    _atomic_bytes(path, buffer.getvalue())


def _integer(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{column} must contain finite integers")
    return pd.Series(values.astype(np.int64), index=frame.index)


def _normalize_alignment(frame: pd.DataFrame, expected_seed: int | None = None) -> pd.DataFrame:
    columns = ["seed", "row_index", "ID", "fold"]
    if any(column not in frame for column in columns) or frame[columns].isna().any().any():
        raise ValueError("alignment columns are incomplete")
    frame = frame[columns].copy()
    for column in ("seed", "row_index", "fold"):
        frame[column] = _integer(frame, column)
    frame["ID"] = frame.ID.astype(str)
    frame = frame.sort_values(columns, kind="stable").reset_index(drop=True)
    seeds = EXPECTED_SEEDS if expected_seed is None else (expected_seed,)
    expected_rows = EXPECTED_ROWS * len(seeds)
    if len(frame) != expected_rows or set(frame.seed) != set(seeds):
        raise ValueError("alignment seed/row coverage is incomplete")
    for seed in seeds:
        part = frame.loc[frame.seed == seed]
        if part.row_index.nunique() != EXPECTED_ROWS or part.ID.nunique() != EXPECTED_ROWS:
            raise ValueError(f"seed {seed} alignment is not one-to-one")
        if not np.array_equal(np.sort(part.row_index), np.arange(EXPECTED_ROWS)):
            raise ValueError(f"seed {seed} row_index domain is invalid")
        if set(part.fold) != set(EXPECTED_FOLDS):
            raise ValueError(f"seed {seed} fold coverage is incomplete")
    mappings = [
        frame.loc[frame.seed == seed, ["row_index", "ID"]].sort_values("row_index").reset_index(drop=True)
        for seed in seeds
    ]
    if any(not mapping.equals(mappings[0]) for mapping in mappings[1:]):
        raise ValueError("canonical row_index-to-ID mapping differs across seeds")
    return frame


def _probability(array: np.ndarray, single_seed: bool) -> np.ndarray:
    shape = (1, EXPECTED_ROWS, EXPECTED_CLASSES) if single_seed else (3, EXPECTED_ROWS, EXPECTED_CLASSES)
    if array.shape != shape:
        raise ValueError(f"OOF probability shape mismatch: {array.shape}")
    values = np.asarray(array, dtype=np.float64)
    if not np.isfinite(values).all() or (values < -1e-8).any() or (values > 1 + 1e-8).any():
        raise ValueError("OOF probability must be finite within [0,1]")
    if not np.allclose(values.sum(axis=-1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("OOF probability rows must sum to one")
    return values


def _live_library_versions(config: dict[str, Any]) -> dict[str, str]:
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
            raise RuntimeError(f"RealTabR runtime snapshot incomplete: {missing}")
    return versions


def _expected_sources(config: dict[str, Any], repo_root: Path) -> list[Path]:
    paths = [
        repo_root / "src/test_007/run_test_007_oof_bank.py",
        *(repo_root / path for path in PROTECTED_FILES),
    ]
    model = repo_root / "src/models" / f"{config['model']['name']}_model.py"
    if model.exists():
        paths.append(model)
    return paths


def _artifact(
    run_dir: Path, name: str, manifest: dict[str, Any], reader: EvidenceReader,
) -> tuple[Path, bytes]:
    artifacts = manifest.get("artifacts")
    expected = artifacts.get(name) if isinstance(artifacts, dict) else None
    if not isinstance(expected, dict) or not _valid_hash(expected.get("sha256")):
        raise ValueError(f"manifest lacks valid artifact metadata: {name}")
    path = reader.guard(run_dir / name, run_dir, f"bank_artifact:{name}", "oof_collection")
    payload = reader.bytes(path, run_dir, f"bank_artifact:{name}", "oof_collection")
    if len(payload) != expected.get("size_bytes") or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
        raise ValueError(f"artifact hash/size mismatch: {path}")
    return path, payload


def _verify_upstream_audit(
    audit: dict[str, Any], expected: dict[str, Path], candidate: str, seed: int,
) -> list[dict[str, Any]]:
    if audit.get("phase") != "oof" or audit.get("test_accessed") is not False:
        raise ValueError(f"{candidate}/{seed}: invalid OOF audit header")
    inputs = audit.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError(f"{candidate}/{seed}: empty upstream OOF audit")
    required = {"config", "train", "test", "submission", "fold_assignments"}
    if not required.issubset(inputs):
        raise ValueError(f"{candidate}/{seed}: incomplete upstream OOF audit inputs")
    events = []
    for role in sorted(required):
        item = inputs[role]
        if not isinstance(item, dict):
            raise ValueError(f"{candidate}/{seed}: malformed audit input {role}")
        path = Path(str(item.get("resolved_path", ""))).resolve()
        if path != expected[role].resolve():
            raise ValueError(f"{candidate}/{seed}: audit path mismatch for {role}")
        permitted = role in {"config", "train", "fold_assignments"}
        if item.get("allowed") is not permitted or item.get("accessed") is not permitted:
            raise ValueError(f"{candidate}/{seed}: audit allowed/accessed mismatch for {role}")
        events.append({
            "stage": "oof", "action": "read" if permitted else "not_read", "role": role,
            "path": str(path), "candidate": candidate, "seed": seed,
            "evidence_kind": "verified_upstream_declaration",
        })
    return events


def _load_run(
    alias: str, config_path: Path, config: dict[str, Any], seed: int, bank_root: Path,
    canonical: pd.DataFrame, canonical_fold_path: Path, canonical_fold_hash: str,
    reader: EvidenceReader,
) -> tuple[np.ndarray, list[str], dict[str, Any], list[dict[str, Any]]]:
    config_hash = reader.sha256(config_path, reader.repo_root, f"candidate_config:{alias}", "oof_identity")
    experiment = _safe_slug(config["project"]["experiment_name"], "experiment_name")
    paths = _configured_paths(reader.repo_root, config)
    train_hash = reader.sha256(paths["train"], reader.repo_root, "train", "oof_identity")
    source_hashes = {
        path.relative_to(reader.repo_root).as_posix(): reader.sha256(path, reader.repo_root, "source", "oof_identity")
        for path in _expected_sources(config, reader.repo_root)
    }
    live_libraries = _live_library_versions(config)
    live_external: dict[str, Any] = {}
    run_dir = _within(bank_root, bank_root / experiment / config_hash / str(seed))
    manifest = reader.json(run_dir / "run_manifest.json", run_dir, f"run_manifest:{alias}", "oof_collection")
    protected = {name: source_hashes[name] for name in PROTECTED_FILES}
    identity = {
        "schema_version": 1, "phase": "oof", "config_path": str(config_path.resolve()),
        "config_hash": config_hash, "source_hashes": source_hashes,
        "source_hash": _canonical_hash(source_hashes), "protected_hashes_before": protected,
        "train_data_hash": train_hash, "fold_assignments_input_hash": canonical_fold_hash,
        "library_versions": live_libraries, "external_runtime_snapshot": live_external,
        "seed": seed, "folds": 5, "test_accessed": False,
    }
    identity["manifest_identity_hash"] = _canonical_hash(identity)
    for key, value in identity.items():
        if manifest.get(key) != value:
            raise ValueError(f"{alias}/{seed}: live manifest identity mismatch: {key}")
    if manifest.get("score_source") != "train_only_outer_oof" or manifest.get("train_rows") != EXPECTED_ROWS:
        raise ValueError(f"{alias}/{seed}: not a complete strict OOF run")
    if manifest.get("protected_hashes_after") != protected:
        raise ValueError(f"{alias}/{seed}: protected hashes changed")
    if manifest.get("probability_mode") == "hard_only":
        raise ValueError(f"{alias}/{seed}: hard-only candidate cannot be collected")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not REQUIRED_ARTIFACTS.issubset(artifacts):
        raise ValueError(f"{alias}/{seed}: required artifact entries missing")
    artifact_payloads = {name: _artifact(run_dir, name, manifest, reader) for name in artifacts}
    for name, (_, payload) in artifact_payloads.items():
        expected = artifacts[name]
        if name.endswith(".npy"):
            array = np.load(io.BytesIO(payload), allow_pickle=False)
            if list(array.shape) != expected.get("shape") or str(array.dtype) != expected.get("dtype"):
                raise ValueError(f"{alias}/{seed}: npy manifest metadata mismatch: {name}")
    probability = _probability(np.load(io.BytesIO(artifact_payloads["oof_probability_by_seed.npy"][1]), allow_pickle=False), True)
    alignment = _normalize_alignment(pd.read_csv(io.BytesIO(artifact_payloads["fold_assignments.csv"][1])), seed)
    expected_alignment = canonical.loc[canonical.seed == seed].reset_index(drop=True)
    if not alignment.equals(expected_alignment):
        raise ValueError(f"{alias}/{seed}: alignment differs from canonical")
    classes = json.loads(artifact_payloads["class_names.json"][1].decode("utf-8"))
    if classes != manifest.get("class_names") or len(classes) != EXPECTED_CLASSES or len(set(classes)) != EXPECTED_CLASSES:
        raise ValueError(f"{alias}/{seed}: invalid class artifact")
    if artifacts["class_names.json"]["sha256"] != manifest.get("class_names_hash"):
        raise ValueError(f"{alias}/{seed}: class hash mismatch")
    if artifacts["fold_assignments.csv"]["sha256"] != manifest.get("fold_assignments_hash"):
        raise ValueError(f"{alias}/{seed}: fold hash mismatch")
    if artifacts["fold_bundles.pkl"]["sha256"] != manifest.get("artifact_bundle_hash"):
        raise ValueError(f"{alias}/{seed}: bundle hash mismatch")
    upstream = json.loads(artifact_payloads["io_audit.json"][1].decode("utf-8"))
    expected_audit = {
        "config": config_path, "train": paths["train"], "test": paths["test"],
        "submission": paths["submission"], "fold_assignments": canonical_fold_path,
    }
    upstream_events = _verify_upstream_audit(upstream, expected_audit, alias, seed)
    run_identity = {
        "seed": seed, "run_manifest_path": str(run_dir / "run_manifest.json"),
        "run_manifest_sha256": hashlib.sha256(reader.bytes(run_dir / "run_manifest.json", run_dir, "run_manifest_hash", "oof_collection")).hexdigest(),
        "config_hash": config_hash, "source_hash": identity["source_hash"], "train_hash": train_hash,
        "fold_input_hash": canonical_fold_hash, "fold_hash": manifest["fold_assignments_hash"],
        "class_hash": manifest["class_names_hash"], "bundle_hash": manifest["artifact_bundle_hash"],
        "runtime_hash": artifacts["runtime_metrics.json"]["sha256"],
        "probability_hash": artifacts["oof_probability_by_seed.npy"]["sha256"],
        "manifest_identity_hash": identity["manifest_identity_hash"],
    }
    return probability[0], classes, run_identity, upstream_events


def _audit_payload(events: list[dict[str, Any]], required_stages: Iterable[str]) -> dict[str, Any]:
    if not events:
        raise ValueError("verified audit evidence cannot be empty")
    unique, seen = [], set()
    for event in events:
        key = json.dumps(event, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key); unique.append(event)
    for stage in required_stages:
        if not any(str(event.get("stage")) == stage for event in unique):
            raise ValueError(f"invoked stage has no verified audit evidence: {stage}")
    contaminated = []
    for event in unique:
        if event.get("action") != "read":
            continue
        name = Path(str(event.get("path", ""))).name.lower()
        if name == "test.csv" or "submission" in name:
            contaminated.append(event)
    if contaminated:
        raise RuntimeError("verified audit contains Test/submission read")
    def count(stage: str, submission: bool) -> int:
        return sum(
            1 for event in unique if event.get("action") == "read" and stage in str(event.get("stage", "")).lower()
            and (("submission" in Path(str(event.get("path", ""))).name.lower()) if submission else Path(str(event.get("path", ""))).name.lower() == "test.csv")
        )
    selection_uses_test = bool(contaminated)
    return {
        "schema_version": 2, "events": unique,
        "summary": {"total_events": len(unique), "oof_test_reads": count("oof", False),
                    "nested_test_reads": count("nested", False), "oof_submission_reads": count("oof", True),
                    "nested_submission_reads": count("nested", True)},
        "selection_uses_test": selection_uses_test,
    }


def _analysis_audits(root: Path, reader: EvidenceReader) -> tuple[list[dict[str, Any]], set[str]]:
    events, invoked = [], set()
    for stage in ANALYSIS_STAGES:
        stage_dir = root / stage
        outputs = [path for path in stage_dir.rglob("*") if path.is_file() and path.name != "io_audit.json"] if stage_dir.exists() else []
        root_nested_outputs = list(root.glob("nested_*.json")) + list(root.glob("nested_*.npy")) if stage == "nested" else []
        audit_candidates = [stage_dir / "io_audit.json"]
        if stage == "nested":
            audit_candidates.append(root / "nested_io_audit.json")
        audit_path = next((path for path in audit_candidates if path.exists()), audit_candidates[0])
        if outputs or root_nested_outputs or audit_path.exists():
            invoked.add(stage)
            audit = reader.json(audit_path, root, f"{stage}_audit", "audit_collection")
            if audit.get("selection_uses_test") is not False:
                raise ValueError(f"{stage}: selection_uses_test must be false")
            summary = audit.get("summary")
            if not isinstance(summary, dict):
                raise ValueError(f"{stage}: upstream audit summary is missing")
            for key, value in summary.items():
                if ("test" in str(key).lower() or "submission" in str(key).lower() or "contaminated" in str(key).lower()) and value != 0:
                    raise RuntimeError(f"{stage}: upstream audit summary reports contamination")
            raw = audit.get("events")
            if not isinstance(raw, list) or not raw:
                raise ValueError(f"{stage}: upstream audit events are empty")
            for item in raw:
                if not isinstance(item, dict):
                    raise ValueError(f"{stage}: malformed audit event")
                event = dict(item); event["stage"] = stage
                event["evidence_kind"] = "verified_upstream_event"
                if str(event.get("action", "")).lower() != "read" or not event.get("path"):
                    raise ValueError(f"{stage}: audit event is not a concrete read")
                recorded = _within(reader.repo_root, Path(str(event["path"])))
                if any(_same_file(recorded, blocked) for blocked in reader.forbidden):
                    raise RuntimeError(f"{stage}: audit records forbidden Test/submission read")
                if event.get("blocked") is True:
                    raise RuntimeError(f"{stage}: audit contains blocked read attempt")
                events.append(event)
    return events, invoked


def _version_files(root: Path, item: dict[str, Any], reader: EvidenceReader) -> tuple[np.ndarray, pd.DataFrame]:
    probability = reader.npy(root / item["probability_path"], root, "cached_probability", "cache_validation")
    alignment = reader.csv(root / item["alignment_path"], root, "cached_alignment", "cache_validation")
    if reader.sha256(root / item["probability_path"], root, "cached_probability_hash", "cache_validation") != item.get("probability_sha256"):
        raise ValueError("cached probability hash mismatch")
    if reader.sha256(root / item["alignment_path"], root, "cached_alignment_hash", "cache_validation") != item.get("alignment_sha256"):
        raise ValueError("cached alignment hash mismatch")
    return _probability(probability, False), _normalize_alignment(alignment)


def _write_version(
    root: Path, alias: str, identity: str, probability: np.ndarray,
    alignment: pd.DataFrame, reader: EvidenceReader,
) -> tuple[Path, Path, Path]:
    base = _within(root, root / "candidates" / alias)
    base.mkdir(parents=True, exist_ok=True)
    destination = _within(root, base / identity)
    if destination.exists():
        raise RuntimeError(f"unreferenced immutable candidate version exists: {alias}/{identity}")
    staging = Path(tempfile.mkdtemp(prefix=f".{identity}.", dir=base))
    try:
        _atomic_npy(staging / "oof_probability_by_seed.npy", probability)
        _atomic_csv(staging / "alignment.csv", alignment)
        stored = reader.npy(staging / "oof_probability_by_seed.npy", staging, "staged_probability", "staging_validation")
        aligned = reader.csv(staging / "alignment.csv", staging, "staged_alignment", "staging_validation")
        if not np.array_equal(stored, probability) or not _normalize_alignment(aligned).equals(alignment):
            raise RuntimeError("staged evidence verification failed")
        os.replace(staging, destination)
    except Exception:
        if staging.exists(): shutil.rmtree(staging)
        raise
    return destination / "oof_probability_by_seed.npy", destination / "alignment.csv", destination


def _canonical_candidate_item(
    root: Path,
    alias: str,
    classes: list[str],
    identities: list[dict[str, Any]],
    identity_hash: str,
    probability_path: Path,
    alignment_path: Path,
    reader: EvidenceReader,
) -> dict[str, Any]:
    return {
        "name": alias,
        "class_names": list(classes),
        "seeds": list(EXPECTED_SEEDS),
        "probability_path": probability_path.relative_to(root).as_posix(),
        "alignment_path": alignment_path.relative_to(root).as_posix(),
        "probability_sha256": reader.sha256(
            probability_path, root, "version_probability_hash", "cache_validation"
        ),
        "alignment_sha256": reader.sha256(
            alignment_path, root, "version_alignment_hash", "cache_validation"
        ),
        "collection_identity_hash": identity_hash,
        "source_runs": [dict(identity) for identity in identities],
        "test_accessed": False,
    }


def _restore_file(path: Path, previous: bytes | None) -> None:
    if previous is None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    else:
        _atomic_bytes(path, previous)


def _remove_created_versions(root: Path, versions: list[Path]) -> None:
    for version in reversed(versions):
        version = _within(root, version)
        if version.is_dir():
            shutil.rmtree(version)
        parent = version.parent
        try:
            if parent != root and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass


def _load_goal(
    path: Path, repo_root: Path,
) -> tuple[dict[str, Any], dict[str, tuple[Path, dict[str, Any]]], list[dict[str, Any]]]:
    path = _control_config(path, repo_root)
    defaults = (repo_root / "data/raw/test.csv", repo_root / "data/raw/sample_submission.csv")
    bootstrap = EvidenceReader(repo_root, defaults)
    # Safe before custom paths are known: configs-only, non-symlink target, nlink==1.
    goal = bootstrap.yaml(path, repo_root, "goal_config", "collection_bootstrap")
    if not isinstance(goal, dict): raise ValueError("goal config must be a mapping")
    configured = _configured_paths(repo_root, goal)
    reader = EvidenceReader(repo_root, (configured["test"], configured["submission"]))
    candidates = {}
    for item in goal.get("candidates", []):
        alias = _safe_slug(item.get("alias", ""), "candidate alias")
        config_path = _control_config(repo_root / str(item.get("config", "")), repo_root)
        config = reader.yaml(config_path, repo_root, f"candidate_config:{alias}", "collection")
        if not alias or alias in candidates: raise ValueError("candidate aliases must be unique")
        candidates[alias] = (config_path, config)
    return goal, candidates, bootstrap.events + reader.events


def collect_evidence(
    goal_config_path: Path, aliases: list[str], repo_root: Path = REPO_ROOT,
    artifact_root: Path | None = None, bank_root: Path | None = None, overwrite: bool = False,
) -> Path:
    del overwrite  # Immutable version directories intentionally never overwrite.
    repo_root = repo_root.resolve()
    goal, available, config_events = _load_goal(goal_config_path, repo_root)
    configured = _configured_paths(repo_root, goal)
    forbidden = [configured["test"], configured["submission"]]
    for _, config in available.values():
        candidate_paths = _configured_paths(repo_root, config)
        forbidden.extend((candidate_paths["test"], candidate_paths["submission"]))
    reader = EvidenceReader(repo_root, forbidden)
    data = goal["data"]
    root = _within(repo_root, artifact_root or repo_root / str(data["artifact_root"]))
    bank = _within(repo_root, bank_root or repo_root / str(data["oof_bank_root"]))
    canonical_path = reader.guard(root / "fold_assignments.csv", root, "canonical_alignment", "collection")
    canonical_hash = reader.sha256(canonical_path, root, "canonical_alignment_hash", "collection")
    canonical = _normalize_alignment(reader.csv(canonical_path, root, "canonical_alignment", "collection"))
    classes = json.loads(reader.text(root / "class_names.json", root, "canonical_classes", "collection"))
    if not isinstance(classes, list) or len(classes) != EXPECTED_CLASSES or len(set(classes)) != EXPECTED_CLASSES:
        raise ValueError("invalid canonical class order")
    manifest_path = root / "candidate_manifest.json"
    audit_path = root / "io_audit.json"
    previous_manifest = reader.bytes(manifest_path, root, "candidate_manifest_snapshot", "transaction") if manifest_path.exists() else None
    previous_audit = reader.bytes(audit_path, root, "io_audit_snapshot", "transaction") if audit_path.exists() else None
    existing_items = {}
    if previous_manifest is not None:
        existing = json.loads(previous_manifest.decode("utf-8"))
        if not isinstance(existing, dict):
            raise ValueError("candidate manifest must be a mapping")
        items = existing.get("candidates")
        if not isinstance(items, list): raise ValueError("candidate manifest requires candidates[]")
        existing_items = {str(item.get("name")): item for item in items}
        if len(existing_items) != len(items) or "" in existing_items:
            raise ValueError("candidate manifest names must be non-empty and unique")
    requested = set(aliases)
    if not requested: raise ValueError("at least one candidate is required")
    all_aliases = sorted(requested | set(existing_items))
    if set(all_aliases) - set(available): raise ValueError("candidate manifest contains unknown candidates")

    final_items, upstream_events, created_versions = [], [], []
    try:
        for alias in all_aliases:
            config_path, config = available[alias]
            probabilities, identities, candidate_classes = [], [], None
            for seed in EXPECTED_SEEDS:
                probability, run_classes, identity, events = _load_run(
                    alias, config_path, config, seed, bank, canonical, canonical_path,
                    canonical_hash, reader,
                )
                probabilities.append(probability); identities.append(identity); upstream_events.extend(events)
                if candidate_classes is None: candidate_classes = run_classes
                elif candidate_classes != run_classes: raise ValueError(f"{alias}: class order differs across seeds")
            if candidate_classes != classes: raise ValueError(f"{alias}: class order differs from canonical")
            expected_probability = np.stack(probabilities)
            identity_hash = _canonical_hash({"alias": alias, "runs": identities, "alignment": canonical_hash, "classes": classes})
            expected_dir = _within(root, root / "candidates" / alias / identity_hash)
            expected_probability_path = expected_dir / "oof_probability_by_seed.npy"
            expected_alignment_path = expected_dir / "alignment.csv"
            old = existing_items.get(alias)
            reusable = False
            if old is not None:
                cached_probability, cached_alignment = _version_files(root, old, reader)
                old_probability_path = _within(root, root / str(old.get("probability_path", "")))
                old_alignment_path = _within(root, root / str(old.get("alignment_path", "")))
                reusable = (
                    old_probability_path == expected_probability_path
                    and old_alignment_path == expected_alignment_path
                    and np.array_equal(cached_probability, expected_probability)
                    and cached_alignment.equals(canonical)
                )
                if not reusable and (
                    not np.array_equal(cached_probability, expected_probability)
                    or not cached_alignment.equals(canonical)
                ):
                    raise RuntimeError(f"{alias}: cached evidence disagrees with revalidated source runs")
            if reusable:
                probability_path, alignment_path = expected_probability_path, expected_alignment_path
            else:
                probability_path, alignment_path, version_dir = _write_version(
                    root, alias, identity_hash, expected_probability, canonical, reader,
                )
                created_versions.append(version_dir)
            final_items.append(
                _canonical_candidate_item(
                    root, alias, classes, identities, identity_hash,
                    probability_path, alignment_path, reader,
                )
            )
        analysis_events, invoked = _analysis_audits(root, reader)
        all_events = config_events + reader.events + upstream_events + analysis_events
        audit = _audit_payload(all_events, {"oof", *invoked})
        candidate_manifest = {
            "schema_version": 2, "seeds": list(EXPECTED_SEEDS), "rows": EXPECTED_ROWS,
            "classes": EXPECTED_CLASSES, "selection_uses_test": audit["selection_uses_test"],
            "candidates": final_items,
        }
        _atomic_json(audit_path, audit)
        _atomic_json(manifest_path, candidate_manifest)  # Commit point: always last.
    except Exception as error:
        rollback_errors = []
        for action in (
            lambda: _remove_created_versions(root, created_versions),
            lambda: _restore_file(audit_path, previous_audit),
            lambda: _restore_file(manifest_path, previous_manifest),
        ):
            try:
                action()
            except Exception as rollback_error:  # pragma: no cover - filesystem catastrophe
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise RuntimeError(
                "evidence transaction failed and rollback was incomplete: "
                + "; ".join(str(value) for value in rollback_errors)
            ) from error
        raise
    return manifest_path


def finalize_protected(goal_config_path: Path, artifact_root: Path, repo_root: Path = REPO_ROOT) -> tuple[Path, bool]:
    repo_root = repo_root.resolve(); goal, available, config_events = _load_goal(goal_config_path, repo_root)
    configured = _configured_paths(repo_root, goal)
    forbidden = [configured["test"], configured["submission"]]
    for _, config in available.values():
        candidate_paths = _configured_paths(repo_root, config)
        forbidden.extend((candidate_paths["test"], candidate_paths["submission"]))
    reader = EvidenceReader(repo_root, forbidden)
    reader.events.extend(config_events)
    root = _within(repo_root, artifact_root); path = root / "protected_file_hashes_before_after.json"
    prior_events: list[dict[str, Any]] = []
    audit_path = root / "io_audit.json"
    if audit_path.exists():
        prior_audit = reader.json(audit_path, root, "root_io_audit", "protected_finalize")
        raw_events = prior_audit.get("events")
        if not isinstance(raw_events, list) or not raw_events:
            raise ValueError("existing root IO audit must contain verified events")
        prior_events = [dict(event) for event in raw_events if isinstance(event, dict)]
        if len(prior_events) != len(raw_events):
            raise ValueError("existing root IO audit contains malformed events")
    data = reader.json(path, root, "protected_baseline", "protected_finalize")
    before = data.get("before")
    if not isinstance(before, dict) or not before: raise ValueError("protected baseline requires before mapping")
    files, after, unchanged = [], {}, True
    for relative, item in sorted(before.items()):
        baseline = item.get("sha256") if isinstance(item, dict) else item
        if not _valid_hash(baseline): raise ValueError(f"invalid baseline hash: {relative}")
        current_path = _within(repo_root, repo_root / relative)
        current = reader.sha256(current_path, repo_root, "protected_source", "protected_finalize")
        same = current == baseline; unchanged &= same
        after[relative] = {"sha256": current, "bytes": current_path.stat().st_size}
        files.append({"path": relative, "before": baseline, "after": current, "unchanged": same})
    result = dict(data); result.update({"after": after, "files": files, "unchanged": unchanged})
    required_stages = {
        str(event.get("stage")) for event in prior_events
        if str(event.get("stage")) in {"oof", *ANALYSIS_STAGES}
    }
    _atomic_json(audit_path, _audit_payload(prior_events + reader.events, required_stages))
    _atomic_json(path, result)
    return path, unchanged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect"); collect.add_argument("--config", type=Path, required=True)
    collect.add_argument("--candidate", action="append", required=True); collect.add_argument("--artifact-root", type=Path)
    collect.add_argument("--bank-root", type=Path); collect.add_argument("--overwrite", action="store_true")
    protected = sub.add_parser("finalize-protected"); protected.add_argument("--config", type=Path, required=True)
    protected.add_argument("--artifact-root", type=Path, default=Path("data/processed/train_only_specialization"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "collect":
        output = collect_evidence(args.config, args.candidate, artifact_root=args.artifact_root, bank_root=args.bank_root, overwrite=args.overwrite)
        payload, code = {"status": "PASS", "output": str(output), "test_accessed": False}, 0
    else:
        root = args.artifact_root if args.artifact_root.is_absolute() else REPO_ROOT / args.artifact_root
        output, unchanged = finalize_protected(args.config, root)
        payload, code = {"status": "PASS" if unchanged else "FAIL", "output": str(output), "unchanged": unchanged}, 0 if unchanged else 1
    print(json.dumps(payload, ensure_ascii=False)); return code


if __name__ == "__main__": raise SystemExit(main())
