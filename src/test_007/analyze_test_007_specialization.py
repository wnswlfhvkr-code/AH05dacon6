"""Train-only analysis for the TEST_007 specialization OOF bank."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from threadpoolctl import threadpool_limits
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCKED_READ_NAMES = {"test.csv", "sample_submission.csv"}
REQUIRED_FINAL_SEEDS = (42, 2026, 777)


def _same_file_or_inode(left: Path, right: Path) -> bool:
    try:
        if left.samefile(right):
            return True
    except (FileNotFoundError, OSError):
        pass
    try:
        left_stat, right_stat = left.stat(), right.stat()
    except (FileNotFoundError, OSError):
        return False
    return (left_stat.st_dev, left_stat.st_ino) == (right_stat.st_dev, right_stat.st_ino)


def _assert_not_forbidden_path(path: Path, forbidden_paths: Iterable[Path], role: str) -> None:
    candidate = path.resolve()
    if candidate.name.lower() in BLOCKED_READ_NAMES or "submission" in candidate.name.lower():
        raise RuntimeError(f"train-only analyzer blocked forbidden {role}: {path.name}")
    for forbidden in forbidden_paths:
        if _same_file_or_inode(candidate, forbidden.resolve()):
            raise RuntimeError(f"train-only analyzer blocked {role} alias of Test/submission")


@dataclass(frozen=True)
class CandidateSpec:
    alias: str
    config_path: Path
    family: str
    expected_role: str | None


@dataclass
class Audit:
    stage: str
    events: list[dict[str, Any]] = field(default_factory=list)
    forbidden_paths: tuple[Path, ...] = ()

    def read(self, path: Path, role: str) -> None:
        _assert_not_forbidden_path(path, self.forbidden_paths, role)
        self.events.append(
            {
                "stage": self.stage,
                "action": "read",
                "role": role,
                "path": str(path.resolve()),
                "blocked": False,
            }
        )

    def payload(self) -> dict[str, Any]:
        contaminated = [event for event in self.events if event["blocked"]]
        return {
            "stage": self.stage,
            "events": self.events,
            "summary": {
                "total_reads": len(self.events),
                "test_reads": sum(Path(event["path"]).name.lower() == "test.csv" for event in self.events),
                "submission_reads": sum("submission" in Path(event["path"]).name.lower() for event in self.events),
                "contaminated_reads": len(contaminated),
            },
            "selection_uses_test": False,
        }


@dataclass(frozen=True)
class BankRun:
    alias: str
    seed: int
    family: str
    expected_role: str | None
    probabilities: np.ndarray
    alignment: pd.DataFrame
    class_names: tuple[str, ...]
    passport: dict[str, Any]
    runtime: dict[str, Any]
    fold_metrics: pd.DataFrame
    run_dir: Path


@dataclass
class Context:
    config: dict[str, Any]
    config_path: Path
    artifact_root: Path
    bank_root: Path
    train_path: Path
    specs: list[CandidateSpec]
    audit: Audit
    train: pd.DataFrame | None = None
    labels: np.ndarray | None = None
    ids: np.ndarray | None = None
    class_names: tuple[str, ...] | None = None


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_text(path: Path, text: str) -> None:
    _atomic_bytes(path, text.encode("utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


def _atomic_yaml(path: Path, value: Any) -> None:
    _atomic_text(path, yaml.safe_dump(value, allow_unicode=True, sort_keys=False))


def _read_json(path: Path, audit: Audit, role: str) -> Any:
    audit.read(path, role)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path, audit: Audit, role: str) -> pd.DataFrame:
    audit.read(path, role)
    return pd.read_csv(path)


def _load_npy(path: Path, audit: Audit, role: str) -> np.ndarray:
    audit.read(path, role)
    return np.load(path, allow_pickle=False)


def _load_context(config_path: Path, stage: str) -> Context:
    config_path = config_path.resolve()
    nearby_forbidden = tuple(
        candidate.resolve()
        for candidate in (
            config_path.parent / "test.csv",
            config_path.parent / "sample_submission.csv",
            config_path.parent / "raw" / "test.csv",
            config_path.parent / "raw" / "sample_submission.csv",
            config_path.parent / "data" / "raw" / "test.csv",
            config_path.parent / "data" / "raw" / "sample_submission.csv",
        )
    )
    default_forbidden = tuple(dict.fromkeys((
        (REPO_ROOT / "data" / "raw" / "test.csv").resolve(),
        (REPO_ROOT / "data" / "raw" / "sample_submission.csv").resolve(),
        *nearby_forbidden,
    )))
    _assert_not_forbidden_path(config_path, default_forbidden, "analysis config")
    audit = Audit(stage=stage, forbidden_paths=default_forbidden)
    audit.read(config_path, "analysis_config")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config.get("data", {})
    raw_dir = _resolve(data.get("raw_dir", "data/raw"))
    configured_forbidden = (
        _resolve(raw_dir / data.get("test_file", "test.csv")),
        _resolve(raw_dir / data.get("submission_file", "sample_submission.csv")),
    )
    audit.forbidden_paths = tuple(dict.fromkeys((*default_forbidden, *configured_forbidden)))
    _assert_not_forbidden_path(config_path, audit.forbidden_paths, "analysis config")
    train_path = _resolve(raw_dir / data.get("train_file", "train.csv"))
    _assert_not_forbidden_path(train_path, audit.forbidden_paths, "train_file")
    final_seeds = tuple(int(value) for value in config.get("validation", {}).get("final_seeds", ()))
    if final_seeds != REQUIRED_FINAL_SEEDS:
        raise ValueError(f"final_seeds must be exactly {REQUIRED_FINAL_SEEDS}")
    specs: list[CandidateSpec] = []
    aliases: set[str] = set()
    for item in config.get("candidates", []):
        alias = str(item.get("alias", ""))
        if not alias or alias in aliases:
            raise ValueError("candidate aliases must be non-empty and unique")
        aliases.add(alias)
        specs.append(
            CandidateSpec(
                alias=alias,
                config_path=_resolve(item["config"]),
                family=str(item.get("family", "unknown")),
                expected_role=(
                    None if item.get("expected_role") is None else str(item["expected_role"])
                ),
            )
        )
    if not specs:
        raise ValueError("analysis config has no strict candidates")
    return Context(
        config=config,
        config_path=config_path,
        artifact_root=_resolve(data.get("artifact_root", "data/processed/train_only_specialization")),
        bank_root=_resolve(data.get("oof_bank_root", "data/processed/oof_bank")),
        train_path=train_path,
        specs=specs,
        audit=audit,
    )


def _assert_all_seed_runs_exist(
    context: Context, specs: Iterable[CandidateSpec], seeds: Iterable[int]
) -> None:
    required = tuple(int(seed) for seed in seeds)
    if required != REQUIRED_FINAL_SEEDS:
        raise ValueError(f"bank validation requires exactly {REQUIRED_FINAL_SEEDS}")
    missing: list[str] = []
    for spec in specs:
        context.audit.read(spec.config_path, f"candidate_config_presence:{spec.alias}")
        candidate_config = yaml.safe_load(spec.config_path.read_text(encoding="utf-8"))
        experiment = str(candidate_config["project"]["experiment_name"])
        config_hash = _sha256(spec.config_path)
        for seed in required:
            manifest = context.bank_root / experiment / config_hash / str(seed) / "run_manifest.json"
            _assert_not_forbidden_path(manifest, context.audit.forbidden_paths, "OOF manifest")
            if not manifest.is_file():
                missing.append(f"{spec.alias}:{seed}")
    if missing:
        raise FileNotFoundError(f"missing required 3-seed OOF bank runs: {', '.join(missing)}")


def _load_train(context: Context) -> None:
    if context.train is not None:
        return
    frame = _read_csv(context.train_path, context.audit, "train")
    data = context.config["data"]
    target, identifier = data["target_column"], data["id_column"]
    if target not in frame or identifier not in frame:
        raise ValueError("train data is missing target or ID column")
    if frame[identifier].astype(str).duplicated().any():
        raise ValueError("train IDs must be unique")
    context.train = frame
    context.labels = frame[target].astype(str).to_numpy()
    context.ids = frame[identifier].astype(str).to_numpy()


def _validate_probability(values: np.ndarray, rows: int, classes: int) -> np.ndarray:
    if values.shape == (1, rows, classes):
        values = values[0]
    if values.shape != (rows, classes):
        raise ValueError(f"OOF probability shape mismatch: {values.shape}")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or (values < -1e-8).any() or (values > 1 + 1e-8).any():
        raise ValueError("OOF probability is not finite within [0,1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("OOF probability rows must sum to one")
    return values


def _artifact_path(run_dir: Path, manifest: dict, name: str, audit: Audit) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or name not in artifacts:
        raise ValueError(f"verified manifest is missing artifact metadata: {name}")
    path = run_dir / name
    expected = artifacts[name]
    audit.read(path, f"bank_artifact:{name}")
    if not path.is_file() or path.stat().st_size != expected.get("size_bytes"):
        raise ValueError(f"artifact size mismatch: {path}")
    if _sha256(path) != expected.get("sha256"):
        raise ValueError(f"artifact hash mismatch: {path}")
    return path


def _load_run(context: Context, spec: CandidateSpec, seed: int) -> BankRun:
    context.audit.read(spec.config_path, f"candidate_config:{spec.alias}")
    candidate_config = yaml.safe_load(spec.config_path.read_text(encoding="utf-8"))
    experiment = str(candidate_config["project"]["experiment_name"])
    config_hash = _sha256(spec.config_path)
    run_dir = context.bank_root / experiment / config_hash / str(seed)
    manifest = _read_json(run_dir / "run_manifest.json", context.audit, f"manifest:{spec.alias}")
    if manifest.get("config_hash") != config_hash or manifest.get("seed") != seed:
        raise ValueError(f"{spec.alias}: manifest config/seed mismatch")
    if manifest.get("test_accessed") is not False or manifest.get("score_source") != "train_only_outer_oof":
        raise ValueError(f"{spec.alias}: manifest is not strict train-only OOF")
    if manifest.get("probability_mode") == "hard_only":
        raise ValueError(f"{spec.alias}: hard-only candidate cannot enter probability analysis")
    before, after = manifest.get("protected_hashes_before"), manifest.get("protected_hashes_after")
    if not isinstance(before, dict) or before != after:
        raise ValueError(f"{spec.alias}: protected hashes changed")
    class_names = tuple(str(value) for value in manifest.get("class_names", []))
    if len(class_names) != 26 or len(set(class_names)) != 26:
        raise ValueError(f"{spec.alias}: invalid fixed class order")
    probability_path = _artifact_path(run_dir, manifest, "oof_probability_by_seed.npy", context.audit)
    alignment_path = _artifact_path(run_dir, manifest, "fold_assignments.csv", context.audit)
    class_path = _artifact_path(run_dir, manifest, "class_names.json", context.audit)
    passport_path = _artifact_path(run_dir, manifest, "model_passport.json", context.audit)
    runtime_path = _artifact_path(run_dir, manifest, "runtime_metrics.json", context.audit)
    fold_metrics_path = _artifact_path(run_dir, manifest, "fold_metrics.csv", context.audit)
    io_path = _artifact_path(run_dir, manifest, "io_audit.json", context.audit)
    stored_classes = tuple(str(value) for value in json.loads(class_path.read_text(encoding="utf-8")))
    if stored_classes != class_names:
        raise ValueError(f"{spec.alias}: class artifact order mismatch")
    io_audit = json.loads(io_path.read_text(encoding="utf-8"))
    if io_audit.get("test_accessed") is not False:
        raise ValueError(f"{spec.alias}: OOF io_audit records Test access")
    alignment = pd.read_csv(alignment_path)
    required = ["row_index", context.config["data"]["id_column"], "seed", "fold"]
    if any(column not in alignment for column in required):
        raise ValueError(f"{spec.alias}: alignment columns missing")
    alignment = alignment[required].copy().sort_values("row_index", kind="stable").reset_index(drop=True)
    rows = int(manifest.get("train_rows", len(alignment)))
    if len(alignment) != rows or not np.array_equal(alignment["row_index"], np.arange(rows)):
        raise ValueError(f"{spec.alias}: row alignment incomplete")
    if set(pd.to_numeric(alignment["fold"]).astype(int)) != set(range(5)):
        raise ValueError(f"{spec.alias}: fold coverage must be 0..4")
    if set(pd.to_numeric(alignment["seed"]).astype(int)) != {seed}:
        raise ValueError(f"{spec.alias}: alignment seed mismatch")
    probability = _validate_probability(np.load(probability_path, allow_pickle=False), rows, 26)
    return BankRun(
        alias=spec.alias,
        seed=seed,
        family=spec.family,
        expected_role=spec.expected_role,
        probabilities=probability,
        alignment=alignment,
        class_names=class_names,
        passport=json.loads(passport_path.read_text(encoding="utf-8")),
        runtime=json.loads(runtime_path.read_text(encoding="utf-8")),
        fold_metrics=pd.read_csv(fold_metrics_path),
        run_dir=run_dir,
    )


def _assert_fixed_alignment(context: Context, runs: Iterable[BankRun]) -> list[BankRun]:
    runs = list(runs)
    if not runs:
        raise ValueError("no OOF bank runs loaded")
    _load_train(context)
    reference_by_seed: dict[int, pd.DataFrame] = {}
    identifier = context.config["data"]["id_column"]
    for run in runs:
        if context.class_names is None:
            context.class_names = run.class_names
        elif context.class_names != run.class_names:
            raise ValueError(f"{run.alias}: class order differs across bank")
        if len(run.alignment) != len(context.ids):
            raise ValueError(f"{run.alias}: OOF rows differ from train")
        if not np.array_equal(run.alignment[identifier].astype(str), context.ids):
            raise ValueError(f"{run.alias}: ID order differs from train")
        columns = ["row_index", identifier, "seed", "fold"]
        if run.seed not in reference_by_seed:
            reference_by_seed[run.seed] = run.alignment[columns]
        elif not run.alignment[columns].equals(reference_by_seed[run.seed]):
            raise ValueError(f"{run.alias}: fixed fold/ID alignment differs")
    if set(context.labels) - set(context.class_names or ()):
        raise ValueError("train labels are absent from fixed class order")
    return runs


def _label_index(labels: np.ndarray, class_names: tuple[str, ...]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(class_names)}
    return np.asarray([lookup[value] for value in labels], dtype=np.int16)


def _macro_f1(y: np.ndarray, prediction: np.ndarray, classes: int = 26) -> float:
    scores = []
    for value in range(classes):
        true = y == value
        predicted = prediction == value
        tp = int(np.count_nonzero(true & predicted))
        fp = int(np.count_nonzero(~true & predicted))
        fn = int(np.count_nonzero(true & ~predicted))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


def _calibration(y: np.ndarray, probabilities: np.ndarray, bins: list[float]) -> dict[str, float]:
    rows = np.arange(len(y))
    true_probability = np.clip(probabilities[rows, y], 1e-15, 1.0)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[y]
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == y
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    high = confidence >= 0.8
    return {
        "logloss": float(-np.log(true_probability).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece": float(ece),
        "high_confidence_error": float((~correct[high]).mean()) if high.any() else 0.0,
    }


def _class_rows(alias: str, y: np.ndarray, prediction: np.ndarray, names: tuple[str, ...]) -> list[dict]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=np.arange(len(names)), zero_division=0
    )
    return [
        {
            "alias": alias,
            "class_index": index,
            "class_name": name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(names)
    ]


def _pair_mask(y: np.ndarray, names: tuple[str, ...], pair: tuple[str, str]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(names)}
    return np.isin(y, [lookup[pair[0]], lookup[pair[1]]])


def _pair_f1(y: np.ndarray, prediction: np.ndarray, names: tuple[str, ...], pair: tuple[str, str]) -> float:
    mask = _pair_mask(y, names, pair)
    if not mask.any():
        return float("nan")
    lookup = {name: index for index, name in enumerate(names)}
    values = [lookup[pair[0]], lookup[pair[1]]]
    scores = []
    for value in values:
        true, predicted = y[mask] == value, prediction[mask] == value
        tp = np.count_nonzero(true & predicted)
        fp = np.count_nonzero(~true & predicted)
        fn = np.count_nonzero(true & ~predicted)
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


def _ordered_collision_directions(
    y: np.ndarray,
    baseline_prediction: np.ndarray,
    names: tuple[str, ...],
    minimum_support: int = 3,
) -> list[dict[str, Any]]:
    """Derive directed collisions exclusively from a baseline's Train OOF errors."""
    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(names):
        source_mask = y == source_index
        source_support = int(source_mask.sum())
        if source_support == 0:
            continue
        targets, counts = np.unique(baseline_prediction[source_mask], return_counts=True)
        for target_index, count in zip(targets, counts):
            target_index, count = int(target_index), int(count)
            if target_index == source_index or count < minimum_support:
                continue
            rows.append(
                {
                    "source": source,
                    "target": names[target_index],
                    "source_support": source_support,
                    "confusion_support": count,
                    "confusion_rate": count / source_support,
                }
            )
    return sorted(rows, key=lambda row: (-row["confusion_support"], row["source"], row["target"]))


def _stratified_paired_bootstrap_se(
    y: np.ndarray,
    candidate_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    *,
    repeats: int,
    seed: int,
    mask: np.ndarray | None = None,
    classes: int | None = None,
) -> tuple[float, float, float]:
    """Vectorized class-stratified paired bootstrap with shared row-ID resampling."""
    selected = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not selected.any() or repeats < 2:
        return 0.0, 0.0, 0.0
    classes = classes or int(
        max(y.max(initial=0), candidate_prediction.max(initial=0), baseline_prediction.max(initial=0)) + 1
    )
    copies = 1
    for candidate_copies in (3,):
        if len(y) % candidate_copies:
            continue
        rows = len(y) // candidate_copies
        y_blocks = y.reshape(candidate_copies, rows)
        selected_blocks = selected.reshape(candidate_copies, rows)
        if all(np.array_equal(y_blocks[0], block) for block in y_blocks[1:]) and all(
            np.array_equal(selected_blocks[0], block) for block in selected_blocks[1:]
        ):
            copies = candidate_copies
            break
    rows = len(y) // copies
    y_blocks = y.reshape(copies, rows)
    candidate_blocks = candidate_prediction.reshape(copies, rows)
    baseline_blocks = baseline_prediction.reshape(copies, rows)
    selected_rows = selected.reshape(copies, rows)[0]
    candidate_confusion = np.zeros((repeats, classes, classes), dtype=np.int32)
    baseline_confusion = np.zeros_like(candidate_confusion)
    child_seeds = np.random.SeedSequence(seed).spawn(classes)
    identity = np.eye(classes, dtype=np.int8)
    powers = np.power(classes, np.arange(2 * copies, dtype=np.int64))

    def bootstrap_source(source: int) -> tuple[int, np.ndarray, np.ndarray]:
        row_indices = np.flatnonzero(selected_rows & (y_blocks[0] == source))
        if not len(row_indices):
            empty = np.zeros((repeats, classes), dtype=np.int32)
            return source, empty, empty.copy()
        states = np.concatenate(
            (
                candidate_blocks[:, row_indices].T,
                baseline_blocks[:, row_indices].T,
            ),
            axis=1,
        )
        encoded = states.astype(np.int64, copy=False) @ powers
        unique_encoded, counts = np.unique(encoded, return_counts=True)
        unique_states = np.empty((len(unique_encoded), states.shape[1]), dtype=np.int16)
        remainder = unique_encoded.copy()
        for column in range(states.shape[1]):
            unique_states[:, column] = remainder % classes
            remainder //= classes
        sampled = np.random.default_rng(child_seeds[source]).multinomial(
            len(row_indices), counts.astype(np.float64) / counts.sum(), size=repeats
        )
        candidate_coefficients = identity[unique_states[:, :copies]].sum(axis=1)
        baseline_coefficients = identity[unique_states[:, copies:]].sum(axis=1)
        return (
            source,
            sampled @ candidate_coefficients,
            sampled @ baseline_coefficients,
        )

    with ThreadPoolExecutor(max_workers=min(8, classes)) as executor:
        for source, candidate_counts, baseline_counts in executor.map(
            bootstrap_source, range(classes)
        ):
            candidate_confusion[:, source, :] = candidate_counts
            baseline_confusion[:, source, :] = baseline_counts

    def macro_from_confusion(confusion: np.ndarray) -> np.ndarray:
        true_positive = np.diagonal(confusion, axis1=1, axis2=2).astype(np.float64)
        false_positive = confusion.sum(axis=1) - true_positive
        false_negative = confusion.sum(axis=2) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros_like(true_positive),
            where=denominator > 0,
        )
        return f1.mean(axis=1)

    deltas = macro_from_confusion(candidate_confusion) - macro_from_confusion(
        baseline_confusion
    )
    return (
        float(np.std(deltas, ddof=1)),
        float(np.quantile(deltas, 0.025)),
        float(np.quantile(deltas, 0.975)),
    )


def _paired_evidence(
    seed_deltas: list[float],
    y: np.ndarray,
    candidate_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    *,
    repeats: int,
    bootstrap_seed: int,
    mask: np.ndarray | None = None,
    classes: int | None = None,
) -> dict[str, float | bool]:
    se_seed = float(np.std(seed_deltas, ddof=1) / np.sqrt(len(seed_deltas))) if len(seed_deltas) > 1 else 0.0
    se_boot, ci_lower, ci_upper = _stratified_paired_bootstrap_se(
        y,
        candidate_prediction,
        baseline_prediction,
        repeats=repeats,
        seed=bootstrap_seed,
        mask=mask,
        classes=classes,
    )
    epsilon = max(se_seed, se_boot)
    delta = float(np.mean(seed_deltas))
    return {
        "delta": delta,
        "se_seed": se_seed,
        "se_boot": se_boot,
        "epsilon": epsilon,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "noninferiority_pass": delta >= -epsilon,
    }


def _equal_blends(runs: list[BankRun], y: np.ndarray, size: int) -> pd.DataFrame:
    rows = []
    for selected in combinations(runs, size):
        probability = np.mean([run.probabilities for run in selected], axis=0)
        rows.append(
            {
                "models": "+".join(run.alias for run in selected),
                "size": size,
                "macro_f1": _macro_f1(y, probability.argmax(axis=1)),
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def _screen(context: Context) -> Path:
    seed = int(context.config["validation"].get("screening_seed", 42))
    if seed != REQUIRED_FINAL_SEEDS[0]:
        raise ValueError(f"screening_seed must be {REQUIRED_FINAL_SEEDS[0]}")
    _assert_all_seed_runs_exist(context, context.specs, REQUIRED_FINAL_SEEDS)
    runs = _assert_fixed_alignment(context, [_load_run(context, spec, seed) for spec in context.specs])
    names = context.class_names
    assert names is not None and context.labels is not None
    y = _label_index(context.labels, names)
    bins = [float(value) for value in context.config.get("subgroups", {}).get("confidence_bins", [0, .4, .6, .8, 1])]
    leaderboard, passports, class_rows, calibration_rows = [], [], [], []
    prediction_by_alias: dict[str, np.ndarray] = {}
    class_f1_by_alias: dict[str, dict[str, float]] = {}
    for run in runs:
        prediction = run.probabilities.argmax(axis=1)
        prediction_by_alias[run.alias] = prediction
        fold_scores = []
        for fold in range(5):
            mask = run.alignment["fold"].to_numpy(dtype=int) == fold
            fold_scores.append(_macro_f1(y[mask], prediction[mask]))
        score = _macro_f1(y, prediction)
        leaderboard.append(
            {
                "alias": run.alias,
                "family": run.family,
                "macro_f1": score,
                "fold_mean": float(np.mean(fold_scores)),
                "fold_std": float(np.std(fold_scores)),
                "runtime_seconds": run.runtime.get("total_seconds"),
                "probability_mode": run.passport.get("probability_mode"),
            }
        )
        rows = _class_rows(run.alias, y, prediction, names)
        class_rows.extend(rows)
        class_f1_by_alias[run.alias] = {row["class_name"]: row["f1"] for row in rows}
        calibration_rows.append({"alias": run.alias, **_calibration(y, run.probabilities, bins)})
        passports.append(
            {
                "alias": run.alias,
                "family": run.family,
                "expected_role": run.expected_role,
                "seed": seed,
                "run_dir": str(run.run_dir),
                "manifest_verified": True,
                "bank_passport": run.passport,
                "runtime": run.runtime,
                "class_names_hash": _canonical_hash(names),
                "alignment_hash": _canonical_hash(run.alignment.astype(str).to_dict("records")),
            }
        )
    leaderboard_frame = pd.DataFrame(leaderboard).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    leader_alias = str(leaderboard_frame.iloc[0]["alias"])
    leader_score = float(leaderboard_frame.iloc[0]["macro_f1"])
    leader_prediction = prediction_by_alias[leader_alias]
    collision_minimum_support = int(context.config.get("screening", {}).get("collision_minimum_support", 3))
    collision_directions = _ordered_collision_directions(
        y, leader_prediction, names, collision_minimum_support
    )
    collision_rows = []
    for run in runs:
        prediction = prediction_by_alias[run.alias]
        for direction in collision_directions:
            source, target = str(direction["source"]), str(direction["target"])
            pair = (source, target)
            pair_score = _pair_f1(y, prediction, names, pair)
            leader_pair = _pair_f1(y, leader_prediction, names, pair)
            lookup = {name: index for index, name in enumerate(names)}
            mask = y == lookup[source]
            baseline_confusion = mask & (leader_prediction == lookup[target])
            unique_rescue = baseline_confusion & (prediction == lookup[source])
            newly_harmed = mask & (leader_prediction == lookup[source]) & (prediction == lookup[target])
            collision_rows.append(
                {
                    "alias": run.alias,
                    "pair": f"{source}|{target}",
                    "source": source,
                    "target": target,
                    "support": int(mask.sum()),
                    "baseline_confusion_support": int(direction["confusion_support"]),
                    "direction_recall": float((prediction[mask] == lookup[source]).mean()) if mask.any() else np.nan,
                    "direction_confusion_rate": float((prediction[mask] == lookup[target]).mean()) if mask.any() else np.nan,
                    "unique_rescue": int(unique_rescue.sum()),
                    "newly_harmed": int(newly_harmed.sum()),
                    "net_rescue": int(unique_rescue.sum() - newly_harmed.sum()),
                    "pair_macro_f1": pair_score,
                    "delta_vs_global_leader": pair_score - leader_pair,
                    "direction_source": "train_oof_baseline_confusion",
                }
            )
    diversity_rows, synergy_rows = [], []
    for left, right in combinations(runs, 2):
        left_prediction, right_prediction = prediction_by_alias[left.alias], prediction_by_alias[right.alias]
        left_correct, right_correct = left_prediction == y, right_prediction == y
        blend = (left.probabilities + right.probabilities) / 2
        blend_score = _macro_f1(y, blend.argmax(axis=1))
        left_score = float(leaderboard_frame.loc[leaderboard_frame.alias == left.alias, "macro_f1"].iloc[0])
        right_score = float(leaderboard_frame.loc[leaderboard_frame.alias == right.alias, "macro_f1"].iloc[0])
        oracle = np.where(left_correct, left_prediction, right_prediction)
        diversity_rows.append(
            {
                "model_a": left.alias,
                "model_b": right.alias,
                "disagreement": float((left_prediction != right_prediction).mean()),
                "double_fault": float((~left_correct & ~right_correct).mean()),
                "rescue_a_over_b": int(np.count_nonzero(left_correct & ~right_correct)),
                "rescue_b_over_a": int(np.count_nonzero(right_correct & ~left_correct)),
                "unique_rescue": int(np.count_nonzero(left_correct ^ right_correct)),
                "harm": int(np.count_nonzero(~left_correct & ~right_correct)),
                "proba_corr": float(np.corrcoef(left.probabilities.ravel(), right.probabilities.ravel())[0, 1]),
                "oracle_macro_f1": _macro_f1(y, oracle),
            }
        )
        synergy_rows.append(
            {
                "model_a": left.alias,
                "model_b": right.alias,
                "equal_blend_macro_f1": blend_score,
                "best_single_macro_f1": max(left_score, right_score),
                "delta": blend_score - max(left_score, right_score),
            }
        )
    pair_blends = _equal_blends(runs, y, 2)
    triple_blends = _equal_blends(runs, y, 3) if len(runs) >= 3 else pd.DataFrame(columns=["models", "size", "macro_f1"])
    screening = context.config.get("screening", {})
    cutoff = float(screening.get("global_f1_within_best", 0.025))
    specialist_min = float(screening.get("specialist_delta_minimum", 0.05))
    rescue_min = int(screening.get("unique_rescue_minimum", 3))
    blend_min = float(screening.get("blend_gain_minimum", 0.0015))
    diversity_frame, synergy_frame = pd.DataFrame(diversity_rows), pd.DataFrame(synergy_rows)
    collision_frame = pd.DataFrame(collision_rows)
    role_map: dict[str, Any] = {"screening_seed": seed, "models": {}}
    for run in runs:
        score = float(leaderboard_frame.loc[leaderboard_frame.alias == run.alias, "macro_f1"].iloc[0])
        global_pass = score >= leader_score - cutoff
        class_delta = max(
            (class_f1_by_alias[run.alias][name] - class_f1_by_alias[leader_alias][name] for name in names),
            default=-math.inf,
        )
        class_pass = class_delta >= specialist_min
        if collision_frame.empty:
            collision_delta = -math.inf
        else:
            values = collision_frame.loc[collision_frame.alias == run.alias, "delta_vs_global_leader"]
            collision_delta = float(values.max()) if len(values) else -math.inf
        collision_pass = collision_delta >= specialist_min
        rescue_values = []
        synergy_values = []
        if run.alias != leader_alias and not diversity_frame.empty:
            row = diversity_frame.loc[
                ((diversity_frame.model_a == run.alias) & (diversity_frame.model_b == leader_alias))
                | ((diversity_frame.model_b == run.alias) & (diversity_frame.model_a == leader_alias))
            ]
            if len(row):
                item = row.iloc[0]
                rescue_values.append(
                    int(item.rescue_a_over_b if item.model_a == run.alias else item.rescue_b_over_a)
                )
            pair = synergy_frame.loc[
                ((synergy_frame.model_a == run.alias) & (synergy_frame.model_b == leader_alias))
                | ((synergy_frame.model_b == run.alias) & (synergy_frame.model_a == leader_alias))
            ]
            if len(pair):
                synergy_values.append(float(pair.iloc[0].delta))
        rescue = max(rescue_values, default=0)
        synergy = max(synergy_values, default=-math.inf)
        rescue_pass, synergy_pass = rescue >= rescue_min, synergy >= blend_min
        criteria = {
            "global_cutoff": global_pass,
            "class_specialization": class_pass,
            "collision_specialization": collision_pass,
            "unique_rescue": rescue_pass,
            "equal_50_50_synergy": synergy_pass,
        }
        survives = any(criteria.values())
        reasons = [name for name, passed in criteria.items() if passed]
        if not reasons:
            reasons = ["모든 정량 생존 기준 미달"]
        role = run.expected_role or (
            "COLLISION_EXPERT" if collision_pass else "SPECIALIST" if class_pass else "GLOBAL_CHALLENGER"
        )
        role_map["models"][run.alias] = {
            "family": run.family,
            "role": role,
            "survives": survives,
            "criteria": criteria,
            "metrics": {
                "macro_f1": score,
                "class_delta_max": class_delta,
                "collision_delta_max": collision_delta,
                "unique_rescue_vs_leader": rescue,
                "synergy_vs_leader": synergy,
            },
            "reasons": reasons,
        }
    output = context.artifact_root / "screen"
    _atomic_csv(output / "model_leaderboard.csv", leaderboard_frame)
    _atomic_json(output / "model_leaderboard.json", leaderboard_frame.to_dict("records"))
    _atomic_json(output / "model_passports.json", passports)
    _atomic_csv(output / "class_specialization.csv", pd.DataFrame(class_rows))
    _atomic_csv(output / "collision_direction_specialization.csv", collision_frame)
    _atomic_csv(output / "calibration_metrics.csv", pd.DataFrame(calibration_rows))
    _atomic_csv(output / "pairwise_diversity.csv", diversity_frame)
    _atomic_csv(output / "pairwise_synergy.csv", synergy_frame)
    _atomic_csv(output / "equal_pair_blends.csv", pair_blends)
    _atomic_csv(output / "equal_triple_blends.csv", triple_blends)
    _atomic_yaml(output / "model_role_map.yaml", role_map)
    _atomic_json(output / "io_audit.json", context.audit.payload())
    return output


def _survivor_aliases(context: Context) -> list[str]:
    path = context.artifact_root / "screen" / "model_role_map.yaml"
    context.audit.read(path, "screen_role_map")
    mapping = yaml.safe_load(path.read_text(encoding="utf-8"))
    aliases = [alias for alias, item in mapping.get("models", {}).items() if item.get("survives")]
    if not aliases:
        raise ValueError("screen stage has no surviving models")
    return aliases


def _mutation_long(context: Context) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    assert context.train is not None
    data = context.config["data"]
    raw = context.train.drop(columns=[data["target_column"], data["id_column"]])
    chunks = []
    for column in raw.columns:
        values = raw[column].astype(str)
        normalized = values.str.strip()
        mask = ~normalized.str.upper().isin({"", "WT", "NAN", "NONE", "NA"})
        if mask.any():
            token = normalized[mask]
            chunks.append(
                pd.DataFrame(
                    {
                        "row_index": token.index.to_numpy(dtype=int),
                        "gene": str(column),
                        "token": token.to_numpy(),
                        "functional": ~token.str.lower().str.contains(
                            "synonymous|silent|intron", regex=True
                        ).to_numpy(),
                    }
                )
            )
    long = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["row_index", "gene", "token", "functional"]
    )
    burden = np.bincount(long.row_index.astype(int), minlength=len(raw)).astype(float) if len(long) else np.zeros(len(raw))
    functional = (
        np.bincount(
            long.loc[long.functional, "row_index"].astype(int), minlength=len(raw)
        ).astype(float)
        if len(long)
        else np.zeros(len(raw))
    )
    functional_ratio = np.divide(functional, burden, out=np.zeros_like(functional), where=burden > 0)
    return long, burden, functional_ratio


def _groups(values: np.ndarray, train_mask: np.ndarray, quantiles: list[float]) -> tuple[np.ndarray, list[float]]:
    thresholds = [float(np.quantile(values[train_mask], value)) for value in quantiles]
    return np.digitize(values, thresholds, right=True), thresholds


def _novelty(long: pd.DataFrame, train_mask: np.ndarray, rows: int) -> np.ndarray:
    if long.empty:
        return np.zeros(rows)
    train_long = long.loc[train_mask[long.row_index.to_numpy(dtype=int)]]
    counts = train_long.groupby(["gene", "token"], sort=False).size()
    known = set(counts.index.tolist())
    rare = set(counts.loc[counts <= 1].index.tolist())
    result = np.zeros(rows)
    burden = np.zeros(rows)
    for row, gene, token in long[["row_index", "gene", "token"]].itertuples(index=False):
        row = int(row)
        burden[row] += 1
        key = (gene, token)
        if (train_mask[row] and key in rare) or (not train_mask[row] and key not in known):
            result[row] += 1
    return np.divide(result, burden, out=np.zeros_like(result), where=burden > 0)


def _three_seed(context: Context) -> Path:
    _load_train(context)
    aliases = _survivor_aliases(context)
    specs = [spec for spec in context.specs if spec.alias in aliases]
    seeds = list(REQUIRED_FINAL_SEEDS)
    _assert_all_seed_runs_exist(context, specs, seeds)
    runs = _assert_fixed_alignment(
        context, [_load_run(context, spec, seed) for spec in specs for seed in seeds]
    )
    names = context.class_names
    assert names is not None and context.labels is not None
    y = _label_index(context.labels, names)
    class_rows, fold_rows, stability_rows, subgroup_rows, threshold_rows = [], [], [], [], []
    calibration_rows: list[dict[str, Any]] = []
    long, burden, functional_ratio = _mutation_long(context)
    run_lookup = {(run.alias, run.seed): run for run in runs}
    for alias in aliases:
        seed_scores = []
        for seed in seeds:
            run = run_lookup[(alias, seed)]
            prediction = run.probabilities.argmax(axis=1)
            seed_score = _macro_f1(y, prediction)
            seed_scores.append(seed_score)
            calibration_rows.append(
                {
                    "alias": alias,
                    "seed": seed,
                    **_calibration(
                        y,
                        run.probabilities,
                        [float(value) for value in context.config.get("subgroups", {}).get(
                            "confidence_bins", [0, .4, .6, .8, 1]
                        )],
                    ),
                }
            )
            for row in _class_rows(alias, y, prediction, names):
                class_rows.append({"seed": seed, **row})
            metric_by_fold = {
                int(row.fold): row
                for row in run.fold_metrics.itertuples(index=False)
                if hasattr(row, "fold")
            }
            folds = run.alignment["fold"].to_numpy(dtype=int)
            for fold in range(5):
                valid = folds == fold
                train_mask = ~valid
                valid_score = _macro_f1(y[valid], prediction[valid])
                metric = metric_by_fold.get(fold)
                train_f1 = float(getattr(metric, "train_f1", np.nan)) if metric is not None else np.nan
                fold_rows.append(
                    {
                        "alias": alias,
                        "seed": seed,
                        "fold": fold,
                        "validation_macro_f1": valid_score,
                        "train_macro_f1": train_f1,
                        "fold_gap": train_f1 - valid_score,
                    }
                )
                novelty = _novelty(long, train_mask, len(y))
                dimensions = {
                    "burden": (burden, context.config.get("subgroups", {}).get("burden_quantiles", [.25, .75])),
                    "functional_ratio": (functional_ratio, context.config.get("subgroups", {}).get("functional_ratio_quantiles", [.25, .75])),
                    "novelty": (novelty, context.config.get("subgroups", {}).get("novelty_quantiles", [.25, .75])),
                }
                for dimension, (values, qs) in dimensions.items():
                    groups, thresholds = _groups(values, train_mask, [float(value) for value in qs])
                    threshold_rows.append(
                        {
                            "alias": alias,
                            "seed": seed,
                            "fold": fold,
                            "dimension": dimension,
                            "thresholds": json.dumps(thresholds),
                            "fit_rows": int(train_mask.sum()),
                            "threshold_fit_scope": "outer_train_only",
                        }
                    )
                    for group in sorted(np.unique(groups[valid])):
                        mask = valid & (groups == group)
                        subgroup_rows.append(
                            {
                                "alias": alias,
                                "seed": seed,
                                "fold": fold,
                                "dimension": dimension,
                                "group": int(group),
                                "support": int(mask.sum()),
                                "macro_f1": _macro_f1(y[mask], prediction[mask]) if mask.any() else np.nan,
                            }
                        )
        stability_rows.append(
            {
                "alias": alias,
                "seed_mean_macro_f1": float(np.mean(seed_scores)),
                "seed_std_macro_f1": float(np.std(seed_scores)),
                "seed_min_macro_f1": float(np.min(seed_scores)),
            }
        )
    stability = pd.DataFrame(stability_rows).sort_values("seed_mean_macro_f1", ascending=False)
    primary = str(stability.iloc[0].alias)
    bootstrap_repeats = int(context.config.get("validation", {}).get("bootstrap_repeats", 2000))
    bootstrap_seed = int(context.config.get("validation", {}).get("bootstrap_seed", 730))
    y_stacked = np.tile(y, len(seeds))
    baseline_by_seed = {
        seed: run_lookup[(primary, seed)].probabilities.argmax(axis=1) for seed in seeds
    }
    baseline_stacked = np.concatenate([baseline_by_seed[seed] for seed in seeds])
    hypothesis_rows: list[dict[str, Any]] = []
    for alias in aliases:
        candidate_by_seed = {
            seed: run_lookup[(alias, seed)].probabilities.argmax(axis=1) for seed in seeds
        }
        differences: list[float] = []
        for seed in seeds:
            differences.append(
                _macro_f1(y, candidate_by_seed[seed]) - _macro_f1(y, baseline_by_seed[seed])
            )
        evidence = _paired_evidence(
            differences,
            y_stacked,
            np.concatenate([candidate_by_seed[seed] for seed in seeds]),
            baseline_stacked,
            repeats=bootstrap_repeats,
            bootstrap_seed=bootstrap_seed + aliases.index(alias),
            classes=len(names),
        )
        hypothesis_rows.append(
            {
                "candidate": alias,
                "baseline": primary,
                "mean_delta": evidence["delta"],
                **{key: value for key, value in evidence.items() if key != "delta"},
            }
        )

    class_frame = pd.DataFrame(class_rows)
    fold_frame = pd.DataFrame(fold_rows)
    subgroup_frame = pd.DataFrame(subgroup_rows)
    calibration_frame = pd.DataFrame(calibration_rows)
    collision_seed_rows: list[dict[str, Any]] = []
    minimum_support = int(context.config.get("screening", {}).get("collision_minimum_support", 3))
    name_lookup = {name: index for index, name in enumerate(names)}
    for seed_index, seed in enumerate(seeds):
        baseline = baseline_by_seed[seed]
        directions = _ordered_collision_directions(y, baseline, names, minimum_support)
        for direction in directions:
            source, target = str(direction["source"]), str(direction["target"])
            source_index, target_index = name_lookup[source], name_lookup[target]
            source_mask = y == source_index
            pair_mask = np.isin(y, [source_index, target_index])
            baseline_confusion = source_mask & (baseline == target_index)
            for alias in aliases:
                candidate = run_lookup[(alias, seed)].probabilities.argmax(axis=1)
                unique_rescue = int(np.count_nonzero(baseline_confusion & (candidate == source_index)))
                newly_harmed = int(
                    np.count_nonzero(source_mask & (baseline == source_index) & (candidate == target_index))
                )
                folds = run_lookup[(alias, seed)].alignment["fold"].to_numpy(dtype=int)
                rescue_mask = baseline_confusion & (candidate == source_index)
                harm_mask = source_mask & (baseline == source_index) & (candidate == target_index)
                fold_evidence = {
                    str(fold): {
                        "baseline_confusion_support": int(np.count_nonzero(baseline_confusion & (folds == fold))),
                        "unique_rescue": int(np.count_nonzero(rescue_mask & (folds == fold))),
                        "newly_harmed": int(np.count_nonzero(harm_mask & (folds == fold))),
                    }
                    for fold in range(5)
                }
                collision_seed_rows.append(
                    {
                        "alias": alias,
                        "seed": seed,
                        "source": source,
                        "target": target,
                        "direction": f"{source}->{target}",
                        "source_support": int(direction["source_support"]),
                        "baseline_confusion_support": int(direction["confusion_support"]),
                        "unique_rescue": unique_rescue,
                        "unique_rescue_rows": json.dumps(np.flatnonzero(rescue_mask).astype(int).tolist()),
                        "newly_harmed": newly_harmed,
                        "net_rescue": unique_rescue - newly_harmed,
                        "fold_evidence": json.dumps(fold_evidence, sort_keys=True),
                        "positive_net_rescue_folds": int(sum(
                            item["unique_rescue"] > item["newly_harmed"]
                            for item in fold_evidence.values()
                        )),
                        "pair_delta": _pair_f1(y, candidate, names, (source, target))
                        - _pair_f1(y, baseline, names, (source, target)),
                        "outside_delta": _macro_f1(y[~pair_mask], candidate[~pair_mask])
                        - _macro_f1(y[~pair_mask], baseline[~pair_mask]),
                        "direction_source": "per_seed_train_oof_baseline_confusion",
                    }
                )
    collision_seed_frame = pd.DataFrame(collision_seed_rows)
    collision_summary_rows: list[dict[str, Any]] = []
    if not collision_seed_frame.empty:
        for (alias, source, target), group in collision_seed_frame.groupby(
            ["alias", "source", "target"], sort=False
        ):
            observed_seeds = sorted(group.seed.astype(int).unique())
            outside_deltas = []
            pair_masks, candidate_parts, baseline_parts, y_parts = [], [], [], []
            for seed in seeds:
                pair_mask = np.isin(y, [name_lookup[source], name_lookup[target]])
                candidate = run_lookup[(alias, seed)].probabilities.argmax(axis=1)
                baseline = baseline_by_seed[seed]
                outside_deltas.append(
                    _macro_f1(y[~pair_mask], candidate[~pair_mask])
                    - _macro_f1(y[~pair_mask], baseline[~pair_mask])
                )
                pair_masks.append(~pair_mask)
                candidate_parts.append(candidate)
                baseline_parts.append(baseline)
                y_parts.append(y)
            outside_evidence = _paired_evidence(
                outside_deltas,
                np.concatenate(y_parts),
                np.concatenate(candidate_parts),
                np.concatenate(baseline_parts),
                repeats=bootstrap_repeats,
                bootstrap_seed=bootstrap_seed + 1000 + len(collision_summary_rows),
                mask=np.concatenate(pair_masks),
                classes=len(names),
            )
            positive_seeds = int((group.net_rescue > 0).sum())
            outside_collapse_seed_count = int(
                sum(delta < -float(outside_evidence["epsilon"]) for delta in outside_deltas)
            )
            repeated_outside_collapse = outside_collapse_seed_count >= 2
            unique_rescue_rows = {
                int(row_index)
                for payload in group.unique_rescue_rows.astype(str)
                for row_index in json.loads(payload)
            }
            collision_summary_rows.append(
                {
                    "alias": alias,
                    "source": source,
                    "target": target,
                    "direction": f"{source}->{target}",
                    "observed_seed_count": len(observed_seeds),
                    "outside_evaluated_seed_count": len(seeds),
                    "positive_net_rescue_seeds": positive_seeds,
                    "repeated_direction_2_of_3": positive_seeds >= 2,
                    "unique_rescue_seed_total": int(group.unique_rescue.sum()),
                    "unique_rescue_count": len(unique_rescue_rows),
                    "positive_net_rescue_fold_total": int(group.positive_net_rescue_folds.sum()),
                    "net_rescue_total": int(group.net_rescue.sum()),
                    "mean_pair_delta": float(group.pair_delta.mean()),
                    "outside_delta": outside_evidence["delta"],
                    "outside_se_seed": outside_evidence["se_seed"],
                    "outside_se_boot": outside_evidence["se_boot"],
                    "outside_epsilon": outside_evidence["epsilon"],
                    "outside_noninferiority_pass": outside_evidence["noninferiority_pass"],
                    "outside_collapse_seed_count": outside_collapse_seed_count,
                    "repeated_outside_collapse": repeated_outside_collapse,
                    "eligible_collision_expert": bool(
                        positive_seeds >= 2
                        and len(unique_rescue_rows) >= 3
                        and float(group.pair_delta.mean()) >= 0.05
                        and outside_evidence["noninferiority_pass"]
                        and not repeated_outside_collapse
                    ),
                }
            )
    collision_summary = pd.DataFrame(collision_summary_rows)

    baseline_class = class_frame.loc[class_frame.alias == primary].set_index(["seed", "class_name"])
    baseline_subgroup = subgroup_frame.loc[subgroup_frame.alias == primary].set_index(
        ["seed", "fold", "dimension", "group"]
    )
    role_map: dict[str, Any] = {
        "evidence_scope": "three_seed_train_oof_only",
        "seeds": seeds,
        "baseline": primary,
        "expected_role_used": False,
        "models": {},
    }
    stability_cutoff = float(stability.seed_std_macro_f1.quantile(0.25))
    best_mean = float(stability.seed_mean_macro_f1.max())
    for alias in aliases:
        stable_row = stability.loc[stability.alias == alias].iloc[0]
        candidate_class = class_frame.loc[class_frame.alias == alias].set_index(["seed", "class_name"])
        class_delta = candidate_class.f1 - baseline_class.f1
        repeated_class_rescues = [
            class_name
            for class_name in names
            if int((class_delta.xs(class_name, level="class_name") >= 0.05).sum()) >= 2
        ]
        repeated_class_collapses = [
            class_name
            for class_name in names
            if int((class_delta.xs(class_name, level="class_name") <= -0.05).sum()) >= 2
        ]
        candidate_subgroup = subgroup_frame.loc[subgroup_frame.alias == alias].set_index(
            ["seed", "fold", "dimension", "group"]
        )
        subgroup_delta = (candidate_subgroup.macro_f1 - baseline_subgroup.macro_f1).dropna()
        subgroup_evidence: dict[str, Any] = {}
        for dimension in ("burden", "functional_ratio", "novelty"):
            dimension_delta = subgroup_delta.xs(dimension, level="dimension")
            group_means = dimension_delta.groupby("group").mean()
            for group_value in sorted(group_means.index):
                seed_means = dimension_delta.xs(group_value, level="group").groupby("seed").mean()
                subgroup_evidence[f"{dimension}:{int(group_value)}"] = {
                    "mean_delta": float(group_means.loc[group_value]),
                    "positive_seed_count": int((seed_means > 0).sum()),
                    "nonnegative_seed_count": int((seed_means >= 0).sum()),
                }
        collisions = (
            collision_summary.loc[collision_summary.alias == alias].sort_values(
                ["eligible_collision_expert", "net_rescue_total"], ascending=False
            )
            if not collision_summary.empty
            else pd.DataFrame()
        )
        qualifying_collisions = (
            collisions.loc[collisions.eligible_collision_expert].direction.astype(str).tolist()
            if not collisions.empty
            else []
        )
        calibration_delta = (
            calibration_frame.loc[calibration_frame.alias == alias, "ece"].mean()
            - calibration_frame.loc[calibration_frame.alias == primary, "ece"].mean()
        )
        diversity_seed_rows = []
        for seed in seeds:
            candidate = run_lookup[(alias, seed)].probabilities.argmax(axis=1)
            baseline = baseline_by_seed[seed]
            baseline_correct, candidate_correct = baseline == y, candidate == y
            diversity_seed_rows.append(
                {
                    "seed": seed,
                    "unique_rescue": int(np.count_nonzero(~baseline_correct & candidate_correct)),
                    "double_fault": float(np.mean(~baseline_correct & ~candidate_correct)),
                }
            )
        low_burden = subgroup_evidence.get("burden:0", {})
        high_burden_key = max(
            (key for key in subgroup_evidence if key.startswith("burden:")),
            key=lambda key: int(key.split(":")[1]),
            default="burden:0",
        )
        high_novelty_key = max(
            (key for key in subgroup_evidence if key.startswith("novelty:")),
            key=lambda key: int(key.split(":")[1]),
            default="novelty:0",
        )
        roles: list[str] = []
        if float(stable_row.seed_mean_macro_f1) >= best_mean - 0.005 and not repeated_class_collapses:
            roles.append("GLOBAL_BACKBONE")
        if (
            float(stable_row.seed_mean_macro_f1) >= best_mean - 0.020
            and float(stable_row.seed_std_macro_f1) <= stability_cutoff
        ):
            roles.append("STABILITY_ANCHOR")
        if repeated_class_rescues:
            roles.append("RARE_CLASS_RESCUER")
        if qualifying_collisions:
            roles.append("COLLISION_EXPERT")
        if low_burden.get("positive_seed_count", 0) >= 2:
            roles.append("SPARSE_EXPERT")
        if subgroup_evidence.get(high_burden_key, {}).get("positive_seed_count", 0) >= 2:
            roles.append("DENSE_EXPERT")
        if subgroup_evidence.get(high_novelty_key, {}).get("positive_seed_count", 0) >= 2:
            roles.append("NOVELTY_EXPERT")
        worst_group_delta = min(
            (float(item["mean_delta"]) for item in subgroup_evidence.values()), default=0.0
        )
        if worst_group_delta >= 0 and float(stable_row.seed_mean_macro_f1) >= best_mean - 0.020:
            roles.append("ROBUSTNESS_ANCHOR")
        if calibration_delta < 0 and sum(row["unique_rescue"] >= 3 for row in diversity_seed_rows) >= 2:
            roles.append("CONFIDENCE_CORRECTOR")
        if alias != primary and sum(row["unique_rescue"] >= 3 for row in diversity_seed_rows) >= 2:
            roles.append("DIVERSITY_MODEL")
        if not roles:
            roles.append("GLOBAL_CHALLENGER")
        role_map["models"][alias] = {
            "role": roles[0],
            "roles": roles,
            "evidence": {
                "seed_mean_macro_f1": float(stable_row.seed_mean_macro_f1),
                "seed_min_macro_f1": float(stable_row.seed_min_macro_f1),
                "seed_std_macro_f1": float(stable_row.seed_std_macro_f1),
                "fold_std_macro_f1": float(
                    fold_frame.loc[fold_frame.alias == alias, "validation_macro_f1"].std(ddof=0)
                ),
                "repeated_class_rescues": repeated_class_rescues,
                "repeated_class_collapses": repeated_class_collapses,
                "qualifying_collision_directions": qualifying_collisions,
                "subgroups": subgroup_evidence,
                "calibration_ece_delta": float(calibration_delta),
                "diversity_by_seed": diversity_seed_rows,
                "worst_group_delta": worst_group_delta,
                "gap_used_for_role_or_objective": False,
            },
        }
    collapse = pd.concat(
        [
            class_frame.assign(metric_type="class", metric_key=lambda x: x.class_name),
            subgroup_frame.assign(
                metric_type="subgroup",
                metric_key=lambda x: x.dimension.astype(str) + ":" + x.group.astype(str),
                f1=lambda x: x.macro_f1,
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    output = context.artifact_root / "three-seed"
    _atomic_csv(output / "class_metrics.csv", pd.DataFrame(class_rows))
    _atomic_csv(output / "fold_metrics.csv", pd.DataFrame(fold_rows))
    _atomic_csv(output / "seed_stability.csv", stability)
    _atomic_csv(output / "subgroup_specialization.csv", pd.DataFrame(subgroup_rows))
    _atomic_csv(output / "subgroup_thresholds.csv", pd.DataFrame(threshold_rows))
    _atomic_csv(output / "robustness_hypothesis_tests.csv", pd.DataFrame(hypothesis_rows))
    _atomic_csv(output / "calibration_by_seed.csv", calibration_frame)
    _atomic_csv(output / "collision_direction_by_seed.csv", collision_seed_frame)
    _atomic_csv(output / "collision_direction_evidence.csv", collision_summary)
    _atomic_yaml(output / "model_role_map.yaml", role_map)
    _atomic_csv(output / "collapse_basis.csv", collapse)
    _atomic_json(output / "io_audit.json", context.audit.payload())
    return output


def _temperature(probability: np.ndarray, value: float) -> np.ndarray:
    logits = np.log(np.clip(probability, 1e-12, 1.0)) / value
    logits -= logits.max(axis=-1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=-1, keepdims=True)


def _dirichlet_weights(seed: int, trials: int, roles: int) -> np.ndarray:
    return np.random.default_rng(seed).dirichlet(np.ones(roles), size=trials)


def _select_best_trial(
    rows: pd.DataFrame,
    held_fold: int | None = None,
    *,
    held_seed: int | None = None,
    held_unit: str | None = None,
) -> pd.Series:
    if "holdout_id" in rows:
        if held_unit is None:
            if held_seed is None or held_fold is None:
                raise ValueError("held_seed and held_fold are required for seed-fold crossfit")
            held_unit = f"{held_seed}:{held_fold}"
        training = rows.loc[rows["holdout_id"] != held_unit]
    else:
        if held_fold is None:
            raise ValueError("held_fold is required")
        training = rows.loc[rows["fold"] != held_fold]
    scores = training.groupby("trial_id", sort=False)["macro_f1"].mean()
    best_id = int(scores.idxmax())
    result = rows.loc[rows.trial_id == best_id].iloc[0].copy()
    if "holdout_id" in rows:
        result["selection_holdouts"] = json.dumps(sorted(set(training.holdout_id.astype(str))))
        result["held_seed"] = int(str(held_unit).split(":")[0])
        result["held_fold"] = int(str(held_unit).split(":")[1])
        result["holdout_id"] = held_unit
    else:
        result["selection_folds"] = json.dumps(sorted(set(training.fold.astype(int))))
        result["held_fold"] = held_fold
    result["selection_macro_f1"] = float(scores.loc[best_id])
    return result


def _trial_fold_scores(
    probability: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    trial_id: int,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    if probability.ndim == 2:
        prediction = probability.argmax(axis=1)
        return [
            {
                "trial_id": trial_id,
                "fold": fold,
                "macro_f1": _macro_f1(y[folds == fold], prediction[folds == fold]),
            }
            for fold in range(5)
        ]
    if probability.ndim != 3 or folds.ndim != 2 or seeds is None:
        raise ValueError("seed-fold scoring requires [seed,row,class] probabilities and [seed,row] folds")
    return [
        {
            "trial_id": trial_id,
            "seed": int(seed),
            "fold": fold,
            "holdout_id": f"{int(seed)}:{fold}",
            "macro_f1": _macro_f1(
                y[folds[seed_index] == fold],
                probability[seed_index, folds[seed_index] == fold].argmax(axis=1),
            ),
        }
        for seed_index, seed in enumerate(seeds)
        for fold in range(5)
    ]


def _draw_model_trial(
    rng: np.random.Generator,
    aliases: list[str],
    role_map: dict[str, Any],
    max_models: int,
    constrained: bool,
) -> list[str]:
    size = int(rng.integers(2, min(max_models, len(aliases)) + 1))
    if not constrained:
        return sorted(rng.choice(aliases, size=size, replace=False).tolist())
    global_models = [
        alias
        for alias in aliases
        if "GLOBAL_BACKBONE" in role_map[alias].get("roles", [role_map[alias].get("role")])
    ]
    complement_models = [
        alias
        for alias in aliases
        if set(role_map[alias].get("roles", [role_map[alias].get("role")]))
        & {
            "COLLISION_EXPERT", "RARE_CLASS_RESCUER", "ROBUSTNESS_ANCHOR",
            "STABILITY_ANCHOR", "NOVELTY_EXPERT", "DIVERSITY_MODEL",
        }
    ]
    if not global_models or not complement_models:
        raise ValueError("role-constrained search requires both backbone and complementary-role pools")
    first = str(rng.choice(global_models))
    complement_pool = [alias for alias in complement_models if alias != first]
    if not complement_pool:
        raise ValueError("role-constrained search pools must contain two distinct models")
    second = str(rng.choice(complement_pool))
    selected = [first, second]
    remaining = [alias for alias in aliases if alias not in selected]
    if size > 2:
        selected.extend(rng.choice(remaining, size=size - 2, replace=False).tolist())
    return sorted(selected)


def _draw_constrained_weights(
    rng: np.random.Generator,
    size: int,
    minimum: float,
    maximum: float,
    attempts: int = 1000,
) -> np.ndarray:
    if minimum * size > 1 + 1e-12 or maximum * size < 1 - 1e-12:
        raise ValueError(
            f"infeasible model-weight bounds for subset size {size}: "
            f"minimum={minimum}, maximum={maximum}"
        )
    for _ in range(attempts):
        weight = rng.dirichlet(np.ones(size))
        if np.all(weight >= minimum) and np.all(weight <= maximum):
            return weight
    raise RuntimeError("failed to draw Dirichlet model weights within configured bounds")


def _holdout_masks(
    fold_by_seed: np.ndarray, held_seed_index: int, held_fold: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    held_rows = np.flatnonzero(fold_by_seed[held_seed_index] == held_fold)
    selection = np.ones(fold_by_seed.shape, dtype=bool)
    selection[:, held_rows] = False
    held = np.zeros(fold_by_seed.shape, dtype=bool)
    held[held_seed_index, held_rows] = True
    if np.intersect1d(np.flatnonzero(selection.any(axis=0)), held_rows).size:
        raise AssertionError("held row IDs leaked into another seed's selection rows")
    return selection, held, held_rows


def _masked_macro_f1(
    y_grid: np.ndarray, probability: np.ndarray, mask: np.ndarray
) -> float:
    return _macro_f1(y_grid[mask], probability[mask].argmax(axis=1))


def _derive_selection_role_map(
    aliases: list[str],
    probability_by_alias: dict[str, np.ndarray],
    y_grid: np.ndarray,
    selection_mask: np.ndarray,
) -> dict[str, Any]:
    scores = {
        alias: _masked_macro_f1(y_grid, probability_by_alias[alias], selection_mask)
        for alias in aliases
    }
    baseline = max(aliases, key=lambda alias: (scores[alias], alias))
    baseline_prediction = probability_by_alias[baseline][selection_mask].argmax(axis=1)
    selected_y = y_grid[selection_mask]
    baseline_correct = baseline_prediction == selected_y
    mapping: dict[str, Any] = {}
    for alias in aliases:
        candidate_prediction = probability_by_alias[alias][selection_mask].argmax(axis=1)
        unique_rescue = int(np.count_nonzero(~baseline_correct & (candidate_prediction == selected_y)))
        roles: list[str] = []
        if scores[alias] >= scores[baseline] - 0.005:
            roles.append("GLOBAL_BACKBONE")
        if alias != baseline and unique_rescue >= 3:
            roles.append("DIVERSITY_MODEL")
        if not roles:
            roles.append("GLOBAL_CHALLENGER")
        mapping[alias] = {
            "role": roles[0],
            "roles": roles,
            "selection_macro_f1": scores[alias],
            "unique_rescue": unique_rescue,
            "evidence_scope": "holdout_selection_rows_only",
        }
    return mapping


class _TrialChunkStream:
    def __init__(self, output: Path) -> None:
        output.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.temporary = Path(tempfile.mkdtemp(prefix=".trial_chunks.", dir=output))
        self.chunks: list[dict[str, Any]] = []
        self.rows = 0
        self.format: str | None = None
        self.closed = False

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        frame = pd.DataFrame(rows)
        index = len(self.chunks)
        if self.format in (None, "parquet"):
            path = self.temporary / f"chunk_{index:05d}.parquet"
            try:
                frame.to_parquet(path, index=False)
                self.format = "parquet"
            except (ImportError, ModuleNotFoundError, ValueError):
                if path.exists():
                    path.unlink()
                if self.format == "parquet":
                    raise
                self.format = "csv"
        if self.format == "csv":
            path = self.temporary / f"chunk_{index:05d}.csv"
            _atomic_csv(path, frame)
        chunk = {
            "file": path.name,
            "rows": len(frame),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "holdout_id": str(frame.holdout_id.iloc[0]),
            "random_seed": int(frame.random_seed.iloc[0]),
            "local_id_first": int(frame.local_trial_id.iloc[0]),
            "local_id_last": int(frame.local_trial_id.iloc[-1]),
        }
        self.chunks.append(chunk)
        self.rows += len(frame)

    def finalize(self) -> dict[str, Any]:
        identity = _canonical_hash(
            [{key: value for key, value in chunk.items() if key != "file"} for chunk in self.chunks]
        )
        directory_name = f"random_search_trial_chunks_{identity[:16]}"
        target = self.output / directory_name
        quarantine = target.with_name(
            f".{target.name}.quarantine.{os.getpid()}.{time.time_ns()}"
        )
        expected = {chunk["file"]: chunk for chunk in self.chunks}
        if target.exists() and self._verify_directory(target, expected):
            shutil.rmtree(self.temporary)
        elif target.exists():
            os.replace(target, quarantine)
            try:
                os.replace(self.temporary, target)
                if not self._verify_directory(target, expected):
                    raise RuntimeError("replacement trial chunk generation failed verification")
                shutil.rmtree(quarantine)
            except Exception as error:
                if target.exists():
                    shutil.rmtree(target)
                if quarantine.exists():
                    os.replace(quarantine, target)
                _atomic_json(
                    self.output / "random_search_trials_generation_invalid.json",
                    {
                        "status": "INVALIDATED",
                        "directory": directory_name,
                        "reason": str(error),
                        "previous_generation_restored": target.exists(),
                    },
                )
                raise
        else:
            os.replace(self.temporary, target)
            if not self._verify_directory(target, expected):
                shutil.rmtree(target)
                raise RuntimeError("new trial chunk generation failed verification")
        invalidation = self.output / "random_search_trials_generation_invalid.json"
        if invalidation.exists():
            invalidation.unlink()
        self.closed = True
        return {
            "format": self.format,
            "directory": directory_name,
            "rows": self.rows,
            "sha256": identity,
            "chunks": [
                {**chunk, "path": f"{directory_name}/{chunk['file']}"}
                for chunk in self.chunks
            ],
        }

    @staticmethod
    def _verify_directory(directory: Path, expected: dict[str, dict[str, Any]]) -> bool:
        try:
            actual_files = {path.name for path in directory.iterdir() if path.is_file()}
        except (FileNotFoundError, OSError):
            return False
        if actual_files != set(expected):
            return False
        for name, metadata in expected.items():
            path = directory / name
            try:
                if path.stat().st_size != int(metadata["size_bytes"]):
                    return False
                if _sha256(path) != metadata["sha256"]:
                    return False
            except (FileNotFoundError, OSError):
                return False
        return True

    def abort(self) -> None:
        if not self.closed and self.temporary.exists():
            shutil.rmtree(self.temporary)
        self.closed = True


def _batch_macro_f1(
    y: np.ndarray, predictions: np.ndarray, classes: int = 26
) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.int64)
    offsets = np.arange(len(predictions), dtype=np.int64)[:, None] * classes * classes
    joint = offsets + y.astype(np.int64, copy=False)[None, :] * classes + predictions
    confusion = np.bincount(
        joint.ravel(), minlength=len(predictions) * classes * classes
    ).reshape(len(predictions), classes, classes)
    true_positive = np.diagonal(confusion, axis1=1, axis2=2).astype(np.float64)
    false_positive = confusion.sum(axis=1) - true_positive
    false_negative = confusion.sum(axis=2) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return f1.mean(axis=1)


def _score_weight_batch(
    dense_weights: np.ndarray,
    selection_bank_flat: np.ndarray,
    selection_y: np.ndarray,
    classes: int,
) -> np.ndarray:
    with threadpool_limits(limits=1, user_api="blas"):
        blended = (dense_weights @ selection_bank_flat).reshape(
            len(dense_weights), len(selection_y), classes
        )
    scores = _batch_macro_f1(selection_y, blended.argmax(axis=2), classes)
    del blended
    return scores


def _estimated_search_peak_bytes(
    models: int,
    seeds: int,
    rows: int,
    classes: int,
    batch_size: int,
    folds: int = 5,
) -> int:
    probability_bank = models * seeds * rows * classes * 8
    selection_rows = seeds * rows * (folds - 1) // folds
    selection_bank = models * selection_rows * classes * 8
    batch_blends = batch_size * selection_rows * classes * 8
    equal_predictions = (
        math.comb(models, 2) + math.comb(models, 3)
    ) * seeds * rows
    return probability_bank + selection_bank + batch_blends + equal_predictions


def _benchmark_search_kernel(
    *,
    models: int = 20,
    seeds: int = 3,
    rows: int = 6201,
    classes: int = 26,
    batch_size: int = 64,
    repeats: int = 3,
    random_seed: int = 730,
) -> dict[str, Any]:
    """Deterministic median benchmark for the legacy scalar and batched kernels."""
    rng = np.random.default_rng(random_seed)
    full_bank = rng.random((models, seeds, rows, classes), dtype=np.float64)
    full_bank /= full_bank.sum(axis=3, keepdims=True)
    selection_mask = np.ones((seeds, rows), dtype=bool)
    selection_mask[:, ::5] = False
    selection_rows = int(selection_mask.sum())
    bank = np.empty((models, selection_rows, classes), dtype=np.float64)
    for model_index in range(models):
        bank[model_index] = full_bank[model_index][selection_mask]
    flat = bank.reshape(models, -1)
    y_single = rng.integers(0, classes, rows, dtype=np.int16)
    y = np.broadcast_to(y_single, (seeds, rows))[selection_mask]
    dense = np.zeros((batch_size, models), dtype=np.float64)
    for index in range(batch_size):
        selected = rng.choice(models, size=3, replace=False)
        dense[index, selected] = rng.dirichlet(np.ones(3))
    _score_weight_batch(dense[:2], flat, y, classes)
    batched_times, scalar_times = [], []
    batched_scores = scalar_scores = None
    for _ in range(repeats):
        started = time.perf_counter()
        batched_scores = _score_weight_batch(dense, flat, y, classes)
        batched_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        scalar_values = []
        with threadpool_limits(limits=1, user_api="blas"):
            for weight in dense:
                active = np.flatnonzero(weight)
                probability = np.tensordot(
                    weight[active], full_bank[active], axes=(0, 0)
                )
                scalar_values.append(
                    _macro_f1(y, probability[selection_mask].argmax(axis=1), classes)
                )
        scalar_scores = np.asarray(scalar_values)
        scalar_times.append(time.perf_counter() - started)
    assert batched_scores is not None and scalar_scores is not None
    batched_median = float(np.median(batched_times))
    scalar_median = float(np.median(scalar_times))
    return {
        "shape": [models, seeds, rows, classes],
        "batch_size": batch_size,
        "repeats": repeats,
        "batched_median_seconds": batched_median,
        "scalar_median_seconds": scalar_median,
        "speedup": scalar_median / batched_median,
        "projected_900k_minutes": batched_median / batch_size * 900_000 / 60,
        "scores_equal": bool(np.allclose(batched_scores, scalar_scores, rtol=0, atol=1e-15)),
    }


def _search(context: Context) -> Path:
    _load_train(context)
    aliases = _survivor_aliases(context)
    specs = [spec for spec in context.specs if spec.alias in aliases]
    seeds = list(REQUIRED_FINAL_SEEDS)
    _assert_all_seed_runs_exist(context, specs, seeds)
    runs = _assert_fixed_alignment(
        context, [_load_run(context, spec, seed) for spec in specs for seed in seeds]
    )
    names = context.class_names
    assert names is not None and context.labels is not None
    y_single = _label_index(context.labels, names)
    run_lookup = {(run.alias, run.seed): run for run in runs}
    probability_by_alias = {
        alias: np.stack([run_lookup[(alias, seed)].probabilities for seed in seeds])
        for alias in aliases
    }
    fold_by_seed = np.stack(
        [run_lookup[(aliases[0], seed)].alignment["fold"].to_numpy(dtype=int) for seed in seeds]
    )
    y_grid = np.broadcast_to(y_single, fold_by_seed.shape)
    assert context.ids is not None
    output = context.artifact_root / "search"
    holdout_registry = {
        f"{seed}:{fold}": {
            "seed": seed,
            "fold": fold,
            "row_indices": np.flatnonzero(fold_by_seed[seed_index] == fold).astype(int).tolist(),
            "row_ids": context.ids[fold_by_seed[seed_index] == fold].astype(str).tolist(),
            "selection_id_overlap": 0,
        }
        for seed_index, seed in enumerate(seeds)
        for fold in range(5)
    }
    _atomic_json(output / "holdout_registry.json", holdout_registry)
    pair_rows, triple_rows, equal_trial_models = [], [], {}
    equal_prediction_cache: list[np.ndarray] = []
    equal_trial_id = 0
    for size, target in ((2, pair_rows), (3, triple_rows)):
        for selected in combinations(aliases, size):
            probability = np.mean([probability_by_alias[alias] for alias in selected], axis=0)
            prediction = probability.argmax(axis=2).astype(np.uint8, copy=False)
            model_names = "+".join(selected)
            target.append(
                {
                    "models": model_names,
                    "size": size,
                    "macro_f1": float(np.mean([
                        _macro_f1(y_single, prediction[index])
                        for index in range(len(seeds))
                    ])),
                }
            )
            equal_trial_models[equal_trial_id] = {"models": model_names, "size": size}
            equal_prediction_cache.append(prediction)
            equal_trial_id += 1
    equal_predictions = np.stack(equal_prediction_cache, axis=0) if equal_prediction_cache else np.empty(
        (0, len(seeds), len(y_single)), dtype=np.uint8
    )
    del equal_prediction_cache
    crossfit_equal_rows = []
    for size in (2, 3):
        trials_for_size = [
            (trial_id, metadata)
            for trial_id, metadata in equal_trial_models.items()
            if metadata["size"] == size
        ]
        if not trials_for_size:
            continue
        for seed_index, held_seed in enumerate(seeds):
            for held_fold in range(5):
                selection_mask, held_mask, held_rows = _holdout_masks(
                    fold_by_seed, seed_index, held_fold
                )
                candidates = []
                for trial_id, metadata in trials_for_size:
                    prediction = equal_predictions[trial_id]
                    candidates.append(
                        (
                            _macro_f1(y_grid[selection_mask], prediction[selection_mask]),
                            trial_id,
                            metadata,
                            prediction,
                        )
                    )
                selection_score, trial_id, metadata, prediction = max(
                    candidates, key=lambda item: (item[0], -item[1])
                )
                crossfit_equal_rows.append(
                    {
                        **metadata,
                        "held_seed": held_seed,
                        "held_fold": held_fold,
                        "holdout_id": f"{held_seed}:{held_fold}",
                        "trial_id": trial_id,
                        "selection_id_overlap": 0,
                        "selection_macro_f1": selection_score,
                        "held_macro_f1": _macro_f1(
                            y_grid[held_mask], prediction[held_mask]
                        ),
                    }
                )
    temperature_rows = []
    temperature_grid = (0.7, 0.85, 1.0, 1.15, 1.3)
    for alias in aliases:
        raw = probability_by_alias[alias]
        for seed_index, held_seed in enumerate(seeds):
            for held_fold in range(5):
                losses = []
                selection_mask, held_mask, held_rows = _holdout_masks(
                    fold_by_seed, seed_index, held_fold
                )
                for temperature in temperature_grid:
                    calibrated = _temperature(raw[selection_mask], temperature)
                    losses.append(float(-np.log(np.clip(
                        calibrated[np.arange(selection_mask.sum()), y_grid[selection_mask]], 1e-12, 1
                    )).mean()))
                best_index = int(np.argmin(losses))
                best_temperature = float(temperature_grid[best_index])
                held_probability = _temperature(raw[held_mask], best_temperature)
                temperature_rows.append(
                    {
                        "alias": alias,
                        "held_seed": held_seed,
                        "held_fold": held_fold,
                        "holdout_id": f"{held_seed}:{held_fold}",
                        "selection_id_overlap": 0,
                        "temperature": best_temperature,
                        "selection_logloss": float(losses[best_index]),
                        "held_macro_f1": _macro_f1(
                            y_grid[held_mask], held_probability.argmax(axis=1)
                        ),
                    }
                )
    role_path = context.artifact_root / "three-seed" / "model_role_map.yaml"
    context.audit.read(role_path, "three_seed_role_map")
    three_seed_role_map = yaml.safe_load(role_path.read_text(encoding="utf-8"))["models"]
    if set(aliases) - set(three_seed_role_map):
        raise ValueError("three-seed role map is missing survivor aliases")
    search = context.config.get("search", {})
    trials_per_seed = int(search.get("random_trials", 20000))
    random_seeds = [int(value) for value in search.get("random_seeds", [1001, 1002, 1003])]
    min_weight = float(search.get("min_active_weight", 0.05))
    max_weight = float(search.get("max_model_weight", 0.80))
    max_models = min(int(search.get("max_models", 5)), len(aliases))
    patience = int(search.get("early_stop_patience", 5000))
    if len(random_seeds) != 3 or len(set(random_seeds)) != 3:
        raise ValueError("random search requires exactly three distinct search seeds")
    if trials_per_seed <= 0 or trials_per_seed % 5:
        raise ValueError("random_trials must be a positive multiple of 5 for exact 80/20 role sampling")
    if len(aliases) < 2 or max_models < 2:
        raise ValueError("random search requires at least two surviving models")
    selected_rows: list[dict[str, Any]] = []
    crossfit_role_maps: dict[str, Any] = {}
    top_candidates: list[dict[str, Any]] = []
    executed_counts: dict[str, int] = {}
    stop_points: dict[str, Any] = {}
    constrained_count = 0
    unconstrained_count = 0
    batch_size = int(search.get("gemm_batch_size", 256))
    chunk_rows_limit = int(search.get("stream_chunk_rows", 1000))
    if batch_size <= 0 or chunk_rows_limit <= 0:
        raise ValueError("gemm_batch_size and stream_chunk_rows must be positive")
    trial_id = 0
    crossfit_prediction = np.empty((len(seeds), len(y_single)), dtype=np.int16)
    alias_index = {alias: index for index, alias in enumerate(aliases)}
    stream = _TrialChunkStream(output)
    try:
        for seed_index, held_seed in enumerate(seeds):
            for held_fold in range(5):
                holdout_id = f"{held_seed}:{held_fold}"
                selection_mask, held_mask, _ = _holdout_masks(
                    fold_by_seed, seed_index, held_fold
                )
                role_map = _derive_selection_role_map(
                    aliases, probability_by_alias, y_grid, selection_mask
                )
                crossfit_role_maps[holdout_id] = role_map
                selection_y = np.ascontiguousarray(y_grid[selection_mask])
                selection_bank = np.empty(
                    (len(aliases), len(selection_y), len(names)), dtype=np.float64
                )
                for model_index, alias in enumerate(aliases):
                    selection_bank[model_index] = probability_by_alias[alias][selection_mask]
                selection_bank_flat = selection_bank.reshape(len(aliases), -1)
                best_holdout: dict[str, Any] | None = None
                for random_seed in random_seeds:
                    rng = np.random.default_rng(
                        np.random.SeedSequence([random_seed, held_seed, held_fold])
                    )
                    best_score, best_at = -math.inf, -1
                    local_id = 0
                    stopped_early = False
                    chunk_buffer: list[dict[str, Any]] = []
                    accepted = 0
                    while local_id < trials_per_seed and not stopped_early:
                        generated = min(batch_size, trials_per_seed - local_id)
                        dense_weights = np.zeros((generated, len(aliases)), dtype=np.float64)
                        generated_metadata: list[tuple[list[str], np.ndarray, bool]] = []
                        for offset in range(generated):
                            candidate_local_id = local_id + offset
                            constrained = candidate_local_id % 5 != 4
                            selected_aliases = _draw_model_trial(
                                rng, aliases, role_map, max_models, constrained
                            )
                            weight = _draw_constrained_weights(
                                rng, len(selected_aliases), min_weight, max_weight
                            )
                            dense_weights[
                                offset, [alias_index[alias] for alias in selected_aliases]
                            ] = weight
                            generated_metadata.append((selected_aliases, weight, constrained))
                        scores = _score_weight_batch(
                            dense_weights, selection_bank_flat, selection_y, len(names)
                        )
                        for offset, selection_score in enumerate(scores):
                            candidate_local_id = local_id + offset
                            selected_aliases, weight, constrained = generated_metadata[offset]
                            row = {
                                "trial_id": trial_id,
                                "holdout_id": holdout_id,
                                "random_seed": random_seed,
                                "local_trial_id": candidate_local_id,
                                "models": "+".join(selected_aliases),
                                "model_count": len(selected_aliases),
                                "model_weights": json.dumps(
                                    dict(zip(selected_aliases, map(float, weight))), sort_keys=True
                                ),
                                "role_constrained": constrained,
                                "selection_macro_f1": float(selection_score),
                                "macro_f1": float(selection_score),
                                "eligible": True,
                                "objective": "selection_macro_f1",
                                "overfit_gap": np.nan,
                            }
                            chunk_buffer.append(row)
                            accepted += 1
                            constrained_count += int(constrained)
                            unconstrained_count += int(not constrained)
                            if selection_score > best_score:
                                best_score, best_at = float(selection_score), candidate_local_id
                            if best_holdout is None or (
                                selection_score > best_holdout["selection_macro_f1"]
                            ):
                                best_holdout = row.copy()
                            top_candidates.append(row.copy())
                            top_candidates.sort(
                                key=lambda item: (-item["selection_macro_f1"], item["trial_id"])
                            )
                            del top_candidates[20:]
                            trial_id += 1
                            if len(chunk_buffer) >= chunk_rows_limit:
                                stream.write(chunk_buffer)
                                chunk_buffer.clear()
                            if (
                                patience > 0
                                and (candidate_local_id + 1) % 5 == 0
                                and candidate_local_id - best_at >= patience
                            ):
                                stopped_early = True
                                break
                        local_id += generated
                    stream.write(chunk_buffer)
                    key = f"{holdout_id}|{random_seed}"
                    executed_counts[key] = accepted
                    stop_points[key] = {
                        "last_local_id": accepted - 1,
                        "best_local_id": best_at,
                        "stopped_early": stopped_early,
                    }
                del selection_bank_flat, selection_bank, selection_y
                if best_holdout is None:
                    raise RuntimeError(f"{holdout_id}: random search produced no trial")
                weights = json.loads(str(best_holdout["model_weights"]))
                held_probability = sum(
                    float(weight) * probability_by_alias[alias][held_mask]
                    for alias, weight in weights.items()
                )
                held_score = _macro_f1(
                    y_grid[held_mask], held_probability.argmax(axis=1)
                )
                crossfit_prediction[seed_index, held_mask[seed_index]] = held_probability.argmax(axis=1)
                selected_rows.append(
                    {
                        "held_seed": held_seed,
                        "held_fold": held_fold,
                        "holdout_id": holdout_id,
                        "selection_id_overlap": 0,
                        "trial_id": int(best_holdout["trial_id"]),
                        "random_seed": int(best_holdout["random_seed"]),
                        "models": best_holdout["models"],
                        "model_weights": best_holdout["model_weights"],
                        "role_constrained": bool(best_holdout["role_constrained"]),
                        "selection_macro_f1": float(best_holdout["selection_macro_f1"]),
                        "held_macro_f1": held_score,
                        "objective": "selection_macro_f1",
                        "gap_used_in_objective": False,
                    }
                )
        trial_manifest = stream.finalize()
    except Exception:
        stream.abort()
        raise
    _atomic_csv(output / "equal_pair_search.csv", pd.DataFrame(pair_rows).sort_values("macro_f1", ascending=False))
    _atomic_csv(output / "equal_triple_search.csv", pd.DataFrame(triple_rows).sort_values("macro_f1", ascending=False))
    _atomic_csv(output / "crossfit_equal_selected.csv", pd.DataFrame(crossfit_equal_rows))
    _atomic_csv(output / "crossfit_temperature.csv", pd.DataFrame(temperature_rows))
    _atomic_json(
        output / "random_search_trials_manifest.json",
        {
            **trial_manifest,
            "trial_count_per_random_seed": trials_per_seed,
            "executed_trials_by_seed": {
                str(seed): int(sum(
                    count for key, count in executed_counts.items() if key.endswith(f"|{seed}")
                ))
                for seed in random_seeds
            },
            "executed_trials_by_holdout_search_seed": executed_counts,
            "stop_points": stop_points,
            "random_seeds": random_seeds,
            "subset_size": [2, max_models],
            "min_active_weight": min_weight,
            "max_model_weight": max_weight,
            "early_stop_patience": patience,
            "role_constrained_fraction_target": 0.8,
            "unconstrained_fraction_target": 0.2,
            "role_constrained_fraction_actual": constrained_count / trial_manifest["rows"],
            "unconstrained_fraction_actual": unconstrained_count / trial_manifest["rows"],
            "holdout_count": len(holdout_registry),
            "held_id_overlap_max": 0,
            "gemm_batch_size": batch_size,
            "stream_chunk_rows": chunk_rows_limit,
            "metadata_accumulation": "streamed_chunks_top20_and_counters_only",
            "trial_generation_scope": "independent_per_holdout_selection_rows_only",
            "objective_fields": ["selection_macro_f1"],
            "gap_usage": "not_used_in_search_or_promotion",
            "reproducible": True,
        },
    )
    selected_frame = pd.DataFrame(selected_rows)
    _atomic_csv(output / "crossfit_selected_trials.csv", selected_frame)
    _atomic_json(output / "crossfit_role_maps.json", crossfit_role_maps)
    baseline_alias = str(
        max(
            aliases,
            key=lambda alias: np.mean([
                _macro_f1(y_single, probability_by_alias[alias][index].argmax(axis=1))
                for index in range(len(seeds))
            ]),
        )
    )
    baseline_prediction = np.stack([
        probability_by_alias[baseline_alias][index].argmax(axis=1)
        for index in range(len(seeds))
    ])
    seed_deltas = [
        _macro_f1(y_single, crossfit_prediction[index])
        - _macro_f1(y_single, baseline_prediction[index])
        for index in range(len(seeds))
    ]
    promotion = _paired_evidence(
        seed_deltas,
        np.tile(y_single, len(seeds)),
        crossfit_prediction.reshape(-1),
        baseline_prediction.reshape(-1),
        repeats=int(context.config.get("validation", {}).get("bootstrap_repeats", 2000)),
        bootstrap_seed=int(context.config.get("validation", {}).get("bootstrap_seed", 730)) + 9000,
        classes=len(names),
    )
    _atomic_json(
        output / "promotion_evidence.json",
        {
            "candidate": "crossfit_random_search",
            "baseline": baseline_alias,
            **promotion,
            "promotion_rule": "delta_vs_dynamic_epsilon",
            "gap_used_in_objective": False,
        },
    )
    _atomic_json(output / "top_candidates.json", top_candidates)
    _atomic_json(
        output / "ensemble_ablation_draft.json",
        {
            "status": "draft",
            "equal_pair_best": pair_rows and max(pair_rows, key=lambda row: row["macro_f1"]),
            "equal_triple_best": triple_rows and max(triple_rows, key=lambda row: row["macro_f1"]),
            "crossfit_random_mean": float(selected_frame.held_macro_f1.mean()),
            "gap_used_in_objective": False,
        },
    )
    _atomic_json(output / "io_audit.json", context.audit.payload())
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen", "three-seed", "search"), required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    context = _load_context(args.config, args.stage)
    output = {
        "screen": _screen,
        "three-seed": _three_seed,
        "search": _search,
    }[args.stage](context)
    print(
        json.dumps(
            {
                "status": "PASS",
                "stage": args.stage,
                "output": str(output),
                "selection_uses_test": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
