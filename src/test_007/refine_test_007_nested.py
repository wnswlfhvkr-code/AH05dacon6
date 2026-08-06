"""Deterministic train-only second-stage refinement of completed TEST_007 Nested OOF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import stat
import zipfile
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


SCHEMA_VERSION = 1
SEEDS = (42, 2026, 777)
OUTER_FOLDS = tuple(range(5))
INNER_FOLDS = tuple(range(4))
LANES = ("primary", "robustness_challenger", "specialization_challenger")
GRID = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
WEIGHT_GRID = (0.0, *GRID)
OUTPUTS = (
    "refined_strategy_probability.npz",
    "refined_inner_probability.npz",
    "refined_selection_records.json",
    "refined_cv_metrics.json",
    "refined_manifest.json",
)
UPSTREAM_FILES = (
    "nested_run_manifest.json",
    "nested_inner_selection_probability.npz",
    "nested_inner_selection_records.json",
    "nested_selection_records.json",
    "nested_strategy_probability.npz",
)
FORBIDDEN_BASENAMES = frozenset({"test.csv", "sample_submission.csv"})
NUMERICAL_TOLERANCE = 1e-15
MIN_GROUP_AGGREGATE_SUPPORT = 4
MIN_GROUP_OBSERVED_FOLDS = 2
REPARSE_POINT_ATTRIBUTE = 0x400
STAGE_MARKER = ".refinement-stage.json"


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


def _is_reparse(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & REPARSE_POINT_ATTRIBUTE)


def _assert_plain(path: Path, *, directory: bool, role: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"missing {role}: {path}") from error
    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if not expected or stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
        raise RuntimeError(f"{role} must be a plain non-reparse {'directory' if directory else 'file'}")
    return metadata


def _exclusive_file(path: Path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags, 0o600)


def _atomic_write(path: Path, writer) -> None:
    _assert_plain(path.parent, directory=True, role="atomic output parent")
    if path.exists() or path.is_symlink():
        _assert_plain(path, directory=False, role="atomic replace target")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
    descriptor = _exclusive_file(temporary)
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            descriptor = -1
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write(path, lambda stream: stream.write(payload))


def _deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    def write(stream) -> None:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    _atomic_write(path, write)


def _guard_input(path: Path, role: str) -> Path:
    original = Path(path)
    resolved = original.resolve()
    names = {original.name.casefold(), resolved.name.casefold()}
    if names & FORBIDDEN_BASENAMES:
        raise ValueError(f"{role} aliases a forbidden Test/submission path")
    repo = Path(__file__).resolve().parents[2]
    protected = (repo / "data/raw/test.csv", repo / "data/raw/sample_submission.csv")
    for candidate in protected:
        try:
            aliases_protected = resolved == candidate.resolve() or (
                resolved.exists() and candidate.exists() and resolved.samefile(candidate)
            )
        except OSError:
            aliases_protected = False
        if aliases_protected:
            raise ValueError(f"{role} aliases a forbidden Test/submission path")
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} is unavailable: {resolved}")
    return resolved


def _probability(values: np.ndarray, shape: tuple[int, ...], role: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{role} shape mismatch: {result.shape} != {shape}")
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError(f"{role} contains invalid probabilities")
    if not np.allclose(result.sum(axis=-1), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError(f"{role} probability rows must sum to one")
    return result


def _predictions(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    return array.argmax(axis=1) if array.ndim == 2 else array.astype(np.int64, copy=False)


def _per_class_f1(labels: np.ndarray, values: np.ndarray, classes: int) -> np.ndarray:
    truth = np.asarray(labels, dtype=np.int64)
    prediction = _predictions(values)
    confusion = np.bincount(
        truth * classes + prediction, minlength=classes * classes
    ).reshape(classes, classes)
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    return np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros(classes, dtype=np.float64),
        where=denominator != 0,
    )


def _macro(labels: np.ndarray, probability: np.ndarray, classes: int) -> float:
    return float(_per_class_f1(labels, probability, classes).mean())


def _fold_scores(
    labels: np.ndarray, probability: np.ndarray, fold_ids: np.ndarray, classes: int
) -> list[float]:
    return [
        _macro(labels[fold_ids == fold], probability[fold_ids == fold], classes)
        for fold in INNER_FOLDS
    ]


def _mutation_mask(frame: pd.DataFrame) -> np.ndarray:
    values = frame.fillna("WT").astype(str).to_numpy()
    normalized = np.char.upper(np.char.strip(values.astype(str)))
    return ~np.isin(normalized, ("", "WT", "WILDTYPE", "NAN", "NONE"))


def _assign_quantile_groups(
    values: np.ndarray, lower: float, upper: float
) -> np.ndarray:
    return np.where(values <= lower, 0, np.where(values >= upper, 2, 1)).astype(
        np.int8
    )


def _fold_safe_groups(
    train_x: pd.DataFrame, valid_x: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    """Exact refinement copy of run_test_007_nested._fold_safe_groups."""
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


def _inner_fold_safe_groups(
    features: pd.DataFrame, fold_ids: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, list[float]]]]:
    groups = {
        "burden": np.full(len(features), -1, dtype=np.int8),
        "novelty": np.full(len(features), -1, dtype=np.int8),
    }
    thresholds: dict[str, dict[str, list[float]]] = {}
    for fold in INNER_FOLDS:
        valid = fold_ids == fold
        fold_groups, fold_thresholds = _fold_safe_groups(
            features.loc[~valid], features.loc[valid]
        )
        for dimension in groups:
            groups[dimension][valid] = fold_groups[dimension]
        thresholds[str(fold)] = fold_thresholds
    if any(np.any(values < 0) for values in groups.values()):
        raise RuntimeError("fold-safe inner group coverage is incomplete")
    return groups, thresholds


def _group_scores(
    labels: np.ndarray,
    values: np.ndarray,
    groups: dict[str, np.ndarray],
    classes: int,
    mask: np.ndarray | None = None,
) -> dict[str, float | None]:
    active = np.ones(len(labels), dtype=bool) if mask is None else mask
    scores: dict[str, float | None] = {}
    for dimension in ("burden", "novelty"):
        for group in (0, 1, 2):
            selected = active & (groups[dimension] == group)
            scores[f"{dimension}:{group}"] = (
                _macro(labels[selected], np.asarray(values)[selected], classes)
                if selected.any()
                else None
            )
    return scores


def _group_safety_evidence(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    groups: dict[str, np.ndarray],
    fold_ids: np.ndarray,
    classes: int,
) -> dict[str, Any]:
    base_scores = _group_scores(labels, baseline, groups, classes)
    scores = _group_scores(labels, candidate, groups, classes)
    keys = [f"{dimension}:{group}" for dimension in ("burden", "novelty") for group in (0, 1, 2)]
    aggregate_support = {
        key: int(np.sum(groups[key.split(":", 1)[0]] == int(key.split(":", 1)[1])))
        for key in keys
    }
    fold_support: dict[str, list[int]] = {key: [] for key in keys}
    fold_deltas: dict[str, list[float | None]] = {key: [] for key in keys}
    for fold in INNER_FOLDS:
        mask = fold_ids == fold
        base_fold = _group_scores(labels, baseline, groups, classes, mask)
        candidate_fold = _group_scores(labels, candidate, groups, classes, mask)
        for key in keys:
            dimension, group = key.split(":", 1)
            support = int(np.sum(mask & (groups[dimension] == int(group))))
            fold_support[key].append(support)
            fold_deltas[key].append(
                None
                if support == 0 or base_fold[key] is None or candidate_fold[key] is None
                else float(candidate_fold[key] - base_fold[key])
            )
    observed_fold_counts = {
        key: sum(value > 0 for value in values) for key, values in fold_support.items()
    }
    sufficiently_supported = {
        key: (
            aggregate_support[key] >= MIN_GROUP_AGGREGATE_SUPPORT
            and observed_fold_counts[key] >= MIN_GROUP_OBSERVED_FOLDS
        )
        for key in keys
    }
    insufficient_reasons = {
        key: [
            reason
            for failed, reason in (
                (
                    aggregate_support[key] < MIN_GROUP_AGGREGATE_SUPPORT,
                    "aggregate_support_below_minimum",
                ),
                (
                    observed_fold_counts[key] < MIN_GROUP_OBSERVED_FOLDS,
                    "observed_fold_count_below_minimum",
                ),
            )
            if failed
        ]
        for key in keys
        if not sufficiently_supported[key]
    }
    fold_delta_se: dict[str, float | None] = {}
    noninferiority_margin: dict[str, float | None] = {}
    decline_counts: dict[str, int] = {}
    for key, values in fold_deltas.items():
        observed = np.asarray(
            [value for value in values if value is not None], dtype=np.float64
        )
        se = None
        if sufficiently_supported[key]:
            se = float(np.std(observed, ddof=1) / np.sqrt(len(observed)))
        fold_delta_se[key] = se
        noninferiority_margin[key] = (
            None
            if se is None or scores[key] is None or base_scores[key] is None
            else float(scores[key] - base_scores[key] + se)
        )
        decline_counts[key] = (
            0
            if se is None
            else sum(
                value is not None and value < -se - NUMERICAL_TOLERANCE
                for value in values
            )
        )
    supported_keys = [key for key in keys if sufficiently_supported[key]]
    sufficiently_supported_dimensions = {
        key.split(":", 1)[0] for key in supported_keys
    }
    evidence_sufficient = bool(supported_keys) and sufficiently_supported_dimensions == set(groups)
    floor = (
        min(float(scores[key]) for key in supported_keys)
        if evidence_sufficient
        else None
    )
    base_floor = (
        min(float(base_scores[key]) for key in supported_keys)
        if evidence_sufficient
        else None
    )
    floor_delta = None if floor is None or base_floor is None else floor - base_floor
    supported_margins = [noninferiority_margin[key] for key in supported_keys]
    all_supported_noninferior = bool(evidence_sufficient) and all(
        margin is not None and margin >= -NUMERICAL_TOLERANCE
        for margin in supported_margins
    )
    return {
        "base_group_scores": base_scores,
        "group_scores": scores,
        "group_score_delta": {
            key: (
                None
                if scores[key] is None or base_scores[key] is None
                else scores[key] - base_scores[key]
            )
            for key in keys
        },
        "aggregate_group_support": aggregate_support,
        "inner_fold_group_support": fold_support,
        "observed_group_fold_counts": observed_fold_counts,
        "minimum_aggregate_support": MIN_GROUP_AGGREGATE_SUPPORT,
        "minimum_observed_folds": MIN_GROUP_OBSERVED_FOLDS,
        "sufficiently_supported_groups": supported_keys,
        "insufficient_evidence_groups": [
            key for key in keys if not sufficiently_supported[key]
        ],
        "insufficient_evidence_reasons": insufficient_reasons,
        "group_evidence_sufficient": evidence_sufficient,
        "base_worst_group_floor": base_floor,
        "worst_group_floor": floor,
        "worst_group_floor_delta": floor_delta,
        "fold_group_delta": fold_deltas,
        "fold_group_delta_se": fold_delta_se,
        "group_noninferiority_margin": noninferiority_margin,
        "all_sufficient_groups_noninferior": all_supported_noninferior,
        "group_declining_fold_counts": decline_counts,
        "repeated_group_decline": any(
            decline_counts[key] >= 2 for key in supported_keys
        ),
        "numerical_tolerance": NUMERICAL_TOLERANCE,
    }


def _safety_evidence(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    fold_ids: np.ndarray,
    classes: int,
) -> dict[str, Any]:
    base_prediction = _predictions(baseline)
    candidate_prediction = _predictions(candidate)
    base_per_class = _per_class_f1(labels, base_prediction, classes)
    candidate_per_class = _per_class_f1(labels, candidate_prediction, classes)
    base_fold_macro = []
    candidate_fold_macro = []
    class_decline_counts = np.zeros(classes, dtype=np.int64)
    for fold in INNER_FOLDS:
        mask = fold_ids == fold
        base_fold_class = _per_class_f1(labels[mask], base_prediction[mask], classes)
        candidate_fold_class = _per_class_f1(
            labels[mask], candidate_prediction[mask], classes
        )
        base_fold_macro.append(float(base_fold_class.mean()))
        candidate_fold_macro.append(float(candidate_fold_class.mean()))
        class_decline_counts += candidate_fold_class < base_fold_class - NUMERICAL_TOLERANCE
    return {
        "base_per_class_f1": base_per_class.tolist(),
        "per_class_f1": candidate_per_class.tolist(),
        "per_class_f1_delta": (candidate_per_class - base_per_class).tolist(),
        "base_fold_macro_f1": base_fold_macro,
        "fold_macro_f1": candidate_fold_macro,
        "overall_declining_folds": int(
            np.sum(
                np.asarray(candidate_fold_macro)
                < np.asarray(base_fold_macro) - NUMERICAL_TOLERANCE
            )
        ),
        "per_class_declining_fold_counts": class_decline_counts.tolist(),
        "numerical_tolerance": NUMERICAL_TOLERANCE,
    }


def _pair_candidate_prediction(
    probability: np.ndarray,
    base_prediction: np.ndarray,
    top1: np.ndarray,
    top2: np.ndarray,
    gap: np.ndarray,
    pair: tuple[int, int],
    margin: float,
    alpha: float,
) -> np.ndarray:
    source, target = pair
    active = (
        (top1 == target)
        & (top2 == source)
        & (gap <= margin + NUMERICAL_TOLERANCE)
    )
    prediction = base_prediction.copy()
    if not np.any(active):
        return prediction
    target_probability = probability[active, target]
    moved = alpha * target_probability
    source_probability = probability[active, source] + moved
    target_probability = target_probability - moved
    source_wins = source_probability > target_probability
    if source < target:
        source_wins |= source_probability == target_probability
    active_rows = np.flatnonzero(active)
    prediction[active_rows[source_wins]] = source
    return prediction


def _apply_pair_bias(
    probability: np.ndarray, pair: tuple[int, int], margin: float, alpha: float
) -> np.ndarray:
    source, target = pair
    result = np.asarray(probability, dtype=np.float64).copy()
    order = np.argsort(-result, axis=1, kind="stable")[:, :2]
    top1 = order[:, 0]
    top2 = order[:, 1]
    gap = result[np.arange(len(result)), top1] - result[np.arange(len(result)), top2]
    active = (
        (top1 == target)
        & (((top1 == source) & (top2 == target)) | ((top1 == target) & (top2 == source)))
        & (gap <= margin + 1e-15)
    )
    moved = alpha * result[active, target]
    result[active, target] -= moved
    result[active, source] += moved
    return result / result.sum(axis=1, keepdims=True)


def _identity_rule(reason: str) -> dict[str, Any]:
    return {
        "kind": "identity",
        "qualified": False,
        "reason": reason,
        "ordered_pair": None,
        "margin": 0.0,
        "alpha": 0.0,
    }


def _select_pair_rule(
    probability: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    classes: int,
    *,
    ranking: str,
    excluded_pair: tuple[int, int] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    order = np.argsort(-probability, axis=1, kind="stable")[:, :2]
    top1 = order[:, 0]
    top2 = order[:, 1]
    gap = probability[np.arange(len(probability)), top1] - probability[
        np.arange(len(probability)), top2
    ]
    base_prediction = top1
    base_score = _macro(labels, probability, classes)
    base_fold_scores = _fold_scores(labels, probability, fold_ids, classes)
    base_standard_error = float(np.std(base_fold_scores, ddof=1) / np.sqrt(4.0))
    support: dict[tuple[int, int], int] = {}
    for truth, prediction in zip(labels.tolist(), base_prediction.tolist()):
        if truth != prediction:
            pair = (int(truth), int(prediction))
            support[pair] = support.get(pair, 0) + 1
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for pair in sorted(pair for pair, count in support.items() if count >= 3):
        if pair == excluded_pair:
            continue
        for margin in GRID:
            for alpha in GRID:
                prediction = _pair_candidate_prediction(
                    probability,
                    base_prediction,
                    top1,
                    top2,
                    gap,
                    pair,
                    margin,
                    alpha,
                )
                rescued = (base_prediction != labels) & (prediction == labels)
                harmed = (base_prediction == labels) & (prediction != labels)
                rescue = int(rescued.sum())
                harm = int(harmed.sum())
                net = rescue - harm
                fold_net = [
                    int(rescued[fold_ids == fold].sum() - harmed[fold_ids == fold].sum())
                    for fold in INNER_FOLDS
                ]
                safety = _safety_evidence(
                    labels, base_prediction, prediction, fold_ids, classes
                )
                fold_scores = safety["fold_macro_f1"]
                score = float(np.mean(_per_class_f1(labels, prediction, classes)))
                if (
                    rescue < 3
                    or net <= 0
                    or sum(value > 0 for value in fold_net) < 2
                    or score < base_score - NUMERICAL_TOLERANCE
                    or safety["overall_declining_folds"] > 1
                    or min(safety["per_class_f1_delta"]) < -NUMERICAL_TOLERANCE
                ):
                    continue
                rule = {
                    "kind": "pair_bias",
                    "qualified": True,
                    "ordered_pair": list(pair),
                    "support": support[pair],
                    "margin": margin,
                    "alpha": alpha,
                    "inner_macro_f1": score,
                    "base_inner_macro_f1": base_score,
                    "base_fold_macro_f1": base_fold_scores,
                    "fold_macro_f1": fold_scores,
                    "dynamic_one_se": base_standard_error,
                    "rescue": rescue,
                    "harm": harm,
                    "net_rescue": net,
                    "positive_net_folds": sum(value > 0 for value in fold_net),
                    "fold_net_rescue": fold_net,
                    "safety_evidence": safety,
                }
                if ranking == "primary":
                    key = (score, net, rescue, -harm, -alpha, -margin, -pair[0], -pair[1])
                elif ranking == "specialization":
                    key = (net, score, rescue, -harm, -alpha, -margin, -pair[0], -pair[1])
                else:
                    raise ValueError(f"unknown pair ranking: {ranking}")
                candidates.append((key, rule))
    if not candidates:
        reason = "no_qualified_primary_rule" if ranking == "primary" else "no_qualified_secondary_rule"
        return _identity_rule(reason), probability.copy()
    _, rule = max(candidates, key=lambda item: item[0])
    transformed = _apply_pair_bias(
        probability,
        tuple(rule["ordered_pair"]),
        rule["margin"],
        rule["alpha"],
    )
    return rule, transformed


def _select_robustness_weight(
    primary: np.ndarray,
    robustness: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    classes: int,
    groups: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    primary_prediction = primary.argmax(axis=1)
    candidates = []
    prediction_evidence_cache: dict[
        bytes, tuple[dict[str, Any], dict[str, Any] | None]
    ] = {}
    for weight in WEIGHT_GRID:
        mixed = (1.0 - weight) * primary + weight * robustness
        mixed /= mixed.sum(axis=1, keepdims=True)
        mixed_prediction = mixed.argmax(axis=1)
        scores = _fold_scores(labels, mixed_prediction, fold_ids, classes)
        fingerprint = np.ascontiguousarray(mixed_prediction).tobytes()
        cached = prediction_evidence_cache.get(fingerprint)
        if cached is None:
            safety = _safety_evidence(
                labels, primary_prediction, mixed_prediction, fold_ids, classes
            )
            group_safety = (
                _group_safety_evidence(
                    labels,
                    primary_prediction,
                    mixed_prediction,
                    groups,
                    fold_ids,
                    classes,
                )
                if groups is not None
                else None
            )
            prediction_evidence_cache[fingerprint] = (safety, group_safety)
        else:
            safety, group_safety = cached
        candidates.append(
            {
                "weight": weight,
                "probability": mixed,
                "fold_macro_f1": scores,
                "mean_macro_f1": float(np.mean(scores)),
                "std_macro_f1": float(np.std(scores, ddof=1)),
                "minimum_fold_macro_f1": float(np.min(scores)),
                "safety_evidence": safety,
                "group_safety_evidence": group_safety,
            }
        )
    best_mean = max(item["mean_macro_f1"] for item in candidates)
    best_for_se = max(
        (
            item
            for item in candidates
            if np.isclose(
                item["mean_macro_f1"],
                best_mean,
                atol=NUMERICAL_TOLERANCE,
                rtol=0.0,
            )
        ),
        key=lambda item: (-item["std_macro_f1"], -item["weight"]),
    )
    one_se = best_for_se["std_macro_f1"] / np.sqrt(4.0)
    within_one_se = [
        item
        for item in candidates
        if item["weight"] > 0.0
        and item["mean_macro_f1"] >= best_mean - one_se - NUMERICAL_TOLERANCE
    ]
    eligible = [
        item
        for item in within_one_se
        if max(item["safety_evidence"]["per_class_declining_fold_counts"]) < 2
        and (
            item["group_safety_evidence"] is None
            or (
                not item["group_safety_evidence"]["repeated_group_decline"]
                and item["group_safety_evidence"]["group_evidence_sufficient"]
                and item["group_safety_evidence"][
                    "all_sufficient_groups_noninferior"
                ]
                and item["group_safety_evidence"]["worst_group_floor_delta"]
                > NUMERICAL_TOLERANCE
            )
        )
    ]
    if not eligible:
        rule = {
            "kind": "identity_mix",
            "qualified": False,
            "reason": "no_safe_positive_weight_within_best_mean_one_se",
            "weight": 0.0,
            "best_inner_mean": best_mean,
            "best_dynamic_one_se": one_se,
            "safety_evidence": candidates[0]["safety_evidence"],
            "group_safety_evidence": candidates[0]["group_safety_evidence"],
            "rejected_positive_weight_safety_evidence": [
                {
                    "weight": item["weight"],
                    "safety_evidence": item["safety_evidence"],
                    "group_safety_evidence": item["group_safety_evidence"],
                }
                for item in within_one_se
            ],
        }
        return rule, primary.copy()
    selected = max(
        eligible,
        key=lambda item: (
            (
                item["group_safety_evidence"]["worst_group_floor"]
                if item["group_safety_evidence"] is not None
                else float("-inf")
            ),
            item["minimum_fold_macro_f1"],
            item["mean_macro_f1"],
            -item["std_macro_f1"],
            -item["weight"],
        ),
    )
    rule = {
        "kind": "robustness_mix",
        "qualified": True,
        "weight": selected["weight"],
        "fold_macro_f1": selected["fold_macro_f1"],
        "mean_macro_f1": selected["mean_macro_f1"],
        "std_macro_f1": selected["std_macro_f1"],
        "minimum_fold_macro_f1": selected["minimum_fold_macro_f1"],
        "best_inner_mean": best_mean,
        "best_dynamic_one_se": one_se,
        "safety_evidence": selected["safety_evidence"],
        "group_safety_evidence": selected["group_safety_evidence"],
    }
    return rule, selected["probability"]


def _apply_bidirectional_pair_bias(
    probability: np.ndarray,
    primary_pair: tuple[int, int],
    primary_margin: float,
    primary_alpha: float,
    reverse_margin: float,
    reverse_alpha: float,
) -> np.ndarray:
    result = np.asarray(probability, dtype=np.float64).copy()
    order = np.argsort(-result, axis=1, kind="stable")[:, :2]
    top1, top2 = order[:, 0], order[:, 1]
    gap = result[np.arange(len(result)), top1] - result[np.arange(len(result)), top2]
    for (source, target), margin, alpha in (
        (primary_pair, primary_margin, primary_alpha),
        ((primary_pair[1], primary_pair[0]), reverse_margin, reverse_alpha),
    ):
        active = (
            (top1 == target)
            & (top2 == source)
            & (gap <= margin + NUMERICAL_TOLERANCE)
        )
        moved = alpha * result[active, target]
        result[active, target] -= moved
        result[active, source] += moved
    return result / result.sum(axis=1, keepdims=True)


def _recall(labels: np.ndarray, prediction: np.ndarray, class_index: int) -> float:
    selected = labels == class_index
    return float(np.mean(prediction[selected] == class_index)) if selected.any() else 0.0


def _select_bidirectional_specialist_rule(
    probability: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    classes: int,
    groups: dict[str, np.ndarray],
    primary_pair: tuple[int, int],
    consensus_hash: str,
) -> tuple[dict[str, Any], np.ndarray]:
    source, target = primary_pair
    reverse_pair = (target, source)
    order = np.argsort(-probability, axis=1, kind="stable")[:, :2]
    top1, top2 = order[:, 0], order[:, 1]
    gap = probability[np.arange(len(probability)), top1] - probability[
        np.arange(len(probability)), top2
    ]
    base_prediction = top1
    base_score = _macro(labels, base_prediction, classes)
    base_fold_scores = np.asarray(
        _fold_scores(labels, base_prediction, fold_ids, classes), dtype=float
    )
    base_one_se = float(np.std(base_fold_scores, ddof=1) / np.sqrt(4.0))
    base_fold_class = np.asarray(
        [
            _per_class_f1(labels[fold_ids == fold], base_prediction[fold_ids == fold], classes)
            for fold in INNER_FOLDS
        ]
    )
    base_class_se = np.std(base_fold_class, axis=0, ddof=1) / np.sqrt(4.0)
    base_class = _per_class_f1(labels, base_prediction, classes)
    pair_indices = np.asarray(sorted(primary_pair), dtype=np.int64)
    outside_indices = np.asarray(
        [index for index in range(classes) if index not in set(primary_pair)], dtype=np.int64
    )
    base_pair_f1 = float(np.mean(base_class[pair_indices]))
    base_outside_f1 = (
        float(np.mean(base_class[outside_indices])) if len(outside_indices) else 0.0
    )
    base_source_recall = _recall(labels, base_prediction, source)
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    seen_predictions: set[bytes] = set()
    for primary_margin in GRID:
        for primary_alpha in GRID:
            primary_prediction = _pair_candidate_prediction(
                probability, base_prediction, top1, top2, gap,
                primary_pair, primary_margin, primary_alpha,
            )
            for reverse_margin in GRID:
                for reverse_alpha in GRID:
                    prediction = _pair_candidate_prediction(
                        probability, primary_prediction, top1, top2, gap,
                        reverse_pair, reverse_margin, reverse_alpha,
                    )
                    fingerprint = np.ascontiguousarray(prediction).tobytes()
                    if fingerprint in seen_predictions:
                        continue
                    seen_predictions.add(fingerprint)
                    rescued = (base_prediction != labels) & (prediction == labels)
                    harmed = (base_prediction == labels) & (prediction != labels)
                    rescue, harm = int(rescued.sum()), int(harmed.sum())
                    net = rescue - harm
                    fold_net = [
                        int(rescued[fold_ids == fold].sum() - harmed[fold_ids == fold].sum())
                        for fold in INNER_FOLDS
                    ]
                    score = _macro(labels, prediction, classes)
                    fold_scores = np.asarray(
                        _fold_scores(labels, prediction, fold_ids, classes), dtype=float
                    )
                    fold_delta = fold_scores - base_fold_scores
                    overall_delta_se = float(
                        np.std(fold_delta, ddof=1) / np.sqrt(4.0)
                    )
                    class_score = _per_class_f1(labels, prediction, classes)
                    class_delta = class_score - base_class
                    fold_class_delta = np.asarray(
                        [
                            _per_class_f1(
                                labels[fold_ids == fold], prediction[fold_ids == fold], classes
                            ) - base_fold_class[fold]
                            for fold in INNER_FOLDS
                        ]
                    )
                    class_delta_se = np.std(fold_class_delta, axis=0, ddof=1) / np.sqrt(4.0)
                    class_decline_counts = np.sum(
                        fold_class_delta < -NUMERICAL_TOLERANCE, axis=0
                    )
                    pair_decline_counts = np.sum(
                        fold_class_delta[:, pair_indices]
                        < -class_delta_se[pair_indices] - NUMERICAL_TOLERANCE,
                        axis=0,
                    )
                    pair_f1 = float(np.mean(class_score[pair_indices]))
                    source_recall = _recall(labels, prediction, source)
                    outside_f1 = (
                        float(np.mean(class_score[outside_indices]))
                        if len(outside_indices) else 0.0
                    )
                    outside_fold_delta = (
                        np.mean(fold_class_delta[:, outside_indices], axis=1)
                        if len(outside_indices) else np.zeros(4)
                    )
                    outside_delta_se = float(
                        np.std(outside_fold_delta, ddof=1) / np.sqrt(4.0)
                    )
                    group_safety = _group_safety_evidence(
                        labels, base_prediction, prediction, groups, fold_ids, classes
                    )
                    class_margin = float(np.min(class_delta + base_class_se))
                    supported_group_margins = [
                        margin
                        for margin in group_safety[
                            "group_noninferiority_margin"
                        ].values()
                        if margin is not None
                    ]
                    group_margin = (
                        float(min(supported_group_margins))
                        if group_safety["group_evidence_sufficient"]
                        and supported_group_margins
                        else float("-inf")
                    )
                    materially_declining = int(
                        np.sum(fold_delta < -overall_delta_se - NUMERICAL_TOLERANCE)
                    )
                    outside_repeated = bool(
                        len(outside_indices)
                        and np.any(class_decline_counts[outside_indices] >= 2)
                    )
                    if (
                        rescue < 3
                        or net <= 0
                        or sum(value > 0 for value in fold_net) < 2
                        or score < base_score - base_one_se - NUMERICAL_TOLERANCE
                        or materially_declining > 1
                        or np.any(class_delta < -base_class_se - NUMERICAL_TOLERANCE)
                        or outside_repeated
                        or np.any(pair_decline_counts >= 2)
                        or (
                            pair_f1 - base_pair_f1 < 0.05 - NUMERICAL_TOLERANCE
                            and source_recall - base_source_recall
                            < 0.05 - NUMERICAL_TOLERANCE
                        )
                        or outside_f1 - base_outside_f1
                        < -outside_delta_se - NUMERICAL_TOLERANCE
                        or group_safety["repeated_group_decline"]
                        or not group_safety["group_evidence_sufficient"]
                        or not group_safety["all_sufficient_groups_noninferior"]
                    ):
                        continue
                    rule = {
                        "kind": "bidirectional_pair_bias",
                        "qualified": True,
                        "ordered_pair": list(primary_pair),
                        "reverse_ordered_pair": list(reverse_pair),
                        "margin": primary_margin,
                        "alpha": primary_alpha,
                        "reverse_margin": reverse_margin,
                        "reverse_alpha": reverse_alpha,
                        "inner_macro_f1": score,
                        "base_inner_macro_f1": base_score,
                        "base_fold_macro_f1": base_fold_scores.tolist(),
                        "fold_macro_f1": fold_scores.tolist(),
                        "base_dynamic_one_se": base_one_se,
                        "materially_declining_overall_folds": materially_declining,
                        "rescue": rescue,
                        "harm": harm,
                        "net_rescue": net,
                        "positive_net_folds": sum(value > 0 for value in fold_net),
                        "fold_net_rescue": fold_net,
                        "pair_macro_f1": pair_f1,
                        "base_pair_macro_f1": base_pair_f1,
                        "pair_macro_f1_delta": pair_f1 - base_pair_f1,
                        "source_recall": source_recall,
                        "base_source_recall": base_source_recall,
                        "source_recall_delta": source_recall - base_source_recall,
                        "outside_pair_macro_f1": outside_f1,
                        "base_outside_pair_macro_f1": base_outside_f1,
                        "outside_pair_delta_se": outside_delta_se,
                        "per_class_f1_delta": class_delta.tolist(),
                        "base_per_class_fold_se": base_class_se.tolist(),
                        "per_class_fold_delta_se": class_delta_se.tolist(),
                        "per_class_declining_fold_counts": class_decline_counts.tolist(),
                        "pair_class_material_declining_fold_counts": pair_decline_counts.tolist(),
                        "minimum_class_noninferiority_margin": class_margin,
                        "minimum_group_noninferiority_margin": group_margin,
                        "group_safety_evidence": group_safety,
                        "selection_transcript_dependency_hash": consensus_hash,
                    }
                    key = (
                        class_margin, group_margin, score, pair_f1, source_recall,
                        net, rescue, -harm, -primary_alpha, -reverse_alpha,
                        -primary_margin, -reverse_margin,
                    )
                    candidates.append((key, rule))
    if not candidates:
        rule = _identity_rule("no_safe_bidirectional_consensus_rule")
        rule.update(
            {
                "reverse_ordered_pair": None,
                "reverse_margin": 0.0,
                "reverse_alpha": 0.0,
                "consensus_ordered_pair": list(primary_pair),
                "selection_transcript_dependency_hash": consensus_hash,
            }
        )
        return rule, probability.copy()
    _, rule = max(candidates, key=lambda item: item[0])
    transformed = _apply_bidirectional_pair_bias(
        probability,
        primary_pair,
        rule["margin"],
        rule["alpha"],
        rule["reverse_margin"],
        rule["reverse_alpha"],
    )
    return rule, transformed


def _derive_consensus_collision_pair(
    rows: list[dict[str, Any]], classes: tuple[str, ...]
) -> dict[str, Any]:
    aggregates: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("strategy") != "specialization_challenger":
            continue
        variant = row.get("selected_variant")
        evidence = row.get("gate_evidence")
        if not isinstance(variant, dict) or not isinstance(evidence, dict):
            raise ValueError("specialization transcript lacks selected gate evidence")
        ordered = variant.get("ordered_pair")
        if not isinstance(ordered, (list, tuple)) or len(ordered) != 2:
            raise ValueError("specialization transcript ordered pair is invalid")
        source, target = map(int, ordered)
        if source == target or min(source, target) < 0 or max(source, target) >= len(classes):
            raise ValueError("specialization transcript class pair is invalid")
        net = int(evidence.get("net_rescue"))
        unordered = tuple(sorted((source, target)))
        item = aggregates.setdefault(
            unordered, {"total_net_rescue": 0, "count": 0, "orientations": {}}
        )
        item["total_net_rescue"] += net
        item["count"] += 1
        orientation = item["orientations"].setdefault(
            (source, target), {"total_net_rescue": 0, "count": 0}
        )
        orientation["total_net_rescue"] += net
        orientation["count"] += 1
    if not aggregates:
        raise ValueError("specialization transcript contains no collision pair evidence")
    unordered = max(
        aggregates,
        key=lambda pair: (
            aggregates[pair]["total_net_rescue"],
            aggregates[pair]["count"],
            -pair[0],
            -pair[1],
        ),
    )
    orientations = aggregates[unordered]["orientations"]
    primary = max(
        orientations,
        key=lambda pair: (
            orientations[pair]["total_net_rescue"],
            orientations[pair]["count"],
            -pair[0],
            -pair[1],
        ),
    )
    serializable = {
        f"{pair[0]}:{pair[1]}": {
            "total_net_rescue": value["total_net_rescue"],
            "count": value["count"],
            "orientations": {
                f"{orientation[0]}:{orientation[1]}": detail
                for orientation, detail in sorted(value["orientations"].items())
            },
        }
        for pair, value in sorted(aggregates.items())
    }
    evidence = {
        "source": "nested_selection_records.specialization_challenger.inner_selection",
        "unordered_pair": list(unordered),
        "unordered_class_names": [classes[index] for index in unordered],
        "primary_ordered_pair": list(primary),
        "primary_class_names": [classes[index] for index in primary],
        "reverse_ordered_pair": [primary[1], primary[0]],
        "aggregates": serializable,
        "tie_break": ["total_net_rescue", "count", "smaller_class_indices"],
    }
    return {**evidence, "derivation_hash": _canonical_hash(evidence)}


def _derive_unit_collision_pairs(
    rows: list[dict[str, Any]], classes: tuple[str, ...]
) -> dict[str, Any]:
    units: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        for fold in OUTER_FOLDS:
            unit_rows = [
                row
                for row in rows
                if row.get("strategy") == "specialization_challenger"
                and int(row.get("seed", -1)) == seed
                and int(row.get("fold", -1)) == fold
            ]
            if len(unit_rows) != 1:
                raise ValueError(
                    f"seed {seed} outer fold {fold}: specialization transcript must be unique"
                )
            unit = _derive_consensus_collision_pair(unit_rows, classes)
            stable = {
                **{key: value for key, value in unit.items() if key != "derivation_hash"},
                "seed": seed,
                "outer_fold": fold,
                "transcript_row_hash": _canonical_hash(unit_rows[0]),
                "scope": "this_seed_outer_fold_inner_selection_only",
            }
            units[_unit_key(seed, fold)] = {
                **stable,
                "derivation_hash": _canonical_hash(stable),
            }
    evidence = {
        "source": "nested_selection_records.specialization_challenger.per_seed_outer_fold_inner_selection",
        "scope": "independent_seed_outer_fold_units",
        "units": dict(sorted(units.items())),
    }
    return {**evidence, "derivation_hash": _canonical_hash(evidence)}


def _inner_key(lane: str, seed: int, fold: int) -> str:
    return f"{lane}__seed_{seed}__fold_{fold}"


def _expected_inner_fold_ids(labels: np.ndarray, seed: int, outer_fold: int) -> np.ndarray:
    splitter = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=(seed * 1009 + outer_fold * 9173) % (2**31 - 1),
    )
    result = np.full(len(labels), -1, dtype=np.int8)
    for inner_fold, (_, valid) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        result[valid] = inner_fold
    return result


@dataclass(frozen=True)
class Inputs:
    raw: Path
    manifest_path: Path
    inner_path: Path
    records_path: Path
    selection_transcript_path: Path
    strategy_path: Path
    train_path: Path
    fold_path: Path
    manifest: dict[str, Any]
    input_hashes: dict[str, str]
    train: pd.DataFrame
    features: pd.DataFrame
    labels: np.ndarray
    classes: tuple[str, ...]
    folds: dict[int, np.ndarray]
    inner_arrays: dict[str, np.ndarray]
    inner_records: dict[str, dict[str, Any]]
    outer_arrays: dict[str, np.ndarray]
    unit_collision_derivations: dict[str, Any]
    group_cache: dict[tuple[int, int], tuple[dict[str, np.ndarray], dict[str, Any]]]


def _load_inputs(
    nested_raw: Path,
    train_path: Path,
    fold_assignments_path: Path,
    target: str,
    identifier: str,
) -> Inputs:
    raw = Path(nested_raw).resolve()
    if raw.name.casefold() in FORBIDDEN_BASENAMES or not raw.is_dir():
        raise ValueError("nested_raw must be an existing artifact directory")
    paths = {name: _guard_input(raw / name, name) for name in UPSTREAM_FILES}
    train_path = _guard_input(train_path, "train")
    fold_path = _guard_input(fold_assignments_path, "fold_assignments")
    all_paths = [*paths.values(), train_path, fold_path]
    if len({str(path).casefold() for path in all_paths}) != len(all_paths):
        raise ValueError("refinement inputs must be distinct files")
    input_hashes = {
        **{name: _sha256(path) for name, path in paths.items()},
        "train": _sha256(train_path),
        "fold_assignments": _sha256(fold_path),
    }
    manifest = json.loads(paths["nested_run_manifest.json"].read_text(encoding="utf-8"))
    if manifest.get("outer_validation_used_for_selection") is not False or manifest.get("test_accessed") is not False:
        raise ValueError("upstream Nested manifest is not train-only")
    if tuple(manifest.get("seeds", ())) != SEEDS or int(manifest.get("outer_folds", -1)) != 5 or int(manifest.get("inner_folds", -1)) != 4:
        raise ValueError("upstream Nested seed/fold contract mismatch")
    expected_artifacts = manifest.get("artifact_hashes", {})
    for name in UPSTREAM_FILES[1:]:
        if expected_artifacts.get(name) != input_hashes[name]:
            raise ValueError(f"upstream artifact hash mismatch: {name}")
    if manifest.get("fold_hash") != input_hashes["fold_assignments"]:
        raise ValueError("upstream fold assignment hash mismatch")
    if manifest.get("train_hash") != input_hashes["train"]:
        raise ValueError("upstream Train hash mismatch")

    train = pd.read_csv(train_path)
    if (
        identifier not in train
        or target not in train
        or train[identifier].astype(str).duplicated().any()
    ):
        raise ValueError("full Train feature/ID/label input is invalid")
    features = train.drop(columns=[identifier, target])
    if features.shape[1] == 0:
        raise ValueError("full Train input requires at least one feature column")
    classes = tuple(map(str, manifest.get("class_names", ())))
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("upstream class order is invalid")
    encoded = train[target].astype(str).map({name: index for index, name in enumerate(classes)})
    if encoded.isna().any():
        raise ValueError("Train labels do not match upstream class order")
    labels = encoded.to_numpy(dtype=np.int32)

    assignments = pd.read_csv(fold_path)
    required = {"row_index", identifier, "seed", "fold"}
    if not required.issubset(assignments):
        raise ValueError(f"fold assignments require {sorted(required)}")
    folds: dict[int, np.ndarray] = {}
    expected_rows = np.arange(len(train), dtype=np.int64)
    expected_ids = train[identifier].astype(str).to_numpy()
    for seed in SEEDS:
        frame = assignments.loc[assignments.seed == seed].sort_values("row_index", kind="stable")
        if len(frame) != len(train) or not np.array_equal(frame.row_index.to_numpy(), expected_rows):
            raise ValueError(f"seed {seed}: fold row order mismatch")
        if not np.array_equal(frame[identifier].astype(str).to_numpy(), expected_ids):
            raise ValueError(f"seed {seed}: fold ID order mismatch")
        values = frame.fold.to_numpy(dtype=np.int16)
        if set(values.tolist()) != set(OUTER_FOLDS):
            raise ValueError(f"seed {seed}: outer fold coverage mismatch")
        folds[seed] = values
    if len(assignments) != len(train) * len(SEEDS):
        raise ValueError("fold assignments contain unexpected rows")

    selection_payload = json.loads(
        paths["nested_selection_records.json"].read_text(encoding="utf-8")
    )
    selection_rows = selection_payload.get("selections")
    if not isinstance(selection_rows, list) or len(selection_rows) != 45:
        raise ValueError("selected variant transcript must contain exactly 45 records")
    selected_keys = {
        (str(row.get("strategy")), int(row.get("seed", -1)), int(row.get("fold", -1)))
        for row in selection_rows
    }
    expected_selected_keys = {
        (lane, seed, fold)
        for lane in LANES
        for seed in SEEDS
        for fold in OUTER_FOLDS
    }
    if len(selected_keys) != len(selection_rows) or selected_keys != expected_selected_keys:
        raise ValueError("selected variant transcript key coverage mismatch")
    if any(
        row.get("outer_validation_used_for_selection") is not False
        or row.get("selection_source") != "outer_train_inner_oof_only"
        for row in selection_rows
    ):
        raise ValueError("selected variant transcript permits outer-label selection")
    unit_collision_derivations = _derive_unit_collision_pairs(selection_rows, classes)

    record_payload = json.loads(paths["nested_inner_selection_records.json"].read_text(encoding="utf-8"))
    if record_payload.get("selection_uses_outer_validation") is not False:
        raise ValueError("inner transcript permits outer-label selection")
    rows = record_payload.get("records")
    if not isinstance(rows, list) or len(rows) != 45:
        raise ValueError("inner transcript must contain exactly 45 records")
    records = {str(row.get("array_key")): row for row in rows}
    expected_keys = {_inner_key(lane, seed, fold) for lane in LANES for seed in SEEDS for fold in OUTER_FOLDS}
    if set(records) != expected_keys or len(records) != len(rows):
        raise ValueError("inner transcript key coverage mismatch")
    inner_arrays: dict[str, np.ndarray] = {}
    with np.load(paths["nested_inner_selection_probability.npz"], allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError("inner probability key coverage mismatch")
        for key in sorted(expected_keys):
            row = records[key]
            seed = int(row.get("seed", -1)); outer_fold = int(row.get("outer_fold", -1))
            lane = str(row.get("strategy"))
            if key != _inner_key(lane, seed, outer_fold) or row.get("outer_validation_used") is not False:
                raise ValueError(f"{key}: inner record identity/leakage mismatch")
            expected_index = np.flatnonzero(folds[seed] != outer_fold)
            row_index = np.asarray(row.get("row_indices", ()), dtype=np.int64)
            if not np.array_equal(row_index, expected_index) or np.any(np.diff(row_index) <= 0):
                raise ValueError(f"{key}: inner row index alignment mismatch")
            if list(map(str, row.get("row_ids", ()))) != train.iloc[row_index][identifier].astype(str).tolist():
                raise ValueError(f"{key}: inner row ID alignment mismatch")
            fold_ids = np.asarray(row.get("inner_fold_ids", ()), dtype=np.int8)
            if fold_ids.shape != (len(row_index),) or set(fold_ids.tolist()) != set(INNER_FOLDS):
                raise ValueError(f"{key}: inner fold coverage mismatch")
            expected_inner_folds = _expected_inner_fold_ids(labels[row_index], seed, outer_fold)
            if not np.array_equal(fold_ids, expected_inner_folds):
                raise ValueError(f"{key}: inner fold IDs differ from deterministic split")
            coverage = np.asarray(row.get("inner_oof_coverage", ()), dtype=np.int8)
            if coverage.shape != fold_ids.shape or not np.all(coverage == 1):
                raise ValueError(f"{key}: inner OOF coverage must equal one")
            probability = _probability(archive[key], (len(row_index), len(classes)), key)
            if _array_hash(probability) != row.get("array_hash"):
                raise ValueError(f"{key}: inner probability hash mismatch")
            inner_arrays[key] = probability.copy()

    outer_arrays = {}
    with np.load(paths["nested_strategy_probability.npz"], allow_pickle=False) as archive:
        if set(archive.files) != set(LANES):
            raise ValueError("outer strategy lane coverage mismatch")
        for lane in LANES:
            outer_arrays[lane] = _probability(
                archive[lane], (len(SEEDS), len(train), len(classes)), f"outer:{lane}"
            ).copy()
    return Inputs(
        raw=raw,
        manifest_path=paths["nested_run_manifest.json"],
        inner_path=paths["nested_inner_selection_probability.npz"],
        records_path=paths["nested_inner_selection_records.json"],
        selection_transcript_path=paths["nested_selection_records.json"],
        strategy_path=paths["nested_strategy_probability.npz"],
        train_path=train_path,
        fold_path=fold_path,
        manifest=manifest,
        input_hashes=input_hashes,
        train=train,
        features=features,
        labels=labels,
        classes=classes,
        folds=folds,
        inner_arrays=inner_arrays,
        inner_records=records,
        outer_arrays=outer_arrays,
        unit_collision_derivations=unit_collision_derivations,
        group_cache={},
    )


def _unit_key(seed: int, fold: int) -> str:
    return f"seed_{seed}__fold_{fold}"


def _selection_record(
    lane: str, seed: int, fold: int, selection: dict[str, Any], inner: np.ndarray,
    labels: np.ndarray, classes: int, row_index: np.ndarray,
    inner_dependency_hashes: dict[str, str],
    parent_selection_hashes: dict[str, str],
) -> dict[str, Any]:
    dependency_hashes = dict(sorted(inner_dependency_hashes.items()))
    parent_hashes = dict(sorted(parent_selection_hashes.items()))
    if not dependency_hashes or any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, str)
        or len(value) != 64
        for name, value in dependency_hashes.items()
    ):
        raise ValueError("selection inner dependency hashes are invalid")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, str)
        or len(value) != 64
        for name, value in parent_hashes.items()
    ):
        raise ValueError("selection parent hashes are invalid")
    stable = {
        "strategy": lane,
        "seed": seed,
        "outer_fold": fold,
        "selection": selection,
        "qualified": bool(selection.get("qualified")),
        "inner_macro_f1": _macro(labels, inner, classes),
        "inner_array_key": _inner_key(lane, seed, fold),
        "inner_row_indices_hash": _canonical_hash(row_index.tolist()),
        "inner_dependency_hashes": dependency_hashes,
        "parent_selection_hashes": parent_hashes,
        "refined_inner_array_hash": _array_hash(inner),
        "selection_source": "outer_train_inner_oof_only",
        "selection_uses_outer_validation": False,
    }
    return {**stable, "selection_hash": _canonical_hash(stable)}


def _compute_unit_impl(
    inputs: Inputs, seed: int, fold: int
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    seed_index = SEEDS.index(seed)
    primary_key = _inner_key("primary", seed, fold)
    robust_key = _inner_key("robustness_challenger", seed, fold)
    primary_inner = inputs.inner_arrays[primary_key]
    robust_inner = inputs.inner_arrays[robust_key]
    record = inputs.inner_records[primary_key]
    row_index = np.asarray(record["row_indices"], dtype=np.int64)
    fold_ids = np.asarray(record["inner_fold_ids"], dtype=np.int8)
    inner_labels = inputs.labels[row_index]
    classes = len(inputs.classes)
    cache_key = (seed, fold)
    if cache_key not in inputs.group_cache:
        inputs.group_cache[cache_key] = _inner_fold_safe_groups(
            inputs.features.iloc[row_index].reset_index(drop=True), fold_ids
        )
    inner_groups, group_thresholds = inputs.group_cache[cache_key]

    primary_rule, refined_primary_inner = _select_pair_rule(
        primary_inner, inner_labels, fold_ids, classes, ranking="primary"
    )
    robust_rule, refined_robust_inner = _select_robustness_weight(
        refined_primary_inner,
        robust_inner,
        inner_labels,
        fold_ids,
        classes,
        inner_groups,
    )
    robust_rule["fold_safe_group_thresholds"] = group_thresholds
    unit_collision = inputs.unit_collision_derivations["units"][_unit_key(seed, fold)]
    collision_pair = tuple(map(int, unit_collision["primary_ordered_pair"]))
    specialization_rule, refined_specialization_inner = (
        _select_bidirectional_specialist_rule(
            refined_primary_inner,
            inner_labels,
            fold_ids,
            classes,
            inner_groups,
            collision_pair,
            unit_collision["transcript_row_hash"],
        )
    )
    specialization_rule["unit_collision_derivation"] = unit_collision
    specialization_rule["unit_collision_derivation_hash"] = unit_collision[
        "derivation_hash"
    ]
    specialization_rule["fold_safe_group_thresholds"] = group_thresholds

    valid_index = np.flatnonzero(inputs.folds[seed] == fold)
    base_primary_outer = inputs.outer_arrays["primary"][seed_index, valid_index]
    base_robust_outer = inputs.outer_arrays["robustness_challenger"][seed_index, valid_index]
    refined_primary_outer = (
        base_primary_outer.copy()
        if primary_rule["kind"] == "identity"
        else _apply_pair_bias(
            base_primary_outer,
            tuple(primary_rule["ordered_pair"]),
            primary_rule["margin"],
            primary_rule["alpha"],
        )
    )
    weight = float(robust_rule["weight"])
    refined_robust_outer = (1.0 - weight) * refined_primary_outer + weight * base_robust_outer
    refined_robust_outer /= refined_robust_outer.sum(axis=1, keepdims=True)
    refined_specialization_outer = (
        refined_primary_outer.copy()
        if specialization_rule["kind"] == "identity"
        else _apply_bidirectional_pair_bias(
            refined_primary_outer,
            tuple(specialization_rule["ordered_pair"]),
            specialization_rule["margin"],
            specialization_rule["alpha"],
            specialization_rule["reverse_margin"],
            specialization_rule["reverse_alpha"],
        )
    )
    arrays = {
        "inner_primary": refined_primary_inner,
        "inner_robustness_challenger": refined_robust_inner,
        "inner_specialization_challenger": refined_specialization_inner,
        "outer_row_indices": valid_index,
        "outer_primary": refined_primary_outer,
        "outer_robustness_challenger": refined_robust_outer,
        "outer_specialization_challenger": refined_specialization_outer,
    }
    upstream_primary_hash = inputs.inner_records[primary_key]["array_hash"]
    refined_primary_hash = _array_hash(refined_primary_inner)
    primary_record = _selection_record(
        "primary",
        seed,
        fold,
        primary_rule,
        refined_primary_inner,
        inner_labels,
        classes,
        row_index,
        {"upstream_primary": upstream_primary_hash},
        {},
    )
    primary_parent = {"primary": primary_record["selection_hash"]}
    robust_record = _selection_record(
        "robustness_challenger",
        seed,
        fold,
        robust_rule,
        refined_robust_inner,
        inner_labels,
        classes,
        row_index,
        {
            "refined_primary": refined_primary_hash,
            "upstream_primary": upstream_primary_hash,
            "upstream_robustness_challenger": inputs.inner_records[robust_key][
                "array_hash"
            ],
        },
        primary_parent,
    )
    specialization_record = _selection_record(
        "specialization_challenger",
        seed,
        fold,
        specialization_rule,
        refined_specialization_inner,
        inner_labels,
        classes,
        row_index,
        {
            "refined_primary": refined_primary_hash,
            "upstream_primary": upstream_primary_hash,
        },
        primary_parent,
    )
    selections = [primary_record, robust_record, specialization_record]
    return arrays, selections


def _compute_unit(
    inputs: Inputs, seed: int, fold: int
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Compute one persisted unit; semantic replay calls the implementation directly."""
    return _compute_unit_impl(inputs, seed, fold)


def _algorithm_contract(
    unit_collision_derivations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = {
        "name": "test_007_nested_refinement_v4",
        "margin_alpha_grid": list(GRID),
        "robustness_weight_grid": list(WEIGHT_GRID),
        "minimum_ordered_error_support": 3,
        "minimum_rescues": 3,
        "minimum_positive_net_inner_folds": 2,
        "pair_safety_constraints": {
            "aggregate_macro_f1": "candidate >= baseline",
            "maximum_declining_inner_folds": 1,
            "aggregate_per_class_f1": "no class below baseline beyond numerical tolerance",
            "numerical_tolerance": NUMERICAL_TOLERANCE,
        },
        "primary_ranking": ["inner_macro_f1", "net_rescue", "rescue", "-harm", "smaller_alpha", "smaller_margin"],
        "robustness_ranking": ["worst_group_floor", "minimum_inner_fold_macro_f1", "mean", "-std", "smaller_weight"],
        "robustness_safety_constraint": "exact overall one-SE; no repeated class or fold-safe burden/novelty decline; strict worst-group-floor improvement versus refined primary",
        "subgroup_safety": {
            "status": "fold_safe",
            "dimensions": ["burden", "novelty"],
            "groups_per_dimension": 3,
            "minimum_aggregate_group_support": MIN_GROUP_AGGREGATE_SUPPORT,
            "minimum_observed_inner_folds": MIN_GROUP_OBSERVED_FOLDS,
            "under_supported_group_status": "insufficient_evidence_with_null_standard_error",
            "eligibility": "all sufficiently-supported groups must have noninferiority margin >= 0; at least one sufficiently-supported group is required in every dimension",
            "contract": "for each seed/outer-fold and each of 4 inner folds, burden quantiles and token frequencies/novelty quantiles are fitted only on inner-train and applied once to inner-valid; assignments are cached once per unit and never recomputed per candidate",
            "input_scope": "full Train features and labels only",
        },
        "specialization_kind": "per_seed_outer_fold_bidirectional_pair_bias",
        "specialization_pair_oracle": "each seed/outer-fold uses only its own immutable nested_selection_records specialization inner-selection transcript",
        "specialization_ranking": ["minimum_class_noninferiority_margin", "minimum_group_noninferiority_margin", "inner_macro_f1", "pair_macro_f1", "source_recall", "net_rescue", "rescue", "-harm", "smaller_alphas", "smaller_margins"],
        "specialization_material_gain_threshold": 0.05,
        "one_se": "sample_std_of_4_inner_fold_macro_f1/sqrt(4)",
        "selection_uses_outer_validation": False,
    }
    if unit_collision_derivations is not None:
        contract["unit_collision_derivations"] = unit_collision_derivations
    return contract


def _identity_payload(inputs: Inputs, target: str, identifier: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "input_hashes": inputs.input_hashes,
        "upstream_run_identity_hash": inputs.manifest.get("run_identity_hash"),
        "upstream_source_bundle": inputs.manifest.get("source_bundle_before"),
        "source_hash": _sha256(Path(__file__).resolve()),
        "algorithm": _algorithm_contract(inputs.unit_collision_derivations),
        "unit_collision_derivations": inputs.unit_collision_derivations,
        "target_column": target,
        "id_column": identifier,
        "row_order_hash": _canonical_hash(inputs.train[identifier].astype(str).tolist()),
    }


def _validate_completed(output: Path, identity_hash: str) -> Path:
    _assert_plain(output, directory=True, role="refinement output")
    if any(not (output / name).is_file() for name in OUTPUTS):
        raise RuntimeError("refinement output is incomplete")
    for name in OUTPUTS:
        _assert_plain(output / name, directory=False, role=f"refinement output {name}")
    manifest = json.loads((output / "refined_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("run_identity_hash") != identity_hash:
        raise RuntimeError("existing refinement output has a different run identity")
    artifact_hashes = manifest.get("artifact_hashes", {})
    expected_names = set(OUTPUTS) - {"refined_manifest.json"}
    if set(artifact_hashes) != expected_names:
        raise RuntimeError("existing refinement artifact hash inventory is incomplete")
    for name, expected in artifact_hashes.items():
        if _sha256(output / name) != expected:
            raise RuntimeError(f"existing refinement artifact hash mismatch: {name}")
    return output


def _read_plain_json(path: Path, role: str) -> dict[str, Any]:
    _assert_plain(path, directory=False, role=role)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{role} must contain a JSON object")
    return value


def _checkpoint_manifest(identity_hash: str, nonce: str, expected_units: set[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_identity_hash": identity_hash,
        "stage_nonce": nonce,
        "expected_units": sorted(expected_units),
        "completed": {},
    }


def _validate_checkpoint_manifest(
    value: dict[str, Any], identity_hash: str, nonce: str, expected_units: set[str]
) -> dict[str, Any]:
    if set(value) != {"schema_version", "run_identity_hash", "stage_nonce", "expected_units", "completed"}:
        raise RuntimeError("checkpoint manifest schema mismatch")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["run_identity_hash"] != identity_hash
        or value["stage_nonce"] != nonce
        or value["expected_units"] != sorted(expected_units)
        or not isinstance(value["completed"], dict)
        or not set(value["completed"]).issubset(expected_units)
    ):
        raise RuntimeError("checkpoint manifest identity/coverage mismatch")
    for key, entry in value["completed"].items():
        if not isinstance(entry, dict) or set(entry) != {"npz_sha256", "json_sha256", "member_hashes"}:
            raise RuntimeError(f"checkpoint manifest entry schema mismatch: {key}")
        if not isinstance(entry["member_hashes"], dict):
            raise RuntimeError(f"checkpoint member hashes are invalid: {key}")
    return value["completed"]


def _validate_selection_rows(rows: Any, seed: int, fold: int) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(LANES):
        raise RuntimeError(f"{_unit_key(seed, fold)}: checkpoint requires exactly three selections")
    expected_fields = {
        "strategy", "seed", "outer_fold", "selection", "qualified", "inner_macro_f1",
        "inner_array_key", "inner_row_indices_hash", "inner_dependency_hashes",
        "parent_selection_hashes",
        "refined_inner_array_hash", "selection_source", "selection_uses_outer_validation",
        "selection_hash",
    }
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise RuntimeError(f"{_unit_key(seed, fold)}: checkpoint selection schema mismatch")
        lane = str(row["strategy"])
        if (
            lane not in LANES
            or lane in indexed
            or row["seed"] != seed
            or row["outer_fold"] != fold
            or row["inner_array_key"] != _inner_key(lane, seed, fold)
            or row["selection_source"] != "outer_train_inner_oof_only"
            or row["selection_uses_outer_validation"] is not False
            or not isinstance(row["selection"], dict)
            or not isinstance(row["inner_dependency_hashes"], dict)
            or not row["inner_dependency_hashes"]
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(value, str)
                or len(value) != 64
                for name, value in row["inner_dependency_hashes"].items()
            )
            or not isinstance(row["parent_selection_hashes"], dict)
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(value, str)
                or len(value) != 64
                for name, value in row["parent_selection_hashes"].items()
            )
            or row["qualified"] is not bool(row["selection"].get("qualified"))
        ):
            raise RuntimeError(f"{_unit_key(seed, fold)}: checkpoint selection identity mismatch")
        stable = row.copy()
        recorded_hash = stable.pop("selection_hash")
        if recorded_hash != _canonical_hash(stable):
            raise RuntimeError(f"{_unit_key(seed, fold)}: checkpoint selection hash mismatch")
        indexed[lane] = row
    if set(indexed) != set(LANES):
        raise RuntimeError(f"{_unit_key(seed, fold)}: checkpoint lane coverage mismatch")
    return [indexed[lane] for lane in LANES]


def _validate_checkpoint_unit(
    inputs: Inputs,
    seed: int,
    fold: int,
    npz_path: Path,
    json_path: Path,
    entry: dict[str, Any],
    *,
    verify_semantics: bool = True,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    key = _unit_key(seed, fold)
    _assert_plain(npz_path, directory=False, role=f"{key} checkpoint NPZ")
    _assert_plain(json_path, directory=False, role=f"{key} checkpoint JSON")
    if _sha256(npz_path) != entry["npz_sha256"] or _sha256(json_path) != entry["json_sha256"]:
        raise RuntimeError(f"corrupt refinement checkpoint: {key}")
    payload = _read_plain_json(json_path, f"{key} checkpoint JSON")
    if set(payload) != {"unit", "selections"} or payload["unit"] != key:
        raise RuntimeError(f"{key}: checkpoint JSON identity mismatch")
    selections = _validate_selection_rows(payload["selections"], seed, fold)
    valid_index = np.flatnonzero(inputs.folds[seed] == fold)
    inner_index = np.flatnonzero(inputs.folds[seed] != fold)
    classes = len(inputs.classes)
    expected_members = {
        "inner_primary", "inner_robustness_challenger", "inner_specialization_challenger",
        "outer_row_indices", "outer_primary", "outer_robustness_challenger",
        "outer_specialization_challenger",
    }
    arrays: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_members or set(entry["member_hashes"]) != expected_members:
            raise RuntimeError(f"{key}: checkpoint NPZ member inventory mismatch")
        for name in sorted(expected_members):
            array = np.asarray(archive[name])
            if _array_hash(array) != entry["member_hashes"][name]:
                raise RuntimeError(f"{key}: checkpoint NPZ member hash mismatch: {name}")
            arrays[name] = array.copy()
    indices = np.asarray(arrays["outer_row_indices"], dtype=np.int64)
    if arrays["outer_row_indices"].dtype.kind not in "iu" or not np.array_equal(indices, valid_index):
        raise RuntimeError(f"{key}: checkpoint outer row alignment mismatch")
    for lane, selection in zip(LANES, selections):
        inner = _probability(arrays[f"inner_{lane}"], (len(inner_index), classes), f"{key}:inner:{lane}")
        _probability(arrays[f"outer_{lane}"], (len(valid_index), classes), f"{key}:outer:{lane}")
        if selection["refined_inner_array_hash"] != _array_hash(inner):
            raise RuntimeError(f"{key}: selection/refined inner hash mismatch: {lane}")
        if selection["inner_row_indices_hash"] != _canonical_hash(inner_index.tolist()):
            raise RuntimeError(f"{key}: selection inner row hash mismatch: {lane}")
    if verify_semantics:
        expected_arrays, expected_selections = _compute_unit_impl(inputs, seed, fold)
        if selections != expected_selections:
            raise RuntimeError(f"{key}: checkpoint semantic selection mismatch")
        for name in sorted(expected_members):
            if not np.array_equal(arrays[name], expected_arrays[name]):
                raise RuntimeError(
                    f"{key}: checkpoint semantic array mismatch: {name}"
                )
    return arrays, selections


def run_refinement(
    nested_raw: Path,
    train_path: Path,
    fold_assignments_path: Path,
    output_dir: Path,
    *,
    target: str = "SUBCLASS",
    identifier: str = "ID",
    resume: bool = False,
) -> Path:
    """Refine immutable Nested probabilities without fitting models or reading Test."""
    inputs = _load_inputs(nested_raw, train_path, fold_assignments_path, target, identifier)
    output = Path(output_dir).resolve()
    if output == inputs.raw or inputs.raw in output.parents:
        raise ValueError("refinement output must not overwrite or nest inside nested_raw")
    identity_payload = _identity_payload(inputs, target, identifier)
    identity_hash = _canonical_hash(identity_payload)
    if os.path.lexists(output):
        if resume:
            return _validate_completed(output, identity_hash)
        raise FileExistsError(f"refinement output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _assert_plain(output.parent, directory=True, role="refinement output parent")
    stage = output.with_name(f".{output.name}.refine-staging")
    checkpoint_dir = stage / "checkpoints"
    checkpoint_manifest_path = stage / "checkpoint_manifest.json"
    marker_path = stage / STAGE_MARKER
    stage_exists = os.path.lexists(stage)
    if stage_exists and not resume:
        raise FileExistsError(f"refinement checkpoint exists; use --resume: {stage}")
    expected_units = {_unit_key(seed, fold) for seed in SEEDS for fold in OUTER_FOLDS}
    if not stage_exists:
        os.mkdir(stage, 0o700)
        _assert_plain(stage, directory=True, role="refinement staging")
        nonce = secrets.token_hex(32)
        _atomic_json(
            marker_path,
            {"schema_version": SCHEMA_VERSION, "run_identity_hash": identity_hash, "stage_nonce": nonce},
        )
        os.mkdir(checkpoint_dir, 0o700)
        checkpoint_manifest = _checkpoint_manifest(identity_hash, nonce, expected_units)
        _atomic_json(checkpoint_manifest_path, checkpoint_manifest)
    else:
        _assert_plain(stage, directory=True, role="refinement staging")
        if {path.name for path in stage.iterdir()} != {STAGE_MARKER, "checkpoints", "checkpoint_manifest.json"}:
            raise RuntimeError("unexpected refinement staging entry")
        marker = _read_plain_json(marker_path, "refinement staging marker")
        if (
            set(marker) != {"schema_version", "run_identity_hash", "stage_nonce"}
            or marker["schema_version"] != SCHEMA_VERSION
            or marker["run_identity_hash"] != identity_hash
            or not isinstance(marker["stage_nonce"], str)
            or len(marker["stage_nonce"]) != 64
        ):
            raise RuntimeError("refinement staging marker identity mismatch")
        nonce = marker["stage_nonce"]
        _assert_plain(checkpoint_dir, directory=True, role="refinement checkpoint directory")
        checkpoint_manifest = _read_plain_json(
            checkpoint_manifest_path, "refinement checkpoint manifest"
        )
    completed = _validate_checkpoint_manifest(
        checkpoint_manifest, identity_hash, nonce, expected_units
    )
    expected_checkpoint_entries = {
        f"{unit}.{suffix}" for unit in completed for suffix in ("npz", "json")
    }
    actual_checkpoint_entries = {path.name for path in checkpoint_dir.iterdir()}
    if actual_checkpoint_entries != expected_checkpoint_entries:
        raise RuntimeError("unexpected or missing refinement checkpoint entry")
    for path in checkpoint_dir.iterdir():
        _assert_plain(path, directory=False, role=f"checkpoint entry {path.name}")
    semantically_verified: set[str] = set()
    for seed in SEEDS:
        for fold in OUTER_FOLDS:
            key = _unit_key(seed, fold)
            npz_path = checkpoint_dir / f"{key}.npz"
            json_path = checkpoint_dir / f"{key}.json"
            if key in completed:
                _validate_checkpoint_unit(inputs, seed, fold, npz_path, json_path, completed[key])
                semantically_verified.add(key)
                continue
            arrays, selections = _compute_unit(inputs, seed, fold)
            _deterministic_npz(npz_path, arrays)
            _atomic_json(json_path, {"unit": key, "selections": selections})
            completed[key] = {
                "npz_sha256": _sha256(npz_path),
                "json_sha256": _sha256(json_path),
                "member_hashes": {name: _array_hash(array) for name, array in sorted(arrays.items())},
            }
            checkpoint_manifest["completed"] = dict(sorted(completed.items()))
            _atomic_json(checkpoint_manifest_path, checkpoint_manifest)
            _validate_checkpoint_unit(
                inputs,
                seed,
                fold,
                npz_path,
                json_path,
                completed[key],
                verify_semantics=False,
            )
    if set(completed) != expected_units:
        raise RuntimeError("refinement checkpoint coverage is incomplete")

    classes = len(inputs.classes)
    outer = {lane: np.full((3, len(inputs.train), classes), np.nan) for lane in LANES}
    coverage = {lane: np.zeros((3, len(inputs.train)), dtype=np.int8) for lane in LANES}
    inner: dict[str, np.ndarray] = {}
    selections: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(SEEDS):
        for fold in OUTER_FOLDS:
            key = _unit_key(seed, fold)
            arrays, unit_selections = _validate_checkpoint_unit(
                inputs,
                seed,
                fold,
                checkpoint_dir / f"{key}.npz",
                checkpoint_dir / f"{key}.json",
                completed[key],
                verify_semantics=key not in semantically_verified,
            )
            semantically_verified.add(key)
            indices = np.asarray(arrays["outer_row_indices"], dtype=np.int64)
            for lane in LANES:
                values = np.asarray(arrays[f"outer_{lane}"], dtype=np.float64)
                outer[lane][seed_index, indices] = values
                coverage[lane][seed_index, indices] += 1
                inner_key = _inner_key(lane, seed, fold)
                expected_rows = np.flatnonzero(inputs.folds[seed] != fold)
                inner[inner_key] = _probability(
                    arrays[f"inner_{lane}"], (len(expected_rows), classes), inner_key
                )
                evaluations.append(
                    {
                        "strategy": lane,
                        "seed": seed,
                        "outer_fold": fold,
                        "macro_f1": _macro(inputs.labels[indices], values, classes),
                        "selection_uses_outer_validation": False,
                    }
                )
            selections.extend(unit_selections)
    for lane in LANES:
        if not np.all(coverage[lane] == 1):
            raise RuntimeError(f"{lane}: each outer row must have exactly one prediction")
        _probability(outer[lane], (3, len(inputs.train), classes), lane)
    expected_final_keys = {
        (lane, seed, fold) for lane in LANES for seed in SEEDS for fold in OUTER_FOLDS
    }
    selection_keys = {
        (row["strategy"], row["seed"], row["outer_fold"]) for row in selections
    }
    evaluation_keys = {
        (row["strategy"], row["seed"], row["outer_fold"]) for row in evaluations
    }
    if (
        len(selections) != 45
        or selection_keys != expected_final_keys
        or len(inner) != 45
        or set(inner) != {_inner_key(*key) for key in expected_final_keys}
        or len(evaluations) != 45
        or evaluation_keys != expected_final_keys
    ):
        raise RuntimeError("refinement final selection/inner/evaluation coverage mismatch")

    _deterministic_npz(stage / "refined_strategy_probability.npz", outer)
    _deterministic_npz(stage / "refined_inner_probability.npz", inner)
    selections.sort(key=lambda row: (LANES.index(row["strategy"]), SEEDS.index(row["seed"]), row["outer_fold"]))
    selection_payload = {
        "schema_version": SCHEMA_VERSION,
        "selection_uses_outer_validation": False,
        "selection_hash": _canonical_hash([row["selection_hash"] for row in selections]),
        "records": selections,
    }
    _atomic_json(stage / "refined_selection_records.json", selection_payload)
    summaries = []
    for lane in LANES:
        seed_scores = {
            str(seed): _macro(inputs.labels, outer[lane][index], classes)
            for index, seed in enumerate(SEEDS)
        }
        summaries.append(
            {
                "strategy": lane,
                "seed_macro_f1": seed_scores,
                "mean_macro_f1": float(np.mean(list(seed_scores.values()))),
                "std_macro_f1": float(np.std(list(seed_scores.values()), ddof=1)),
            }
        )
    _atomic_json(
        stage / "refined_cv_metrics.json",
        {
            "schema_version": SCHEMA_VERSION,
            "outer_validation_used_for_evaluation_only": True,
            "selection_uses_outer_validation": False,
            "outer_evaluations": evaluations,
            "strategy_summary": summaries,
        },
    )
    artifact_hashes = {
        name: _sha256(stage / name)
        for name in OUTPUTS
        if name != "refined_manifest.json"
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_identity": identity_payload,
        "run_identity_hash": identity_hash,
        "input_hashes": inputs.input_hashes,
        "source_hash": identity_payload["source_hash"],
        "algorithm": _algorithm_contract(inputs.unit_collision_derivations),
        "class_names": list(inputs.classes),
        "seeds": list(SEEDS),
        "outer_folds": list(OUTER_FOLDS),
        "inner_folds": list(INNER_FOLDS),
        "row_order": inputs.train[identifier].astype(str).tolist(),
        "row_order_hash": identity_payload["row_order_hash"],
        "selection_uses_outer_validation": False,
        "outer_validation_used_for_evaluation_only": True,
        "no_test_audit": {
            "test_reads": 0,
            "submission_reads": 0,
            "allowed_inputs": [*UPSTREAM_FILES, "full_train_features_labels", "fold_assignments"],
            "forbidden_basenames": sorted(FORBIDDEN_BASENAMES),
        },
        "checkpoint_resume": {
            "unit": "seed_x_outer_fold",
            "expected_units": 15,
            "completed_units": len(completed),
            "checkpoint_hash": _canonical_hash(
                {"expected_units": sorted(expected_units), "completed": completed}
            ),
        },
        "artifact_hashes": artifact_hashes,
    }
    _atomic_json(stage / "refined_manifest.json", manifest)
    for path in checkpoint_dir.iterdir():
        _assert_plain(path, directory=False, role=f"checkpoint cleanup entry {path.name}")
        path.unlink()
    os.rmdir(checkpoint_dir)
    _assert_plain(checkpoint_manifest_path, directory=False, role="checkpoint cleanup manifest")
    checkpoint_manifest_path.unlink()
    _assert_plain(marker_path, directory=False, role="checkpoint cleanup marker")
    marker_path.unlink()
    os.replace(stage, output)
    return _validate_completed(output, identity_hash)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nested-raw", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True, help="Train CSV; only ID/label columns are loaded")
    parser.add_argument("--fold-assignments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="SUBCLASS")
    parser.add_argument("--identifier", default="ID")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = run_refinement(
        args.nested_raw,
        args.train,
        args.fold_assignments,
        args.output,
        target=args.target,
        identifier=args.identifier,
        resume=args.resume,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
