"""Checkpointed vectorized bootstrap adapter for TEST_007 Evidence.

This module changes only how statistically equivalent class-stratified paired
bootstrap draws are evaluated.  It does not change the Nested probabilities,
selection rules, Test boundary, or finalizer outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import f1_score

from src.test_007 import finalize_test_007_specialization as finalizer


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_ROOT = (
    finalizer.REPO_ROOT.parents[1]
    / ".omx"
    / "checkpoints"
    / "test_007_evidence_bootstrap"
)
GROUP_TRAIN_HASH: str | None = None
ORIGINAL_FOLD_SAFE_GROUPS = finalizer.nested._fold_safe_groups
GROUP_CONTEXT_ACTIVE: np.ndarray | None = None
GROUP_CONTEXT_VALUES: np.ndarray | None = None
GROUP_CONTEXT_COLUMNS: tuple[str, ...] | None = None


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _score(
    labels: np.ndarray,
    prediction: np.ndarray,
    metric_labels: list[int],
) -> float:
    if not len(labels):
        return 0.0
    return float(
        f1_score(
            labels,
            prediction,
            labels=metric_labels,
            average="macro",
            zero_division=0,
        )
    )


def _f1_from_confusion(
    confusion: np.ndarray,
    metric_labels: list[int],
) -> np.ndarray:
    selected = np.asarray(metric_labels, dtype=int)
    true_positive = confusion[:, selected, selected]
    predicted = confusion[:, :, selected].sum(axis=1)
    actual = confusion[:, selected, :].sum(axis=2)
    denominator = predicted + actual
    per_class = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=float),
        where=denominator != 0,
    )
    return per_class.mean(axis=1)


def _bootstrap_from_predictions(
    labels: np.ndarray,
    candidate_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    masks: np.ndarray,
    *,
    classes: int,
    iterations: int,
    seed: int,
    metric_labels: Iterable[int],
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int16)
    candidate_prediction = np.asarray(candidate_prediction, dtype=np.int16)
    baseline_prediction = np.asarray(baseline_prediction, dtype=np.int16)
    masks = np.asarray(masks, dtype=bool)
    if candidate_prediction.shape != baseline_prediction.shape or candidate_prediction.shape != masks.shape:
        raise ValueError("bootstrap prediction/mask shapes differ")
    if candidate_prediction.shape != (len(finalizer.SEEDS), len(labels)):
        raise ValueError("bootstrap requires three aligned seed predictions")
    if iterations < 2:
        raise ValueError("bootstrap requires at least two draws")
    metric_labels = [int(value) for value in metric_labels]
    if not metric_labels:
        raise ValueError("bootstrap metric label set is empty")

    seed_delta = []
    support = []
    for seed_index in range(len(finalizer.SEEDS)):
        active = masks[seed_index]
        support.append(int(active.sum()))
        seed_delta.append(
            _score(labels[active], candidate_prediction[seed_index, active], metric_labels)
            - _score(labels[active], baseline_prediction[seed_index, active], metric_labels)
        )

    # [draw, seed, candidate_or_baseline, true, predicted]
    confusion = np.zeros(
        (iterations, len(finalizer.SEEDS), 2, classes, classes),
        dtype=np.int32,
    )
    rng = np.random.default_rng(seed)
    for true_class in range(classes):
        indices = np.flatnonzero(labels == true_class)
        if not len(indices):
            continue
        signature_columns = []
        for seed_index in range(len(finalizer.SEEDS)):
            signature_columns.extend(
                (
                    masks[seed_index, indices].astype(np.int16),
                    candidate_prediction[seed_index, indices],
                    baseline_prediction[seed_index, indices],
                )
            )
        signatures = np.column_stack(signature_columns)
        categories, frequencies = np.unique(signatures, axis=0, return_counts=True)
        sampled_counts = rng.multinomial(
            len(indices),
            frequencies.astype(float) / frequencies.sum(),
            size=iterations,
        )
        for category_index, category in enumerate(categories):
            draw_counts = sampled_counts[:, category_index]
            for seed_index in range(len(finalizer.SEEDS)):
                offset = seed_index * 3
                if not bool(category[offset]):
                    continue
                candidate_class = int(category[offset + 1])
                baseline_class = int(category[offset + 2])
                confusion[:, seed_index, 0, true_class, candidate_class] += draw_counts
                confusion[:, seed_index, 1, true_class, baseline_class] += draw_counts

    per_seed_draws = []
    for seed_index in range(len(finalizer.SEEDS)):
        candidate_score = _f1_from_confusion(
            confusion[:, seed_index, 0], metric_labels
        )
        baseline_score = _f1_from_confusion(
            confusion[:, seed_index, 1], metric_labels
        )
        per_seed_draws.append(candidate_score - baseline_score)
    draws = np.mean(np.stack(per_seed_draws, axis=1), axis=1)
    seed_vector = np.asarray(seed_delta, dtype=float)
    se_seed = float(np.std(seed_vector, ddof=1) / math.sqrt(len(finalizer.SEEDS)))
    se_boot = float(np.std(draws, ddof=1))
    return {
        "seed_delta": dict(zip(map(str, finalizer.SEEDS), map(float, seed_delta))),
        "support": dict(zip(map(str, finalizer.SEEDS), support)),
        "delta": float(seed_vector.mean()),
        "se_seed": se_seed,
        "se_boot": se_boot,
        "epsilon": max(se_seed, se_boot),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
    }


def _checkpointed(
    kind: str,
    identity: dict[str, Any],
    compute: Any,
) -> dict[str, Any]:
    key = finalizer._canonical_hash(identity)
    path = CHECKPOINT_ROOT / kind / f"{key}.json"
    if path.is_file():
        try:
            envelope = finalizer._read_json(path)
            payload = envelope.get("payload")
            if (
                envelope.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
                and envelope.get("identity") == identity
                and envelope.get("payload_sha256")
                == finalizer._canonical_hash(payload)
                and isinstance(payload, dict)
            ):
                return payload
        except (OSError, ValueError, TypeError):
            pass
    payload = compute()
    finalizer._atomic_json(
        path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "identity": identity,
            "payload": payload,
            "payload_sha256": finalizer._canonical_hash(payload),
        },
    )
    return payload


def _fold_safe_group_identity(train_x: Any, valid_x: Any) -> dict[str, Any]:
    if not GROUP_TRAIN_HASH:
        raise RuntimeError("fold-safe group checkpoint context is not initialized")
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "algorithm": "fold_safe_groups_vectorized_v2",
        "train_file_sha256": GROUP_TRAIN_HASH,
        "train_indices_sha256": _array_hash(np.asarray(train_x.index, dtype=np.int64)),
        "valid_indices_sha256": _array_hash(np.asarray(valid_x.index, dtype=np.int64)),
        "columns_sha256": finalizer._canonical_hash(list(map(str, train_x.columns))),
        "dtypes_sha256": finalizer._canonical_hash(list(map(str, train_x.dtypes))),
    }


def _fold_safe_group_checkpoint_paths(
    train_x: Any, valid_x: Any
) -> tuple[dict[str, Any], Path, Path]:
    identity = _fold_safe_group_identity(train_x, valid_x)
    key = finalizer._canonical_hash(identity)
    directory = CHECKPOINT_ROOT / "fold_safe_groups"
    return identity, directory / f"{key}.npz", directory / f"{key}.json"


def _valid_fold_safe_group_checkpoint(train_x: Any, valid_x: Any) -> bool:
    identity, array_path, receipt_path = _fold_safe_group_checkpoint_paths(
        train_x, valid_x
    )
    if array_path.is_file() and receipt_path.is_file():
        try:
            receipt = finalizer._read_json(receipt_path)
            if (
                receipt.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
                and receipt.get("identity") == identity
                and receipt.get("array_sha256") == finalizer._sha256(array_path)
                and _valid_group_thresholds(receipt.get("thresholds"))
            ):
                with np.load(array_path, allow_pickle=False) as archive:
                    if set(archive.files) == {"burden", "novelty"}:
                        groups = {
                            name: np.asarray(archive[name], dtype=np.int8)
                            for name in ("burden", "novelty")
                        }
                        if all(
                            value.shape == (len(valid_x),)
                            and np.isin(value, (0, 1, 2)).all()
                            for value in groups.values()
                        ):
                            return True
        except (OSError, ValueError, TypeError, KeyError):
            pass
    return False


def _valid_group_thresholds(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"burden", "novelty"}:
        return False
    for thresholds in value.values():
        if not isinstance(thresholds, list) or len(thresholds) != 2:
            return False
        try:
            numeric = np.asarray(thresholds, dtype=float)
        except (TypeError, ValueError):
            return False
        if (
            numeric.shape != (2,)
            or not np.isfinite(numeric).all()
            or numeric[0] > numeric[1]
        ):
            return False
    return True


def fold_safe_group_checkpoints_complete(
    features: Any, assignments: Any
) -> bool:
    """Return true only when all 3-seed x 5-fold group shards validate."""
    normalized = finalizer._normalize_alignment(assignments)
    for seed in finalizer.SEEDS:
        seeded = normalized.loc[normalized.seed == seed].set_index("row_index")
        for fold in finalizer.FOLDS:
            valid = seeded.index[seeded.fold == fold].to_numpy(int)
            train = seeded.index[seeded.fold != fold].to_numpy(int)
            if not _valid_fold_safe_group_checkpoint(
                features.iloc[train], features.iloc[valid]
            ):
                return False
    return True


def checkpointed_fold_safe_groups(
    train_x: Any,
    valid_x: Any,
) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    identity, array_path, receipt_path = _fold_safe_group_checkpoint_paths(
        train_x, valid_x
    )
    if _valid_fold_safe_group_checkpoint(train_x, valid_x):
        receipt = finalizer._read_json(receipt_path)
        with np.load(array_path, allow_pickle=False) as archive:
            groups = {
                name: np.asarray(archive[name], dtype=np.int8)
                for name in ("burden", "novelty")
            }
        return groups, receipt["thresholds"]
    groups, thresholds = vectorized_fold_safe_groups(train_x, valid_x)
    groups = {
        name: np.asarray(groups[name], dtype=np.int8)
        for name in ("burden", "novelty")
    }
    finalizer._atomic_npz(array_path, **groups)
    finalizer._atomic_json(
        receipt_path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "identity": identity,
            "thresholds": thresholds,
            "array_sha256": finalizer._sha256(array_path),
        },
    )
    return groups, thresholds


def vectorized_fold_safe_groups(
    train_x: Any,
    valid_x: Any,
) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    """Exact vectorization of ``nested._fold_safe_groups`` row loops."""
    use_context = (
        GROUP_CONTEXT_ACTIVE is not None
        and GROUP_CONTEXT_VALUES is not None
        and GROUP_CONTEXT_COLUMNS == tuple(map(str, train_x.columns))
        and GROUP_CONTEXT_COLUMNS == tuple(map(str, valid_x.columns))
    )
    if use_context:
        train_indices = np.asarray(train_x.index, dtype=np.int64)
        valid_indices = np.asarray(valid_x.index, dtype=np.int64)
        train_active = GROUP_CONTEXT_ACTIVE[train_indices]
        valid_active = GROUP_CONTEXT_ACTIVE[valid_indices]
        train_values = GROUP_CONTEXT_VALUES[train_indices]
        valid_values = GROUP_CONTEXT_VALUES[valid_indices]
    else:
        train_active = finalizer.nested._mutation_mask(train_x)
        valid_active = finalizer.nested._mutation_mask(valid_x)
        train_values = train_x.fillna("WT").astype(str).to_numpy()
        valid_values = valid_x.fillna("WT").astype(str).to_numpy()
    train_burden = train_active.sum(axis=1).astype(float)
    valid_burden = valid_active.sum(axis=1).astype(float)
    burden_thresholds = np.quantile(train_burden, [0.25, 0.75]).astype(float)
    train_rare = np.zeros(len(train_x), dtype=float)
    valid_unseen = np.zeros(len(valid_x), dtype=float)
    train_denominator = np.maximum(train_active.sum(axis=1), 1)
    valid_denominator = np.maximum(valid_active.sum(axis=1), 1)
    for column in range(train_values.shape[1]):
        train_rows = np.flatnonzero(train_active[:, column])
        if len(train_rows):
            unique, inverse, counts = np.unique(
                train_values[train_rows, column],
                return_inverse=True,
                return_counts=True,
            )
            train_rare[train_rows] += counts[inverse] <= 1
        else:
            unique = np.empty(0, dtype=train_values.dtype)
        valid_rows = np.flatnonzero(valid_active[:, column])
        if len(valid_rows):
            valid_unseen[valid_rows] += ~np.isin(
                valid_values[valid_rows, column], unique
            )
    train_novelty = train_rare / train_denominator
    valid_novelty = valid_unseen / valid_denominator
    novelty_thresholds = np.quantile(train_novelty, [0.25, 0.75]).astype(float)
    return (
        {
            "burden": finalizer.nested._assign_quantile_groups(
                valid_burden, burden_thresholds[0], burden_thresholds[1]
            ),
            "novelty": finalizer.nested._assign_quantile_groups(
                valid_novelty, novelty_thresholds[0], novelty_thresholds[1]
            ),
        },
        {
            "burden": burden_thresholds.tolist(),
            "novelty": novelty_thresholds.tolist(),
        },
    )


def initialize_group_context(features: Any) -> None:
    global GROUP_CONTEXT_ACTIVE, GROUP_CONTEXT_VALUES, GROUP_CONTEXT_COLUMNS
    GROUP_CONTEXT_COLUMNS = tuple(map(str, features.columns))
    GROUP_CONTEXT_ACTIVE = finalizer.nested._mutation_mask(features)
    GROUP_CONTEXT_VALUES = features.fillna("WT").astype(str).to_numpy()


def fast_masked_paired_bootstrap(
    labels: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    masks: np.ndarray,
    iterations: int = finalizer.N_BOOTSTRAP,
    seed: int = 730,
    metric_labels: Iterable[int] | None = None,
) -> dict[str, Any]:
    classes = int(candidate.shape[-1])
    candidate_prediction = np.asarray(candidate).argmax(axis=2).astype(np.int16)
    baseline_prediction = np.asarray(baseline).argmax(axis=2).astype(np.int16)
    masks = np.asarray(masks, dtype=bool)
    metric = list(metric_labels) if metric_labels is not None else list(range(classes))
    identity = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "algorithm": "joint_category_multinomial_v1",
        "kind": "masked",
        "iterations": int(iterations),
        "seed": int(seed),
        "classes": classes,
        "metric_labels": list(map(int, metric)),
        "labels_sha256": _array_hash(np.asarray(labels)),
        "candidate_prediction_sha256": _array_hash(candidate_prediction),
        "baseline_prediction_sha256": _array_hash(baseline_prediction),
        "masks_sha256": _array_hash(masks),
    }
    return _checkpointed(
        "masked",
        identity,
        lambda: _bootstrap_from_predictions(
            labels,
            candidate_prediction,
            baseline_prediction,
            masks,
            classes=classes,
            iterations=iterations,
            seed=seed,
            metric_labels=metric,
        ),
    )


def fast_paired_bootstrap(
    labels: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    iterations: int = finalizer.N_BOOTSTRAP,
    seed: int = 7007,
) -> dict[str, float]:
    classes = int(candidate.shape[-1])
    if any(np.count_nonzero(np.asarray(labels) == value) == 0 for value in range(classes)):
        raise ValueError("paired bootstrap requires every canonical class")
    result = fast_masked_paired_bootstrap(
        labels,
        candidate,
        baseline,
        np.ones((len(finalizer.SEEDS), len(labels)), dtype=bool),
        iterations=iterations,
        seed=seed,
        metric_labels=range(classes),
    )
    return {
        key: result[key]
        for key in (
            "delta",
            "se_seed",
            "se_boot",
            "epsilon",
            "ci_lower",
            "ci_upper",
        )
    }


def run(args: argparse.Namespace) -> Path:
    global GROUP_TRAIN_HASH
    goal = finalizer.load_config(Path(args.config).resolve())
    data = goal["data"]
    raw = (finalizer.REPO_ROOT / data["raw_dir"]).resolve()
    train_path = raw / data["train_file"]
    GROUP_TRAIN_HASH = finalizer._sha256(train_path)
    train = finalizer.pd.read_csv(train_path)
    features = train.drop(columns=[data["target_column"], data["id_column"]])
    assignments = finalizer.pd.read_csv(
        Path(args.artifact_root).resolve() / "fold_assignments.csv"
    )
    if not fold_safe_group_checkpoints_complete(features, assignments):
        initialize_group_context(features)
    finalizer.nested._fold_safe_groups = checkpointed_fold_safe_groups
    finalizer._masked_paired_bootstrap = fast_masked_paired_bootstrap
    finalizer.paired_bootstrap = fast_paired_bootstrap
    return finalizer.build_evidence(
        Path(args.artifact_root),
        Path(args.nested_source),
        Path(args.config),
        Path(args.search_universe),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--nested-source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--search-universe", required=True)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
