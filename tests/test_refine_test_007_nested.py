import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.test_007 import refine_test_007_nested as refine


CLASSES = ("C0", "C1", "C2")


def _probabilities(labels: np.ndarray, *, robust: bool = False) -> np.ndarray:
    values = np.full((len(labels), len(CLASSES)), 0.05, dtype=np.float64)
    for index, label in enumerate(labels):
        if robust:
            values[index] = 0.05
            values[index, label] = 0.90
        elif label == 0:
            values[index] = [0.48, 0.52, 0.0]
        elif label == 2:
            values[index] = [0.0, 0.52, 0.48]
        else:
            values[index] = [0.05, 0.90, 0.05]
    return values


def _write_fixture(
    root: Path,
    labels: np.ndarray | None = None,
    probability_labels: np.ndarray | None = None,
):
    raw = root / "nested_raw"
    raw.mkdir(parents=True)
    labels = np.asarray(labels if labels is not None else np.arange(60) % 3, dtype=np.int32)
    probability_labels = np.asarray(
        labels if probability_labels is None else probability_labels, dtype=np.int32
    )
    train = pd.DataFrame(
        {
            "ID": [f"R{i:03d}" for i in range(len(labels))],
            "SUBCLASS": np.asarray(CLASSES)[labels],
            "G0": np.where(np.arange(len(labels)) % 4 == 0, "A", "WT"),
            "G1": np.where(np.arange(len(labels)) % 5 == 0, "B", "WT"),
            "G2": np.where(np.arange(len(labels)) % 7 == 0, "C", "WT"),
        }
    )
    train_path = root / "train_labels.csv"
    train.to_csv(train_path, index=False)
    assignments = pd.concat(
        [
            pd.DataFrame(
                {
                    "row_index": np.arange(len(train)),
                    "ID": train.ID,
                    "seed": seed,
                    "fold": np.arange(len(train)) % 5,
                }
            )
            for seed in refine.SEEDS
        ],
        ignore_index=True,
    )
    fold_path = root / "fold_assignments.csv"
    assignments.to_csv(fold_path, index=False)

    inner_arrays = {}
    records = []
    outer_arrays = {
        lane: np.empty((3, len(train), len(CLASSES)), dtype=np.float64)
        for lane in refine.LANES
    }
    for seed_index, seed in enumerate(refine.SEEDS):
        outer_arrays["primary"][seed_index] = _probabilities(probability_labels)
        outer_arrays["robustness_challenger"][seed_index] = _probabilities(
            probability_labels, robust=True
        )
        outer_arrays["specialization_challenger"][seed_index] = _probabilities(
            probability_labels
        )
        folds = np.arange(len(train)) % 5
        for fold in refine.OUTER_FOLDS:
            row_index = np.flatnonzero(folds != fold)
            fold_ids = refine._expected_inner_fold_ids(labels[row_index], seed, fold)
            for lane in refine.LANES:
                probability = _probabilities(
                    probability_labels[row_index], robust=lane == "robustness_challenger"
                )
                key = refine._inner_key(lane, seed, fold)
                inner_arrays[key] = probability
                records.append(
                    {
                        "array_key": key,
                        "strategy": lane,
                        "seed": seed,
                        "outer_fold": fold,
                        "row_indices": row_index.tolist(),
                        "row_ids": train.iloc[row_index].ID.tolist(),
                        "inner_oof_coverage": np.ones(len(row_index), dtype=int).tolist(),
                        "inner_fold_ids": fold_ids.tolist(),
                        "array_hash": refine._array_hash(probability),
                        "outer_validation_used": False,
                    }
                )
    refine._deterministic_npz(raw / "nested_inner_selection_probability.npz", inner_arrays)
    refine._deterministic_npz(raw / "nested_strategy_probability.npz", outer_arrays)
    refine._atomic_json(
        raw / "nested_inner_selection_records.json",
        {
            "schema_version": 2,
            "selection_uses_outer_validation": False,
            "records": records,
        },
    )
    refine._atomic_json(
        raw / "nested_selection_records.json",
        {
            "selections": [
                {
                    "strategy": lane,
                    "seed": seed,
                    "fold": fold,
                    "selected_variant": (
                        {"ordered_pair": [2, 1]}
                        if lane == "specialization_challenger"
                        else {}
                    ),
                    "gate_evidence": (
                        {"net_rescue": 5}
                        if lane == "specialization_challenger"
                        else None
                    ),
                    "selection_source": "outer_train_inner_oof_only",
                    "outer_validation_used_for_selection": False,
                }
                for lane in refine.LANES
                for seed in refine.SEEDS
                for fold in refine.OUTER_FOLDS
            ]
        },
    )
    manifest = {
        "schema_version": 2,
        "class_names": list(CLASSES),
        "seeds": list(refine.SEEDS),
        "outer_folds": 5,
        "inner_folds": 4,
        "outer_validation_used_for_selection": False,
        "test_accessed": False,
        "train_hash": refine._sha256(train_path),
        "fold_hash": refine._sha256(fold_path),
        "source_bundle_before": "upstream-source",
        "run_identity_hash": "upstream-run",
        "artifact_hashes": {
            name: refine._sha256(raw / name)
            for name in (
                "nested_inner_selection_probability.npz",
                "nested_inner_selection_records.json",
                "nested_selection_records.json",
                "nested_strategy_probability.npz",
            )
        },
    }
    refine._atomic_json(raw / "nested_run_manifest.json", manifest)
    return raw, train_path, fold_path


def _selection(output: Path, seed: int = 42, fold: int = 0):
    payload = json.loads((output / "refined_selection_records.json").read_text(encoding="utf-8"))
    return {
        row["strategy"]: row
        for row in payload["records"]
        if row["seed"] == seed and row["outer_fold"] == fold
    }


def test_pair_transform_moves_target_mass_only_for_active_top2_pair():
    probability = np.asarray(
        [[0.48, 0.52, 0.0], [0.30, 0.60, 0.10], [0.48, 0.0, 0.52]],
        dtype=np.float64,
    )
    transformed = refine._apply_pair_bias(probability, (0, 1), margin=0.05, alpha=0.10)

    np.testing.assert_allclose(transformed[0], [0.532, 0.468, 0.0])
    np.testing.assert_allclose(transformed[1:], probability[1:])
    np.testing.assert_allclose(transformed.sum(axis=1), 1.0)


def test_identity_fallbacks_are_explicit():
    labels = np.asarray([0, 1] * 8)
    folds = np.arange(len(labels)) % 4
    perfect = np.where(labels[:, None] == np.arange(2), 0.9, 0.1)
    primary_rule, primary = refine._select_pair_rule(
        perfect, labels, folds, 2, ranking="primary"
    )
    near_boundary = np.where(labels[:, None] == np.arange(2), 0.5001, 0.4999)
    opposite = np.where(labels[:, None] == np.arange(2), 0.0, 1.0)
    robust_rule, robust = refine._select_robustness_weight(
        near_boundary, opposite, labels, folds, 2
    )

    assert primary_rule == refine._identity_rule("no_qualified_primary_rule")
    np.testing.assert_array_equal(primary, perfect)
    assert robust_rule["qualified"] is False and robust_rule["weight"] == 0.0
    np.testing.assert_array_equal(robust, near_boundary)


def _pair_safety_case(*, harm_target: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = []
    rows = []
    folds = []
    for fold in refine.INNER_FOLDS:
        labels.extend([0, 0])
        rows.extend([[0.48, 0.52], [0.48, 0.52]])
        folds.extend([fold, fold])
        labels.append(1)
        rows.append([0.48, 0.52] if harm_target else [0.10, 0.90])
        folds.append(fold)
    return np.asarray(rows), np.asarray(labels), np.asarray(folds)


def test_pair_rule_rejects_aggregate_target_class_harm():
    probability, labels, folds = _pair_safety_case(harm_target=True)

    rule, transformed = refine._select_pair_rule(
        probability, labels, folds, 2, ranking="primary"
    )

    assert rule == refine._identity_rule("no_qualified_primary_rule")
    np.testing.assert_array_equal(transformed, probability)


def test_pair_rule_accepts_safe_candidate_and_materializes_only_winner(monkeypatch):
    probability, labels, folds = _pair_safety_case(harm_target=False)
    original = refine._apply_pair_bias
    calls = []

    def recording(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(refine, "_apply_pair_bias", recording)

    rule, transformed = refine._select_pair_rule(
        probability, labels, folds, 2, ranking="primary"
    )

    assert rule["qualified"] is True
    assert rule["ordered_pair"] == [0, 1]
    assert min(rule["safety_evidence"]["per_class_f1_delta"]) >= 0.0
    assert rule["safety_evidence"]["overall_declining_folds"] <= 1
    assert len(calls) == 1
    np.testing.assert_array_equal(transformed.argmax(axis=1), labels)


def test_robustness_rejects_repeated_per_class_fold_degradation(monkeypatch):
    primary, labels, folds = _pair_safety_case(harm_target=True)
    candidate_prediction = np.where(labels == 0, 0, 0)
    robustness = np.where(
        candidate_prediction[:, None] == np.arange(2), 0.9, 0.1
    )
    monkeypatch.setattr(refine, "WEIGHT_GRID", (0.0, 1.0))

    rule, transformed = refine._select_robustness_weight(
        primary, robustness, labels, folds, 2
    )

    assert rule["qualified"] is False
    assert rule["weight"] == 0.0
    assert max(
        rule["rejected_positive_weight_safety_evidence"][0]["safety_evidence"][
            "per_class_declining_fold_counts"
        ]
    ) >= 2
    np.testing.assert_array_equal(transformed, primary)


def test_robustness_one_se_uses_the_exact_best_mean_candidate(monkeypatch):
    primary = np.asarray([[0.8, 0.2], [0.2, 0.8]] * 4, dtype=np.float64)
    robustness = primary.copy()
    labels = np.asarray([0, 1] * 4)
    folds = np.arange(len(labels)) % 4
    score_rows = iter(
        (
            [0.5, 0.5, 0.5, 0.5],
            [0.3, 0.4, 0.6, 0.70000002],
            [0.48, 0.48, 0.48, 0.48],
        )
    )
    safe = {
        "overall_delta": 0.0,
        "overall_fold_delta": [0.0] * 4,
        "overall_declining_folds": 0,
        "per_class_f1_delta": [0.0, 0.0],
        "per_class_fold_delta": [[0.0] * 4, [0.0] * 4],
        "per_class_declining_fold_counts": [0, 0],
    }
    monkeypatch.setattr(refine, "WEIGHT_GRID", (0.0, 0.5, 1.0))
    monkeypatch.setattr(refine, "_fold_scores", lambda *args: next(score_rows))
    monkeypatch.setattr(refine, "_safety_evidence", lambda *args: safe)

    rule, _ = refine._select_robustness_weight(
        primary, robustness, labels, folds, 2
    )

    assert rule["qualified"] is True
    assert rule["weight"] == 1.0
    assert rule["best_inner_mean"] == pytest.approx(0.500000005)
    assert rule["best_dynamic_one_se"] > 0.05


def test_fold_safe_groups_cover_valid_once_and_fit_thresholds_on_train_only():
    train = pd.DataFrame(
        {"A": ["WT", "X", "WT", "Y"], "B": ["WT", "WT", "Z", "Z"]}
    )
    valid = pd.DataFrame({"A": ["NEW", "WT"], "B": ["NEW", "WT"]})
    groups, thresholds = refine._fold_safe_groups(train, valid)
    changed_valid = valid.copy()
    changed_valid.loc[:, :] = "OTHER"
    _, changed_thresholds = refine._fold_safe_groups(train, changed_valid)

    assert set(groups) == {"burden", "novelty"}
    assert all(values.shape == (2,) for values in groups.values())
    assert thresholds == changed_thresholds
    assert thresholds["burden"] == pytest.approx([0.75, 1.25])

    features = pd.concat([train, valid], ignore_index=True)
    fold_ids = np.asarray([0, 1, 2, 3, 0, 1])
    inner_groups, inner_thresholds = refine._inner_fold_safe_groups(features, fold_ids)
    assert all(set(values.tolist()).issubset({0, 1, 2}) for values in inner_groups.values())
    assert set(inner_thresholds) == {"0", "1", "2", "3"}


def test_sparse_singleton_group_is_recorded_as_insufficient_evidence():
    labels = np.asarray([0, 1] * 8)
    folds = np.arange(len(labels)) % 4
    baseline = labels.copy()
    candidate = baseline.copy()
    candidate[-1] = 1 - candidate[-1]
    groups = {
        "burden": np.asarray([0] * 15 + [2]),
        "novelty": np.arange(len(labels)) % 2,
    }

    evidence = refine._group_safety_evidence(
        labels, baseline, candidate, groups, folds, 2
    )

    assert evidence["aggregate_group_support"]["burden:2"] == 1
    assert evidence["inner_fold_group_support"]["burden:2"] == [0, 0, 0, 1]
    assert evidence["observed_group_fold_counts"]["burden:2"] == 1
    assert "burden:2" in evidence["insufficient_evidence_groups"]
    assert evidence["fold_group_delta_se"]["burden:2"] is None
    assert evidence["group_noninferiority_margin"]["burden:2"] is None
    assert evidence["group_declining_fold_counts"]["burden:2"] == 0
    assert evidence["group_evidence_sufficient"] is True


def test_zero_support_groups_are_explicitly_recorded_and_never_gate_or_collapse():
    labels = np.asarray([0, 1] * 8)
    folds = np.arange(len(labels)) % 4
    baseline = labels.copy()
    candidate = baseline.copy()
    groups = {
        "burden": np.arange(len(labels)) % 2,
        "novelty": np.arange(len(labels)) % 2,
    }

    evidence = refine._group_safety_evidence(
        labels, baseline, candidate, groups, folds, 2
    )

    expected_keys = {
        f"{dimension}:{group}"
        for dimension in ("burden", "novelty")
        for group in (0, 1, 2)
    }
    assert set(evidence["aggregate_group_support"]) == expected_keys
    for key in ("burden:2", "novelty:2"):
        assert evidence["aggregate_group_support"][key] == 0
        assert evidence["inner_fold_group_support"][key] == [0, 0, 0, 0]
        assert evidence["observed_group_fold_counts"][key] == 0
        assert evidence["base_group_scores"][key] is None
        assert evidence["group_scores"][key] is None
        assert evidence["group_score_delta"][key] is None
        assert evidence["fold_group_delta"][key] == [None, None, None, None]
        assert evidence["fold_group_delta_se"][key] is None
        assert evidence["group_noninferiority_margin"][key] is None
        assert evidence["group_declining_fold_counts"][key] == 0
        assert key not in evidence["sufficiently_supported_groups"]
        assert key in evidence["insufficient_evidence_groups"]
        assert evidence["insufficient_evidence_reasons"][key] == [
            "aggregate_support_below_minimum",
            "observed_fold_count_below_minimum",
        ]
    assert evidence["repeated_group_decline"] is False
    assert evidence["all_sufficient_groups_noninferior"] is True


def test_robustness_requires_strict_group_floor_improvement(monkeypatch):
    primary, labels, folds = _pair_safety_case(harm_target=False)
    groups = {"burden": np.arange(len(labels)) % 3, "novelty": np.arange(len(labels)) % 3}
    monkeypatch.setattr(refine, "WEIGHT_GRID", (0.0, 1.0))

    rule, transformed = refine._select_robustness_weight(
        primary, primary.copy(), labels, folds, 2, groups
    )

    assert rule["qualified"] is False
    evidence = rule["rejected_positive_weight_safety_evidence"][0][
        "group_safety_evidence"
    ]
    assert evidence["worst_group_floor_delta"] == pytest.approx(0.0)
    np.testing.assert_array_equal(transformed, primary)


def test_robustness_rejects_repeated_group_collapse(monkeypatch):
    primary, labels, folds = _pair_safety_case(harm_target=False)
    candidate_prediction = np.where(np.arange(len(labels)) % 3 == 0, 1 - labels, labels)
    robustness = np.where(candidate_prediction[:, None] == np.arange(2), 0.9, 0.1)
    groups = {"burden": np.arange(len(labels)) % 3, "novelty": np.arange(len(labels)) % 3}
    safe = {
        "per_class_declining_fold_counts": [0, 0],
        "overall_declining_folds": 0,
    }
    monkeypatch.setattr(refine, "WEIGHT_GRID", (0.0, 1.0))
    monkeypatch.setattr(refine, "_fold_scores", lambda *args: [0.5] * 4)
    monkeypatch.setattr(refine, "_safety_evidence", lambda *args: safe)
    original_group = refine._group_safety_evidence

    def group_evidence(*args, **kwargs):
        evidence = original_group(*args, **kwargs)
        if not np.array_equal(np.asarray(args[1]), np.asarray(args[2])):
            evidence["repeated_group_decline"] = True
            evidence["group_declining_fold_counts"]["burden:0"] = 2
        return evidence

    monkeypatch.setattr(refine, "_group_safety_evidence", group_evidence)

    rule, _ = refine._select_robustness_weight(
        primary, robustness, labels, folds, 2, groups
    )

    rejected = rule["rejected_positive_weight_safety_evidence"][0]
    assert rejected["group_safety_evidence"]["repeated_group_decline"] is True


def test_robustness_rejects_supported_group_decline_despite_floor_improvement(
    monkeypatch,
):
    labels = np.tile([0, 1], 12)
    folds = np.tile(np.arange(4), 6)
    groups = {
        "burden": np.repeat([0, 1], 12),
        "novelty": np.repeat([0, 1], 12),
    }
    primary_prediction = labels.copy()
    primary_prediction[12:] = 0
    robust_prediction = labels.copy()
    robust_prediction[[0, 3, 4, 7]] = 1 - robust_prediction[[0, 3, 4, 7]]
    primary = np.where(
        primary_prediction[:, None] == np.arange(2), 0.9, 0.1
    ).astype(np.float64)
    robustness = np.where(
        robust_prediction[:, None] == np.arange(2), 0.9, 0.1
    ).astype(np.float64)
    safe = {
        "per_class_declining_fold_counts": [0, 0],
        "overall_declining_folds": 0,
    }
    monkeypatch.setattr(refine, "WEIGHT_GRID", (0.0, 1.0))
    monkeypatch.setattr(refine, "_safety_evidence", lambda *args: safe)

    rule, transformed = refine._select_robustness_weight(
        primary, robustness, labels, folds, 2, groups
    )

    rejected = rule["rejected_positive_weight_safety_evidence"][0][
        "group_safety_evidence"
    ]
    assert rejected["worst_group_floor_delta"] > 0
    assert rejected["group_noninferiority_margin"]["burden:0"] < 0
    assert rejected["all_sufficient_groups_noninferior"] is False
    assert rule["qualified"] is False
    np.testing.assert_array_equal(transformed, primary)


def test_robustness_reuses_group_statistics_for_identical_predictions(monkeypatch):
    primary, labels, folds = _pair_safety_case(harm_target=False)
    groups = {
        "burden": np.arange(len(labels)) % 3,
        "novelty": np.arange(len(labels)) % 3,
    }
    calls = 0
    original = refine._group_safety_evidence

    def recording(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(refine, "WEIGHT_GRID", (0.0, 0.5, 1.0))
    monkeypatch.setattr(refine, "_group_safety_evidence", recording)

    refine._select_robustness_weight(
        primary, primary.copy(), labels, folds, 2, groups
    )

    assert calls == 1


def test_consensus_pair_derivation_is_deterministic_with_tie_breaks():
    rows = []
    for pair in ([1, 2], [2, 1], [0, 2], [2, 0]):
        rows.append(
            {
                "strategy": "specialization_challenger",
                "selected_variant": {"ordered_pair": pair},
                "gate_evidence": {"net_rescue": 3},
            }
        )
    result = refine._derive_consensus_collision_pair(rows, CLASSES)

    assert result["unordered_pair"] == [0, 2]
    assert result["primary_ordered_pair"] == [0, 2]
    assert result["reverse_ordered_pair"] == [2, 0]
    assert result["derivation_hash"] == refine._canonical_hash(
        {key: value for key, value in result.items() if key != "derivation_hash"}
    )


def test_collision_pair_derivation_is_independent_per_seed_outer_fold():
    rows = [
        {
            "strategy": "specialization_challenger",
            "seed": seed,
            "fold": fold,
            "selected_variant": {"ordered_pair": [2, 1] if fold == 0 else [0, 2]},
            "gate_evidence": {"net_rescue": 5 + fold},
        }
        for seed in refine.SEEDS
        for fold in refine.OUTER_FOLDS
    ]

    derivations = refine._derive_unit_collision_pairs(rows, CLASSES)

    assert len(derivations["units"]) == 15
    assert derivations["units"]["seed_42__fold_0"]["primary_ordered_pair"] == [2, 1]
    assert derivations["units"]["seed_42__fold_1"]["primary_ordered_pair"] == [0, 2]
    assert derivations["derivation_hash"] == refine._canonical_hash(
        {key: value for key, value in derivations.items() if key != "derivation_hash"}
    )


def test_bidirectional_specialist_accepts_safe_rescue_and_materializes_once(monkeypatch):
    probability, labels, folds = _pair_safety_case(harm_target=False)
    groups = {"burden": np.arange(len(labels)) % 3, "novelty": np.arange(len(labels)) % 3}
    original = refine._apply_bidirectional_pair_bias
    calls = []

    def recording(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(refine, "_apply_bidirectional_pair_bias", recording)
    rule, transformed = refine._select_bidirectional_specialist_rule(
        probability, labels, folds, 2, groups, (0, 1), "a" * 64
    )

    assert rule["kind"] == "bidirectional_pair_bias"
    assert rule["ordered_pair"] == [0, 1]
    assert rule["reverse_ordered_pair"] == [1, 0]
    assert rule["source_recall_delta"] >= 0.05
    assert len(calls) == 1
    np.testing.assert_array_equal(transformed.argmax(axis=1), labels)


def test_bidirectional_specialist_rejects_group_and_pair_class_collapse(monkeypatch):
    probability, labels, folds = _pair_safety_case(harm_target=False)
    groups = {"burden": np.arange(len(labels)) % 3, "novelty": np.arange(len(labels)) % 3}
    original_group = refine._group_safety_evidence

    def collapsed(*args, **kwargs):
        evidence = original_group(*args, **kwargs)
        evidence["repeated_group_decline"] = True
        return evidence

    monkeypatch.setattr(refine, "_group_safety_evidence", collapsed)
    group_rule, _ = refine._select_bidirectional_specialist_rule(
        probability, labels, folds, 2, groups, (0, 1), "a" * 64
    )
    assert group_rule["kind"] == "identity"

    monkeypatch.setattr(refine, "_group_safety_evidence", original_group)
    harmful, harmful_labels, harmful_folds = _pair_safety_case(harm_target=True)
    pair_rule, _ = refine._select_bidirectional_specialist_rule(
        harmful, harmful_labels, harmful_folds, 2, groups, (0, 1), "a" * 64
    )
    assert pair_rule["kind"] == "identity"


def test_optimized_pair_predictions_match_materialized_transform_with_ties():
    probability = np.asarray(
        [
            [0.50, 0.50, 0.00],
            [0.48, 0.52, 0.00],
            [0.52, 0.48, 0.00],
            [0.34, 0.34, 0.32],
            [0.10, 0.45, 0.45],
            [0.20, 0.40, 0.40],
        ],
        dtype=np.float64,
    )
    order = np.argsort(-probability, axis=1, kind="stable")[:, :2]
    top1, top2 = order[:, 0], order[:, 1]
    gap = probability[np.arange(len(probability)), top1] - probability[
        np.arange(len(probability)), top2
    ]
    for pair in ((0, 1), (1, 0), (1, 2), (2, 1)):
        for margin in refine.GRID:
            for alpha in refine.GRID:
                optimized = refine._pair_candidate_prediction(
                    probability,
                    top1,
                    top1,
                    top2,
                    gap,
                    pair,
                    margin,
                    alpha,
                )
                expected = refine._apply_pair_bias(
                    probability, pair, margin, alpha
                ).argmax(axis=1)
                np.testing.assert_array_equal(optimized, expected)


def test_full_refinement_outputs_are_deterministic_and_valid(tmp_path):
    raw, train, folds = _write_fixture(tmp_path / "inputs")
    first = refine.run_refinement(raw, train, folds, tmp_path / "first")
    second = refine.run_refinement(raw, train, folds, tmp_path / "second")

    for name in refine.OUTPUTS:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    with np.load(first / "refined_strategy_probability.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(refine.LANES)
        for lane in refine.LANES:
            assert archive[lane].shape == (3, 60, 3)
            assert np.isfinite(archive[lane]).all()
            np.testing.assert_allclose(archive[lane].sum(axis=2), 1.0)
    with np.load(first / "refined_inner_probability.npz", allow_pickle=False) as archive:
        assert len(archive.files) == 45
        assert all(np.isfinite(archive[key]).all() for key in archive.files)
    selected = _selection(first)
    assert selected["primary"]["selection"]["ordered_pair"] == [0, 1]
    assert selected["specialization_challenger"]["selection"]["ordered_pair"] == [2, 1]
    primary = selected["primary"]
    robust = selected["robustness_challenger"]
    specialist = selected["specialization_challenger"]
    assert set(primary["inner_dependency_hashes"]) == {"upstream_primary"}
    assert primary["parent_selection_hashes"] == {}
    assert set(robust["inner_dependency_hashes"]) == {
        "refined_primary",
        "upstream_primary",
        "upstream_robustness_challenger",
    }
    assert set(specialist["inner_dependency_hashes"]) == {
        "refined_primary",
        "upstream_primary",
    }
    assert robust["inner_dependency_hashes"]["refined_primary"] == primary[
        "refined_inner_array_hash"
    ]
    assert specialist["inner_dependency_hashes"]["refined_primary"] == primary[
        "refined_inner_array_hash"
    ]
    assert robust["parent_selection_hashes"] == {
        "primary": primary["selection_hash"]
    }
    assert specialist["parent_selection_hashes"] == {
        "primary": primary["selection_hash"]
    }
    manifest = json.loads((first / "refined_manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithm"]["name"] == "test_007_nested_refinement_v4"
    assert manifest["algorithm"]["subgroup_safety"]["status"] == "fold_safe"
    assert "nested_selection_records.json" in manifest["input_hashes"]
    assert specialist["selection"]["kind"] == "bidirectional_pair_bias"
    assert specialist["selection"]["reverse_ordered_pair"] == [1, 2]
    unit_derivation = manifest["algorithm"]["unit_collision_derivations"]["units"][
        "seed_42__fold_0"
    ]
    assert specialist["selection"]["selection_transcript_dependency_hash"] == unit_derivation[
        "transcript_row_hash"
    ]
    assert specialist["selection"]["unit_collision_derivation_hash"] == unit_derivation[
        "derivation_hash"
    ]
    assert len(manifest["algorithm"]["unit_collision_derivations"]["units"]) == 15
    assert manifest["selection_uses_outer_validation"] is False
    assert manifest["no_test_audit"]["test_reads"] == 0
    assert manifest["checkpoint_resume"]["completed_units"] == 15


def test_outer_label_change_does_not_change_that_fold_selection(tmp_path):
    labels = np.arange(60) % 3
    raw_a, train_a, folds_a = _write_fixture(tmp_path / "a", labels)
    out_a = refine.run_refinement(raw_a, train_a, folds_a, tmp_path / "out_a")
    changed = labels.copy()
    changed[np.arange(60) % 5 == 0] = (changed[np.arange(60) % 5 == 0] + 1) % 3
    raw_b, train_b, folds_b = _write_fixture(
        tmp_path / "b", changed, probability_labels=labels
    )
    out_b = refine.run_refinement(raw_b, train_b, folds_b, tmp_path / "out_b")

    select_a = _selection(out_a)
    select_b = _selection(out_b)
    assert {lane: row["selection_hash"] for lane, row in select_a.items()} == {
        lane: row["selection_hash"] for lane, row in select_b.items()
    }
    metrics_a = json.loads((out_a / "refined_cv_metrics.json").read_text(encoding="utf-8"))
    metrics_b = json.loads((out_b / "refined_cv_metrics.json").read_text(encoding="utf-8"))
    fold_a = [row for row in metrics_a["outer_evaluations"] if row["seed"] == 42 and row["outer_fold"] == 0]
    fold_b = [row for row in metrics_b["outer_evaluations"] if row["seed"] == 42 and row["outer_fold"] == 0]
    assert fold_a != fold_b


def test_regenerated_transcript_outer_label_isolation_for_specialist(tmp_path):
    labels = np.arange(60) % 3
    raw_a, train_a, folds_a = _write_fixture(tmp_path / "a", labels)
    out_a = refine.run_refinement(raw_a, train_a, folds_a, tmp_path / "out_a")

    changed = labels.copy()
    outer_zero = np.arange(60) % 5 == 0
    changed[outer_zero] = (changed[outer_zero] + 1) % 3
    raw_b, train_b, folds_b = _write_fixture(
        tmp_path / "b", changed, probability_labels=labels
    )
    transcript_path = raw_b / "nested_selection_records.json"
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    for row in transcript["selections"]:
        if row["strategy"] == "specialization_challenger" and row["fold"] != 0:
            row["selected_variant"]["ordered_pair"] = [0, 2]
            row["gate_evidence"]["net_rescue"] = 100
    refine._atomic_json(transcript_path, transcript)
    manifest_path = raw_b / "nested_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"][transcript_path.name] = refine._sha256(transcript_path)
    refine._atomic_json(manifest_path, manifest)

    out_b = refine.run_refinement(raw_b, train_b, folds_b, tmp_path / "out_b")
    specialist_a = _selection(out_a, seed=42, fold=0)["specialization_challenger"]
    specialist_b = _selection(out_b, seed=42, fold=0)["specialization_challenger"]

    assert specialist_a["selection_hash"] == specialist_b["selection_hash"]
    assert specialist_a["selection"]["ordered_pair"] == [2, 1]
    assert specialist_b["selection"]["ordered_pair"] == [2, 1]
    manifest_b = json.loads((out_b / "refined_manifest.json").read_text(encoding="utf-8"))
    manifest_a = json.loads((out_a / "refined_manifest.json").read_text(encoding="utf-8"))
    assert manifest_a["run_identity_hash"] != manifest_b["run_identity_hash"]
    assert manifest_b["algorithm"]["unit_collision_derivations"]["units"][
        "seed_42__fold_1"
    ]["primary_ordered_pair"] == [0, 2]


@pytest.mark.filterwarnings("ignore:The least populated class.*")
def test_inner_label_change_can_change_selection(tmp_path):
    labels = np.arange(60) % 3
    raw_a, train_a, folds_a = _write_fixture(tmp_path / "a", labels)
    out_a = refine.run_refinement(raw_a, train_a, folds_a, tmp_path / "out_a")
    changed = labels.copy()
    inner_rows = np.arange(60) % 5 != 0
    changed[np.flatnonzero(inner_rows & (changed == 0))[:14]] = 1
    raw_b, train_b, folds_b = _write_fixture(
        tmp_path / "b", changed, probability_labels=labels
    )
    out_b = refine.run_refinement(raw_b, train_b, folds_b, tmp_path / "out_b")

    assert _selection(out_a)["primary"]["selection_hash"] != _selection(out_b)["primary"]["selection_hash"]


def test_resume_no_overwrite_and_integrity(tmp_path):
    raw, train, folds = _write_fixture(tmp_path / "inputs")
    output = refine.run_refinement(raw, train, folds, tmp_path / "output")
    hashes = {name: refine._sha256(output / name) for name in refine.OUTPUTS}

    with pytest.raises(FileExistsError):
        refine.run_refinement(raw, train, folds, output)
    assert refine.run_refinement(raw, train, folds, output, resume=True) == output
    assert hashes == {name: refine._sha256(output / name) for name in refine.OUTPUTS}


def test_interrupted_checkpoint_resumes_without_recomputing_completed_unit(
    tmp_path, monkeypatch
):
    raw, train, folds = _write_fixture(tmp_path / "inputs")
    output = tmp_path / "resumed"
    original = refine._compute_unit
    calls = []

    def interrupted(inputs, seed, fold):
        calls.append((seed, fold))
        if len(calls) == 2:
            raise RuntimeError("forced interruption")
        return original(inputs, seed, fold)

    monkeypatch.setattr(refine, "_compute_unit", interrupted)
    with pytest.raises(RuntimeError, match="forced interruption"):
        refine.run_refinement(raw, train, folds, output)
    monkeypatch.setattr(refine, "_compute_unit", original)
    resumed_calls = []

    def recording(inputs, seed, fold):
        resumed_calls.append((seed, fold))
        return original(inputs, seed, fold)

    monkeypatch.setattr(refine, "_compute_unit", recording)
    resumed = refine.run_refinement(raw, train, folds, output, resume=True)
    resumed_only_calls = resumed_calls.copy()
    clean = refine.run_refinement(raw, train, folds, tmp_path / "clean")

    assert calls[0] not in resumed_only_calls
    assert len(resumed_only_calls) == 14
    for name in refine.OUTPUTS:
        assert (resumed / name).read_bytes() == (clean / name).read_bytes()


def test_resume_rejects_staging_symlink_without_touching_external_sentinel(tmp_path):
    raw, train, folds = _write_fixture(tmp_path / "inputs")
    output = tmp_path / "output"
    stage = output.with_name(f".{output.name}.refine-staging")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    sentinel = unrelated / "sentinel.txt"
    sentinel.write_text("preserve-me", encoding="utf-8")
    try:
        os.symlink(unrelated, stage, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink unavailable: {error}")

    with pytest.raises(RuntimeError, match="non-reparse"):
        refine.run_refinement(raw, train, folds, output, resume=True)

    assert sentinel.read_text(encoding="utf-8") == "preserve-me"
    assert stage.is_symlink()


def test_resume_rejects_empty_rehashed_checkpoint_selections(tmp_path, monkeypatch):
    raw, train, folds = _write_fixture(tmp_path / "inputs")
    output = tmp_path / "output"
    original = refine._compute_unit
    calls = 0

    def interrupted(inputs, seed, fold):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("forced interruption")
        return original(inputs, seed, fold)

    monkeypatch.setattr(refine, "_compute_unit", interrupted)
    with pytest.raises(RuntimeError, match="forced interruption"):
        refine.run_refinement(raw, train, folds, output)
    monkeypatch.setattr(refine, "_compute_unit", original)

    stage = output.with_name(f".{output.name}.refine-staging")
    unit_path = stage / "checkpoints" / "seed_42__fold_0.json"
    unit = json.loads(unit_path.read_text(encoding="utf-8"))
    unit["selections"] = []
    unit_path.write_text(json.dumps(unit, sort_keys=True), encoding="utf-8")
    manifest_path = stage / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["completed"]["seed_42__fold_0"]["json_sha256"] = refine._sha256(unit_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(RuntimeError, match="exactly three selections"):
        refine.run_refinement(raw, train, folds, output, resume=True)


def test_resume_rejects_valid_shaped_rehashed_checkpoint_probability(
    tmp_path, monkeypatch
):
    raw, train, folds = _write_fixture(tmp_path / "inputs")
    output = tmp_path / "output"
    original = refine._compute_unit
    calls = 0

    def interrupted(inputs, seed, fold):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("forced interruption")
        return original(inputs, seed, fold)

    monkeypatch.setattr(refine, "_compute_unit", interrupted)
    with pytest.raises(RuntimeError, match="forced interruption"):
        refine.run_refinement(raw, train, folds, output)
    monkeypatch.setattr(refine, "_compute_unit", original)

    stage = output.with_name(f".{output.name}.refine-staging")
    unit_name = "seed_42__fold_0"
    npz_path = stage / "checkpoints" / f"{unit_name}.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    forged = arrays["outer_primary"]
    forged[:] = 1.0 / forged.shape[1]
    refine._deterministic_npz(npz_path, arrays)

    manifest_path = stage / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["completed"][unit_name]
    entry["npz_sha256"] = refine._sha256(npz_path)
    entry["member_hashes"] = {
        name: refine._array_hash(array) for name, array in sorted(arrays.items())
    }
    refine._atomic_json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match="semantic array mismatch: outer_primary"):
        refine.run_refinement(raw, train, folds, output, resume=True)


def test_corrupted_upstream_hash_is_rejected(tmp_path):
    raw, train, folds = _write_fixture(tmp_path / "inputs")
    with (raw / "nested_strategy_probability.npz").open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        refine.run_refinement(raw, train, folds, tmp_path / "output")


@pytest.mark.parametrize("forbidden", ["test.csv", "sample_submission.csv"])
def test_test_and_submission_input_paths_are_blocked_before_read(tmp_path, forbidden):
    raw, _, folds = _write_fixture(tmp_path / "inputs")
    protected = tmp_path / forbidden
    protected.write_text("must-not-read", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden Test/submission"):
        refine.run_refinement(raw, protected, folds, tmp_path / "output")


def test_probability_and_record_corruption_are_rejected(tmp_path):
    raw, train, folds = _write_fixture(tmp_path / "inputs")
    records_path = raw / "nested_inner_selection_records.json"
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    payload["records"][0]["row_indices"] = list(reversed(payload["records"][0]["row_indices"]))
    refine._atomic_json(records_path, payload)
    manifest_path = raw / "nested_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"][records_path.name] = refine._sha256(records_path)
    refine._atomic_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="row index alignment"):
        refine.run_refinement(raw, train, folds, tmp_path / "output")
