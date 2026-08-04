import hashlib

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import src.pipelines.pipeline_em_v24 as module
from src.pipelines.pipeline_em_v24 import (
    EMV24PreprocessingPipeline,
    add_signature_channel,
    create_oof_dual_signatures,
    learn_class_weights,
)


def _signature_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, np.ndarray]:
    labels = pd.Series(np.tile(np.asarray(["A", "B", "C"], dtype=object), 12))
    groups = np.repeat(
        np.asarray([f"patient-{index}" for index in range(12)], dtype=object), 3
    )
    mutation = pd.DataFrame(
        {
            "GA": labels.eq("A").astype("int8"),
            "GB": labels.eq("B").astype("int8"),
            "GC": labels.eq("C").astype("int8"),
            "COMMON": (np.arange(len(labels)) % 2 == 0).astype("int8"),
        }
    )
    functional = mutation.copy()
    return mutation, functional, labels, groups


def _kwargs() -> dict[str, int | float]:
    return {
        "top_genes_per_class": 1,
        "smoothing": 0.5,
        "max_log2_odds": 8.0,
        "shrinkage": 10.0,
        "folds": 3,
        "random_state": 17,
    }


def _manual_oof(
    mutation: pd.DataFrame,
    functional: pd.DataFrame,
    labels: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    output = pd.DataFrame(index=mutation.index)
    parameters = _kwargs()
    for train_positions, valid_positions in splits:
        fold_labels = labels.iloc[train_positions]
        fold_output = pd.DataFrame(index=mutation.iloc[valid_positions].index)
        for channel, source in (("all", mutation), ("functional", functional)):
            weights = learn_class_weights(
                source.iloc[train_positions],
                fold_labels,
                int(parameters["top_genes_per_class"]),
                float(parameters["smoothing"]),
                float(parameters["max_log2_odds"]),
                float(parameters["shrinkage"]),
                channel,
            )
            add_signature_channel(
                fold_output, source.iloc[valid_positions], weights, channel
            )
        for column in fold_output:
            output.loc[fold_output.index, column] = fold_output[column]
    return output.astype("float32")


def _index_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _pipeline_inputs() -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    _, _, labels, groups = _signature_inputs()
    features = pd.DataFrame(
        {
            "GA": np.where(labels.eq("A"), "A1V", "WT"),
            "GB": np.where(labels.eq("B"), "B2C", "WT"),
            "GC": np.where(labels.eq("C"), "C3D", "WT"),
        }
    )
    return features, labels, groups


def _pipeline() -> EMV24PreprocessingPipeline:
    return EMV24PreprocessingPipeline(
        min_mutation_count=1,
        min_functional_mutation_count=1,
        top_genes_per_class=1,
        min_hotspot_count=2,
        max_hotspots=3,
        inner_signature_folds=3,
        signature_random_state=17,
    )


def test_legacy_oof_is_exactly_the_existing_stratified_kfold_formula() -> None:
    mutation, functional, labels, _ = _signature_inputs()
    expected_splits = list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=17).split(
            mutation, labels
        )
    )

    actual = create_oof_dual_signatures(
        mutation, functional, labels, **_kwargs()
    )
    repeated = create_oof_dual_signatures(
        mutation, functional, labels, **_kwargs()
    )
    expected = _manual_oof(mutation, functional, labels, expected_splits)

    pdt.assert_frame_equal(actual, expected, check_exact=True)
    pdt.assert_frame_equal(actual, repeated, check_exact=True)


def test_grouped_oof_is_deterministic_zero_overlap_and_auditable() -> None:
    features, labels, groups = _pipeline_inputs()
    first = _pipeline()
    second = _pipeline()

    first_output = first.fit_transform(features, labels, groups=groups)
    second_output = second.fit_transform(features, labels, groups=groups.copy())
    encoded_groups = np.repeat(np.arange(12), 3)
    expected_splits = list(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=17).split(
            features, labels, encoded_groups
        )
    )
    diagnostics = first.get_diagnostics()
    audits = diagnostics["signature_oof_fold_audits"]

    pdt.assert_frame_equal(first_output, second_output, check_exact=True)
    assert first.signature_oof_fold_audits_ == second.signature_oof_fold_audits_
    assert diagnostics["signature_oof_group_safe"] is True
    assert diagnostics["signature_oof_group_count"] == 12
    assert diagnostics["signature_oof_fold_count"] == 3
    assert [audit["train_index_hash"] for audit in audits] == [
        _index_hash(train) for train, _ in expected_splits
    ]
    assert [audit["valid_index_hash"] for audit in audits] == [
        _index_hash(valid) for _, valid in expected_splits
    ]
    assert all(audit["group_overlap_count"] == 0 for audit in audits)
    assert all(audit["group_overlap_free"] is True for audit in audits)
    assert all(
        audit["train_group_count"] + audit["valid_group_count"] == 12
        for audit in audits
    )


def test_grouped_signature_shape_order_and_formula_are_unchanged() -> None:
    mutation, functional, labels, groups = _signature_inputs()
    encoded_groups = np.repeat(np.arange(12), 3)
    expected_splits = list(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=17).split(
            mutation, labels, encoded_groups
        )
    )
    expected_columns = [
        f"signature_{channel}_{class_name}_{suffix}"
        for channel in ("all", "functional")
        for class_name in ("A", "B", "C")
        for suffix in ("weighted", "match_count")
    ]

    actual = create_oof_dual_signatures(
        mutation, functional, labels, **_kwargs(), groups=groups
    )
    expected = _manual_oof(mutation, functional, labels, expected_splits)

    assert actual.shape == (36, 12)
    assert actual.columns.tolist() == expected_columns
    pdt.assert_frame_equal(actual, expected, check_exact=True)


def test_legacy_pipeline_reports_non_grouped_audit_without_changing_output() -> None:
    features, labels, _ = _pipeline_inputs()
    first = _pipeline()
    second = _pipeline()

    first_output = first.fit_transform(features, labels)
    second_output = second.fit_transform(features, labels, groups=None)

    pdt.assert_frame_equal(first_output, second_output, check_exact=True)
    summary = first.summary()
    diagnostics = first.get_diagnostics()
    assert summary["signature_oof_group_safe"] is False
    assert summary["signature_oof_group_count"] is None
    assert summary["signature_oof_fold_count"] == 3
    assert len(diagnostics["signature_oof_fold_audits"]) == 3
    assert all(
        audit["group_overlap_count"] is None
        and audit["group_overlap_free"] is None
        for audit in diagnostics["signature_oof_fold_audits"]
    )


@pytest.mark.parametrize(
    ("groups", "match"),
    [
        (np.zeros((36, 1), dtype=int), "one-dimensional"),
        (np.arange(35), "row count"),
        ([*range(35), None], "missing"),
        ([*range(35), []], "hashable"),
    ],
)
def test_malformed_groups_fail(groups: object, match: str) -> None:
    mutation, functional, labels, _ = _signature_inputs()
    with pytest.raises(ValueError, match=match):
        create_oof_dual_signatures(
            mutation, functional, labels, **_kwargs(), groups=groups
        )


def test_grouped_fold_missing_a_full_class_fails_closed_before_nan_output() -> None:
    labels = pd.Series(["A", "A", "B", "B", "C", "C"])
    groups = np.asarray(["ga", "ga", "gb1", "gb2", "gc1", "gc2"], dtype=object)
    mutation = pd.DataFrame(
        {
            "GA": labels.eq("A").astype("int8"),
            "GB": labels.eq("B").astype("int8"),
            "GC": labels.eq("C").astype("int8"),
        }
    )
    functional = mutation.copy()
    encoded_groups = np.asarray([0, 0, 1, 2, 3, 4])
    splits = list(
        StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=17).split(
            mutation, labels, encoded_groups
        )
    )

    assert all(
        not set(encoded_groups[train]).intersection(encoded_groups[valid])
        for train, valid in splits
    )
    assert any(set(labels.iloc[train]) != {"A", "B", "C"} for train, _ in splits)
    with pytest.raises(
        ValueError,
        match="training partition does not contain all full fitted classes",
    ):
        create_oof_dual_signatures(
            mutation,
            functional,
            labels,
            top_genes_per_class=1,
            smoothing=0.5,
            max_log2_odds=8.0,
            shrinkage=10.0,
            folds=2,
            random_state=17,
            groups=groups,
        )


def test_fold_channel_weight_keyset_must_equal_full_class_set(monkeypatch) -> None:
    mutation, functional, labels, groups = _signature_inputs()
    original = module.learn_class_weights

    def missing_class_weights(*args, **kwargs):
        weights = original(*args, **kwargs)
        weights.pop("C")
        return weights

    monkeypatch.setattr(module, "learn_class_weights", missing_class_weights)
    with pytest.raises(RuntimeError, match="class keyset differs"):
        create_oof_dual_signatures(
            mutation, functional, labels, **_kwargs(), groups=groups
        )


def test_nonfinite_fold_signature_fails_closed(monkeypatch) -> None:
    mutation, functional, labels, groups = _signature_inputs()
    original = module.add_signature_channel

    def add_nonfinite_signature(output, matrix, weights_by_class, channel):
        original(output, matrix, weights_by_class, channel)
        output.iloc[0, -1] = np.nan

    monkeypatch.setattr(module, "add_signature_channel", add_nonfinite_signature)
    with pytest.raises(RuntimeError, match="contains NaN or infinity"):
        create_oof_dual_signatures(
            mutation, functional, labels, **_kwargs(), groups=groups
        )


def test_pipeline_fit_transform_rejects_partial_oof_schema(monkeypatch) -> None:
    features, labels, groups = _pipeline_inputs()

    def partial_oof(*args, **kwargs):
        frame = pd.DataFrame(
            {"signature_all_A_weighted": np.zeros(len(features), dtype=np.float32)},
            index=features.index,
        )
        return frame, (), 12

    monkeypatch.setattr(module, "_create_oof_dual_signatures", partial_oof)
    with pytest.raises(RuntimeError, match="canonical dual-signature schema"):
        _pipeline().fit_transform(features, labels, groups=groups)
