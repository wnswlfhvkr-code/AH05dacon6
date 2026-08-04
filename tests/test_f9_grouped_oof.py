import hashlib

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.pipelines.jyp_preprocessing.pipeline_jyp_f9 import (
    F9GlobalAAPairNoRawPreprocessingPipeline,
    fit_with_oof,
)


def _inputs() -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray]:
    labels = np.tile(np.asarray(["A", "B"], dtype=object), 8)
    groups = np.repeat(np.asarray([f"patient-{index}" for index in range(8)], dtype=object), 2)
    mutations = sparse.csr_matrix(
        np.column_stack(
            [labels == "A", labels == "B", np.arange(len(labels)) % 3 == 0]
        ).astype(np.float32)
    )
    missing = sparse.csr_matrix(mutations.shape, dtype=np.float32)
    return mutations, missing, labels, groups


def _fit(*, groups: object = None):
    mutations, missing, labels, _ = _inputs()
    return fit_with_oof(
        mutations,
        missing,
        labels,
        ["GA", "GB", "GC"],
        [("A", "B")],
        groups=groups,
        oof_folds=2,
        top_k_per_direction=1,
        minimum_pair_mutation_support=1,
        burden_quantiles=2,
        stability_folds=4,
        minimum_direction_consistency=4,
        minimum_selection_frequency=3,
        random_state=17,
    )


def _hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def test_legacy_oof_remains_row_stratified_and_deterministic() -> None:
    _, _, labels, _ = _inputs()
    first_model, first_oof = _fit()
    second_model, second_oof = _fit()
    expected = list(
        StratifiedKFold(n_splits=2, shuffle=True, random_state=17).split(
            np.zeros(len(labels)), labels
        )
    )

    assert first_model.oof_group_safe is False
    assert first_model.oof_group_count is None
    assert [audit.valid_index_hash for audit in first_model.oof_fold_audits] == [
        _hash(valid) for _, valid in expected
    ]
    assert first_model.oof_fold_audits == second_model.oof_fold_audits
    assert (first_oof != second_oof).nnz == 0


def test_grouped_oof_has_zero_overlap_and_is_deterministic() -> None:
    _, _, labels, groups = _inputs()
    first_model, first_oof = _fit(groups=groups)
    second_model, second_oof = _fit(groups=groups.copy())
    encoded_groups = np.repeat(np.arange(8), 2)
    expected = list(
        StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=17).split(
            np.zeros(len(labels)), labels, encoded_groups
        )
    )

    assert first_model.oof_group_safe is True
    assert first_model.oof_group_count == 8
    assert [audit.valid_index_hash for audit in first_model.oof_fold_audits] == [
        _hash(valid) for _, valid in expected
    ]
    assert all(audit.group_overlap_free for audit in first_model.oof_fold_audits)
    assert all(audit.group_overlap_count == 0 for audit in first_model.oof_fold_audits)
    assert first_model.oof_fold_audits == second_model.oof_fold_audits
    assert (first_oof != second_oof).nnz == 0


def test_f9_pipeline_exposes_grouped_oof_audit_in_summary_and_diagnostics() -> None:
    mutations, _, labels, groups = _inputs()
    values = np.where(mutations.toarray() > 0, "A1C", "WT")
    features = pd.DataFrame(values, columns=["GA", "GB", "GC"])
    pipeline = F9GlobalAAPairNoRawPreprocessingPipeline(
        f3_position_min_support=1,
        f3_aa_min_support=1,
        f4_min_support=1,
        f7_pairs=(("A", "B"),),
        f7_top_k_per_direction=1,
        f7_min_gene_support=1,
        f7_burden_quantiles=2,
        f7_stability_folds=4,
        f7_min_direction_consistency=4,
        f7_min_selection_frequency=3,
        f7_random_state=17,
        show_progress=False,
    )

    output = pipeline.fit_transform(features, pd.Series(labels), groups=pd.Series(groups))
    summary = pipeline.summary()
    diagnostics = pipeline.get_diagnostics()

    assert output.shape[1] == len(pipeline.get_feature_names_out())
    assert summary["f7_oof_group_safe"] is True
    assert summary["f7_oof_group_count"] == 8
    assert summary["f7_oof_fold_count"] == 4
    assert len(diagnostics["f7_oof_fold_audits"]) == 4
    assert all(
        audit["group_overlap_free"] and audit["group_overlap_count"] == 0
        for audit in diagnostics["f7_oof_fold_audits"]
    )


@pytest.mark.parametrize(
    ("groups", "match"),
    [
        (np.zeros((16, 1), dtype=int), "one-dimensional"),
        (np.arange(15), "row count"),
        ([*range(15), None], "missing"),
        ([*range(15), []], "hashable"),
    ],
)
def test_malformed_groups_fail(groups: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _fit(groups=groups)
