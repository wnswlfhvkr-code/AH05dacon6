import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

import src.validate_preprocessing_stability as stability_validation
from src.models.xgboost_model import create_model
from src.pipelines.base import PreprocessingPipeline
from src.pipelines.jyp_preprocessing.pipeline_jyp_f8 import (
    _population_stability_index,
    build_f5_n10_baseline_config,
    build_f7_baseline_config,
    run_repeated_paired_validation,
)
from src.pipelines.jyp_preprocessing.pipeline_jyp_f10 import (
    AMINO_ACIDS,
    F10_AA_COMPOSITION_FEATURE_NAMES,
    F9_AA_SUBSTITUTIONS,
    build_f10_global_aa_composition_matrix,
    scan_mutation_frame as scan_f10_mutation_frame,
)
from src.pipelines.jyp_preprocessing.pipeline_jyp_f11 import (
    discover_f11_confusion_pairs,
)
from src.pipelines.preprocessing_registry import create_preprocessing_pipeline
from src.validate_preprocessing_stability import (
    EM16_INTERNAL_SUMMARY_COLUMNS,
    EM24_INTERNAL_SUMMARY_COLUMNS,
    _align_probability_columns,
    _blend_probabilities,
    _build_internal_fusion_matrix,
    _safe_output_stem,
    _select_internal_companion_columns,
    _select_em24_internal_columns,
    _validate_config,
    _validate_xgb_internal_fusion_config,
    _validate_xgb_blend_config,
    _xgb_blend_contract_hash,
    _xgb_internal_fusion_contract_hash,
    run_stability_validation,
    run_xgboost_internal_fusion_validation,
    run_xgboost_blend_validation,
)


JYP_PIPELINES = (
    "jyp_raw",
    "jyp_f0",
    "jyp_f0_no_raw",
    "jyp_f1",
    "jyp_f2",
    "jyp_f3_position",
    "jyp_f3",
    "jyp_f3_no_raw",
    "jyp_f4",
    "jyp_f4_no_raw",
    "jyp_f5",
    "jyp_f5_no_raw",
    "jyp_f5_no_raw_missmask",
    "jyp_f5_selective_no_raw",
    "jyp_f8",
    "jyp_f6",
    "jyp_f7",
    "jyp_f9",
    "jyp_f10",
    "jyp_f11",
)

OVERFIT_ROW_FIELDS = {
    "train_macro_f1",
    "validation_macro_f1",
    "train_validation_gap",
    "overfit_threshold",
    "is_overfitting",
}

OVERFIT_SUMMARY_FIELDS = {
    "mean_train_macro_f1",
    "mean_validation_macro_f1",
    "mean_train_validation_gap",
    "min_train_validation_gap",
    "max_train_validation_gap",
    "overfit_threshold",
    "overfit_fold_count",
    "total_fold_count",
    "is_overfitting",
}


def _assert_overfit_row(row: dict[str, object]) -> None:
    assert OVERFIT_ROW_FIELDS <= row.keys()
    assert row["validation_macro_f1"] == row["macro_f1"]
    assert row["train_validation_gap"] == pytest.approx(
        float(row["train_macro_f1"]) - float(row["validation_macro_f1"])
    )
    assert row["overfit_threshold"] == 0.1
    assert row["is_overfitting"] == (row["train_validation_gap"] > 0.1)


def _assert_overfit_summary(
    summary: dict[str, object],
    rows: list[dict[str, object]],
) -> None:
    assert OVERFIT_SUMMARY_FIELDS <= summary.keys()
    gaps = np.asarray(
        [float(row["train_validation_gap"]) for row in rows], dtype=np.float64
    )
    assert summary["mean_train_macro_f1"] == pytest.approx(
        np.mean([float(row["train_macro_f1"]) for row in rows])
    )
    assert summary["mean_validation_macro_f1"] == pytest.approx(
        np.mean([float(row["validation_macro_f1"]) for row in rows])
    )
    assert summary["mean_train_validation_gap"] == pytest.approx(gaps.mean())
    assert summary["min_train_validation_gap"] == pytest.approx(gaps.min())
    assert summary["max_train_validation_gap"] == pytest.approx(gaps.max())
    assert summary["overfit_threshold"] == 0.1
    assert summary["overfit_fold_count"] == sum(
        bool(row["is_overfitting"]) for row in rows
    )
    assert summary["total_fold_count"] == len(rows)
    assert summary["is_overfitting"] == (
        summary["mean_train_validation_gap"] > 0.1
    )


def test_preprocessor_removes_only_constant_columns() -> None:
    features = pd.DataFrame({
        "numeric": ["1.0", None, "100.0", "3.0"],
        "duplicate_numeric": ["1.0", None, "100.0", "3.0"],
        "constant": ["same"] * 4,
        "category": ["a", None, "b", "a"],
    })
    labels = pd.Series(["x", "y", "x", "y"])
    preprocessor = PreprocessingPipeline().fit(features, labels)
    transformed = preprocessor.transform(features)

    assert preprocessor.dropped_constant_columns == ["constant"]
    assert transformed.columns.tolist() == ["numeric", "duplicate_numeric", "category"]
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in transformed.dtypes)

    unseen = pd.DataFrame({
        "numeric": ["not_seen"],
        "duplicate_numeric": ["1.0"],
        "constant": ["same"],
        "category": ["not_seen"],
    })
    assert preprocessor.transform(unseen).iloc[0].tolist() == [-1.0, 0.0, -1.0]


def test_pipeline_factory_creates_baseline() -> None:
    pipeline = create_preprocessing_pipeline({"name": "baseline"})

    assert pipeline.name == "baseline"
    assert pipeline.steps == ("상수 열 제거", "범주형 순서 인코딩")


def test_xgboost_accepts_ordinal_encoded_features() -> None:
    features = pd.DataFrame({
        "gene_a": ["WT", "M1", "WT", "M2", "WT", "M1", "WT", "M2"],
        "gene_b": ["WT", "WT", "M3", "WT", "M3", "WT", "M3", "WT"],
    })
    labels = pd.Series(["A", "B", "A", "B", "A", "B", "A", "B"])
    preprocessor = PreprocessingPipeline().fit(features, labels)
    transformed = preprocessor.transform(features)
    model = create_model(
        {
            "n_estimators": 2,
            "learning_rate": 0.1,
            "max_depth": 2,
            "n_jobs": 1,
        },
        seed=42,
    )

    assert model.get_params()["device"] == "cuda"
    model.fit(transformed, preprocessor.encode_labels(labels))
    booster_config = json.loads(model.get_booster().save_config())

    assert len(model.predict(transformed)) == len(features)
    assert booster_config["learner"]["generic_param"]["device"] == "cuda:0"


def test_xgboost_explicit_cpu_device_overrides_cuda_default() -> None:
    model = create_model({"n_estimators": 1, "device": "cpu"}, seed=42)

    assert model.get_params()["device"] == "cpu"


def test_jyp_registry_uses_distinct_direct_pipeline_classes() -> None:
    pipelines = [
        create_preprocessing_pipeline({"name": name, "show_progress": False})
        for name in JYP_PIPELINES
    ]

    assert len({type(pipeline) for pipeline in pipelines}) == len(JYP_PIPELINES)
    assert len({type(pipeline).__module__ for pipeline in pipelines}) == len(
        JYP_PIPELINES
    )
    assert all(
        type(pipeline).__bases__ == (PreprocessingPipeline,)
        for pipeline in pipelines
    )


def test_jyp_promoted_recipes_are_code_owned_defaults() -> None:
    common_selected_names = (
        "jyp_f4_no_raw",
        "jyp_f5_no_raw",
        "jyp_f5_no_raw_missmask",
        "jyp_f5_selective_no_raw",
        "jyp_f6",
        "jyp_f7",
        "jyp_f8",
        "jyp_f9",
        "jyp_f10",
        "jyp_f11",
    )
    pipelines = {
        name: create_preprocessing_pipeline({"name": name})
        for name in common_selected_names
    }
    for pipeline in pipelines.values():
        assert (
            pipeline.burden_clip_quantile,
            pipeline.f3_position_min_support,
            pipeline.f3_aa_min_support,
            pipeline.f4_min_support,
        ) == (0.99, 3, 3, 5)

    for name in (
        "jyp_f5_no_raw",
        "jyp_f5_no_raw_missmask",
        "jyp_f5_selective_no_raw",
        "jyp_f8",
    ):
        pipeline = pipelines[name]
        assert (
            pipeline.f5_top_k_per_class,
            pipeline.f5_min_gene_support,
            pipeline.f5_laplace_alpha,
            pipeline.f5_stability_folds,
            pipeline.f5_min_direction_consistency,
            pipeline.f5_min_selection_frequency,
            pipeline.f5_random_state,
        ) == (10, 3, 1.0, 5, 3, 2, 42)
    assert pipelines["jyp_f5_selective_no_raw"].f5_output_rare_class_count == 10
    assert pipelines["jyp_f8"].f5_output_rare_class_count == 10

    for name in ("jyp_f7", "jyp_f8", "jyp_f9", "jyp_f10", "jyp_f11"):
        pipeline = pipelines[name]
        assert (
            pipeline.f7_pairs,
            pipeline.f7_top_k_per_direction,
            pipeline.f7_min_gene_support,
            pipeline.f7_laplace_alpha,
            pipeline.f7_burden_quantiles,
            pipeline.f7_stability_folds,
            pipeline.f7_min_direction_consistency,
            pipeline.f7_min_selection_frequency,
            pipeline.f7_random_state,
        ) == (
            (("KIRC", "KIPAN"), ("LGG", "GBMLGG")),
            3,
            10,
            4.0,
            5,
            5,
            4,
            3,
            42,
        )
    assert pipelines["jyp_f9"].show_progress is False
    assert (
        pipelines["jyp_f6"].f6_min_document_frequency,
        pipelines["jyp_f6"].f6_max_components,
        pipelines["jyp_f6"].f6_random_state,
    ) == (2, 128, 42)

    pipe_comb = create_preprocessing_pipeline({"name": "pipeComb_v3"})
    assert pipe_comb.f9_parameters == {
        "burden_clip_quantile": 0.99,
        "f3_position_min_support": 3,
        "f3_aa_min_support": 3,
        "f4_min_support": 5,
        "f7_pairs": (("KIRC", "KIPAN"), ("LGG", "GBMLGG")),
        "f7_top_k_per_direction": 3,
        "f7_min_gene_support": 10,
        "f7_laplace_alpha": 4.0,
        "f7_burden_quantiles": 5,
        "f7_stability_folds": 5,
        "f7_min_direction_consistency": 4,
        "f7_min_selection_frequency": 3,
        "f7_random_state": 42,
        "show_progress": True,
        "progress_interval": 25000,
    }
    assert pipe_comb.em24_parameters == {
        "min_mutation_count": 5,
        "min_functional_mutation_count": 5,
        "top_genes_per_class": 20,
        "smoothing": 0.5,
        "max_log2_odds": 8.0,
        "shrinkage": 10.0,
        "min_hotspot_count": 5,
        "max_hotspots": 384,
        "inner_signature_folds": 5,
        "signature_random_state": 42,
    }


def test_pipe_comb_v3_adds_only_em24_dual_signatures_after_f9() -> None:
    classes = ["KIRC", "KIPAN", "LGG", "GBMLGG"]
    genes = ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "PASSENGER"]
    rows: list[dict[str, str]] = []
    label_values: list[str] = []
    for class_index, class_name in enumerate(classes):
        for sample_index in range(8):
            row = {gene: "WT" for gene in genes}
            row[genes[class_index]] = f"A{class_index + 1}V"
            if sample_index % 2 == 0:
                row["PASSENGER"] = "R9Q"
            rows.append(row)
            label_values.append(class_name)

    features = pd.DataFrame(rows)
    labels = pd.Series(label_values, index=features.index)
    pipeline = create_preprocessing_pipeline({
        "name": "pipeComb_v3",
        "f3_position_min_support": 1,
        "f3_aa_min_support": 1,
        "f4_min_support": 1,
        "f7_top_k_per_direction": 1,
        "f7_min_gene_support": 1,
        "f7_stability_folds": 4,
        "f7_min_direction_consistency": 4,
        "f7_min_selection_frequency": 3,
        "em24_min_mutation_count": 1,
        "em24_min_functional_mutation_count": 1,
        "em24_top_genes_per_class": 2,
        "em24_min_hotspot_count": 1,
        "em24_inner_signature_folds": 4,
        "show_progress": False,
    })

    train_matrix = pipeline.fit_transform(features, labels)
    full_fit_train_matrix = pipeline.transform(features)
    test_matrix = pipeline.transform(features.iloc[:4])
    feature_names = pipeline.get_feature_names_out().astype(str).tolist()
    em24_names = [name for name in feature_names if name.startswith("EM24__")]

    assert type(pipeline).__bases__ == (PreprocessingPipeline,)
    assert sparse.isspmatrix_csr(train_matrix)
    assert train_matrix.dtype == np.float32
    assert train_matrix.shape[1] == test_matrix.shape[1] == len(feature_names)
    assert len(em24_names) == 4 * len(classes)
    assert all(
        name.startswith("EM24__signature_all_")
        or name.startswith("EM24__signature_functional_")
        for name in em24_names
    )
    assert feature_names[-len(em24_names):] == em24_names
    assert pipeline.summary()["em24_dual_signature_features"] == 16
    assert not np.allclose(
        train_matrix[:, -len(em24_names):].toarray(),
        full_fit_train_matrix[:, -len(em24_names):].toarray(),
    )

    reference_f9 = create_preprocessing_pipeline({
        "name": "jyp_f9",
        **pipeline.f9_parameters,
    })
    reference_em24 = create_preprocessing_pipeline({
        "name": "em_v24",
        **pipeline.em24_parameters,
    })
    reference_matrix, reference_suffix_names = _build_internal_fusion_matrix(
        reference_f9.fit_transform(features, labels),
        reference_em24.fit_transform(features, labels),
        ["signature_all", "signature_functional"],
    )
    np.testing.assert_allclose(train_matrix.toarray(), reference_matrix.toarray())
    assert feature_names == [
        *reference_f9.get_feature_names_out().astype(str).tolist(),
        *reference_suffix_names,
    ]

    restored = pickle.loads(pickle.dumps(pipeline))
    np.testing.assert_allclose(
        restored.transform(features.iloc[:4]).toarray(),
        test_matrix.toarray(),
    )


def test_jyp_pipeline_rejects_stage_and_irrelevant_parameters() -> None:
    with pytest.raises(TypeError):
        create_preprocessing_pipeline({"name": "jyp_f0", "feature_stage": "f4"})
    with pytest.raises(TypeError):
        create_preprocessing_pipeline({"name": "jyp_f0", "f4_min_support": 5})
    with pytest.raises(ValueError):
        create_preprocessing_pipeline({"name": "f7_paircontrast_no_raw"})


def test_jyp_f0_preserves_raw_and_adds_binary_gene_features() -> None:
    features = pd.DataFrame({
        "GENE_A": ["WT", "R1Q", None, "WT"],
        "GENE_B": ["WT", "WT", "Q2*", "A3V B4V"],
    })
    labels = pd.Series(["A", "B", "A", "B"])
    pipeline = create_preprocessing_pipeline({
        "name": "jyp_f0",
        "show_progress": False,
    })
    matrix = pipeline.fit_transform(features, labels)

    assert matrix.shape == (4, 4)
    assert pipeline.get_feature_names_out().tolist() == [
        "RAW__gene__GENE_A",
        "RAW__gene__GENE_B",
        "F0__gene__GENE_A",
        "F0__gene__GENE_B",
    ]
    assert matrix[:, -2:].toarray().tolist() == [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]
    assert pipeline.summary()["pipeline_name"] == "jyp_f0"


def test_jyp_f4_no_raw_fit_transform_matches_full_transform() -> None:
    features = pd.DataFrame({
        "GENE_A": ["WT", "R1Q", None, "R1Q"],
        "GENE_B": ["WT", "WT", "Q2*", "A3V B4V"],
    })
    labels = pd.Series(["A", "B", "A", "B"])
    config = {
        "name": "jyp_f4_no_raw",
        "f3_position_min_support": 1,
        "f3_aa_min_support": 1,
        "f4_min_support": 1,
        "show_progress": False,
    }

    pipeline = create_preprocessing_pipeline(config)
    fit_matrix = pipeline.fit_transform(features, labels)
    full_transform_matrix = pipeline.transform(features.copy())
    restored_matrix = pickle.loads(pickle.dumps(pipeline)).transform(features.copy())
    feature_names = pipeline.get_feature_names_out().tolist()

    assert (fit_matrix != full_transform_matrix).nnz == 0
    assert (restored_matrix != full_transform_matrix).nnz == 0
    assert fit_matrix.shape[1] == len(feature_names)
    assert not [name for name in feature_names if name.startswith("RAW__")]


def test_jyp_f7_uses_oof_for_shared_trainer_fit_transform_sequence() -> None:
    rows = []
    labels = []
    for repeat in range(10):
        for label in ("KIRC", "KIPAN", "LGG", "GBMLGG"):
            labels.append(label)
            rows.append({
                "VHL": "V100A" if label == "KIRC" else "WT",
                "PBRM1": "P200L" if label == "KIPAN" else "WT",
                "IDH1": "R132H" if label == "LGG" else "WT",
                "EGFR": "A289V" if label == "GBMLGG" else "WT",
                "TP53": None if repeat == 0 else ("R175H" if repeat % 2 else "WT"),
            })
    features = pd.DataFrame(rows)
    labels = pd.Series(labels)
    config = {
        "name": "jyp_f7",
        "f3_position_min_support": 1,
        "f3_aa_min_support": 1,
        "f4_min_support": 2,
        "f7_top_k_per_direction": 1,
        "f7_min_gene_support": 2,
        "f7_burden_quantiles": 2,
        "f7_stability_folds": 5,
        "f7_min_direction_consistency": 4,
        "f7_min_selection_frequency": 3,
        "show_progress": False,
    }

    pipeline = create_preprocessing_pipeline(config)
    oof_matrix = pipeline.fit_transform(features, labels)
    full_fit_matrix = pipeline.transform(features)
    copied_full_fit_matrix = pipeline.transform(features.copy())
    restored = pickle.loads(pickle.dumps(pipeline))
    restored_matrix = restored.transform(features.copy())
    test_matrix = pipeline.transform(features.iloc[:4].copy())
    feature_names = pipeline.get_feature_names_out().tolist()
    f7_names = [name for name in feature_names if name.startswith("F7__")]

    assert (oof_matrix[:, -4:] != full_fit_matrix[:, -4:]).nnz > 0
    assert (full_fit_matrix != copied_full_fit_matrix).nnz == 0
    assert (restored_matrix != full_fit_matrix).nnz == 0
    assert restored.__dict__.get("_fit_scan_") is None
    assert restored.__dict__.get("_fit_f7_oof_matrix_") is None
    assert not [name for name in feature_names if name.startswith("RAW__")]
    assert f7_names == [
        "F7__KIRC_vs_KIPAN__mutated_log_odds_mean",
        "F7__KIRC_vs_KIPAN__bernoulli_llr_mean",
        "F7__LGG_vs_GBMLGG__mutated_log_odds_mean",
        "F7__LGG_vs_GBMLGG__bernoulli_llr_mean",
    ]
    assert oof_matrix.shape[1] == test_matrix.shape[1] == len(feature_names)


def test_jyp_f10_global_aa_composition_has_fixed_values_and_zero_policy() -> None:
    features = pd.DataFrame({
        "GENE_A": ["A1C A2C A3A A4*", "A9C", "WT"],
        "GENE_B": ["A5C C6A", "WT", None],
    })
    scan = scan_f10_mutation_frame(
        features,
        stage="unit",
        show_progress=False,
    )
    matrix = build_f10_global_aa_composition_matrix(scan).toarray()
    amino_a = AMINO_ACIDS.index("A")
    amino_c = AMINO_ACIDS.index("C")
    expected_entropy = -(
        0.75 * np.log(0.75) + 0.25 * np.log(0.25)
    ) / np.log(len(F9_AA_SUBSTITUTIONS))
    expected_names = [
        *(f"F10__global_aa_from_fraction__{amino_acid}" for amino_acid in AMINO_ACIDS),
        *(f"F10__global_aa_to_fraction__{amino_acid}" for amino_acid in AMINO_ACIDS),
        "F10__global_aa_pair_entropy_normalized",
        "F10__global_aa_pair_dominant_fraction",
    ]

    assert list(F10_AA_COMPOSITION_FEATURE_NAMES) == expected_names
    assert matrix.shape == (3, 42)
    assert matrix.dtype == np.float32
    assert np.isclose(matrix[0, amino_a], 0.75)
    assert np.isclose(matrix[0, amino_c], 0.25)
    assert np.isclose(matrix[0, 20 + amino_a], 0.25)
    assert np.isclose(matrix[0, 20 + amino_c], 0.75)
    assert np.isclose(matrix[0, 40], expected_entropy)
    assert np.isclose(matrix[0, 41], 0.75)
    assert np.isclose(matrix[0, :20].sum(), 1.0)
    assert np.isclose(matrix[0, 20:40].sum(), 1.0)
    assert np.isclose(matrix[1, 40], 0.0)
    assert np.isclose(matrix[1, 41], 1.0)
    assert not matrix[2].any()


def test_jyp_f10_preserves_f9_prefix_and_target_independent_suffix() -> None:
    rows = []
    labels = []
    for repeat in range(10):
        for label in ("KIRC", "KIPAN", "LGG", "GBMLGG"):
            labels.append(label)
            rows.append({
                "VHL": "A1C A2C" if label == "KIRC" else "WT",
                "PBRM1": "P200L" if label == "KIPAN" else "WT",
                "IDH1": "R132H" if label == "LGG" else "WT",
                "EGFR": "A289V" if label == "GBMLGG" else "WT",
                "TP53": None if repeat == 0 else ("R175H" if repeat % 2 else "WT"),
            })
    features = pd.DataFrame(rows)
    labels = pd.Series(labels)
    common = {
        "f3_position_min_support": 1,
        "f3_aa_min_support": 1,
        "f4_min_support": 2,
        "f7_top_k_per_direction": 1,
        "f7_min_gene_support": 2,
        "f7_burden_quantiles": 2,
        "f7_stability_folds": 5,
        "f7_min_direction_consistency": 4,
        "f7_min_selection_frequency": 3,
        "show_progress": False,
    }
    f9 = create_preprocessing_pipeline({"name": "jyp_f9", **common})
    f10 = create_preprocessing_pipeline({"name": "jyp_f10", **common})
    relabeled_f10 = create_preprocessing_pipeline({"name": "jyp_f10", **common})

    f9.fit_transform(features, labels)
    f9_full = f9.transform(features)
    f10_oof = f10.fit_transform(features, labels)
    f10_full = f10.transform(features)
    relabeled_oof = relabeled_f10.fit_transform(
        features,
        pd.Series(np.roll(labels.to_numpy(), 1), index=labels.index),
    )
    feature_names = f10.get_feature_names_out().tolist()
    f10_indices = [
        index for index, name in enumerate(feature_names) if name.startswith("F10__")
    ]

    assert len(f10_indices) == 42
    assert f10_indices == list(range(len(feature_names) - 42, len(feature_names)))
    assert f10_full.shape[1] == f9_full.shape[1] + 42
    assert (f10_full[:, :f9_full.shape[1]] != f9_full).nnz == 0
    assert (f10_oof[:, f10_indices] != f10_full[:, f10_indices]).nnz == 0
    assert (f10_oof[:, f10_indices] != relabeled_oof[:, f10_indices]).nnz == 0

    restored = pickle.loads(pickle.dumps(f10))
    assert (restored.transform(features) != f10_full).nnz == 0
    assert restored.get_feature_names_out().tolist() == feature_names
    assert restored.summary()["f10_global_aa_composition_features"] == 42


def test_jyp_f11_confusion_discovery_is_deterministic_and_excludes_manual_pair() -> None:
    labels = np.repeat(np.asarray(["A", "B", "C", "D"], dtype=object), 9)
    matrix = sparse.csr_matrix(
        np.vstack([
            np.tile([1.0, 0.0], (18, 1)),
            np.tile([0.0, 1.0], (18, 1)),
        ]),
        dtype=np.float32,
    )
    kwargs = {
        "pair_count": 2,
        "discovery_folds": 3,
        "regularization_c": 0.07,
        "max_iter": 500,
        "random_state": 42,
    }

    first = discover_f11_confusion_pairs(
        matrix,
        labels,
        [("B", "A")],
        **kwargs,
    )
    second = discover_f11_confusion_pairs(
        matrix,
        labels,
        [("A", "B")],
        **kwargs,
    )
    first_catalog, first_labels, first_confusion, first_warnings = first
    second_catalog, second_labels, second_confusion, second_warnings = second
    selected = sorted(
        (
            candidate
            for candidate in first_catalog
            if candidate.selected_rank is not None
        ),
        key=lambda candidate: int(candidate.selected_rank),
    )

    assert first_catalog == second_catalog
    assert first_labels == second_labels == ("A", "B", "C", "D")
    assert np.array_equal(first_confusion, second_confusion)
    assert first_warnings == second_warnings == 0
    assert len(selected) == 2
    assert (selected[0].pair.first_label, selected[0].pair.second_label) == ("C", "D")
    assert all(
        frozenset((candidate.pair.first_label, candidate.pair.second_label))
        != frozenset(("A", "B"))
        for candidate in selected
    )


def test_jyp_f11_preserves_f9_prefix_and_uses_oof_auto_pair_values() -> None:
    class_labels = ("KIRC", "KIPAN", "LGG", "GBMLGG", "PCPG", "THYM")
    rows = []
    labels = []
    for repeat in range(10):
        for label in class_labels:
            labels.append(label)
            rows.append({
                "VHL": "V100A" if label == "KIRC" else "WT",
                "PBRM1": "P200L" if label == "KIPAN" else "WT",
                "IDH1": "R132H" if label == "LGG" else "WT",
                "EGFR": "A289V" if label == "GBMLGG" else "WT",
                "RET": "A10V" if label in {"PCPG", "THYM"} else "WT",
                "TP53": "R175H" if label == "PCPG" and repeat % 2 else "WT",
                "MISSING_GENE": None if repeat == 0 else "WT",
            })
    features = pd.DataFrame(rows)
    labels = pd.Series(labels)
    common = {
        "f3_position_min_support": 1,
        "f3_aa_min_support": 1,
        "f4_min_support": 2,
        "f7_top_k_per_direction": 1,
        "f7_min_gene_support": 2,
        "f7_burden_quantiles": 2,
        "f7_stability_folds": 5,
        "f7_min_direction_consistency": 4,
        "f7_min_selection_frequency": 3,
        "show_progress": False,
    }
    f9 = create_preprocessing_pipeline({"name": "jyp_f9", **common})
    f11 = create_preprocessing_pipeline({
        "name": "jyp_f11",
        **common,
        "f11_discovery_folds": 3,
        "f11_discovery_max_iter": 1000,
    })

    f9_oof = f9.fit_transform(features, labels)
    f9_full = f9.transform(features)
    f11_oof = f11.fit_transform(features, labels)
    f11_full = f11.transform(features)
    feature_names = f11.get_feature_names_out().tolist()
    f11_indices = [
        index for index, name in enumerate(feature_names) if name.startswith("F11__")
    ]
    manual_pair_keys = {
        frozenset(("KIRC", "KIPAN")),
        frozenset(("LGG", "GBMLGG")),
    }

    assert len(f11_indices) == 12
    assert not [name for name in feature_names if name.startswith("F10__")]
    assert f11_oof.shape[1] == f9_oof.shape[1] + 12
    assert (f11_oof[:, :f9_oof.shape[1]] != f9_oof).nnz == 0
    assert (f11_full[:, :f9_full.shape[1]] != f9_full).nnz == 0
    assert (f11_oof[:, f11_indices] != f11_full[:, f11_indices]).nnz > 0
    assert all(
        frozenset((pair.first_label, pair.second_label)) not in manual_pair_keys
        for pair in f11.f11_pairs_
    )

    diagnostics = f11.get_diagnostics()
    assert diagnostics["f11_discovery_confusion_matrix"].shape == (6, 6)
    assert f11.summary()["f11_selected_pair_count"] == 6
    assert f11.summary()["f11_training_value_semantics"] == (
        "pair_schema_supervised_full_training__values_oof__outer_validation_train_fitted"
    )

    restored = pickle.loads(pickle.dumps(f11))
    assert restored.__dict__.get("_fit_scan_") is None
    assert restored.__dict__.get("_fit_f7_oof_matrix_") is None
    assert restored.__dict__.get("_fit_f11_oof_matrix_") is None
    assert (restored.transform(features) != f11_full).nnz == 0
    assert restored.get_feature_names_out().tolist() == feature_names


@pytest.mark.parametrize(
    "config",
    [
        {"f11_auto_pair_count": True},
        {"f11_discovery_folds": 1},
        {"f11_discovery_c": 0.0},
        {"f11_discovery_c": np.inf},
        {"f11_discovery_max_iter": 0},
        {"f11_random_state": True},
        {"f11_random_state": 1.5},
        {"f11_random_state": -1},
    ],
)
def test_jyp_f11_rejects_invalid_discovery_parameters(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        create_preprocessing_pipeline({
            "name": "jyp_f11",
            "show_progress": False,
            **config,
        })


@pytest.mark.parametrize(
    "pipeline_name",
    [
        "jyp_f5",
        "jyp_f5_no_raw",
        "jyp_f5_no_raw_missmask",
        "jyp_f5_selective_no_raw",
    ],
)
def test_jyp_f5_uses_oof_only_in_fit_transform(pipeline_name: str) -> None:
    rows = []
    labels = []
    for repeat in range(10):
        for label in ("KIRC", "KIPAN", "LGG", "GBMLGG"):
            labels.append(label)
            rows.append({
                "VHL": "V100A" if label == "KIRC" else "WT",
                "PBRM1": "P200L" if label == "KIPAN" else "WT",
                "IDH1": "R132H" if label == "LGG" else "WT",
                "EGFR": "A289V" if label == "GBMLGG" else "WT",
                "TP53": None if repeat == 0 else ("R175H" if repeat % 2 else "WT"),
            })
    features = pd.DataFrame(rows)
    labels = pd.Series(labels)
    config = {
        "name": pipeline_name,
        "f3_position_min_support": 1,
        "f3_aa_min_support": 1,
        "f4_min_support": 2,
        "f5_top_k_per_class": 1,
        "f5_min_gene_support": 2,
        "f5_stability_folds": 5,
        "f5_min_direction_consistency": 4,
        "f5_min_selection_frequency": 3,
        "show_progress": False,
    }
    if pipeline_name == "jyp_f5_selective_no_raw":
        config["f5_output_rare_class_count"] = None

    pipeline = create_preprocessing_pipeline(config)
    oof_matrix = pipeline.fit_transform(features, labels)
    full_fit_matrix = pipeline.transform(features)
    copied_full_fit_matrix = pipeline.transform(features.copy())
    restored = pickle.loads(pickle.dumps(pipeline))
    restored_matrix = restored.transform(features.copy())
    feature_names = pipeline.get_feature_names_out().tolist()
    f5_names = [name for name in feature_names if name.startswith("F5__")]

    assert len(f5_names) == 16
    assert (oof_matrix[:, -16:] != full_fit_matrix[:, -16:]).nnz > 0
    assert (full_fit_matrix != copied_full_fit_matrix).nnz == 0
    assert (restored_matrix != full_fit_matrix).nnz == 0
    assert restored.__dict__.get("_fit_scan_") is None
    assert restored.__dict__.get("_fit_f5_oof_matrix_") is None
    assert oof_matrix.shape == full_fit_matrix.shape


def test_jyp_f8_combines_f5_selective_and_f7_oof_blocks() -> None:
    rows = []
    labels = []
    class_labels = [f"C{index:02d}" for index in range(11)]
    gene_columns = [f"GENE_{index:02d}" for index in range(11)]
    for class_index, label in enumerate(class_labels):
        repeat_count = 10 if class_index < 10 else 11
        for repeat in range(repeat_count):
            row = {gene: "WT" for gene in gene_columns}
            row[gene_columns[class_index]] = f"A{class_index + 1}V"
            row["MISSING_GENE"] = None if repeat == 0 else "WT"
            rows.append(row)
            labels.append(label)
    features = pd.DataFrame(rows)
    labels = pd.Series(labels)
    config = {
        "name": "jyp_f8",
        "f3_position_min_support": 1,
        "f3_aa_min_support": 1,
        "f4_min_support": 2,
        "f5_top_k_per_class": 1,
        "f5_min_gene_support": 2,
        "f5_stability_folds": 5,
        "f5_min_direction_consistency": 4,
        "f5_min_selection_frequency": 3,
        "f5_output_rare_class_count": 10,
        "f7_pairs": [["C00", "C01"], ["C02", "C03"]],
        "f7_top_k_per_direction": 1,
        "f7_min_gene_support": 2,
        "f7_burden_quantiles": 2,
        "f7_stability_folds": 5,
        "f7_min_direction_consistency": 4,
        "f7_min_selection_frequency": 3,
        "show_progress": False,
    }

    pipeline = create_preprocessing_pipeline(config)
    oof_matrix = pipeline.fit_transform(features, labels)
    full_fit_matrix = pipeline.transform(features.copy())
    feature_names = pipeline.get_feature_names_out().tolist()
    f5_indices = [
        index for index, name in enumerate(feature_names) if name.startswith("F5__")
    ]
    f7_indices = [
        index for index, name in enumerate(feature_names) if name.startswith("F7__")
    ]

    assert len(f5_indices) == 40
    assert len(f7_indices) == 4
    assert max(f5_indices) < min(f7_indices)
    assert not [name for name in feature_names if name.startswith("RAW__")]
    assert (oof_matrix[:, f5_indices] != full_fit_matrix[:, f5_indices]).nnz > 0
    assert (oof_matrix[:, f7_indices] != full_fit_matrix[:, f7_indices]).nnz > 0

    all_missing = pd.DataFrame(
        [{column: " NA " for column in features.columns}],
        columns=features.columns,
    )
    missing_matrix = pipeline.transform(all_missing).toarray()[0]
    likelihood_indices = [
        index
        for index, name in enumerate(feature_names)
        if name.endswith("__bernoulli_log_likelihood")
        or name.endswith("__bernoulli_llr_mean")
    ]
    assert likelihood_indices
    assert all(missing_matrix[index] == 0.0 for index in likelihood_indices)

    restored = pickle.loads(pickle.dumps(pipeline))
    restored_matrix = restored.transform(features.copy())
    assert (restored_matrix != full_fit_matrix).nnz == 0
    assert restored.__dict__.get("_fit_scan_") is None
    assert restored.__dict__.get("_fit_f5_oof_matrix_") is None
    assert restored.__dict__.get("_fit_f7_oof_matrix_") is None
    assert restored.get_feature_names_out().tolist() == feature_names

    summary = pipeline.summary()
    assert summary["f5_signature_features"] == 40
    assert summary["f7_pair_contrast_features"] == 4
    assert summary["includes_raw_ordinal"] is False


def test_jyp_f8_repeated_validation_uses_paired_folds() -> None:
    features = pd.DataFrame({
        "signal": [0, 0, 0, 0, 1, 1, 1, 1],
        "candidate_signal": [0, 1, 0, 1, 1, 0, 1, 0],
    })
    labels = pd.Series(["A"] * 4 + ["B"] * 4, index=features.index)
    calls: list[tuple[str, str, tuple[int, ...]]] = []
    factory_configs: list[dict[str, object]] = []

    class RecordingPipeline:
        def __init__(self, recipe: str) -> None:
            self.recipe = recipe
            self.encoder = LabelEncoder()

        def _matrix(self, frame: pd.DataFrame):
            columns = [frame["signal"].to_numpy(dtype=np.float32)]
            if self.recipe == "jyp_f8":
                columns.append(frame["candidate_signal"].to_numpy(dtype=np.float32))
            return sparse.csr_matrix(np.column_stack(columns))

        def fit_transform(self, frame: pd.DataFrame, target: pd.Series):
            calls.append(("fit", self.recipe, tuple(frame.index)))
            self.encoder.fit(target.to_numpy())
            return self._matrix(frame)

        def transform(self, frame: pd.DataFrame):
            calls.append(("transform", self.recipe, tuple(frame.index)))
            return self._matrix(frame)

        def encode_labels(self, target: pd.Series) -> np.ndarray:
            return self.encoder.transform(target.to_numpy())

        def decode_labels(self, target: np.ndarray) -> np.ndarray:
            return self.encoder.inverse_transform(np.asarray(target, dtype=int))

        def get_feature_names_out(self) -> np.ndarray:
            names = ["F0__signal"]
            if self.recipe == "jyp_f8":
                names.append("F7__dummy")
            return np.asarray(names, dtype=object)

    class SignalClassifier:
        def fit(self, matrix, target):
            return self

        def predict(self, matrix) -> np.ndarray:
            return (matrix[:, 0].toarray().ravel() > 0.5).astype(int)

    def pipeline_factory(config: dict[str, object]) -> RecordingPipeline:
        factory_configs.append(dict(config))
        return RecordingPipeline(str(config["name"]))

    def model_builder(config: dict[str, object], seed: int) -> SignalClassifier:
        assert config["name"] == "xgboost"
        assert isinstance(seed, int)
        return SignalClassifier()

    config = {
        "model": {"name": "xgboost"},
        "preprocessing": {
            "name": "jyp_f8",
            "f5_output_rare_class_count": 10,
            "f7_top_k_per_direction": 3,
        },
    }
    result = run_repeated_paired_validation(
        config,
        features,
        labels,
        seeds=[11, 22],
        n_splits=2,
        pipeline_factory=pipeline_factory,
        model_builder=model_builder,
        show_progress=False,
    )

    assert result["fold_count"] == 4
    assert result["summary"]["mean_delta_oof_macro_f1"] == 0.0
    assert result["summary"]["fold_ties"] == 4
    assert result["summary"]["decision"] == "hold_no_mean_gain"
    assert factory_configs
    assert all(config["show_progress"] is False for config in factory_configs)
    fit_calls = [call for call in calls if call[0] == "fit"]
    transform_calls = [call for call in calls if call[0] == "transform"]
    assert len(fit_calls) == len(transform_calls) == 8
    for offset in range(0, len(fit_calls), 2):
        assert fit_calls[offset][1:] == ("jyp_f5_selective_no_raw", fit_calls[offset][2])
        assert fit_calls[offset + 1][1:] == ("jyp_f8", fit_calls[offset + 1][2])
        assert fit_calls[offset][2] == fit_calls[offset + 1][2]
        assert transform_calls[offset][2] == transform_calls[offset + 1][2]

    calls.clear()
    factory_configs.clear()
    f7_result = run_repeated_paired_validation(
        config,
        features,
        labels,
        baseline="jyp_f7",
        seeds=[11],
        n_splits=2,
        pipeline_factory=pipeline_factory,
        model_builder=model_builder,
        show_progress=False,
    )

    assert f7_result["comparison"]["baseline"] == "jyp_f7"
    assert f7_result["fold_count"] == 2
    assert "f7_mean_oof_macro_f1" in f7_result["summary"]
    assert factory_configs[0]["name"] == "jyp_f7"
    assert "f5_output_rare_class_count" not in factory_configs[0]
    assert factory_configs[0]["f7_top_k_per_direction"] == 3


def test_jyp_f8_repeated_validation_builds_real_n10_baseline_config() -> None:
    baseline = build_f5_n10_baseline_config({
        "name": "jyp_f8",
        "f3_aa_min_support": 3,
        "f5_output_rare_class_count": 10,
        "f7_pairs": [["KIRC", "KIPAN"]],
        "f7_laplace_alpha": 4.0,
    })

    assert baseline == {
        "name": "jyp_f5_selective_no_raw",
        "f3_aa_min_support": 3,
        "f5_output_rare_class_count": 10,
    }

    f7_baseline = build_f7_baseline_config({
        "name": "jyp_f8",
        "f3_aa_min_support": 3,
        "f5_output_rare_class_count": 10,
        "f7_pairs": [["KIRC", "KIPAN"]],
        "f7_laplace_alpha": 4.0,
    })
    assert f7_baseline == {
        "name": "jyp_f7",
        "f3_aa_min_support": 3,
        "f7_pairs": [["KIRC", "KIPAN"]],
        "f7_laplace_alpha": 4.0,
    }


def test_jyp_f8_repeated_validation_psi_detects_binary_shift() -> None:
    psi = _population_stability_index(
        np.asarray([0.0, 0.0, 1.0, 1.0]),
        np.asarray([0.0, 0.0, 0.0, 0.0]),
    )

    assert psi > 0.0


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_jyp_f8_repeated_validation_psi_rejects_non_finite_values(
    invalid: float,
) -> None:
    with pytest.raises(ValueError, match="유한한 값"):
        _population_stability_index(
            np.asarray([0.0, invalid]),
            np.asarray([0.0, 1.0]),
        )


def test_jyp_pipeline_fit_and_pickle_round_trip() -> None:
    features = pd.DataFrame({
        "GENE_A": ["WT", "R1Q", None, "WT"],
        "GENE_B": ["WT", "WT", "Q2*", "A3V B4V"],
    })
    labels = pd.Series(["A", "B", "A", "B"])
    pipeline = create_preprocessing_pipeline({
        "name": "jyp_f0_no_raw",
        "show_progress": False,
    })

    assert pipeline.fit(features, labels) is pipeline
    expected = pipeline.transform(features)
    restored = pickle.loads(pickle.dumps(pipeline))
    actual = restored.transform(features)

    assert actual.shape == expected.shape
    assert (actual != expected).nnz == 0
    assert restored.get_feature_names_out().tolist() == (
        pipeline.get_feature_names_out().tolist()
    )


def test_stability_validator_accepts_standalone_config_without_reference() -> None:
    config = {
        "model": {
            "name": "xgboost",
            "n_estimators": 1,
            "learning_rate": 0.1,
            "max_depth": 1,
            "n_jobs": 1,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
        },
        "preprocessing": {"name": "baseline"},
        "stability_validation": {
            "seeds": [101, 2027, 7301],
            "n_splits": 5,
            "metric": "macro_f1",
        },
    }

    _, _, validation_config, seeds, n_splits = _validate_config(config)

    assert validation_config.get("reference_results") is None
    assert validation_config.get("stability_reference") is None
    assert seeds == (101, 2027, 7301)
    assert n_splits == 5


@pytest.mark.parametrize("threshold", [True, -0.01, float("inf"), "invalid"])
def test_stability_validator_rejects_invalid_overfit_threshold(
    threshold: object,
) -> None:
    config = {
        "model": {"name": "xgboost", "n_estimators": 1},
        "preprocessing": {"name": "baseline"},
        "stability_validation": {
            "seeds": [42],
            "n_splits": 2,
            "overfit_threshold": threshold,
        },
    }

    with pytest.raises(ValueError, match="overfit_threshold"):
        _validate_config(config)


def test_stability_validator_runs_standalone_without_promotion_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = pd.DataFrame({
        "ID": [f"SAMPLE_{index}" for index in range(8)],
        "SUBCLASS": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "GENE_A": ["WT", "R1Q", "WT", "A2V", "WT", "Q3*", "WT", "R1Q"],
        "GENE_B": ["WT", "WT", "A4V", "WT", "R5Q", "WT", "A4V", "WT"],
    })
    train.to_csv(tmp_path / "train.csv", index=False)
    processed_dir = tmp_path / "processed"
    config = {
        "project": {"experiment_name": "standalone_3seed_test"},
        "data": {
            "raw_dir": ".",
            "processed_dir": "processed",
            "train_file": "train.csv",
            "target_column": "SUBCLASS",
            "id_column": "ID",
        },
        "model": {
            "name": "xgboost",
            "n_estimators": 1,
            "learning_rate": 0.1,
            "max_depth": 1,
            "n_jobs": 1,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
        },
        "preprocessing": {"name": "baseline"},
        "stability_validation": {
            "seeds": [11],
            "n_splits": 2,
            "metric": "macro_f1",
            "show_progress": False,
            "resume": True,
        },
    }
    monkeypatch.setattr(
        "src.validate_preprocessing_stability._implementation_identity",
        lambda **_: ({}, {}),
    )

    result = run_stability_validation(
        config,
        project_root=tmp_path,
        quiet=True,
    )

    assert result["status"] == "complete"
    assert result["reference_results"] == {}
    assert result["reference_comparisons"] == {}
    assert result["stability_rule"] == {}
    assert result["summary"]["stability_reference"] is None
    assert result["summary"]["promotion_passed"] is None
    assert result["summary"]["decision"] == "standalone_validation"
    assert len(result["fold_results"]) == 2
    for row in result["fold_results"]:
        _assert_overfit_row(row)
    _assert_overfit_summary(result["summary"], result["fold_results"])

    result_path = processed_dir / "standalone_3seed_test.json"
    fold_path = tmp_path / result["artifacts"]["fold_csv"]
    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    persisted_fold_rows = pd.read_csv(fold_path).to_dict("records")
    assert result_path.is_file()
    assert OVERFIT_SUMMARY_FIELDS <= persisted["summary"].keys()
    assert OVERFIT_ROW_FIELDS <= persisted["fold_results"][0].keys()
    assert OVERFIT_ROW_FIELDS <= persisted_fold_rows[0].keys()


def test_xgb_blend_aligns_probability_columns_to_original_labels() -> None:
    class ReversedModel:
        classes_ = np.asarray([1, 0])

    class Decoder:
        def decode_labels(self, labels):
            return np.asarray(["A", "B"])[np.asarray(labels, dtype=int)]

    aligned = _align_probability_columns(
        np.asarray([[0.3, 0.7]]),
        model=ReversedModel(),
        preprocessor=Decoder(),
        global_class_labels=np.asarray(["A", "B"]),
        context="test",
    )

    np.testing.assert_allclose(aligned, np.asarray([[0.7, 0.3]]))
    with pytest.raises(RuntimeError, match="SUBCLASS 집합"):
        _align_probability_columns(
            np.asarray([[1.0]]),
            model=type("OneClassModel", (), {"classes_": np.asarray([0])})(),
            preprocessor=Decoder(),
            global_class_labels=np.asarray(["A", "B"]),
            context="missing class",
        )


def test_xgb_blend_applies_configured_convex_weights() -> None:
    combined = _blend_probabilities(
        {"jyp_f9": 0.6, "em_v24": 0.4},
        {
            "jyp_f9": np.asarray([[0.9, 0.1], [0.2, 0.8]]),
            "em_v24": np.asarray([[0.7, 0.3], [0.4, 0.6]]),
        },
        context="test blend",
    )

    np.testing.assert_allclose(
        combined,
        np.asarray([[0.82, 0.18], [0.28, 0.72]]),
        rtol=0.0,
        atol=1e-7,
    )


@pytest.mark.parametrize("stem", ["../escape", "nested/output", "nested\\output", "bad name"])
def test_xgb_blend_rejects_unsafe_output_stems(stem: str) -> None:
    with pytest.raises(ValueError, match="단일 파일명"):
        _safe_output_stem(stem)


def test_xgb_blend_contract_ignores_resume_and_progress_flags() -> None:
    base = {
        "project": {"experiment_name": "blend"},
        "data": {"train_file": "train.csv"},
        "model": {"name": "xgboost"},
        "xgb_blend_validation": {
            "seeds": [42],
            "n_splits": 5,
            "resume": True,
            "show_progress": True,
        },
    }
    changed = {
        **base,
        "xgb_blend_validation": {
            **base["xgb_blend_validation"],
            "resume": False,
            "show_progress": False,
        },
    }

    first = _xgb_blend_contract_hash(
        base,
        data_hashes={},
        source_hashes={},
        runtime_versions={},
    )
    second = _xgb_blend_contract_hash(
        changed,
        data_hashes={},
        source_hashes={},
        runtime_versions={},
    )

    assert first == second


@pytest.mark.parametrize(
    "weights",
    [
        {"left": 0.8, "right": 0.3},
        {"left": 1.0, "right": 0.0},
        {"left": 1.1, "right": -0.1},
    ],
)
def test_xgb_blend_rejects_invalid_candidate_weights(
    weights: dict[str, float],
) -> None:
    config = {
        "model": {"name": "xgboost", "n_estimators": 1},
        "xgb_blend_validation": {
            "seeds": [42],
            "n_splits": 2,
            "members": {
                "left": {"name": "baseline"},
                "right": {"name": "baseline"},
            },
            "baseline_member": "left",
            "candidates": {"candidate": {"weights": weights}},
        },
    }

    with pytest.raises(ValueError, match="가중치"):
        _validate_xgb_blend_config(config)


@pytest.mark.parametrize("threshold", [True, -0.01, float("inf"), "invalid"])
def test_xgb_blend_rejects_invalid_overfit_threshold(threshold: object) -> None:
    config = {
        "model": {"name": "xgboost", "n_estimators": 1},
        "xgb_blend_validation": {
            "seeds": [42],
            "n_splits": 2,
            "overfit_threshold": threshold,
            "members": {
                "left": {"name": "baseline"},
                "right": {"name": "baseline"},
            },
            "baseline_member": "left",
            "candidates": {
                "candidate": {"weights": {"left": 0.5, "right": 0.5}}
            },
        },
    }

    with pytest.raises(ValueError, match="overfit_threshold"):
        _validate_xgb_blend_config(config)


def test_xgb_blend_fits_each_member_once_per_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = pd.DataFrame({
        "ID": [f"TRAIN_{index}" for index in range(8)],
        "SUBCLASS": ["A"] * 4 + ["B"] * 4,
        "signal": [0.0] * 4 + [1.0] * 4,
    })
    test = pd.DataFrame({
        "ID": ["TEST_0", "TEST_1"],
        "signal": [0.0, 1.0],
    })
    submission = pd.DataFrame({
        "ID": test["ID"],
        "SUBCLASS": ["A", "A"],
    })
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    submission.to_csv(tmp_path / "sample_submission.csv", index=False)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    stale_submission = processed_dir / "blend_smoke_old_submission.csv"
    stale_submission.write_text("stale", encoding="utf-8")

    fit_calls: list[str] = []

    class RecordingPipeline:
        def __init__(self, name: str) -> None:
            self.name = name
            self.encoder = LabelEncoder()

        def fit_transform(self, frame: pd.DataFrame, labels: pd.Series):
            fit_calls.append(self.name)
            self.encoder.fit(labels)
            return frame[["signal"]].astype("float32")

        def transform(self, frame: pd.DataFrame):
            return frame[["signal"]].astype("float32")

        def encode_labels(self, labels: pd.Series):
            return self.encoder.transform(labels)

        def decode_labels(self, labels):
            return self.encoder.inverse_transform(np.asarray(labels, dtype=int))

    class FixedModel:
        def fit(self, features, labels):
            self.classes_ = np.unique(labels)
            return self

        def predict_proba(self, features):
            signal = np.asarray(features)[:, 0]
            right = np.where(signal > 0.5, 0.8, 0.2)
            return np.column_stack([1.0 - right, right])

    def create_recording_pipeline(config: dict[str, object]):
        return RecordingPipeline(str(config["name"]))

    monkeypatch.setattr(
        stability_validation,
        "create_preprocessing_pipeline",
        create_recording_pipeline,
    )
    monkeypatch.setitem(
        stability_validation.MODEL_BUILDERS,
        "xgboost",
        lambda _config, _seed: FixedModel(),
    )
    monkeypatch.setattr(
        stability_validation,
        "_blend_implementation_identity",
        lambda **_: ({}, {}),
    )

    members = {
        "jyp_f9": {"name": "jyp_f9"},
        "em_v24": {"name": "em_v24"},
        "em_v20": {"name": "em_v20"},
        "em_v19": {"name": "em_v19"},
    }
    config = {
        "project": {"experiment_name": "blend_smoke"},
        "data": {
            "raw_dir": ".",
            "processed_dir": "processed",
            "train_file": "train.csv",
            "test_file": "test.csv",
            "submission_file": "sample_submission.csv",
            "target_column": "SUBCLASS",
            "id_column": "ID",
        },
        "model": {"name": "xgboost", "n_estimators": 1},
        "xgb_blend_validation": {
            "seeds": [42],
            "n_splits": 2,
            "show_progress": False,
            "resume": True,
            "submission_policy": "screening_only",
            "baseline_member": "jyp_f9",
            "members": members,
            "candidates": {
                "jypF9_EM24": {"weights": {"jyp_f9": 0.6, "em_v24": 0.4}},
                "jypF9_EM20": {"weights": {"jyp_f9": 0.65, "em_v20": 0.35}},
                "jypF9_EM19": {"weights": {"jyp_f9": 0.65, "em_v19": 0.35}},
            },
        },
    }

    result = run_xgboost_blend_validation(
        config,
        project_root=tmp_path,
        force=True,
        quiet=True,
    )

    assert result["training_count"] == 8
    assert {name: fit_calls.count(name) for name in members} == {
        name: 2 for name in members
    }
    assert result["summary"]["baseline_mean_seed_oof_macro_f1"] == 1.0
    assert set(result["summary"]["candidates"]) == {
        "jypF9_EM24",
        "jypF9_EM20",
        "jypF9_EM19",
    }
    scores_path = tmp_path / result["artifacts"]["scores_csv"]
    score_rows = pd.read_csv(scores_path).to_dict("records")
    assert score_rows
    for row in score_rows:
        _assert_overfit_row(row)
    rows_by_name = {
        name: [row for row in score_rows if row["name"] == name]
        for name in result["summary"]["candidates"]
    }
    for name, summary in result["summary"]["candidates"].items():
        _assert_overfit_summary(summary, rows_by_name[name])

    result_path = tmp_path / result["artifacts"]["result_json"]
    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    assert OVERFIT_SUMMARY_FIELDS <= persisted["summary"]["candidates"][
        "jypF9_EM24"
    ].keys()
    assert OVERFIT_ROW_FIELDS <= set(pd.read_csv(scores_path).columns)
    assert (tmp_path / "processed" / "blend_smoke_probabilities.npz").is_file()
    assert not list((tmp_path / "processed").glob("*_submission.csv"))

    completed_fit_count = len(fit_calls)
    repeated = run_xgboost_blend_validation(
        config,
        project_root=tmp_path,
        quiet=True,
    )
    assert repeated["status"] == "complete"
    assert len(fit_calls) == completed_fit_count


def _internal_fusion_config() -> dict[str, object]:
    return {
        "project": {"experiment_name": "internal_fusion_smoke"},
        "data": {
            "raw_dir": ".",
            "processed_dir": "processed",
            "train_file": "train.csv",
            "test_file": "test.csv",
            "submission_file": "sample_submission.csv",
            "target_column": "SUBCLASS",
            "id_column": "ID",
        },
        "model": {"name": "xgboost", "n_estimators": 1},
        "xgb_internal_fusion_validation": {
            "seeds": [42],
            "n_splits": 2,
            "show_progress": False,
            "resume": True,
            "submission_policy": "screening_only",
            "baseline_candidate": "f9",
            "f9_preprocessing": {"name": "jyp_f9"},
            "em24_preprocessing": {"name": "em_v24"},
            "candidates": {
                "f9": {"em24_blocks": []},
                "summary": {"em24_blocks": ["summary"]},
                "dual_signature": {
                    "em24_blocks": ["signature_all", "signature_functional"]
                },
                "both": {
                    "em24_blocks": [
                        "summary",
                        "signature_all",
                        "signature_functional",
                    ]
                },
            },
        },
    }


def test_internal_fusion_selects_approved_em24_blocks_in_canonical_order() -> None:
    columns = [
        "signature_functional_B_mean",
        "unused_gene_severity",
        *reversed(EM24_INTERNAL_SUMMARY_COLUMNS),
        "signature_all_B_sum",
        "signature_all_A_sum",
        "signature_functional_A_mean",
    ]
    frame = pd.DataFrame(np.zeros((2, len(columns))), columns=columns)

    selected = _select_em24_internal_columns(
        frame,
        ["signature_functional", "summary", "signature_all"],
    )

    assert selected[:16] == list(EM24_INTERNAL_SUMMARY_COLUMNS)
    assert selected[16:18] == ["signature_all_B_sum", "signature_all_A_sum"]
    assert selected[18:] == [
        "signature_functional_B_mean",
        "signature_functional_A_mean",
    ]


def test_internal_fusion_preserves_f9_csr_prefix_shape_dtype_and_names() -> None:
    f9 = sparse.csr_matrix(
        np.asarray([[0.0, 2.0], [3.0, 0.0]], dtype=np.float64)
    )
    em24 = pd.DataFrame(
        {
            **{
                name: np.asarray([index, index + 1], dtype=np.float64)
                for index, name in enumerate(EM24_INTERNAL_SUMMARY_COLUMNS)
            },
            "signature_all_A_sum": [0.25, 0.5],
            "signature_all_B_sum": [0.75, 1.0],
            "signature_functional_A_mean": [1.25, 1.5],
            "signature_functional_B_mean": [1.75, 2.0],
        }
    )

    combined, suffix_names = _build_internal_fusion_matrix(
        f9,
        em24,
        ["summary", "signature_all", "signature_functional"],
    )

    assert sparse.isspmatrix_csr(combined)
    assert combined.dtype == np.float32
    assert combined.shape == (2, 22)
    np.testing.assert_array_equal(combined[:, :2].toarray(), f9.toarray())
    assert len(suffix_names) == 20
    assert suffix_names == [
        *(f"EM24__{name}" for name in EM24_INTERNAL_SUMMARY_COLUMNS),
        "EM24__signature_all_A_sum",
        "EM24__signature_all_B_sum",
        "EM24__signature_functional_A_mean",
        "EM24__signature_functional_B_mean",
    ]


def test_internal_fusion_selects_and_names_em16_blocks() -> None:
    columns = [
        "GENE_A",
        "GENE_B",
        *EM16_INTERNAL_SUMMARY_COLUMNS,
        "signature_A_weighted",
        "signature_A_match_count",
        "hotspot_0000",
    ]
    em16 = pd.DataFrame(np.ones((2, len(columns))), columns=columns)
    selected = _select_internal_companion_columns(
        em16,
        ["hotspot", "signature", "summary", "severity"],
        companion_name="em_v16",
    )

    assert selected == columns
    combined, suffix_names = _build_internal_fusion_matrix(
        sparse.csr_matrix(np.zeros((2, 1), dtype=np.float32)),
        em16,
        ["severity", "signature"],
        companion_name="em_v16",
    )
    assert combined.shape == (2, 5)
    assert suffix_names == [
        "EM16__GENE_A",
        "EM16__GENE_B",
        "EM16__signature_A_weighted",
        "EM16__signature_A_match_count",
    ]


def test_internal_fusion_rejects_unsupported_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _internal_fusion_config()
    config["xgb_internal_fusion_validation"]["candidates"]["both"] = {
        "em24_blocks": ["severity"]
    }
    monkeypatch.setattr(
        stability_validation,
        "create_preprocessing_pipeline",
        lambda _config: object(),
    )
    monkeypatch.setitem(
        stability_validation.MODEL_BUILDERS,
        "xgboost",
        lambda _config, _seed: object(),
    )

    with pytest.raises(ValueError, match="지원하지 않는 블록"):
        _validate_xgb_internal_fusion_config(config)


def test_internal_fusion_rejects_baseline_only_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _internal_fusion_config()
    config["xgb_internal_fusion_validation"]["candidates"] = {
        "f9": {"em24_blocks": []}
    }
    monkeypatch.setattr(
        stability_validation,
        "create_preprocessing_pipeline",
        lambda _config: object(),
    )
    monkeypatch.setitem(
        stability_validation.MODEL_BUILDERS,
        "xgboost",
        lambda _config, _seed: object(),
    )

    with pytest.raises(ValueError, match="비교 후보"):
        _validate_xgb_internal_fusion_config(config)


@pytest.mark.parametrize("threshold", [True, -0.01, float("inf"), "invalid"])
def test_internal_fusion_rejects_invalid_overfit_threshold(
    threshold: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _internal_fusion_config()
    config["xgb_internal_fusion_validation"]["overfit_threshold"] = threshold
    monkeypatch.setattr(
        stability_validation,
        "create_preprocessing_pipeline",
        lambda _config: object(),
    )
    monkeypatch.setitem(
        stability_validation.MODEL_BUILDERS,
        "xgboost",
        lambda _config, _seed: object(),
    )

    with pytest.raises(ValueError, match="overfit_threshold"):
        _validate_xgb_internal_fusion_config(config)


def test_internal_fusion_contract_ignores_resume_and_progress_flags() -> None:
    base = _internal_fusion_config()
    changed = {
        **base,
        "xgb_internal_fusion_validation": {
            **base["xgb_internal_fusion_validation"],
            "resume": False,
            "show_progress": True,
        },
    }

    first = _xgb_internal_fusion_contract_hash(
        base,
        data_hashes={},
        source_hashes={},
        runtime_versions={},
    )
    second = _xgb_internal_fusion_contract_hash(
        changed,
        data_hashes={},
        source_hashes={},
        runtime_versions={},
    )

    assert first == second


def test_internal_fusion_reuses_preprocessing_per_fold_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = pd.DataFrame({
        "ID": [f"TRAIN_{index}" for index in range(8)],
        "SUBCLASS": ["A"] * 4 + ["B"] * 4,
        "signal": [0.0] * 4 + [1.0] * 4,
    })
    test = pd.DataFrame({
        "ID": ["TEST_0", "TEST_1"],
        "signal": [0.0, 1.0],
    })
    submission = pd.DataFrame({"ID": test["ID"], "SUBCLASS": ["A", "A"]})
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    submission.to_csv(tmp_path / "sample_submission.csv", index=False)

    fit_calls: list[str] = []
    model_feature_counts: list[int] = []

    class RecordingPipeline:
        def __init__(self, name: str) -> None:
            self.name = name
            self.label_encoder = LabelEncoder()

        def fit_transform(self, frame: pd.DataFrame, labels: pd.Series):
            fit_calls.append(self.name)
            self.label_encoder.fit(labels)
            return self.transform(frame)

        def transform(self, frame: pd.DataFrame):
            if self.name == "jyp_f9":
                return sparse.csr_matrix(
                    frame[["signal"]].to_numpy(dtype=np.float32)
                )
            values: dict[str, np.ndarray] = {
                name: np.zeros(len(frame), dtype=np.float32)
                for name in EM24_INTERNAL_SUMMARY_COLUMNS
            }
            signal = frame["signal"].to_numpy(dtype=np.float32)
            values.update({
                "signature_all_A_sum": 1.0 - signal,
                "signature_all_B_sum": signal,
                "signature_functional_A_mean": 1.0 - signal,
                "signature_functional_B_mean": signal,
                "unused_gene_severity": signal,
            })
            return pd.DataFrame(values, index=frame.index)

        def get_feature_names_out(self):
            return np.asarray(["F9__signal"], dtype=object)

        def encode_labels(self, labels: pd.Series):
            return self.label_encoder.transform(labels)

        def decode_labels(self, labels):
            return self.label_encoder.inverse_transform(np.asarray(labels, dtype=int))

    class RecordingModel:
        def fit(self, features, labels):
            self.classes_ = np.unique(labels)
            model_feature_counts.append(features.shape[1])
            return self

        def predict_proba(self, features):
            signal = sparse.csr_matrix(features)[:, 0].toarray().ravel()
            right = np.where(signal > 0.5, 0.8, 0.2)
            return np.column_stack([1.0 - right, right])

    monkeypatch.setattr(
        stability_validation,
        "create_preprocessing_pipeline",
        lambda config: RecordingPipeline(str(config["name"])),
    )
    monkeypatch.setitem(
        stability_validation.MODEL_BUILDERS,
        "xgboost",
        lambda _config, _seed: RecordingModel(),
    )
    monkeypatch.setattr(
        stability_validation,
        "_blend_implementation_identity",
        lambda **_: ({}, {}),
    )

    config = _internal_fusion_config()
    result = run_xgboost_internal_fusion_validation(
        config,
        project_root=tmp_path,
        force=True,
        quiet=True,
    )

    assert result["training_count"] == 8
    assert result["preprocessing_fit_count"] == 4
    assert fit_calls.count("jyp_f9") == 2
    assert fit_calls.count("em_v24") == 2
    assert model_feature_counts == [1, 17, 5, 21] * 2
    assert result["summary"]["baseline_mean_seed_oof_macro_f1"] == 1.0
    assert result["summary"]["best_candidate"] == "f9"
    assert result["summary"]["best_nonbaseline_candidate"] == "summary"
    assert result["summary"]["promotion_candidate"] is None
    scores_path = tmp_path / result["artifacts"]["scores_csv"]
    score_rows = pd.read_csv(scores_path).to_dict("records")
    assert score_rows
    for row in score_rows:
        _assert_overfit_row(row)
    rows_by_name = {
        name: [row for row in score_rows if row["name"] == name]
        for name in result["summary"]["candidates"]
    }
    for name, summary in result["summary"]["candidates"].items():
        _assert_overfit_summary(summary, rows_by_name[name])

    result_path = tmp_path / result["artifacts"]["result_json"]
    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    assert OVERFIT_SUMMARY_FIELDS <= persisted["summary"]["candidates"][
        "summary"
    ].keys()
    assert OVERFIT_ROW_FIELDS <= set(pd.read_csv(scores_path).columns)
    assert not list((tmp_path / "processed").glob("*_submission.csv"))

    completed_fit_calls = len(fit_calls)
    completed_model_fits = len(model_feature_counts)
    repeated = run_xgboost_internal_fusion_validation(
        config,
        project_root=tmp_path,
        quiet=True,
    )
    assert repeated["status"] == "complete"
    assert len(fit_calls) == completed_fit_calls
    assert len(model_feature_counts) == completed_model_fits
