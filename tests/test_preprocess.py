import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

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
    _validate_config,
    run_stability_validation,
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

    model.fit(transformed, preprocessor.encode_labels(labels))

    assert len(model.predict(transformed)) == len(features)


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
    assert (processed_dir / "standalone_3seed_test.json").is_file()
