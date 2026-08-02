import pickle

import pandas as pd
import pytest

from src.models.xgboost_model import create_model
from src.pipelines.base import PreprocessingPipeline
from src.pipelines.preprocessing_registry import create_preprocessing_pipeline


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
    "jyp_f6",
    "jyp_f7",
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
