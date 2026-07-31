import pandas as pd

from src.models.xgboost_model import create_model
from src.pipelines.base import PreprocessingPipeline
from src.pipelines.preprocessing_registry import create_preprocessing_pipeline


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


def test_pipeline_factory_creates_em_v1() -> None:
    pipeline = create_preprocessing_pipeline({"name": "em_v1"})

    assert pipeline.name == "em_v1"
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
