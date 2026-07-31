from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import yaml

from src.models import MODEL_BUILDERS
from src.pipelines.pipeline_jsj_v1 import (
    JSJV1PreprocessingPipeline,
    TextTreeFeatureBundle,
)
from src.pipelines.preprocessing_registry import PIPELINES


def test_new_models_and_pipeline_are_registered() -> None:
    assert {"xgboost", "lightgbm", "linear_svc", "wc_tfidf_lsvc_lgbm"} <= set(
        MODEL_BUILDERS
    )
    assert "jsj_v1" in PIPELINES


def test_shared_test_config_references_registered_components() -> None:
    root = Path(__file__).parents[1]
    with (root / "configs" / "test_001.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    assert config["model"]["name"] in MODEL_BUILDERS
    assert config["preprocessing"]["name"] in PIPELINES


def test_wc_tfidf_pipeline_splits_train_and_transform_without_refitting() -> None:
    features = pd.DataFrame(
        {
            "TP53": ["R175H", "WT", "R248Q R273H", "WT"],
            "BRAF": ["WT", "V600E", "WT", "V600E"],
            "EGFR": ["WT", "WT", "L858R", None],
        }
    )
    labels = pd.Series(["A", "B", "A", "B"])
    pipeline = JSJV1PreprocessingPipeline(
        word_min_df=1,
        word_max_features=100,
        char_min_df=1,
        char_max_features=100,
        split_multi_event=True,
        return_bundle=True,
    )

    transformed = pipeline.fit_transform(features, labels)
    assert isinstance(transformed, TextTreeFeatureBundle)
    assert transformed.text.shape[0] == len(features)
    assert transformed.tree.shape[0] == len(features)
    assert transformed.text.shape[1] > 0
    assert transformed.tree.shape[1] >= 4
    assert np.isfinite(transformed.tree.data).all()

    second = pipeline.transform(features.iloc[:2])
    assert second.text.shape[1] == transformed.text.shape[1]
    assert second.tree.shape[1] == transformed.tree.shape[1]


def test_lightgbm_and_linear_svc_fit_and_predict() -> None:
    features = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.1, 0.9],
        ]
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2])

    linear = MODEL_BUILDERS["linear_svc"](
        {"C": 0.2, "class_weight": "balanced", "max_iter": 1000},
        42,
    )
    linear.fit(csr_matrix(features), labels)
    assert linear.predict(csr_matrix(features)).shape == labels.shape

    lightgbm = MODEL_BUILDERS["lightgbm"](
        {
            "n_estimators": 5,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 1,
            "n_jobs": 1,
            "verbosity": -1,
        },
        42,
    )
    lightgbm.fit(features, labels)
    assert lightgbm.predict(features).shape == labels.shape


def test_wc_tfidf_ensemble_fits_feature_bundle() -> None:
    text = csr_matrix(
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.1, 0.9],
            ]
        )
    )
    tree = text.copy()
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    bundle = TextTreeFeatureBundle(text=text, tree=tree)
    model = MODEL_BUILDERS["wc_tfidf_lsvc_lgbm"](
        {
            "linear_weight": 0.95,
            "tree_weight": 0.05,
            "temperature": 0.5,
            "linear": {"C": 0.2, "max_iter": 1000},
            "tree": {
                "n_estimators": 5,
                "learning_rate": 0.1,
                "num_leaves": 7,
                "min_child_samples": 1,
                "n_jobs": 1,
                "verbosity": -1,
            },
        },
        42,
    )

    model.fit(bundle, labels)
    probabilities = model.predict_proba(bundle)
    assert probabilities.shape == (len(labels), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert model.predict(bundle).shape == labels.shape
