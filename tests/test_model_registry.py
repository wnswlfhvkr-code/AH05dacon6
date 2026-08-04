from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import yaml

from src.models import MODEL_BUILDERS
from src.models.tabpfn3_model import TabPFN3Classifier
from src.pipelines.pipeline_jsj_v1 import (
    JSJV1PreprocessingPipeline,
    TextTreeFeatureBundle,
)
from src.pipelines.pipeline_jsj_v2 import JSJV2PreprocessingPipeline
from src.pipelines.preprocessing_registry import PIPELINES


def test_new_models_and_pipeline_are_registered() -> None:
    assert {
        "xgboost",
        "lightgbm",
        "linear_svc",
        "wc_tfidf_lsvc_lgbm",
        "oncobert",
        "muat",
        "mutation_projector",
        "tabpfn3",
    } <= set(MODEL_BUILDERS)
    assert {"jsj_v1", "jsj_v2"} <= set(PIPELINES)


def test_jsj_test_config_references_registered_components() -> None:
    root = Path(__file__).parents[1]
    with (root / "configs" / "test_004.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    assert config["model"]["name"] in MODEL_BUILDERS
    assert config["preprocessing"]["name"] in PIPELINES


def test_tabpfn3_config_references_registered_components() -> None:
    root = Path(__file__).parents[1]
    with (root / "configs" / "test_006_m9.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    assert config["model"]["name"] == "tabpfn3"
    assert config["model"]["name"] in MODEL_BUILDERS
    assert config["model"]["model_version"] == "v3"
    assert config["preprocessing"]["name"] in PIPELINES


def test_tabpfn3_reduces_features_inside_model_fit(monkeypatch) -> None:
    class FakeTabPFN:
        def fit(self, features, labels):
            self.fit_shape = features.shape
            self.classes_ = np.unique(labels)
            return self

        def predict(self, features):
            return np.repeat(self.classes_[0], len(features))

        def predict_proba(self, features):
            return np.full(
                (len(features), len(self.classes_)),
                1.0 / len(self.classes_),
            )

    features = pd.DataFrame(
        {
            "TP53": [1, 1, 0, 0, 1, 0],
            "BRAF": [0, 0, 1, 1, 0, 1],
            "EGFR": [0, 1, 0, 1, 0, 1],
            "KRAS": [1, 0, 1, 0, 1, 0],
        },
        dtype="float32",
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    fake = FakeTabPFN()
    model = TabPFN3Classifier(
        {"reduction_method": "chi2", "max_features": 2},
        seed=42,
    )
    monkeypatch.setattr(model, "_create_tabpfn", lambda: fake)

    model.fit(features, labels)

    assert fake.fit_shape == (6, 2)
    assert model.summary()["removed_features"] == 2
    assert model.predict(features).shape == labels.shape
    assert model.predict_proba(features).shape == (6, 3)


def test_jsj_v2_pipeline_creates_compact_numeric_features() -> None:
    features = pd.DataFrame(
        {
            "TP53": ["R175H", "WT", "R248Q R273H", "WT"],
            "BRAF": ["WT", "V600E", "WT", "V600E"],
            "EGFR": ["WT", "WT", "L858R", None],
        }
    )
    labels = pd.Series(["A", "B", "A", "B"])
    pipeline = JSJV2PreprocessingPipeline()

    transformed = pipeline.fit_transform(features, labels)
    assert transformed.shape[0] == len(features)
    assert 0 < transformed.shape[1] < 40
    assert transformed.select_dtypes(exclude="number").empty
    assert "total_event_count" in transformed.columns
    assert "multi_event_gene_count" in transformed.columns
    assert np.isfinite(transformed.to_numpy()).all()


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
