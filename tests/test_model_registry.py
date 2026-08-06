from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
import yaml

from src.models import MODEL_BUILDERS
from src.pipelines.pipeline_jsj_v1 import (
    JSJV1PreprocessingPipeline,
    TextTreeFeatureBundle,
)
from src.pipelines.pipeline_jsj_v2 import JSJV2PreprocessingPipeline
from src.pipelines.preprocessing_registry import PIPELINES
from src.train import used_tree_count


def test_new_models_and_pipeline_are_registered() -> None:
    assert {
        "xgboost",
        "lightgbm",
        "catboost",
        "torch_linear",
        "logistic_regression_gpu",
        "torch_mlp",
        "tabpfn",
        "tabicl",
        "tabicl_collision_expert",
        "tabfm_collision_expert",
        "tabm",
        "modernnca",
        "modernnca_collision_expert",
        "realmlp",
        "realtabr_collision_expert",
        "xrfm",
        "linear_svc",
        "wc_tfidf_lsvc_lgbm",
        "extra_trees",
        "balanced_random_forest",
        "random_forest",
    } <= set(MODEL_BUILDERS)
    assert {"jsj_v1", "jsj_v2"} <= set(PIPELINES)


def test_jsj_test_config_references_registered_components() -> None:
    root = Path(__file__).parents[1]
    with (root / "configs" / "test_004.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    assert config["model"]["name"] in MODEL_BUILDERS
    assert config["preprocessing"]["name"] in PIPELINES


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

    cpu_lightgbm = MODEL_BUILDERS["lightgbm"](
        {"n_estimators": 2, "device_type": "cpu", "verbosity": -1},
        42,
    )
    assert cpu_lightgbm.get_params()["device_type"] == "cpu"


@pytest.mark.parametrize(
    ("model_name", "expected_class_weight"),
    [
        ("extra_trees", "balanced"),
        ("balanced_random_forest", None),
        ("random_forest", "balanced_subsample"),
    ],
)
def test_cpu_forest_builders_fit_sparse_multiclass_data(
    model_name: str,
    expected_class_weight: str | None,
) -> None:
    features = csr_matrix(np.eye(9, dtype=np.float32))
    labels = np.repeat(np.arange(3), 3)
    model = MODEL_BUILDERS[model_name](
        {
            "name": model_name,
            "n_estimators": 5,
            "max_depth": 3,
            "min_samples_leaf": 1,
            "n_jobs": 1,
        },
        7301,
    )

    model.fit(features, labels)
    assert model.predict(features).shape == labels.shape
    assert model.predict_proba(features).shape == (len(labels), 3)
    assert model.get_params()["random_state"] == 7301
    assert model.get_params()["class_weight"] == expected_class_weight


def test_catboost_builder_enforces_gpu_defaults() -> None:
    model = MODEL_BUILDERS["catboost"](
        {"name": "catboost", "n_estimators": 5, "depth": 3},
        7301,
    )

    parameters = model.get_params()
    assert parameters["task_type"] == "GPU"
    assert parameters["devices"] == "0"
    assert parameters["random_seed"] == 7301
    assert parameters["allow_writing_files"] is False
    assert parameters["auto_class_weights"] == "Balanced"
    assert parameters["verbose"] is False


def test_catboost_builder_rejects_cpu_task_type() -> None:
    with pytest.raises(ValueError, match="task_type=GPU"):
        MODEL_BUILDERS["catboost"]({"task_type": "CPU"}, 42)


@pytest.mark.parametrize(
    "model_name",
    ["torch_linear", "logistic_regression_gpu", "torch_mlp"],
)
def test_torch_builders_enforce_cuda(model_name: str) -> None:
    with pytest.raises(ValueError, match="device=cuda"):
        MODEL_BUILDERS[model_name]({"device": "cpu"}, 42)
    model = MODEL_BUILDERS[model_name]({"device": "cuda"}, 42)
    with pytest.raises(ValueError, match="device=cuda"):
        model.set_params(device="cpu")


def test_tabpfn_builder_enforces_cuda_and_exposes_non_tree_params() -> None:
    with pytest.raises(ValueError, match="expert_device=cuda"):
        MODEL_BUILDERS["tabpfn"](
            {
                "class_names": ["A", "B"],
                "collision_pairs": [["A", "B"]],
                "expert_device": "cpu",
            },
            42,
        )

    model = MODEL_BUILDERS["tabpfn"](
        {
            "class_names": ["A", "B"],
            "collision_pairs": [["A", "B"]],
            "expert_device": "cuda",
            "svd_components": 2,
            "ensemble_size": 3,
        },
        42,
    )
    assert model.get_params()["ensemble_size"] == 3
    assert used_tree_count(model) is None
    with pytest.raises(ValueError, match="expert_device=cuda"):
        model.set_params(expert_device="cpu")


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(
    not _torch_cuda_available(),
    reason="PyTorch GPU integration test requires a CUDA-enabled torch install.",
)
@pytest.mark.parametrize(
    ("model_name", "extra_config"),
    [
        ("torch_linear", {}),
        ("logistic_regression_gpu", {}),
        ("torch_mlp", {"hidden_dims": [8], "dropout": 0.0}),
    ],
)
def test_torch_gpu_models_fit_sparse_features_and_pickle(
    model_name: str,
    extra_config: dict,
) -> None:
    base_features = np.eye(3, dtype=np.float32)
    features = csr_matrix(np.tile(base_features, (10, 1)))
    labels = np.tile(np.arange(3), 10)
    model = MODEL_BUILDERS[model_name](
        {
            "epochs": 2,
            "batch_size": 10,
            "learning_rate": 0.01,
            "device": "cuda",
            **extra_config,
        },
        42,
    )

    model.fit(features, labels)
    probabilities = model.predict_proba(features)

    assert probabilities.shape == (len(labels), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert model.predict(features).shape == labels.shape
    assert used_tree_count(model) is None

    restored = pickle.loads(pickle.dumps(model))
    assert restored.predict(features).shape == labels.shape


def test_tabpfn_adapter_densifies_with_fold_local_svd(monkeypatch) -> None:
    from src.models.tabpfn_model import SVDTabPFNClassifier

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorch:
        cuda = FakeCuda()

    class FakeTabPFNClassifier:
        def __init__(self, **parameters) -> None:
            self.parameters = parameters

        def fit(self, features, labels) -> None:
            assert isinstance(features, np.ndarray)
            assert features.dtype == np.float32
            self.classes_ = np.unique(labels)

        def predict_proba(self, features) -> np.ndarray:
            probabilities = np.full(
                (len(features), len(self.classes_)),
                1.0 / len(self.classes_),
            )
            return probabilities

        def predict(self, features) -> np.ndarray:
            return np.resize(self.classes_, len(features))

    model = SVDTabPFNClassifier(
        svd_components=2,
        ensemble_size=3,
        device="cuda",
        seed=42,
    )
    monkeypatch.setattr(
        model,
        "_import_dependencies",
        lambda: (FakeTorch, FakeTabPFNClassifier),
    )
    features = csr_matrix(np.tile(np.eye(4, dtype=np.float32), (3, 1)))
    labels = np.tile(np.arange(3), 4)

    model.fit(features, labels)
    probabilities = model.predict_proba(features)

    assert model.reducer_.n_components == 2
    assert model.classifier_.parameters["device"] == "cuda"
    assert model.classifier_.parameters["n_estimators"] == 3
    assert model.classifier_.parameters["auto_scale_n_estimators"] is False
    assert model.classifier_.parameters["balance_probabilities"] is True
    assert model.classifier_.parameters["fit_mode"] == "low_memory"
    assert model.classifier_.parameters["keep_cache_on_device"] is False
    assert probabilities.shape == (len(labels), 3)
    assert model.predict(features).shape == labels.shape


def test_tabpfn_base_adapter_rejects_more_than_ten_classes(monkeypatch) -> None:
    from src.models.tabpfn_model import SVDTabPFNClassifier

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorch:
        cuda = FakeCuda()

    model = SVDTabPFNClassifier(device="cuda")
    monkeypatch.setattr(model, "_import_dependencies", lambda: (FakeTorch, object))
    features = csr_matrix(np.eye(11, dtype=np.float32))

    with pytest.raises(ValueError, match="최대 10개 클래스"):
        model.fit(features, np.arange(11))


def test_tabpfn_collision_expert_only_redistributes_active_pair_mass(monkeypatch) -> None:
    from src.models.tabpfn_model import TabPFNCollisionExpertClassifier

    class FakeBaseModel:
        classes_ = np.asarray([0, 1, 2])

        def fit(self, features, labels):
            return self

        def predict_proba(self, features) -> np.ndarray:
            return np.asarray(
                [
                    [0.6, 0.3, 0.1],
                    [0.2, 0.3, 0.5],
                    [0.1, 0.7, 0.2],
                ]
            )[: features.shape[0]]

    class FakeExpert:
        classes_ = np.asarray([0, 1])

        def fit(self, features, labels):
            return self

        def predict_proba(self, features) -> np.ndarray:
            return np.tile(np.asarray([[0.25, 0.75]]), (features.shape[0], 1))

    model = TabPFNCollisionExpertClassifier(
        class_names=("A", "B", "C"),
        collision_pairs=(("A", "B"),),
        expert_device="cuda",
        seed=42,
    )
    monkeypatch.setattr(model, "_build_base_model", lambda: FakeBaseModel())
    monkeypatch.setattr(model, "_build_expert", lambda: FakeExpert())
    features = csr_matrix(np.eye(3, dtype=np.float32))
    labels = np.asarray([0, 1, 2])

    model.fit(features, labels)
    probabilities = model.predict_proba(features)

    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.allclose(probabilities[0], [0.225, 0.675, 0.1])
    assert np.allclose(probabilities[1], [0.2, 0.3, 0.5])
    assert np.allclose(probabilities[2], [0.2, 0.6, 0.2])
    assert model.predict(features).shape == labels.shape


def test_tabfm_collision_expert_enforces_cuda_and_non_tree_metadata() -> None:
    with pytest.raises(ValueError, match="expert_device=cuda"):
        MODEL_BUILDERS["tabfm_collision_expert"](
            {
                "class_names": ["A", "B"],
                "collision_pairs": [["A", "B"]],
                "expert_device": "cpu",
            },
            42,
        )
    model = MODEL_BUILDERS["tabfm_collision_expert"](
        {
            "class_names": ["A", "B"],
            "collision_pairs": [["A", "B"]],
            "expert_device": "cuda",
            "ensemble_size": 2,
        },
        42,
    )
    assert model.get_params()["ensemble_size"] == 2
    assert used_tree_count(model) is None
    with pytest.raises(ValueError, match="expert_device=cuda"):
        model.set_params(expert_device="cpu")


def test_tabfm_collision_expert_preserves_pair_mass_and_non_target_rows(
    monkeypatch,
) -> None:
    from src.models.tabfm_collision_expert_model import (
        TabFMCollisionExpertClassifier,
    )

    class FakeBaseModel:
        classes_ = np.asarray([0, 1, 2])

        def fit(self, features, labels):
            return self

        def predict_proba(self, features) -> np.ndarray:
            return np.asarray(
                [
                    [0.6, 0.3, 0.1],
                    [0.2, 0.3, 0.5],
                    [0.1, 0.7, 0.2],
                ]
            )[: features.shape[0]]

    class FakeExpert:
        classes_ = np.asarray([0, 1])

        def fit(self, features, labels):
            return self

        def predict_proba(self, features) -> np.ndarray:
            return np.tile(np.asarray([[0.25, 0.75]]), (features.shape[0], 1))

    model = TabFMCollisionExpertClassifier(
        class_names=("A", "B", "C"),
        collision_pairs=(("A", "B"),),
        expert_device="cuda",
        seed=42,
    )
    monkeypatch.setattr(model, "_build_base_model", lambda: FakeBaseModel())
    monkeypatch.setattr(model, "_load_foundation_model", lambda: object())
    monkeypatch.setattr(model, "_build_expert", lambda foundation_model: FakeExpert())
    features = csr_matrix(np.eye(3, dtype=np.float32))
    labels = np.asarray([0, 1, 2])

    model.fit(features, labels)
    probabilities = model.predict_proba(features)

    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.allclose(probabilities[0], [0.225, 0.675, 0.1])
    assert np.allclose(probabilities[1], [0.2, 0.3, 0.5])
    assert np.allclose(probabilities[2], [0.2, 0.6, 0.2])
    assert model.predict(features).shape == labels.shape


def test_tabfm_collision_expert_falls_back_on_nonfinite_expert_probabilities(
    monkeypatch,
) -> None:
    from src.models.tabfm_collision_expert_model import (
        TabFMCollisionExpertClassifier,
    )

    class FakeBaseModel:
        classes_ = np.asarray([0, 1, 2])

        def fit(self, features, labels):
            return self

        def predict_proba(self, features) -> np.ndarray:
            return np.tile(np.asarray([[0.6, 0.3, 0.1]]), (features.shape[0], 1))

    class NonFiniteExpert:
        classes_ = np.asarray([0, 1])

        def fit(self, features, labels):
            return self

        def predict_proba(self, features) -> np.ndarray:
            return np.full((features.shape[0], 2), np.nan)

    model = TabFMCollisionExpertClassifier(
        class_names=("A", "B", "C"),
        collision_pairs=(("A", "B"),),
        expert_device="cuda",
        seed=42,
    )
    monkeypatch.setattr(model, "_build_base_model", lambda: FakeBaseModel())
    monkeypatch.setattr(model, "_load_foundation_model", lambda: object())
    monkeypatch.setattr(
        model,
        "_build_expert",
        lambda foundation_model: NonFiniteExpert(),
    )
    features = csr_matrix(np.eye(3, dtype=np.float32))
    labels = np.asarray([0, 1, 2])

    model.fit(features, labels)
    probabilities = model.predict_proba(features)

    assert np.allclose(probabilities, np.tile([0.6, 0.3, 0.1], (3, 1)))
    assert model.invalid_expert_probability_rows_ == 3


def test_tabicl_builders_enforce_cuda_and_non_tree_metadata() -> None:
    with pytest.raises(ValueError, match="device='cuda'"):
        MODEL_BUILDERS["tabicl"](
            {"class_names": ["A", "B"], "device": "cpu"},
            42,
        )
    global_model = MODEL_BUILDERS["tabicl"](
        {"class_names": ["A", "B"], "device": "cuda", "ensemble_size": 2},
        42,
    )
    assert used_tree_count(global_model) is None
    with pytest.raises(ValueError, match="device='cuda'"):
        global_model.set_params(device="cpu")

    with pytest.raises(ValueError, match="expert_device='cuda'"):
        MODEL_BUILDERS["tabicl_collision_expert"](
            {
                "class_names": ["A", "B"],
                "collision_pairs": [["A", "B"]],
                "expert_device": "cpu",
            },
            42,
        )


def test_tabicl_global_adapter_uses_fold_local_svd_and_many_classes(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorch:
        cuda = FakeCuda()

    class FakeTabICLClassifier:
        def __init__(self, **parameters) -> None:
            self.parameters = parameters

        def fit(self, features, labels):
            assert isinstance(features, np.ndarray)
            assert features.dtype == np.float32
            self.classes_ = np.unique(labels)
            return self

        def predict_proba(self, features) -> np.ndarray:
            return np.full(
                (len(features), len(self.classes_)),
                1.0 / len(self.classes_),
            )

    model = MODEL_BUILDERS["tabicl"](
        {
            "class_names": ["A", "B", "C"],
            "svd_components": 2,
            "ensemble_size": 3,
            "batch_size": 1,
            "device": "cuda",
        },
        42,
    )
    monkeypatch.setattr(
        model,
        "_import_dependencies",
        lambda: (FakeTorch, FakeTabICLClassifier),
    )
    features = csr_matrix(np.tile(np.eye(4, dtype=np.float32), (3, 1)))
    labels = np.tile(np.arange(3), 4)

    model.fit(features, labels)
    probabilities = model.predict_proba(features)

    assert model.reducer_.n_components == 2
    assert model.classifier_.parameters["device"] == "cuda"
    assert model.classifier_.parameters["support_many_classes"] is True
    assert model.classifier_.parameters["n_estimators"] == 3
    assert probabilities.shape == (len(labels), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_tabicl_collision_expert_preserves_pair_mass(monkeypatch) -> None:
    from src.models.tabicl_collision_expert_model import (
        TabICLCollisionExpertClassifier,
    )

    class FakeBaseModel:
        classes_ = np.asarray([0, 1, 2])

        def fit(self, features, labels):
            return self

        def predict_proba(self, features) -> np.ndarray:
            return np.asarray(
                [
                    [0.6, 0.3, 0.1],
                    [0.2, 0.3, 0.5],
                    [0.1, 0.7, 0.2],
                ]
            )[: features.shape[0]]

    class FakeExpert:
        classes_ = np.asarray([0, 1])

        def fit(self, features, labels):
            assert set(labels) == {0, 1}
            return self

        def predict_proba(self, features) -> np.ndarray:
            return np.tile(np.asarray([[0.25, 0.75]]), (features.shape[0], 1))

    model = TabICLCollisionExpertClassifier(
        class_names=("A", "B", "C"),
        collision_pairs=(("A", "B"),),
        expert_device="cuda",
        seed=42,
    )
    monkeypatch.setattr(model, "_build_base_model", lambda: FakeBaseModel())
    monkeypatch.setattr(model, "_build_expert", lambda pair: FakeExpert())
    features = csr_matrix(np.eye(3, dtype=np.float32))
    labels = np.asarray([0, 1, 2])

    model.fit(features, labels)
    probabilities = model.predict_proba(features)

    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.allclose(probabilities[0], [0.225, 0.675, 0.1])
    assert np.allclose(probabilities[1], [0.2, 0.3, 0.5])
    assert np.allclose(probabilities[2], [0.2, 0.6, 0.2])


@pytest.mark.skipif(
    shutil.which("nvidia-smi") is None,
    reason="CatBoost GPU integration test requires an NVIDIA GPU.",
)
def test_catboost_gpu_fits_sparse_features() -> None:
    base_features = np.eye(3, dtype=np.float32)
    features = csr_matrix(np.tile(base_features, (20, 1)))
    labels = np.tile(np.arange(3), 20)
    model = MODEL_BUILDERS["catboost"](
        {
            "n_estimators": 2,
            "depth": 2,
            "learning_rate": 0.1,
        },
        42,
    )

    model.fit(features, labels)
    probabilities = model.predict_proba(features)

    assert probabilities.shape == (len(labels), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert model.predict(features).shape == labels.shape
    assert used_tree_count(model) == 2

    restored = pickle.loads(pickle.dumps(model))
    assert restored.predict(features).shape == labels.shape


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
