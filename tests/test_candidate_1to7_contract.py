from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
import yaml

from src.models import MODEL_BUILDERS
from src.models.modernnca_collision_expert_model import (
    ModernNCACollisionExpertClassifier,
)
from src.models.realtabr_collision_expert_model import (
    RealTabRCollisionExpertClassifier,
)
from src.train import used_tree_count


ROOT = Path(__file__).parents[1]
CLASS_NAMES = (
    "ACC", "BLCA", "BRCA", "CESC", "COAD", "DLBC", "GBMLGG", "HNSC",
    "KIPAN", "KIRC", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "OV",
    "PAAD", "PCPG", "PRAD", "SARC", "SKCM", "STES", "TGCT", "THCA",
    "THYM", "UCEC",
)
ROSTER = {
    "test_007_tabm_global_c1.yaml": "tabm",
    "test_007_modernnca_global_c1.yaml": "modernnca",
    "test_007_modernnca_expert_c1.yaml": "modernnca_collision_expert",
    "test_007_realmlp_global_c1.yaml": "realmlp",
    "test_007_realtabr_expert_c1.yaml": "realtabr_collision_expert",
    "test_007_xrfm_c1.yaml": "xrfm",
}


def _config(filename: str) -> dict:
    return yaml.safe_load(
        (ROOT / "data" / "backup" / "yaml" / filename).read_text(
            encoding="utf-8"
        )
    )


def test_corrected_candidate_roster_is_registered_and_uses_comb3() -> None:
    for filename, model_name in ROSTER.items():
        config = _config(filename)
        assert config["model"]["name"] == model_name
        assert model_name in MODEL_BUILDERS
        assert config["preprocessing"] == {"name": "pipeComb_v3"}
        assert config["project"]["seed"] == 42
    assert not (ROOT / "src/pipelines/jyp_preprocessing/pipeline_pipe_comb_v3_2.py").exists()


def test_candidate_roles_and_devices_match_the_corrected_instruction() -> None:
    tabm = _config("test_007_tabm_global_c1.yaml")["model"]
    modern_global = _config("test_007_modernnca_global_c1.yaml")["model"]
    modern_expert = _config("test_007_modernnca_expert_c1.yaml")["model"]
    realmlp = _config("test_007_realmlp_global_c1.yaml")["model"]
    realtabr = _config("test_007_realtabr_expert_c1.yaml")["model"]
    xrfm = _config("test_007_xrfm_c1.yaml")["model"]

    assert len(tabm["class_names"]) == 26 and tabm["device"] == "cuda"
    assert len(modern_global["class_names"]) == 26
    assert modern_global["device"] == "cuda"
    assert modern_expert["base_model"]["device"] == "cuda"
    assert modern_expert["expert_model"]["device"] == "cuda"
    assert len(realmlp["class_names"]) == 26 and realmlp["device"] == "cuda:0"
    assert realtabr["expert_device"] == "cpu"
    assert realtabr["base_model"]["device"] == "cuda"
    assert xrfm["device"] == "cpu"


@pytest.mark.parametrize(
    "filename",
    [
        "test_007_tabm_global_c1.yaml",
        "test_007_modernnca_global_c1.yaml",
        "test_007_modernnca_expert_c1.yaml",
        "test_007_realmlp_global_c1.yaml",
        "test_007_realtabr_expert_c1.yaml",
        "test_007_xrfm_c1.yaml",
    ],
)
def test_new_candidates_are_non_tree_registry_models(filename: str) -> None:
    config = _config(filename)
    model = MODEL_BUILDERS[config["model"]["name"]](config["model"], 42)
    expected = 1 if config["model"]["name"] == "xrfm" else None
    assert used_tree_count(model) == expected


class _FakeBase:
    classes_ = np.arange(26)

    def predict_proba(self, features):
        probabilities = np.full((int(features.shape[0]), 26), 0.01, dtype=np.float64)
        probabilities[:, 9] = 0.40
        probabilities[:, 8] = 0.35
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities


class _FakePairExpert:
    def predict_proba(self, features):
        return np.tile(np.asarray([[0.2, 0.8]]), (int(features.shape[0]), 1))


@pytest.mark.parametrize(
    "model_type",
    [
        ModernNCACollisionExpertClassifier,
        RealTabRCollisionExpertClassifier,
    ],
)
def test_collision_candidates_preserve_pair_mass_and_nonpair_columns(model_type) -> None:
    if model_type is ModernNCACollisionExpertClassifier:
        model = model_type(class_names=CLASS_NAMES)
    else:
        model = model_type(class_names=CLASS_NAMES)
    model.base_model_ = _FakeBase()
    model.classes_ = np.arange(26)
    model.experts_ = {(9, 8): _FakePairExpert()}
    features = csr_matrix(np.ones((4, 6), dtype=np.float32))

    base = model.base_model_.predict_proba(features)
    combined = model.predict_proba(features)

    outside = [column for column in range(26) if column not in {8, 9}]
    assert np.array_equal(combined[:, outside], base[:, outside])
    assert np.allclose(combined[:, [9, 8]].sum(axis=1), base[:, [9, 8]].sum(axis=1))
    assert np.allclose(combined.sum(axis=1), 1.0)
    assert np.all(np.argmax(combined, axis=1) == 8)
