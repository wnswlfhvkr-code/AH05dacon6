import numpy as np
import pandas as pd

from src.models import MODEL_BUILDERS
from src.models.tabpfn3_model import TabPFN3Classifier


class _FakeTabPFN:
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


def test_tabpfn3_is_registered_and_reduces_fold_train_features(monkeypatch) -> None:
    features = pd.DataFrame({
        "TP53": [1, 1, 0, 0, 1, 0],
        "BRAF": [0, 0, 1, 1, 0, 1],
        "EGFR": [0, 1, 0, 1, 0, 1],
        "KRAS": [1, 0, 1, 0, 1, 0],
    }, dtype="float32")
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    fake = _FakeTabPFN()
    model = TabPFN3Classifier({
        "model_version": "v3",
        "reduction_method": "chi2",
        "max_features": 2,
    }, seed=42)
    monkeypatch.setattr(model, "_create_tabpfn", lambda: fake)

    model.fit(features, labels)
    probabilities = model.predict_proba(features)

    assert "tabpfn3" in MODEL_BUILDERS
    assert fake.fit_shape == (6, 2)
    assert model.summary() == {
        "input_features": 4,
        "selected_features": 2,
        "removed_features": 2,
    }
    assert probabilities.shape == (6, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert model.predict(features).shape == labels.shape


def test_tabpfn3_chi2_rejects_negative_features(monkeypatch) -> None:
    features = pd.DataFrame({"signed": [-1.0, 0.0, 1.0, 2.0]})
    labels = np.asarray([0, 0, 1, 1])
    model = TabPFN3Classifier({
        "reduction_method": "chi2",
        "max_features": 1,
    }, seed=42)
    monkeypatch.setattr(model, "_create_tabpfn", _FakeTabPFN)

    try:
        model.fit(features, labels)
    except ValueError as error:
        assert "음수가 없는 입력" in str(error)
    else:
        raise AssertionError("음수 chi2 입력은 거부되어야 합니다.")
