import pickle

import numpy as np
import pytest

from src.models.realtabr_collision_expert_model import SVDRealTabRExpert


class _InferenceClassifier:
    fit_calls = 0

    def __init__(self) -> None:
        self.classes_ = np.asarray([0, 1], dtype=np.int64)
        self._score = lambda values: values[:, 0] - 0.5 * values[:, 1]

    def fit(self, values, labels):
        type(self).fit_calls += 1
        return self

    def predict_proba(self, values) -> np.ndarray:
        score = self._score(np.asarray(values, dtype=np.float32))
        right = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack((1.0 - right, right))


def _expert() -> SVDRealTabRExpert:
    return SVDRealTabRExpert(
        svd_components=4,
        device="cpu",
        n_epochs=1,
        batch_size=4,
        eval_batch_size=4,
        context_size=2,
        d_main=4,
        patience=1,
        val_fraction=0.25,
        verbosity=0,
        seed=7,
    )


def _fit_fake_expert(monkeypatch) -> tuple[SVDRealTabRExpert, np.ndarray]:
    features = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    monkeypatch.setattr(
        SVDRealTabRExpert,
        "_fit_classifier",
        lambda self, dense, labels_array: _InferenceClassifier(),
    )
    return _expert().fit(features, labels), features


def test_pickle_roundtrip_preserves_prediction(monkeypatch):
    expert, features = _fit_fake_expert(monkeypatch)
    expected = expert.predict_proba(features)

    restored = pickle.loads(pickle.dumps(expert, protocol=pickle.HIGHEST_PROTOCOL))
    resumed = pickle.loads(pickle.dumps(restored, protocol=pickle.HIGHEST_PROTOCOL))

    np.testing.assert_allclose(restored.predict_proba(features), expected)
    np.testing.assert_allclose(resumed.predict_proba(features), expected)


def test_pickle_roundtrip_predict_never_refits(monkeypatch):
    expert, features = _fit_fake_expert(monkeypatch)
    payload = pickle.dumps(expert, protocol=pickle.HIGHEST_PROTOCOL)
    fit_calls = 0
    _InferenceClassifier.fit_calls = 0

    def forbidden_fit(self, dense, labels_array):
        nonlocal fit_calls
        fit_calls += 1
        raise AssertionError("inference must not fit RealTabR")

    monkeypatch.setattr(SVDRealTabRExpert, "_fit_classifier", forbidden_fit)
    restored = pickle.loads(payload)
    restored.predict_proba(features)

    assert fit_calls == 0
    assert _InferenceClassifier.fit_calls == 0


def test_legacy_fitted_state_without_inference_snapshot_is_rejected(monkeypatch):
    expert, _ = _fit_fake_expert(monkeypatch)

    def legacy_getstate(self):
        state = self.__dict__.copy()
        state.pop("classifier_", None)
        return state

    monkeypatch.setattr(SVDRealTabRExpert, "__getstate__", legacy_getstate)
    legacy_payload = pickle.dumps(expert, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(RuntimeError, match="구형 checkpoint.*다시 생성"):
        pickle.loads(legacy_payload)


def test_unfitted_expert_remains_pickle_compatible():
    restored = pickle.loads(pickle.dumps(_expert(), protocol=pickle.HIGHEST_PROTOCOL))

    with pytest.raises(RuntimeError, match="먼저 fit"):
        restored.predict_proba(np.zeros((1, 2), dtype=np.float32))
