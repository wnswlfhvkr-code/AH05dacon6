import numpy as np
import pandas as pd

from src.models import MODEL_BUILDERS
from src.models.macro_f1_calibrated_classifier_model import (
    MacroF1CalibratedClassifier,
)


def _config() -> dict:
    return {
        "base_model": {
            "name": "logistic_regression",
            "C": 0.2,
            "class_weight": "balanced",
            "solver": "saga",
            "penalty": "l2",
            "max_iter": 1000,
        },
        "adjustment_method": "class_prior_power",
        "gamma_candidates": [-0.25, 0.0, 0.25],
        "temperature_candidates": [0.85, 1.0, 1.15],
        "calibration_folds": 3,
    }


def test_macro_f1_calibrated_classifier_is_registered_and_predicts() -> None:
    features = pd.DataFrame({
        "x1": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
        "x2": [0.0, 0.2, 0.1, 1.0, 1.2, 1.1, 0.0, 0.2, 0.1],
    })
    labels = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2])
    model = MacroF1CalibratedClassifier(_config(), seed=42)

    model.fit(features, labels)
    probabilities = model.predict_proba(features)
    predictions = model.predict(features)

    assert "macro_f1_calibrated_classifier" in MODEL_BUILDERS
    assert model.selected_gamma_ in _config()["gamma_candidates"]
    assert model.selected_temperature_ in _config()["temperature_candidates"]
    assert len(model.calibration_results_) == 9
    assert probabilities.shape == (len(labels), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert predictions.shape == labels.shape


def test_positive_gamma_increases_rare_class_relative_probability() -> None:
    probabilities = np.asarray([[0.6, 0.4]])
    class_prior = np.asarray([0.9, 0.1])

    baseline = MacroF1CalibratedClassifier._adjust_probabilities(
        probabilities, class_prior, gamma=0.0, temperature=1.0
    )
    adjusted = MacroF1CalibratedClassifier._adjust_probabilities(
        probabilities, class_prior, gamma=0.5, temperature=1.0
    )

    assert adjusted[0, 1] > baseline[0, 1]
