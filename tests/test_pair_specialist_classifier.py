import numpy as np
from scipy.sparse import csr_matrix

from src.models import MODEL_BUILDERS
from src.models.pair_specialist_classifier_model import PairSpecialistClassifier
from src.train import build_model


def _model_config() -> dict:
    return {
        "name": "pair_specialist_classifier",
        "base_model": {
            "name": "linear_svc",
            "C": 0.2,
            "class_weight": "balanced",
            "max_iter": 5000,
            "dual": "auto",
        },
        "activation": "top2_pair_match",
        "temperature": 1.0,
        "pairs": [
            {
                "labels": ["KIRC", "KIPAN"],
                "blend_weight": 0.25,
                "minimum_activation_probability": 0.0,
            },
            {
                "labels": ["LGG", "GBMLGG"],
                "blend_weight": 0.25,
                "minimum_activation_probability": 0.0,
            },
        ],
    }


def test_pair_specialist_is_registered_and_receives_label_names() -> None:
    config = {
        "project": {"seed": 42},
        "model": _model_config(),
    }
    class_names = np.asarray(["GBMLGG", "KIPAN", "KIRC", "LGG"])

    model = build_model(config, class_names)

    assert "pair_specialist_classifier" in MODEL_BUILDERS
    assert isinstance(model, PairSpecialistClassifier)
    assert model.class_names == class_names.tolist()


def test_pair_specialist_fits_predicts_and_preserves_classes() -> None:
    features = csr_matrix(np.asarray([
        [2.0, 0.0, 0.0, 0.0],
        [1.8, 0.2, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.2, 1.8, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.2, 1.8, 0.0],
        [0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.2, 1.8],
    ]))
    labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    model_config = _model_config()
    model_config["class_names"] = ["GBMLGG", "KIPAN", "KIRC", "LGG"]
    model = PairSpecialistClassifier(model_config, seed=42)

    model.fit(features, labels)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    assert model.summary() == {
        "configured_pairs": 2,
        "fitted_pair_specialists": 2,
        "skipped_pair_specialists": 0,
    }
    assert predictions.shape == labels.shape
    assert set(predictions) <= set(labels)
    assert probabilities.shape == (len(labels), 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
