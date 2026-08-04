import numpy as np
import pandas as pd

from src.models import MODEL_BUILDERS
from src.models.pattern_posterior_classifier_model import (
    PatternPosteriorClassifier,
)
from src.pipelines.pipeline_em_v16 import (
    EMV16PreprocessingPipeline,
    RAW_PATTERN_KEY_COLUMN,
    create_raw_mutation_pattern_keys,
)


def _config() -> dict:
    return {
        "base_model": {
            "name": "logistic_regression",
            "C": 0.2,
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 1000,
        },
        "posterior_blend_weight": 0.25,
        "pattern_key_column": RAW_PATTERN_KEY_COLUMN,
        "smoothing_alpha": 1.0,
        "minimum_group_size": 2,
        "unseen_pattern_action": "base_model_only",
    }


def test_raw_pattern_key_is_stable_and_uses_all_columns() -> None:
    raw = pd.DataFrame({
        "TP53": ["WT", "WT", "R175H"],
        "BRAF": ["V600E", "V600E", "V600E"],
    })

    keys = create_raw_mutation_pattern_keys(raw)

    assert keys.iloc[0] == keys.iloc[1]
    assert keys.iloc[0] != keys.iloc[2]
    assert keys.name == RAW_PATTERN_KEY_COLUMN


def test_pattern_posterior_is_registered_and_uses_train_lookup_only() -> None:
    features = pd.DataFrame({
        "feature_1": [0.0, 0.1, 1.0, 1.1, 0.2, 0.8],
        "feature_2": [1.0, 0.9, 0.0, 0.1, 0.8, 0.2],
        RAW_PATTERN_KEY_COLUMN: ["duplicate", "duplicate", "b", "c", "d", "e"],
    })
    labels = np.asarray([0, 1, 1, 1, 0, 1])
    model = PatternPosteriorClassifier(_config(), seed=42)

    model.fit(features, labels)
    probabilities = model.predict_proba(features)
    unseen = features.iloc[[0]].copy()
    unseen[RAW_PATTERN_KEY_COLUMN] = "never_seen"
    unseen_probabilities = model.predict_proba(unseen)

    assert "pattern_posterior_classifier" in MODEL_BUILDERS
    assert model.summary() == {
        "matched_pattern_groups": 1,
        "conflicting_pattern_groups": 1,
        "matched_training_samples": 2,
    }
    assert probabilities.shape == (len(labels), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.allclose(
        unseen_probabilities,
        model.base_model.predict_proba(unseen.drop(columns=[RAW_PATTERN_KEY_COLUMN])),
    )


def test_em_v16_pattern_key_and_classifier_work_together() -> None:
    raw = pd.DataFrame({
        "TP53": ["WT", "WT", "R1H", "R1H", "WT", "R2H"],
        "BRAF": ["V1E", "V1E", "WT", "WT", "V2E", "WT"],
    })
    labels = pd.Series(["A", "B", "A", "B", "A", "B"])
    pipeline = EMV16PreprocessingPipeline(
        min_mutation_count=1,
        top_genes_per_class=1,
        min_hotspot_count=1,
        max_hotspots=4,
        inner_signature_folds=2,
        include_raw_pattern_key=True,
    )

    transformed = pipeline.fit_transform(raw, labels)
    model = PatternPosteriorClassifier(_config(), seed=42)
    model.fit(transformed, pipeline.encode_labels(labels))

    assert RAW_PATTERN_KEY_COLUMN in transformed
    assert model.predict(transformed).shape == (len(labels),)
    assert model.summary()["matched_pattern_groups"] == 2
