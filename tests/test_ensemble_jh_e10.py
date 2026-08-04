import numpy as np
from pathlib import Path

from src.ensembles.train_jh_e10 import (
    combine_logit_scores,
    load_e9_test_probability,
    score_to_probability,
)


def test_score_to_probability_normalizes_last_axis() -> None:
    score = np.array(
        [
            [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
            [[3.0, 2.0, 1.0], [-1.0, 1.0, 0.0]],
        ]
    )
    probability = score_to_probability(score)
    assert probability.shape == score.shape
    np.testing.assert_allclose(probability.sum(axis=-1), 1.0)


def test_zero_e9_weight_uses_only_scaled_e8_score() -> None:
    e8 = np.array([[1.0, 2.0]])
    e9 = np.array([[0.9, 0.1]])
    combined = combine_logit_scores(e8, e9, 0.5, 2.0, 0.0)
    np.testing.assert_allclose(combined, e8 / 0.5)


def test_e9_mean_probability_is_expanded_for_legacy_output(
    tmp_path: Path,
) -> None:
    mean_probability = np.full((4, 3), 1.0 / 3.0)
    np.save(tmp_path / "test_probability_mean.npy", mean_probability)
    loaded, source = load_e9_test_probability(tmp_path, 3, 2)
    assert loaded.shape == (3, 2, 4, 3)
    assert source == "mean_probability_fallback"
    np.testing.assert_allclose(loaded[2, 1], mean_probability)
