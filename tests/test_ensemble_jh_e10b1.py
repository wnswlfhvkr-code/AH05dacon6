import numpy as np

from src.ensembles.train_jh_e10b1 import (
    paired_bootstrap_mean_probability,
)


def test_mean_probability_bootstrap_uses_seed_average() -> None:
    y = np.array([0, 0, 1, 1])
    baseline = np.array([
        [[0.6, 0.4], [0.6, 0.4], [0.4, 0.6], [0.4, 0.6]],
        [[0.6, 0.4], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6]],
    ])
    candidate = baseline.copy()
    candidate[:, 1] = [0.8, 0.2]
    result = paired_bootstrap_mean_probability(
        y, candidate, baseline, repeats=50, random_state=42, n_classes=2,
    )
    assert result["candidate_macro_f1"] >= result["baseline_macro_f1"]
    assert result["observed_delta"] >= 0
