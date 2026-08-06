import numpy as np

from src.ensembles.train_jh_e10a import (
    apply_pair_experts,
    mix_team_probability,
)


def test_mix_team_probability_is_normalized() -> None:
    em16 = np.array([[0.7, 0.3], [0.2, 0.8]])
    em24 = np.array([[0.4, 0.6], [0.5, 0.5]])
    mixed = mix_team_probability(em16, em24, 0.4)
    np.testing.assert_allclose(mixed.sum(axis=1), 1.0)


def test_pair_expert_preserves_probability_and_pair_mass() -> None:
    base = np.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    team = np.array([[0.2, 0.7, 0.1], [0.4, 0.1, 0.5]])
    adjusted = apply_pair_experts(
        base, team,
        {"KIRC_KIPAN": (0, 1), "LGG_GBMLGG": (1, 2)},
        {"KIRC_KIPAN": 0.5, "LGG_GBMLGG": 0.3},
        temperature=1.0,
        kirc_pair_weight=0.25,
        lgg_pair_weight=0.30,
    )
    np.testing.assert_allclose(adjusted.sum(axis=1), 1.0)
    np.testing.assert_allclose(adjusted[0, :2].sum(), base[0, :2].sum())
    np.testing.assert_allclose(adjusted[1, 1:].sum(), base[1, 1:].sum())
