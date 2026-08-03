import numpy as np
import pandas as pd
import pytest

from src.train import fit_model, split_preprocessing_config, used_tree_count


class EarlyStoppingModelStub:
    early_stopping_rounds = 30
    best_iteration = 4

    def __init__(self) -> None:
        self.fit_arguments = None

    def fit(self, train_x, train_y, **kwargs) -> None:
        self.fit_arguments = (train_x, train_y, kwargs)

    def get_params(self) -> dict:
        return {"n_estimators": 500}


def test_fit_model_passes_validation_set_for_early_stopping() -> None:
    model = EarlyStoppingModelStub()
    train_x = pd.DataFrame({"feature": [0.0, 1.0]})
    train_y = np.array([0, 1])
    valid_x = pd.DataFrame({"feature": [1.0]})
    valid_y = np.array([1])

    fit_model(model, train_x, train_y, valid_x, valid_y)

    assert model.fit_arguments is not None
    assert model.fit_arguments[2]["eval_set"] == [(valid_x, valid_y)]
    assert model.fit_arguments[2]["verbose"] is False
    assert used_tree_count(model) == 5


def test_fit_model_rejects_early_stopping_without_validation_set() -> None:
    model = EarlyStoppingModelStub()

    with pytest.raises(ValueError, match="검증 데이터"):
        fit_model(
            model,
            pd.DataFrame({"feature": [0.0, 1.0]}),
            np.array([0, 1]),
        )


def test_split_preprocessing_config_separates_em_v20_support_cv() -> None:
    pipeline_config, tuning_config = split_preprocessing_config({
        "name": "em_v20",
        "min_functional_mutation_count": 5,
        "min_functional_mutation_count_cv": {
            "enabled": True,
            "candidates": [5, 8, 10],
            "folds": 3,
        },
    })

    assert pipeline_config == {
        "name": "em_v20",
        "min_functional_mutation_count": 5,
    }
    assert tuning_config["candidates"] == [5, 8, 10]
