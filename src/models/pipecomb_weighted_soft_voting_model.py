"""과적합 모델 비중을 낮추고 규제형 E14 비중을 높인 soft voting."""

from __future__ import annotations

import numpy as np

from src.models.pipecomb_ensemble_model import PipeCombEnsembleClassifier
from src.pipelines.pipeline_pipeComb_em_v1_001 import (
    WeightedSoftVotingFeatureBundle,
)


class PipeCombWeightedSoftVotingClassifier:
    """broad·E14·F05·F10 확률을 가중 평균합니다."""

    def __init__(self, model_config: dict, seed: int) -> None:
        self.seed = int(seed)
        voting_weights = model_config.get(
            "voting_weights",
            {
                "broad_ensemble": 0.05,
                "regularized_e14": 5.0,
                "em_f05_xgboost": 7.5,
                "em_f10_xgboost": 10.0,
            },
        )
        self.voting_weights = np.asarray([
            float(voting_weights.get("broad_ensemble", 0.0)),
            float(voting_weights.get("regularized_e14", 0.0)),
            float(voting_weights.get("em_f05_xgboost", 0.0)),
            float(voting_weights.get("em_f10_xgboost", 0.0)),
        ])
        if np.any(self.voting_weights < 0) or self.voting_weights.sum() <= 0:
            raise ValueError("soft voting 가중치는 음수가 아니며 합이 0보다 커야 합니다.")
        self.voting_weights /= self.voting_weights.sum()

        broad_config = dict(model_config.get("broad_ensemble", {}))
        broad_config["class_names"] = model_config.get("class_names", [])
        self.broad_model = PipeCombEnsembleClassifier(broad_config, self.seed)
        self.e14_config = dict(model_config.get("regularized_e14", {}))
        self.f05_config = dict(model_config.get("em_f05_xgboost", {}))
        self.f10_config = dict(model_config.get("em_f10_xgboost", {}))
        self.e14_model = None
        self.f05_model = None
        self.f10_model = None
        self.classes_: np.ndarray | None = None

    @staticmethod
    def _require_bundle(features) -> WeightedSoftVotingFeatureBundle:
        if not isinstance(features, WeightedSoftVotingFeatureBundle):
            raise TypeError(
                "pipecomb_weighted_soft_voting에는 "
                "preprocessing.name=pipeComb_em_v1_001이 필요합니다."
            )
        return features

    def _create_xgboost(self, config: dict, *, strongly_regularized: bool = False):
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError(
                "pipecomb_weighted_soft_voting에는 xgboost가 필요합니다."
            ) from error
        parameters = dict(config)
        parameters.setdefault("n_estimators", 260 if strongly_regularized else 500)
        parameters.setdefault("learning_rate", 0.025 if strongly_regularized else 0.03)
        parameters.setdefault("max_depth", 2 if strongly_regularized else 3)
        parameters.setdefault("min_child_weight", 12.0 if strongly_regularized else 5.0)
        parameters.setdefault("subsample", 0.70 if strongly_regularized else 0.75)
        parameters.setdefault("colsample_bytree", 0.50 if strongly_regularized else 0.60)
        parameters.setdefault("reg_alpha", 2.0 if strongly_regularized else 1.0)
        parameters.setdefault("reg_lambda", 20.0 if strongly_regularized else 12.0)
        parameters.setdefault("eval_metric", "mlogloss")
        parameters.setdefault("tree_method", "hist")
        parameters.setdefault("n_jobs", -1)
        parameters["random_state"] = self.seed
        return XGBClassifier(**parameters)

    def fit(self, features, labels):
        bundle = self._require_bundle(features)
        self.broad_model.fit(bundle.broad, labels)
        self.classes_ = np.asarray(self.broad_model.classes_)
        fitted_models = []
        if self.voting_weights[1] > 0:
            self.e14_model = self._create_xgboost(
                self.e14_config, strongly_regularized=True
            )
            self.e14_model.fit(bundle.regularized_e14, labels)
            fitted_models.append(self.e14_model)
        if self.voting_weights[2] > 0:
            self.f05_model = self._create_xgboost(self.f05_config)
            self.f05_model.fit(bundle.em_f05, labels)
            fitted_models.append(self.f05_model)
        if self.voting_weights[3] > 0:
            self.f10_model = self._create_xgboost(self.f10_config)
            self.f10_model.fit(bundle.em_f10, labels)
            fitted_models.append(self.f10_model)
        if not all(
            np.array_equal(self.classes_, model.classes_)
            for model in fitted_models
        ):
            raise ValueError("soft voting 구성 모델의 클래스 순서가 다릅니다.")
        return self

    def predict_proba(self, features) -> np.ndarray:
        required_models = (
            (self.voting_weights[1], self.e14_model),
            (self.voting_weights[2], self.f05_model),
            (self.voting_weights[3], self.f10_model),
        )
        if any(weight > 0 and model is None for weight, model in required_models):
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        bundle = self._require_bundle(features)
        probabilities = self.voting_weights[0] * self.broad_model.predict_proba(
            bundle.broad
        )
        if self.e14_model is not None:
            probabilities += self.voting_weights[1] * self.e14_model.predict_proba(
                bundle.regularized_e14
            )
        if self.f05_model is not None:
            probabilities += self.voting_weights[2] * self.f05_model.predict_proba(
                bundle.em_f05
            )
        if self.f10_model is not None:
            probabilities += self.voting_weights[3] * self.f10_model.predict_proba(
                bundle.em_f10
            )
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, features) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        return self.classes_[self.predict_proba(features).argmax(axis=1)]

    def summary(self) -> dict[str, object]:
        broad_weights = self.broad_model.weights * self.voting_weights[0]
        return {
            "voting": "weighted_soft_voting",
            "effective_weights": {
                "tfidf_linear_svc": float(broad_weights[0]),
                "emv45_xgboost": float(broad_weights[1]),
                "emv45_lightgbm": float(broad_weights[2]),
                "regularized_e14_xgboost": float(self.voting_weights[1]),
                "em_f05_xgboost": float(self.voting_weights[2]),
                "em_f10_xgboost": float(self.voting_weights[3]),
            },
        }


def create_model(
    model_config: dict, seed: int
) -> PipeCombWeightedSoftVotingClassifier:
    return PipeCombWeightedSoftVotingClassifier(model_config, seed)
