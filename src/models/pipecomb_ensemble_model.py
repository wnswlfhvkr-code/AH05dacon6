"""pipeComb_em_v1의 텍스트·수치 뷰를 위한 저상관 앙상블입니다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, softmax
from sklearn.svm import LinearSVC

from src.pipelines.pipeline_jsj_v1 import TextTreeFeatureBundle


@dataclass
class _PairExpert:
    left_class: int
    right_class: int
    left_column: int
    right_column: int
    blend_weight: float
    minimum_activation_probability: float
    temperature: float
    model: LinearSVC


class PipeCombEnsembleClassifier:
    """TF-IDF LinearSVC와 두 종류의 규제 트리 모델을 결합합니다.

    전역 예측은 LinearSVC·XGBoost·LightGBM의 확률 평균이며, 두 중첩 암종
    쌍은 해당 두 클래스가 top-2일 때만 fold-train 이진 전문가로 약하게
    보정합니다. 원본 다중분류 레이블은 병합하거나 분해하지 않습니다.
    """

    def __init__(self, model_config: dict, seed: int) -> None:
        self.seed = int(seed)
        self.class_names = [str(value) for value in model_config.get("class_names", [])]
        self.temperature = float(model_config.get("linear_temperature", 0.8))
        if self.temperature <= 0:
            raise ValueError("linear_temperature는 0보다 커야 합니다.")

        weights = model_config.get(
            "weights", {"linear_svc": 0.50, "xgboost": 0.30, "lightgbm": 0.20}
        )
        self.weights = np.asarray([
            float(weights.get("linear_svc", 0.0)),
            float(weights.get("xgboost", 0.0)),
            float(weights.get("lightgbm", 0.0)),
        ])
        if np.any(self.weights < 0) or self.weights.sum() <= 0:
            raise ValueError("앙상블 가중치는 음수가 아니며 합이 0보다 커야 합니다.")
        self.weights /= self.weights.sum()

        self.linear_config = dict(model_config.get("linear_svc", {}))
        self.xgb_config = dict(model_config.get("xgboost", {}))
        self.lgbm_config = dict(model_config.get("lightgbm", {}))
        self.pair_configs = list(model_config.get("pair_specialists", []))
        self.linear_model = self._create_linear(self.linear_config)
        self.xgb_model = None
        self.lgbm_model = None
        self.classes_: np.ndarray | None = None
        self.pair_experts_: list[_PairExpert] = []

    @staticmethod
    def _require_bundle(features) -> TextTreeFeatureBundle:
        if not isinstance(features, TextTreeFeatureBundle):
            raise TypeError(
                "pipecomb_ensemble에는 preprocessing.name=pipeComb_em_v1이 필요합니다."
            )
        return features

    def _create_linear(self, config: dict) -> LinearSVC:
        return LinearSVC(
            C=float(config.get("C", 0.15)),
            class_weight=config.get("class_weight", "balanced"),
            max_iter=int(config.get("max_iter", 20_000)),
            tol=float(config.get("tol", 1e-4)),
            dual=config.get("dual", "auto"),
            random_state=self.seed,
        )

    def _create_xgboost(self):
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError("pipecomb_ensemble에는 xgboost가 필요합니다.") from error
        config = dict(self.xgb_config)
        config.setdefault("n_estimators", 350)
        config.setdefault("learning_rate", 0.03)
        config.setdefault("max_depth", 3)
        config.setdefault("min_child_weight", 8.0)
        config.setdefault("subsample", 0.75)
        config.setdefault("colsample_bytree", 0.55)
        config.setdefault("reg_alpha", 1.0)
        config.setdefault("reg_lambda", 12.0)
        config.setdefault("eval_metric", "mlogloss")
        config.setdefault("tree_method", "hist")
        config.setdefault("n_jobs", -1)
        config["random_state"] = self.seed
        return XGBClassifier(**config)

    def _create_lightgbm(self):
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise ImportError("pipecomb_ensemble에는 lightgbm이 필요합니다.") from error
        config = dict(self.lgbm_config)
        config.setdefault("objective", "multiclass")
        config.setdefault("n_estimators", 300)
        config.setdefault("learning_rate", 0.03)
        config.setdefault("num_leaves", 15)
        config.setdefault("max_depth", 5)
        config.setdefault("min_child_samples", 35)
        config.setdefault("subsample", 0.75)
        config.setdefault("subsample_freq", 1)
        config.setdefault("colsample_bytree", 0.55)
        config.setdefault("reg_alpha", 1.0)
        config.setdefault("reg_lambda", 12.0)
        config.setdefault("verbosity", -1)
        config.setdefault("n_jobs", -1)
        config["random_state"] = self.seed
        return LGBMClassifier(**config)

    def _resolve_class(self, label: object) -> int:
        if isinstance(label, (int, np.integer)):
            return int(label)
        if not self.class_names:
            raise ValueError("문자열 pair 레이블에는 class_names가 필요합니다.")
        try:
            return self.class_names.index(str(label))
        except ValueError as error:
            raise ValueError(f"학습 레이블에 없는 pair 클래스입니다: {label}") from error

    def _fit_pair_experts(self, text, labels: np.ndarray) -> None:
        if self.classes_ is None:
            raise RuntimeError("전역 모델 클래스가 먼저 확정되어야 합니다.")
        class_to_column = {
            int(value): column for column, value in enumerate(self.classes_)
        }
        self.pair_experts_ = []
        for config in self.pair_configs:
            pair = list(config.get("labels", []))
            if len(pair) != 2:
                raise ValueError("pair_specialists.labels는 레이블 2개여야 합니다.")
            left, right = self._resolve_class(pair[0]), self._resolve_class(pair[1])
            if left not in class_to_column or right not in class_to_column:
                continue
            mask = np.isin(labels, [left, right])
            if np.unique(labels[mask]).size != 2:
                continue
            specialist_config = dict(self.linear_config)
            specialist_config.update(config.get("model", {}))
            blend_weight = float(config.get("blend_weight", 0.15))
            minimum_probability = float(
                config.get("minimum_activation_probability", 0.55)
            )
            temperature = float(config.get("temperature", 1.5))
            if not 0.0 <= blend_weight <= 1.0:
                raise ValueError("pair blend_weight는 0 이상 1 이하여야 합니다.")
            if not 0.0 <= minimum_probability <= 1.0:
                raise ValueError(
                    "pair minimum_activation_probability는 0 이상 1 이하여야 합니다."
                )
            if temperature <= 0:
                raise ValueError("pair temperature는 0보다 커야 합니다.")
            specialist = self._create_linear(specialist_config)
            specialist.fit(text[mask], labels[mask])
            self.pair_experts_.append(_PairExpert(
                left_class=left,
                right_class=right,
                left_column=class_to_column[left],
                right_column=class_to_column[right],
                blend_weight=blend_weight,
                minimum_activation_probability=minimum_probability,
                temperature=temperature,
                model=specialist,
            ))

    def fit(self, features, labels):
        bundle = self._require_bundle(features)
        encoded_labels = np.asarray(labels)
        self.linear_model.fit(bundle.text, encoded_labels)
        self.xgb_model = self._create_xgboost()
        self.lgbm_model = self._create_lightgbm()
        self.xgb_model.fit(bundle.tree, encoded_labels)
        self.lgbm_model.fit(bundle.tree, encoded_labels)
        self.classes_ = np.asarray(self.linear_model.classes_)
        if not (
            np.array_equal(self.classes_, self.xgb_model.classes_)
            and np.array_equal(self.classes_, self.lgbm_model.classes_)
        ):
            raise ValueError("앙상블 구성 모델의 클래스 순서가 다릅니다.")
        self._fit_pair_experts(bundle.text, encoded_labels)
        return self

    def _global_probabilities(self, bundle: TextTreeFeatureBundle) -> np.ndarray:
        if self.xgb_model is None or self.lgbm_model is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        linear_scores = np.asarray(self.linear_model.decision_function(bundle.text))
        if linear_scores.ndim == 1:
            linear_scores = np.column_stack((-linear_scores, linear_scores))
        probabilities = (
            self.weights[0] * softmax(linear_scores / self.temperature, axis=1)
            + self.weights[1] * self.xgb_model.predict_proba(bundle.tree)
            + self.weights[2] * self.lgbm_model.predict_proba(bundle.tree)
        )
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def _apply_pair_experts(
        self, probabilities: np.ndarray, text
    ) -> np.ndarray:
        output = probabilities.copy()
        top_two = np.argpartition(output, -2, axis=1)[:, -2:]
        for expert in self.pair_experts_:
            columns = np.asarray([expert.left_column, expert.right_column])
            active = (
                np.all(np.sort(top_two, axis=1) == np.sort(columns), axis=1)
                & (output[:, columns].sum(axis=1) >= expert.minimum_activation_probability)
            )
            if not np.any(active):
                continue
            margin = np.asarray(
                expert.model.decision_function(text[active]), dtype="float64"
            ).reshape(-1)
            if int(expert.model.classes_[1]) != expert.right_class:
                margin = -margin
            expert_right = expit(margin / expert.temperature)
            pair_mass = output[active][:, columns].sum(axis=1)
            base_right = output[active, expert.right_column] / np.maximum(pair_mass, 1e-12)
            mixed_right = (
                (1.0 - expert.blend_weight) * base_right
                + expert.blend_weight * expert_right
            )
            output[active, expert.left_column] = pair_mass * (1.0 - mixed_right)
            output[active, expert.right_column] = pair_mass * mixed_right
        return output / output.sum(axis=1, keepdims=True)

    def predict_proba(self, features) -> np.ndarray:
        bundle = self._require_bundle(features)
        return self._apply_pair_experts(
            self._global_probabilities(bundle), bundle.text
        )

    def predict(self, features) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        return self.classes_[self.predict_proba(features).argmax(axis=1)]

    def summary(self) -> dict[str, object]:
        return {
            "global_models": 3,
            "fitted_pair_specialists": len(self.pair_experts_),
            "weights": self.weights.tolist(),
        }


def create_model(model_config: dict, seed: int) -> PipeCombEnsembleClassifier:
    return PipeCombEnsembleClassifier(model_config, seed)
