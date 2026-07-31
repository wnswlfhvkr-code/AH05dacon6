"""Word+Char TF-IDF용 LinearSVC와 LightGBM 결합 분류기입니다."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.svm import LinearSVC

from src.pipelines.pipeline_jsj_v1 import TextTreeFeatureBundle


class WCTfidfLinearSVCAndLightGBM:
    """LinearSVC 점수와 LightGBM 확률을 가중 평균하여 예측합니다."""

    def __init__(self, model_config: dict, seed: int) -> None:
        self.linear_weight = float(model_config.get("linear_weight", 0.95))
        self.tree_weight = float(model_config.get("tree_weight", 0.05))
        total_weight = self.linear_weight + self.tree_weight
        if total_weight <= 0:
            raise ValueError("linear_weight와 tree_weight의 합은 0보다 커야 합니다.")
        self.linear_weight /= total_weight
        self.tree_weight /= total_weight
        self.temperature = float(model_config.get("temperature", 0.5))
        if self.temperature <= 0:
            raise ValueError("temperature는 0보다 커야 합니다.")

        linear_config = model_config.get("linear", {})
        self.linear_model = LinearSVC(
            C=linear_config.get("C", 0.2),
            class_weight=linear_config.get("class_weight", "balanced"),
            max_iter=linear_config.get("max_iter", 10_000),
            tol=linear_config.get("tol", 1e-4),
            random_state=seed,
        )
        self.tree_config = model_config.get("tree", {})
        self.seed = seed
        multipliers = model_config.get("class_multipliers")
        self.class_multipliers = (
            None if multipliers is None else np.asarray(multipliers, dtype=np.float64)
        )
        self.tree_model = None
        self.classes_: np.ndarray | None = None

    @staticmethod
    def _require_bundle(features) -> TextTreeFeatureBundle:
        if not isinstance(features, TextTreeFeatureBundle):
            raise TypeError(
                "wc_tfidf_lsvc_lgbm 모델에는 preprocessing.name=jsj_v1과 "
                "return_bundle=true가 필요합니다."
            )
        return features

    def _create_tree_model(self):
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise ImportError(
                "결합 모델을 사용하려면 `pip install -r requirements.txt`를 실행하세요."
            ) from error

        return LGBMClassifier(
            objective=self.tree_config.get("objective", "multiclass"),
            n_estimators=self.tree_config.get("n_estimators", 300),
            learning_rate=self.tree_config.get("learning_rate", 0.05),
            num_leaves=self.tree_config.get("num_leaves", 31),
            max_depth=self.tree_config.get("max_depth", -1),
            min_child_samples=self.tree_config.get("min_child_samples", 20),
            subsample=self.tree_config.get("subsample", 1.0),
            colsample_bytree=self.tree_config.get("colsample_bytree", 1.0),
            reg_alpha=self.tree_config.get("reg_alpha", 0.0),
            reg_lambda=self.tree_config.get("reg_lambda", 0.0),
            class_weight=self.tree_config.get("class_weight"),
            random_state=self.seed,
            n_jobs=self.tree_config.get("n_jobs", -1),
            verbosity=self.tree_config.get("verbosity", -1),
        )

    def fit(self, features, labels):
        """텍스트 피처에는 LinearSVC, 구조 피처에는 LightGBM을 학습합니다."""
        bundle = self._require_bundle(features)
        self.linear_model.fit(bundle.text, labels)
        self.tree_model = self._create_tree_model()
        self.tree_model.fit(bundle.tree, labels)
        self.classes_ = self.linear_model.classes_
        if not np.array_equal(self.classes_, self.tree_model.classes_):
            raise ValueError("LinearSVC와 LightGBM의 클래스 순서가 다릅니다.")
        if (
            self.class_multipliers is not None
            and len(self.class_multipliers) != len(self.classes_)
        ):
            raise ValueError(
                "class_multipliers 길이는 학습 데이터의 클래스 수와 같아야 합니다."
            )
        return self

    def predict_proba(self, features) -> np.ndarray:
        """온도 보정한 LinearSVC 점수와 LightGBM 확률을 결합합니다."""
        if self.tree_model is None or self.classes_ is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        bundle = self._require_bundle(features)
        linear_scores = self.linear_model.decision_function(bundle.text)
        if linear_scores.ndim == 1:
            linear_scores = np.column_stack((-linear_scores, linear_scores))
        linear_probabilities = softmax(
            linear_scores / self.temperature, axis=1
        )
        tree_probabilities = self.tree_model.predict_proba(bundle.tree)
        probabilities = (
            self.linear_weight * linear_probabilities
            + self.tree_weight * tree_probabilities
        )
        if self.class_multipliers is not None:
            probabilities = probabilities * self.class_multipliers
            probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def predict(self, features) -> np.ndarray:
        """결합 확률이 가장 큰 클래스를 반환합니다."""
        if self.classes_ is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        indices = self.predict_proba(features).argmax(axis=1)
        return self.classes_[indices]


def create_model(model_config: dict, seed: int) -> WCTfidfLinearSVCAndLightGBM:
    """현재 최고 제출 조합을 재현 가능한 형태로 구성합니다."""
    return WCTfidfLinearSVCAndLightGBM(model_config, seed)
