"""동일한 원시 변이 패턴의 train-only posterior를 혼합하는 모델입니다."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.pipelines.pipeline_em_v16 import RAW_PATTERN_KEY_COLUMN


class PatternPosteriorClassifier:
    """기본 모델 확률과 학습 데이터의 동일 패턴 레이블 분포를 혼합합니다.

    패턴별 레이블 분포는 ``fit``에 전달된 데이터에서만 계산됩니다. 따라서
    OOF에서는 fold-train에만 fit되고 validation에는 고정 lookup만 적용되며,
    최종 학습에서는 전체 train에 fit된 lookup을 test에 그대로 적용합니다.
    """

    def __init__(self, model_config: dict, seed: int) -> None:
        self.seed = seed
        self.pattern_key_column = str(
            model_config.get("pattern_key_column", RAW_PATTERN_KEY_COLUMN)
        )
        self.posterior_blend_weight = float(
            model_config.get("posterior_blend_weight", 0.25)
        )
        self.smoothing_alpha = float(model_config.get("smoothing_alpha", 1.0))
        self.minimum_group_size = int(model_config.get("minimum_group_size", 2))
        self.unseen_pattern_action = str(
            model_config.get("unseen_pattern_action", "base_model_only")
        )
        if not 0.0 <= self.posterior_blend_weight <= 1.0:
            raise ValueError("posterior_blend_weight는 0 이상 1 이하여야 합니다.")
        if self.smoothing_alpha < 0:
            raise ValueError("smoothing_alpha는 0 이상이어야 합니다.")
        if self.minimum_group_size < 2:
            raise ValueError("minimum_group_size는 2 이상이어야 합니다.")
        if self.unseen_pattern_action != "base_model_only":
            raise ValueError("unseen_pattern_action은 'base_model_only'만 지원합니다.")

        self.base_config = dict(model_config.get("base_model", {}))
        base_name = self.base_config.pop("name", "logistic_regression")
        if base_name != "logistic_regression":
            raise ValueError("현재 base_model은 logistic_regression만 지원합니다.")

        self.base_model: LogisticRegression | None = None
        self.classes_: np.ndarray | None = None
        self.pattern_posteriors_: dict[str, np.ndarray] = {}
        self.pattern_group_sizes_: dict[str, int] = {}
        self.matched_pattern_groups_: int = 0
        self.conflicting_pattern_groups_: int = 0

    def _create_base_model(self) -> LogisticRegression:
        penalty = str(self.base_config.get("penalty", "l2"))
        l1_ratio_by_penalty = {"l2": 0.0, "l1": 1.0}
        if penalty == "elasticnet":
            l1_ratio = float(self.base_config.get("l1_ratio", 0.5))
        elif penalty in l1_ratio_by_penalty:
            l1_ratio = l1_ratio_by_penalty[penalty]
        else:
            raise ValueError("base_model.penalty는 l1, l2, elasticnet만 지원합니다.")
        return LogisticRegression(
            C=float(self.base_config.get("C", 1.0)),
            class_weight=self.base_config.get("class_weight", "balanced"),
            solver=self.base_config.get("solver", "saga"),
            l1_ratio=l1_ratio,
            max_iter=int(self.base_config.get("max_iter", 3000)),
            tol=float(self.base_config.get("tol", 1e-4)),
            random_state=self.seed,
        )

    def _split_features(self, features) -> tuple[pd.DataFrame, pd.Series]:
        if not isinstance(features, pd.DataFrame):
            raise TypeError(
                "pattern_posterior_classifier에는 패턴 키가 포함된 pandas DataFrame이 필요합니다."
            )
        if self.pattern_key_column not in features.columns:
            raise ValueError(
                f"패턴 키 피처가 없습니다: {self.pattern_key_column}. "
                "preprocessing.include_raw_pattern_key=true를 설정하세요."
            )
        pattern_keys = features[self.pattern_key_column].astype("string")
        model_features = features.drop(columns=[self.pattern_key_column])
        return model_features, pattern_keys

    def fit(self, features, labels):
        """기본 분류기와 train-only 패턴 posterior lookup을 학습합니다."""
        model_features, pattern_keys = self._split_features(features)
        encoded_labels = np.asarray(labels)
        self.base_model = self._create_base_model()
        self.base_model.fit(model_features, encoded_labels)
        self.classes_ = np.asarray(self.base_model.classes_)
        class_to_column = {
            int(class_value): column
            for column, class_value in enumerate(self.classes_)
        }

        class_counts = np.asarray([
            np.count_nonzero(encoded_labels == class_value)
            for class_value in self.classes_
        ], dtype="float64")
        global_prior = class_counts / class_counts.sum()
        pattern_frame = pd.DataFrame({
            "pattern_key": pattern_keys.to_numpy(),
            "label": encoded_labels,
        })
        self.pattern_posteriors_ = {}
        self.pattern_group_sizes_ = {}
        self.conflicting_pattern_groups_ = 0

        for pattern_key, group in pattern_frame.groupby("pattern_key", sort=False):
            group_size = len(group)
            if group_size < self.minimum_group_size:
                continue
            counts = np.zeros(len(self.classes_), dtype="float64")
            for class_value, count in group["label"].value_counts().items():
                counts[class_to_column[int(class_value)]] = float(count)
            posterior = (
                counts + self.smoothing_alpha * global_prior
            ) / (group_size + self.smoothing_alpha)
            key = str(pattern_key)
            self.pattern_posteriors_[key] = posterior
            self.pattern_group_sizes_[key] = group_size
            if np.count_nonzero(counts) > 1:
                self.conflicting_pattern_groups_ += 1

        self.matched_pattern_groups_ = len(self.pattern_posteriors_)
        return self

    def predict_proba(self, features) -> np.ndarray:
        """관찰된 패턴만 고정 posterior와 기본 확률을 혼합합니다."""
        if self.base_model is None or self.classes_ is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        model_features, pattern_keys = self._split_features(features)
        probabilities = np.asarray(
            self.base_model.predict_proba(model_features), dtype="float64"
        )
        for row, pattern_key in enumerate(pattern_keys):
            posterior = self.pattern_posteriors_.get(str(pattern_key))
            if posterior is None:
                continue
            probabilities[row] = (
                (1.0 - self.posterior_blend_weight) * probabilities[row]
                + self.posterior_blend_weight * posterior
            )
        return probabilities

    def predict(self, features) -> np.ndarray:
        """혼합 확률이 가장 큰 원본 클래스를 반환합니다."""
        if self.classes_ is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        return self.classes_[self.predict_proba(features).argmax(axis=1)]

    def summary(self) -> dict[str, int]:
        return {
            "matched_pattern_groups": self.matched_pattern_groups_,
            "conflicting_pattern_groups": self.conflicting_pattern_groups_,
            "matched_training_samples": sum(self.pattern_group_sizes_.values()),
        }


def create_model(model_config: dict, seed: int) -> PatternPosteriorClassifier:
    """설정으로 train-only 패턴 posterior 분류기를 생성합니다."""
    return PatternPosteriorClassifier(model_config, seed)
