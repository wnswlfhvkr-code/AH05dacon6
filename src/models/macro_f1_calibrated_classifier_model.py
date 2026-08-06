"""Train 내부 OOF로 Macro F1 확률 보정을 선택하는 분류 모델입니다."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


class MacroF1CalibratedClassifier:
    """클래스 사전확률과 temperature를 OOF Macro F1으로 선택합니다.

    각 ``fit`` 호출 안에서 Stratified inner-OOF 확률을 생성하고 보정 후보를
    비교합니다. 선택이 끝난 후 기본 Logistic Regression은 전달된 학습 데이터
    전체에 다시 fit됩니다. validation/test 분포는 보정값 선택에 사용하지 않습니다.
    """

    def __init__(self, model_config: dict, seed: int) -> None:
        self.seed = seed
        self.adjustment_method = str(
            model_config.get("adjustment_method", "class_prior_power")
        )
        if self.adjustment_method != "class_prior_power":
            raise ValueError("adjustment_method는 'class_prior_power'만 지원합니다.")

        self.gamma_candidates = self._validated_candidates(
            model_config.get("gamma_candidates", [-0.25, 0.0, 0.25, 0.5]),
            "gamma_candidates",
            positive_only=False,
        )
        self.temperature_candidates = self._validated_candidates(
            model_config.get("temperature_candidates", [0.85, 1.0, 1.15]),
            "temperature_candidates",
            positive_only=True,
        )
        self.calibration_folds = int(model_config.get("calibration_folds", 3))
        if self.calibration_folds < 2:
            raise ValueError("calibration_folds는 2 이상이어야 합니다.")

        self.base_config = dict(model_config.get("base_model", {}))
        base_name = self.base_config.pop("name", "logistic_regression")
        if base_name != "logistic_regression":
            raise ValueError("현재 base_model은 logistic_regression만 지원합니다.")

        self.base_model: LogisticRegression | None = None
        self.classes_: np.ndarray | None = None
        self.class_prior_: np.ndarray | None = None
        self.selected_gamma_: float = 0.0
        self.selected_temperature_: float = 1.0
        self.selected_oof_macro_f1_: float | None = None
        self.calibration_results_: list[dict[str, float]] = []
        self.effective_calibration_folds_: int = 0

    @staticmethod
    def _validated_candidates(
        values,
        name: str,
        positive_only: bool,
    ) -> list[float]:
        candidates = sorted({float(value) for value in values})
        if not candidates or not np.isfinite(candidates).all():
            raise ValueError(f"{name}에는 유한한 숫자 후보가 필요합니다.")
        if positive_only and any(value <= 0 for value in candidates):
            raise ValueError(f"{name}의 값은 모두 0보다 커야 합니다.")
        return candidates

    def _create_base_model(self) -> LogisticRegression:
        penalty = str(self.base_config.get("penalty", "l2"))
        if penalty == "elasticnet":
            l1_ratio = float(self.base_config.get("l1_ratio", 0.5))
        elif penalty == "l2":
            l1_ratio = 0.0
        elif penalty == "l1":
            l1_ratio = 1.0
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

    @staticmethod
    def _slice_rows(features, indices):
        if hasattr(features, "iloc"):
            return features.iloc[indices]
        return features[indices]

    @staticmethod
    def _adjust_probabilities(
        probabilities: np.ndarray,
        class_prior: np.ndarray,
        gamma: float,
        temperature: float,
    ) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype="float64"), 1e-12, 1.0)
        adjusted = np.power(clipped, 1.0 / temperature)
        adjusted /= np.power(np.clip(class_prior, 1e-12, 1.0), gamma)
        normalizer = adjusted.sum(axis=1, keepdims=True)
        return adjusted / np.clip(normalizer, 1e-12, None)

    def _create_inner_oof_probabilities(
        self,
        features,
        labels: np.ndarray,
    ) -> np.ndarray | None:
        if self.classes_ is None:
            raise RuntimeError("클래스가 초기화되지 않았습니다.")
        class_counts = np.asarray([
            np.count_nonzero(labels == class_value)
            for class_value in self.classes_
        ])
        n_splits = min(self.calibration_folds, int(class_counts.min()))
        self.effective_calibration_folds_ = n_splits
        if n_splits < 2:
            return None

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.seed,
        )
        probabilities = np.zeros(
            (len(labels), len(self.classes_)), dtype="float64"
        )
        class_to_column = {
            int(class_value): column
            for column, class_value in enumerate(self.classes_)
        }
        for train_indices, valid_indices in splitter.split(features, labels):
            inner_model = self._create_base_model()
            inner_model.fit(
                self._slice_rows(features, train_indices), labels[train_indices]
            )
            fold_probabilities = inner_model.predict_proba(
                self._slice_rows(features, valid_indices)
            )
            for inner_column, class_value in enumerate(inner_model.classes_):
                probabilities[
                    valid_indices, class_to_column[int(class_value)]
                ] = fold_probabilities[:, inner_column]
        return probabilities

    def _select_adjustment(
        self,
        labels: np.ndarray,
        oof_probabilities: np.ndarray | None,
    ) -> None:
        self.selected_gamma_ = 0.0
        self.selected_temperature_ = 1.0
        self.selected_oof_macro_f1_ = None
        self.calibration_results_ = []
        if oof_probabilities is None or self.classes_ is None or self.class_prior_ is None:
            return

        best_key: tuple[float, float, float] | None = None
        for gamma in self.gamma_candidates:
            for temperature in self.temperature_candidates:
                adjusted = self._adjust_probabilities(
                    oof_probabilities,
                    self.class_prior_,
                    gamma,
                    temperature,
                )
                predictions = self.classes_[adjusted.argmax(axis=1)]
                score = float(f1_score(labels, predictions, average="macro"))
                self.calibration_results_.append({
                    "gamma": gamma,
                    "temperature": temperature,
                    "macro_f1": score,
                })
                candidate_key = (
                    score,
                    -abs(gamma),
                    -abs(temperature - 1.0),
                )
                if best_key is None or candidate_key > best_key:
                    best_key = candidate_key
                    self.selected_gamma_ = gamma
                    self.selected_temperature_ = temperature
                    self.selected_oof_macro_f1_ = score

    def fit(self, features, labels):
        """inner-OOF로 보정값을 선택한 뒤 기본 모델을 전체 train에 fit합니다."""
        encoded_labels = np.asarray(labels)
        self.classes_ = np.unique(encoded_labels)
        class_counts = np.asarray([
            np.count_nonzero(encoded_labels == class_value)
            for class_value in self.classes_
        ], dtype="float64")
        self.class_prior_ = class_counts / class_counts.sum()
        oof_probabilities = self._create_inner_oof_probabilities(
            features, encoded_labels
        )
        self._select_adjustment(encoded_labels, oof_probabilities)
        self.base_model = self._create_base_model()
        self.base_model.fit(features, encoded_labels)
        print(
            "[macro_f1_calibration] "
            f"inner_folds={self.effective_calibration_folds_}, "
            f"gamma={self.selected_gamma_:.2f}, "
            f"temperature={self.selected_temperature_:.2f}, "
            f"oof_macro_f1={self.selected_oof_macro_f1_}"
        )
        return self

    def predict_proba(self, features) -> np.ndarray:
        """선택된 보정값을 고정해 기본 모델 확률을 변환합니다."""
        if self.base_model is None or self.class_prior_ is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        probabilities = self.base_model.predict_proba(features)
        return self._adjust_probabilities(
            probabilities,
            self.class_prior_,
            self.selected_gamma_,
            self.selected_temperature_,
        )

    def predict(self, features) -> np.ndarray:
        """보정된 확률이 가장 높은 원본 클래스를 반환합니다."""
        if self.classes_ is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        return self.classes_[self.predict_proba(features).argmax(axis=1)]

    def summary(self) -> dict[str, float | int | None]:
        return {
            "calibration_folds": self.effective_calibration_folds_,
            "selected_gamma": self.selected_gamma_,
            "selected_temperature": self.selected_temperature_,
            "selected_oof_macro_f1": self.selected_oof_macro_f1_,
            "candidate_count": len(self.calibration_results_),
        }


def create_model(model_config: dict, seed: int) -> MacroF1CalibratedClassifier:
    """설정으로 Macro F1 보정 분류기를 생성합니다."""
    return MacroF1CalibratedClassifier(model_config, seed)
