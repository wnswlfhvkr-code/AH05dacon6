"""혼동이 큰 암종 쌍을 이진 전문가로 보정하는 분류 모델입니다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import softmax
from sklearn.svm import LinearSVC


@dataclass
class _FittedPairSpecialist:
    """학습된 암종 쌍 전문가와 혼합 설정입니다."""

    left_class: int
    right_class: int
    left_column: int
    right_column: int
    blend_weight: float
    minimum_activation_probability: float
    model: LinearSVC


class PairSpecialistClassifier:
    """전체 다중분류 점수에서 지정된 암종 쌍의 경계만 보정합니다.

    먼저 전체 클래스용 LinearSVC를 학습합니다. 예측할 때 지정된 두 암종이
    기본 모델의 상위 2개 후보이고 두 클래스의 확률 질량이 임계값 이상이면,
    해당 두 클래스 표본만으로 학습한 이진 LinearSVC의 margin을 혼합합니다.
    최종 출력 클래스는 기존 다중분류 클래스 중 하나이며 레이블을 병합하거나
    하위 암종으로 분해하지 않습니다.
    """

    def __init__(self, model_config: dict, seed: int) -> None:
        self.seed = seed
        self.class_names = [str(value) for value in model_config.get("class_names", [])]
        if not bool(model_config.get("preserve_original_labels", True)):
            raise ValueError(
                "pair specialist는 원본 다중분류 레이블을 유지하는 설정만 지원합니다."
            )
        self.activation = str(model_config.get("activation", "top2_pair_match"))
        if self.activation != "top2_pair_match":
            raise ValueError(
                "pair specialist activation은 'top2_pair_match'만 지원합니다."
            )

        self.temperature = float(model_config.get("temperature", 1.0))
        if self.temperature <= 0:
            raise ValueError("temperature는 0보다 커야 합니다.")

        self.base_config = dict(model_config.get("base_model", {}))
        base_name = self.base_config.pop("name", "linear_svc")
        if base_name != "linear_svc":
            raise ValueError("현재 base_model은 linear_svc만 지원합니다.")

        default_specialist_config = dict(self.base_config)
        default_specialist_config.update(model_config.get("specialist_model", {}))
        self.specialist_config = default_specialist_config
        self.pair_configs = list(model_config.get("pairs", []))
        if not self.pair_configs:
            raise ValueError("pair specialist에 하나 이상의 pairs 설정이 필요합니다.")

        self.base_model: LinearSVC | None = None
        self.classes_: np.ndarray | None = None
        self.specialists_: list[_FittedPairSpecialist] = []
        self.skipped_pairs_: list[dict[str, object]] = []

    def _create_linear_svc(self, config: dict) -> LinearSVC:
        return LinearSVC(
            C=float(config.get("C", 0.2)),
            class_weight=config.get("class_weight", "balanced"),
            max_iter=int(config.get("max_iter", 20_000)),
            tol=float(config.get("tol", 1e-4)),
            random_state=self.seed,
            dual=config.get("dual", "auto"),
        )

    def _resolve_pair(self, labels: list[object]) -> tuple[int, int]:
        if len(labels) != 2 or labels[0] == labels[1]:
            raise ValueError("각 pairs.labels에는 서로 다른 레이블 2개가 필요합니다.")

        if all(isinstance(label, (int, np.integer)) for label in labels):
            return int(labels[0]), int(labels[1])

        if not self.class_names:
            raise ValueError(
                "문자열 pairs.labels를 사용하려면 전처리 LabelEncoder의 "
                "class_names가 모델에 전달되어야 합니다."
            )
        class_to_value = {name: index for index, name in enumerate(self.class_names)}
        missing = [str(label) for label in labels if str(label) not in class_to_value]
        if missing:
            raise ValueError(f"학습 레이블에 없는 pair specialist 클래스입니다: {missing}")
        return class_to_value[str(labels[0])], class_to_value[str(labels[1])]

    def fit(self, features, labels):
        """전체 모델과 fold-train 내부의 이진 전문가를 학습합니다."""
        encoded_labels = np.asarray(labels)
        self.base_model = self._create_linear_svc(self.base_config)
        self.base_model.fit(features, encoded_labels)
        self.classes_ = np.asarray(self.base_model.classes_)
        class_to_column = {
            int(class_value): column
            for column, class_value in enumerate(self.classes_)
        }
        self.specialists_ = []
        self.skipped_pairs_ = []

        for pair_config in self.pair_configs:
            pair_labels = list(pair_config.get("labels", []))
            left_class, right_class = self._resolve_pair(pair_labels)
            if left_class not in class_to_column or right_class not in class_to_column:
                self.skipped_pairs_.append({
                    "labels": pair_labels,
                    "reason": "fold-train에 두 클래스가 모두 존재하지 않음",
                })
                continue

            blend_weight = float(pair_config.get("blend_weight", 0.25))
            minimum_probability = float(
                pair_config.get("minimum_activation_probability", 0.0)
            )
            if not 0.0 <= blend_weight <= 1.0:
                raise ValueError("blend_weight는 0 이상 1 이하여야 합니다.")
            if not 0.0 <= minimum_probability <= 1.0:
                raise ValueError(
                    "minimum_activation_probability는 0 이상 1 이하여야 합니다."
                )

            pair_mask = np.isin(encoded_labels, [left_class, right_class])
            pair_targets = encoded_labels[pair_mask]
            if np.unique(pair_targets).size != 2:
                self.skipped_pairs_.append({
                    "labels": pair_labels,
                    "reason": "이진 전문가 학습 표본 부족",
                })
                continue

            specialist = self._create_linear_svc(self.specialist_config)
            specialist.fit(features[pair_mask], pair_targets)
            self.specialists_.append(_FittedPairSpecialist(
                left_class=left_class,
                right_class=right_class,
                left_column=class_to_column[left_class],
                right_column=class_to_column[right_class],
                blend_weight=blend_weight,
                minimum_activation_probability=minimum_probability,
                model=specialist,
            ))
        return self

    def _base_scores(self, features) -> np.ndarray:
        if self.base_model is None or self.classes_ is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        scores = np.asarray(self.base_model.decision_function(features), dtype="float64")
        if scores.ndim == 1:
            scores = np.column_stack((-scores, scores))
        return scores

    def decision_function(self, features) -> np.ndarray:
        """활성화 조건을 만족하는 행의 암종 쌍 margin을 보정합니다."""
        scores = self._base_scores(features)
        base_scores = scores.copy()
        base_probabilities = softmax(base_scores / self.temperature, axis=1)
        top_two = np.argpartition(base_scores, -2, axis=1)[:, -2:]

        for specialist in self.specialists_:
            pair_columns = np.asarray(
                [specialist.left_column, specialist.right_column]
            )
            top_two_match = np.all(
                np.sort(top_two, axis=1) == np.sort(pair_columns), axis=1
            )
            pair_probability = base_probabilities[:, pair_columns].sum(axis=1)
            active = top_two_match & (
                pair_probability >= specialist.minimum_activation_probability
            )
            if not np.any(active):
                continue

            expert_margin = np.asarray(
                specialist.model.decision_function(features[active]),
                dtype="float64",
            ).reshape(-1)
            if int(specialist.model.classes_[1]) != specialist.right_class:
                expert_margin = -expert_margin

            left_column = specialist.left_column
            right_column = specialist.right_column
            base_margin = (
                base_scores[active, right_column]
                - base_scores[active, left_column]
            )
            mixed_margin = (
                (1.0 - specialist.blend_weight) * base_margin
                + specialist.blend_weight * expert_margin
            )
            pair_center = (
                base_scores[active, left_column]
                + base_scores[active, right_column]
            ) / 2.0
            scores[active, left_column] = pair_center - mixed_margin / 2.0
            scores[active, right_column] = pair_center + mixed_margin / 2.0
        return scores

    def predict_proba(self, features) -> np.ndarray:
        """혼합 점수를 softmax 확률로 변환합니다."""
        return softmax(self.decision_function(features) / self.temperature, axis=1)

    def predict(self, features) -> np.ndarray:
        """보정된 점수가 가장 높은 원본 클래스를 반환합니다."""
        if self.classes_ is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        indices = self.decision_function(features).argmax(axis=1)
        return self.classes_[indices]

    def summary(self) -> dict[str, int]:
        return {
            "configured_pairs": len(self.pair_configs),
            "fitted_pair_specialists": len(self.specialists_),
            "skipped_pair_specialists": len(self.skipped_pairs_),
        }


def create_model(model_config: dict, seed: int) -> PairSpecialistClassifier:
    """설정으로 암종 쌍 전문가 분류기를 생성합니다."""
    return PairSpecialistClassifier(model_config, seed)
