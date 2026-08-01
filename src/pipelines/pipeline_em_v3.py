"""최소 변이 빈도 필터를 적용하는 EM 실험 버전 3 파이프라인입니다."""

from __future__ import annotations

import pandas as pd

from src.pipelines.base import PreprocessingPipeline


class EMV3PreprocessingPipeline(PreprocessingPipeline):
    """학습 데이터에서 변이가 너무 적은 유전자 피처를 제거합니다.

    변이 빈도는 ``fit``에 전달된 데이터에서만 계산하므로 교차검증에서
    검증 fold의 정보를 사용하지 않습니다. WT가 아닌 값은 변이로 계산하며,
    빈도 계산용 이진 피처는 모델 입력에 추가하지 않습니다.
    """

    name = "em_v3"

    def __init__(
        self,
        min_mutation_count: int = 5,
        min_class_mutation_count: int | None = None,
        **parameters: object,
    ) -> None:
        super().__init__(**parameters)
        if (
            isinstance(min_mutation_count, bool)
            or not isinstance(min_mutation_count, int)
            or min_mutation_count < 1
        ):
            raise ValueError("min_mutation_count는 1 이상의 정수여야 합니다.")
        if (
            min_class_mutation_count is not None
            and (
                isinstance(min_class_mutation_count, bool)
                or not isinstance(min_class_mutation_count, int)
                or min_class_mutation_count < 1
            )
        ):
            raise ValueError("min_class_mutation_count는 None 또는 1 이상의 정수여야 합니다.")

        self.min_mutation_count = min_mutation_count
        self.min_class_mutation_count = min_class_mutation_count
        self.mutation_counts_: dict[str, int] = {}
        self.max_class_mutation_counts_: dict[str, int] = {}
        self.dropped_rare_columns: list[str] = []
        self.steps = ("최소 변이 빈도 피처 제거", *self.steps)

    @staticmethod
    def _count_mutations(
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> tuple[dict[str, int], dict[str, int]]:
        """전체 및 클래스별 최대 변이 환자 수를 계산합니다."""
        mutation_counts: dict[str, int] = {}
        max_class_mutation_counts: dict[str, int] = {}
        aligned_labels = labels.reindex(features.index)
        for column in features.columns:
            normalized = features[column].astype("string").str.strip().str.upper()
            mutated = normalized.ne("WT").fillna(False)
            mutation_counts[column] = int(mutated.sum())
            class_counts = mutated.groupby(aligned_labels, observed=True).sum()
            max_class_mutation_counts[column] = int(class_counts.max())
        return mutation_counts, max_class_mutation_counts

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "EMV3PreprocessingPipeline":
        """학습 데이터의 변이 횟수로 유지할 피처를 결정합니다."""
        (
            self.mutation_counts_,
            self.max_class_mutation_counts_,
        ) = self._count_mutations(features, labels)
        self.dropped_rare_columns = [
            column
            for column in features.columns
            if self.mutation_counts_[column] < self.min_mutation_count
            and (
                self.min_class_mutation_count is None
                or self.max_class_mutation_counts_[column]
                < self.min_class_mutation_count
            )
        ]

        filtered = features.drop(columns=self.dropped_rare_columns)
        super().fit(filtered, labels)
        self._print_dropped_feature_count()
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """학습 단계에서 결정한 희귀 유전자 피처를 동일하게 제거합니다."""
        filtered = features.drop(
            columns=self.dropped_rare_columns,
            errors="ignore",
        )
        return super().transform(filtered)

    def _print_dropped_feature_count(self) -> None:
        rare_count = len(self.dropped_rare_columns)
        constant_count = len(self.dropped_constant_columns)
        total_count = rare_count + constant_count

        print(
            f"[{self.name}] 최소 변이 {self.min_mutation_count}건 미만으로 "
            f"삭제된 피처: {rare_count}개"
        )
        if self.min_class_mutation_count is not None:
            print(
                f"[{self.name}] 클래스 내 {self.min_class_mutation_count}건 이상 변이 피처는 "
                "삭제 대상에서 보호"
            )
        print(f"[{self.name}] 상수값으로 삭제된 피처: {constant_count}개")
        print(f"[{self.name}] 전체 삭제된 피처: {total_count}개")

    def summary(self) -> dict[str, int]:
        summary = super().summary()
        rare_count = len(self.dropped_rare_columns)
        summary["dropped_rare_features"] = rare_count
        summary["total_dropped_features"] = (
            rare_count + summary["dropped_constant_features"]
        )
        return summary
