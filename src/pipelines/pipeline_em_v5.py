"""이진 유전자 행렬에 종양 변이 부담을 추가하는 EM 실험 버전 5입니다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipelines.base import PreprocessingPipeline


def create_mutation_presence_matrix(
    features: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """v5 입력을 WT=0, 변이=1인 이진 행렬로 변환합니다."""
    selected_columns = features.columns.tolist() if columns is None else columns
    missing_columns = set(selected_columns) - set(features.columns)
    if missing_columns:
        raise ValueError(f"변이 행렬 생성에 필요한 피처가 없습니다: {sorted(missing_columns)}")
    mutation_columns = {
        column: features[column].astype("string").str.strip().str.upper()
        .ne("WT").fillna(False).astype("int8")
        for column in selected_columns
    }
    return pd.DataFrame(mutation_columns, index=features.index)


class EMV5PreprocessingPipeline(PreprocessingPipeline):
    """유전자 변이 여부와 표본별 변이 부담을 함께 표현합니다."""

    name = "em_v5"
    burden_columns = (
        "mutation_burden_total",
        "mutation_burden_log1p",
        "mutation_burden_rate",
    )

    def __init__(self, min_mutation_count: int = 5, **parameters: object) -> None:
        super().__init__(**parameters)
        if (
            isinstance(min_mutation_count, bool)
            or not isinstance(min_mutation_count, int)
            or min_mutation_count < 1
        ):
            raise ValueError("min_mutation_count는 1 이상의 정수여야 합니다.")

        self.min_mutation_count = min_mutation_count
        self.input_gene_columns: list[str] = []
        self.selected_gene_columns: list[str] = []
        self.dropped_rare_columns: list[str] = []
        self.mutation_counts_: dict[str, int] = {}
        self.steps = (
            "최소 변이 빈도 필터",
            "전체 유전자 이진 변이 표현",
            "표본별 변이 부담 피처 추가",
        )

    def _build_features(self, features: pd.DataFrame) -> pd.DataFrame:
        mutation_matrix = create_mutation_presence_matrix(
            features,
            self.input_gene_columns,
        )
        engineered = mutation_matrix[self.selected_gene_columns].copy()
        burden = mutation_matrix.sum(axis=1).astype("float32")
        engineered["mutation_burden_total"] = burden
        engineered["mutation_burden_log1p"] = np.log1p(burden).astype("float32")
        engineered["mutation_burden_rate"] = (
            burden / max(len(self.input_gene_columns), 1)
        ).astype("float32")
        return engineered

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "EMV5PreprocessingPipeline":
        self.input_gene_columns = features.columns.tolist()
        mutation_matrix = create_mutation_presence_matrix(features, self.input_gene_columns)
        self.mutation_counts_ = mutation_matrix.sum(axis=0).astype(int).to_dict()
        self.selected_gene_columns = [
            column for column in self.input_gene_columns
            if self.mutation_counts_[column] >= self.min_mutation_count
        ]
        self.dropped_rare_columns = [
            column for column in self.input_gene_columns
            if self.mutation_counts_[column] < self.min_mutation_count
        ]
        if not self.selected_gene_columns:
            raise ValueError("최소 변이 빈도를 만족하는 유전자 피처가 없습니다.")

        super().fit(self._build_features(features), labels)
        print(f"[{self.name}] 희귀 변이로 삭제된 피처: {len(self.dropped_rare_columns)}개")
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return super().transform(self._build_features(features)).astype("float32")

    def summary(self) -> dict[str, int]:
        summary = super().summary()
        summary["dropped_rare_features"] = len(self.dropped_rare_columns)
        summary["burden_features"] = len(self.burden_columns)
        return summary
