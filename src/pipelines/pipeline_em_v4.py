"""전체 유전자를 이진 변이 행렬로 표현하는 EM 실험 버전 4입니다."""

from __future__ import annotations

import pandas as pd

from src.pipelines.base import PreprocessingPipeline


def create_mutation_presence_matrix(
    features: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """v4 입력을 WT=0, 변이=1인 이진 행렬로 변환합니다."""
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


def select_genes_by_mutation_count(
    features: pd.DataFrame,
    minimum_count: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    """v4 학습 데이터에서 최소 변이 횟수를 만족하는 유전자를 선택합니다."""
    mutation_counts = (
        create_mutation_presence_matrix(features).sum(axis=0).astype(int).to_dict()
    )
    selected = [column for column in features.columns if mutation_counts[column] >= minimum_count]
    dropped = [column for column in features.columns if mutation_counts[column] < minimum_count]
    return selected, dropped, mutation_counts


class EMV4PreprocessingPipeline(PreprocessingPipeline):
    """빈도 필터 후 원본 문자열을 WT=0, 변이=1로 대체합니다."""

    name = "em_v4"

    def __init__(self, min_mutation_count: int = 5, **parameters: object) -> None:
        super().__init__(**parameters)
        if (
            isinstance(min_mutation_count, bool)
            or not isinstance(min_mutation_count, int)
            or min_mutation_count < 1
        ):
            raise ValueError("min_mutation_count는 1 이상의 정수여야 합니다.")

        self.min_mutation_count = min_mutation_count
        self.selected_gene_columns: list[str] = []
        self.dropped_rare_columns: list[str] = []
        self.mutation_counts_: dict[str, int] = {}
        self.steps = ("최소 변이 빈도 필터", "전체 유전자 이진 변이 표현")

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "EMV4PreprocessingPipeline":
        (
            self.selected_gene_columns,
            self.dropped_rare_columns,
            self.mutation_counts_,
        ) = select_genes_by_mutation_count(features, self.min_mutation_count)
        if not self.selected_gene_columns:
            raise ValueError("최소 변이 빈도를 만족하는 유전자 피처가 없습니다.")

        mutation_matrix = create_mutation_presence_matrix(
            features,
            self.selected_gene_columns,
        )
        super().fit(mutation_matrix, labels)
        print(f"[{self.name}] 희귀 변이로 삭제된 피처: {len(self.dropped_rare_columns)}개")
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        mutation_matrix = create_mutation_presence_matrix(
            features,
            self.selected_gene_columns,
        )
        return super().transform(mutation_matrix).astype("float32")

    def summary(self) -> dict[str, int]:
        summary = super().summary()
        summary["dropped_rare_features"] = len(self.dropped_rare_columns)
        summary["binary_gene_features"] = len(self.feature_columns)
        return summary
