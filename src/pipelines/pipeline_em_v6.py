"""단백질 변이 결과 유형을 수치화하는 EM 실험 버전 6입니다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipelines.base import PreprocessingPipeline


def create_mutation_presence_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """v6의 최소 빈도 계산을 위해 WT/변이 이진 행렬을 만듭니다."""
    mutation_columns = {
        column: features[column].astype("string").str.strip().str.upper()
        .ne("WT").fillna(False).astype("int8")
        for column in features.columns
    }
    return pd.DataFrame(mutation_columns, index=features.index)


def select_genes_by_mutation_count(
    features: pd.DataFrame,
    minimum_count: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    """v6 학습 데이터에서 최소 변이 횟수를 만족하는 유전자를 선택합니다."""
    mutation_counts = (
        create_mutation_presence_matrix(features).sum(axis=0).astype(int).to_dict()
    )
    selected = [column for column in features.columns if mutation_counts[column] >= minimum_count]
    dropped = [column for column in features.columns if mutation_counts[column] < minimum_count]
    return selected, dropped, mutation_counts


class EMV6PreprocessingPipeline(PreprocessingPipeline):
    """유전자별 변이를 기능 결과 그룹과 표본별 구성비로 표현합니다.

    심각도 코드는 WT=0, 동의치환=1, 미스센스/기타=2,
    in-frame/복합변이=3, 조기종결/frameshift=4입니다.
    """

    name = "em_v6"
    consequence_names = ("synonymous", "missense", "inframe_complex", "truncating")

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
        self.steps = (
            "최소 변이 빈도 필터",
            "단백질 변이 결과 심각도 인코딩",
            "표본별 변이 결과 구성 피처 추가",
        )

    @staticmethod
    def _encode_consequence(values: pd.Series) -> pd.Series:
        normalized = values.astype("string").str.strip().str.upper()
        mutated = normalized.ne("WT").fillna(False)
        severity = pd.Series(0, index=values.index, dtype="int8")
        severity.loc[mutated] = 2

        synonymous = normalized.str.fullmatch(r"([A-Z])\d+\1", na=False)
        structural = normalized.str.contains(r"DEL|INS|DUP|>", regex=True, na=False)
        truncating = normalized.str.contains(r"FS|\*|TER", regex=True, na=False)
        severity.loc[synonymous] = 1
        severity.loc[structural] = 3
        severity.loc[truncating] = 4
        return severity

    def _build_features(self, features: pd.DataFrame) -> pd.DataFrame:
        severity_columns = {
            column: self._encode_consequence(features[column])
            for column in self.selected_gene_columns
        }
        engineered = pd.DataFrame(severity_columns, index=features.index)
        burden = engineered.ne(0).sum(axis=1).astype("float32")
        denominator = burden.clip(lower=1.0)

        engineered["mutation_burden_log1p"] = np.log1p(burden).astype("float32")
        for severity, name in enumerate(self.consequence_names, start=1):
            count = engineered.eq(severity).sum(axis=1).astype("float32")
            engineered[f"consequence_count_{name}"] = count
            engineered[f"consequence_ratio_{name}"] = (count / denominator).astype(
                "float32"
            )
        return engineered

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "EMV6PreprocessingPipeline":
        (
            self.selected_gene_columns,
            self.dropped_rare_columns,
            self.mutation_counts_,
        ) = select_genes_by_mutation_count(features, self.min_mutation_count)
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
        summary["consequence_summary_features"] = 1 + 2 * len(self.consequence_names)
        return summary
