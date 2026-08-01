"""변이 부담(v5)과 기능 결과(v6)를 중복 없이 결합한 EM 버전 10입니다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipelines.base import PreprocessingPipeline


def create_mutation_presence_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """v10의 최소 빈도 계산을 위한 WT/변이 이진 행렬입니다."""
    return pd.DataFrame(
        {
            column: features[column].astype("string").str.strip().str.upper()
            .ne("WT").fillna(False).astype("int8")
            for column in features.columns
        },
        index=features.index,
    )


def select_genes_by_mutation_count(
    features: pd.DataFrame,
    minimum_count: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    """v10 학습 데이터에서 최소 변이 횟수를 만족하는 유전자를 선택합니다."""
    mutation_counts = (
        create_mutation_presence_matrix(features).sum(axis=0).astype(int).to_dict()
    )
    selected = [column for column in features.columns if mutation_counts[column] >= minimum_count]
    dropped = [column for column in features.columns if mutation_counts[column] < minimum_count]
    return selected, dropped, mutation_counts


def create_consequence_severity_matrix(
    features: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """v10 유전자 변이를 WT=0부터 truncating=4까지로 인코딩합니다."""
    missing_columns = set(columns) - set(features.columns)
    if missing_columns:
        raise ValueError(f"결과 유형 생성에 필요한 피처가 없습니다: {sorted(missing_columns)}")
    severity_columns: dict[str, pd.Series] = {}
    for column in columns:
        normalized = features[column].astype("string").str.strip().str.upper()
        severity = pd.Series(0, index=features.index, dtype="int8")
        severity.loc[normalized.ne("WT").fillna(False)] = 2
        severity.loc[normalized.str.fullmatch(r"([A-Z])\d+\1", na=False)] = 1
        severity.loc[normalized.str.contains(r"DEL|INS|DUP|>", regex=True, na=False)] = 3
        severity.loc[normalized.str.contains(r"FS|\*|TER", regex=True, na=False)] = 4
        severity_columns[column] = severity
    return pd.DataFrame(severity_columns, index=features.index)


def add_mutation_burden_features(
    engineered: pd.DataFrame,
    mutation_matrix: pd.DataFrame,
    denominator_gene_count: int,
) -> pd.DataFrame:
    """v10의 변이 부담 total/log1p/rate를 한 번만 추가합니다."""
    burden = mutation_matrix.sum(axis=1).astype("float32")
    engineered["mutation_burden_total"] = burden
    engineered["mutation_burden_log1p"] = np.log1p(burden).astype("float32")
    engineered["mutation_burden_rate"] = (burden / max(denominator_gene_count, 1)).astype("float32")
    return engineered


def add_consequence_summary_features(
    engineered: pd.DataFrame,
    severity_matrix: pd.DataFrame,
    include_burden_log1p: bool,
) -> pd.DataFrame:
    """v10의 결과 유형별 개수와 비율을 추가합니다."""
    burden = severity_matrix.ne(0).sum(axis=1).astype("float32")
    denominator = burden.clip(lower=1.0)
    if include_burden_log1p and "mutation_burden_log1p" not in engineered:
        engineered["mutation_burden_log1p"] = np.log1p(burden).astype("float32")
    for severity, name in enumerate(
        ("synonymous", "missense", "inframe_complex", "truncating"), start=1
    ):
        count = severity_matrix.eq(severity).sum(axis=1).astype("float32")
        engineered[f"consequence_count_{name}"] = count
        engineered[f"consequence_ratio_{name}"] = (count / denominator).astype("float32")
    return engineered


class EMV10PreprocessingPipeline(PreprocessingPipeline):
    """이진 피처 대신 더 풍부한 severity를 유지하고 burden은 한 번만 추가합니다."""

    name = "em_v10"

    def __init__(self, min_mutation_count: int = 5, **parameters: object) -> None:
        super().__init__(**parameters)
        if isinstance(min_mutation_count, bool) or not isinstance(min_mutation_count, int) or min_mutation_count < 1:
            raise ValueError("min_mutation_count는 1 이상의 정수여야 합니다.")
        self.min_mutation_count = min_mutation_count
        self.input_gene_columns: list[str] = []
        self.selected_gene_columns: list[str] = []
        self.dropped_rare_columns: list[str] = []
        self.mutation_counts_: dict[str, int] = {}
        self.steps = ("최소 변이 빈도 필터", "기능 결과 심각도", "변이 부담", "기능 결과 구성")

    def _build_features(self, features: pd.DataFrame) -> pd.DataFrame:
        severity_all = create_consequence_severity_matrix(features, self.input_gene_columns)
        severity = severity_all[self.selected_gene_columns]
        mutation_all = severity_all.ne(0).astype("int8")
        engineered = severity.copy()
        add_mutation_burden_features(engineered, mutation_all, len(self.input_gene_columns))
        add_consequence_summary_features(engineered, severity, include_burden_log1p=False)
        return engineered

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "EMV10PreprocessingPipeline":
        self.input_gene_columns = features.columns.tolist()
        self.selected_gene_columns, self.dropped_rare_columns, self.mutation_counts_ = (
            select_genes_by_mutation_count(features, self.min_mutation_count)
        )
        super().fit(self._build_features(features), labels)
        print(f"[{self.name}]")
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return super().transform(self._build_features(features)).astype("float32")

    def summary(self) -> dict[str, int]:
        summary = super().summary()
        summary.update({"dropped_rare_features": len(self.dropped_rare_columns), "burden_features": 3, "consequence_features": 8})
        return summary
