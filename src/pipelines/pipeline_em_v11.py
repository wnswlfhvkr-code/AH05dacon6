"""변이 부담(v5)과 암종 signature(v9)를 결합한 EM 버전 11입니다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipelines.base import PreprocessingPipeline


def create_mutation_presence_matrix(
    features: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """v11 입력을 WT=0, 변이=1인 이진 행렬로 변환합니다."""
    selected_columns = features.columns.tolist() if columns is None else columns
    missing_columns = set(selected_columns) - set(features.columns)
    if missing_columns:
        raise ValueError(f"변이 행렬 생성에 필요한 피처가 없습니다: {sorted(missing_columns)}")
    return pd.DataFrame(
        {
            column: features[column].astype("string").str.strip().str.upper()
            .ne("WT").fillna(False).astype("int8")
            for column in selected_columns
        },
        index=features.index,
    )


def select_genes_by_mutation_count(
    features: pd.DataFrame,
    minimum_count: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    """v11 학습 데이터에서 최소 변이 횟수를 만족하는 유전자를 선택합니다."""
    mutation_counts = (
        create_mutation_presence_matrix(features).sum(axis=0).astype(int).to_dict()
    )
    selected = [column for column in features.columns if mutation_counts[column] >= minimum_count]
    dropped = [column for column in features.columns if mutation_counts[column] < minimum_count]
    return selected, dropped, mutation_counts


def add_mutation_burden_features(
    engineered: pd.DataFrame,
    mutation_matrix: pd.DataFrame,
    denominator_gene_count: int,
) -> pd.DataFrame:
    """v11의 변이 부담 total/log1p/rate를 추가합니다."""
    burden = mutation_matrix.sum(axis=1).astype("float32")
    engineered["mutation_burden_total"] = burden
    engineered["mutation_burden_log1p"] = np.log1p(burden).astype("float32")
    engineered["mutation_burden_rate"] = (burden / max(denominator_gene_count, 1)).astype("float32")
    return engineered


def learn_class_gene_weights(
    mutation_matrix: pd.DataFrame,
    labels: pd.Series,
    top_genes_per_class: int,
    smoothing: float,
    max_log2_odds: float,
    shrinkage: float,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """v11 학습 fold에서 암종별 안정화 log2 odds 가중치를 학습합니다."""
    aligned_labels = labels.reindex(mutation_matrix.index).astype("string")
    class_names = sorted(aligned_labels.dropna().unique().tolist())
    class_gene_weights: dict[str, dict[str, float]] = {}
    for class_name in class_names:
        in_class = aligned_labels.eq(class_name).fillna(False)
        class_size = int(in_class.sum())
        other_size = len(in_class) - class_size
        class_mutations = mutation_matrix.loc[in_class].sum(axis=0).astype("float64")
        other_mutations = mutation_matrix.loc[~in_class].sum(axis=0).astype("float64")
        class_odds = (class_mutations + smoothing) / (class_size - class_mutations + smoothing)
        other_odds = (other_mutations + smoothing) / (other_size - other_mutations + smoothing)
        log2_odds = np.log2(class_odds / other_odds).clip(lower=0.0, upper=max_log2_odds)
        total_mutations = class_mutations + other_mutations
        reliability = np.sqrt(total_mutations / (total_mutations + shrinkage))
        top_scores = (log2_odds * reliability).nlargest(top_genes_per_class)
        top_scores = top_scores[top_scores.gt(0)]
        if top_scores.empty:
            raise ValueError(f"{class_name}의 특징 유전자 그룹을 만들 수 없습니다.")
        class_gene_weights[class_name] = {
            gene: float(score) for gene, score in top_scores.items()
        }
    return class_names, class_gene_weights


def add_class_signature_features(
    engineered: pd.DataFrame,
    mutation_matrix: pd.DataFrame,
    class_names: list[str],
    class_gene_weights: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """v11의 암종별 weighted/match signature를 추가합니다."""
    for class_name in class_names:
        weights = class_gene_weights[class_name]
        genes = list(weights)
        class_matrix = mutation_matrix[genes].astype("float32")
        weight_vector = np.array([weights[gene] for gene in genes], dtype="float32")
        engineered[f"signature_{class_name}_weighted"] = class_matrix.to_numpy().dot(weight_vector)
        engineered[f"signature_{class_name}_match_count"] = class_matrix.sum(axis=1)
    return engineered


class EMV11PreprocessingPipeline(PreprocessingPipeline):
    """이진 유전자와 burden을 유지하고 signature의 중복 burden은 제외합니다."""

    name = "em_v11"

    def __init__(
        self,
        min_mutation_count: int = 5,
        top_genes_per_class: int = 20,
        smoothing: float = 0.5,
        max_log2_odds: float = 8.0,
        shrinkage: float = 10.0,
        **parameters: object,
    ) -> None:
        super().__init__(**parameters)
        if min_mutation_count < 1 or top_genes_per_class < 1:
            raise ValueError("변이 빈도와 클래스별 유전자 수는 1 이상이어야 합니다.")
        if smoothing <= 0 or max_log2_odds <= 0 or shrinkage <= 0:
            raise ValueError("signature 안정화 파라미터는 0보다 커야 합니다.")
        self.min_mutation_count = min_mutation_count
        self.top_genes_per_class = top_genes_per_class
        self.smoothing = float(smoothing)
        self.max_log2_odds = float(max_log2_odds)
        self.shrinkage = float(shrinkage)
        self.input_gene_columns: list[str] = []
        self.selected_gene_columns: list[str] = []
        self.dropped_rare_columns: list[str] = []
        self.mutation_counts_: dict[str, int] = {}
        self.class_names: list[str] = []
        self.class_gene_weights_: dict[str, dict[str, float]] = {}
        self.steps = ("최소 변이 빈도 필터", "이진 변이", "변이 부담", "암종 signature")

    def _build_features(self, features: pd.DataFrame) -> pd.DataFrame:
        mutation_all = create_mutation_presence_matrix(features, self.input_gene_columns)
        mutation = mutation_all[self.selected_gene_columns]
        engineered = mutation.copy()
        add_mutation_burden_features(engineered, mutation_all, len(self.input_gene_columns))
        add_class_signature_features(engineered, mutation, self.class_names, self.class_gene_weights_)
        return engineered

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "EMV11PreprocessingPipeline":
        self.input_gene_columns = features.columns.tolist()
        self.selected_gene_columns, self.dropped_rare_columns, self.mutation_counts_ = select_genes_by_mutation_count(features, self.min_mutation_count)
        mutation = create_mutation_presence_matrix(features, self.selected_gene_columns)
        self.class_names, self.class_gene_weights_ = learn_class_gene_weights(
            mutation,
            labels,
            self.top_genes_per_class,
            self.smoothing,
            self.max_log2_odds,
            self.shrinkage,
        )
        super().fit(self._build_features(features), labels)        
        print(f"[{self.name}]")
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return super().transform(self._build_features(features)).astype("float32")

    def summary(self) -> dict[str, int]:
        summary = super().summary()
        summary.update({"dropped_rare_features": len(self.dropped_rare_columns), "burden_features": 3, "signature_features": 2 * len(self.class_names)})
        return summary
