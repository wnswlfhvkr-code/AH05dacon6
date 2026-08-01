"""암종별 특징 유전자 그룹 점수를 만드는 EM 실험 버전 9입니다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipelines.base import PreprocessingPipeline


def create_mutation_presence_matrix(
    features: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """v9 입력을 WT=0, 변이=1인 signature 학습 행렬로 변환합니다."""
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
    """v9 학습 데이터에서 최소 변이 횟수를 만족하는 유전자를 선택합니다."""
    mutation_counts = (
        create_mutation_presence_matrix(features).sum(axis=0).astype(int).to_dict()
    )
    selected = [column for column in features.columns if mutation_counts[column] >= minimum_count]
    dropped = [column for column in features.columns if mutation_counts[column] < minimum_count]
    return selected, dropped, mutation_counts


class EMV9PreprocessingPipeline(PreprocessingPipeline):
    """학습 fold에서 암종별 고특이 유전자 그룹과 가중 점수를 학습합니다."""

    name = "em_v9"

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
        for value, name in (
            (min_mutation_count, "min_mutation_count"),
            (top_genes_per_class, "top_genes_per_class"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name}는 1 이상의 정수여야 합니다.")
        for value, name in (
            (smoothing, "smoothing"),
            (max_log2_odds, "max_log2_odds"),
            (shrinkage, "shrinkage"),
        ):
            if value <= 0:
                raise ValueError(f"{name}는 0보다 커야 합니다.")

        self.min_mutation_count = min_mutation_count
        self.top_genes_per_class = top_genes_per_class
        self.smoothing = float(smoothing)
        self.max_log2_odds = float(max_log2_odds)
        self.shrinkage = float(shrinkage)
        self.selected_gene_columns: list[str] = []
        self.dropped_rare_columns: list[str] = []
        self.mutation_counts_: dict[str, int] = {}
        self.class_names: list[str] = []
        self.class_gene_weights_: dict[str, dict[str, float]] = {}
        self.steps = (
            "최소 변이 빈도 필터",
            "학습 fold 암종별 log2 odds 유전자 그룹 선택",
            "암종별 가중 변이 점수 생성",
        )

    def _mutation_matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        return create_mutation_presence_matrix(
            features,
            self.selected_gene_columns,
        )

    def _build_signature_features(self, features: pd.DataFrame) -> pd.DataFrame:
        matrix = self._mutation_matrix(features)
        signature = pd.DataFrame(index=features.index)
        for class_name in self.class_names:
            weights = self.class_gene_weights_[class_name]
            genes = list(weights)
            class_matrix = matrix[genes].astype("float32")
            weight_vector = np.array([weights[gene] for gene in genes], dtype="float32")
            signature[f"signature_{class_name}_weighted"] = class_matrix.to_numpy().dot(
                weight_vector
            )
            signature[f"signature_{class_name}_match_count"] = class_matrix.sum(axis=1)

        burden = matrix.sum(axis=1).astype("float32")
        signature["mutation_burden_log1p"] = np.log1p(burden).astype("float32")
        return signature.astype("float32")

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "EMV9PreprocessingPipeline":
        (
            self.selected_gene_columns,
            self.dropped_rare_columns,
            self.mutation_counts_,
        ) = select_genes_by_mutation_count(features, self.min_mutation_count)
        if not self.selected_gene_columns:
            raise ValueError("최소 변이 빈도를 만족하는 유전자 피처가 없습니다.")

        matrix = self._mutation_matrix(features)
        aligned_labels = labels.reindex(features.index).astype("string")
        self.class_names = sorted(aligned_labels.dropna().unique().tolist())
        self.class_gene_weights_ = {}

        for class_name in self.class_names:
            in_class = aligned_labels.eq(class_name).fillna(False)
            class_size = int(in_class.sum())
            other_size = len(in_class) - class_size
            class_mutations = matrix.loc[in_class].sum(axis=0).astype("float64")
            other_mutations = matrix.loc[~in_class].sum(axis=0).astype("float64")

            class_odds = (class_mutations + self.smoothing) / (
                class_size - class_mutations + self.smoothing
            )
            other_odds = (other_mutations + self.smoothing) / (
                other_size - other_mutations + self.smoothing
            )
            log2_odds = np.log2(class_odds / other_odds).clip(
                lower=0.0,
                upper=self.max_log2_odds,
            )
            total_mutations = class_mutations + other_mutations
            reliability = np.sqrt(
                total_mutations / (total_mutations + self.shrinkage)
            )
            adjusted_score = log2_odds * reliability
            top_scores = adjusted_score.nlargest(self.top_genes_per_class)
            top_scores = top_scores[top_scores.gt(0)]
            if top_scores.empty:
                raise ValueError(f"{class_name}의 특징 유전자 그룹을 만들 수 없습니다.")
            self.class_gene_weights_[class_name] = {
                gene: float(score) for gene, score in top_scores.items()
            }

        super().fit(self._build_signature_features(features), labels)
        print(
            f"[{self.name}] {len(self.class_names)}개 암종별 특징 유전자 그룹 생성"
        )
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return super().transform(self._build_signature_features(features)).astype(
            "float32"
        )

    def summary(self) -> dict[str, int]:
        summary = super().summary()
        summary["dropped_rare_features"] = len(self.dropped_rare_columns)
        summary["class_signature_groups"] = len(self.class_names)
        summary["signature_features"] = 2 * len(self.class_names) + 1
        return summary
