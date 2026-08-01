"""비음수 잠재 변이 모듈을 학습하는 EM 실험 버전 8입니다."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import MiniBatchNMF

from src.pipelines.base import PreprocessingPipeline


def create_mutation_presence_matrix(
    features: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """v8 입력을 WT=0, 변이=1인 NMF 입력 행렬로 변환합니다."""
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
    """v8 학습 데이터에서 최소 변이 횟수를 만족하는 유전자를 선택합니다."""
    mutation_counts = (
        create_mutation_presence_matrix(features).sum(axis=0).astype(int).to_dict()
    )
    selected = [column for column in features.columns if mutation_counts[column] >= minimum_count]
    dropped = [column for column in features.columns if mutation_counts[column] < minimum_count]
    return selected, dropped, mutation_counts


class EMV8PreprocessingPipeline(PreprocessingPipeline):
    """희소 유전자 변이 행렬을 데이터 기반의 비음수 변이 모듈로 압축합니다."""

    name = "em_v8"

    def __init__(
        self,
        min_mutation_count: int = 5,
        n_components: int = 64,
        max_iter: int = 200,
        batch_size: int = 256,
        random_state: int = 42,
        **parameters: object,
    ) -> None:
        super().__init__(**parameters)
        for value, name in (
            (min_mutation_count, "min_mutation_count"),
            (n_components, "n_components"),
            (max_iter, "max_iter"),
            (batch_size, "batch_size"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name}는 1 이상의 정수여야 합니다.")

        self.min_mutation_count = min_mutation_count
        self.requested_components = n_components
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.selected_gene_columns: list[str] = []
        self.dropped_rare_columns: list[str] = []
        self.mutation_counts_: dict[str, int] = {}
        self.effective_components = 0
        self.nmf: MiniBatchNMF | None = None
        self.steps = (
            "최소 변이 빈도 필터",
            "전체 유전자 이진 변이 표현",
            "MiniBatch NMF 잠재 변이 모듈 생성",
        )

    def _mutation_matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        return create_mutation_presence_matrix(
            features,
            self.selected_gene_columns,
        ).astype("float32")

    def _latent_frame(
        self,
        matrix: pd.DataFrame,
        latent_values: np.ndarray,
    ) -> pd.DataFrame:
        columns = [f"mutation_module_{index:03d}" for index in range(self.effective_components)]
        latent = pd.DataFrame(latent_values, index=matrix.index, columns=columns)
        burden = matrix.sum(axis=1).astype("float32")
        latent["mutation_burden_log1p"] = np.log1p(burden).astype("float32")
        return latent.astype("float32")

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "EMV8PreprocessingPipeline":
        (
            self.selected_gene_columns,
            self.dropped_rare_columns,
            self.mutation_counts_,
        ) = select_genes_by_mutation_count(features, self.min_mutation_count)
        if not self.selected_gene_columns:
            raise ValueError("최소 변이 빈도를 만족하는 유전자 피처가 없습니다.")

        matrix = self._mutation_matrix(features)
        self.effective_components = min(
            self.requested_components,
            matrix.shape[0],
            matrix.shape[1],
        )
        self.nmf = MiniBatchNMF(
            n_components=self.effective_components,
            init="nndsvda",
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )
        latent_values = self.nmf.fit_transform(matrix)
        super().fit(self._latent_frame(matrix, latent_values), labels)
        print(
            f"[{self.name}] {len(self.selected_gene_columns)}개 유전자를 "
            f"{self.effective_components}개 변이 모듈로 압축"
        )
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.nmf is None:
            raise RuntimeError("fit()을 먼저 호출해야 합니다.")
        matrix = self._mutation_matrix(features)
        latent_values = self.nmf.transform(matrix)
        return super().transform(self._latent_frame(matrix, latent_values)).astype(
            "float32"
        )

    def summary(self) -> dict[str, int]:
        summary = super().summary()
        summary["dropped_rare_features"] = len(self.dropped_rare_columns)
        summary["mutation_modules"] = self.effective_components
        return summary
