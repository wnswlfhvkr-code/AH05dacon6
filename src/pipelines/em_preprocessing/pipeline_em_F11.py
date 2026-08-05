"""EM F11: 유전자별 multi-hit 여부."""

from __future__ import annotations

import pandas as pd

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.pipeline_em_v45 import EMV45PreprocessingPipeline


def _token_count(value: object) -> int:
    if pd.isna(value):
        return 0
    normalized = str(value).strip().upper()
    if not normalized or normalized in {"WT", "<NA>"}:
        return 0
    return len(set(normalized.split()))


class _F11DerivedFeaturePipeline(PreprocessingPipeline):
    """유전자별 multi-hit 여부를 fold-train 지지도 필터 후 생성합니다."""

    name = "em_F11"
    evaluation_folds = 5

    def __init__(self, min_feature_support: int = 2, **_: object) -> None:
        super().__init__()
        if min_feature_support < 1:
            raise ValueError("min_feature_support는 1 이상이어야 합니다.")
        self.min_feature_support = int(min_feature_support)
        self.gene_columns: list[str] = []
        self.selected_genes_: list[str] = []
        self.steps = ("결측·WT 정규화", "fold-train 지지도 필터", "유전자별 multi-hit 여부")

    def _counts(self, features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        missing = set(columns) - set(features.columns)
        if missing:
            raise ValueError(f"누락된 유전자 컬럼이 있습니다: {sorted(missing)}")
        return pd.DataFrame({
            gene: features[gene].map(_token_count).astype("int16")
            for gene in columns
        }, index=features.index)

    def _build_features(self, features: pd.DataFrame) -> pd.DataFrame:
        counts = self._counts(features, self.selected_genes_)
        return pd.DataFrame({
            f"gene_multi_hit__{gene}": counts[gene].gt(1).astype("int8")
            for gene in self.selected_genes_
        }, index=features.index)

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "EMF11PreprocessingPipeline":
        self.gene_columns = features.columns.tolist()
        counts = self._counts(features, self.gene_columns)
        support = counts.gt(1).sum(axis=0)
        self.selected_genes_ = support[support >= self.min_feature_support].index.tolist()
        if not self.selected_genes_:
            raise ValueError("유전자별 multi-hit 여부 지지도를 만족하는 유전자가 없습니다.")
        PreprocessingPipeline.fit(self, self._build_features(features), labels)
        return self

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        return self.fit(features, labels).transform(features)

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return PreprocessingPipeline.transform(self, self._build_features(features))

    def summary(self) -> dict[str, int]:
        result = PreprocessingPipeline.summary(self)
        result.update({
            "selected_gene_features": len(self.selected_genes_),
            "dropped_gene_features": len(self.gene_columns) - len(self.selected_genes_),
        })
        return result


class EMF11PreprocessingPipeline(PreprocessingPipeline):
    """EMV45를 베이스로 F11 전용 파생 피처를 결합합니다."""

    name = "em_F11"
    evaluation_folds = 5

    def __init__(self, **parameters: object) -> None:
        super().__init__()
        self.v45_pipeline = EMV45PreprocessingPipeline(**parameters)
        self.derived_pipeline = _F11DerivedFeaturePipeline(**parameters)
        self.v45_feature_count_ = 0
        self.derived_feature_count_ = 0
        self.steps = (
            "EMV45 베이스 피처",
            "F11 전용 파생 피처",
            "F11 접두사로 컬럼 충돌 제거",
            "학습 행 OOF·validation/test train-fit transform",
        )

    @staticmethod
    def _combine(v45: pd.DataFrame, derived: pd.DataFrame) -> pd.DataFrame:
        prefixed = derived.copy()
        prefixed.columns = [f"F11__{column}" for column in prefixed.columns]
        return pd.concat([v45, prefixed], axis=1)

    def fit(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> "EMF11PreprocessingPipeline":
        self.v45_pipeline.fit(features, labels)
        self.derived_pipeline.fit(features, labels)
        v45 = self.v45_pipeline.transform(features)
        derived = self.derived_pipeline.transform(features)
        combined = self._combine(v45, derived)
        self.v45_feature_count_ = v45.shape[1]
        self.derived_feature_count_ = derived.shape[1]
        PreprocessingPipeline.fit(self, combined, labels)
        return self

    def fit_transform(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> pd.DataFrame:
        v45 = self.v45_pipeline.fit_transform(features, labels)
        derived = self.derived_pipeline.fit_transform(features, labels)
        self.v45_feature_count_ = v45.shape[1]
        self.derived_feature_count_ = derived.shape[1]
        combined = self._combine(v45, derived)
        PreprocessingPipeline.fit(self, combined, labels)
        return PreprocessingPipeline.transform(self, combined).astype("float32")

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        combined = self._combine(
            self.v45_pipeline.transform(features),
            self.derived_pipeline.transform(features),
        )
        return PreprocessingPipeline.transform(self, combined).astype("float32")

    def summary(self) -> dict[str, int]:
        result = PreprocessingPipeline.summary(self)
        result.update({
            "v45_base_features": self.v45_feature_count_,
            "f11_derived_features": self.derived_feature_count_,
            "combined_before_constant_filter": (
                self.v45_feature_count_ + self.derived_feature_count_
            ),
        })
        return result
