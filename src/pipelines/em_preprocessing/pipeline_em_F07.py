"""EM F07: 변이 없는 표본 여부 파생 피처."""

from __future__ import annotations

import numpy as np
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


def _event_counts(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"누락된 유전자 컬럼이 있습니다: {sorted(missing)}")
    return pd.DataFrame({
        column: features[column].map(_token_count).astype("int16")
        for column in columns
    }, index=features.index)


class _F07DerivedFeaturePipeline(PreprocessingPipeline):
    """변이 없는 표본 여부만 생성하는 독립 실험 파이프라인입니다."""

    name = "em_F07"
    evaluation_folds = 5

    def __init__(self, **_: object) -> None:
        super().__init__()
        self.gene_columns: list[str] = []
        self.steps = ("결측·WT 정규화", "셀 내 중복 토큰 제거", "변이 없는 표본 여부")

    def _build_features(self, features: pd.DataFrame) -> pd.DataFrame:
        counts = _event_counts(features, self.gene_columns)
        return pd.DataFrame({
            "no_mutation_sample": counts.sum(axis=1).eq(0).astype("int8"),
        }, index=features.index)

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "EMF07PreprocessingPipeline":
        self.gene_columns = features.columns.tolist()
        PreprocessingPipeline.fit(self, self._build_features(features), labels)
        return self

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        return self.fit(features, labels).transform(features)

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return PreprocessingPipeline.transform(self, self._build_features(features))

    def summary(self) -> dict[str, int]:
        result = PreprocessingPipeline.summary(self)
        result["derived_feature_count"] = 1
        return result


class EMF07PreprocessingPipeline(PreprocessingPipeline):
    """EMV45를 베이스로 F07 전용 파생 피처를 결합합니다."""

    name = "em_F07"
    evaluation_folds = 5

    def __init__(self, **parameters: object) -> None:
        super().__init__()
        self.v45_pipeline = EMV45PreprocessingPipeline(**parameters)
        self.derived_pipeline = _F07DerivedFeaturePipeline(**parameters)
        self.v45_feature_count_ = 0
        self.derived_feature_count_ = 0
        self.steps = (
            "EMV45 베이스 피처",
            "F07 전용 파생 피처",
            "F07 접두사로 컬럼 충돌 제거",
            "학습 행 OOF·validation/test train-fit transform",
        )

    @staticmethod
    def _combine(v45: pd.DataFrame, derived: pd.DataFrame) -> pd.DataFrame:
        prefixed = derived.copy()
        prefixed.columns = [f"F07__{column}" for column in prefixed.columns]
        return pd.concat([v45, prefixed], axis=1)

    def fit(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> "EMF07PreprocessingPipeline":
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
            "f07_derived_features": self.derived_feature_count_,
            "combined_before_constant_filter": (
                self.v45_feature_count_ + self.derived_feature_count_
            ),
        })
        return result
