"""EM F05: fold-train recurrent hotspot의 표본별 변이 수."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.pipeline_em_v45 import EMV45PreprocessingPipeline


def _tokens(value: object) -> frozenset[str]:
    if pd.isna(value):
        return frozenset()
    normalized = str(value).strip().upper()
    if not normalized or normalized in {"WT", "<NA>"}:
        return frozenset()
    return frozenset(normalized.split())


class _F05DerivedFeaturePipeline(PreprocessingPipeline):
    """train에서 반복된 gene-token hotspot의 표본별 개수를 생성합니다."""

    name = "em_F05"
    evaluation_folds = 5

    def __init__(
        self, min_hotspot_count: int = 5, max_hotspots: int = 384, **_: object
    ) -> None:
        super().__init__()
        if min_hotspot_count < 2 or max_hotspots < 1:
            raise ValueError("min_hotspot_count는 2 이상, max_hotspots는 1 이상이어야 합니다.")
        self.min_hotspot_count = int(min_hotspot_count)
        self.max_hotspots = int(max_hotspots)
        self.gene_columns: list[str] = []
        self.hotspots_: list[tuple[str, str]] = []
        self.steps = ("결측·WT 정규화", "fold-train recurrent hotspot 학습", "표본 hotspot 수")

    def _learn_hotspots(self, features: pd.DataFrame) -> None:
        support: Counter[tuple[str, str]] = Counter()
        for gene in self.gene_columns:
            for value in features[gene]:
                for token in _tokens(value):
                    support[(gene, token)] += 1
        eligible = [pair for pair, count in support.items() if count >= self.min_hotspot_count]
        self.hotspots_ = sorted(eligible, key=lambda pair: (-support[pair], pair))[:self.max_hotspots]

    def _build_features(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.gene_columns) - set(features.columns)
        if missing:
            raise ValueError(f"누락된 유전자 컬럼이 있습니다: {sorted(missing)}")
        token_cache = {gene: features[gene].map(_tokens) for gene, _ in self.hotspots_}
        count = pd.Series(0, index=features.index, dtype="int32")
        for gene, token in self.hotspots_:
            count += token_cache[gene].map(lambda values: token in values).astype("int32")
        return pd.DataFrame({"hotspot_mutation_count": count}, index=features.index)

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "EMF05PreprocessingPipeline":
        self.gene_columns = features.columns.tolist()
        self._learn_hotspots(features)
        PreprocessingPipeline.fit(self, self._build_features(features), labels)
        return self

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        return self.fit(features, labels).transform(features)

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return PreprocessingPipeline.transform(self, self._build_features(features))

    def summary(self) -> dict[str, int]:
        result = PreprocessingPipeline.summary(self)
        result.update({"derived_feature_count": 1, "learned_hotspots": len(self.hotspots_)})
        return result


class EMF05PreprocessingPipeline(PreprocessingPipeline):
    """EMV45를 베이스로 F05 전용 파생 피처를 결합합니다."""

    name = "em_F05"
    evaluation_folds = 5

    def __init__(self, **parameters: object) -> None:
        super().__init__()
        self.v45_pipeline = EMV45PreprocessingPipeline(**parameters)
        self.derived_pipeline = _F05DerivedFeaturePipeline(**parameters)
        self.v45_feature_count_ = 0
        self.derived_feature_count_ = 0
        self.steps = (
            "EMV45 베이스 피처",
            "F05 전용 파생 피처",
            "F05 접두사로 컬럼 충돌 제거",
            "학습 행 OOF·validation/test train-fit transform",
        )

    @staticmethod
    def _combine(v45: pd.DataFrame, derived: pd.DataFrame) -> pd.DataFrame:
        prefixed = derived.copy()
        prefixed.columns = [f"F05__{column}" for column in prefixed.columns]
        return pd.concat([v45, prefixed], axis=1)

    def fit(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> "EMF05PreprocessingPipeline":
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
            "f05_derived_features": self.derived_feature_count_,
            "combined_before_constant_filter": (
                self.v45_feature_count_ + self.derived_feature_count_
            ),
        })
        return result
