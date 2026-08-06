"""EM G01: 저빈도·중복·고상관 피처를 fold-safe 통합 기준으로 정리합니다."""

from __future__ import annotations

import pandas as pd

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.em_preprocessing.em_redundancy_filter import (
    EMV46RedundancyFilterEngine,
)


class EMG01PreprocessingPipeline(PreprocessingPipeline):
    """EMV46 이후 중복 신호를 제거하고 일반화 가능한 대표 피처를 유지합니다."""

    name = "em_G01"
    evaluation_folds = 5

    def __init__(self, **parameters: object) -> None:
        super().__init__()
        self.engine = EMV46RedundancyFilterEngine(parameters)
        self.steps = self.engine.steps

    def fit(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> "EMG01PreprocessingPipeline":
        self.engine.fit(self, features, labels)
        return self

    def fit_transform(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> pd.DataFrame:
        return self.engine.fit_transform(self, features, labels)

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.engine.transform(self, features)

    def summary(self) -> dict[str, int]:
        return self.engine.summary(self)
