"""E7 패턴 불확실성과 E5 안정 hotspot을 EMV16에 결합한 파이프라인입니다."""

from __future__ import annotations

import pandas as pd

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.em_preprocessing.em_feature_engine import (
    EMExperimentSpec,
    EMFeatureEngine,
)


class EME13PreprocessingPipeline(PreprocessingPipeline):
    """OOF 패턴 불확실성과 안정 hotspot 선택을 한 번씩 적용합니다."""

    name = "em_E13"
    evaluation_folds = 5
    feature_spec = EMExperimentSpec(
        name="E13",
        description="E7 OOF 패턴 불확실성 + E5 fold 안정 hotspot",
        pattern_uncertainty=True,
        stable_hotspots=True,
    )

    def __init__(self, **parameters: object) -> None:
        super().__init__()
        self.engine = EMFeatureEngine(self.feature_spec, parameters)
        self.steps = self.engine.steps

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "EME13PreprocessingPipeline":
        self.engine.fit(self, features, labels)
        return self

    def fit_transform(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        return self.engine.fit_transform(self, features, labels)

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.engine.transform(self, features)

    def summary(self) -> dict[str, int]:
        return self.engine.summary(self)
