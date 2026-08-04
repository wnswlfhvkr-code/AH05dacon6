"""여러 fold에서 반복 선택되는 안정 hotspot를 EMV16과 결합한 독립 실험 파이프라인입니다."""

from __future__ import annotations

import pandas as pd

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.em_preprocessing.em_feature_engine import (
    EMExperimentSpec,
    EMFeatureEngine,
)


class EME5PreprocessingPipeline(PreprocessingPipeline):
    """여러 fold에서 반복 선택되는 안정 hotspot; 다른 버전 파이프라인을 상속하지 않습니다."""

    name = "em_E5"
    evaluation_folds = 5
    feature_spec = EMExperimentSpec(
        name="E5",
        description="여러 fold에서 반복 선택되는 안정 hotspot",
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
    ) -> "EME5PreprocessingPipeline":
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

