"""E7 패턴 불확실성과 E3 multi-hit을 EMV16에 결합한 파이프라인입니다."""

from __future__ import annotations

import pandas as pd

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.em_preprocessing.em_feature_engine import (
    EMExperimentSpec,
    EMFeatureEngine,
)


class EME11PreprocessingPipeline(PreprocessingPipeline):
    """OOF 패턴 불확실성과 token 복잡도를 각각 한 번 적용합니다."""

    name = "em_E11"
    evaluation_folds = 5
    feature_spec = EMExperimentSpec(
        name="E11",
        description="E7 OOF 패턴 불확실성 + E3 multi-hit token complexity",
        pattern_uncertainty=True,
        token_complexity=True,
    )

    def __init__(self, **parameters: object) -> None:
        super().__init__()
        self.engine = EMFeatureEngine(self.feature_spec, parameters)
        self.steps = self.engine.steps

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "EME11PreprocessingPipeline":
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

