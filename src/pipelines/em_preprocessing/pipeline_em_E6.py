"""KIRC-KIPAN·LGG-GBMLGG class signature contrast를 EMV16과 결합한 독립 실험 파이프라인입니다."""

from __future__ import annotations

import pandas as pd

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.em_preprocessing.em_feature_engine import (
    EMExperimentSpec,
    EMFeatureEngine,
)


class EME6PreprocessingPipeline(PreprocessingPipeline):
    """KIRC-KIPAN·LGG-GBMLGG class signature contrast; 다른 버전 파이프라인을 상속하지 않습니다."""

    name = "em_E6"
    evaluation_folds = 5
    feature_spec = EMExperimentSpec(
        name="E6",
        description="KIRC-KIPAN·LGG-GBMLGG class signature contrast",
        pair_contrasts=True,
    )

    def __init__(self, **parameters: object) -> None:
        super().__init__()
        self.engine = EMFeatureEngine(self.feature_spec, parameters)
        self.steps = self.engine.steps

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "EME6PreprocessingPipeline":
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

