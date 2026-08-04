"""E7·E3·E6을 EMV16에 중복 없이 결합한 파이프라인입니다."""

from __future__ import annotations

import pandas as pd

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.em_preprocessing.em_feature_engine import (
    EMExperimentSpec,
    EMFeatureEngine,
)


class EME14PreprocessingPipeline(PreprocessingPipeline):
    """OOF 패턴 불확실성·multi-hit·pair contrast를 한 번씩 적용합니다."""

    name = "em_E14"
    evaluation_folds = 5
    feature_spec = EMExperimentSpec(
        name="E14",
        description=(
            "E7 OOF 패턴 불확실성 + E3 multi-hit token complexity + "
            "E6 class signature pair contrast"
        ),
        pattern_uncertainty=True,
        token_complexity=True,
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
    ) -> "EME14PreprocessingPipeline":
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

