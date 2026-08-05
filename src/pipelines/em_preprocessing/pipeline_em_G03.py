"""EM G03: 고상관 그룹에서 활성 변이 빈도가 높은 피처를 유지합니다."""

import pandas as pd

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.em_preprocessing.em_redundancy_filter import (
    EMV46RedundancyFilterEngine,
)


class EMG03PreprocessingPipeline(PreprocessingPipeline):
    name = "em_G03"
    evaluation_folds = 5

    def __init__(self, **parameters: object) -> None:
        super().__init__()
        self.engine = EMV46RedundancyFilterEngine("g03_mutation_frequency", parameters)
        self.steps = self.engine.steps

    def fit(self, features: pd.DataFrame, labels: pd.Series):
        self.engine.fit(self, features, labels)
        return self

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        return self.engine.fit_transform(self, features, labels)

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.engine.transform(self, features)

    def summary(self) -> dict[str, int]:
        return self.engine.summary(self)
