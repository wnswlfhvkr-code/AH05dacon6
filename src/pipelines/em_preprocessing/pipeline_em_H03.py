"""H03: Mutual information 상위 유전자를 선택합니다."""

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.em_preprocessing.em_dimensionality_engine import (
    EMDimensionalityEngine,
)


class EMH03PreprocessingPipeline(PreprocessingPipeline):
    name = "em_H03"
    evaluation_folds = 5

    def __init__(self, **parameters: object) -> None:
        super().__init__()
        self.engine = EMDimensionalityEngine("mutual_information", parameters)
        self.steps = self.engine.steps

    def fit(self, features, labels):
        self.engine.fit(self, features, labels)
        return self

    def fit_transform(self, features, labels):
        return self.engine.fit_transform(self, features, labels)

    def transform(self, features):
        return self.engine.transform(self, features)

    def summary(self):
        return self.engine.summary(self)
