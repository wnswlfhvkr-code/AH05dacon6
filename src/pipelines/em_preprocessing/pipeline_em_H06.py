"""H06: SVD 잠재 변이 피처를 생성합니다."""

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.em_preprocessing.em_dimensionality_engine import (
    EMDimensionalityEngine,
)


class EMH06PreprocessingPipeline(PreprocessingPipeline):
    name = "em_H06"
    evaluation_folds = 5

    def __init__(self, **parameters: object) -> None:
        super().__init__()
        self.engine = EMDimensionalityEngine("svd", parameters)
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
