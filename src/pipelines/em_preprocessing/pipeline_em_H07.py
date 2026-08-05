"""H07: 원본 유전자 이진 피처와 burden·hotspot을 결합합니다."""

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.em_preprocessing.em_dimensionality_engine import (
    EMDimensionalityEngine,
)


class EMH07PreprocessingPipeline(PreprocessingPipeline):
    name = "em_H07"
    evaluation_folds = 5

    def __init__(self, **parameters: object) -> None:
        super().__init__()
        self.engine = EMDimensionalityEngine("genes_burden_hotspot", parameters)
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
