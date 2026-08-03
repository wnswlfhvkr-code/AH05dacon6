"""JH v06: F0+F1+gene-aware F3, Fold-Train support 5."""

from src.pipelines.pipeline_jh_v05 import (
    JHV05PreprocessingPipeline,
)
from src.pipelines.base import PreprocessingPipeline


class JHV06PreprocessingPipeline(PreprocessingPipeline):
    name = "jh_v06"
    min_support = 5

    def __init__(self, **kwargs: object) -> None:
        kwargs["min_support"] = 5
        self.base_pipeline = JHV05PreprocessingPipeline(**kwargs)

    def fit(self, features, labels):
        self.base_pipeline.fit(features, labels)
        return self

    def transform(self, features):
        return self.base_pipeline.transform(features)

    def fit_transform(self, features, labels):
        return self.fit(features, labels).transform(features)

    def encode_labels(self, labels):
        return self.base_pipeline.encode_labels(labels)

    def decode_labels(self, labels):
        return self.base_pipeline.decode_labels(labels)

    def summary(self):
        result = self.base_pipeline.summary()
        result["f3_min_support"] = 5
        return result
