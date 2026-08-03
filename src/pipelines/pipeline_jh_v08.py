"""JH v08: F0+F1+global F3+exact hotspot F4, support 5."""

from src.pipelines.pipeline_jh_v07 import (
    JHV07PreprocessingPipeline,
)
from src.pipelines.base import PreprocessingPipeline


class JHV08PreprocessingPipeline(PreprocessingPipeline):
    name = "jh_v08"
    f4_min_support = 5

    def __init__(self, **kwargs: object) -> None:
        kwargs["f4_min_support"] = 5
        self.base_pipeline = JHV07PreprocessingPipeline(**kwargs)

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
        result["f4_min_support"] = 5
        return result
