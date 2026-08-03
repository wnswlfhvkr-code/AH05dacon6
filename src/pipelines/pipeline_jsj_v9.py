"""JSJ v9: JSJ v8에 EM v16/v24 충돌 전문가를 결합하는 재현 파이프라인."""

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.pipeline_jsj_v3 import JSJV3PreprocessingPipeline


class JSJV9PreprocessingPipeline(PreprocessingPipeline):
    """기본 문자열 피처를 만들고 v9 앙상블 설정을 공개합니다."""

    name = "jsj_v9"
    conflict_mode = "soft"
    conflict_weights = {"KIRC_KIPAN": 0.35, "LGG_GBMLGG": 0.30}
    conflict_specialist_c = {"KIRC_KIPAN": 0.15, "LGG_GBMLGG": 0.06}
    conflict_right_offset = {"KIRC_KIPAN": -0.05, "LGG_GBMLGG": 0.06}
    team_experts = {
        "pipelines": {"em_v16": 0.40, "em_v24": 0.60},
        "temperature": 1.75,
        "right_offset": {"KIRC_KIPAN": 0.50, "LGG_GBMLGG": 0.30},
    }

    def __init__(self, **parameters):
        super().__init__()
        self._delegate = JSJV3PreprocessingPipeline(**parameters)

    def fit(self, features, labels):
        self._delegate.fit(features, labels)
        return self

    def fit_transform(self, features, labels):
        return self._delegate.fit_transform(features, labels)

    def transform(self, features):
        return self._delegate.transform(features)

    def summary(self):
        summary = self._delegate.summary()
        summary["team_experts"] = 2
        return summary
