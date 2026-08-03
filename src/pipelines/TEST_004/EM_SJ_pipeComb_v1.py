"""EM_SJ_pipeComb_v1: JSJ 문자열 기반과 EM v16/v24 전문가 결합."""

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.pipeline_jsj_v3 import JSJV3PreprocessingPipeline


class EMSJPipeCombV1PreprocessingPipeline(PreprocessingPipeline):
    """기본 문자열 피처와 충돌 암종군 앙상블 설정을 제공합니다."""

    name = "EM_SJ_pipeComb_v1"
    base_pipeline = "jsj_v8"
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


# 기존 jsj_v9 설정 및 import 경로와의 하위 호환성을 유지합니다.
JSJV9PreprocessingPipeline = EMSJPipeCombV1PreprocessingPipeline
