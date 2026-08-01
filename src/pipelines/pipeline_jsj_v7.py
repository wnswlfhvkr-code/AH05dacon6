"""JSJ v7: 충돌 쌍별 강도, C, 임계 오프셋을 적용하는 TEST_004_5 파이프라인."""

from src.pipelines.pipeline_jsj_v3 import JSJV3PreprocessingPipeline


class JSJV7PreprocessingPipeline(JSJV3PreprocessingPipeline):
    name = "jsj_v7"
    conflict_mode = "soft"
    conflict_weights = {"KIRC_KIPAN": 0.10, "LGG_GBMLGG": 1.0}
    conflict_specialist_c = {"KIRC_KIPAN": 0.20, "LGG_GBMLGG": 0.10}
    conflict_right_offset = {"KIRC_KIPAN": 0.0, "LGG_GBMLGG": 0.10}
