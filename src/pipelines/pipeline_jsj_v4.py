"""JSJ v4: 두 충돌 쌍을 soft 방식으로 보정하는 TEST_004_2 파이프라인."""

from src.pipelines.pipeline_jsj_v3 import JSJV3PreprocessingPipeline


class JSJV4PreprocessingPipeline(JSJV3PreprocessingPipeline):
    name = "jsj_v4"
    conflict_mode = "soft"
    conflict_weights = {"KIRC_KIPAN": 1.0, "LGG_GBMLGG": 1.0}

