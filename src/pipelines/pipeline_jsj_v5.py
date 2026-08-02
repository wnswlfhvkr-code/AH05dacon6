"""JSJ v5: 충돌 쌍 내부에서만 재판정하는 TEST_004_3 파이프라인."""

from src.pipelines.pipeline_jsj_v3 import JSJV3PreprocessingPipeline


class JSJV5PreprocessingPipeline(JSJV3PreprocessingPipeline):
    name = "jsj_v5"
    conflict_mode = "strict"
    conflict_weights = {"KIRC_KIPAN": 1.0, "LGG_GBMLGG": 1.0}

