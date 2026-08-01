"""JSJ v6: LGG 전면/KIRC 10% 비대칭 보정 TEST_004_4 파이프라인."""

from src.pipelines.pipeline_jsj_v3 import JSJV3PreprocessingPipeline


class JSJV6PreprocessingPipeline(JSJV3PreprocessingPipeline):
    name = "jsj_v6"
    conflict_mode = "soft"
    conflict_weights = {"KIRC_KIPAN": 0.10, "LGG_GBMLGG": 1.0}

