"""JSJ v8: JSJ 문자열 앙상블에 EM v14 KIRC 전문가를 결합합니다."""

from src.pipelines.pipeline_jsj_v3 import JSJV3PreprocessingPipeline


class JSJV8PreprocessingPipeline(JSJV3PreprocessingPipeline):
    name = "jsj_v8"
    conflict_mode = "soft"
    conflict_weights = {"KIRC_KIPAN": 0.10, "LGG_GBMLGG": 1.0}
    conflict_specialist_c = {"KIRC_KIPAN": 0.15, "LGG_GBMLGG": 0.06}
    conflict_right_offset = {"KIRC_KIPAN": -0.05, "LGG_GBMLGG": 0.06}
    auxiliary_specialist = {
        "name": "em_v14_kirc_specialist",
        "pair": "KIRC_KIPAN",
        "weight": 0.30,
        "temperature": 0.50,
        "right_offset": 0.30,
    }
