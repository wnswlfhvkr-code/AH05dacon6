"""JSJ v11: lower-shrinkage vote-pattern selector used by TEST_004_9."""

from src.pipelines.pipeline_jsj_v10 import JSJV10PreprocessingPipeline


class JSJV11PreprocessingPipeline(JSJV10PreprocessingPipeline):
    name = "jsj_v11"
    selector_version = "v2"
    selector_min_count = 3
    selector_shrinkage = 0.5
    selector_inputs = (
        "MultiPair v9", "FullMeta v7", "LGBMeta v1", "corrected", "CPEM f003"
    )
