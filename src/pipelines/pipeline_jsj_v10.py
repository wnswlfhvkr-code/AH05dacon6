"""JSJ v10: fold-safe exact vote-pattern selector used by TEST_004_8."""

from src.pipelines.pipeline_jsj_v9 import JSJV9PreprocessingPipeline


class JSJV10PreprocessingPipeline(JSJV9PreprocessingPipeline):
    name = "jsj_v10"
    selector_version = "v1"
    selector_min_count = 3
    selector_shrinkage = 1.0
    selector_inputs = (
        "MultiPair v8", "FullMeta v6", "LGBMeta v1", "corrected", "CPEM f003"
    )
