"""기존 jsj_v9 import 경로를 유지하는 호환 모듈."""

from src.pipelines.TEST_004.EM_SJ_pipeComb_v1 import (
    EMSJPipeCombV1PreprocessingPipeline,
)


JSJV9PreprocessingPipeline = EMSJPipeCombV1PreprocessingPipeline
