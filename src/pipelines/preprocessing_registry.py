"""설정 파일에서 전처리 파이프라인을 선택하는 레지스트리입니다."""

from src.pipelines.pipeline_baseline import BaselinePreprocessingPipeline
from src.pipelines.pipeline_em_v1 import EMV1PreprocessingPipeline
from src.pipelines.pipeline_jsj_v1 import JSJV1PreprocessingPipeline
from src.pipelines.pipeline_jsj_v2 import JSJV2PreprocessingPipeline
from src.pipelines.pipeline_jsj_v3 import JSJV3PreprocessingPipeline
from src.pipelines.pipeline_jsj_v4 import JSJV4PreprocessingPipeline
from src.pipelines.pipeline_jsj_v5 import JSJV5PreprocessingPipeline
from src.pipelines.pipeline_jsj_v6 import JSJV6PreprocessingPipeline

PIPELINES = {
    "baseline": BaselinePreprocessingPipeline,
    "em_v1": EMV1PreprocessingPipeline,
    "jsj_v1": JSJV1PreprocessingPipeline,
    "jsj_v2": JSJV2PreprocessingPipeline,
    "jsj_v3": JSJV3PreprocessingPipeline,
    "jsj_v4": JSJV4PreprocessingPipeline,
    "jsj_v5": JSJV5PreprocessingPipeline,
    "jsj_v6": JSJV6PreprocessingPipeline,
}


def create_preprocessing_pipeline(config: dict):
    """설정의 이름과 파라미터에 맞는 전처리 파이프라인을 생성합니다."""
    pipeline_name = config.get("name", "baseline")
    try:
        pipeline_class = PIPELINES[pipeline_name]
    except KeyError as error:
        available = ", ".join(sorted(PIPELINES))
        raise ValueError(
            f"지원하지 않는 전처리 파이프라인입니다: {pipeline_name}. 사용 가능: {available}"
        ) from error
    parameters = {key: value for key, value in config.items() if key != "name"}
    return pipeline_class(**parameters)
