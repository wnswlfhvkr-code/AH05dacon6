"""설정 파일에서 전처리 파이프라인을 선택하는 레지스트리입니다."""

from src.pipelines.pipeline_baseline import BaselinePreprocessingPipeline
from src.pipelines.pipeline_em_v1 import EMV1PreprocessingPipeline
from src.pipelines.pipeline_em_v2 import EMV2PreprocessingPipeline
from src.pipelines.pipeline_em_v3 import EMV3PreprocessingPipeline
from src.pipelines.pipeline_em_v4 import EMV4PreprocessingPipeline
from src.pipelines.pipeline_em_v5 import EMV5PreprocessingPipeline
from src.pipelines.pipeline_em_v6 import EMV6PreprocessingPipeline
from src.pipelines.pipeline_em_v7 import EMV7PreprocessingPipeline
from src.pipelines.pipeline_em_v8 import EMV8PreprocessingPipeline
from src.pipelines.pipeline_em_v9 import EMV9PreprocessingPipeline
from src.pipelines.pipeline_em_v10 import EMV10PreprocessingPipeline
from src.pipelines.pipeline_em_v11 import EMV11PreprocessingPipeline
from src.pipelines.pipeline_em_v12 import EMV12PreprocessingPipeline
from src.pipelines.pipeline_em_v13 import EMV13PreprocessingPipeline
from src.pipelines.pipeline_em_v14 import EMV14PreprocessingPipeline
from src.pipelines.pipeline_jsj_v1 import JSJV1PreprocessingPipeline
from src.pipelines.pipeline_jsj_v2 import JSJV2PreprocessingPipeline
from src.pipelines.pipeline_jsj_v3 import JSJV3PreprocessingPipeline
from src.pipelines.pipeline_jsj_v4 import JSJV4PreprocessingPipeline
from src.pipelines.pipeline_jsj_v5 import JSJV5PreprocessingPipeline
from src.pipelines.pipeline_jsj_v6 import JSJV6PreprocessingPipeline
from src.pipelines.pipeline_jsj_v7 import JSJV7PreprocessingPipeline
from src.pipelines.pipeline_jsj_v8 import JSJV8PreprocessingPipeline

PIPELINES = {
    "baseline": BaselinePreprocessingPipeline,
    "em_v1": EMV1PreprocessingPipeline,
    "em_v2": EMV2PreprocessingPipeline,
    "em_v3": EMV3PreprocessingPipeline,
    "em_v4": EMV4PreprocessingPipeline,
    "em_v5": EMV5PreprocessingPipeline,
    "em_v6": EMV6PreprocessingPipeline,
    "em_v7": EMV7PreprocessingPipeline,
    "em_v8": EMV8PreprocessingPipeline,
    "em_v9": EMV9PreprocessingPipeline,
    "em_v10": EMV10PreprocessingPipeline,
    "em_v11": EMV11PreprocessingPipeline,
    "em_v12": EMV12PreprocessingPipeline,
    "em_v13": EMV13PreprocessingPipeline,
    "em_v14": EMV14PreprocessingPipeline,
    "jsj_v1": JSJV1PreprocessingPipeline,
    "jsj_v2": JSJV2PreprocessingPipeline,
    "jsj_v3": JSJV3PreprocessingPipeline,
    "jsj_v4": JSJV4PreprocessingPipeline,
    "jsj_v5": JSJV5PreprocessingPipeline,
    "jsj_v6": JSJV6PreprocessingPipeline,
    "jsj_v7": JSJV7PreprocessingPipeline,
    "jsj_v8": JSJV8PreprocessingPipeline,
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
