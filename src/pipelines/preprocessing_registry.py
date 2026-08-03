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
from src.pipelines.pipeline_em_v15 import EMV15PreprocessingPipeline
from src.pipelines.pipeline_em_v16 import EMV16PreprocessingPipeline
from src.pipelines.pipeline_em_v17 import EMV17PreprocessingPipeline
from src.pipelines.pipeline_em_v18 import EMV18PreprocessingPipeline
from src.pipelines.pipeline_em_v19 import EMV19PreprocessingPipeline
from src.pipelines.pipeline_em_v20 import EMV20PreprocessingPipeline
from src.pipelines.pipeline_em_v21 import EMV21PreprocessingPipeline
from src.pipelines.pipeline_em_v22 import EMV22PreprocessingPipeline
from src.pipelines.pipeline_em_v23 import EMV23PreprocessingPipeline
from src.pipelines.pipeline_em_v24 import EMV24PreprocessingPipeline
from src.pipelines.pipeline_em_v25 import EMV25PreprocessingPipeline
from src.pipelines.pipeline_em_v26 import EMV26PreprocessingPipeline
from src.pipelines.pipeline_em_v27 import EMV27PreprocessingPipeline

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
    "em_v15": EMV15PreprocessingPipeline,
    "em_v16": EMV16PreprocessingPipeline,
    "em_v17": EMV17PreprocessingPipeline,
    "em_v18": EMV18PreprocessingPipeline,
    "em_v19": EMV19PreprocessingPipeline,
    "em_v20": EMV20PreprocessingPipeline,
    "em_v21": EMV21PreprocessingPipeline,
    "em_v22": EMV22PreprocessingPipeline,
    "em_v23": EMV23PreprocessingPipeline,
    "em_v24": EMV24PreprocessingPipeline,
    "em_v25": EMV25PreprocessingPipeline,
    "em_v26": EMV26PreprocessingPipeline,
    "em_v27": EMV27PreprocessingPipeline,
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
