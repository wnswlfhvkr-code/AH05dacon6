"""원본 XGBoost 베이스라인용 전처리 파이프라인입니다."""

from src.pipelines.base import PreprocessingPipeline


class BaselinePreprocessingPipeline(PreprocessingPipeline):
    """베이스라인의 상수 제거와 범주형·타깃 인코딩을 적용합니다."""

    name = "baseline"
