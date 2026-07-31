"""EM 실험 버전 1용 전처리 파이프라인입니다."""

from src.pipelines.base import PreprocessingPipeline


class EMV1PreprocessingPipeline(PreprocessingPipeline):
    """EM v1 실험의 상수 제거와 범주형·타깃 인코딩을 적용합니다."""
    print(f"[pipeline : em_v1]:{'-'*50}")
    
    name = "em_v1"
