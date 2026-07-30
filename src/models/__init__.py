"""모델별 생성 함수를 모아 둔 패키지입니다."""

from src.models.xgboost_model import create_model

MODEL_BUILDERS = {
    "xgboost": create_model,
}
