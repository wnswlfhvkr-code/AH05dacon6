"""모델별 생성 함수를 모아 둔 패키지입니다."""

def create_logistic_regression(model_config: dict, seed: int):
    """필요할 때만 Logistic Regression 구현을 불러옵니다."""
    from src.models.logistic_regression_model import create_model

    return create_model(model_config, seed)


def create_xgboost(model_config: dict, seed: int):
    """필요할 때만 XGBoost 구현을 불러옵니다."""
    from src.models.xgboost_model import create_model

    return create_model(model_config, seed)


MODEL_BUILDERS = {
    "logistic_regression": create_logistic_regression,
    "xgboost": create_xgboost,
}
