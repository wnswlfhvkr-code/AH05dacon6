"""모델별 생성 함수를 모아 둔 패키지입니다."""


def create_logistic_regression(model_config: dict, seed: int):
    from src.models.logistic_regression_model import create_model

    return create_model(model_config, seed)


def create_xgboost(model_config: dict, seed: int):
    from src.models.xgboost_model import create_model

    return create_model(model_config, seed)


def create_lightgbm(model_config: dict, seed: int):
    from src.models.lightgbm_model import create_model

    return create_model(model_config, seed)


def create_linear_svc(model_config: dict, seed: int):
    from src.models.linear_svc_model import create_model

    return create_model(model_config, seed)


def create_wc_tfidf_lsvc_lgbm(model_config: dict, seed: int):
    from src.models.wc_tfidf_lsvc_lgbm_model import create_model

    return create_model(model_config, seed)


MODEL_BUILDERS = {
    "logistic_regression": create_logistic_regression,
    "xgboost": create_xgboost,
    "lightgbm": create_lightgbm,
    "linear_svc": create_linear_svc,
    "wc_tfidf_lsvc_lgbm": create_wc_tfidf_lsvc_lgbm,
}