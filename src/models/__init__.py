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


def create_muat(model_config: dict, seed: int):
    from src.models.muat_model import create_model

    return create_model(model_config, seed)


def create_pair_specialist_classifier(model_config: dict, seed: int):
    from src.models.pair_specialist_classifier_model import create_model

    return create_model(model_config, seed)


def create_pattern_posterior_classifier(model_config: dict, seed: int):
    from src.models.pattern_posterior_classifier_model import create_model

    return create_model(model_config, seed)


def create_macro_f1_calibrated_classifier(model_config: dict, seed: int):
    from src.models.macro_f1_calibrated_classifier_model import create_model

    return create_model(model_config, seed)


def create_tabpfn3(model_config: dict, seed: int):
    from src.models.tabpfn3_model import create_model

    return create_model(model_config, seed)


def create_pipecomb_ensemble(model_config: dict, seed: int):
    from src.models.pipecomb_ensemble_model import create_model

    return create_model(model_config, seed)


def create_pipecomb_weighted_soft_voting(model_config: dict, seed: int):
    from src.models.pipecomb_weighted_soft_voting_model import create_model

    return create_model(model_config, seed)


def create_pipecomb_oof_stacking(model_config: dict, seed: int):
    from src.models.pipecomb_oof_stacking_model import create_model

    return create_model(model_config, seed)


MODEL_BUILDERS = {
    "logistic_regression": create_logistic_regression,
    "xgboost": create_xgboost,
    "lightgbm": create_lightgbm,
    "linear_svc": create_linear_svc,
    "wc_tfidf_lsvc_lgbm": create_wc_tfidf_lsvc_lgbm,
    "muat": create_muat,
    "pair_specialist_classifier": create_pair_specialist_classifier,
    "pattern_posterior_classifier": create_pattern_posterior_classifier,
    "macro_f1_calibrated_classifier": create_macro_f1_calibrated_classifier,
    "tabpfn3": create_tabpfn3,
    "pipecomb_ensemble": create_pipecomb_ensemble,
    "pipecomb_weighted_soft_voting": create_pipecomb_weighted_soft_voting,
    "pipecomb_oof_stacking": create_pipecomb_oof_stacking,
}
