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


def create_catboost(model_config: dict, seed: int):
    from src.models.catboost_model import create_model

    return create_model(model_config, seed)


def create_torch_linear(model_config: dict, seed: int):
    from src.models.torch_linear_model import create_model

    return create_model(model_config, seed)


def create_logistic_regression_gpu(model_config: dict, seed: int):
    from src.models.logistic_regression_gpu_model import create_model

    return create_model(model_config, seed)


def create_torch_mlp(model_config: dict, seed: int):
    from src.models.torch_mlp_model import create_model

    return create_model(model_config, seed)


def create_tabpfn(model_config: dict, seed: int):
    from src.models.tabpfn_model import create_model

    return create_model(model_config, seed)


def create_tabicl(model_config: dict, seed: int):
    from src.models.tabicl_model import create_model

    return create_model(model_config, seed)


def create_tabicl_collision_expert(model_config: dict, seed: int):
    from src.models.tabicl_collision_expert_model import create_model

    return create_model(model_config, seed)


def create_tabfm_collision_expert(model_config: dict, seed: int):
    from src.models.tabfm_collision_expert_model import create_model

    return create_model(model_config, seed)


def create_tabm(model_config: dict, seed: int):
    from src.models.tabm_model import create_model

    return create_model(model_config, seed)


def create_modernnca(model_config: dict, seed: int):
    from src.models.modernnca_model import create_model

    return create_model(model_config, seed)


def create_modernnca_collision_expert(model_config: dict, seed: int):
    from src.models.modernnca_collision_expert_model import create_model

    return create_model(model_config, seed)


def create_realmlp(model_config: dict, seed: int):
    from src.models.realmlp_model import create_model

    return create_model(model_config, seed)


def create_realtabr_collision_expert(model_config: dict, seed: int):
    from src.models.realtabr_collision_expert_model import create_model

    return create_model(model_config, seed)


def create_xrfm(model_config: dict, seed: int):
    from src.models.xrfm_model import create_model

    return create_model(model_config, seed)


def create_linear_svc(model_config: dict, seed: int):
    from src.models.linear_svc_model import create_model

    return create_model(model_config, seed)


def create_wc_tfidf_lsvc_lgbm(model_config: dict, seed: int):
    from src.models.wc_tfidf_lsvc_lgbm_model import create_model

    return create_model(model_config, seed)


def create_extra_trees(model_config: dict, seed: int):
    from src.models.extra_trees_model import create_model
    
    return create_model(model_config, seed)
  
  
def create_muat(model_config: dict, seed: int):
    from src.models.muat_model import create_model

    return create_model(model_config, seed)


def create_balanced_random_forest(model_config: dict, seed: int):
    from src.models.balanced_random_forest_model import create_model
    
    return create_model(model_config, seed)
  
  
def create_pair_specialist_classifier(model_config: dict, seed: int):
    from src.models.pair_specialist_classifier_model import create_model

    return create_model(model_config, seed)


def create_random_forest(model_config: dict, seed: int):
    from src.models.random_forest_model import create_model
    
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


def create_pipecomb_tabpfn_subset_stacking(model_config: dict, seed: int):
    from src.models.pipecomb_tabpfn_subset_stacking_model import create_model

    return create_model(model_config, seed)


MODEL_BUILDERS = {
    "logistic_regression": create_logistic_regression,
    "xgboost": create_xgboost,
    "lightgbm": create_lightgbm,
    "catboost": create_catboost,
    "torch_linear": create_torch_linear,
    "logistic_regression_gpu": create_logistic_regression_gpu,
    "torch_mlp": create_torch_mlp,
    "tabpfn": create_tabpfn,
    "tabicl": create_tabicl,
    "tabicl_collision_expert": create_tabicl_collision_expert,
    "tabfm_collision_expert": create_tabfm_collision_expert,
    "tabm": create_tabm,
    "modernnca": create_modernnca,
    "modernnca_collision_expert": create_modernnca_collision_expert,
    "realmlp": create_realmlp,
    "realtabr_collision_expert": create_realtabr_collision_expert,
    "xrfm": create_xrfm,
    "linear_svc": create_linear_svc,
    "wc_tfidf_lsvc_lgbm": create_wc_tfidf_lsvc_lgbm,
    "extra_trees": create_extra_trees,
    "balanced_random_forest": create_balanced_random_forest,
    "random_forest": create_random_forest,
    "muat": create_muat,
    "pair_specialist_classifier": create_pair_specialist_classifier,
    "pattern_posterior_classifier": create_pattern_posterior_classifier,
    "macro_f1_calibrated_classifier": create_macro_f1_calibrated_classifier,
    "tabpfn3": create_tabpfn3,
    "pipecomb_ensemble": create_pipecomb_ensemble,
    "pipecomb_weighted_soft_voting": create_pipecomb_weighted_soft_voting,
    "pipecomb_oof_stacking": create_pipecomb_oof_stacking,
    "pipecomb_tabpfn_subset_stacking": create_pipecomb_tabpfn_subset_stacking,
}
