"""TEST_004_7: JSJ v8 + EM v16/v24 충돌 암종군 전문가 앙상블."""

SPEC = {
    "name": "TEST_004_7",
    "base_experiment": "TEST_004_6",
    "team_pipeline_weights": {"em_v16": 0.40, "em_v24": 0.60},
    "xgboost_profile": "e4",
    "temperature": 1.75,
    "right_offset": {"KIRC_KIPAN": 0.50, "LGG_GBMLGG": 0.30},
    "expert_weight": {"KIRC_KIPAN": 0.35, "LGG_GBMLGG": 0.30},
    "historical_oof_macro_f1": 0.528569659777493,
    "historical_public_macro_f1": None,
}
