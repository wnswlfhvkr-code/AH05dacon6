"""모델별 생성 함수를 모아 둔 패키지입니다."""

from src.models.lightgbm_model import create_model as create_lightgbm
from src.models.linear_svc_model import create_model as create_linear_svc
from src.models.wc_tfidf_lsvc_lgbm_model import (
    create_model as create_wc_tfidf_lsvc_lgbm,
)
from src.models.xgboost_model import create_model as create_xgboost

MODEL_BUILDERS = {
    "xgboost": create_xgboost,
    "lightgbm": create_lightgbm,
    "linear_svc": create_linear_svc,
    "wc_tfidf_lsvc_lgbm": create_wc_tfidf_lsvc_lgbm,
}
