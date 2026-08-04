"""XGBoost 분류 모델 생성 함수입니다."""

from __future__ import annotations

from xgboost import XGBClassifier


def create_model(model_config: dict, seed: int) -> XGBClassifier:
    """설정 파일의 하이퍼파라미터로 XGBoost 분류기를 만듭니다."""
    parameters = {
        key: value for key, value in model_config.items() if key != "name"
    }
    parameters["random_state"] = seed
    parameters.setdefault("eval_metric", "mlogloss")
    parameters.setdefault("tree_method", "hist")
    parameters.setdefault("device", "cuda")
    return XGBClassifier(**parameters)
