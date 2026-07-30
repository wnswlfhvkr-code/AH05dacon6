"""XGBoost 분류 모델 생성 함수입니다."""

from __future__ import annotations

from xgboost import XGBClassifier


def create_model(model_config: dict, seed: int) -> XGBClassifier:
    """설정 파일의 하이퍼파라미터로 XGBoost 분류기를 만듭니다."""
    return XGBClassifier(
        n_estimators=model_config["n_estimators"],
        learning_rate=model_config["learning_rate"],
        max_depth=model_config["max_depth"],
        random_state=seed,
        n_jobs=model_config["n_jobs"],
        eval_metric="mlogloss",
    )
