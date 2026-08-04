"""LightGBM 분류 모델 생성 함수입니다."""

from __future__ import annotations


def create_model(model_config: dict, seed: int):
    """설정 파일의 하이퍼파라미터로 LightGBM 분류기를 만듭니다."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:
        raise ImportError(
            "LightGBM 모델을 사용하려면 `pip install -r requirements.txt`를 실행하세요."
        ) from error

    return LGBMClassifier(
        objective=model_config.get("objective", "multiclass"),
        n_estimators=model_config.get("n_estimators", 300),
        learning_rate=model_config.get("learning_rate", 0.05),
        num_leaves=model_config.get("num_leaves", 31),
        max_depth=model_config.get("max_depth", -1),
        min_child_samples=model_config.get("min_child_samples", 20),
        subsample=model_config.get("subsample", 1.0),
        subsample_freq=model_config.get("subsample_freq", 0),
        colsample_bytree=model_config.get("colsample_bytree", 1.0),
        reg_alpha=model_config.get("reg_alpha", 0.0),
        reg_lambda=model_config.get("reg_lambda", 0.0),
        class_weight=model_config.get("class_weight"),
        random_state=seed,
        n_jobs=model_config.get("n_jobs", -1),
        verbosity=model_config.get("verbosity", -1),
        device_type=model_config.get("device_type", "cpu"),
    )
