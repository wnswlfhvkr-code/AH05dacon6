"""CPU Extra Trees 분류 모델 생성 함수입니다."""

from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier


def create_model(model_config: dict, seed: int) -> ExtraTreesClassifier:
    """설정 파일의 하이퍼파라미터로 Extra Trees 분류기를 만듭니다."""
    parameters = {
        key: value for key, value in model_config.items() if key != "name"
    }
    parameters.pop("random_state", None)
    parameters["random_state"] = seed
    parameters.setdefault("n_estimators", 600)
    parameters.setdefault("criterion", "gini")
    parameters.setdefault("max_depth", 28)
    parameters.setdefault("min_samples_split", 4)
    parameters.setdefault("min_samples_leaf", 2)
    parameters.setdefault("max_features", "sqrt")
    parameters.setdefault("bootstrap", False)
    parameters.setdefault("class_weight", "balanced")
    parameters.setdefault("n_jobs", -1)
    return ExtraTreesClassifier(**parameters)
