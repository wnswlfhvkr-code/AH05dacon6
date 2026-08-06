"""CPU Balanced Random Forest 분류 모델 생성 함수입니다."""

from __future__ import annotations

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except ImportError as error:
    raise ImportError(
        "Balanced Random Forest를 사용하려면 "
        "`pip install -r requirements.txt`를 실행하세요."
    ) from error


def create_model(
    model_config: dict,
    seed: int,
) -> BalancedRandomForestClassifier:
    """명시적 균형 샘플링 설정으로 Balanced Random Forest를 만듭니다."""
    parameters = {
        key: value for key, value in model_config.items() if key != "name"
    }
    parameters.pop("random_state", None)
    parameters["random_state"] = seed
    parameters.setdefault("n_estimators", 500)
    parameters.setdefault("criterion", "gini")
    parameters.setdefault("max_depth", 28)
    parameters.setdefault("min_samples_split", 4)
    parameters.setdefault("min_samples_leaf", 2)
    parameters.setdefault("max_features", "sqrt")
    parameters.setdefault("sampling_strategy", "all")
    parameters.setdefault("replacement", True)
    parameters.setdefault("bootstrap", False)
    parameters.setdefault("class_weight", None)
    parameters.setdefault("n_jobs", -1)
    return BalancedRandomForestClassifier(**parameters)
