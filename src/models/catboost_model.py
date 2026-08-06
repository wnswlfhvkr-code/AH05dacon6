"""CatBoost GPU 분류 모델 생성 함수입니다."""

from __future__ import annotations

import numpy as np

try:
    from catboost import CatBoostClassifier
except ImportError as error:
    raise ImportError(
        "CatBoost 모델을 사용하려면 `pip install -r requirements.txt`를 실행하세요."
    ) from error


class GpuCatBoostClassifier(CatBoostClassifier):
    """CatBoost의 클래스 예측을 공용 학습기용 1차원 배열로 맞춥니다."""

    def predict(self, data, *args, **kwargs) -> np.ndarray:
        predictions = super().predict(data, *args, **kwargs)
        return np.asarray(predictions).reshape(-1)


def create_model(model_config: dict, seed: int) -> GpuCatBoostClassifier:
    """설정 파일의 하이퍼파라미터로 GPU CatBoost 분류기를 만듭니다."""
    parameters = {
        key: value for key, value in model_config.items() if key != "name"
    }
    task_type = str(parameters.get("task_type", "GPU")).upper()
    if task_type != "GPU":
        raise ValueError("CatBoost 모델은 task_type=GPU만 지원합니다.")

    parameters.pop("random_state", None)
    parameters["task_type"] = "GPU"
    parameters["random_seed"] = seed
    parameters["allow_writing_files"] = False
    parameters.setdefault("devices", "0")
    parameters.setdefault("loss_function", "MultiClass")
    parameters.setdefault("eval_metric", "MultiClass")
    parameters.setdefault("n_estimators", 500)
    parameters.setdefault("verbose", False)
    if "class_weights" not in parameters:
        parameters.setdefault("auto_class_weights", "Balanced")

    return GpuCatBoostClassifier(**parameters)
