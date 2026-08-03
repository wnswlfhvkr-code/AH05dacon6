"""LinearSVC 분류 모델 생성 함수입니다."""

from __future__ import annotations

from sklearn.svm import LinearSVC


def create_model(model_config: dict, seed: int) -> LinearSVC:
    """설정 파일의 하이퍼파라미터로 LinearSVC 분류기를 만듭니다."""
    return LinearSVC(
        C=model_config.get("C", 0.2),
        class_weight=model_config.get("class_weight", "balanced"),
        max_iter=model_config.get("max_iter", 10_000),
        tol=model_config.get("tol", 1e-4),
        random_state=seed,
        dual=model_config.get("dual", "auto"),
    )
