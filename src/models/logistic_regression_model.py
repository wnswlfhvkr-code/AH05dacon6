"""희소 변이 피처용 Logistic Regression 모델 생성 함수입니다."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def create_model(model_config: dict, seed: int) -> LogisticRegression:
    """설정 파일의 파라미터로 균형 가중 Logistic Regression을 만듭니다."""
    return LogisticRegression(
        C=float(model_config.get("C", 1.0)),
        class_weight=model_config.get("class_weight", "balanced"),
        solver=model_config.get("solver", "saga"),
        penalty=model_config.get("penalty", "l2"),
        max_iter=int(model_config.get("max_iter", 3000)),
        tol=float(model_config.get("tol", 1e-4)),
        n_jobs=int(model_config.get("n_jobs", -1)),
        random_state=seed,
    )
