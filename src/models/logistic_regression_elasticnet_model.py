"""Class-balanced multinomial ElasticNet logistic regression factory."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def create_model(model_config: dict, seed: int) -> LogisticRegression:
    """Build a validated SAGA ElasticNet multinomial logistic model."""
    penalty = str(model_config.get("penalty", "elasticnet"))
    solver = str(model_config.get("solver", "saga"))
    if penalty != "elasticnet":
        raise ValueError("ElasticNet logistic regression requires penalty='elasticnet'.")
    if solver != "saga":
        raise ValueError("ElasticNet logistic regression requires solver='saga'.")

    regularization = float(model_config.get("C", 1.0))
    l1_ratio = float(model_config.get("l1_ratio", 0.5))
    if regularization <= 0:
        raise ValueError("C must be greater than zero.")
    if not 0.0 <= l1_ratio <= 1.0:
        raise ValueError("l1_ratio must be in [0, 1].")

    return LogisticRegression(
        C=regularization,
        class_weight=model_config.get("class_weight", "balanced"),
        solver=solver,
        penalty=penalty,
        l1_ratio=l1_ratio,
        max_iter=int(model_config.get("max_iter", 400)),
        tol=float(model_config.get("tol", 1e-3)),
        n_jobs=int(model_config.get("n_jobs", 1)),
        random_state=int(seed),
        verbose=int(model_config.get("verbose", 0)),
    )
