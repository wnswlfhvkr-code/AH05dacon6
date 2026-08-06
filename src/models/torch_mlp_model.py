"""TEST_007용 PyTorch GPU MLP 분류 모델입니다."""

from __future__ import annotations

from src.models.torch_tabular_base import TorchTabularClassifier


def create_model(model_config: dict, seed: int) -> TorchTabularClassifier:
    hidden_dims = tuple(model_config.get("hidden_dims", [256, 128]))
    if not hidden_dims or any(dimension < 1 for dimension in hidden_dims):
        raise ValueError("hidden_dims는 1 이상의 정수 목록이어야 합니다.")
    return TorchTabularClassifier(
        hidden_dims=hidden_dims,
        learning_rate=model_config.get("learning_rate", 1e-3),
        epochs=model_config.get("epochs", 50),
        batch_size=model_config.get("batch_size", 256),
        weight_decay=model_config.get("weight_decay", 1e-4),
        dropout=model_config.get("dropout", 0.2),
        class_weight=model_config.get("class_weight", "balanced"),
        label_smoothing=model_config.get("label_smoothing", 0.02),
        gradient_clip_norm=model_config.get("gradient_clip_norm", 5.0),
        device=model_config.get("device", "cuda"),
        verbose=model_config.get("verbose", False),
        seed=seed,
    )
