"""TEST_007용 PyTorch GPU 선형 분류 모델입니다."""

from __future__ import annotations

from src.models.torch_tabular_base import TorchTabularClassifier


def create_model(model_config: dict, seed: int) -> TorchTabularClassifier:
    return TorchTabularClassifier(
        hidden_dims=(),
        learning_rate=model_config.get("learning_rate", 1e-3),
        epochs=model_config.get("epochs", 40),
        batch_size=model_config.get("batch_size", 256),
        weight_decay=model_config.get("weight_decay", 1e-4),
        dropout=0.0,
        class_weight=model_config.get("class_weight", "balanced"),
        label_smoothing=model_config.get("label_smoothing", 0.0),
        gradient_clip_norm=model_config.get("gradient_clip_norm", 5.0),
        device=model_config.get("device", "cuda"),
        verbose=model_config.get("verbose", False),
        seed=seed,
    )
