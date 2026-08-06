"""희소 표형 피처를 미니배치로 GPU 학습하는 PyTorch 분류기입니다."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse


class TorchTabularClassifier:
    """scikit-learn 형태의 GPU 전용 PyTorch 다중 분류기."""

    def __init__(
        self,
        *,
        hidden_dims: tuple[int, ...] = (),
        learning_rate: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        dropout: float = 0.2,
        class_weight: str | None = "balanced",
        label_smoothing: float = 0.0,
        gradient_clip_norm: float | None = 5.0,
        device: str = "cuda",
        verbose: bool = False,
        seed: int = 42,
    ) -> None:
        if device != "cuda":
            raise ValueError("PyTorch TEST_007 모델은 device=cuda만 지원합니다.")
        self.hidden_dims = tuple(hidden_dims)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.class_weight = class_weight
        self.label_smoothing = label_smoothing
        self.gradient_clip_norm = gradient_clip_norm
        self.device = device
        self.verbose = verbose
        self.seed = seed

    @staticmethod
    def _import_torch():
        try:
            import torch
        except ImportError as error:
            raise ImportError(
                "PyTorch 모델을 사용하려면 `pip install -r requirements.txt`를 실행하세요."
            ) from error
        return torch

    def _require_cuda(self, torch) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch TEST_007 모델 학습에는 CUDA GPU가 필요합니다.")

    def _build_network(self, torch, input_dim: int, class_count: int):
        layers: list[Any] = []
        previous_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend(
                [
                    torch.nn.Linear(previous_dim, hidden_dim),
                    torch.nn.LayerNorm(hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(self.dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(torch.nn.Linear(previous_dim, class_count))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def _rows_as_float32(features, indices: np.ndarray) -> np.ndarray:
        if sparse.issparse(features):
            batch = features[indices].toarray()
        elif hasattr(features, "iloc"):
            batch = features.iloc[indices].to_numpy()
        else:
            batch = np.asarray(features)[indices]
        return np.asarray(batch, dtype=np.float32)

    def fit(self, features, labels):
        torch = self._import_torch()
        self._require_cuda(torch)
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs와 batch_size는 1 이상이어야 합니다.")

        labels_array = np.asarray(labels)
        self.classes_, encoded_labels = np.unique(labels_array, return_inverse=True)
        self.n_features_in_ = int(features.shape[1])
        self._class_count = int(len(self.classes_))
        if self._class_count < 2:
            raise ValueError("PyTorch 분류 학습에는 두 개 이상의 클래스가 필요합니다.")

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        network = self._build_network(
            torch,
            self.n_features_in_,
            self._class_count,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_weights = None
        if self.class_weight == "balanced":
            counts = np.bincount(encoded_labels, minlength=self._class_count)
            weights = len(encoded_labels) / (self._class_count * counts)
            loss_weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        elif self.class_weight is not None:
            raise ValueError("class_weight는 'balanced' 또는 None만 지원합니다.")
        criterion = torch.nn.CrossEntropyLoss(
            weight=loss_weights,
            label_smoothing=self.label_smoothing,
        )

        generator = np.random.default_rng(self.seed)
        for epoch in range(self.epochs):
            network.train()
            permutation = generator.permutation(len(encoded_labels))
            epoch_loss = 0.0
            for start in range(0, len(permutation), self.batch_size):
                indices = permutation[start : start + self.batch_size]
                batch_x = torch.from_numpy(
                    self._rows_as_float32(features, indices)
                ).to(self.device)
                batch_y = torch.as_tensor(
                    encoded_labels[indices], dtype=torch.long, device=self.device
                )
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(network(batch_x), batch_y)
                loss.backward()
                if self.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        network.parameters(), self.gradient_clip_norm
                    )
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(indices)
            if self.verbose:
                print(
                    f"torch_epoch={epoch + 1} "
                    f"loss={epoch_loss / len(encoded_labels):.6f}"
                )

        self._state_dict = {
            key: value.detach().cpu()
            for key, value in network.state_dict().items()
        }
        del network
        torch.cuda.empty_cache()
        return self

    def _fitted_network(self, torch):
        if not hasattr(self, "_state_dict"):
            raise RuntimeError("모델을 먼저 fit 해야 합니다.")
        network = self._build_network(
            torch,
            self.n_features_in_,
            self._class_count,
        )
        network.load_state_dict(self._state_dict)
        network.to(self.device)
        network.eval()
        return network

    def predict_proba(self, features) -> np.ndarray:
        torch = self._import_torch()
        self._require_cuda(torch)
        network = self._fitted_network(torch)
        probabilities: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, features.shape[0], self.batch_size):
                indices = np.arange(start, min(start + self.batch_size, features.shape[0]))
                batch_x = torch.from_numpy(
                    self._rows_as_float32(features, indices)
                ).to(self.device)
                batch_probabilities = torch.softmax(network(batch_x), dim=1)
                probabilities.append(batch_probabilities.cpu().numpy())
        del network
        torch.cuda.empty_cache()
        return np.concatenate(probabilities, axis=0)

    def predict(self, features) -> np.ndarray:
        encoded = np.argmax(self.predict_proba(features), axis=1)
        return np.asarray(self.classes_[encoded])

    def get_params(self, deep: bool = True) -> dict:
        del deep
        return {
            "hidden_dims": self.hidden_dims,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "class_weight": self.class_weight,
            "label_smoothing": self.label_smoothing,
            "gradient_clip_norm": self.gradient_clip_norm,
            "device": self.device,
            "verbose": self.verbose,
            "seed": self.seed,
        }

    def set_params(self, **parameters):
        if parameters.get("device", self.device) != "cuda":
            raise ValueError("PyTorch TEST_007 모델은 device=cuda만 지원합니다.")
        for key, value in parameters.items():
            if not hasattr(self, key):
                raise ValueError(f"알 수 없는 파라미터입니다: {key}")
            setattr(self, key, value)
        return self
