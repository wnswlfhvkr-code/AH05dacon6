"""Fold-safe SVD와 공식 TabM API를 사용하는 GPU 전역 분류기입니다."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


class SVDTabMClassifier:
    """희소 입력을 fold 내부에서 축소해 TabM으로 학습하는 sklearn 어댑터."""

    def __init__(
        self,
        *,
        class_names: tuple[str, ...],
        svd_components: int = 128,
        k: int = 16,
        n_blocks: int = 2,
        d_block: int = 128,
        dropout: float = 0.1,
        epochs: int = 20,
        batch_size: int = 128,
        learning_rate: float = 2e-3,
        weight_decay: float = 3e-4,
        class_weight: str | None = "balanced",
        device: str = "cuda",
        verbose: bool = False,
        seed: int = 42,
    ) -> None:
        self.class_names = tuple(class_names)
        self.svd_components = svd_components
        self.k = k
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weight = class_weight
        self.device = device
        self.verbose = verbose
        self.seed = seed
        self._validate_params()

    def _validate_params(self) -> None:
        if self.device != "cuda":
            raise ValueError("TabM 모델은 device='cuda'만 지원합니다.")
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(
            self.class_names
        ):
            raise ValueError("class_names에는 중복 없는 전체 클래스 순서가 필요합니다.")
        for name in (
            "svd_components",
            "k",
            "n_blocks",
            "d_block",
            "epochs",
            "batch_size",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name}는 1 이상의 정수여야 합니다.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout은 0 이상 1 미만이어야 합니다.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate는 양수이고 weight_decay는 0 이상이어야 합니다.")
        if self.class_weight not in ("balanced", None):
            raise ValueError("class_weight는 'balanced' 또는 None만 지원합니다.")

    @staticmethod
    def _import_dependencies():
        try:
            import torch
            from tabm import TabM
        except ImportError as error:
            raise ImportError(
                "TabM 모델을 사용하려면 torch와 tabm==0.0.3이 필요합니다."
            ) from error
        return torch, TabM

    @staticmethod
    def _dense_float32(features) -> np.ndarray:
        if sparse.issparse(features):
            features = features.toarray()
        elif hasattr(features, "to_numpy"):
            features = features.to_numpy()
        result = np.asarray(features, dtype=np.float32)
        if result.ndim != 2:
            raise ValueError("features는 2차원 행렬이어야 합니다.")
        if not np.isfinite(result).all():
            raise ValueError("features에 NaN 또는 무한대가 있습니다.")
        return result

    @staticmethod
    def _seed_all(torch, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _fit_reducer(self, features) -> np.ndarray:
        if getattr(features, "ndim", 2) != 2:
            raise ValueError("features는 2차원 행렬이어야 합니다.")
        row_count, feature_count = features.shape
        if row_count < 2 or feature_count < 1:
            raise ValueError("TabM 학습에는 2개 이상의 행과 1개 이상의 피처가 필요합니다.")
        max_components = min(row_count - 1, feature_count - 1)
        if max_components >= 1 and feature_count > self.svd_components:
            self.reducer_ = TruncatedSVD(
                n_components=min(self.svd_components, max_components),
                algorithm="randomized",
                random_state=self.seed,
            )
            result = np.asarray(
                self.reducer_.fit_transform(features), dtype=np.float32
            )
            if not np.isfinite(result).all():
                raise ValueError("SVD 결과에 NaN 또는 무한대가 있습니다.")
            return result
        self.reducer_ = None
        return self._dense_float32(features)

    def _transform(self, features) -> np.ndarray:
        if not hasattr(self, "reducer_"):
            raise RuntimeError("모델을 먼저 fit 해야 합니다.")
        if getattr(features, "ndim", 2) != 2:
            raise ValueError("features는 2차원 행렬이어야 합니다.")
        if int(features.shape[1]) != self.n_features_in_:
            raise ValueError(
                f"피처 수가 다릅니다: expected={self.n_features_in_}, "
                f"observed={features.shape[1]}"
            )
        if self.reducer_ is None:
            return self._dense_float32(features)
        result = np.asarray(self.reducer_.transform(features), dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("SVD 결과에 NaN 또는 무한대가 있습니다.")
        return result

    def _build_model(self, TabM, input_dim: int, class_count: int):
        return TabM.make(
            n_num_features=input_dim,
            cat_cardinalities=None,
            d_out=class_count,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            dropout=self.dropout,
            k=self.k,
        )

    def fit(self, features, labels):
        self._validate_params()
        labels_array = np.asarray(labels).reshape(-1)
        if int(features.shape[0]) != len(labels_array):
            raise ValueError("features와 labels의 행 수가 다릅니다.")
        expected_classes = np.arange(len(self.class_names))
        observed_classes = np.unique(labels_array)
        if not np.array_equal(observed_classes, expected_classes):
            raise ValueError(
                "class_names 순서와 인코딩 레이블이 일치하지 않습니다: "
                f"expected={expected_classes.tolist()}, observed={observed_classes.tolist()}"
            )

        torch, TabM = self._import_dependencies()
        if not torch.cuda.is_available():
            raise RuntimeError("TabM 학습 및 추론에는 CUDA GPU가 필요합니다.")
        self._seed_all(torch, self.seed)
        dense_features = self._fit_reducer(features)
        self.n_features_in_ = int(features.shape[1])
        self._reduced_features_in_ = int(dense_features.shape[1])
        self.classes_ = expected_classes

        model = self._build_model(
            TabM, self._reduced_features_in_, len(self.classes_)
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_weights = None
        if self.class_weight == "balanced":
            counts = np.bincount(labels_array, minlength=len(self.classes_))
            weights = len(labels_array) / (len(self.classes_) * counts)
            loss_weights = torch.as_tensor(
                weights, dtype=torch.float32, device=self.device
            )
        criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)
        generator = np.random.default_rng(self.seed)

        encoded_labels = labels_array.astype(np.int64, copy=False)
        for epoch in range(self.epochs):
            model.train()
            permutation = generator.permutation(len(encoded_labels))
            epoch_loss = 0.0
            for start in range(0, len(permutation), self.batch_size):
                indices = permutation[start : start + self.batch_size]
                batch_x = torch.as_tensor(
                    dense_features[indices], dtype=torch.float32, device=self.device
                )
                batch_y = torch.as_tensor(
                    encoded_labels[indices], dtype=torch.long, device=self.device
                )
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x, None)
                targets = batch_y[:, None].expand(-1, self.k).reshape(-1)
                loss = criterion(logits.reshape(-1, len(self.classes_)), targets)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(indices)
            if self.verbose:
                print(
                    f"tabm_epoch={epoch + 1} "
                    f"loss={epoch_loss / len(encoded_labels):.6f}"
                )

        self._state_dict = {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        }
        del model, optimizer
        torch.cuda.empty_cache()
        return self

    def _fitted_model(self, torch, TabM):
        if not hasattr(self, "_state_dict"):
            raise RuntimeError("모델을 먼저 fit 해야 합니다.")
        model = self._build_model(
            TabM, self._reduced_features_in_, len(self.classes_)
        )
        model.load_state_dict(self._state_dict)
        model.to(self.device)
        model.eval()
        return model

    def predict_proba(self, features) -> np.ndarray:
        torch, TabM = self._import_dependencies()
        if not torch.cuda.is_available():
            raise RuntimeError("TabM 학습 및 추론에는 CUDA GPU가 필요합니다.")
        dense_features = self._transform(features)
        if len(dense_features) == 0:
            return np.empty((0, len(self.classes_)), dtype=np.float64)
        self._seed_all(torch, self.seed)
        model = self._fitted_model(torch, TabM)
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(dense_features), self.batch_size):
                batch_x = torch.as_tensor(
                    dense_features[start : start + self.batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                member_probabilities = torch.softmax(model(batch_x, None), dim=-1)
                chunks.append(member_probabilities.mean(dim=1).cpu().numpy())
        del model
        torch.cuda.empty_cache()
        probabilities = np.asarray(np.concatenate(chunks), dtype=np.float64)
        row_sums = probabilities.sum(axis=1)
        if not np.isfinite(probabilities).all() or not np.allclose(
            row_sums, 1.0, rtol=1e-5, atol=1e-6
        ):
            raise RuntimeError("TabM 예측 확률이 유한하지 않거나 행 합이 1이 아닙니다.")
        return probabilities

    def predict(self, features) -> np.ndarray:
        indices = np.argmax(self.predict_proba(features), axis=1)
        return np.asarray(self.classes_[indices]).reshape(-1)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "class_names": self.class_names,
            "svd_components": self.svd_components,
            "k": self.k,
            "n_blocks": self.n_blocks,
            "d_block": self.d_block,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "class_weight": self.class_weight,
            "device": self.device,
            "verbose": self.verbose,
            "seed": self.seed,
        }

    def set_params(self, **parameters):
        current = self.get_params()
        unknown = sorted(set(parameters) - set(current))
        if unknown:
            raise ValueError(f"알 수 없는 파라미터입니다: {unknown}")
        candidate = current | parameters
        previous = current
        try:
            for key, value in candidate.items():
                setattr(self, key, tuple(value) if key == "class_names" else value)
            self._validate_params()
        except Exception:
            for key, value in previous.items():
                setattr(self, key, value)
            raise
        return self

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if "_state_dict" in state:
            state["_state_dict"] = {
                key: value.detach().cpu()
                for key, value in state["_state_dict"].items()
            }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


def create_model(model_config: dict, seed: int) -> SVDTabMClassifier:
    return SVDTabMClassifier(
        class_names=tuple(model_config.get("class_names", ())),
        svd_components=model_config.get("svd_components", 128),
        k=model_config.get("k", 16),
        n_blocks=model_config.get("n_blocks", 2),
        d_block=model_config.get("d_block", 128),
        dropout=model_config.get("dropout", 0.1),
        epochs=model_config.get("epochs", 20),
        batch_size=model_config.get("batch_size", 128),
        learning_rate=model_config.get("learning_rate", 2e-3),
        weight_decay=model_config.get("weight_decay", 3e-4),
        class_weight=model_config.get("class_weight", "balanced"),
        device=model_config.get("device", "cuda"),
        verbose=model_config.get("verbose", False),
        seed=seed,
    )
