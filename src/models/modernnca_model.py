"""Fold-local SVD와 TALENT ModernNCA를 결합한 sklearn 호환 분류기."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


TALENT_COMMIT = "08301d670a7c854bcf3a73298763484ba58eecdb"
TALENT_INSTALL = (
    "pip install git+https://github.com/LAMDA-Tabular/TALENT.git@"
    f"{TALENT_COMMIT}"
)


class SVDModernNCAClassifier:
    """Sparse input -> train-only SVD -> CUDA ModernNCA classifier.

    The fitted candidate bank contains only the rows passed to :meth:`fit`.
    During each optimization step, the current mini-batch is removed from the
    candidate argument exactly as in TALENT's official training method.
    """

    def __init__(
        self,
        *,
        class_names: tuple[str, ...],
        svd_components: int = 128,
        dim: int = 64,
        dropout: float = 0.0,
        d_block: int = 128,
        n_blocks: int = 1,
        temperature: float = 1.0,
        sample_rate: float = 0.8,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 5,
        batch_size: int = 128,
        predict_batch_size: int = 512,
        class_weight: str = "balanced",
        device: str = "cuda",
        verbose: bool = False,
        seed: int = 42,
    ) -> None:
        self.class_names = tuple(class_names)
        self.svd_components = svd_components
        self.dim = dim
        self.dropout = dropout
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.temperature = temperature
        self.sample_rate = sample_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size
        self.class_weight = class_weight
        self.device = device
        self.verbose = verbose
        self.seed = seed
        self._validate_params()

    def _validate_params(self) -> None:
        if self.device != "cuda":
            raise ValueError("ModernNCA는 device='cuda'만 지원합니다.")
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names에는 중복 없는 클래스 순서가 필요합니다.")
        integer_params = {
            "svd_components": self.svd_components,
            "dim": self.dim,
            "d_block": self.d_block,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "predict_batch_size": self.predict_batch_size,
        }
        if any(isinstance(value, bool) or int(value) != value or value < 1 for value in integer_params.values()):
            raise ValueError("SVD/차원/epoch/batch 파라미터는 모두 1 이상의 정수여야 합니다.")
        if isinstance(self.n_blocks, bool) or int(self.n_blocks) != self.n_blocks or self.n_blocks < 0:
            raise ValueError("n_blocks는 0 이상의 정수여야 합니다.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout은 0 이상 1 미만이어야 합니다.")
        if self.temperature <= 0 or not 0 < self.sample_rate <= 1:
            raise ValueError("temperature는 양수이고 sample_rate는 (0, 1] 범위여야 합니다.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate는 양수이고 weight_decay는 음수가 아니어야 합니다.")
        if self.class_weight != "balanced":
            raise ValueError("ModernNCA는 class_weight='balanced'만 지원합니다.")

    @staticmethod
    def _import_dependencies():
        try:
            torch = importlib.import_module("torch")
            module = importlib.import_module("TALENT.model.models.modernNCA")
            model_type = module.ModernNCA
        except (ImportError, AttributeError) as error:
            raise ImportError(
                "ModernNCA를 사용하려면 고정된 TALENT upstream을 설치하세요: "
                f"`{TALENT_INSTALL}`"
            ) from error
        return torch, model_type

    @staticmethod
    def _dense_float32(features) -> np.ndarray:
        if sparse.issparse(features):
            features = features.toarray()
        elif hasattr(features, "to_numpy"):
            features = features.to_numpy()
        dense = np.asarray(features, dtype=np.float32)
        if dense.ndim != 2 or not np.isfinite(dense).all():
            raise ValueError("features는 유한한 값으로 구성된 2차원 행렬이어야 합니다.")
        return dense

    def _fit_reducer(self, features) -> np.ndarray:
        if getattr(features, "ndim", 2) != 2:
            raise ValueError("features는 2차원 행렬이어야 합니다.")
        row_count, feature_count = features.shape
        if row_count < 2 or feature_count < 1:
            raise ValueError("ModernNCA 학습에는 2개 이상의 행과 1개 이상의 피처가 필요합니다.")
        component_count = min(self.svd_components, row_count - 1, feature_count - 1)
        if component_count >= 1 and component_count < feature_count:
            self.reducer_ = TruncatedSVD(
                n_components=component_count,
                algorithm="randomized",
                random_state=self.seed,
            )
            reduced = self.reducer_.fit_transform(features)
            return self._dense_float32(reduced)
        self.reducer_ = None
        return self._dense_float32(features)

    def _transform(self, features) -> np.ndarray:
        if not hasattr(self, "reducer_"):
            raise RuntimeError("모델을 먼저 fit 해야 합니다.")
        if getattr(features, "ndim", 2) != 2 or int(features.shape[1]) != self.n_features_in_:
            observed = getattr(features, "shape", (None, None))[1]
            raise ValueError(
                f"피처 수가 다릅니다: expected={self.n_features_in_}, observed={observed}"
            )
        transformed = features if self.reducer_ is None else self.reducer_.transform(features)
        return self._dense_float32(transformed)

    def _new_model(self, model_type, input_dimension: int):
        return model_type(
            d_in=input_dimension,
            d_num=input_dimension,
            d_out=len(self.class_names),
            dim=self.dim,
            dropout=self.dropout,
            d_block=self.d_block,
            n_blocks=self.n_blocks,
            num_embeddings=None,
            temperature=self.temperature,
            sample_rate=self.sample_rate,
        )

    def _check_cuda(self, torch) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("ModernNCA 학습 및 추론에는 CUDA GPU가 필요합니다.")

    def fit(self, features, labels):
        self._validate_params()
        labels_array = np.asarray(labels).reshape(-1)
        if int(features.shape[0]) != len(labels_array):
            raise ValueError("features와 labels의 행 수가 다릅니다.")
        expected_classes = np.arange(len(self.class_names))
        if not np.array_equal(np.unique(labels_array), expected_classes):
            raise ValueError(
                "class_names 순서와 인코딩 레이블이 일치하지 않습니다: "
                f"expected={expected_classes.tolist()}, observed={np.unique(labels_array).tolist()}"
            )
        if len(labels_array) < 2:
            raise ValueError("ModernNCA 학습에는 2개 이상의 행이 필요합니다.")

        torch, model_type = self._import_dependencies()
        self._check_cuda(torch)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        candidate_features = self._fit_reducer(features)
        self.n_features_in_ = int(features.shape[1])
        self.candidate_features_ = np.ascontiguousarray(candidate_features, dtype=np.float32)
        self.candidate_labels_ = np.ascontiguousarray(labels_array, dtype=np.int64)
        self.classes_ = expected_classes

        self.model_ = self._new_model(model_type, candidate_features.shape[1]).to("cuda")
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        candidate_x = torch.as_tensor(self.candidate_features_, device="cuda")
        candidate_y = torch.as_tensor(self.candidate_labels_, device="cuda")
        class_counts = np.bincount(self.candidate_labels_, minlength=len(self.class_names))
        balanced = len(labels_array) / (len(self.class_names) * class_counts)
        loss_weight = torch.as_tensor(balanced, dtype=torch.float32, device="cuda")
        all_indices = torch.arange(len(labels_array), device="cuda")

        for epoch in range(self.epochs):
            self.model_.train()
            permutation = torch.randperm(len(labels_array), device="cuda")
            epoch_loss = 0.0
            steps = 0
            for batch_indices in permutation.split(self.batch_size):
                candidate_indices = all_indices[~torch.isin(all_indices, batch_indices)]
                log_probabilities = self.model_(
                    x=candidate_x[batch_indices],
                    y=candidate_y[batch_indices],
                    candidate_x=candidate_x[candidate_indices],
                    candidate_y=candidate_y[candidate_indices],
                    is_train=True,
                )
                loss = torch.nn.functional.nll_loss(
                    log_probabilities,
                    candidate_y[batch_indices],
                    weight=loss_weight,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("ModernNCA 학습 중 유한하지 않은 NLL loss가 발생했습니다.")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                steps += 1
            if self.verbose:
                print(f"ModernNCA epoch {epoch + 1}/{self.epochs} loss={epoch_loss / steps:.6f}")
        self.model_.eval()
        self._model_state_dict_ = None
        return self

    def _ensure_model_loaded(self):
        if hasattr(self, "model_"):
            return
        if not hasattr(self, "_model_state_dict_") or self._model_state_dict_ is None:
            raise RuntimeError("모델을 먼저 fit 해야 합니다.")
        torch, model_type = self._import_dependencies()
        self._check_cuda(torch)
        model = self._new_model(model_type, self.candidate_features_.shape[1])
        model.load_state_dict(self._model_state_dict_)
        self.model_ = model.to("cuda")
        self.model_.eval()

    @staticmethod
    def _probabilities_from_log(torch, log_probabilities) -> np.ndarray:
        if log_probabilities.ndim != 2 or not torch.isfinite(log_probabilities).all():
            raise RuntimeError("ModernNCA가 유효한 2차원 log-probability를 반환하지 않았습니다.")
        probabilities = torch.exp(log_probabilities)
        row_sums = probabilities.sum(dim=1, keepdim=True)
        if (
            not torch.isfinite(probabilities).all()
            or torch.any(probabilities < 0)
            or torch.any(row_sums <= 0)
            or not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)
        ):
            raise RuntimeError("ModernNCA log-probability를 exp한 결과가 유효한 확률이 아닙니다.")
        probabilities = probabilities / row_sums
        return probabilities.detach().cpu().numpy().astype(np.float64, copy=False)

    def predict_proba(self, features) -> np.ndarray:
        self._ensure_model_loaded()
        torch, _ = self._import_dependencies()
        transformed = self._transform(features)
        candidate_x = torch.as_tensor(self.candidate_features_, device="cuda")
        candidate_y = torch.as_tensor(self.candidate_labels_, device="cuda")
        outputs: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(transformed), self.predict_batch_size):
                query = torch.as_tensor(
                    transformed[start : start + self.predict_batch_size], device="cuda"
                )
                log_probabilities = self.model_(
                    x=query,
                    y=None,
                    candidate_x=candidate_x,
                    candidate_y=candidate_y,
                    is_train=False,
                )
                outputs.append(self._probabilities_from_log(torch, log_probabilities))
        if not outputs:
            return np.empty((0, len(self.class_names)), dtype=np.float64)
        probabilities = np.concatenate(outputs, axis=0)
        if probabilities.shape != (len(transformed), len(self.class_names)):
            raise RuntimeError("ModernNCA 확률 행렬 shape이 기대값과 다릅니다.")
        return probabilities

    def predict(self, features) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "class_names": self.class_names,
            "svd_components": self.svd_components,
            "dim": self.dim,
            "dropout": self.dropout,
            "d_block": self.d_block,
            "n_blocks": self.n_blocks,
            "temperature": self.temperature,
            "sample_rate": self.sample_rate,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "predict_batch_size": self.predict_batch_size,
            "class_weight": self.class_weight,
            "device": self.device,
            "verbose": self.verbose,
            "seed": self.seed,
        }

    def set_params(self, **parameters):
        unknown = sorted(set(parameters) - set(self.get_params()))
        if unknown:
            raise ValueError(f"알 수 없는 파라미터입니다: {unknown}")
        previous = self.get_params()
        try:
            for key, value in parameters.items():
                setattr(self, key, tuple(value) if key == "class_names" else value)
            self._validate_params()
        except Exception:
            for key, value in previous.items():
                setattr(self, key, value)
            raise
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        model = state.pop("model_", None)
        if model is not None:
            state["_model_state_dict_"] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def create_model(model_config: dict, seed: int) -> SVDModernNCAClassifier:
    class_names = tuple(model_config.get("class_names", ()))
    if len(class_names) != 26:
        raise ValueError("TEST_007 ModernNCA 전역 모델은 정확히 26개 class_names가 필요합니다.")
    return SVDModernNCAClassifier(
        class_names=class_names,
        svd_components=model_config.get("svd_components", 128),
        dim=model_config.get("dim", 64),
        dropout=model_config.get("dropout", 0.0),
        d_block=model_config.get("d_block", 128),
        n_blocks=model_config.get("n_blocks", 1),
        temperature=model_config.get("temperature", 1.0),
        sample_rate=model_config.get("sample_rate", 0.8),
        learning_rate=model_config.get("learning_rate", 1e-3),
        weight_decay=model_config.get("weight_decay", 1e-4),
        epochs=model_config.get("epochs", 5),
        batch_size=model_config.get("batch_size", 128),
        predict_batch_size=model_config.get("predict_batch_size", 512),
        class_weight=model_config.get("class_weight", "balanced"),
        device=model_config.get("device", "cuda"),
        verbose=model_config.get("verbose", False),
        seed=seed,
    )
