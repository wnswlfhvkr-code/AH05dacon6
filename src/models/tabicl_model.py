"""희소 피처를 fold 내부 SVD 후 TabICLv2 GPU로 학습합니다."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


TABICL_V2_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"


class SVDTabICLClassifier:
    """TabICLv2용 누수 방지 SVD 및 CUDA 전용 sklearn 호환 어댑터."""

    def __init__(
        self,
        *,
        class_names: tuple[str, ...],
        svd_components: int = 128,
        ensemble_size: int = 8,
        batch_size: int = 1,
        support_many_classes: bool = True,
        checkpoint_version: str = TABICL_V2_CHECKPOINT,
        device: str = "cuda",
        use_amp: str | bool = "auto",
        offload_mode: str = "auto",
        verbose: bool = False,
        seed: int = 42,
    ) -> None:
        self.class_names = tuple(class_names)
        self.svd_components = svd_components
        self.ensemble_size = ensemble_size
        self.batch_size = batch_size
        self.support_many_classes = support_many_classes
        self.checkpoint_version = checkpoint_version
        self.device = device
        self.use_amp = use_amp
        self.offload_mode = offload_mode
        self.verbose = verbose
        self.seed = seed
        self._validate_params()

    def _validate_params(self) -> None:
        if self.device != "cuda":
            raise ValueError("TabICLv2 모델은 device='cuda'만 지원합니다.")
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(
            self.class_names
        ):
            raise ValueError("class_names에는 중복 없는 전체 클래스 순서가 필요합니다.")
        if self.svd_components < 1:
            raise ValueError("svd_components는 1 이상이어야 합니다.")
        if self.ensemble_size < 1:
            raise ValueError("ensemble_size는 1 이상이어야 합니다.")
        if self.batch_size < 1:
            raise ValueError("batch_size는 1 이상이어야 합니다.")
        if self.support_many_classes is not True:
            raise ValueError("26클래스 TabICLv2는 support_many_classes=True가 필요합니다.")
        if self.checkpoint_version != TABICL_V2_CHECKPOINT:
            raise ValueError(
                "TabICLv2 checkpoint_version은 "
                f"{TABICL_V2_CHECKPOINT!r}이어야 합니다."
            )

    @staticmethod
    def _import_dependencies():
        try:
            import torch
            from tabicl import TabICLClassifier
        except ImportError as error:
            raise ImportError(
                "TabICLv2 모델을 사용하려면 `pip install tabicl`을 실행하세요."
            ) from error
        return torch, TabICLClassifier

    @staticmethod
    def _dense_float32(features) -> np.ndarray:
        if sparse.issparse(features):
            features = features.toarray()
        elif hasattr(features, "to_numpy"):
            features = features.to_numpy()
        return np.asarray(features, dtype=np.float32)

    def _fit_reducer(self, features) -> np.ndarray:
        if getattr(features, "ndim", 2) != 2:
            raise ValueError("features는 2차원 행렬이어야 합니다.")
        row_count, feature_count = features.shape
        if row_count < 2 or feature_count < 1:
            raise ValueError("TabICLv2 학습에는 2개 이상의 행과 1개 이상의 피처가 필요합니다.")
        max_components = min(row_count - 1, feature_count - 1)
        if max_components >= 1 and feature_count > self.svd_components:
            self.reducer_ = TruncatedSVD(
                n_components=min(self.svd_components, max_components),
                algorithm="randomized",
                random_state=self.seed,
            )
            return np.asarray(self.reducer_.fit_transform(features), dtype=np.float32)
        self.reducer_ = None
        return self._dense_float32(features)

    def _transform(self, features) -> np.ndarray:
        if not hasattr(self, "reducer_"):
            raise RuntimeError("모델을 먼저 fit 해야 합니다.")
        if int(features.shape[1]) != self.n_features_in_:
            raise ValueError(
                f"피처 수가 다릅니다: expected={self.n_features_in_}, "
                f"observed={features.shape[1]}"
            )
        if self.reducer_ is None:
            return self._dense_float32(features)
        return np.asarray(self.reducer_.transform(features), dtype=np.float32)

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

        torch, classifier_type = self._import_dependencies()
        if not torch.cuda.is_available():
            raise RuntimeError("TabICLv2 학습 및 추론에는 CUDA GPU가 필요합니다.")

        dense_features = self._fit_reducer(features)
        self.classifier_ = classifier_type(
            n_estimators=self.ensemble_size,
            support_many_classes=True,
            batch_size=self.batch_size,
            checkpoint_version=self.checkpoint_version,
            device="cuda",
            use_amp=self.use_amp,
            offload_mode=self.offload_mode,
            random_state=self.seed,
            verbose=self.verbose,
        )
        self.classifier_.fit(dense_features, labels_array)
        self.classes_ = np.asarray(self.classifier_.classes_)
        if not np.array_equal(self.classes_, expected_classes):
            raise RuntimeError(
                "TabICLv2의 클래스 순서가 class_names 설정과 일치하지 않습니다."
            )
        self.n_features_in_ = int(features.shape[1])
        return self

    def predict_proba(self, features) -> np.ndarray:
        if not hasattr(self, "classifier_"):
            raise RuntimeError("모델을 먼저 fit 해야 합니다.")
        probabilities = np.asarray(
            self.classifier_.predict_proba(self._transform(features)),
            dtype=np.float64,
        )
        if probabilities.shape[1] != len(self.class_names):
            raise RuntimeError("TabICLv2 확률 열 수가 class_names와 일치하지 않습니다.")
        return probabilities

    def predict(self, features) -> np.ndarray:
        encoded = np.argmax(self.predict_proba(features), axis=1)
        return np.asarray(self.classes_[encoded]).reshape(-1)

    def get_params(self, deep: bool = True) -> dict:
        del deep
        return {
            "class_names": self.class_names,
            "svd_components": self.svd_components,
            "ensemble_size": self.ensemble_size,
            "batch_size": self.batch_size,
            "support_many_classes": self.support_many_classes,
            "checkpoint_version": self.checkpoint_version,
            "device": self.device,
            "use_amp": self.use_amp,
            "offload_mode": self.offload_mode,
            "verbose": self.verbose,
            "seed": self.seed,
        }

    def set_params(self, **parameters):
        valid = set(self.get_params())
        unknown = sorted(set(parameters) - valid)
        if unknown:
            raise ValueError(f"알 수 없는 파라미터입니다: {unknown}")
        candidate = self.get_params()
        candidate.update(parameters)
        if candidate["device"] != "cuda":
            raise ValueError("TabICLv2 모델은 device='cuda'만 지원합니다.")
        if candidate["support_many_classes"] is not True:
            raise ValueError("26클래스 TabICLv2는 support_many_classes=True가 필요합니다.")
        previous = self.get_params()
        try:
            for key, value in candidate.items():
                setattr(self, key, tuple(value) if key == "class_names" else value)
            self._validate_params()
        except Exception:
            for key, value in previous.items():
                setattr(self, key, value)
            raise
        return self


def create_model(model_config: dict, seed: int) -> SVDTabICLClassifier:
    return SVDTabICLClassifier(
        class_names=tuple(model_config.get("class_names", ())),
        svd_components=model_config.get("svd_components", 128),
        ensemble_size=model_config.get("ensemble_size", 8),
        batch_size=model_config.get("batch_size", 1),
        support_many_classes=model_config.get("support_many_classes", True),
        checkpoint_version=model_config.get(
            "checkpoint_version", TABICL_V2_CHECKPOINT
        ),
        device=model_config.get("device", "cuda"),
        use_amp=model_config.get("use_amp", "auto"),
        offload_mode=model_config.get("offload_mode", "auto"),
        verbose=model_config.get("verbose", False),
        seed=seed,
    )
