"""콤3 희소 피처를 fold-local SVD 후 RealMLP GPU로 학습합니다."""

from __future__ import annotations

import copy
from pathlib import Path
import shutil
import tempfile

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


class SVDRealMLPClassifier:
    """PyTabKit RealMLP의 누수 방지 SVD sklearn 어댑터."""

    def __init__(
        self,
        *,
        class_names: tuple[str, ...],
        svd_components: int = 128,
        device: str = "cuda:0",
        n_epochs: int = 32,
        batch_size: int = 256,
        predict_batch_size: int = 2048,
        val_fraction: float = 0.15,
        hidden_sizes: tuple[int, ...] | None = None,
        verbosity: int = 0,
        seed: int = 42,
    ) -> None:
        self.class_names = tuple(class_names)
        self.svd_components = int(svd_components)
        self.device = str(device)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.predict_batch_size = int(predict_batch_size)
        self.val_fraction = float(val_fraction)
        self.hidden_sizes = (
            None if hidden_sizes is None else tuple(int(value) for value in hidden_sizes)
        )
        self.verbosity = int(verbosity)
        self.seed = int(seed)
        self._validate_params()

    def _validate_params(self) -> None:
        if not self.device.startswith("cuda"):
            raise ValueError("RealMLP global은 CUDA device만 지원합니다.")
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(
            self.class_names
        ):
            raise ValueError("class_names에는 중복 없는 전체 클래스 순서가 필요합니다.")
        if self.svd_components < 1 or self.n_epochs < 1:
            raise ValueError("SVD 차원과 epoch는 1 이상이어야 합니다.")
        if self.batch_size < 1 or self.predict_batch_size < 1:
            raise ValueError("batch size는 1 이상이어야 합니다.")
        if not 0.0 < self.val_fraction < 0.5:
            raise ValueError("val_fraction은 0과 0.5 사이여야 합니다.")

    @staticmethod
    def _import_dependencies():
        try:
            import torch
            from pytabkit import RealMLP_TD_Classifier
        except ImportError as error:
            raise ImportError(
                "RealMLP를 사용하려면 `pip install pytabkit==1.7.3`이 필요합니다."
            ) from error
        return torch, RealMLP_TD_Classifier

    @staticmethod
    def _dense_float32(features) -> np.ndarray:
        if sparse.issparse(features):
            features = features.toarray()
        elif hasattr(features, "to_numpy"):
            features = features.to_numpy()
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("RealMLP 입력은 유한한 2차원 float32여야 합니다.")
        return values

    def _fit_reducer(self, features) -> np.ndarray:
        rows, columns = map(int, features.shape)
        if rows < 2 or columns < 1:
            raise ValueError("RealMLP 학습에는 2개 이상 행과 피처가 필요합니다.")
        max_components = min(rows - 1, columns - 1)
        if columns > self.svd_components and max_components >= 1:
            self.reducer_ = TruncatedSVD(
                n_components=min(self.svd_components, max_components),
                algorithm="randomized",
                random_state=self.seed,
            )
            values = self.reducer_.fit_transform(features)
        else:
            self.reducer_ = None
            values = self._dense_float32(features)
        return np.asarray(values, dtype=np.float32)

    def _transform(self, features) -> np.ndarray:
        if int(features.shape[1]) != self.n_features_in_:
            raise ValueError(
                f"피처 수가 다릅니다: expected={self.n_features_in_}, "
                f"observed={features.shape[1]}"
            )
        values = (
            self._dense_float32(features)
            if self.reducer_ is None
            else np.asarray(self.reducer_.transform(features), dtype=np.float32)
        )
        if not np.isfinite(values).all():
            raise ValueError("RealMLP SVD 결과에 NaN 또는 무한대가 있습니다.")
        return values

    def _move_model(self, device: str) -> None:
        if hasattr(self, "classifier_") and hasattr(self.classifier_, "to"):
            self.classifier_.to(device)

    def fit(self, features, labels):
        self._validate_params()
        labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
        expected = np.arange(len(self.class_names), dtype=np.int64)
        if int(features.shape[0]) != len(labels_array):
            raise ValueError("features와 labels 행 수가 다릅니다.")
        if not np.array_equal(np.unique(labels_array), expected):
            raise ValueError("RealMLP 레이블과 class_names 순서가 일치하지 않습니다.")

        torch, classifier_type = self._import_dependencies()
        if not torch.cuda.is_available():
            raise RuntimeError("RealMLP global 실행에는 CUDA GPU가 필요합니다.")
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        dense = self._fit_reducer(features)
        kwargs = {
            "device": self.device,
            "random_state": self.seed,
            "n_cv": 1,
            "n_refit": 0,
            "val_fraction": self.val_fraction,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "predict_batch_size": self.predict_batch_size,
            "val_metric_name": "cross_entropy",
            "use_ls": False,
            "verbosity": self.verbosity,
        }
        if self.hidden_sizes is not None:
            kwargs["hidden_sizes"] = list(self.hidden_sizes)
        temporary_folder = Path(tempfile.mkdtemp(prefix="realmlp_pytabkit_"))
        kwargs["tmp_folder"] = temporary_folder
        self.classifier_ = classifier_type(**kwargs)
        try:
            self.classifier_.fit(dense, labels_array)
        finally:
            shutil.rmtree(temporary_folder, ignore_errors=True)
        self.classes_ = np.asarray(self.classifier_.classes_, dtype=np.int64)
        if not np.array_equal(self.classes_, expected):
            raise RuntimeError("RealMLP 확률 열 순서가 class_names와 다릅니다.")
        self.n_features_in_ = int(features.shape[1])
        return self

    def predict_proba(self, features) -> np.ndarray:
        if not hasattr(self, "classifier_"):
            raise RuntimeError("RealMLP를 먼저 fit 해야 합니다.")
        self._move_model(self.device)
        probabilities = np.asarray(
            self.classifier_.predict_proba(self._transform(features)),
            dtype=np.float64,
        )
        if probabilities.shape != (int(features.shape[0]), len(self.class_names)):
            raise RuntimeError("RealMLP 확률 shape가 잘못됐습니다.")
        row_sums = probabilities.sum(axis=1)
        if not np.isfinite(probabilities).all() or np.any(row_sums <= 0):
            raise RuntimeError("RealMLP가 유효하지 않은 확률을 반환했습니다.")
        return probabilities / row_sums[:, None]

    def predict(self, features) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        classifier = state.get("classifier_")
        if classifier is not None:
            try:
                classifier = copy.deepcopy(classifier)
                classifier.to("cpu")
            except Exception:
                self._move_model("cpu")
                classifier = self.classifier_
            state["classifier_"] = classifier
        return state

    def get_params(self, deep: bool = True) -> dict:
        del deep
        return {
            "class_names": self.class_names,
            "svd_components": self.svd_components,
            "device": self.device,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "predict_batch_size": self.predict_batch_size,
            "val_fraction": self.val_fraction,
            "hidden_sizes": self.hidden_sizes,
            "verbosity": self.verbosity,
            "seed": self.seed,
        }

    def set_params(self, **parameters):
        valid = set(self.get_params())
        unknown = sorted(set(parameters) - valid)
        if unknown:
            raise ValueError(f"알 수 없는 파라미터입니다: {unknown}")
        for key, value in parameters.items():
            if key == "class_names":
                value = tuple(value)
            elif key == "hidden_sizes" and value is not None:
                value = tuple(value)
            setattr(self, key, value)
        self._validate_params()
        return self


def create_model(model_config: dict, seed: int) -> SVDRealMLPClassifier:
    return SVDRealMLPClassifier(
        class_names=tuple(model_config.get("class_names", ())),
        svd_components=model_config.get("svd_components", 128),
        device=model_config.get("device", "cuda:0"),
        n_epochs=model_config.get("n_epochs", 32),
        batch_size=model_config.get("batch_size", 256),
        predict_batch_size=model_config.get("predict_batch_size", 2048),
        val_fraction=model_config.get("val_fraction", 0.15),
        hidden_sizes=(
            None
            if model_config.get("hidden_sizes") is None
            else tuple(model_config["hidden_sizes"])
        ),
        verbosity=model_config.get("verbosity", 0),
        seed=seed,
    )
