"""콤3 희소 피처를 fold-local SVD 후 upstream xRFM CPU로 학습합니다."""

from __future__ import annotations

import os

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split


class SVDXRFMClassifier:
    """xRFM v0.4.5의 CPU 전용 sklearn 스타일 어댑터."""

    def __init__(
        self,
        *,
        class_names: tuple[str, ...],
        svd_components: int = 128,
        device: str = "cpu",
        validation_fraction: float = 0.15,
        iterations: int = 1,
        max_leaf_size: int = 512,
        matrix_batch_size: int = 128,
        regularization: float = 1e-3,
        bandwidth: float = 10.0,
        time_limit_seconds: int = 300,
        n_threads: int | None = None,
        verbose: bool = False,
        seed: int = 42,
    ) -> None:
        self.class_names = tuple(class_names)
        self.svd_components = int(svd_components)
        self.device = str(device)
        self.validation_fraction = float(validation_fraction)
        self.iterations = int(iterations)
        self.max_leaf_size = int(max_leaf_size)
        self.matrix_batch_size = int(matrix_batch_size)
        self.regularization = float(regularization)
        self.bandwidth = float(bandwidth)
        self.time_limit_seconds = int(time_limit_seconds)
        self.n_threads = None if n_threads is None else int(n_threads)
        self.verbose = bool(verbose)
        self.seed = int(seed)
        self._validate_params()

    def _validate_params(self) -> None:
        if self.device != "cpu":
            raise ValueError("6번 xRFM은 사용자 지시에 따라 device='cpu'만 지원합니다.")
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(
            self.class_names
        ):
            raise ValueError("class_names에는 중복 없는 전체 클래스 순서가 필요합니다.")
        if self.svd_components < 1 or self.iterations < 1:
            raise ValueError("SVD 차원과 xRFM iteration은 1 이상이어야 합니다.")
        if self.max_leaf_size < 2 or self.matrix_batch_size < 1:
            raise ValueError("xRFM leaf/batch 크기가 잘못됐습니다.")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction은 0과 0.5 사이여야 합니다.")

    @staticmethod
    def _import_classifier():
        try:
            from xrfm import xRFM
        except ImportError as error:
            raise ImportError(
                "xRFM을 사용하려면 `pip install xrfm==0.4.5`가 필요합니다."
            ) from error
        return xRFM

    @staticmethod
    def _dense_float32(features) -> np.ndarray:
        if sparse.issparse(features):
            features = features.toarray()
        elif hasattr(features, "to_numpy"):
            features = features.to_numpy()
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("xRFM 입력은 유한한 2차원 float32여야 합니다.")
        return values

    def _fit_reducer(self, features) -> np.ndarray:
        rows, columns = map(int, features.shape)
        max_components = min(rows - 1, columns - 1)
        if max_components < 1:
            raise ValueError("xRFM 학습에는 두 행 이상이 필요합니다.")
        if columns > self.svd_components:
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
            raise ValueError("xRFM 입력 피처 수가 학습 시점과 다릅니다.")
        values = (
            self._dense_float32(features)
            if self.reducer_ is None
            else np.asarray(self.reducer_.transform(features), dtype=np.float32)
        )
        if not np.isfinite(values).all():
            raise ValueError("xRFM SVD 결과에 NaN 또는 무한대가 있습니다.")
        return values

    def fit(self, features, labels):
        self._validate_params()
        labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
        expected = np.arange(len(self.class_names), dtype=np.int64)
        if int(features.shape[0]) != len(labels_array):
            raise ValueError("features와 labels 행 수가 다릅니다.")
        if not np.array_equal(np.unique(labels_array), expected):
            raise ValueError("xRFM labels와 class_names 순서가 일치하지 않습니다.")
        dense = self._fit_reducer(features)
        fit_x, valid_x, fit_y, valid_y = train_test_split(
            dense,
            labels_array,
            test_size=self.validation_fraction,
            random_state=self.seed,
            stratify=labels_array,
        )
        parameters = {
            "model": {
                "kernel": "l2",
                "bandwidth": self.bandwidth,
                "exponent": 1.0,
                "diag": True,
                "bandwidth_mode": "constant",
            },
            "fit": {
                "reg": self.regularization,
                "iters": self.iterations,
                "M_batch_size": self.matrix_batch_size,
                "verbose": self.verbose,
                "early_stop_rfm": True,
            },
        }
        classifier_type = self._import_classifier()
        self.classifier_ = classifier_type(
            rfm_params=parameters,
            max_leaf_size=self.max_leaf_size,
            device="cpu",
            tuning_metric="logloss",
            classification_mode="prevalence",
            split_method="top_vector_agop_on_subset",
            time_limit_s=self.time_limit_seconds,
            n_threads=(
                self.n_threads
                if self.n_threads is not None
                else max(1, (os.cpu_count() or 2) // 2)
            ),
            random_state=self.seed,
            verbose=self.verbose,
        )
        self.classifier_.fit(fit_x, fit_y, valid_x, valid_y)
        self.classes_ = expected
        self.n_features_in_ = int(features.shape[1])
        return self

    def predict_proba(self, features) -> np.ndarray:
        if not hasattr(self, "classifier_"):
            raise RuntimeError("xRFM을 먼저 fit 해야 합니다.")
        probabilities = np.asarray(
            self.classifier_.predict_proba(self._transform(features)),
            dtype=np.float64,
        )
        if probabilities.shape != (int(features.shape[0]), len(self.class_names)):
            raise RuntimeError("xRFM 확률 shape가 class_names와 다릅니다.")
        row_sums = probabilities.sum(axis=1)
        if not np.isfinite(probabilities).all() or np.any(row_sums <= 0):
            raise RuntimeError("xRFM 확률이 유효하지 않습니다.")
        return probabilities / row_sums[:, None]

    def predict(self, features) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        classifier = state.get("classifier_")
        if classifier is not None and hasattr(classifier, "to"):
            classifier.to("cpu")
        return state

    def get_params(self, deep: bool = True) -> dict:
        del deep
        return {
            "class_names": self.class_names,
            "svd_components": self.svd_components,
            "device": self.device,
            "validation_fraction": self.validation_fraction,
            "iterations": self.iterations,
            "max_leaf_size": self.max_leaf_size,
            "matrix_batch_size": self.matrix_batch_size,
            "regularization": self.regularization,
            "bandwidth": self.bandwidth,
            "time_limit_seconds": self.time_limit_seconds,
            "n_threads": self.n_threads,
            "verbose": self.verbose,
            "seed": self.seed,
        }

    def set_params(self, **parameters):
        valid = set(self.get_params())
        unknown = sorted(set(parameters) - valid)
        if unknown:
            raise ValueError(f"알 수 없는 파라미터입니다: {unknown}")
        for key, value in parameters.items():
            setattr(self, key, tuple(value) if key == "class_names" else value)
        self._validate_params()
        return self


def create_model(model_config: dict, seed: int) -> SVDXRFMClassifier:
    return SVDXRFMClassifier(
        class_names=tuple(model_config.get("class_names", ())),
        svd_components=model_config.get("svd_components", 128),
        device=model_config.get("device", "cpu"),
        validation_fraction=model_config.get("validation_fraction", 0.15),
        iterations=model_config.get("iterations", 1),
        max_leaf_size=model_config.get("max_leaf_size", 512),
        matrix_batch_size=model_config.get("matrix_batch_size", 128),
        regularization=model_config.get("regularization", 1e-3),
        bandwidth=model_config.get("bandwidth", 10.0),
        time_limit_seconds=model_config.get("time_limit_seconds", 300),
        n_threads=model_config.get("n_threads"),
        verbose=model_config.get("verbose", False),
        seed=seed,
    )
