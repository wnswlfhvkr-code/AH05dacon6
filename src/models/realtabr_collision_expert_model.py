"""XGBoost 전역 모델에 CPU RealTabR 충돌 암종 전문가를 결합합니다."""

from __future__ import annotations

from pathlib import Path
import pickle
import shutil
import tempfile

from joblib.externals import cloudpickle
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


DEFAULT_COLLISION_PAIRS = (("GBMLGG", "LGG"), ("KIPAN", "KIRC"))
_INFERENCE_STATE_VERSION = 1


def _select_rows(features, selector: np.ndarray):
    positions = np.flatnonzero(selector) if selector.dtype == bool else selector
    if hasattr(features, "iloc"):
        return features.iloc[positions]
    return features[positions]


def _dense_float32(features) -> np.ndarray:
    if sparse.issparse(features):
        features = features.toarray()
    elif hasattr(features, "to_numpy"):
        features = features.to_numpy()
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("RealTabR 입력은 유한한 2차원 float32여야 합니다.")
    return values


class SVDRealTabRExpert:
    """충돌 쌍의 train 행에서만 SVD와 RealTabR retrieval bank를 적합합니다."""

    def __init__(
        self,
        *,
        svd_components: int,
        device: str,
        n_epochs: int,
        batch_size: int,
        eval_batch_size: int,
        context_size: int,
        d_main: int,
        patience: int,
        val_fraction: float,
        verbosity: int,
        seed: int,
    ) -> None:
        self.svd_components = int(svd_components)
        self.device = str(device)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.eval_batch_size = int(eval_batch_size)
        self.context_size = int(context_size)
        self.d_main = int(d_main)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self.verbosity = int(verbosity)
        self.seed = int(seed)
        if self.device != "cpu":
            raise ValueError(
                "Windows native RealTabR는 FAISS-GPU가 없어 device='cpu'만 지원합니다."
            )

    @staticmethod
    def _import_classifier():
        try:
            from pytabkit import RealTabR_D_Classifier
        except ImportError as error:
            raise ImportError(
                "RealTabR를 사용하려면 `pip install pytabkit==1.7.3 "
                "faiss-cpu==1.14.3`이 필요합니다."
            ) from error
        return RealTabR_D_Classifier

    def _fit_reducer(self, features) -> np.ndarray:
        rows, columns = map(int, features.shape)
        max_components = min(rows - 1, columns - 1)
        if max_components < 1:
            raise ValueError("RealTabR pair에는 두 행 이상이 필요합니다.")
        if columns > self.svd_components:
            self.reducer_ = TruncatedSVD(
                n_components=min(self.svd_components, max_components),
                algorithm="randomized",
                random_state=self.seed,
            )
            values = self.reducer_.fit_transform(features)
        else:
            self.reducer_ = None
            values = _dense_float32(features)
        return np.asarray(values, dtype=np.float32)

    def _transform(self, features) -> np.ndarray:
        if int(features.shape[1]) != self.n_features_in_:
            raise ValueError("RealTabR expert 입력 피처 수가 학습 시점과 다릅니다.")
        values = (
            _dense_float32(features)
            if self.reducer_ is None
            else np.asarray(self.reducer_.transform(features), dtype=np.float32)
        )
        if not np.isfinite(values).all():
            raise ValueError("RealTabR SVD 결과에 NaN 또는 무한대가 있습니다.")
        return values

    def fit(self, features, labels):
        labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
        if not np.array_equal(np.unique(labels_array), np.asarray([0, 1])):
            raise ValueError("RealTabR collision expert에는 local label 0과 1이 필요합니다.")
        dense = self._fit_reducer(features)
        self.x_context_ = np.asarray(dense, dtype=np.float32)
        self.y_context_ = labels_array.copy()
        self.classifier_ = self._fit_classifier(self.x_context_, self.y_context_)
        self.classes_ = np.asarray(self.classifier_.classes_, dtype=np.int64)
        if not np.array_equal(self.classes_, np.asarray([0, 1])):
            raise RuntimeError("RealTabR local class 순서가 0,1이 아닙니다.")
        self.n_features_in_ = int(features.shape[1])
        return self

    def _fit_classifier(self, dense: np.ndarray, labels_array: np.ndarray):
        classifier_type = self._import_classifier()
        temporary_folder = Path(tempfile.mkdtemp(prefix="realtabr_pytabkit_"))
        classifier = classifier_type(
            device="cpu",
            random_state=self.seed,
            n_cv=1,
            n_refit=0,
            val_fraction=self.val_fraction,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            eval_batch_size=self.eval_batch_size,
            context_size=min(self.context_size, max(2, len(labels_array) - 1)),
            d_main=self.d_main,
            patience=self.patience,
            tmp_folder=temporary_folder,
            verbosity=self.verbosity,
        )
        try:
            classifier.fit(dense, labels_array)
        finally:
            shutil.rmtree(temporary_folder, ignore_errors=True)
        return classifier

    def _ensure_classifier(self) -> None:
        if not hasattr(self, "classifier_"):
            raise RuntimeError(
                "RealTabR inference state가 없습니다. 구형 checkpoint는 Train에서 다시 "
                "생성해야 하며 predict 시점 재학습은 허용되지 않습니다."
            )

    def predict_proba(self, features) -> np.ndarray:
        if not hasattr(self, "x_context_"):
            raise RuntimeError("RealTabR expert를 먼저 fit 해야 합니다.")
        self._ensure_classifier()
        probabilities = np.asarray(
            self.classifier_.predict_proba(self._transform(features)),
            dtype=np.float64,
        )
        if probabilities.shape != (int(features.shape[0]), 2):
            raise RuntimeError("RealTabR pair 확률 shape가 잘못됐습니다.")
        row_sums = probabilities.sum(axis=1)
        if not np.isfinite(probabilities).all() or np.any(row_sums <= 0):
            raise RuntimeError("RealTabR pair 확률이 유효하지 않습니다.")
        return probabilities / row_sums[:, None]

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        classifier = state.pop("classifier_", None)
        state["_inference_state_version"] = _INFERENCE_STATE_VERSION
        state["_classifier_pickle"] = (
            None
            if classifier is None
            else cloudpickle.dumps(classifier, protocol=pickle.HIGHEST_PROTOCOL)
        )
        return state

    def __setstate__(self, state: dict) -> None:
        state = state.copy()
        version = state.pop("_inference_state_version", None)
        classifier_pickle = state.pop("_classifier_pickle", None)
        self.__dict__.update(state)
        if not hasattr(self, "x_context_"):
            return
        if version != _INFERENCE_STATE_VERSION or classifier_pickle is None:
            raise RuntimeError(
                "RealTabR checkpoint에 직렬화된 inference state가 없습니다. "
                "구형 checkpoint는 Train에서 다시 생성해야 합니다."
            )
        self.classifier_ = pickle.loads(classifier_pickle)


class RealTabRCollisionExpertClassifier:
    """전역 XGBoost top-1 충돌 행에서만 RealTabR 확률로 재분배합니다."""

    def __init__(
        self,
        *,
        class_names: tuple[str, ...],
        collision_pairs: tuple[tuple[str, str], ...] = DEFAULT_COLLISION_PAIRS,
        base_model_config: dict | None = None,
        svd_components: int = 128,
        expert_device: str = "cpu",
        n_epochs: int = 4,
        batch_size: int = 64,
        eval_batch_size: int = 256,
        context_size: int = 32,
        d_main: int = 64,
        patience: int = 2,
        val_fraction: float = 0.15,
        verbosity: int = 0,
        seed: int = 42,
    ) -> None:
        self.class_names = tuple(class_names)
        self.collision_pairs = tuple(tuple(pair) for pair in collision_pairs)
        self.base_model_config = dict(base_model_config or {})
        self.svd_components = int(svd_components)
        self.expert_device = str(expert_device)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.eval_batch_size = int(eval_batch_size)
        self.context_size = int(context_size)
        self.d_main = int(d_main)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self.verbosity = int(verbosity)
        self.seed = int(seed)
        self._validate_params()

    def _validate_params(self) -> None:
        if self.expert_device != "cpu":
            raise ValueError("Windows RealTabR collision expert는 CPU만 지원합니다.")
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(
            self.class_names
        ):
            raise ValueError("class_names에는 중복 없는 전체 클래스 순서가 필요합니다.")
        flattened = [name for pair in self.collision_pairs for name in pair]
        if any(len(pair) != 2 or pair[0] == pair[1] for pair in self.collision_pairs):
            raise ValueError("collision pair는 서로 다른 두 클래스여야 합니다.")
        unknown = sorted(set(flattened) - set(self.class_names))
        if unknown:
            raise ValueError(f"class_names에 없는 충돌 클래스입니다: {unknown}")
        if len(flattened) != len(set(flattened)):
            raise ValueError("한 클래스는 둘 이상의 collision pair에 포함될 수 없습니다.")
        if self.base_model_config.get("device", "cuda") != "cuda":
            raise ValueError("RealTabR expert의 global XGBoost base는 CUDA여야 합니다.")

    def _build_base_model(self):
        from src.models.xgboost_model import create_model

        config = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_jobs": -1,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "device": "cuda",
            **self.base_model_config,
        }
        return create_model(config, self.seed)

    def _build_expert(self) -> SVDRealTabRExpert:
        return SVDRealTabRExpert(
            svd_components=self.svd_components,
            device="cpu",
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            eval_batch_size=self.eval_batch_size,
            context_size=self.context_size,
            d_main=self.d_main,
            patience=self.patience,
            val_fraction=self.val_fraction,
            verbosity=self.verbosity,
            seed=self.seed,
        )

    def fit(self, features, labels):
        self._validate_params()
        labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
        expected = np.arange(len(self.class_names), dtype=np.int64)
        if not np.array_equal(np.unique(labels_array), expected):
            raise ValueError("RealTabR global labels와 class_names가 일치하지 않습니다.")
        self.base_model_ = self._build_base_model()
        self.base_model_.fit(features, labels_array)
        self.classes_ = np.asarray(self.base_model_.classes_, dtype=np.int64)
        if not np.array_equal(self.classes_, expected):
            raise RuntimeError("XGBoost class 순서가 class_names와 다릅니다.")

        class_to_id = {name: index for index, name in enumerate(self.class_names)}
        self.experts_: dict[tuple[int, int], SVDRealTabRExpert] = {}
        for left_name, right_name in self.collision_pairs:
            pair = (class_to_id[left_name], class_to_id[right_name])
            mask = np.isin(labels_array, pair)
            local = np.where(labels_array[mask] == pair[0], 0, 1).astype(np.int64)
            expert = self._build_expert()
            expert.fit(_select_rows(features, mask), local)
            self.experts_[pair] = expert
        self.n_features_in_ = int(features.shape[1])
        self.last_prediction_stats_: dict[str, object] = {}
        return self

    def predict_base_proba(self, features) -> np.ndarray:
        if not hasattr(self, "base_model_"):
            raise RuntimeError("모델을 먼저 fit 해야 합니다.")
        return np.asarray(self.base_model_.predict_proba(features), dtype=np.float64)

    def predict_proba(self, features) -> np.ndarray:
        base = self.predict_base_proba(features)
        probabilities = base.copy()
        top1 = np.argmax(base, axis=1)
        active_counts: dict[str, int] = {}
        invalid_rows = 0
        for pair, expert in self.experts_.items():
            active = np.isin(top1, pair)
            active_counts[f"{pair[0]}:{pair[1]}"] = int(active.sum())
            if not np.any(active):
                continue
            expert_probabilities = expert.predict_proba(_select_rows(features, active))
            valid = np.isfinite(expert_probabilities).all(axis=1)
            invalid_rows += int((~valid).sum())
            active_rows = np.flatnonzero(active)[valid]
            if not len(active_rows):
                continue
            pair_mass = probabilities[np.ix_(active_rows, pair)].sum(axis=1)
            probabilities[np.ix_(active_rows, pair)] = (
                expert_probabilities[valid] * pair_mass[:, None]
            )
        row_sums = probabilities.sum(axis=1)
        if not np.isfinite(probabilities).all() or not np.allclose(
            row_sums, 1.0, rtol=1e-6, atol=1e-8
        ):
            raise RuntimeError("RealTabR 결합 확률이 유효하지 않습니다.")
        self.last_prediction_stats_ = {
            "active_rows_by_pair": active_counts,
            "invalid_expert_probability_rows": invalid_rows,
            "changed_rows": int(
                np.count_nonzero(np.argmax(base, axis=1) != np.argmax(probabilities, axis=1))
            ),
        }
        return probabilities

    def predict(self, features) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]

    def get_params(self, deep: bool = True) -> dict:
        del deep
        return {
            "class_names": self.class_names,
            "collision_pairs": self.collision_pairs,
            "base_model_config": self.base_model_config,
            "svd_components": self.svd_components,
            "expert_device": self.expert_device,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "context_size": self.context_size,
            "d_main": self.d_main,
            "patience": self.patience,
            "val_fraction": self.val_fraction,
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
            elif key == "collision_pairs":
                value = tuple(tuple(pair) for pair in value)
            elif key == "base_model_config":
                value = dict(value)
            setattr(self, key, value)
        self._validate_params()
        return self


def create_model(model_config: dict, seed: int) -> RealTabRCollisionExpertClassifier:
    return RealTabRCollisionExpertClassifier(
        class_names=tuple(model_config.get("class_names", ())),
        collision_pairs=tuple(
            tuple(pair)
            for pair in model_config.get("collision_pairs", DEFAULT_COLLISION_PAIRS)
        ),
        base_model_config=model_config.get("base_model", {}),
        svd_components=model_config.get("svd_components", 128),
        expert_device=model_config.get("expert_device", "cpu"),
        n_epochs=model_config.get("n_epochs", 4),
        batch_size=model_config.get("batch_size", 64),
        eval_batch_size=model_config.get("eval_batch_size", 256),
        context_size=model_config.get("context_size", 32),
        d_main=model_config.get("d_main", 64),
        patience=model_config.get("patience", 2),
        val_fraction=model_config.get("val_fraction", 0.15),
        verbosity=model_config.get("verbosity", 0),
        seed=seed,
    )
