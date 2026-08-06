"""XGBoost 전역 모델에 Google TabFM 충돌 암종 전문가를 결합합니다."""

from __future__ import annotations

import copy

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


DEFAULT_COLLISION_PAIRS = (("GBMLGG", "LGG"), ("KIPAN", "KIRC"))
_TABFM_MODEL_CACHE: dict[str, object] = {}


def _load_tabfm_foundation_model(device: str):
    try:
        import torch
        from tabfm import tabfm_v1_0_0_pytorch
    except ImportError as error:
        raise ImportError(
            "TabFM 모델을 사용하려면 `pip install -r requirements.txt`를 실행하세요."
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("TabFM 충돌 전문가 학습에는 CUDA GPU가 필요합니다.")
    if device not in _TABFM_MODEL_CACHE:
        _TABFM_MODEL_CACHE[device] = tabfm_v1_0_0_pytorch.load(
            model_type="classification",
            device=device,
            dtype=torch.bfloat16,
            use_cache=True,
        )
    return _TABFM_MODEL_CACHE[device]


class SVDTabFMExpert:
    """희소 피처를 fold 내부 SVD로 축소해 공유 TabFM 모델에 연결합니다."""

    def __init__(
        self,
        *,
        foundation_model,
        svd_components: int = 128,
        ensemble_size: int = 4,
        batch_size: int = 1,
        expert_device: str = "cuda",
        seed: int = 42,
    ) -> None:
        self.foundation_model = foundation_model
        self.svd_components = svd_components
        self.ensemble_size = ensemble_size
        self.batch_size = batch_size
        self.expert_device = expert_device
        self.seed = seed

    @staticmethod
    def _dense_float32(features) -> np.ndarray:
        if sparse.issparse(features):
            features = features.toarray()
        elif hasattr(features, "to_numpy"):
            features = features.to_numpy()
        return np.asarray(features, dtype=np.float32)

    def _fit_reducer(self, features) -> np.ndarray:
        row_count, feature_count = features.shape
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
            raise RuntimeError("전문가를 먼저 fit 해야 합니다.")
        if self.reducer_ is None:
            return self._dense_float32(features)
        return np.asarray(self.reducer_.transform(features), dtype=np.float32)

    def fit(self, features, labels):
        try:
            from tabfm import TabFMClassifier
        except ImportError as error:
            raise ImportError(
                "TabFM 모델을 사용하려면 `pip install -r requirements.txt`를 실행하세요."
            ) from error
        if self.foundation_model is None:
            self.foundation_model = _load_tabfm_foundation_model(self.expert_device)
        self.classifier_ = TabFMClassifier(
            model=self.foundation_model,
            n_estimators=self.ensemble_size,
            max_num_features=self.svd_components,
            batch_size=self.batch_size,
            use_amp=True,
            random_state=self.seed,
            verbose=False,
        )
        self.classifier_.fit(self._fit_reducer(features), np.asarray(labels))
        self.classes_ = np.asarray(self.classifier_.classes_)
        return self

    def predict_proba(self, features) -> np.ndarray:
        if not hasattr(self, "classifier_"):
            raise RuntimeError("전문가를 먼저 fit 해야 합니다.")
        if self.classifier_.model is None:
            self.classifier_.model = _load_tabfm_foundation_model(self.expert_device)
        return np.asarray(
            self.classifier_.predict_proba(self._transform(features)),
            dtype=np.float64,
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["foundation_model"] = None
        classifier = state.get("classifier_")
        if classifier is not None:
            classifier = copy.copy(classifier)
            classifier.model = None
            state["classifier_"] = classifier
        return state


class TabFMCollisionExpertClassifier:
    """초기 XGBoost top-1이 충돌 쌍일 때만 TabFM 확률로 재분배합니다."""

    def __init__(
        self,
        *,
        class_names: tuple[str, ...],
        collision_pairs: tuple[tuple[str, str], ...] = DEFAULT_COLLISION_PAIRS,
        base_model_config: dict | None = None,
        svd_components: int = 128,
        ensemble_size: int = 4,
        batch_size: int = 1,
        expert_device: str = "cuda",
        seed: int = 42,
    ) -> None:
        if expert_device != "cuda":
            raise ValueError("TabFM 충돌 전문가는 expert_device=cuda만 지원합니다.")
        if not 1 <= svd_components <= 500:
            raise ValueError("TabFM svd_components는 1~500이어야 합니다.")
        if len(class_names) < 2 or len(set(class_names)) != len(class_names):
            raise ValueError("class_names에는 중복 없는 전체 클래스 순서가 필요합니다.")
        unknown_names = {
            name
            for pair in collision_pairs
            for name in pair
            if name not in class_names
        }
        if unknown_names:
            raise ValueError(f"class_names에 없는 충돌 클래스입니다: {sorted(unknown_names)}")
        self.class_names = tuple(class_names)
        self.collision_pairs = tuple(tuple(pair) for pair in collision_pairs)
        self.base_model_config = dict(base_model_config or {})
        self.svd_components = svd_components
        self.ensemble_size = ensemble_size
        self.batch_size = batch_size
        self.expert_device = expert_device
        self.seed = seed

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
        if config.get("device") != "cuda":
            raise ValueError("TabFM 전문가의 전역 XGBoost도 device=cuda여야 합니다.")
        return create_model(config, self.seed)

    def _load_foundation_model(self):
        return _load_tabfm_foundation_model(self.expert_device)

    @staticmethod
    def _select_rows(features, mask: np.ndarray):
        if hasattr(features, "iloc"):
            return features.iloc[np.flatnonzero(mask)]
        return features[mask]

    def _build_expert(self, foundation_model) -> SVDTabFMExpert:
        return SVDTabFMExpert(
            foundation_model=foundation_model,
            svd_components=self.svd_components,
            ensemble_size=self.ensemble_size,
            batch_size=self.batch_size,
            expert_device=self.expert_device,
            seed=self.seed,
        )

    def fit(self, features, labels):
        labels_array = np.asarray(labels)
        expected_classes = np.arange(len(self.class_names))
        observed_classes = np.unique(labels_array)
        if not np.array_equal(observed_classes, expected_classes):
            raise ValueError(
                "class_names 순서와 인코딩 레이블이 일치하지 않습니다: "
                f"expected={expected_classes.tolist()}, observed={observed_classes.tolist()}"
            )
        self.base_model_ = self._build_base_model()
        self.base_model_.fit(features, labels_array)
        self.classes_ = np.asarray(self.base_model_.classes_)
        class_to_id = {name: index for index, name in enumerate(self.class_names)}
        self.foundation_model_ = self._load_foundation_model()
        self.invalid_expert_probability_rows_ = 0
        self.experts_: dict[tuple[int, int], SVDTabFMExpert] = {}
        for left_name, right_name in self.collision_pairs:
            pair = (class_to_id[left_name], class_to_id[right_name])
            mask = np.isin(labels_array, pair)
            expert = self._build_expert(self.foundation_model_)
            expert.fit(self._select_rows(features, mask), labels_array[mask])
            self.experts_[pair] = expert
        self.n_features_in_ = int(features.shape[1])
        return self

    def predict_proba(self, features) -> np.ndarray:
        if not hasattr(self, "experts_"):
            raise RuntimeError("모델을 먼저 fit 해야 합니다.")
        base_probabilities = np.asarray(
            self.base_model_.predict_proba(features), dtype=np.float64
        )
        probabilities = base_probabilities.copy()
        initial_predictions = self.classes_[np.argmax(base_probabilities, axis=1)]
        for pair, expert in self.experts_.items():
            active = np.isin(initial_predictions, pair)
            if not np.any(active):
                continue
            expert_probabilities = expert.predict_proba(
                self._select_rows(features, active)
            )
            row_sums = expert_probabilities.sum(axis=1)
            valid = (
                np.isfinite(expert_probabilities).all(axis=1)
                & np.isfinite(row_sums)
                & (row_sums > 0)
            )
            self.invalid_expert_probability_rows_ += int((~valid).sum())
            if not np.any(valid):
                continue
            expert_probabilities = expert_probabilities[valid] / row_sums[valid, None]
            expert_columns = {
                int(class_id): column
                for column, class_id in enumerate(expert.classes_)
            }
            active_rows = np.flatnonzero(active)[valid]
            pair_mass = probabilities[np.ix_(active_rows, pair)].sum(axis=1)
            for class_id in pair:
                probabilities[active_rows, class_id] = (
                    pair_mass * expert_probabilities[:, expert_columns[class_id]]
                )
        return probabilities

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["foundation_model_"] = None
        return state

    def predict(self, features) -> np.ndarray:
        encoded = np.argmax(self.predict_proba(features), axis=1)
        return np.asarray(self.classes_[encoded]).reshape(-1)

    def get_params(self, deep: bool = True) -> dict:
        del deep
        return {
            "class_names": self.class_names,
            "collision_pairs": self.collision_pairs,
            "base_model_config": self.base_model_config,
            "svd_components": self.svd_components,
            "ensemble_size": self.ensemble_size,
            "batch_size": self.batch_size,
            "expert_device": self.expert_device,
            "seed": self.seed,
        }

    def set_params(self, **parameters):
        if parameters.get("expert_device", self.expert_device) != "cuda":
            raise ValueError("TabFM 충돌 전문가는 expert_device=cuda만 지원합니다.")
        for key, value in parameters.items():
            if not hasattr(self, key):
                raise ValueError(f"알 수 없는 파라미터입니다: {key}")
            setattr(self, key, value)
        return self


def create_model(model_config: dict, seed: int) -> TabFMCollisionExpertClassifier:
    return TabFMCollisionExpertClassifier(
        class_names=tuple(model_config.get("class_names", ())),
        collision_pairs=tuple(
            tuple(pair)
            for pair in model_config.get("collision_pairs", DEFAULT_COLLISION_PAIRS)
        ),
        base_model_config=model_config.get("base_model", {}),
        svd_components=model_config.get("svd_components", 128),
        ensemble_size=model_config.get("ensemble_size", 4),
        batch_size=model_config.get("batch_size", 1),
        expert_device=model_config.get("expert_device", "cuda"),
        seed=seed,
    )
