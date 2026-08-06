"""희소 pipeComb 피처를 fold 내부 SVD 후 TabPFN GPU로 학습합니다."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


DEFAULT_COLLISION_PAIRS = (("GBMLGG", "LGG"), ("KIPAN", "KIRC"))


class SVDTabPFNClassifier:
    """TabPFN이 받을 수 있는 dense 표현을 누수 없이 학습하는 어댑터."""

    def __init__(
        self,
        *,
        svd_components: int = 128,
        ensemble_size: int = 8,
        balance_probabilities: bool = True,
        device: str = "cuda",
        ignore_pretraining_limits: bool = False,
        seed: int = 42,
    ) -> None:
        if device != "cuda":
            raise ValueError("TabPFN TEST_007 모델은 device=cuda만 지원합니다.")
        if svd_components < 1:
            raise ValueError("svd_components는 1 이상이어야 합니다.")
        if ensemble_size < 1:
            raise ValueError("ensemble_size는 1 이상이어야 합니다.")
        self.svd_components = svd_components
        self.ensemble_size = ensemble_size
        self.balance_probabilities = balance_probabilities
        self.device = device
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.seed = seed

    @staticmethod
    def _import_dependencies():
        try:
            import torch
            from tabpfn import TabPFNClassifier
        except ImportError as error:
            raise ImportError(
                "TabPFN 모델을 사용하려면 `pip install -r requirements.txt`를 실행하세요."
            ) from error
        return torch, TabPFNClassifier

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
            component_count = min(self.svd_components, max_components)
            self.reducer_ = TruncatedSVD(
                n_components=component_count,
                algorithm="randomized",
                random_state=self.seed,
            )
            return np.asarray(
                self.reducer_.fit_transform(features), dtype=np.float32
            )
        self.reducer_ = None
        return self._dense_float32(features)

    def _transform(self, features) -> np.ndarray:
        if not hasattr(self, "reducer_"):
            raise RuntimeError("모델을 먼저 fit 해야 합니다.")
        if self.reducer_ is None:
            return self._dense_float32(features)
        return np.asarray(self.reducer_.transform(features), dtype=np.float32)

    def fit(self, features, labels):
        torch, classifier_type = self._import_dependencies()
        if not torch.cuda.is_available():
            raise RuntimeError("TabPFN TEST_007 모델 학습에는 CUDA GPU가 필요합니다.")
        if len(np.unique(labels)) > 10:
            raise ValueError(
                "TabPFN 8.2.0 기본 모델은 최대 10개 클래스만 지원합니다. "
                "26개 암종에는 충돌 암종 2진 전문가 구조를 사용하세요."
            )
        dense_features = self._fit_reducer(features)
        self.classifier_ = classifier_type(
            device=self.device,
            n_estimators=self.ensemble_size,
            auto_scale_n_estimators=False,
            balance_probabilities=self.balance_probabilities,
            random_state=self.seed,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            fit_mode="low_memory",
            keep_cache_on_device=False,
            show_progress_bar=False,
        )
        self.classifier_.fit(dense_features, np.asarray(labels))
        self.classes_ = np.asarray(self.classifier_.classes_)
        self.n_features_in_ = int(features.shape[1])
        return self

    def predict_proba(self, features) -> np.ndarray:
        if not hasattr(self, "classifier_"):
            raise RuntimeError("모델을 먼저 fit 해야 합니다.")
        return np.asarray(
            self.classifier_.predict_proba(self._transform(features)),
            dtype=np.float64,
        )

    def predict(self, features) -> np.ndarray:
        if not hasattr(self, "classifier_"):
            raise RuntimeError("모델을 먼저 fit 해야 합니다.")
        return np.asarray(self.classifier_.predict(self._transform(features))).reshape(-1)

    def get_params(self, deep: bool = True) -> dict:
        del deep
        return {
            "svd_components": self.svd_components,
            "ensemble_size": self.ensemble_size,
            "balance_probabilities": self.balance_probabilities,
            "device": self.device,
            "ignore_pretraining_limits": self.ignore_pretraining_limits,
            "seed": self.seed,
        }

    def set_params(self, **parameters):
        if parameters.get("device", self.device) != "cuda":
            raise ValueError("TabPFN TEST_007 모델은 device=cuda만 지원합니다.")
        for key, value in parameters.items():
            if not hasattr(self, key):
                raise ValueError(f"알 수 없는 파라미터입니다: {key}")
            setattr(self, key, value)
        return self


class TabPFNCollisionExpertClassifier:
    """XGBoost 전역 확률에서 지정된 충돌 쌍만 TabPFN으로 재분배합니다."""

    def __init__(
        self,
        *,
        class_names: tuple[str, ...],
        collision_pairs: tuple[tuple[str, str], ...] = DEFAULT_COLLISION_PAIRS,
        base_model_config: dict | None = None,
        svd_components: int = 128,
        ensemble_size: int = 8,
        balance_probabilities: bool = True,
        expert_device: str = "cuda",
        ignore_pretraining_limits: bool = False,
        seed: int = 42,
    ) -> None:
        if expert_device != "cuda":
            raise ValueError("TabPFN 충돌 전문가는 expert_device=cuda만 지원합니다.")
        if len(class_names) < 2 or len(set(class_names)) != len(class_names):
            raise ValueError("class_names에는 중복 없는 전체 클래스 순서가 필요합니다.")
        if not collision_pairs:
            raise ValueError("collision_pairs를 하나 이상 지정해야 합니다.")
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
        self.balance_probabilities = balance_probabilities
        self.expert_device = expert_device
        self.ignore_pretraining_limits = ignore_pretraining_limits
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
            raise ValueError("TabPFN 충돌 전문가의 전역 XGBoost도 device=cuda여야 합니다.")
        return create_model(config, self.seed)

    @staticmethod
    def _select_rows(features, mask: np.ndarray):
        if hasattr(features, "iloc"):
            return features.iloc[np.flatnonzero(mask)]
        return features[mask]

    def _build_expert(self) -> SVDTabPFNClassifier:
        return SVDTabPFNClassifier(
            svd_components=self.svd_components,
            ensemble_size=self.ensemble_size,
            balance_probabilities=self.balance_probabilities,
            device=self.expert_device,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
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
        self.experts_: dict[tuple[int, int], SVDTabPFNClassifier] = {}
        for left_name, right_name in self.collision_pairs:
            pair = (class_to_id[left_name], class_to_id[right_name])
            mask = np.isin(labels_array, pair)
            expert = self._build_expert()
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
            expert_columns = {
                int(class_id): column
                for column, class_id in enumerate(expert.classes_)
            }
            pair_mass = probabilities[np.ix_(active, pair)].sum(axis=1)
            for class_id in pair:
                probabilities[active, class_id] = (
                    pair_mass * expert_probabilities[:, expert_columns[class_id]]
                )
        return probabilities

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
            "balance_probabilities": self.balance_probabilities,
            "expert_device": self.expert_device,
            "ignore_pretraining_limits": self.ignore_pretraining_limits,
            "seed": self.seed,
        }

    def set_params(self, **parameters):
        if parameters.get("expert_device", self.expert_device) != "cuda":
            raise ValueError("TabPFN 충돌 전문가는 expert_device=cuda만 지원합니다.")
        for key, value in parameters.items():
            if not hasattr(self, key):
                raise ValueError(f"알 수 없는 파라미터입니다: {key}")
            setattr(self, key, value)
        return self


def create_model(model_config: dict, seed: int) -> TabPFNCollisionExpertClassifier:
    class_names = tuple(model_config.get("class_names", ()))
    collision_pairs = tuple(
        tuple(pair)
        for pair in model_config.get("collision_pairs", DEFAULT_COLLISION_PAIRS)
    )
    return TabPFNCollisionExpertClassifier(
        class_names=class_names,
        collision_pairs=collision_pairs,
        base_model_config=model_config.get("base_model", {}),
        svd_components=model_config.get("svd_components", 128),
        ensemble_size=model_config.get("ensemble_size", 8),
        balance_probabilities=model_config.get("balance_probabilities", True),
        expert_device=model_config.get("expert_device", "cuda"),
        ignore_pretraining_limits=model_config.get(
            "ignore_pretraining_limits", False
        ),
        seed=seed,
    )
