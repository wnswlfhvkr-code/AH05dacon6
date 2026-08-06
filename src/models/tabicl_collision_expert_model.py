"""XGBoost 전역 모델에 TabICLv2 충돌 암종 전문가를 결합합니다."""

from __future__ import annotations

import numpy as np

from src.models.tabicl_model import SVDTabICLClassifier, TABICL_V2_CHECKPOINT


DEFAULT_COLLISION_PAIRS = (("GBMLGG", "LGG"), ("KIPAN", "KIRC"))


class TabICLCollisionExpertClassifier:
    """전역 top-1이 충돌 쌍일 때만 쌍의 확률 질량을 재분배합니다."""

    def __init__(
        self,
        *,
        class_names: tuple[str, ...],
        collision_pairs: tuple[tuple[str, str], ...] = DEFAULT_COLLISION_PAIRS,
        base_model_config: dict | None = None,
        svd_components: int = 128,
        ensemble_size: int = 8,
        batch_size: int = 1,
        checkpoint_version: str = TABICL_V2_CHECKPOINT,
        expert_device: str = "cuda",
        use_amp: str | bool = "auto",
        offload_mode: str = "auto",
        verbose: bool = False,
        seed: int = 42,
    ) -> None:
        self.class_names = tuple(class_names)
        self.collision_pairs = tuple(tuple(pair) for pair in collision_pairs)
        self.base_model_config = dict(base_model_config or {})
        self.svd_components = svd_components
        self.ensemble_size = ensemble_size
        self.batch_size = batch_size
        self.checkpoint_version = checkpoint_version
        self.expert_device = expert_device
        self.use_amp = use_amp
        self.offload_mode = offload_mode
        self.verbose = verbose
        self.seed = seed
        self._validate_params()

    def _validate_params(self) -> None:
        if self.expert_device != "cuda":
            raise ValueError("TabICLv2 충돌 전문가는 expert_device='cuda'만 지원합니다.")
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(
            self.class_names
        ):
            raise ValueError("class_names에는 중복 없는 전체 클래스 순서가 필요합니다.")
        if not self.collision_pairs:
            raise ValueError("collision_pairs를 하나 이상 지정해야 합니다.")
        if any(len(pair) != 2 or pair[0] == pair[1] for pair in self.collision_pairs):
            raise ValueError("각 collision_pairs 항목은 서로 다른 두 클래스여야 합니다.")
        flattened = [name for pair in self.collision_pairs for name in pair]
        unknown = sorted(set(flattened) - set(self.class_names))
        if unknown:
            raise ValueError(f"class_names에 없는 충돌 클래스입니다: {unknown}")
        if len(flattened) != len(set(flattened)):
            raise ValueError("한 클래스는 둘 이상의 collision_pairs에 포함될 수 없습니다.")
        if self.base_model_config.get("device", "cuda") != "cuda":
            raise ValueError("전역 XGBoost 모델은 device='cuda'만 지원합니다.")
        if self.svd_components < 1 or self.ensemble_size < 1 or self.batch_size < 1:
            raise ValueError("SVD/ensemble/batch 크기는 모두 1 이상이어야 합니다.")
        if self.checkpoint_version != TABICL_V2_CHECKPOINT:
            raise ValueError(
                "TabICLv2 checkpoint_version은 "
                f"{TABICL_V2_CHECKPOINT!r}이어야 합니다."
            )

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
            raise ValueError("전역 XGBoost 모델은 device='cuda'만 지원합니다.")
        return create_model(config, self.seed)

    def _build_expert(self, pair: tuple[int, int]) -> SVDTabICLClassifier:
        return SVDTabICLClassifier(
            class_names=tuple(self.class_names[class_id] for class_id in pair),
            svd_components=self.svd_components,
            ensemble_size=self.ensemble_size,
            batch_size=self.batch_size,
            support_many_classes=True,
            checkpoint_version=self.checkpoint_version,
            device="cuda",
            use_amp=self.use_amp,
            offload_mode=self.offload_mode,
            verbose=self.verbose,
            seed=self.seed,
        )

    @staticmethod
    def _select_rows(features, mask: np.ndarray):
        indices = np.flatnonzero(mask)
        if hasattr(features, "iloc"):
            return features.iloc[indices]
        return features[indices]

    def fit(self, features, labels):
        self._validate_params()
        labels_array = np.asarray(labels).reshape(-1)
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
        if not np.array_equal(self.classes_, expected_classes):
            raise RuntimeError("XGBoost 클래스 순서가 class_names 설정과 일치하지 않습니다.")

        class_to_id = {name: index for index, name in enumerate(self.class_names)}
        self.experts_: dict[tuple[int, int], SVDTabICLClassifier] = {}
        for left_name, right_name in self.collision_pairs:
            pair = (class_to_id[left_name], class_to_id[right_name])
            mask = np.isin(labels_array, pair)
            pair_to_local = {class_id: index for index, class_id in enumerate(pair)}
            local_labels = np.fromiter(
                (pair_to_local[int(label)] for label in labels_array[mask]),
                dtype=np.int64,
                count=int(mask.sum()),
            )
            expert = self._build_expert(pair)
            expert.fit(self._select_rows(features, mask), local_labels)
            self.experts_[pair] = expert
        self.n_features_in_ = int(features.shape[1])
        return self

    def predict_proba(self, features) -> np.ndarray:
        if not hasattr(self, "experts_"):
            raise RuntimeError("모델을 먼저 fit 해야 합니다.")
        base_probabilities = np.asarray(
            self.base_model_.predict_proba(features), dtype=np.float64
        )
        if base_probabilities.shape[1] != len(self.class_names):
            raise RuntimeError("XGBoost 확률 열 수가 class_names와 일치하지 않습니다.")
        probabilities = base_probabilities.copy()
        top1 = np.argmax(base_probabilities, axis=1)
        for pair, expert in self.experts_.items():
            active = np.isin(top1, pair)
            if not np.any(active):
                continue
            pair_mass = probabilities[np.ix_(active, pair)].sum(axis=1)
            expert_probabilities = expert.predict_proba(
                self._select_rows(features, active)
            )
            probabilities[np.ix_(active, pair)] = (
                expert_probabilities * pair_mass[:, None]
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
            "batch_size": self.batch_size,
            "checkpoint_version": self.checkpoint_version,
            "expert_device": self.expert_device,
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
        if candidate["expert_device"] != "cuda":
            raise ValueError("TabICLv2 충돌 전문가는 expert_device='cuda'만 지원합니다.")
        if dict(candidate["base_model_config"]).get("device", "cuda") != "cuda":
            raise ValueError("전역 XGBoost 모델은 device='cuda'만 지원합니다.")
        previous = self.get_params()
        try:
            for key, value in candidate.items():
                if key in {"class_names", "collision_pairs"}:
                    value = tuple(tuple(item) if key == "collision_pairs" else item for item in value)
                elif key == "base_model_config":
                    value = dict(value)
                setattr(self, key, value)
            self._validate_params()
        except Exception:
            for key, value in previous.items():
                setattr(self, key, value)
            raise
        return self


def create_model(model_config: dict, seed: int) -> TabICLCollisionExpertClassifier:
    return TabICLCollisionExpertClassifier(
        class_names=tuple(model_config.get("class_names", ())),
        collision_pairs=tuple(
            tuple(pair)
            for pair in model_config.get("collision_pairs", DEFAULT_COLLISION_PAIRS)
        ),
        base_model_config=model_config.get("base_model", {}),
        svd_components=model_config.get("svd_components", 128),
        ensemble_size=model_config.get("ensemble_size", 8),
        batch_size=model_config.get("batch_size", 1),
        checkpoint_version=model_config.get(
            "checkpoint_version", TABICL_V2_CHECKPOINT
        ),
        expert_device=model_config.get("expert_device", "cuda"),
        use_amp=model_config.get("use_amp", "auto"),
        offload_mode=model_config.get("offload_mode", "auto"),
        verbose=model_config.get("verbose", False),
        seed=seed,
    )
