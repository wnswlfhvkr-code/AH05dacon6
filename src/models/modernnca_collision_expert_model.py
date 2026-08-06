"""ModernNCA 전역 모델에 pair-local ModernNCA 충돌 전문가를 결합합니다."""

from __future__ import annotations

import numpy as np

from src.models.modernnca_model import SVDModernNCAClassifier


DEFAULT_COLLISION_PAIRS = (("KIRC", "KIPAN"), ("LGG", "GBMLGG"))


class ModernNCACollisionExpertClassifier:
    """Base top-1이 충돌 쌍일 때만 그 쌍의 확률 질량을 재분배합니다."""

    def __init__(
        self,
        *,
        class_names: tuple[str, ...],
        collision_pairs: tuple[tuple[str, str], ...] = DEFAULT_COLLISION_PAIRS,
        base_model_config: dict | None = None,
        expert_model_config: dict | None = None,
        seed: int = 42,
    ) -> None:
        self.class_names = tuple(class_names)
        self.collision_pairs = tuple(tuple(pair) for pair in collision_pairs)
        self.base_model_config = dict(base_model_config or {})
        self.expert_model_config = dict(expert_model_config or {})
        self.seed = seed
        self._validate_params()

    def _validate_params(self) -> None:
        if len(self.class_names) != 26 or len(set(self.class_names)) != 26:
            raise ValueError("ModernNCA collision expert는 중복 없는 26개 class_names가 필요합니다.")
        if self.collision_pairs != DEFAULT_COLLISION_PAIRS:
            raise ValueError(
                "collision_pairs는 ((KIRC, KIPAN), (LGG, GBMLGG))로 고정됩니다."
            )
        missing = sorted(
            {name for pair in self.collision_pairs for name in pair} - set(self.class_names)
        )
        if missing:
            raise ValueError(f"class_names에 없는 충돌 클래스입니다: {missing}")
        for config_name, config in (
            ("base_model_config", self.base_model_config),
            ("expert_model_config", self.expert_model_config),
        ):
            forbidden = sorted(set(config) & {"class_names", "seed"})
            if forbidden:
                raise ValueError(f"{config_name}에서 설정할 수 없는 키입니다: {forbidden}")
            if config.get("device", "cuda") != "cuda":
                raise ValueError(f"{config_name}는 device='cuda'만 지원합니다.")

    def _build_base_model(self) -> SVDModernNCAClassifier:
        return SVDModernNCAClassifier(
            class_names=self.class_names,
            seed=self.seed,
            **self.base_model_config,
        )

    def _build_expert(self, pair: tuple[int, int]) -> SVDModernNCAClassifier:
        return SVDModernNCAClassifier(
            class_names=tuple(self.class_names[index] for index in pair),
            seed=self.seed,
            **self.expert_model_config,
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
        if int(features.shape[0]) != len(labels_array):
            raise ValueError("features와 labels의 행 수가 다릅니다.")
        if not np.array_equal(np.unique(labels_array), expected_classes):
            raise ValueError(
                "class_names 순서와 인코딩 레이블이 일치하지 않습니다: "
                f"expected={expected_classes.tolist()}, observed={np.unique(labels_array).tolist()}"
            )

        self.base_model_ = self._build_base_model()
        self.base_model_.fit(features, labels_array)
        self.classes_ = expected_classes
        class_to_id = {name: index for index, name in enumerate(self.class_names)}
        self.experts_: dict[tuple[int, int], SVDModernNCAClassifier] = {}
        for left_name, right_name in self.collision_pairs:
            pair = (class_to_id[left_name], class_to_id[right_name])
            mask = np.isin(labels_array, pair)
            local_labels = (labels_array[mask] == pair[1]).astype(np.int64)
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
        expected_shape = (int(features.shape[0]), len(self.class_names))
        if base_probabilities.shape != expected_shape:
            raise RuntimeError("ModernNCA base 확률 행렬 shape이 기대값과 다릅니다.")
        probabilities = base_probabilities.copy()
        top1 = np.argmax(base_probabilities, axis=1)
        for pair, expert in self.experts_.items():
            active = np.isin(top1, pair)
            if not np.any(active):
                continue
            pair_mass = base_probabilities[np.ix_(active, pair)].sum(axis=1)
            expert_probabilities = np.asarray(
                expert.predict_proba(self._select_rows(features, active)),
                dtype=np.float64,
            )
            if expert_probabilities.shape != (int(active.sum()), 2):
                raise RuntimeError("ModernNCA pair expert 확률 행렬 shape이 기대값과 다릅니다.")
            probabilities[np.ix_(active, pair)] = expert_probabilities * pair_mass[:, None]
        return probabilities

    def predict(self, features) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]

    def get_params(self, deep: bool = True) -> dict:
        del deep
        return {
            "class_names": self.class_names,
            "collision_pairs": self.collision_pairs,
            "base_model_config": self.base_model_config,
            "expert_model_config": self.expert_model_config,
            "seed": self.seed,
        }

    def set_params(self, **parameters):
        unknown = sorted(set(parameters) - set(self.get_params()))
        if unknown:
            raise ValueError(f"알 수 없는 파라미터입니다: {unknown}")
        previous = self.get_params()
        try:
            for key, value in parameters.items():
                if key == "class_names":
                    value = tuple(value)
                elif key == "collision_pairs":
                    value = tuple(tuple(pair) for pair in value)
                elif key in {"base_model_config", "expert_model_config"}:
                    value = dict(value)
                setattr(self, key, value)
            self._validate_params()
        except Exception:
            for key, value in previous.items():
                setattr(self, key, value)
            raise
        return self


def create_model(model_config: dict, seed: int) -> ModernNCACollisionExpertClassifier:
    return ModernNCACollisionExpertClassifier(
        class_names=tuple(model_config.get("class_names", ())),
        collision_pairs=tuple(
            tuple(pair)
            for pair in model_config.get("collision_pairs", DEFAULT_COLLISION_PAIRS)
        ),
        base_model_config=model_config.get("base_model", {}),
        expert_model_config=model_config.get("expert_model", {}),
        seed=seed,
    )
