"""고차원 변이 행렬을 축소해 TabPFN-3에 전달하는 분류 모델입니다."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif


class TabPFN3Classifier:
    """fold-train에서 피처를 선택한 뒤 사전학습 TabPFN-3로 분류합니다."""

    def __init__(self, model_config: dict, seed: int) -> None:
        self.seed = seed
        self.model_version = str(model_config.get("model_version", "v3"))
        if self.model_version != "v3":
            raise ValueError("tabpfn3 모델은 model_version='v3'만 지원합니다.")

        self.reduction_method = str(model_config.get("reduction_method", "chi2"))
        if self.reduction_method not in {"chi2", "mutual_info"}:
            raise ValueError("reduction_method는 chi2 또는 mutual_info여야 합니다.")
        self.max_features = int(model_config.get("max_features", 1000))
        if self.max_features < 1:
            raise ValueError("max_features는 1 이상이어야 합니다.")

        self.device = model_config.get("device", "auto")
        self.n_estimators = int(model_config.get("n_estimators", 8))
        self.softmax_temperature = float(
            model_config.get("softmax_temperature", 0.9)
        )
        self.ignore_pretraining_limits = bool(
            model_config.get("ignore_pretraining_limits", False)
        )
        self.model_path = model_config.get("model_path") or "auto"
        self.fit_mode = str(model_config.get("fit_mode", "low_memory"))
        self.memory_saving_mode = model_config.get("memory_saving_mode", "auto")
        self.n_preprocessing_jobs = int(
            model_config.get("n_preprocessing_jobs", 1)
        )
        self.show_progress_bar = bool(model_config.get("show_progress_bar", True))
        categorical = model_config.get("categorical_features", [])
        self.categorical_features_indices = (
            None if not categorical else [int(index) for index in categorical]
        )

        self.selector_: SelectKBest | None = None
        self.model_ = None
        self.classes_: np.ndarray | None = None
        self.feature_columns_: list[str] | None = None
        self.selected_feature_names_: list[str] = []
        self.input_feature_count_: int = 0
        self.selected_feature_count_: int = 0

    def _create_tabpfn(self):
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as error:
            raise ImportError(
                "TabPFN-3 모델에는 tabpfn 패키지가 필요합니다. "
                "`pip install -r requirements.txt`를 실행하세요."
            ) from error

        return TabPFNClassifier(
            n_estimators=self.n_estimators,
            categorical_features_indices=self.categorical_features_indices,
            softmax_temperature=self.softmax_temperature,
            model_path=self.model_path,
            device=self.device,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            fit_mode=self.fit_mode,
            memory_saving_mode=self.memory_saving_mode,
            random_state=self.seed,
            n_preprocessing_jobs=self.n_preprocessing_jobs,
            show_progress_bar=self.show_progress_bar,
        )

    def _prepare_input(self, features, *, fitting: bool) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            if fitting:
                self.feature_columns_ = features.columns.astype(str).tolist()
            elif self.feature_columns_ is not None:
                missing = set(self.feature_columns_) - set(features.columns)
                if missing:
                    raise ValueError(
                        f"TabPFN 입력에 누락된 피처가 있습니다: {sorted(missing)}"
                    )
                features = features[self.feature_columns_]
            values = features.to_numpy(dtype="float32", copy=False)
        else:
            values = np.asarray(features, dtype="float32")
        if values.ndim != 2:
            raise ValueError("TabPFN 입력은 2차원 행렬이어야 합니다.")
        if not np.isfinite(values).all():
            raise ValueError("TabPFN 입력에 NaN 또는 무한대가 있습니다.")
        if not fitting and values.shape[1] != self.input_feature_count_:
            raise ValueError(
                "TabPFN 학습과 예측 입력의 피처 수가 다릅니다: "
                f"{self.input_feature_count_} != {values.shape[1]}"
            )
        return values

    def _create_selector(self, feature_count: int) -> SelectKBest | None:
        selected_count = min(self.max_features, feature_count)
        if selected_count >= feature_count:
            return None
        if self.reduction_method == "chi2":
            score_function = chi2
        else:
            score_function = lambda features, labels: mutual_info_classif(
                features, labels, random_state=self.seed
            )
        return SelectKBest(score_func=score_function, k=selected_count)

    def _fit_reduce(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if self.reduction_method == "chi2" and np.any(features < 0):
            raise ValueError(
                "chi2 피처 선택에는 음수가 없는 입력이 필요합니다. "
                "preprocessing.name=em_v4를 사용하세요."
            )
        self.selector_ = self._create_selector(features.shape[1])
        if self.selector_ is None:
            return features
        return np.asarray(
            self.selector_.fit_transform(features, labels), dtype="float32"
        )

    def _reduce(self, features: np.ndarray) -> np.ndarray:
        if self.selector_ is None:
            return features
        return np.asarray(self.selector_.transform(features), dtype="float32")

    @staticmethod
    def _raise_actionable_fit_error(error: Exception) -> None:
        if error.__class__.__name__ == "TabPFNHuggingFaceGatedRepoError":
            raise RuntimeError(
                "TabPFN-3 가중치는 gated 저장소입니다. "
                "https://huggingface.co/Prior-Labs/tabpfn_3 에서 약관에 동의한 뒤 "
                "`hf auth login`을 실행하거나 HF_TOKEN을 설정하세요."
            ) from error
        raise error

    def fit(self, features, labels):
        """학습 데이터로만 피처 선택기를 fit하고 TabPFN-3를 준비합니다."""
        values = self._prepare_input(features, fitting=True)
        encoded_labels = np.asarray(labels)
        self.input_feature_count_ = values.shape[1]
        reduced = self._fit_reduce(values, encoded_labels)
        self.selected_feature_count_ = reduced.shape[1]

        if self.feature_columns_ is not None:
            if self.selector_ is None:
                self.selected_feature_names_ = list(self.feature_columns_)
            else:
                support = self.selector_.get_support(indices=True)
                self.selected_feature_names_ = [
                    self.feature_columns_[index] for index in support
                ]

        try:
            self.model_ = self._create_tabpfn()
            self.model_.fit(reduced, encoded_labels)
        except Exception as error:
            self._raise_actionable_fit_error(error)
        self.classes_ = np.asarray(self.model_.classes_)
        print(
            "[tabpfn3] train-only 피처 선택: "
            f"{self.input_feature_count_}개 -> {self.selected_feature_count_}개"
        )
        return self

    def predict_proba(self, features) -> np.ndarray:
        """학습 때 고정된 피처 선택을 적용해 클래스 확률을 반환합니다."""
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        values = self._prepare_input(features, fitting=False)
        return np.asarray(self.model_.predict_proba(self._reduce(values)))

    def predict(self, features) -> np.ndarray:
        """학습 때 고정된 피처 선택을 적용해 클래스를 반환합니다."""
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        values = self._prepare_input(features, fitting=False)
        return np.asarray(self.model_.predict(self._reduce(values)))

    def summary(self) -> dict[str, int]:
        return {
            "input_features": self.input_feature_count_,
            "selected_features": self.selected_feature_count_,
            "removed_features": (
                self.input_feature_count_ - self.selected_feature_count_
            ),
        }


def create_model(model_config: dict, seed: int) -> TabPFN3Classifier:
    """설정으로 TabPFN-3 분류기를 생성합니다."""
    return TabPFN3Classifier(model_config, seed)
