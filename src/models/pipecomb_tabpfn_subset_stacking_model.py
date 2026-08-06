"""pipeComb v3 수치·텍스트 모델과 API TabPFN의 OOF 부분집합 스태킹."""

from __future__ import annotations

from itertools import combinations
import os
from pathlib import Path
import warnings

import numpy as np
from scipy import sparse
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.models.pipecomb_oof_stacking_model import PipeCombOOFStackingClassifier


class _ColumnIndexSelector:
    """fold-train에서 고정한 열 위치만 이후 데이터에 동일하게 적용합니다."""

    def __init__(self, indices: np.ndarray, eligible_count: int) -> None:
        self.indices = np.asarray(indices, dtype="int64")
        self.eligible_count = int(eligible_count)

    def transform(self, features):
        return features[:, self.indices]


class PipeCombTabPFNSubsetStackingClassifier(PipeCombOOFStackingClassifier):
    """TabPFN을 유지하면서 기존 base learner의 최적 부분집합을 OOF로 선택합니다."""

    _EXISTING_MODELS = ("text", "xgboost", "lightgbm")

    def __init__(self, model_config: dict, seed: int) -> None:
        base_config = dict(model_config)
        base_config.pop("fourth_model", None)
        base_config["ensemble_method"] = "regularized_stacking"
        super().__init__(base_config, seed)

        self.tabpfn_config = dict(model_config.get("tabpfn", {}))
        self.tabpfn_max_features = int(self.tabpfn_config.get("max_features", 512))
        if not 1 <= self.tabpfn_max_features <= 2000:
            raise ValueError("tabpfn.max_features는 API 한도 안의 1~2000이어야 합니다.")
        configured_feature_counts = self.tabpfn_config.get(
            "feature_count_candidates"
        )
        self.tabpfn_feature_count_candidates = (
            (self.tabpfn_max_features,)
            if configured_feature_counts is None
            else tuple(sorted({int(value) for value in configured_feature_counts}))
        )
        if (
            not self.tabpfn_feature_count_candidates
            or any(
                value < 1 or value > 2000
                for value in self.tabpfn_feature_count_candidates
            )
        ):
            raise ValueError(
                "tabpfn.feature_count_candidates는 1~2000 범위의 목록이어야 합니다."
            )
        self.tabpfn_feature_selection_folds = int(
            self.tabpfn_config.get("feature_count_selection_folds", 3)
        )
        self.tabpfn_feature_count_score_tolerance = float(
            self.tabpfn_config.get("feature_count_score_tolerance", 0.0)
        )
        self.tabpfn_selector_mode = str(
            self.tabpfn_config.get("selector_mode", "anova")
        ).lower()
        if self.tabpfn_selector_mode not in {"anova", "stable_anova"}:
            raise ValueError("tabpfn.selector_mode는 anova 또는 stable_anova여야 합니다.")
        self.tabpfn_min_support = int(
            self.tabpfn_config.get("minimum_nonzero_support", 1)
        )
        self.tabpfn_max_prevalence = float(
            self.tabpfn_config.get("maximum_nonzero_prevalence", 1.0)
        )
        self.tabpfn_min_variance = float(
            self.tabpfn_config.get("minimum_variance", 0.0)
        )
        self.tabpfn_stability_folds = int(
            self.tabpfn_config.get("stability_folds", 3)
        )
        self.tabpfn_stability_top_multiplier = float(
            self.tabpfn_config.get("stability_top_multiplier", 2.0)
        )
        self.tabpfn_min_selection_frequency = int(
            self.tabpfn_config.get("minimum_selection_frequency", 2)
        )
        if self.tabpfn_feature_selection_folds < 2:
            raise ValueError("feature_count_selection_folds는 2 이상이어야 합니다.")
        if self.tabpfn_feature_count_score_tolerance < 0:
            raise ValueError("feature_count_score_tolerance는 음수가 아니어야 합니다.")
        if self.tabpfn_min_support < 1:
            raise ValueError("minimum_nonzero_support는 1 이상이어야 합니다.")
        if not 0 < self.tabpfn_max_prevalence <= 1:
            raise ValueError("maximum_nonzero_prevalence는 0보다 크고 1 이하여야 합니다.")
        if self.tabpfn_min_variance < 0:
            raise ValueError("minimum_variance는 음수가 아니어야 합니다.")
        if self.tabpfn_stability_folds < 2:
            raise ValueError("stability_folds는 2 이상이어야 합니다.")
        if self.tabpfn_stability_top_multiplier < 1:
            raise ValueError("stability_top_multiplier는 1 이상이어야 합니다.")
        if self.tabpfn_min_selection_frequency < 1:
            raise ValueError("minimum_selection_frequency는 1 이상이어야 합니다.")
        self.tabpfn_api_key_name = str(
            self.tabpfn_config.get("api_key_name", "tabpfn_api_key")
        )
        self.tabpfn_env_file = Path(self.tabpfn_config.get("env_file", ".env"))
        self.tabpfn_temperature = float(
            self.tabpfn_config.get("temperature", 1.10)
        )
        if self.tabpfn_temperature <= 0:
            raise ValueError("tabpfn.temperature는 0보다 커야 합니다.")

        self.subset_selection_folds = int(
            model_config.get("subset_selection_folds", 5)
        )
        self.subset_score_tolerance = float(
            model_config.get("subset_score_tolerance", 0.001)
        )
        self.minimum_selected_models = int(
            model_config.get("minimum_selected_models", 1)
        )
        self.require_tabpfn = bool(model_config.get("require_tabpfn", True))
        self.reuse_fit_oof_probabilities = bool(
            model_config.get("reuse_fit_oof_probabilities", True)
        )
        if self.subset_selection_folds < 2:
            raise ValueError("subset_selection_folds는 2 이상이어야 합니다.")
        if self.subset_score_tolerance < 0:
            raise ValueError("subset_score_tolerance는 음수가 아니어야 합니다.")
        if not 1 <= self.minimum_selected_models <= 4:
            raise ValueError("minimum_selected_models는 1~4여야 합니다.")

        self.tabpfn_model = None
        self.tabpfn_selector_: SelectKBest | _ColumnIndexSelector | None = None
        self.selected_base_models_: tuple[str, ...] = ()
        self.subset_oof_scores_: dict[str, float] = {}
        self.tabpfn_selected_features_ = 0
        self.tabpfn_selected_feature_counts_by_fold_: list[int] = []
        self.tabpfn_feature_count_oof_scores_by_fold_: list[dict[int, float]] = []
        self.tabpfn_mean_feature_count_oof_scores_: dict[int, float] = {}
        self.tabpfn_final_eligible_features_ = 0
        self._fit_bundle_id_: int | None = None
        self._fit_oof_probabilities_: np.ndarray | None = None
        self.fit_oof_cache_hits_ = 0

    def _load_api_key(self) -> str:
        """환경변수를 우선하고, 없으면 프로젝트 .env의 지정 키만 읽습니다."""
        value = os.getenv(self.tabpfn_api_key_name)
        if value:
            return value.strip()
        if not self.tabpfn_env_file.is_file():
            raise FileNotFoundError(
                f"TabPFN API 키 파일이 없습니다: {self.tabpfn_env_file}"
            )
        for raw_line in self.tabpfn_env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, candidate = line.split("=", 1)
            if name.removeprefix("export ").strip() == self.tabpfn_api_key_name:
                value = candidate.strip().strip("\"'")
                if value:
                    return value
        raise RuntimeError(
            f".env에 {self.tabpfn_api_key_name}=... 값을 설정해야 합니다."
        )

    def _create_tabpfn(self, seed_offset: int = 0):
        try:
            import tabpfn_client
            from tabpfn_client import TabPFNClassifier
        except ImportError as error:
            raise ImportError(
                "API TabPFN 실험에는 tabpfn-client가 필요합니다. "
                "`pip install -r requirements.txt`를 실행하세요."
            ) from error

        tabpfn_client.set_access_token(self._load_api_key())
        parameters = {
            "model_path": self.tabpfn_config.get("model_path", "auto"),
            "n_estimators": int(self.tabpfn_config.get("n_estimators", 4)),
            "softmax_temperature": float(
                self.tabpfn_config.get("softmax_temperature", 1.0)
            ),
            "balance_probabilities": bool(
                self.tabpfn_config.get("balance_probabilities", False)
            ),
            "ignore_pretraining_limits": bool(
                self.tabpfn_config.get("ignore_pretraining_limits", False)
            ),
            "random_state": self.seed + seed_offset,
            "thinking_mode": bool(self.tabpfn_config.get("thinking_mode", False)),
        }
        return TabPFNClassifier(**parameters)

    def _fit_tabpfn_view(
        self,
        tree,
        labels: np.ndarray,
        feature_count: int | None = None,
    ):
        requested = self.tabpfn_max_features if feature_count is None else feature_count
        count = min(int(requested), tree.shape[1])
        if self.tabpfn_selector_mode == "stable_anova":
            selector = self._fit_stable_anova_selector(tree, labels, count)
            selected = selector.transform(tree)
        else:
            selector = SelectKBest(score_func=f_classif, k=count)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                selected = selector.fit_transform(tree, labels)
        if sparse.issparse(selected):
            selected = selected.toarray()
        selected = np.nan_to_num(np.asarray(selected, dtype="float32"), copy=False)
        return selector, selected

    def _fit_stable_anova_selector(
        self,
        tree,
        labels: np.ndarray,
        feature_count: int,
    ) -> _ColumnIndexSelector:
        """support·prevalence·variance와 fold 선택 안정성으로 노이즈 열을 제한합니다."""
        matrix = sparse.csr_matrix(tree, dtype="float32")
        rows = matrix.shape[0]
        support = np.asarray(matrix.getnnz(axis=0)).ravel()
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        squared_mean = np.asarray(matrix.power(2).mean(axis=0)).ravel()
        variance = np.maximum(squared_mean - np.square(mean), 0.0)
        eligible = np.flatnonzero(
            (support >= self.tabpfn_min_support)
            & (support <= self.tabpfn_max_prevalence * rows)
            & (variance >= self.tabpfn_min_variance)
        )
        if not len(eligible):
            raise ValueError("TabPFN 노이즈 필터를 통과한 피처가 없습니다.")
        selected_count = min(int(feature_count), len(eligible))
        eligible_matrix = matrix[:, eligible]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            full_scores, _ = f_classif(eligible_matrix, labels)
        full_scores = np.nan_to_num(
            np.asarray(full_scores, dtype="float64"),
            nan=-np.inf,
            posinf=np.finfo("float64").max,
            neginf=-np.inf,
        )

        _, counts = np.unique(labels, return_counts=True)
        folds = min(self.tabpfn_stability_folds, int(counts.min()))
        frequency = np.zeros(len(eligible), dtype="int16")
        pool_count = min(
            len(eligible),
            max(selected_count, int(np.ceil(
                selected_count * self.tabpfn_stability_top_multiplier
            ))),
        )
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=self.seed + 55_000,
        )
        for train_index, _ in splitter.split(np.zeros(rows), labels):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fold_scores, _ = f_classif(
                    eligible_matrix[train_index], labels[train_index]
                )
            fold_scores = np.nan_to_num(
                np.asarray(fold_scores, dtype="float64"),
                nan=-np.inf,
                posinf=np.finfo("float64").max,
                neginf=-np.inf,
            )
            order = np.lexsort((eligible, -fold_scores))
            frequency[order[:pool_count]] += 1

        stable = frequency >= min(self.tabpfn_min_selection_frequency, folds)
        # 안정 피처가 부족하면 빈도와 전체 fold-train ANOVA 점수 순으로 보충합니다.
        rank = np.lexsort((eligible, -full_scores, -frequency, ~stable))
        selected_indices = eligible[rank[:selected_count]]
        return _ColumnIndexSelector(selected_indices, len(eligible))

    @staticmethod
    def _transform_tabpfn_view(selector: SelectKBest, tree) -> np.ndarray:
        selected = selector.transform(tree)
        if sparse.issparse(selected):
            selected = selected.toarray()
        return np.nan_to_num(np.asarray(selected, dtype="float32"), copy=False)

    def _select_tabpfn_feature_count(
        self,
        tree,
        labels: np.ndarray,
        fold: int,
    ) -> tuple[int, dict[int, float]]:
        """현재 outer-fold 학습 행 안에서만 TabPFN 입력 차원을 OOF 선택합니다."""
        available = tuple(
            value
            for value in self.tabpfn_feature_count_candidates
            if value <= tree.shape[1]
        )
        if not available:
            available = (min(self.tabpfn_feature_count_candidates),)
        if len(available) == 1:
            return available[0], {available[0]: float("nan")}

        _, counts = np.unique(labels, return_counts=True)
        folds = min(self.tabpfn_feature_selection_folds, int(counts.min()))
        if folds < 2:
            raise ValueError("TabPFN 피처 수 OOF 선택에는 클래스별 표본이 2개 이상 필요합니다.")
        candidate_oof = {
            value: np.zeros((len(labels), len(self.classes_)), dtype="float64")
            for value in available
        }
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=self.seed + 50_000 + fold,
        )
        for inner_fold, (train_index, valid_index) in enumerate(
            splitter.split(np.zeros(len(labels)), labels)
        ):
            for feature_count in available:
                selector, selected_train = self._fit_tabpfn_view(
                    tree[train_index], labels[train_index], feature_count
                )
                model = self._create_tabpfn(
                    60_000 + fold * 1_000 + inner_fold * 10 + feature_count
                )
                model.fit(selected_train, labels[train_index])
                selected_valid = self._transform_tabpfn_view(
                    selector, tree[valid_index]
                )
                candidate_oof[feature_count][valid_index] = (
                    self._tabpfn_probabilities(model, selected_valid)
                )

        scores = {
            feature_count: float(
                f1_score(
                    labels,
                    self.classes_[probabilities.argmax(axis=1)],
                    average="macro",
                )
            )
            for feature_count, probabilities in candidate_oof.items()
        }
        best_score = max(scores.values())
        stable = [
            feature_count
            for feature_count, score in scores.items()
            if score >= best_score - self.tabpfn_feature_count_score_tolerance
        ]
        return min(stable), scores

    def _final_tabpfn_feature_count(self) -> int:
        """outer-fold 내부 OOF 점수 평균으로 최종 전체-train 입력 차원을 고정합니다."""
        valid_scores: dict[int, list[float]] = {
            value: [] for value in self.tabpfn_feature_count_candidates
        }
        for fold_scores in self.tabpfn_feature_count_oof_scores_by_fold_:
            for feature_count, score in fold_scores.items():
                if np.isfinite(score):
                    valid_scores.setdefault(feature_count, []).append(score)
        self.tabpfn_mean_feature_count_oof_scores_ = {
            feature_count: float(np.mean(scores))
            for feature_count, scores in valid_scores.items()
            if scores
        }
        if not self.tabpfn_mean_feature_count_oof_scores_:
            return int(np.median(self.tabpfn_selected_feature_counts_by_fold_))
        best_score = max(self.tabpfn_mean_feature_count_oof_scores_.values())
        stable = [
            feature_count
            for feature_count, score in self.tabpfn_mean_feature_count_oof_scores_.items()
            if score >= best_score - self.tabpfn_feature_count_score_tolerance
        ]
        return min(stable)

    def _tabpfn_probabilities(self, model, features: np.ndarray) -> np.ndarray:
        try:
            probabilities = model.predict_proba(features)
        except RuntimeError as error:
            message = str(error)
            if "429" in message and "Daily usage limit" in message:
                reset = ""
                marker = "Resets at "
                if marker in message:
                    reset = message.split(marker, 1)[1].split(".", 1)[0]
                reset_note = f" 재설정 시각: {reset}." if reset else ""
                raise RuntimeError(
                    "TabPFN API 일일 cell prediction 한도에 도달했습니다."
                    f"{reset_note} 제한 재설정 후 다시 실행하거나, "
                    "TabPFN 피처 수·n_estimators·중첩 OOF 횟수를 줄이세요."
                ) from error
            raise
        return self._temperature_scale(
            self._align_probabilities(
                probabilities, model.classes_, self.classes_
            ),
            self.tabpfn_temperature,
        )

    def _candidate_subsets(self) -> list[tuple[str, ...]]:
        """설정에 따라 TabPFN 필수 또는 전체 base learner 부분집합을 비교합니다."""
        candidates: list[tuple[str, ...]] = []
        if self.require_tabpfn:
            for size in range(len(self._EXISTING_MODELS) + 1):
                for existing in combinations(self._EXISTING_MODELS, size):
                    subset = (*existing, "tabpfn")
                    if len(subset) >= self.minimum_selected_models:
                        candidates.append(subset)
            return candidates
        all_models = (*self._EXISTING_MODELS, "tabpfn")
        for size in range(self.minimum_selected_models, len(all_models) + 1):
            candidates.extend(combinations(all_models, size))
        return candidates

    def _cross_fitted_subset_score(
        self,
        oof: dict[str, np.ndarray],
        labels: np.ndarray,
        subset: tuple[str, ...],
    ) -> float:
        counts = np.unique(labels, return_counts=True)[1]
        folds = min(self.subset_selection_folds, int(counts.min()))
        predictions = np.empty(len(labels), dtype=labels.dtype)
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=self.seed + 30_000
        )
        meta = self._meta_features(*(oof[name] for name in subset))
        for train_index, valid_index in splitter.split(meta, labels):
            feature_indices = self._fit_meta_feature_indices(
                meta[train_index], labels[train_index]
            )
            train_meta = self._select_meta_features(
                meta[train_index], feature_indices
            )
            valid_meta = self._select_meta_features(
                meta[valid_index], feature_indices
            )
            scaler = StandardScaler()
            model = self._create_meta_model(labels[train_index])
            self._fit_meta_model(
                model,
                scaler.fit_transform(train_meta),
                labels[train_index],
            )
            predictions[valid_index] = model.predict(
                scaler.transform(valid_meta)
            )
        return float(f1_score(labels, predictions, average="macro"))

    def _select_subset(
        self, oof: dict[str, np.ndarray], labels: np.ndarray
    ) -> tuple[str, ...]:
        scored = [
            (self._cross_fitted_subset_score(oof, labels, subset), subset)
            for subset in self._candidate_subsets()
        ]
        self.subset_oof_scores_ = {
            "+".join(subset): score for score, subset in scored
        }
        best_score = max(score for score, _ in scored)
        stable = [
            (score, subset)
            for score, subset in scored
            if score >= best_score - self.subset_score_tolerance
        ]
        return min(stable, key=lambda item: (len(item[1]), -item[0], item[1]))[1]

    def fit(self, features, labels):
        bundle = self._require_bundle(features)
        xgboost_tree, lightgbm_tree, _ = self._tree_views(bundle)
        y = np.asarray(labels)
        self.classes_, counts = np.unique(y, return_counts=True)
        folds = min(self.stacking_folds, int(counts.min()))
        if folds < 2:
            raise ValueError("OOF stacking에는 클래스별 표본이 최소 2개 필요합니다.")

        oof = {
            name: np.zeros((len(y), len(self.classes_)), dtype="float64")
            for name in (*self._EXISTING_MODELS, "tabpfn")
        }
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        xgb_counts: list[int] = []
        lgbm_counts: list[int] = []
        base_sample_weight = self._base_sample_weight(y)
        self.tabpfn_selected_feature_counts_by_fold_ = []
        self.tabpfn_feature_count_oof_scores_by_fold_ = []
        for fold, (train_index, valid_index) in enumerate(
            splitter.split(np.zeros(len(y)), y)
        ):
            offset = 100 * (fold + 1)
            text_model = self._create_text_model(offset, y[train_index])
            fit_index, stop_index = self._early_stopping_indices(train_index, y, fold)
            xgb_model = self._create_xgboost(
                offset + 1,
                enable_early_stopping=self.xgb_early_stopping_rounds > 0,
            )
            lgbm_model = self._create_lightgbm(offset + 2)
            selected_feature_count, feature_count_scores = (
                self._select_tabpfn_feature_count(
                    xgboost_tree[train_index], y[train_index], fold
                )
            )
            self.tabpfn_selected_feature_counts_by_fold_.append(
                selected_feature_count
            )
            self.tabpfn_feature_count_oof_scores_by_fold_.append(
                feature_count_scores
            )
            selector, tab_train = self._fit_tabpfn_view(
                xgboost_tree[train_index],
                y[train_index],
                selected_feature_count,
            )
            tabpfn_model = self._create_tabpfn(offset + 3)

            text_model.fit(bundle.text[train_index], y[train_index])
            if self.xgb_early_stopping_rounds > 0:
                fit_weight = (
                    None if base_sample_weight is None
                    else base_sample_weight[fit_index]
                )
                stop_weight = (
                    None if base_sample_weight is None
                    else base_sample_weight[stop_index]
                )
                xgb_model.fit(
                    xgboost_tree[fit_index], y[fit_index],
                    sample_weight=fit_weight,
                    eval_set=[(xgboost_tree[stop_index], y[stop_index])],
                    sample_weight_eval_set=(
                        [stop_weight] if stop_weight is not None else None
                    ),
                    verbose=False,
                )
            else:
                train_weight = (
                    None if base_sample_weight is None
                    else base_sample_weight[train_index]
                )
                xgb_model.fit(
                    xgboost_tree[train_index], y[train_index],
                    sample_weight=train_weight,
                )
            self._fit_third_model(
                lgbm_model,
                lightgbm_tree,
                y,
                fit_index if self.lgbm_early_stopping_rounds > 0 else train_index,
                stop_index,
                base_sample_weight,
            )
            tabpfn_model.fit(tab_train, y[train_index])

            xgb_counts.append(self._best_iteration_count(xgb_model, 350))
            lgbm_counts.append(self._best_iteration_count(lgbm_model, 280))
            oof["text"][valid_index] = self._align_probabilities(
                self._text_probabilities(text_model, bundle.text[valid_index]),
                text_model.classes_, self.classes_,
            )
            oof["xgboost"][valid_index] = self._temperature_scale(
                self._align_probabilities(
                    xgb_model.predict_proba(xgboost_tree[valid_index]),
                    xgb_model.classes_, self.classes_,
                ), self.xgb_temperature,
            )
            oof["lightgbm"][valid_index] = self._temperature_scale(
                self._align_probabilities(
                    lgbm_model.predict_proba(lightgbm_tree[valid_index]),
                    lgbm_model.classes_, self.classes_,
                ), self.lgbm_temperature,
            )
            oof["tabpfn"][valid_index] = self._tabpfn_probabilities(
                tabpfn_model,
                self._transform_tabpfn_view(selector, xgboost_tree[valid_index]),
            )

        self.selected_base_models_ = self._select_subset(oof, y)
        selected_meta = self._meta_features(
            *(oof[name] for name in self.selected_base_models_)
        )
        self.meta_input_feature_count_ = selected_meta.shape[1]
        self.meta_feature_indices_ = self._fit_meta_feature_indices(
            selected_meta, y
        )
        selected_meta = self._select_meta_features(
            selected_meta, self.meta_feature_indices_
        )
        self.meta_scaler = StandardScaler()
        self.meta_model = self._create_meta_model(y)
        self._fit_meta_model(
            self.meta_model,
            self.meta_scaler.fit_transform(selected_meta),
            y,
        )
        scaled_selected_meta = self.meta_scaler.transform(selected_meta)
        cached_probabilities = self.meta_model.predict_proba(scaled_selected_meta)
        self._fit_oof_probabilities_ = self._align_probabilities(
            cached_probabilities,
            self.meta_model.classes_,
            self.classes_,
        )
        self._fit_bundle_id_ = id(bundle)
        self.meta_feature_count_ = selected_meta.shape[1]
        self.inner_oof_rows_ = len(y)
        self.selected_xgb_estimators_ = int(np.median(xgb_counts))
        self.selected_lgbm_estimators_ = int(np.median(lgbm_counts))

        if "text" in self.selected_base_models_:
            self.text_model = self._create_text_model(labels=y)
            self.text_model.fit(bundle.text, y)
        if "xgboost" in self.selected_base_models_:
            self.xgb_model = self._create_xgboost(
                1, n_estimators=self.selected_xgb_estimators_
            )
            self.xgb_model.fit(
                xgboost_tree, y, sample_weight=base_sample_weight
            )
        if "lightgbm" in self.selected_base_models_:
            self.lgbm_model = self._create_lightgbm(
                2, n_estimators=self.selected_lgbm_estimators_
            )
            self.lgbm_model.fit(
                lightgbm_tree, y, sample_weight=base_sample_weight
            )
        if "tabpfn" in self.selected_base_models_:
            final_feature_count = self._final_tabpfn_feature_count()
            self.tabpfn_selector_, tab_train = self._fit_tabpfn_view(
                xgboost_tree, y, final_feature_count
            )
            self.tabpfn_selected_features_ = tab_train.shape[1]
            self.tabpfn_final_eligible_features_ = int(
                getattr(
                    self.tabpfn_selector_,
                    "eligible_count",
                    xgboost_tree.shape[1],
                )
            )
            self.tabpfn_model = self._create_tabpfn(3)
            self.tabpfn_model.fit(tab_train, y)
        print(
            "[pipecomb_tabpfn] OOF 선택 base models: "
            + ", ".join(self.selected_base_models_)
        )
        return self

    def _selected_probabilities(self, bundle) -> tuple[np.ndarray, ...]:
        xgboost_tree, lightgbm_tree, _ = self._tree_views(bundle)
        output: list[np.ndarray] = []
        for name in self.selected_base_models_:
            if name == "text":
                output.append(self._align_probabilities(
                    self._text_probabilities(self.text_model, bundle.text),
                    self.text_model.classes_, self.classes_,
                ))
            elif name == "xgboost":
                output.append(self._temperature_scale(
                    self._align_probabilities(
                        self.xgb_model.predict_proba(xgboost_tree),
                        self.xgb_model.classes_, self.classes_,
                    ), self.xgb_temperature,
                ))
            elif name == "lightgbm":
                output.append(self._temperature_scale(
                    self._align_probabilities(
                        self.lgbm_model.predict_proba(lightgbm_tree),
                        self.lgbm_model.classes_, self.classes_,
                    ), self.lgbm_temperature,
                ))
            else:
                tab_features = self._transform_tabpfn_view(
                    self.tabpfn_selector_, xgboost_tree
                )
                output.append(self._tabpfn_probabilities(
                    self.tabpfn_model, tab_features
                ))
        return tuple(output)

    def predict_proba(self, features) -> np.ndarray:
        if self.meta_model is None or not self.selected_base_models_:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        bundle = self._require_bundle(features)
        if (
            self.reuse_fit_oof_probabilities
            and self._fit_bundle_id_ == id(bundle)
            and self._fit_oof_probabilities_ is not None
            and len(self._fit_oof_probabilities_) == bundle.tree.shape[0]
        ):
            self.fit_oof_cache_hits_ += 1
            return self._fit_oof_probabilities_.copy()
        meta = self._meta_features(*self._selected_probabilities(bundle))
        meta = self._select_meta_features(meta, self.meta_feature_indices_)
        probabilities = self.meta_model.predict_proba(self.meta_scaler.transform(meta))
        return self._align_probabilities(
            probabilities, self.meta_model.classes_, self.classes_
        )

    def summary(self) -> dict[str, object]:
        result = super().summary()
        result.update({
            "base_learners": len(self.selected_base_models_),
            "selected_base_models": list(self.selected_base_models_),
            "subset_oof_scores": dict(self.subset_oof_scores_),
            "tabpfn_api_client": True,
            "reuse_fit_oof_probabilities": self.reuse_fit_oof_probabilities,
            "fit_oof_cache_hits": self.fit_oof_cache_hits_,
            "tabpfn_required_in_selected_subset": self.require_tabpfn,
            "tabpfn_selector_mode": self.tabpfn_selector_mode,
            "tabpfn_final_eligible_features": self.tabpfn_final_eligible_features_,
            "tabpfn_max_features": self.tabpfn_max_features,
            "tabpfn_feature_count_candidates": list(
                self.tabpfn_feature_count_candidates
            ),
            "tabpfn_feature_count_selection_folds": (
                self.tabpfn_feature_selection_folds
            ),
            "tabpfn_selected_feature_counts_by_fold": list(
                self.tabpfn_selected_feature_counts_by_fold_
            ),
            "tabpfn_feature_count_oof_scores_by_fold": [
                {str(key): value for key, value in scores.items()}
                for scores in self.tabpfn_feature_count_oof_scores_by_fold_
            ],
            "tabpfn_mean_feature_count_oof_scores": {
                str(key): value
                for key, value in self.tabpfn_mean_feature_count_oof_scores_.items()
            },
            "tabpfn_selected_features": self.tabpfn_selected_features_,
            "tabpfn_api_key_name": self.tabpfn_api_key_name,
        })
        return result


def create_model(
    model_config: dict, seed: int
) -> PipeCombTabPFNSubsetStackingClassifier:
    return PipeCombTabPFNSubsetStackingClassifier(model_config, seed)
