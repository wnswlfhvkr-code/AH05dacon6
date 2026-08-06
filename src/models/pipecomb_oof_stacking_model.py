"""JSJ9 text와 EM 계열 tree의 교차 적합 확률을 결합하는 앙상블 모델."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from src.pipelines.em_preprocessing.pipeline_pipeComb_em_v1_006 import OOFStackingFeatureBundle


class PipeCombOOFStackingClassifier:
    """TF-IDF SVC와 EM tree 모델을 OOF stacking 또는 log blend로 결합합니다."""

    def __init__(self, model_config: dict, seed: int) -> None:
        self.seed = int(seed)
        self.stacking_folds = int(model_config.get("stacking_folds", 5))
        if self.stacking_folds < 2:
            raise ValueError("stacking_folds는 2 이상이어야 합니다.")

        self.text_config = dict(model_config.get("text_linear_svc", {}))
        self.ensemble_method = str(
            model_config.get("ensemble_method", "regularized_stacking")
        )
        if self.ensemble_method not in {
            "regularized_stacking",
            "constrained_log_blend",
        }:
            raise ValueError(f"지원하지 않는 앙상블 방식입니다: {self.ensemble_method}")
        self.third_model_name = str(
            model_config.get("third_model", "lightgbm")
        ).lower()
        if self.third_model_name not in {"lightgbm", "catboost"}:
            raise ValueError(f"지원하지 않는 세 번째 모델입니다: {self.third_model_name}")
        fourth_model = model_config.get("fourth_model")
        self.fourth_model_name = (
            None if fourth_model in (None, "", "none") else str(fourth_model).lower()
        )
        if self.fourth_model_name not in {None, "catboost"}:
            raise ValueError(
                f"지원하지 않는 네 번째 모델입니다: {self.fourth_model_name}"
            )
        if self.fourth_model_name == "catboost" and self.third_model_name == "catboost":
            raise ValueError("CatBoost를 세 번째와 네 번째 모델에 중복 지정할 수 없습니다.")
        if self.fourth_model_name is not None and self.ensemble_method != "regularized_stacking":
            raise ValueError("네 번째 모델은 regularized_stacking에서만 지원합니다.")
        self.class_weight_mode = str(
            model_config.get("class_weight_mode", "configured")
        ).lower()
        if self.class_weight_mode not in {
            "configured", "none", "sqrt_balanced", "effective_number"
        }:
            raise ValueError(
                f"지원하지 않는 클래스 가중치 방식입니다: {self.class_weight_mode}"
            )
        class_weight_clip = model_config.get("class_weight_clip", (0.75, 2.5))
        self.class_weight_clip = tuple(float(value) for value in class_weight_clip)
        self.class_weight_power = float(model_config.get("class_weight_power", 0.5))
        self.effective_number_beta = float(
            model_config.get("effective_number_beta", 0.99)
        )
        self.base_sample_weight_enabled = bool(
            model_config.get("base_sample_weight_enabled", False)
        )
        if (
            len(self.class_weight_clip) != 2
            or self.class_weight_clip[0] <= 0
            or self.class_weight_clip[0] > self.class_weight_clip[1]
        ):
            raise ValueError("class_weight_clip은 양수인 [최솟값, 최댓값]이어야 합니다.")
        if not 0.0 <= self.class_weight_power <= 1.0:
            raise ValueError("class_weight_power는 0 이상 1 이하여야 합니다.")
        if not 0.0 < self.effective_number_beta < 1.0:
            raise ValueError("effective_number_beta는 0보다 크고 1보다 작아야 합니다.")
        self.xgb_config = dict(
            model_config.get(
                "emv46_xgboost",
                model_config.get("emv45_xgboost", {}),
            )
        )
        self.lgbm_config = dict(
            model_config.get(
                "emv46_lightgbm",
                model_config.get("emv45_lightgbm", {}),
            )
        )
        self.catboost_config = dict(model_config.get("emv46_catboost", {}))
        self.xgb_early_stopping_rounds = int(
            self.xgb_config.pop("early_stopping_rounds", 0) or 0
        )
        self.lgbm_early_stopping_rounds = int(
            self.lgbm_config.pop("early_stopping_rounds", 0) or 0
        )
        self.catboost_early_stopping_rounds = int(
            self.catboost_config.pop("early_stopping_rounds", 0) or 0
        )
        self.early_stopping_fraction = float(
            model_config.get("early_stopping_fraction", 0.15)
        )
        if not 0.0 < self.early_stopping_fraction < 0.5:
            raise ValueError("early_stopping_fraction은 0보다 크고 0.5보다 작아야 합니다.")
        self.meta_config = dict(model_config.get("meta_learner", {}))
        self.meta_model_type = str(
            self.meta_config.get("type", "logistic_regression")
        ).lower()
        if self.meta_model_type not in {
            "logistic_regression", "xgboost", "focal_mlp", "balanced_batch_mlp"
        }:
            raise ValueError(
                "meta_learner.type은 logistic_regression, xgboost, focal_mlp "
                "또는 balanced_batch_mlp여야 합니다."
            )
        self.meta_feature_selection_config = dict(
            model_config.get("meta_feature_selection", {})
        )
        self.meta_feature_selection_enabled = bool(
            self.meta_feature_selection_config.get("enabled", False)
        )
        self.meta_feature_selection_max_features = int(
            self.meta_feature_selection_config.get("max_features", 48)
        )
        self.meta_feature_selection_min_variance = float(
            self.meta_feature_selection_config.get("minimum_variance", 1e-6)
        )
        self.meta_feature_selection_folds = int(
            self.meta_feature_selection_config.get("stability_folds", 5)
        )
        self.meta_feature_selection_top_multiplier = float(
            self.meta_feature_selection_config.get("stability_top_multiplier", 1.5)
        )
        self.meta_feature_selection_min_frequency = int(
            self.meta_feature_selection_config.get("minimum_selection_frequency", 3)
        )
        if self.meta_feature_selection_max_features < 1:
            raise ValueError("meta_feature_selection.max_features는 1 이상이어야 합니다.")
        if self.meta_feature_selection_min_variance < 0:
            raise ValueError("meta_feature_selection.minimum_variance는 음수가 아니어야 합니다.")
        if self.meta_feature_selection_folds < 2:
            raise ValueError("meta_feature_selection.stability_folds는 2 이상이어야 합니다.")
        if self.meta_feature_selection_top_multiplier < 1:
            raise ValueError("meta_feature_selection.stability_top_multiplier는 1 이상이어야 합니다.")
        if self.meta_feature_selection_min_frequency < 1:
            raise ValueError("minimum_selection_frequency는 1 이상이어야 합니다.")
        self.blend_config = dict(model_config.get("constrained_log_blend", {}))
        self.text_temperature = float(model_config.get("text_temperature", 1.20))
        self.xgb_temperature = float(model_config.get("xgboost_temperature", 1.15))
        default_third_temperature = model_config.get(
            "catboost_temperature",
            model_config.get("lightgbm_temperature", 1.15),
        )
        self.lgbm_temperature = float(
            model_config.get("third_temperature", default_third_temperature)
        )
        self.fourth_temperature = float(
            model_config.get("fourth_temperature", model_config.get("catboost_temperature", 1.20))
        )
        if min(
            self.text_temperature,
            self.xgb_temperature,
            self.lgbm_temperature,
            self.fourth_temperature,
        ) <= 0:
            raise ValueError("확률 temperature는 0보다 커야 합니다.")
        self.postprocess_config = dict(model_config.get("postprocessing", {}))
        self.postprocess_enabled = bool(self.postprocess_config.get("enabled", False))
        self.postprocess_method = str(
            self.postprocess_config.get("method", "oof_class_prior_power")
        )
        supported_postprocessing = {
            "oof_class_prior_power",
            "oof_crossfit_classwise_threshold",
        }
        if (
            self.postprocess_enabled
            and self.postprocess_method not in supported_postprocessing
        ):
            raise ValueError("지원하지 않는 후처리 방식입니다.")
        if self.postprocess_enabled and self.ensemble_method != "regularized_stacking":
            raise ValueError("OOF prior 후처리는 regularized_stacking에서만 지원합니다.")
        self.postprocess_folds = int(self.postprocess_config.get("folds", 5))
        if self.postprocess_folds < 2:
            raise ValueError("postprocessing.folds는 2 이상이어야 합니다.")
        self.postprocess_gamma_candidates = sorted({
            float(value)
            for value in self.postprocess_config.get(
                "gamma_candidates", (-0.10, -0.05, 0.0, 0.05, 0.10)
            )
        })
        self.postprocess_temperature_candidates = sorted({
            float(value)
            for value in self.postprocess_config.get(
                "temperature_candidates", (0.95, 1.0, 1.05)
            )
        })
        if (
            not self.postprocess_gamma_candidates
            or not self.postprocess_temperature_candidates
            or any(value <= 0 for value in self.postprocess_temperature_candidates)
        ):
            raise ValueError("후처리 후보 목록이 올바르지 않습니다.")
        self.postprocess_min_oof_gain = float(
            self.postprocess_config.get("minimum_oof_macro_f1_gain", 0.002)
        )
        if self.postprocess_min_oof_gain < 0:
            raise ValueError("minimum_oof_macro_f1_gain은 음수가 아니어야 합니다.")
        self.postprocess_threshold_candidates = sorted({
            float(value)
            for value in self.postprocess_config.get(
                "threshold_factor_candidates",
                (0.75, 0.85, 0.925, 1.0, 1.075, 1.15, 1.25),
            )
        })
        if (
            not self.postprocess_threshold_candidates
            or 1.0 not in self.postprocess_threshold_candidates
            or any(value <= 0 for value in self.postprocess_threshold_candidates)
        ):
            raise ValueError(
                "threshold_factor_candidates는 1.0을 포함하는 양수 목록이어야 합니다."
            )
        self.postprocess_threshold_rounds = int(
            self.postprocess_config.get("coordinate_descent_rounds", 2)
        )
        self.postprocess_threshold_shrinkage = float(
            self.postprocess_config.get("threshold_log_l2", 0.002)
        )
        self.postprocess_min_class_support = int(
            self.postprocess_config.get("minimum_class_support", 25)
        )
        if self.postprocess_threshold_rounds < 1:
            raise ValueError("coordinate_descent_rounds는 1 이상이어야 합니다.")
        if self.postprocess_threshold_shrinkage < 0:
            raise ValueError("threshold_log_l2는 음수가 아니어야 합니다.")
        if self.postprocess_min_class_support < 1:
            raise ValueError("minimum_class_support는 1 이상이어야 합니다.")

        self.classes_: np.ndarray | None = None
        self.text_model = None
        self.xgb_model = None
        self.lgbm_model = None
        self.fourth_model = None
        self.meta_scaler = StandardScaler()
        self.meta_model = None
        self.inner_oof_rows_ = 0
        self.meta_feature_count_ = 0
        self.meta_input_feature_count_ = 0
        self.meta_feature_indices_ = np.zeros(0, dtype="int64")
        self.meta_feature_eligible_count_ = 0
        self.selected_xgb_estimators_ = 0
        self.selected_lgbm_estimators_ = 0
        self.selected_fourth_estimators_ = 0
        self.calibrated_temperatures_ = np.ones(3, dtype="float64")
        self.blend_weights_ = np.full(3, 1.0 / 3.0, dtype="float64")
        self.postprocess_class_prior_: np.ndarray | None = None
        self.selected_postprocess_gamma_ = 0.0
        self.selected_postprocess_temperature_ = 1.0
        self.postprocess_baseline_oof_macro_f1_: float | None = None
        self.postprocess_selected_oof_macro_f1_: float | None = None
        self.postprocess_candidate_count_ = 0
        self.selected_postprocess_threshold_factors_ = np.ones(0, dtype="float64")

    @staticmethod
    def _require_bundle(features) -> OOFStackingFeatureBundle:
        if not all(hasattr(features, name) for name in ("text", "tree")):
            raise TypeError(
                "pipecomb_oof_stacking에는 "
                "text와 tree 뷰를 제공하는 pipeComb stacking 전처리가 필요합니다."
            )
        return features

    @staticmethod
    def _tree_views(bundle):
        default = bundle.tree
        xgboost = getattr(bundle, "xgboost_tree", None)
        third = getattr(bundle, "third_tree", None)
        fourth = getattr(bundle, "fourth_tree", None)
        xgboost = default if xgboost is None else xgboost
        third = default if third is None else third
        fourth = third if fourth is None else fourth
        return xgboost, third, fourth

    def _class_weights(self, labels: np.ndarray | None):
        if self.class_weight_mode == "configured":
            return None
        if self.class_weight_mode == "none":
            return None
        if labels is None:
            raise ValueError("sqrt_balanced 클래스 가중치에는 학습 labels가 필요합니다.")
        classes, counts = np.unique(labels, return_counts=True)
        if self.class_weight_mode == "effective_number":
            raw = (1.0 - self.effective_number_beta) / (
                1.0 - np.power(self.effective_number_beta, counts.astype("float64"))
            )
            raw /= np.average(raw, weights=counts)
        else:
            raw = (
                len(labels) / (len(classes) * counts.astype("float64"))
            ) ** self.class_weight_power
        clipped = np.clip(raw, *self.class_weight_clip)
        return {
            value.item() if hasattr(value, "item") else value: float(weight)
            for value, weight in zip(classes, clipped)
        }

    def _base_sample_weight(self, labels: np.ndarray) -> np.ndarray | None:
        if not self.base_sample_weight_enabled:
            return None
        weights = self._class_weights(labels)
        if weights is None:
            return None
        return np.asarray([weights[value] for value in labels], dtype="float64")

    def _create_text_model(
        self,
        seed_offset: int = 0,
        labels: np.ndarray | None = None,
    ) -> LinearSVC:
        parameters = dict(self.text_config)
        parameters.setdefault("C", 0.10)
        parameters.setdefault("class_weight", "balanced")
        if self.class_weight_mode == "none":
            parameters["class_weight"] = None
        elif self.class_weight_mode in {"sqrt_balanced", "effective_number"}:
            parameters["class_weight"] = self._class_weights(labels)
        parameters.setdefault("max_iter", 20000)
        parameters.setdefault("tol", 1e-4)
        parameters.setdefault("dual", "auto")
        parameters["random_state"] = self.seed + seed_offset
        return LinearSVC(**parameters)

    def _create_xgboost(
        self,
        seed_offset: int = 0,
        n_estimators: int | None = None,
        enable_early_stopping: bool = False,
    ):
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError("pipecomb_oof_stacking에는 xgboost가 필요합니다.") from error
        parameters = dict(self.xgb_config)
        parameters.setdefault("n_estimators", 350)
        parameters.setdefault("learning_rate", 0.025)
        parameters.setdefault("max_depth", 3)
        parameters.setdefault("min_child_weight", 8.0)
        parameters.setdefault("subsample", 0.75)
        parameters.setdefault("colsample_bytree", 0.55)
        parameters.setdefault("reg_alpha", 1.5)
        parameters.setdefault("reg_lambda", 15.0)
        parameters.setdefault("max_delta_step", 1.0)
        parameters.setdefault("eval_metric", "mlogloss")
        parameters.setdefault("tree_method", "hist")
        parameters.setdefault("n_jobs", -1)
        if n_estimators is not None:
            parameters["n_estimators"] = int(n_estimators)
        if enable_early_stopping and self.xgb_early_stopping_rounds > 0:
            parameters["early_stopping_rounds"] = self.xgb_early_stopping_rounds
        parameters["random_state"] = self.seed + seed_offset
        return XGBClassifier(**parameters)

    def _create_lightgbm(
        self,
        seed_offset: int = 0,
        n_estimators: int | None = None,
    ):
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise ImportError("pipecomb_oof_stacking에는 lightgbm이 필요합니다.") from error
        parameters = dict(self.lgbm_config)
        parameters.setdefault("objective", "multiclass")
        parameters.setdefault("n_estimators", 280)
        parameters.setdefault("learning_rate", 0.025)
        parameters.setdefault("num_leaves", 11)
        parameters.setdefault("max_depth", 4)
        parameters.setdefault("min_child_samples", 45)
        parameters.setdefault("subsample", 0.75)
        parameters.setdefault("subsample_freq", 1)
        parameters.setdefault("colsample_bytree", 0.50)
        parameters.setdefault("reg_alpha", 2.0)
        parameters.setdefault("reg_lambda", 18.0)
        parameters.setdefault("n_jobs", -1)
        parameters.setdefault("verbosity", -1)
        if n_estimators is not None:
            parameters["n_estimators"] = int(n_estimators)
        parameters["random_state"] = self.seed + seed_offset
        return LGBMClassifier(**parameters)

    def _create_catboost(
        self,
        seed_offset: int = 0,
        n_estimators: int | None = None,
    ):
        try:
            from catboost import CatBoostClassifier
        except ImportError as error:
            raise ImportError("CatBoost 교체 실험에는 catboost가 필요합니다.") from error
        parameters = dict(self.catboost_config)
        parameters.setdefault("loss_function", "MultiClass")
        parameters.setdefault("eval_metric", "MultiClass")
        parameters.setdefault("iterations", 350)
        parameters.setdefault("learning_rate", 0.025)
        parameters.setdefault("depth", 4)
        parameters.setdefault("l2_leaf_reg", 18.0)
        parameters.setdefault("random_strength", 1.0)
        parameters.setdefault("bootstrap_type", "Bernoulli")
        parameters.setdefault("subsample", 0.75)
        parameters.setdefault("rsm", 0.50)
        parameters.setdefault("thread_count", -1)
        parameters.setdefault("verbose", False)
        parameters.setdefault("allow_writing_files", False)
        if n_estimators is not None:
            parameters["iterations"] = int(n_estimators)
        parameters["random_seed"] = self.seed + seed_offset
        return CatBoostClassifier(**parameters)

    def _create_third_model(
        self,
        seed_offset: int = 0,
        n_estimators: int | None = None,
    ):
        if self.third_model_name == "catboost":
            return self._create_catboost(seed_offset, n_estimators)
        return self._create_lightgbm(seed_offset, n_estimators)

    def _third_early_stopping_rounds(self) -> int:
        if self.third_model_name == "catboost":
            return self.catboost_early_stopping_rounds
        return self.lgbm_early_stopping_rounds

    def _third_fallback_estimators(self) -> int:
        if self.third_model_name == "catboost":
            return int(self.catboost_config.get("iterations", 350))
        return int(self.lgbm_config.get("n_estimators", 280))

    def _fit_third_model(
        self,
        model,
        tree,
        labels: np.ndarray,
        fit_index: np.ndarray,
        stop_index: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        rounds = self._third_early_stopping_rounds()
        if rounds <= 0:
            fit_weight = None if sample_weight is None else sample_weight[fit_index]
            model.fit(
                tree[fit_index], labels[fit_index], sample_weight=fit_weight
            )
            return
        if self.third_model_name == "catboost":
            self._fit_catboost_model(
                model, tree, labels, fit_index, stop_index, rounds, sample_weight
            )
            return
        from lightgbm import early_stopping, log_evaluation

        fit_weight = None if sample_weight is None else sample_weight[fit_index]
        stop_weight = None if sample_weight is None else sample_weight[stop_index]
        model.fit(
            tree[fit_index],
            labels[fit_index],
            sample_weight=fit_weight,
            eval_X=tree[stop_index],
            eval_y=labels[stop_index],
            eval_metric="multi_logloss",
            eval_sample_weight=[stop_weight] if stop_weight is not None else None,
            callbacks=[
                early_stopping(rounds, verbose=False),
                log_evaluation(period=0),
            ],
        )

    @staticmethod
    def _fit_catboost_model(
        model,
        tree,
        labels: np.ndarray,
        fit_index: np.ndarray,
        stop_index: np.ndarray,
        rounds: int,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        fit_weight = None if sample_weight is None else sample_weight[fit_index]
        if rounds <= 0:
            model.fit(
                tree[fit_index], labels[fit_index], sample_weight=fit_weight
            )
            return
        model.fit(
            tree[fit_index],
            labels[fit_index],
            sample_weight=fit_weight,
            eval_set=(tree[stop_index], labels[stop_index]),
            early_stopping_rounds=rounds,
            use_best_model=True,
            verbose=False,
        )

    @staticmethod
    def _best_iteration_count(model, fallback: int) -> int:
        get_best_iteration = getattr(model, "get_best_iteration", None)
        if callable(get_best_iteration):
            try:
                value = int(get_best_iteration())
            except (TypeError, ValueError):
                value = -1
            if value >= 0:
                return value + 1
        for attribute in ("best_iteration", "best_iteration_"):
            try:
                value = int(getattr(model, attribute))
            except (AttributeError, TypeError, ValueError):
                continue
            if value >= 0:
                # XGBoost best_iteration은 0부터, LightGBM best_iteration_은
                # 1부터 시작합니다.
                return value + 1 if attribute == "best_iteration" else value
        return int(fallback)

    def _early_stopping_indices(
        self,
        train_index: np.ndarray,
        labels: np.ndarray,
        fold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        fit_index, stop_index = train_test_split(
            train_index,
            test_size=self.early_stopping_fraction,
            random_state=self.seed + 10_000 + fold,
            stratify=labels[train_index],
        )
        return np.asarray(fit_index), np.asarray(stop_index)

    @staticmethod
    def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
        logits = np.log(np.clip(probabilities, 1e-7, 1.0)) / temperature
        return softmax(logits, axis=1)

    def _text_probabilities(self, model: LinearSVC, features) -> np.ndarray:
        scores = np.asarray(model.decision_function(features), dtype="float64")
        if scores.ndim == 1:
            scores = np.column_stack((-scores, scores))
        return softmax(scores / self.text_temperature, axis=1)

    @staticmethod
    def _align_probabilities(
        probabilities: np.ndarray,
        model_classes: np.ndarray,
        all_classes: np.ndarray,
    ) -> np.ndarray:
        if np.array_equal(model_classes, all_classes):
            return np.asarray(probabilities, dtype="float64")
        aligned = np.full((len(probabilities), len(all_classes)), 1e-7, dtype="float64")
        positions = {value: index for index, value in enumerate(all_classes)}
        for source, value in enumerate(model_classes):
            aligned[:, positions[value]] = probabilities[:, source]
        return aligned / aligned.sum(axis=1, keepdims=True)

    @staticmethod
    def _meta_block(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(probabilities, 1e-7, 1.0)
        entropy = -(clipped * np.log(clipped)).sum(axis=1, keepdims=True)
        ordered = np.partition(clipped, -2, axis=1)
        margin = (ordered[:, -1] - ordered[:, -2]).reshape(-1, 1)
        return np.hstack((np.log(clipped), entropy, margin))

    def _meta_features(self, *probabilities: np.ndarray) -> np.ndarray:
        return np.hstack(
            tuple(self._meta_block(values) for values in probabilities)
        ).astype("float64", copy=False)

    def _fit_meta_feature_indices(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """OOF 메타 변수에서 분산과 fold 선택 안정성이 낮은 열을 제거합니다."""
        width = features.shape[1]
        if not self.meta_feature_selection_enabled:
            return np.arange(width, dtype="int64")
        variance = np.var(features, axis=0)
        eligible = np.flatnonzero(
            np.isfinite(variance)
            & (variance >= self.meta_feature_selection_min_variance)
        )
        if not len(eligible):
            raise ValueError("메타 피처 분산 필터를 통과한 변수가 없습니다.")
        selected_count = min(
            self.meta_feature_selection_max_features, len(eligible)
        )
        eligible_features = features[:, eligible]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            full_scores, _ = f_classif(eligible_features, labels)
        full_scores = np.nan_to_num(
            np.asarray(full_scores, dtype="float64"),
            nan=-np.inf,
            posinf=np.finfo("float64").max,
            neginf=-np.inf,
        )
        _, counts = np.unique(labels, return_counts=True)
        folds = min(self.meta_feature_selection_folds, int(counts.min()))
        frequency = np.zeros(len(eligible), dtype="int16")
        pool_count = min(
            len(eligible),
            max(
                selected_count,
                int(np.ceil(
                    selected_count * self.meta_feature_selection_top_multiplier
                )),
            ),
        )
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=self.seed + 25_000,
        )
        for train_index, _ in splitter.split(eligible_features, labels):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fold_scores, _ = f_classif(
                    eligible_features[train_index], labels[train_index]
                )
            fold_scores = np.nan_to_num(
                np.asarray(fold_scores, dtype="float64"),
                nan=-np.inf,
                posinf=np.finfo("float64").max,
                neginf=-np.inf,
            )
            order = np.lexsort((eligible, -fold_scores))
            frequency[order[:pool_count]] += 1
        stable = frequency >= min(
            self.meta_feature_selection_min_frequency, folds
        )
        rank = np.lexsort((eligible, -full_scores, -frequency, ~stable))
        self.meta_feature_eligible_count_ = len(eligible)
        return eligible[rank[:selected_count]]

    @staticmethod
    def _select_meta_features(
        features: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(features[:, indices], dtype="float64")

    def _create_meta_model(self, labels: np.ndarray | None = None):
        parameters = dict(self.meta_config)
        parameters.pop("type", None)
        if self.meta_model_type == "balanced_batch_mlp":
            from src.models.balanced_batch_meta_classifier import (
                BalancedBatchMetaClassifier,
            )

            for key in (
                "C", "solver", "class_weight", "max_iter", "l1_ratio",
                "penalty", "tol", "dual", "objective", "n_estimators",
                "max_depth", "min_child_weight", "subsample",
                "colsample_bytree", "reg_alpha", "reg_lambda", "gamma",
                "max_delta_step", "eval_metric", "tree_method", "n_jobs",
            ):
                parameters.pop(key, None)
            return BalancedBatchMetaClassifier(parameters, self.seed)
        if self.meta_model_type == "focal_mlp":
            from src.models.focal_meta_classifier import FocalMetaClassifier

            for key in (
                "C", "solver", "class_weight", "max_iter", "l1_ratio",
                "penalty", "tol", "dual", "objective", "n_estimators",
                "max_depth", "min_child_weight", "subsample",
                "colsample_bytree", "reg_alpha", "reg_lambda", "gamma",
                "max_delta_step", "eval_metric", "tree_method", "n_jobs",
            ):
                parameters.pop(key, None)
            return FocalMetaClassifier(parameters, self.seed)
        if self.meta_model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as error:
                raise ImportError("XGBoost 메타 학습기에는 xgboost가 필요합니다.") from error
            for key in (
                "C", "solver", "class_weight", "max_iter", "l1_ratio",
                "penalty", "tol", "dual",
            ):
                parameters.pop(key, None)
            parameters.setdefault("objective", "multi:softprob")
            parameters.setdefault("n_estimators", 120)
            parameters.setdefault("learning_rate", 0.03)
            parameters.setdefault("max_depth", 1)
            parameters.setdefault("min_child_weight", 10.0)
            parameters.setdefault("subsample", 0.70)
            parameters.setdefault("colsample_bytree", 0.65)
            parameters.setdefault("reg_alpha", 5.0)
            parameters.setdefault("reg_lambda", 30.0)
            parameters.setdefault("gamma", 0.1)
            parameters.setdefault("eval_metric", "mlogloss")
            parameters.setdefault("tree_method", "hist")
            parameters.setdefault("n_jobs", -1)
            parameters["random_state"] = self.seed
            return XGBClassifier(**parameters)
        parameters.setdefault("C", 0.05)
        parameters.setdefault("solver", "lbfgs")
        parameters.setdefault("class_weight", "balanced")
        if self.class_weight_mode == "none":
            parameters["class_weight"] = None
        elif self.class_weight_mode in {"sqrt_balanced", "effective_number"}:
            parameters["class_weight"] = self._class_weights(labels)
        parameters.setdefault("max_iter", 3000)
        parameters["random_state"] = self.seed
        return LogisticRegression(**parameters)

    def _meta_sample_weight(self, labels: np.ndarray) -> np.ndarray | None:
        if self.meta_model_type != "xgboost" or self.class_weight_mode == "none":
            return None
        if self.class_weight_mode in {"sqrt_balanced", "effective_number"}:
            weights = self._class_weights(labels)
            return np.asarray([weights[value] for value in labels], dtype="float64")
        configured = self.meta_config.get("class_weight")
        if configured in (None, "none"):
            return None
        if configured == "balanced":
            return np.asarray(
                compute_sample_weight(class_weight="balanced", y=labels),
                dtype="float64",
            )
        if isinstance(configured, dict):
            return np.asarray(
                [float(configured.get(value, 1.0)) for value in labels],
                dtype="float64",
            )
        raise ValueError("XGBoost meta class_weight는 balanced, dict 또는 none이어야 합니다.")

    def _fit_meta_model(
        self,
        model,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        sample_weight = self._meta_sample_weight(labels)
        if sample_weight is None:
            model.fit(features, labels)
        else:
            model.fit(features, labels, sample_weight=sample_weight)

    def _cross_fitted_meta_probabilities(
        self,
        meta_features: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("클래스가 초기화되지 않았습니다.")
        _, counts = np.unique(labels, return_counts=True)
        folds = min(self.postprocess_folds, int(counts.min()))
        if folds < 2:
            raise ValueError("후처리 OOF에는 클래스별 표본이 최소 2개 필요합니다.")
        probabilities = np.zeros(
            (len(labels), len(self.classes_)), dtype="float64"
        )
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=self.seed + 20_000
        )
        for train_index, valid_index in splitter.split(meta_features, labels):
            feature_indices = self._fit_meta_feature_indices(
                meta_features[train_index], labels[train_index]
            )
            fold_train_features = self._select_meta_features(
                meta_features[train_index], feature_indices
            )
            fold_valid_features = self._select_meta_features(
                meta_features[valid_index], feature_indices
            )
            scaler = StandardScaler()
            train_meta = scaler.fit_transform(fold_train_features)
            model = self._create_meta_model(labels[train_index])
            self._fit_meta_model(model, train_meta, labels[train_index])
            fold_probabilities = model.predict_proba(
                scaler.transform(fold_valid_features)
            )
            probabilities[valid_index] = self._align_probabilities(
                fold_probabilities, model.classes_, self.classes_
            )
        return probabilities

    @staticmethod
    def _apply_prior_postprocessing(
        probabilities: np.ndarray,
        class_prior: np.ndarray,
        gamma: float,
        temperature: float,
    ) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype="float64"), 1e-12, 1.0)
        adjusted = np.power(clipped, 1.0 / temperature)
        adjusted /= np.power(np.clip(class_prior, 1e-12, 1.0), gamma)
        return adjusted / np.clip(adjusted.sum(axis=1, keepdims=True), 1e-12, None)

    def _fit_postprocessing(
        self,
        oof_probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        if self.postprocess_method == "oof_crossfit_classwise_threshold":
            self._fit_classwise_threshold_postprocessing(
                oof_probabilities, labels
            )
            return
        if self.classes_ is None:
            raise RuntimeError("클래스가 초기화되지 않았습니다.")
        counts = np.asarray([
            np.count_nonzero(labels == value) for value in self.classes_
        ], dtype="float64")
        self.postprocess_class_prior_ = counts / counts.sum()
        baseline_predictions = self.classes_[oof_probabilities.argmax(axis=1)]
        baseline_score = float(f1_score(labels, baseline_predictions, average="macro"))
        self.postprocess_baseline_oof_macro_f1_ = baseline_score
        best_score = baseline_score
        best_gamma = 0.0
        best_temperature = 1.0
        self.postprocess_candidate_count_ = 0
        for gamma in self.postprocess_gamma_candidates:
            for temperature in self.postprocess_temperature_candidates:
                self.postprocess_candidate_count_ += 1
                adjusted = self._apply_prior_postprocessing(
                    oof_probabilities,
                    self.postprocess_class_prior_,
                    gamma,
                    temperature,
                )
                predictions = self.classes_[adjusted.argmax(axis=1)]
                score = float(f1_score(labels, predictions, average="macro"))
                candidate = (score, -abs(gamma), -abs(temperature - 1.0))
                current = (
                    best_score,
                    -abs(best_gamma),
                    -abs(best_temperature - 1.0),
                )
                if candidate > current:
                    best_score = score
                    best_gamma = gamma
                    best_temperature = temperature
        if best_score >= baseline_score + self.postprocess_min_oof_gain:
            self.selected_postprocess_gamma_ = best_gamma
            self.selected_postprocess_temperature_ = best_temperature
            self.postprocess_selected_oof_macro_f1_ = best_score
        else:
            self.selected_postprocess_gamma_ = 0.0
            self.selected_postprocess_temperature_ = 1.0
            self.postprocess_selected_oof_macro_f1_ = baseline_score

    @staticmethod
    def _apply_classwise_thresholds(
        probabilities: np.ndarray,
        threshold_factors: np.ndarray,
    ) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype="float64"), 1e-12, 1.0)
        adjusted = clipped / np.clip(threshold_factors, 1e-12, None)[None, :]
        return adjusted / np.clip(adjusted.sum(axis=1, keepdims=True), 1e-12, None)

    def _optimize_classwise_thresholds(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Macro F1 목적과 log-L2 shrinkage로 클래스별 배율을 제한적으로 선택합니다."""
        if self.classes_ is None:
            raise RuntimeError("클래스가 초기화되지 않았습니다.")
        factors = np.ones(len(self.classes_), dtype="float64")
        supports = np.asarray(
            [np.count_nonzero(labels == value) for value in self.classes_],
            dtype="int64",
        )

        def objective(candidate_factors: np.ndarray) -> tuple[float, float]:
            adjusted = self._apply_classwise_thresholds(
                probabilities, candidate_factors
            )
            predictions = self.classes_[adjusted.argmax(axis=1)]
            score = float(f1_score(labels, predictions, average="macro"))
            penalty = self.postprocess_threshold_shrinkage * float(
                np.square(np.log(candidate_factors)).mean()
            )
            return score - penalty, score

        best_objective, _ = objective(factors)
        for _ in range(self.postprocess_threshold_rounds):
            changed = False
            for class_index, support in enumerate(supports):
                if support < self.postprocess_min_class_support:
                    continue
                best_factor = factors[class_index]
                best_candidate_objective = best_objective
                for factor in self.postprocess_threshold_candidates:
                    self.postprocess_candidate_count_ += 1
                    candidate = factors.copy()
                    candidate[class_index] = factor
                    candidate_objective, _ = objective(candidate)
                    proposed = (
                        candidate_objective,
                        -abs(np.log(factor)),
                    )
                    current = (
                        best_candidate_objective,
                        -abs(np.log(best_factor)),
                    )
                    if proposed > current:
                        best_candidate_objective = candidate_objective
                        best_factor = factor
                if best_factor != factors[class_index]:
                    factors[class_index] = best_factor
                    best_objective = best_candidate_objective
                    changed = True
            if not changed:
                break
        return factors

    def _fit_classwise_threshold_postprocessing(
        self,
        oof_probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """임계 배율을 별도 cross-fitting으로 검증하고 이득이 있을 때만 채택합니다."""
        if self.classes_ is None:
            raise RuntimeError("클래스가 초기화되지 않았습니다.")
        baseline_predictions = self.classes_[oof_probabilities.argmax(axis=1)]
        baseline_score = float(f1_score(labels, baseline_predictions, average="macro"))
        self.postprocess_baseline_oof_macro_f1_ = baseline_score
        self.postprocess_candidate_count_ = 0

        _, counts = np.unique(labels, return_counts=True)
        folds = min(self.postprocess_folds, int(counts.min()))
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=self.seed + 40_000,
        )
        cross_fitted = np.zeros_like(oof_probabilities, dtype="float64")
        for train_index, valid_index in splitter.split(oof_probabilities, labels):
            factors = self._optimize_classwise_thresholds(
                oof_probabilities[train_index], labels[train_index]
            )
            cross_fitted[valid_index] = self._apply_classwise_thresholds(
                oof_probabilities[valid_index], factors
            )
        cross_fitted_predictions = self.classes_[cross_fitted.argmax(axis=1)]
        selected_score = float(
            f1_score(labels, cross_fitted_predictions, average="macro")
        )
        if selected_score >= baseline_score + self.postprocess_min_oof_gain:
            self.selected_postprocess_threshold_factors_ = (
                self._optimize_classwise_thresholds(oof_probabilities, labels)
            )
            self.postprocess_selected_oof_macro_f1_ = selected_score
        else:
            self.selected_postprocess_threshold_factors_ = np.ones(
                len(self.classes_), dtype="float64"
            )
            self.postprocess_selected_oof_macro_f1_ = baseline_score

    def _postprocess_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        if not self.postprocess_enabled:
            return probabilities
        if self.postprocess_method == "oof_crossfit_classwise_threshold":
            if len(self.selected_postprocess_threshold_factors_) == 0:
                raise RuntimeError("클래스별 threshold 후처리를 fit한 뒤 예측해야 합니다.")
            return self._apply_classwise_thresholds(
                probabilities,
                self.selected_postprocess_threshold_factors_,
            )
        if self.postprocess_class_prior_ is None:
            raise RuntimeError("후처리 상태를 fit한 뒤 예측해야 합니다.")
        return self._apply_prior_postprocessing(
            probabilities,
            self.postprocess_class_prior_,
            self.selected_postprocess_gamma_,
            self.selected_postprocess_temperature_,
        )

    def _fit_temperature(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        candidates = np.asarray(
            self.blend_config.get(
                "temperature_candidates",
                (0.8, 1.0, 1.2, 1.4, 1.6, 2.0),
            ),
            dtype="float64",
        )
        if candidates.ndim != 1 or len(candidates) == 0 or np.any(candidates <= 0):
            raise ValueError("temperature_candidates는 양수 목록이어야 합니다.")
        class_positions = {value: index for index, value in enumerate(self.classes_)}
        target = np.asarray([class_positions[value] for value in labels], dtype="int64")
        losses = []
        for temperature in candidates:
            calibrated = self._temperature_scale(probabilities, float(temperature))
            losses.append(
                -np.log(np.clip(calibrated[np.arange(len(target)), target], 1e-12, 1.0)).mean()
            )
        return float(candidates[int(np.argmin(losses))])

    @staticmethod
    def _log_probability_blend(
        probabilities: tuple[np.ndarray, np.ndarray, np.ndarray],
        weights: np.ndarray,
    ) -> np.ndarray:
        logits = sum(
            float(weight) * np.log(np.clip(values, 1e-12, 1.0))
            for weight, values in zip(weights, probabilities)
        )
        return softmax(logits, axis=1)

    def _fit_constrained_blend(
        self,
        text_oof: np.ndarray,
        xgb_oof: np.ndarray,
        third_oof: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        raw = (text_oof, xgb_oof, third_oof)
        self.calibrated_temperatures_ = np.asarray(
            [self._fit_temperature(values, labels) for values in raw],
            dtype="float64",
        )
        calibrated = tuple(
            self._temperature_scale(values, temperature)
            for values, temperature in zip(raw, self.calibrated_temperatures_)
        )
        names = ("text", "xgboost", "third")
        default_bounds = ((0.30, 0.70), (0.10, 0.45), (0.10, 0.45))
        configured_bounds = self.blend_config.get("weight_bounds", {})
        bounds = tuple(
            tuple(float(value) for value in configured_bounds.get(name, default))
            for name, default in zip(names, default_bounds)
        )
        if any(len(bound) != 2 or bound[0] < 0 or bound[0] > bound[1] for bound in bounds):
            raise ValueError("각 blend weight_bounds는 [최솟값, 최댓값]이어야 합니다.")
        initial = np.asarray(
            self.blend_config.get("initial_weights", (0.50, 0.25, 0.25)),
            dtype="float64",
        )
        if initial.shape != (3,) or np.any(initial < 0) or initial.sum() <= 0:
            raise ValueError("initial_weights는 음수가 아닌 3개 값이어야 합니다.")
        initial /= initial.sum()
        class_positions = {value: index for index, value in enumerate(self.classes_)}
        target = np.asarray([class_positions[value] for value in labels], dtype="int64")
        shrinkage = float(self.blend_config.get("weight_l2", 0.01))

        def objective(weights: np.ndarray) -> float:
            blended = self._log_probability_blend(calibrated, weights)
            loss = -np.log(
                np.clip(blended[np.arange(len(target)), target], 1e-12, 1.0)
            ).mean()
            return float(loss + shrinkage * np.square(weights - initial).sum())

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
            options={"maxiter": int(self.blend_config.get("max_iter", 300))},
        )
        if not result.success:
            raise RuntimeError(f"제약형 blend 가중치 최적화 실패: {result.message}")
        self.blend_weights_ = np.asarray(result.x, dtype="float64")

    def fit(self, features, labels):
        bundle = self._require_bundle(features)
        xgboost_tree, third_tree, fourth_tree = self._tree_views(bundle)
        y = np.asarray(labels)
        self.classes_, counts = np.unique(y, return_counts=True)
        folds = min(self.stacking_folds, int(counts.min()))
        if folds < 2:
            raise ValueError("OOF stacking에는 클래스별 학습 표본이 최소 2개 필요합니다.")

        rows = len(y)
        class_count = len(self.classes_)
        text_oof = np.zeros((rows, class_count), dtype="float64")
        xgb_oof = np.zeros_like(text_oof)
        lgbm_oof = np.zeros_like(text_oof)
        fourth_oof = (
            np.zeros_like(text_oof) if self.fourth_model_name is not None else None
        )
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        xgb_best_counts: list[int] = []
        lgbm_best_counts: list[int] = []
        fourth_best_counts: list[int] = []
        base_sample_weight = self._base_sample_weight(y)

        for fold, (train_index, valid_index) in enumerate(splitter.split(np.zeros(rows), y)):
            offset = 100 * (fold + 1)
            text_model = self._create_text_model(offset, y[train_index])
            early_stop_enabled = (
                self.xgb_early_stopping_rounds > 0
                or self._third_early_stopping_rounds() > 0
                or (
                    self.fourth_model_name == "catboost"
                    and self.catboost_early_stopping_rounds > 0
                )
            )
            if early_stop_enabled:
                fit_index, stop_index = self._early_stopping_indices(
                    train_index, y, fold
                )
            else:
                fit_index, stop_index = train_index, np.asarray([], dtype="int64")
            xgb_model = self._create_xgboost(
                offset + 1,
                enable_early_stopping=self.xgb_early_stopping_rounds > 0,
            )
            lgbm_model = self._create_third_model(offset + 2)
            fourth_model = (
                self._create_catboost(offset + 3)
                if self.fourth_model_name == "catboost"
                else None
            )
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
                    xgboost_tree[fit_index],
                    y[fit_index],
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
                    xgboost_tree[train_index],
                    y[train_index],
                    sample_weight=train_weight,
                )
            self._fit_third_model(
                lgbm_model,
                third_tree,
                y,
                fit_index if self._third_early_stopping_rounds() > 0 else train_index,
                stop_index,
                base_sample_weight,
            )
            if fourth_model is not None:
                fourth_fit_index = (
                    fit_index
                    if self.catboost_early_stopping_rounds > 0
                    else train_index
                )
                self._fit_catboost_model(
                    fourth_model,
                    fourth_tree,
                    y,
                    fourth_fit_index,
                    stop_index,
                    self.catboost_early_stopping_rounds,
                    base_sample_weight,
                )
            xgb_best_counts.append(
                self._best_iteration_count(
                    xgb_model, self.xgb_config.get("n_estimators", 350)
                )
            )
            lgbm_best_counts.append(
                self._best_iteration_count(
                    lgbm_model, self._third_fallback_estimators()
                )
            )
            if fourth_model is not None:
                fourth_best_counts.append(
                    self._best_iteration_count(
                        fourth_model, self.catboost_config.get("iterations", 350)
                    )
                )

            text_oof[valid_index] = self._align_probabilities(
                self._text_probabilities(text_model, bundle.text[valid_index]),
                text_model.classes_, self.classes_,
            )
            xgb_oof[valid_index] = self._temperature_scale(
                self._align_probabilities(
                    xgb_model.predict_proba(xgboost_tree[valid_index]),
                    xgb_model.classes_, self.classes_,
                ),
                self.xgb_temperature,
            )
            lgbm_oof[valid_index] = self._temperature_scale(
                self._align_probabilities(
                    lgbm_model.predict_proba(third_tree[valid_index]),
                    lgbm_model.classes_, self.classes_,
                ),
                self.lgbm_temperature,
            )
            if fourth_model is not None and fourth_oof is not None:
                fourth_oof[valid_index] = self._temperature_scale(
                    self._align_probabilities(
                        fourth_model.predict_proba(fourth_tree[valid_index]),
                        fourth_model.classes_,
                        self.classes_,
                    ),
                    self.fourth_temperature,
                )

        self.inner_oof_rows_ = rows
        if self.ensemble_method == "constrained_log_blend":
            self._fit_constrained_blend(text_oof, xgb_oof, lgbm_oof, y)
            self.meta_feature_count_ = 0
            self.meta_model = None
        else:
            oof_probabilities = [text_oof, xgb_oof, lgbm_oof]
            if fourth_oof is not None:
                oof_probabilities.append(fourth_oof)
            meta_features = self._meta_features(*oof_probabilities)
            postprocess_oof = (
                self._cross_fitted_meta_probabilities(meta_features, y)
                if self.postprocess_enabled
                else None
            )
            self.meta_input_feature_count_ = int(meta_features.shape[1])
            self.meta_feature_indices_ = self._fit_meta_feature_indices(
                meta_features, y
            )
            selected_meta_features = self._select_meta_features(
                meta_features, self.meta_feature_indices_
            )
            self.meta_feature_count_ = int(selected_meta_features.shape[1])
            scaled_meta = self.meta_scaler.fit_transform(selected_meta_features)
            self.meta_model = self._create_meta_model(y)
            self._fit_meta_model(self.meta_model, scaled_meta, y)
            if postprocess_oof is not None:
                self._fit_postprocessing(postprocess_oof, y)

        self.selected_xgb_estimators_ = int(np.median(xgb_best_counts))
        self.selected_lgbm_estimators_ = int(np.median(lgbm_best_counts))
        self.selected_fourth_estimators_ = (
            int(np.median(fourth_best_counts)) if fourth_best_counts else 0
        )
        self.text_model = self._create_text_model(labels=y)
        self.xgb_model = self._create_xgboost(
            1, n_estimators=self.selected_xgb_estimators_
        )
        self.lgbm_model = self._create_third_model(
            2, n_estimators=self.selected_lgbm_estimators_
        )
        self.fourth_model = (
            self._create_catboost(3, n_estimators=self.selected_fourth_estimators_)
            if self.fourth_model_name == "catboost"
            else None
        )
        self.text_model.fit(bundle.text, y)
        self.xgb_model.fit(xgboost_tree, y, sample_weight=base_sample_weight)
        self.lgbm_model.fit(third_tree, y, sample_weight=base_sample_weight)
        if self.fourth_model is not None:
            self.fourth_model.fit(
                fourth_tree, y, sample_weight=base_sample_weight
            )
        return self

    def _base_probabilities(self, bundle: OOFStackingFeatureBundle):
        if self.classes_ is None or any(
            model is None for model in (self.text_model, self.xgb_model, self.lgbm_model)
        ):
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        xgboost_tree, third_tree, fourth_tree = self._tree_views(bundle)
        text = self._align_probabilities(
            self._text_probabilities(self.text_model, bundle.text),
            self.text_model.classes_, self.classes_,
        )
        xgb = self._temperature_scale(
            self._align_probabilities(
                self.xgb_model.predict_proba(xgboost_tree),
                self.xgb_model.classes_, self.classes_,
            ),
            self.xgb_temperature,
        )
        lgbm = self._temperature_scale(
            self._align_probabilities(
                self.lgbm_model.predict_proba(third_tree),
                self.lgbm_model.classes_, self.classes_,
            ),
            self.lgbm_temperature,
        )
        probabilities = [text, xgb, lgbm]
        if self.fourth_model_name is not None:
            if self.fourth_model is None:
                raise RuntimeError("네 번째 모델을 fit한 뒤 예측해야 합니다.")
            probabilities.append(
                self._temperature_scale(
                    self._align_probabilities(
                        self.fourth_model.predict_proba(fourth_tree),
                        self.fourth_model.classes_,
                        self.classes_,
                    ),
                    self.fourth_temperature,
                )
            )
        return tuple(probabilities)

    def predict_proba(self, features) -> np.ndarray:
        if self.ensemble_method == "regularized_stacking" and self.meta_model is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        bundle = self._require_bundle(features)
        base = self._base_probabilities(bundle)
        if self.ensemble_method == "constrained_log_blend":
            calibrated = tuple(
                self._temperature_scale(values, temperature)
                for values, temperature in zip(base, self.calibrated_temperatures_)
            )
            return self._log_probability_blend(calibrated, self.blend_weights_)
        meta = self._meta_features(*base)
        meta = self._select_meta_features(meta, self.meta_feature_indices_)
        probabilities = self.meta_model.predict_proba(self.meta_scaler.transform(meta))
        aligned = self._align_probabilities(
            probabilities, self.meta_model.classes_, self.classes_
        )
        return self._postprocess_probabilities(aligned)

    def predict(self, features) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        return self.classes_[self.predict_proba(features).argmax(axis=1)]

    def summary(self) -> dict[str, object]:
        return {
            "ensemble_method": self.ensemble_method,
            "base_learners": 4 if self.fourth_model_name is not None else 3,
            "third_model": self.third_model_name,
            "fourth_model": self.fourth_model_name,
            "class_weight_mode": self.class_weight_mode,
            "class_weight_power": self.class_weight_power,
            "effective_number_beta": self.effective_number_beta,
            "base_sample_weight_enabled": self.base_sample_weight_enabled,
            "stacking_folds": self.stacking_folds,
            "meta_features": self.meta_feature_count_,
            "meta_input_features": self.meta_input_feature_count_,
            "meta_feature_selection_enabled": self.meta_feature_selection_enabled,
            "meta_feature_selection_eligible": self.meta_feature_eligible_count_,
            "meta_feature_indices": self.meta_feature_indices_.tolist(),
            "meta_model_type": self.meta_model_type,
            "inner_oof_rows": self.inner_oof_rows_,
            "xgboost_early_stopping_rounds": self.xgb_early_stopping_rounds,
            "lightgbm_early_stopping_rounds": self.lgbm_early_stopping_rounds,
            "catboost_early_stopping_rounds": self.catboost_early_stopping_rounds,
            "selected_xgboost_estimators": self.selected_xgb_estimators_,
            "selected_third_model_estimators": self.selected_lgbm_estimators_,
            "selected_fourth_model_estimators": self.selected_fourth_estimators_,
            "selected_lightgbm_estimators": (
                self.selected_lgbm_estimators_
                if self.third_model_name == "lightgbm"
                else 0
            ),
            "calibrated_temperatures": self.calibrated_temperatures_.tolist(),
            "blend_weights": self.blend_weights_.tolist(),
            "postprocessing_enabled": self.postprocess_enabled,
            "postprocessing_method": self.postprocess_method,
            "postprocessing_candidates": self.postprocess_candidate_count_,
            "postprocessing_gamma": self.selected_postprocess_gamma_,
            "postprocessing_temperature": self.selected_postprocess_temperature_,
            "postprocessing_baseline_oof_macro_f1": self.postprocess_baseline_oof_macro_f1_,
            "postprocessing_selected_oof_macro_f1": self.postprocess_selected_oof_macro_f1_,
            "postprocessing_threshold_factors": (
                self.selected_postprocess_threshold_factors_.tolist()
            ),
            "postprocessing_threshold_factor_min": (
                float(self.selected_postprocess_threshold_factors_.min())
                if len(self.selected_postprocess_threshold_factors_)
                else 1.0
            ),
            "postprocessing_threshold_factor_max": (
                float(self.selected_postprocess_threshold_factors_.max())
                if len(self.selected_postprocess_threshold_factors_)
                else 1.0
            ),
        }


def create_model(model_config: dict, seed: int) -> PipeCombOOFStackingClassifier:
    return PipeCombOOFStackingClassifier(model_config, seed)
