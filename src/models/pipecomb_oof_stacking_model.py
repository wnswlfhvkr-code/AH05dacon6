"""JSJ9 text와 EM 계열 tree의 교차 적합 확률을 결합하는 앙상블 모델."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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
        self.class_weight_mode = str(
            model_config.get("class_weight_mode", "configured")
        ).lower()
        if self.class_weight_mode not in {"configured", "none", "sqrt_balanced"}:
            raise ValueError(
                f"지원하지 않는 클래스 가중치 방식입니다: {self.class_weight_mode}"
            )
        class_weight_clip = model_config.get("class_weight_clip", (0.75, 2.5))
        self.class_weight_clip = tuple(float(value) for value in class_weight_clip)
        self.class_weight_power = float(model_config.get("class_weight_power", 0.5))
        if (
            len(self.class_weight_clip) != 2
            or self.class_weight_clip[0] <= 0
            or self.class_weight_clip[0] > self.class_weight_clip[1]
        ):
            raise ValueError("class_weight_clip은 양수인 [최솟값, 최댓값]이어야 합니다.")
        if not 0.0 <= self.class_weight_power <= 1.0:
            raise ValueError("class_weight_power는 0 이상 1 이하여야 합니다.")
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
        if min(self.text_temperature, self.xgb_temperature, self.lgbm_temperature) <= 0:
            raise ValueError("확률 temperature는 0보다 커야 합니다.")

        self.classes_: np.ndarray | None = None
        self.text_model = None
        self.xgb_model = None
        self.lgbm_model = None
        self.meta_scaler = StandardScaler()
        self.meta_model = None
        self.inner_oof_rows_ = 0
        self.meta_feature_count_ = 0
        self.selected_xgb_estimators_ = 0
        self.selected_lgbm_estimators_ = 0
        self.calibrated_temperatures_ = np.ones(3, dtype="float64")
        self.blend_weights_ = np.full(3, 1.0 / 3.0, dtype="float64")

    @staticmethod
    def _require_bundle(features) -> OOFStackingFeatureBundle:
        if not all(hasattr(features, name) for name in ("text", "tree")):
            raise TypeError(
                "pipecomb_oof_stacking에는 "
                "text와 tree 뷰를 제공하는 pipeComb stacking 전처리가 필요합니다."
            )
        return features

    def _class_weights(self, labels: np.ndarray | None):
        if self.class_weight_mode == "configured":
            return None
        if self.class_weight_mode == "none":
            return None
        if labels is None:
            raise ValueError("sqrt_balanced 클래스 가중치에는 학습 labels가 필요합니다.")
        classes, counts = np.unique(labels, return_counts=True)
        raw = (
            len(labels) / (len(classes) * counts.astype("float64"))
        ) ** self.class_weight_power
        clipped = np.clip(raw, *self.class_weight_clip)
        return {
            value.item() if hasattr(value, "item") else value: float(weight)
            for value, weight in zip(classes, clipped)
        }

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
        elif self.class_weight_mode == "sqrt_balanced":
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
    ) -> None:
        rounds = self._third_early_stopping_rounds()
        if rounds <= 0:
            model.fit(tree[fit_index], labels[fit_index])
            return
        if self.third_model_name == "catboost":
            model.fit(
                tree[fit_index],
                labels[fit_index],
                eval_set=(tree[stop_index], labels[stop_index]),
                early_stopping_rounds=rounds,
                use_best_model=True,
                verbose=False,
            )
            return
        from lightgbm import early_stopping, log_evaluation

        model.fit(
            tree[fit_index],
            labels[fit_index],
            eval_X=tree[stop_index],
            eval_y=labels[stop_index],
            eval_metric="multi_logloss",
            callbacks=[
                early_stopping(rounds, verbose=False),
                log_evaluation(period=0),
            ],
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

    def _meta_features(self, text: np.ndarray, xgb: np.ndarray, lgbm: np.ndarray) -> np.ndarray:
        return np.hstack(
            (self._meta_block(text), self._meta_block(xgb), self._meta_block(lgbm))
        ).astype("float64", copy=False)

    def _create_meta_model(
        self,
        labels: np.ndarray | None = None,
    ) -> LogisticRegression:
        parameters = dict(self.meta_config)
        parameters.setdefault("C", 0.05)
        parameters.setdefault("solver", "lbfgs")
        parameters.setdefault("class_weight", "balanced")
        if self.class_weight_mode == "none":
            parameters["class_weight"] = None
        elif self.class_weight_mode == "sqrt_balanced":
            parameters["class_weight"] = self._class_weights(labels)
        parameters.setdefault("max_iter", 3000)
        parameters["random_state"] = self.seed
        return LogisticRegression(**parameters)

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
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        xgb_best_counts: list[int] = []
        lgbm_best_counts: list[int] = []

        for fold, (train_index, valid_index) in enumerate(splitter.split(np.zeros(rows), y)):
            offset = 100 * (fold + 1)
            text_model = self._create_text_model(offset, y[train_index])
            early_stop_enabled = (
                self.xgb_early_stopping_rounds > 0
                or self._third_early_stopping_rounds() > 0
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
            text_model.fit(bundle.text[train_index], y[train_index])
            if self.xgb_early_stopping_rounds > 0:
                xgb_model.fit(
                    bundle.tree[fit_index],
                    y[fit_index],
                    eval_set=[(bundle.tree[stop_index], y[stop_index])],
                    verbose=False,
                )
            else:
                xgb_model.fit(bundle.tree[train_index], y[train_index])
            self._fit_third_model(
                lgbm_model,
                bundle.tree,
                y,
                fit_index if self._third_early_stopping_rounds() > 0 else train_index,
                stop_index,
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

            text_oof[valid_index] = self._align_probabilities(
                self._text_probabilities(text_model, bundle.text[valid_index]),
                text_model.classes_, self.classes_,
            )
            xgb_oof[valid_index] = self._temperature_scale(
                self._align_probabilities(
                    xgb_model.predict_proba(bundle.tree[valid_index]),
                    xgb_model.classes_, self.classes_,
                ),
                self.xgb_temperature,
            )
            lgbm_oof[valid_index] = self._temperature_scale(
                self._align_probabilities(
                    lgbm_model.predict_proba(bundle.tree[valid_index]),
                    lgbm_model.classes_, self.classes_,
                ),
                self.lgbm_temperature,
            )

        self.inner_oof_rows_ = rows
        if self.ensemble_method == "constrained_log_blend":
            self._fit_constrained_blend(text_oof, xgb_oof, lgbm_oof, y)
            self.meta_feature_count_ = 0
            self.meta_model = None
        else:
            meta_features = self._meta_features(text_oof, xgb_oof, lgbm_oof)
            self.meta_feature_count_ = int(meta_features.shape[1])
            scaled_meta = self.meta_scaler.fit_transform(meta_features)
            self.meta_model = self._create_meta_model(y)
            self.meta_model.fit(scaled_meta, y)

        self.selected_xgb_estimators_ = int(np.median(xgb_best_counts))
        self.selected_lgbm_estimators_ = int(np.median(lgbm_best_counts))
        self.text_model = self._create_text_model(labels=y)
        self.xgb_model = self._create_xgboost(
            1, n_estimators=self.selected_xgb_estimators_
        )
        self.lgbm_model = self._create_third_model(
            2, n_estimators=self.selected_lgbm_estimators_
        )
        self.text_model.fit(bundle.text, y)
        self.xgb_model.fit(bundle.tree, y)
        self.lgbm_model.fit(bundle.tree, y)
        return self

    def _base_probabilities(self, bundle: OOFStackingFeatureBundle):
        if self.classes_ is None or any(
            model is None for model in (self.text_model, self.xgb_model, self.lgbm_model)
        ):
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        text = self._align_probabilities(
            self._text_probabilities(self.text_model, bundle.text),
            self.text_model.classes_, self.classes_,
        )
        xgb = self._temperature_scale(
            self._align_probabilities(
                self.xgb_model.predict_proba(bundle.tree),
                self.xgb_model.classes_, self.classes_,
            ),
            self.xgb_temperature,
        )
        lgbm = self._temperature_scale(
            self._align_probabilities(
                self.lgbm_model.predict_proba(bundle.tree),
                self.lgbm_model.classes_, self.classes_,
            ),
            self.lgbm_temperature,
        )
        return text, xgb, lgbm

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
        probabilities = self.meta_model.predict_proba(self.meta_scaler.transform(meta))
        return self._align_probabilities(
            probabilities, self.meta_model.classes_, self.classes_
        )

    def predict(self, features) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        return self.classes_[self.predict_proba(features).argmax(axis=1)]

    def summary(self) -> dict[str, object]:
        return {
            "ensemble_method": self.ensemble_method,
            "base_learners": 3,
            "third_model": self.third_model_name,
            "class_weight_mode": self.class_weight_mode,
            "class_weight_power": self.class_weight_power,
            "stacking_folds": self.stacking_folds,
            "meta_features": self.meta_feature_count_,
            "inner_oof_rows": self.inner_oof_rows_,
            "xgboost_early_stopping_rounds": self.xgb_early_stopping_rounds,
            "lightgbm_early_stopping_rounds": self.lgbm_early_stopping_rounds,
            "catboost_early_stopping_rounds": self.catboost_early_stopping_rounds,
            "selected_xgboost_estimators": self.selected_xgb_estimators_,
            "selected_third_model_estimators": self.selected_lgbm_estimators_,
            "selected_lightgbm_estimators": (
                self.selected_lgbm_estimators_
                if self.third_model_name == "lightgbm"
                else 0
            ),
            "calibrated_temperatures": self.calibrated_temperatures_.tolist(),
            "blend_weights": self.blend_weights_.tolist(),
        }


def create_model(model_config: dict, seed: int) -> PipeCombOOFStackingClassifier:
    return PipeCombOOFStackingClassifier(model_config, seed)
