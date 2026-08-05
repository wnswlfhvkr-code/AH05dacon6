"""JSJ9·EMV45 교차 적합 확률을 사용하는 규제형 stacking 모델."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.pipelines.em_preprocessing.pipeline_pipeComb_em_v1_006 import OOFStackingFeatureBundle


class PipeCombOOFStackingClassifier:
    """TF-IDF SVC와 EMV45 트리 모델을 OOF meta learner로 결합합니다."""

    def __init__(self, model_config: dict, seed: int) -> None:
        self.seed = int(seed)
        self.stacking_folds = int(model_config.get("stacking_folds", 5))
        if self.stacking_folds < 2:
            raise ValueError("stacking_folds는 2 이상이어야 합니다.")

        self.text_config = dict(model_config.get("text_linear_svc", {}))
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
        self.xgb_early_stopping_rounds = int(
            self.xgb_config.pop("early_stopping_rounds", 0) or 0
        )
        self.lgbm_early_stopping_rounds = int(
            self.lgbm_config.pop("early_stopping_rounds", 0) or 0
        )
        self.early_stopping_fraction = float(
            model_config.get("early_stopping_fraction", 0.15)
        )
        if not 0.0 < self.early_stopping_fraction < 0.5:
            raise ValueError("early_stopping_fraction은 0보다 크고 0.5보다 작아야 합니다.")
        self.meta_config = dict(model_config.get("meta_learner", {}))
        self.text_temperature = float(model_config.get("text_temperature", 1.20))
        self.xgb_temperature = float(model_config.get("xgboost_temperature", 1.15))
        self.lgbm_temperature = float(model_config.get("lightgbm_temperature", 1.15))
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

    @staticmethod
    def _require_bundle(features) -> OOFStackingFeatureBundle:
        if not all(hasattr(features, name) for name in ("text", "tree")):
            raise TypeError(
                "pipecomb_oof_stacking에는 "
                "text와 tree 뷰를 제공하는 pipeComb stacking 전처리가 필요합니다."
            )
        return features

    def _create_text_model(self, seed_offset: int = 0) -> LinearSVC:
        parameters = dict(self.text_config)
        parameters.setdefault("C", 0.10)
        parameters.setdefault("class_weight", "balanced")
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

    @staticmethod
    def _best_iteration_count(model, fallback: int) -> int:
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

    def _create_meta_model(self) -> LogisticRegression:
        parameters = dict(self.meta_config)
        parameters.setdefault("C", 0.05)
        parameters.setdefault("solver", "lbfgs")
        parameters.setdefault("class_weight", "balanced")
        parameters.setdefault("max_iter", 3000)
        parameters["random_state"] = self.seed
        return LogisticRegression(**parameters)

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
            text_model = self._create_text_model(offset)
            early_stop_enabled = (
                self.xgb_early_stopping_rounds > 0
                or self.lgbm_early_stopping_rounds > 0
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
            lgbm_model = self._create_lightgbm(offset + 2)
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
            if self.lgbm_early_stopping_rounds > 0:
                from lightgbm import early_stopping, log_evaluation

                lgbm_model.fit(
                    bundle.tree[fit_index],
                    y[fit_index],
                    eval_X=bundle.tree[stop_index],
                    eval_y=y[stop_index],
                    eval_metric="multi_logloss",
                    callbacks=[
                        early_stopping(
                            self.lgbm_early_stopping_rounds,
                            verbose=False,
                        ),
                        log_evaluation(period=0),
                    ],
                )
            else:
                lgbm_model.fit(bundle.tree[train_index], y[train_index])
            xgb_best_counts.append(
                self._best_iteration_count(
                    xgb_model, self.xgb_config.get("n_estimators", 350)
                )
            )
            lgbm_best_counts.append(
                self._best_iteration_count(
                    lgbm_model, self.lgbm_config.get("n_estimators", 280)
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

        meta_features = self._meta_features(text_oof, xgb_oof, lgbm_oof)
        self.meta_feature_count_ = int(meta_features.shape[1])
        self.inner_oof_rows_ = rows
        scaled_meta = self.meta_scaler.fit_transform(meta_features)
        self.meta_model = self._create_meta_model()
        self.meta_model.fit(scaled_meta, y)

        self.selected_xgb_estimators_ = int(np.median(xgb_best_counts))
        self.selected_lgbm_estimators_ = int(np.median(lgbm_best_counts))
        self.text_model = self._create_text_model()
        self.xgb_model = self._create_xgboost(
            1, n_estimators=self.selected_xgb_estimators_
        )
        self.lgbm_model = self._create_lightgbm(
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
        if self.meta_model is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        bundle = self._require_bundle(features)
        meta = self._meta_features(*self._base_probabilities(bundle))
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
            "ensemble_method": "cross_fitted_regularized_stacking",
            "base_learners": 3,
            "stacking_folds": self.stacking_folds,
            "meta_features": self.meta_feature_count_,
            "inner_oof_rows": self.inner_oof_rows_,
            "xgboost_early_stopping_rounds": self.xgb_early_stopping_rounds,
            "lightgbm_early_stopping_rounds": self.lgbm_early_stopping_rounds,
            "selected_xgboost_estimators": self.selected_xgb_estimators_,
            "selected_lightgbm_estimators": self.selected_lgbm_estimators_,
        }


def create_model(model_config: dict, seed: int) -> PipeCombOOFStackingClassifier:
    return PipeCombOOFStackingClassifier(model_config, seed)
