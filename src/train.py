"""설정 파일에 지정된 모델을 학습하고 제출 파일을 생성합니다."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.models import MODEL_BUILDERS
from src.pipelines.preprocessing_registry import create_preprocessing_pipeline


def _deep_merge_config(base: dict, override: dict) -> dict:
    """중첩 설정을 복사한 뒤 variant에서 지정한 값만 재귀적으로 덮어씁니다."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: Path, _visited: set[Path] | None = None) -> dict:
    """YAML을 읽고 선택적 extends 체인을 순환 없이 병합합니다."""
    resolved = path.resolve()
    visited = set() if _visited is None else set(_visited)
    if resolved in visited:
        chain = " -> ".join(str(item) for item in (*visited, resolved))
        raise ValueError(f"설정 extends 순환이 감지됐습니다: {chain}")
    visited.add(resolved)
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    parent = config.pop("extends", None)
    if parent is None:
        return config
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_merge_config(load_config(parent_path, visited), config)


def build_model(config: dict, class_names=None):
    """설정의 모델 이름에 맞는 생성 함수를 호출합니다."""
    model_name = config["model"]["name"]
    try:
        builder = MODEL_BUILDERS[model_name]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_BUILDERS))
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}. 사용 가능: {available}") from error
    model_config = deepcopy(config["model"])
    if class_names is not None and model_name in {
        "pair_specialist_classifier",
        "pipecomb_ensemble",
        "pipecomb_weighted_soft_voting",
    }:
        model_config["class_names"] = [str(value) for value in class_names]
    return builder(model_config, config["project"]["seed"])


def fit_model(
    model,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: pd.DataFrame | None = None,
    valid_y: np.ndarray | None = None,
) -> None:
    """설정된 경우 검증 데이터로 XGBoost 조기 종료를 적용합니다."""
    early_stopping_rounds = getattr(model, "early_stopping_rounds", None)
    if early_stopping_rounds is None:
        model.fit(train_x, train_y)
        return
    if valid_x is None or valid_y is None:
        raise ValueError("조기 종료를 사용하려면 검증 데이터가 필요합니다.")
    model.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        verbose=False,
    )


def used_tree_count(model) -> int | None:
    """트리 모델의 실제 반복 수를 반환하고 비트리 모델은 ``None``으로 둡니다."""
    try:
        return int(model.best_iteration) + 1
    except (AttributeError, TypeError, ValueError):
        pass

    try:
        parameters = model.get_params()
    except (AttributeError, TypeError, ValueError):
        return None

    for key in ("n_estimators", "iterations"):
        value = parameters.get(key)
        if value is not None:
            return int(value)

    return None
  
  

def evaluate_stratified_oof(
    config: dict,
    preprocessing_config: dict,
    features: pd.DataFrame,
    labels: pd.Series,
) -> dict:
    """파이프라인을 fold 안에서 다시 학습하는 Stratified OOF 평가입니다."""
    template_config, _ = split_preprocessing_config(preprocessing_config)
    template = create_preprocessing_pipeline(template_config)
    folds = int(getattr(template, "evaluation_folds", 1))
    if folds < 2:
        return {}

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=config["project"]["seed"],
    )
    oof_predictions = np.full(len(features), -1, dtype="int32")
    fold_scores: list[float] = []
    fold_train_scores: list[float] = []
    fold_tree_counts: list[int | None] = []
    fold_preprocessing_selections: list[dict] = []

    for fold, (train_index, valid_index) in enumerate(
        splitter.split(features, labels), start=1
    ):
        fold_train_x = features.iloc[train_index]
        fold_valid_x = features.iloc[valid_index]
        fold_train_y = labels.iloc[train_index]
        fold_valid_y = labels.iloc[valid_index]

        fold_config = deepcopy(config)
        fold_config["preprocessing"] = preprocessing_config
        fold_preprocessing_config, fold_preprocessing_cv = select_min_mutation_count(
            fold_config,
            fold_train_x,
            fold_train_y,
        )
        preprocessor = create_preprocessing_pipeline(fold_preprocessing_config)
        encoded_train_x = preprocessor.fit_transform(fold_train_x, fold_train_y)
        encoded_valid_x = preprocessor.transform(fold_valid_x)
        encoded_train_y = preprocessor.encode_labels(fold_train_y)
        encoded_valid_y = preprocessor.encode_labels(fold_valid_y)
        model = build_model(config, preprocessor.label_encoder.classes_)
        fit_model(
            model,
            encoded_train_x,
            encoded_train_y,
            encoded_valid_x,
            encoded_valid_y,
        )
        train_predictions = model.predict(encoded_train_x)
        valid_predictions = model.predict(encoded_valid_x).astype("int32")
        oof_predictions[valid_index] = valid_predictions
        train_score = float(
            f1_score(encoded_train_y, train_predictions, average="macro")
        )
        valid_score = float(
            f1_score(encoded_valid_y, valid_predictions, average="macro")
        )
        fold_train_scores.append(train_score)
        fold_scores.append(valid_score)
        tree_count = used_tree_count(model)
        if tree_count is not None:
            fold_tree_counts.append(tree_count)
        if fold_preprocessing_cv:
            fold_preprocessing_selections.append({
                "fold": fold,
                **fold_preprocessing_cv,
            })
        print(json.dumps({
            "oof_fold": fold,
            "train_macro_f1": train_score,
            "validation_macro_f1": valid_score,
        }, ensure_ascii=False))

    global_encoder = template.label_encoder.fit(labels)
    encoded_labels = global_encoder.transform(labels)
    oof_macro_f1 = float(
        f1_score(encoded_labels, oof_predictions, average="macro")
    )
    score_series = pd.Series(fold_scores, dtype="float64")
    result = {
        "folds": folds,
        "macro_f1": oof_macro_f1,
        "fold_mean_macro_f1": float(score_series.mean()),
        "fold_std_macro_f1": float(score_series.std(ddof=0)),
        "fold_scores": fold_scores,
        "fold_train_scores": fold_train_scores,
        "fold_tree_counts": fold_tree_counts,
    }
    if fold_preprocessing_selections:
        result["fold_preprocessing_selections"] = fold_preprocessing_selections

    return result


def split_preprocessing_config(config: dict) -> tuple[dict, dict]:
    """파이프라인 생성 설정과 교차검증 튜닝 설정을 분리합니다."""
    tuning_keys = (
        "min_mutation_count_cv",
        "min_functional_mutation_count_cv",
        "min_redundancy_support_cv",
    )
    pipeline_config = {
        key: value
        for key, value in config.items()
        if key not in tuning_keys
    }
    tuning_config = next(
        (config[key] for key in tuning_keys if key in config),
        {},
    )
    return pipeline_config, tuning_config


def select_min_mutation_count(
    config: dict,
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[dict, dict]:
    """Stratified K-fold Macro-F1으로 최소 변이 지지도를 선택합니다."""
    pipeline_config, tuning_config = split_preprocessing_config(
        config["preprocessing"]
    )
    pipeline_name = pipeline_config.get("name")
    parameter_by_pipeline = {
        "em_v3": "min_mutation_count",
        "em_v19": "min_functional_mutation_count",
        "em_v20": "min_functional_mutation_count",
        "em_G01": "min_active_count",
        "em_H01": "min_mutation_count",
        "em_H02": "min_mutation_count",
        "em_H03": "min_mutation_count",
        "em_H04": "min_mutation_count",
        "em_H05": "min_mutation_count",
        "em_H06": "min_mutation_count",
        "em_H07": "min_mutation_count",
    }
    parameter_name = parameter_by_pipeline.get(pipeline_name)
    if parameter_name is None or not tuning_config.get("enabled", False):
        return pipeline_config, {}

    candidates = sorted(set(tuning_config.get("candidates", [1, 2, 3, 5, 10, 20])))
    if not candidates or any(
        isinstance(candidate, bool)
        or not isinstance(candidate, int)
        or candidate < 1
        for candidate in candidates
    ):
        raise ValueError("최소 변이 지지도 후보는 1 이상의 정수 목록이어야 합니다.")

    folds = tuning_config.get("folds", 5)
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("최소 변이 지지도 CV folds는 2 이상의 정수여야 합니다.")

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=config["project"]["seed"],
    )
    candidate_results: list[dict] = []

    for candidate in candidates:
        fold_scores: list[float] = []
        for train_index, valid_index in splitter.split(features, labels):
            fold_train_x = features.iloc[train_index]
            fold_valid_x = features.iloc[valid_index]
            fold_train_y = labels.iloc[train_index]
            fold_valid_y = labels.iloc[valid_index]

            candidate_config = {
                **pipeline_config,
                parameter_name: candidate,
            }
            preprocessor = create_preprocessing_pipeline(candidate_config)
            encoded_train_x = preprocessor.fit_transform(fold_train_x, fold_train_y)
            encoded_valid_x = preprocessor.transform(fold_valid_x)
            encoded_train_y = preprocessor.encode_labels(fold_train_y)
            encoded_valid_y = preprocessor.encode_labels(fold_valid_y)

            model = build_model(config, preprocessor.label_encoder.classes_)
            fit_model(
                model,
                encoded_train_x,
                encoded_train_y,
                encoded_valid_x,
                encoded_valid_y,
            )
            predictions = model.predict(encoded_valid_x)
            fold_scores.append(
                float(f1_score(encoded_valid_y, predictions, average="macro"))
            )

        score_series = pd.Series(fold_scores, dtype="float64")
        candidate_results.append({
            parameter_name: candidate,
            "mean_macro_f1": float(score_series.mean()),
            "std_macro_f1": float(score_series.std(ddof=0)),
            "fold_scores": fold_scores,
        })

    best_result = max(
        candidate_results,
        key=lambda result: (
            result["mean_macro_f1"],
            -result["std_macro_f1"],
            result[parameter_name],
        ),
    )
    selected_count = best_result[parameter_name]
    selected_pipeline_config = {
        **pipeline_config,
        parameter_name: selected_count,
    }
    cv_result = {
        f"selected_{parameter_name}": selected_count,
        "parameter": parameter_name,
        "folds": folds,
        "candidates": candidate_results,
    }
    print(json.dumps({"preprocessing_cv": cv_result}, ensure_ascii=False))
    return selected_pipeline_config, cv_result


def write_experiment_report(
    path: Path,
    config_path: Path,
    config: dict,
    train_rows: int,
    feature_count: int,
    metrics: dict,
    submission_path: Path,
    artifact_path: Path,
) -> None:
    """학습 결과를 실험 이름의 Markdown 보고서로 저장합니다."""
    model_config = config["model"]
    path.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# {config['project']['experiment_name']}

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | {datetime.now().astimezone().isoformat(timespec='seconds')} |
| 모델 | {model_config['name']} |
| 전처리 파이프라인 | {config['preprocessing']['name']} |
| 선택된 최소 변이 횟수 | {config['preprocessing'].get('min_mutation_count', config['preprocessing'].get('min_functional_mutation_count', '-'))} |
| 학습 데이터 행 수 | {train_rows} |
| 피처 수 | {feature_count} |
| 최종 Macro F1 | {metrics['final_macro_f1']:.6f} |
| 80% 학습 Macro F1 | {metrics['train_macro_f1_80pct']:.6f} |
| 20% 검증 Macro F1 | {metrics['validation_macro_f1']:.6f} |
| 과적합 격차 | {metrics['overfit_gap']:.6f} |
| 과적합 여부 | {metrics['is_overfitting']} |
| 설정 파일 | `{config_path}` |
| 제출 파일 | `{submission_path}` |
| 모델 아티팩트 | `{artifact_path}` |

## 하이퍼파라미터

```yaml
{yaml.safe_dump(model_config, allow_unicode=True, sort_keys=False).rstrip()}
```

## 전처리 설정

```yaml
{yaml.safe_dump(config['preprocessing'], allow_unicode=True, sort_keys=False).rstrip()}
```
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    experiment_name = config["project"]["experiment_name"]
    model_name = config["model"]["name"]
    data_config = config["data"]
    raw_dir = Path(data_config["raw_dir"])
    output_dir = Path(data_config["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    train = pd.read_csv(raw_dir / data_config["train_file"])
    test = pd.read_csv(raw_dir / data_config["test_file"])
    target = data_config["target_column"]
    identifier = data_config["id_column"]

    features = train.drop(columns=[target, identifier])
    labels = train[target]
    train_features, valid_features, train_labels, valid_labels = train_test_split(
        features, labels,
        test_size=config["training"]["validation_fraction"],
        random_state=config["project"]["seed"],
        stratify=labels if config["training"]["stratify"] else None,
    )
    preprocessing_config, preprocessing_cv = select_min_mutation_count(
        config,
        train_features,
        train_labels,
    )
    effective_config = deepcopy(config)
    effective_config["preprocessing"] = preprocessing_config
    if preprocessing_cv:
        effective_config["preprocessing_cv"] = preprocessing_cv

    oof_evaluation = evaluate_stratified_oof(
        config,
        config["preprocessing"],
        features,
        labels,
    )

    validation_preprocessor = create_preprocessing_pipeline(preprocessing_config)
    train_x = validation_preprocessor.fit_transform(train_features, train_labels)
    valid_x = validation_preprocessor.transform(valid_features)
    train_y = validation_preprocessor.encode_labels(train_labels)
    valid_y = validation_preprocessor.encode_labels(valid_labels)
    validation_model = build_model(
        config, validation_preprocessor.label_encoder.classes_
    )
    fit_model(validation_model, train_x, train_y, valid_x, valid_y)
    train_predictions = validation_model.predict(train_x)
    validation_predictions = validation_model.predict(valid_x)
    train_f1 = float(f1_score(train_y, train_predictions, average="macro"))
    validation_f1 = float(
        f1_score(valid_y, validation_predictions, average="macro")
    )
    overfit_gap = train_f1 - validation_f1
    overfit_threshold = float(
        config["training"].get("overfit_threshold", 0.1)
    )
    is_overfitting = overfit_gap > overfit_threshold

    preprocessor = create_preprocessing_pipeline(preprocessing_config)
    encoded_features = preprocessor.fit_transform(features, labels)
    final_model_config = deepcopy(config)
    if final_model_config["model"].pop("early_stopping_rounds", None) is not None:
        fold_tree_counts = [
            count
            for count in oof_evaluation.get("fold_tree_counts", [])
            if count is not None
        ]
        if fold_tree_counts:
            final_model_config["model"]["n_estimators"] = int(
                np.median(fold_tree_counts)
            )
    final_model = build_model(
        final_model_config, preprocessor.label_encoder.classes_
    )
    fit_model(final_model, encoded_features, preprocessor.encode_labels(labels))
    encoded_test = preprocessor.transform(test.drop(columns=[identifier]))
    encoded_predictions = final_model.predict(encoded_test)
    predictions = preprocessor.decode_labels(encoded_predictions)

    submission = pd.read_csv(raw_dir / data_config["submission_file"])
    submission[target] = predictions
    submission_path = output_dir / f"{experiment_name}_{model_name}_submission.csv"
    submission.to_csv(submission_path, index=False, encoding="utf-8-sig")

    artifact_path = Path("models") / f"{experiment_name}.pkl"
    with artifact_path.open("wb") as file:
        pickle.dump(
            {
                "model": final_model,
                "preprocessor": preprocessor,
                "config": effective_config,
            },
            file,
        )

    final_f1 = float(oof_evaluation.get("macro_f1", validation_f1))
    metrics = {
        "final_macro_f1": final_f1,
        "train_macro_f1_80pct": train_f1,
        "validation_macro_f1": validation_f1,
        "overfit_gap": overfit_gap,
        "overfit_threshold": overfit_threshold,
        "is_overfitting": is_overfitting,
        "submission": str(submission_path),
    }
    if oof_evaluation:
        metrics["oof_evaluation"] = oof_evaluation
    if preprocessing_cv:
        metrics["preprocessing_cv"] = preprocessing_cv
    (output_dir / f"{experiment_name}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_experiment_report(
        Path("experiments") / f"{experiment_name}_{model_name}.md",
        args.config,
        effective_config,
        train_rows=len(train),
        feature_count=preprocessor.summary()["remaining_features"],
        metrics=metrics,
        submission_path=submission_path,
        artifact_path=artifact_path,
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
