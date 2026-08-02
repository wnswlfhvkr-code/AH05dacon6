"""설정 파일에 지정된 모델을 학습하고 제출 파일을 생성합니다."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.models import MODEL_BUILDERS
from src.pipelines.preprocessing_registry import create_preprocessing_pipeline


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_model(config: dict):
    """설정의 모델 이름에 맞는 생성 함수를 호출합니다."""
    model_name = config["model"]["name"]
    try:
        builder = MODEL_BUILDERS[model_name]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_BUILDERS))
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}. 사용 가능: {available}") from error
    return builder(config["model"], config["project"]["seed"])


def split_preprocessing_config(config: dict) -> tuple[dict, dict]:
    """파이프라인 생성 설정과 교차검증 튜닝 설정을 분리합니다."""
    pipeline_config = {
        key: value
        for key, value in config.items()
        if key != "min_mutation_count_cv"
    }
    tuning_config = config.get("min_mutation_count_cv", {})
    return pipeline_config, tuning_config


def select_min_mutation_count(
    config: dict,
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[dict, dict]:
    """Stratified K-fold Macro-F1으로 최소 변이 기준을 선택합니다."""
    pipeline_config, tuning_config = split_preprocessing_config(
        config["preprocessing"]
    )
    if pipeline_config.get("name") != "em_v3" or not tuning_config.get("enabled", False):
        return pipeline_config, {}

    candidates = sorted(set(tuning_config.get("candidates", [1, 2, 3, 5, 10, 20])))
    if not candidates or any(
        isinstance(candidate, bool)
        or not isinstance(candidate, int)
        or candidate < 1
        for candidate in candidates
    ):
        raise ValueError("min_mutation_count_cv.candidates는 1 이상의 정수 목록이어야 합니다.")

    folds = tuning_config.get("folds", 5)
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("min_mutation_count_cv.folds는 2 이상의 정수여야 합니다.")

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
                "min_mutation_count": candidate,
            }
            preprocessor = create_preprocessing_pipeline(candidate_config)
            encoded_train_x = preprocessor.fit_transform(fold_train_x, fold_train_y)
            encoded_valid_x = preprocessor.transform(fold_valid_x)
            encoded_train_y = preprocessor.encode_labels(fold_train_y)
            encoded_valid_y = preprocessor.encode_labels(fold_valid_y)

            model = build_model(config)
            model.fit(encoded_train_x, encoded_train_y)
            predictions = model.predict(encoded_valid_x)
            fold_scores.append(
                float(f1_score(encoded_valid_y, predictions, average="macro"))
            )

        score_series = pd.Series(fold_scores, dtype="float64")
        candidate_results.append({
            "min_mutation_count": candidate,
            "mean_macro_f1": float(score_series.mean()),
            "std_macro_f1": float(score_series.std(ddof=0)),
            "fold_scores": fold_scores,
        })

    best_result = max(
        candidate_results,
        key=lambda result: (
            result["mean_macro_f1"],
            -result["std_macro_f1"],
            result["min_mutation_count"],
        ),
    )
    selected_count = best_result["min_mutation_count"]
    selected_pipeline_config = {
        **pipeline_config,
        "min_mutation_count": selected_count,
    }
    cv_result = {
        "selected_min_mutation_count": selected_count,
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
    validation_f1: float,
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
| 선택된 최소 변이 횟수 | {config['preprocessing'].get('min_mutation_count', '-')} |
| 학습 데이터 행 수 | {train_rows} |
| 피처 수 | {feature_count} |
| 검증 Macro F1 | {validation_f1:.6f} |
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

    validation_preprocessor = create_preprocessing_pipeline(preprocessing_config).fit(train_features, train_labels)
    train_x = validation_preprocessor.transform(train_features)
    valid_x = validation_preprocessor.transform(valid_features)
    train_y = validation_preprocessor.encode_labels(train_labels)
    valid_y = validation_preprocessor.encode_labels(valid_labels)
    validation_model = build_model(config)
    validation_model.fit(train_x, train_y)
    validation_predictions = validation_model.predict(valid_x)
    validation_f1 = f1_score(valid_y, validation_predictions, average="macro")

    preprocessor = create_preprocessing_pipeline(preprocessing_config)
    encoded_features = preprocessor.fit_transform(features, labels)
    final_model = build_model(config)
    final_model.fit(encoded_features, preprocessor.encode_labels(labels))
    encoded_test = preprocessor.transform(test.drop(columns=[identifier]))
    predictions = preprocessor.decode_labels(final_model.predict(encoded_test))

    submission = pd.read_csv(raw_dir / data_config["submission_file"])
    submission[target] = predictions
    submission_path = output_dir / f"{experiment_name}_submission.csv"
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

    metrics = {
        "validation_macro_f1": validation_f1,
        "submission": str(submission_path),
    }
    if preprocessing_cv:
        metrics["preprocessing_cv"] = preprocessing_cv
    (output_dir / f"{experiment_name}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_experiment_report(
        Path("experiments") / f"{experiment_name}.md",
        args.config,
        effective_config,
        train_rows=len(train),
        feature_count=preprocessor.summary()["remaining_features"],
        validation_f1=validation_f1,
        submission_path=submission_path,
        artifact_path=artifact_path,
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
