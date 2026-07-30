"""설정 파일에 지정된 모델을 학습하고 제출 파일을 생성합니다."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from src.models import MODEL_BUILDERS
from src.preprocess import TabularPreprocessor


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
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/xgboost_baseline.yaml"))
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
    preprocessor = TabularPreprocessor().fit(features, labels)
    encoded_features = preprocessor.transform(features)

    train_x, valid_x, train_y, valid_y = train_test_split(
        encoded_features,
        preprocessor.encode_labels(labels),
        test_size=config["training"]["validation_fraction"],
        random_state=config["project"]["seed"],
        stratify=preprocessor.encode_labels(labels) if config["training"]["stratify"] else None,
    )
    validation_model = build_model(config)
    validation_model.fit(train_x, train_y)
    validation_predictions = validation_model.predict(valid_x)
    validation_f1 = f1_score(valid_y, validation_predictions, average="macro")

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
        pickle.dump({"model": final_model, "preprocessor": preprocessor, "config": config}, file)

    metrics = {"validation_macro_f1": validation_f1, "submission": str(submission_path)}
    (output_dir / f"{experiment_name}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_experiment_report(
        Path("experiments") / f"{experiment_name}.md",
        args.config,
        config,
        train_rows=len(train),
        feature_count=encoded_features.shape[1],
        validation_f1=validation_f1,
        submission_path=submission_path,
        artifact_path=artifact_path,
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
