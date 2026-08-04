"""JH 전처리·모델 공용 StratifiedGroupKFold 실행기."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from src.models import MODEL_BUILDERS
from src.pipelines.pipeline_jh_v04 import (
    build_f0_matrix,
    build_profile_groups,
    make_class_burden_strata,
    parse_wide_mutations,
)
from src.pipelines.preprocessing_registry import create_preprocessing_pipeline


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def model_scores(model, matrix: sparse.spmatrix) -> tuple[np.ndarray, str]:
    """모델의 클래스별 출력과 출력 종류를 반환합니다."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix), dtype=np.float64), "probability"
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(matrix), dtype=np.float64)
        if decision.ndim != 2:
            raise ValueError("다중 클래스 decision score는 2차원이어야 합니다.")
        return decision, "decision"
    raise TypeError("모델에 predict_proba 또는 decision_function이 없습니다.")


def dataframe_to_markdown(
    frame: pd.DataFrame,
    float_digits: int = 6,
) -> str:
    """추가 패키지 없이 DataFrame을 Markdown 표로 변환합니다."""
    columns = [str(column) for column in frame.columns]

    def format_value(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        return str(value).replace("|", "\\|")

    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    rows.extend(
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def write_experiment_report(
    path: Path,
    config_path: Path,
    config: dict,
    started_at: datetime,
    finished_at: datetime,
    train_rows: int,
    fold_metrics: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    submission_path: Path,
    summary_path: Path,
) -> None:
    """SGKF 실행 결과를 팀 공통 Markdown 보고서 형식으로 저장합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model_config = config["model"]
    preprocessing_config = config["preprocessing"]
    mean_score = float(seed_metrics["oof_macro_f1"].mean())
    std_score = float(seed_metrics["oof_macro_f1"].std(ddof=1))
    feature_min = int(fold_metrics["feature_count"].min())
    feature_max = int(fold_metrics["feature_count"].max())
    feature_mean = float(fold_metrics["feature_count"].mean())
    elapsed = finished_at - started_at

    seed_table = dataframe_to_markdown(seed_metrics)
    fold_columns = [
        "seed", "fold", "feature_count", "macro_f1",
        "converged", "elapsed_seconds",
    ]
    fold_table = dataframe_to_markdown(fold_metrics[fold_columns])

    report = f"""# {config['project']['experiment_name']}

| 항목 | 결과 |
| --- | --- |
| 실행 시작 | {started_at.isoformat(timespec='seconds')} |
| 실행 종료 | {finished_at.isoformat(timespec='seconds')} |
| 총 실행 시간 | {elapsed} |
| 모델 | {model_config['name']} |
| 전처리 파이프라인 | {preprocessing_config['name']} |
| 검증 | StratifiedGroupKFold {config['validation']['n_splits']}-Fold × {len(config['validation']['seeds'])} seeds |
| 학습 데이터 행 수 | {train_rows} |
| Fold 피처 수 | 평균 {feature_mean:.1f}, 범위 {feature_min}~{feature_max} |
| OOF Macro F1 | {mean_score:.6f} ± {std_score:.6f} |
| 설정 파일 | `{config_path}` |
| 제출 파일 | `{submission_path}` |
| 요약 JSON | `{summary_path}` |

## Seed별 OOF

{seed_table}

## Fold별 결과

{fold_table}

## 하이퍼파라미터

```yaml
{yaml.safe_dump(model_config, allow_unicode=True, sort_keys=False).rstrip()}
```

## 전처리 설정

```yaml
{yaml.safe_dump(preprocessing_config, allow_unicode=True, sort_keys=False).rstrip()}
```
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    started_at = datetime.now().astimezone()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/test_002.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    validation_config = config["validation"]
    feature_config = config["preprocessing"]
    pipeline_name = str(feature_config.get("name", ""))
    if not pipeline_name.startswith("jh_v"):
        raise ValueError(
            "train_jh_sgkf.py는 jh_v 전처리만 지원합니다: "
            f"{pipeline_name!r}"
        )
    model_name = str(config["model"].get("name", ""))
    try:
        model_builder = MODEL_BUILDERS[model_name]
    except KeyError as error:
        raise ValueError(
            f"지원하지 않는 모델입니다: {model_name!r}. "
            f"사용 가능: {', '.join(sorted(MODEL_BUILDERS))}"
        ) from error
    raw_dir = Path(data_config["raw_dir"])
    output_dir = Path(data_config["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(raw_dir / data_config["train_file"])
    test = pd.read_csv(raw_dir / data_config["test_file"])
    target_column = data_config["target_column"]
    id_column = data_config["id_column"]
    gene_columns = [column for column in train.columns if column not in {id_column, target_column}]
    if gene_columns != [column for column in test.columns if column != id_column]:
        raise ValueError("Train/Test 유전자 열 또는 순서가 다릅니다.")

    train_features = train[gene_columns]
    test_features = test[gene_columns]
    labels = train[target_column]
    label_encoder = LabelEncoder().fit(labels)
    y = label_encoder.transform(labels)
    class_names = label_encoder.classes_
    n_classes = len(class_names)

    print(f"원본 변이 파싱 중... pipeline={pipeline_name}")
    train_events = parse_wide_mutations(train_features)
    print(f"Train events={len(train_events):,}")

    train_f0_all = build_f0_matrix(train_features)

    mutation_burden = np.asarray(train_f0_all.getnnz(axis=1)).ravel()
    profile_groups = build_profile_groups(train_events, len(train))
    n_splits = int(validation_config["n_splits"])
    seeds = [int(seed) for seed in validation_config["seeds"]]
    strata, burden_bins = make_class_burden_strata(labels, mutation_burden, n_splits)

    fixed_folds: dict[int, np.ndarray] | None = None
    fixed_split_value = validation_config.get("fixed_split_file")
    if fixed_split_value:
        fixed_split_path = Path(fixed_split_value)
        if not fixed_split_path.exists():
            raise FileNotFoundError(
                f"고정 split 파일이 없습니다: {fixed_split_path}"
            )
        fixed_frame = pd.read_csv(fixed_split_path)
        if not np.array_equal(
            fixed_frame[id_column].astype(str).to_numpy(),
            train[id_column].astype(str).to_numpy(),
        ):
            raise ValueError("Train과 고정 split의 ID 순서가 다릅니다.")
        fixed_folds = {}
        for seed in seeds:
            column = f"fold_seed_{seed}"
            if column not in fixed_frame:
                raise ValueError(f"고정 split에 {column} 열이 없습니다.")
            fold_ids = fixed_frame[column].to_numpy(dtype=np.int8)
            if set(np.unique(fold_ids)) != set(range(n_splits)):
                raise ValueError(f"seed={seed} fold 값이 잘못되었습니다.")
            fixed_folds[seed] = fold_ids

    fold_rows: list[dict] = []
    seed_rows: list[dict] = []
    class_frames: list[pd.DataFrame] = []
    burden_rows: list[dict] = []
    oof_rows: list[pd.DataFrame] = []
    test_score_sum = np.zeros((len(test), n_classes), dtype=np.float64)
    oof_scores_by_seed: list[np.ndarray] = []
    score_kind: str | None = None
    model_count = 0

    for seed in seeds:
        if fixed_folds is None:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            split_iterator = splitter.split(
                np.zeros(len(train)), strata, profile_groups,
            )
        else:
            split_iterator = (
                (
                    np.flatnonzero(fixed_folds[seed] != fold),
                    np.flatnonzero(fixed_folds[seed] == fold),
                )
                for fold in range(n_splits)
            )
        oof_score = np.zeros((len(train), n_classes), dtype=np.float64)
        oof_seen = np.zeros(len(train), dtype=bool)

        for fold, (train_index, valid_index) in enumerate(split_iterator):
            started = time.time()
            fold_feature_config = dict(feature_config)
            if "svd_random_state" in fold_feature_config:
                fold_feature_config["svd_random_state"] = seed
            preprocessor = create_preprocessing_pipeline(fold_feature_config)
            train_matrix = preprocessor.fit_transform(
                train_features.iloc[train_index],
                labels.iloc[train_index],
            )
            valid_matrix = preprocessor.transform(train_features.iloc[valid_index])
            test_matrix = preprocessor.transform(test_features)
            feature_summary = preprocessor.summary()

            model_seed = (
                seed + fold
                if bool(config["model"].get("seed_plus_fold", True))
                else seed
            )
            model = model_builder(config["model"], model_seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                early_stopping_rounds = int(
                    config["model"].get("early_stopping_rounds", 0)
                )
                if model_name == "lightgbm" and early_stopping_rounds > 0:
                    from lightgbm import early_stopping, log_evaluation

                    model.fit(
                        train_matrix,
                        y[train_index],
                        eval_set=[(valid_matrix, y[valid_index])],
                        eval_metric=config["model"].get(
                            "eval_metric", "multi_logloss"
                        ),
                        callbacks=[
                            early_stopping(early_stopping_rounds, verbose=False),
                            log_evaluation(0),
                        ],
                    )
                else:
                    model.fit(train_matrix, y[train_index])
            converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)

            valid_score, current_score_kind = model_scores(model, valid_matrix)
            if score_kind is None:
                score_kind = current_score_kind
            elif score_kind != current_score_kind:
                raise RuntimeError("폴드마다 모델 출력 종류가 다릅니다.")
            valid_prediction = model.classes_[valid_score.argmax(axis=1)]
            fold_score = f1_score(
                y[valid_index], valid_prediction,
                labels=np.arange(n_classes), average="macro", zero_division=0,
            )
            oof_score[valid_index] = valid_score
            oof_seen[valid_index] = True
            test_score, test_score_kind = model_scores(model, test_matrix)
            if test_score_kind != score_kind:
                raise RuntimeError("Validation/Test 모델 출력 종류가 다릅니다.")
            test_score_sum += test_score
            model_count += 1
            fold_row = {
                "seed": seed,
                "fold": fold,
                "pipeline": pipeline_name,
                "train_rows": len(train_index),
                "valid_rows": len(valid_index),
                "feature_count": train_matrix.shape[1],
                "macro_f1": float(fold_score),
                "converged": converged,
                "max_n_iter": int(np.max(np.atleast_1d(getattr(
                    model,
                    "n_iter_",
                    getattr(model, "best_iteration_", 0),
                )))),
                "elapsed_seconds": time.time() - started,
            }
            fold_row.update(feature_summary)
            fold_rows.append(fold_row)
            print(
                f"seed={seed} fold={fold} Macro F1={fold_score:.6f} "
                f"features={train_matrix.shape[1]} converged={converged}"
            )

        if not oof_seen.all():
            raise RuntimeError(f"seed={seed}에서 OOF가 채워지지 않은 샘플이 있습니다.")
        oof_prediction = oof_score.argmax(axis=1)
        oof_scores_by_seed.append(oof_score.copy())
        seed_score = f1_score(
            y, oof_prediction,
            labels=np.arange(n_classes), average="macro", zero_division=0,
        )
        seed_rows.append({"seed": seed, "oof_macro_f1": float(seed_score)})
        report = pd.DataFrame(classification_report(
            y, oof_prediction,
            labels=np.arange(n_classes), target_names=class_names,
            output_dict=True, zero_division=0,
        )).T.loc[class_names].reset_index(names=target_column)
        report.insert(0, "seed", seed)
        class_frames.append(report)
        for burden_bin in validation_config["burden_bins"]["labels"]:
            mask = burden_bins.astype(str).to_numpy() == burden_bin
            burden_score = (
                float(f1_score(
                    y[mask], oof_prediction[mask],
                    labels=np.arange(n_classes), average="macro", zero_division=0,
                ))
                if mask.any()
                else np.nan
            )
            burden_rows.append({
                "seed": seed,
                "burden_bin": burden_bin,
                "sample_count": int(mask.sum()),
                "macro_f1": burden_score,
            })
        oof_rows.append(pd.DataFrame({
            id_column: train[id_column],
            target_column: labels,
            "seed": seed,
            "prediction": label_encoder.inverse_transform(oof_prediction),
        }))
        print(f"seed={seed} OOF Macro F1={seed_score:.6f}")

    fold_metrics = pd.DataFrame(fold_rows)
    seed_metrics = pd.DataFrame(seed_rows)
    class_metrics = pd.concat(class_frames, ignore_index=True)
    burden_metrics = pd.DataFrame(burden_rows)
    oof_predictions = pd.concat(oof_rows, ignore_index=True)
    mean_score = float(seed_metrics["oof_macro_f1"].mean())
    std_score = float(seed_metrics["oof_macro_f1"].std(ddof=1))

    test_mean_score = test_score_sum / model_count
    submission = pd.read_csv(raw_dir / data_config["submission_file"])
    submission[target_column] = label_encoder.inverse_transform(test_mean_score.argmax(axis=1))

    experiment_name = config["project"]["experiment_name"]
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    seed_metrics.to_csv(output_dir / "seed_oof_metrics.csv", index=False, encoding="utf-8-sig")
    class_metrics.to_csv(output_dir / "class_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    burden_metrics.to_csv(output_dir / "burden_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    oof_predictions.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    score_prefix = "probability" if score_kind == "probability" else "decision"
    np.save(output_dir / f"oof_{score_prefix}_by_seed.npy", np.stack(oof_scores_by_seed))
    np.save(output_dir / f"test_{score_prefix}_mean.npy", test_mean_score)
    pd.DataFrame({
        "class_index": np.arange(n_classes),
        target_column: class_names,
    }).to_csv(output_dir / "class_order.csv", index=False, encoding="utf-8-sig")
    submission_path = output_dir / f"submission_{experiment_name}.csv"
    submission.to_csv(submission_path, index=False, encoding="utf-8-sig")

    summary = {
        "experiment": experiment_name,
        "preprocessing": pipeline_name,
        "model": model_name,
        "score_kind": score_kind,
        "cv_strategy": "StratifiedGroupKFold",
        "n_splits": n_splits,
        "seeds": seeds,
        "model_count": model_count,
        "oof_macro_f1_mean": mean_score,
        "oof_macro_f1_std": std_score,
        "submission": str(submission_path),
        "feature_config": feature_config,
    }
    report_path = Path("experiments") / f"{experiment_name}.md"
    summary["experiment_report"] = str(report_path)
    summary_path = output_dir / f"{pipeline_name}_sgkf_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    finished_at = datetime.now().astimezone()
    write_experiment_report(
        path=report_path,
        config_path=args.config,
        config=config,
        started_at=started_at,
        finished_at=finished_at,
        train_rows=len(train),
        fold_metrics=fold_metrics,
        seed_metrics=seed_metrics,
        submission_path=submission_path,
        summary_path=summary_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
