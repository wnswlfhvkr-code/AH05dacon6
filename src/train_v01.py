"""v01: F0+F1+F3, SGKF 3-fold×3-seed Logistic Regression 실행기."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.models.logistic_regression_model import create_model
from src.pipelines.pipeline_jh_v01 import (
    build_f0_matrix,
    build_f1_matrix,
    build_f3_matrix,
    build_profile_groups,
    make_class_burden_strata,
    parse_wide_mutations,
)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/test_002.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    validation_config = config["validation"]
    feature_config = config["preprocessing"]
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

    print("원본 변이 파싱 중...")
    train_events = parse_wide_mutations(train_features)
    test_events = parse_wide_mutations(test_features)
    print(f"Train events={len(train_events):,}, Test events={len(test_events):,}")

    train_f0_all = build_f0_matrix(train_features)
    test_f0_all = build_f0_matrix(test_features)
    train_f1_raw, f1_names = build_f1_matrix(train_events, len(train))
    test_f1_raw, _ = build_f1_matrix(test_events, len(test))
    train_f3_all, f3_names = build_f3_matrix(train_events, len(train))
    test_f3_all, _ = build_f3_matrix(test_events, len(test))

    mutation_burden = np.asarray(train_f0_all.getnnz(axis=1)).ravel()
    profile_groups = build_profile_groups(train_events, len(train))
    n_splits = int(validation_config["n_splits"])
    seeds = [int(seed) for seed in validation_config["seeds"]]
    strata, burden_bins = make_class_burden_strata(labels, mutation_burden, n_splits)

    fold_rows: list[dict] = []
    seed_rows: list[dict] = []
    class_frames: list[pd.DataFrame] = []
    burden_rows: list[dict] = []
    oof_rows: list[pd.DataFrame] = []
    test_probability_sum = np.zeros((len(test), n_classes), dtype=np.float64)
    model_count = 0

    for seed in seeds:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        oof_probability = np.zeros((len(train), n_classes), dtype=np.float64)
        oof_seen = np.zeros(len(train), dtype=bool)

        for fold, (train_index, valid_index) in enumerate(splitter.split(
            np.zeros(len(train)), strata, profile_groups,
        )):
            started = time.time()
            f0_mask = np.asarray(train_f0_all[train_index].getnnz(axis=0)).ravel() > 0
            f3_mask = np.asarray(train_f3_all[train_index].getnnz(axis=0)).ravel() > 0
            f1_scaler = StandardScaler().fit(train_f1_raw[train_index])

            train_matrix = sparse.hstack([
                train_f0_all[train_index][:, f0_mask],
                sparse.csr_matrix(f1_scaler.transform(train_f1_raw[train_index]).astype(np.float32)),
                train_f3_all[train_index][:, f3_mask],
            ], format="csr", dtype=np.float32)
            valid_matrix = sparse.hstack([
                train_f0_all[valid_index][:, f0_mask],
                sparse.csr_matrix(f1_scaler.transform(train_f1_raw[valid_index]).astype(np.float32)),
                train_f3_all[valid_index][:, f3_mask],
            ], format="csr", dtype=np.float32)
            test_matrix = sparse.hstack([
                test_f0_all[:, f0_mask],
                sparse.csr_matrix(f1_scaler.transform(test_f1_raw).astype(np.float32)),
                test_f3_all[:, f3_mask],
            ], format="csr", dtype=np.float32)

            model = create_model(config["model"], seed + fold)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(train_matrix, y[train_index])
            converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)

            valid_probability = model.predict_proba(valid_matrix)
            valid_prediction = model.classes_[valid_probability.argmax(axis=1)]
            fold_score = f1_score(
                y[valid_index], valid_prediction,
                labels=np.arange(n_classes), average="macro", zero_division=0,
            )
            oof_probability[valid_index] = valid_probability
            oof_seen[valid_index] = True
            test_probability_sum += model.predict_proba(test_matrix)
            model_count += 1
            fold_rows.append({
                "seed": seed,
                "fold": fold,
                "train_rows": len(train_index),
                "valid_rows": len(valid_index),
                "feature_count": train_matrix.shape[1],
                "f0_feature_count": int(f0_mask.sum()),
                "f1_feature_count": len(f1_names),
                "f3_feature_count": int(f3_mask.sum()),
                "macro_f1": float(fold_score),
                "converged": converged,
                "max_n_iter": int(np.max(model.n_iter_)),
                "elapsed_seconds": time.time() - started,
            })
            print(
                f"seed={seed} fold={fold} Macro F1={fold_score:.6f} "
                f"features={train_matrix.shape[1]} converged={converged}"
            )

        if not oof_seen.all():
            raise RuntimeError(f"seed={seed}에서 OOF가 채워지지 않은 샘플이 있습니다.")
        oof_prediction = oof_probability.argmax(axis=1)
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

    test_probability = test_probability_sum / model_count
    submission = pd.read_csv(raw_dir / data_config["submission_file"])
    submission[target_column] = label_encoder.inverse_transform(test_probability.argmax(axis=1))

    experiment_name = config["project"]["experiment_name"]
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    seed_metrics.to_csv(output_dir / "seed_oof_metrics.csv", index=False, encoding="utf-8-sig")
    class_metrics.to_csv(output_dir / "class_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    burden_metrics.to_csv(output_dir / "burden_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    oof_predictions.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    submission_path = output_dir / f"submission_{experiment_name}.csv"
    submission.to_csv(submission_path, index=False, encoding="utf-8-sig")

    summary = {
        "experiment": experiment_name,
        "feature": "F0 + F1 + F3",
        "excluded_feature": "F2 gene-consequence",
        "model": "Balanced Logistic Regression",
        "cv_strategy": "StratifiedGroupKFold",
        "n_splits": n_splits,
        "seeds": seeds,
        "model_count": model_count,
        "oof_macro_f1_mean": mean_score,
        "oof_macro_f1_std": std_score,
        "submission": str(submission_path),
        "feature_config": feature_config,
    }
    (output_dir / "v01_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
