"""Run the full mutation preprocessing audit and repeated-CV ablation."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing_ablation import (  # noqa: E402
    collect_preprocessing_audit,
    count_unsplit_other,
    build_mutation_documents,
    build_structural_features,
    create_figures,
    prepare_figure_data,
    run_repeated_cv_ablation,
    summarize_cv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/preprocessing_ablation.yaml",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the configured directory containing train.csv and test.csv.",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_dir = resolve_path(args.data_dir or config["data_dir"])
    output_dir = resolve_path(config["output_dir"])
    figure_dir = resolve_path(config["figure_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"train.csv and test.csv are required under {data_dir}"
        )

    print(f"[LOAD] {train_path}", flush=True)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    gene_columns = [
        column for column in train.columns if column not in {"ID", "SUBCLASS"}
    ]
    labels = train["SUBCLASS"].astype(str)

    print("[AUDIT] parsing all train/test mutation events", flush=True)
    audit, examples, train_long, test_long = collect_preprocessing_audit(
        train,
        test,
    )
    audit["train"]["unsplit_other_cells"] = count_unsplit_other(
        train,
        gene_columns,
    )
    audit["test"]["unsplit_other_cells"] = count_unsplit_other(
        test,
        gene_columns,
    )
    write_json(audit, output_dir / "preprocessing_audit.json")
    examples.to_csv(
        output_dir / "multi_event_examples.csv",
        index=False,
        encoding="utf-8-sig",
    )
    event_share, burden = prepare_figure_data(audit, train_long, test_long)
    del train_long, test_long
    gc.collect()

    print("[FEATURES] building raw and split mutation documents", flush=True)
    raw_documents = build_mutation_documents(
        train,
        gene_columns,
        split_events=False,
    )
    split_documents = build_mutation_documents(
        train,
        gene_columns,
        split_events=True,
    )
    structural = build_structural_features(train, gene_columns)
    structural.describe().T.to_csv(
        output_dir / "structural_feature_summary.csv",
        encoding="utf-8-sig",
    )

    print("[CV] starting repeated stratified ablation", flush=True)
    fold_scores, class_scores = run_repeated_cv_ablation(
        raw_documents,
        split_documents,
        structural,
        labels,
        n_splits=int(config["cv"]["n_splits"]),
        seeds=config["cv"]["seeds"],
        vectorizer_config=config["vectorizer"],
        model_config=config["model"],
    )
    cv_summary, per_class_delta = summarize_cv(fold_scores, class_scores)
    fold_scores.to_csv(
        output_dir / "cv_fold_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    class_scores.to_csv(
        output_dir / "cv_per_class_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_summary.to_csv(
        output_dir / "cv_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    per_class_delta.to_csv(
        output_dir / "per_class_f1_delta.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("[PLOT] creating Git-trackable PNG figures", flush=True)
    create_figures(
        audit=audit,
        event_share=event_share,
        burden=burden,
        cv_summary=cv_summary,
        per_class_delta=per_class_delta,
        figure_dir=figure_dir,
    )

    run_metadata = {
        "train_rows": len(train),
        "test_rows": len(test),
        "gene_columns": len(gene_columns),
        "classes": sorted(labels.unique().tolist()),
        "cv": config["cv"],
        "vectorizer": config["vectorizer"],
        "model": config["model"],
        "best_model": cv_summary.iloc[0].to_dict(),
    }
    write_json(run_metadata, output_dir / "run_metadata.json")
    print(cv_summary.to_string(index=False), flush=True)
    print(f"[DONE] outputs: {output_dir}", flush=True)
    print(f"[DONE] figures: {figure_dir}", flush=True)


if __name__ == "__main__":
    main()
