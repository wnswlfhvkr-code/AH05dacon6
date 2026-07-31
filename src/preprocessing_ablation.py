"""Mutation-string preprocessing ablation utilities.

This module is intentionally independent from the common ``src.train`` entry
point.  It compares raw mutation-cell strings with correctly split mutation
events under the same cross-validation protocol.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.mutation_preprocessing import (
    MUTATION_TYPES,
    MutationStringPreprocessor,
    is_no_mutation,
    mutation_position,
    mutation_type,
    normalize_variant,
    split_mutation_events,
)


MODEL_LABELS = {
    "raw_cell_tfidf": "Raw-cell Word TF-IDF",
    "split_event_tfidf": "Split-event Word TF-IDF",
    "split_event_structural": "Split-event Word TF-IDF + structural",
}


def _safe_token(value: object) -> str:
    return str(value).strip().replace(" ", "_")


def build_mutation_documents(
    frame: pd.DataFrame,
    gene_columns: Iterable[str],
    *,
    split_events: bool,
) -> list[str]:
    """Turn each sample into a mutation-token document.

    In ``split_events=False`` mode, all events in one cell remain one encoded
    token.  In ``split_events=True`` mode, each event becomes an independent
    gene, gene-type, and exact-event token.
    """

    documents: list[list[str]] = [[] for _ in range(len(frame))]
    for gene in gene_columns:
        values = frame[gene].to_numpy(dtype=object)
        for row_index, value in enumerate(values):
            if pd.isna(value) or is_no_mutation(value):
                continue

            if split_events:
                events = split_mutation_events(value)
            else:
                events = (_safe_token(value),)

            documents[row_index].append(f"G={gene}")
            for event in events:
                normalized = normalize_variant(event)
                event_kind = mutation_type(event)
                documents[row_index].append(f"GT={gene}:{event_kind}")
                documents[row_index].append(f"E={gene}:{normalized}")

    return [" ".join(tokens) if tokens else "__NO_MUTATION__" for tokens in documents]


def build_structural_features(
    frame: pd.DataFrame,
    gene_columns: Iterable[str],
) -> pd.DataFrame:
    """Create label-free sample-level mutation structure features."""

    genes = list(gene_columns)
    row_count = len(frame)
    mutation_gene_count = np.zeros(row_count, dtype=np.int32)
    event_count = np.zeros(row_count, dtype=np.int32)
    multi_event_gene_count = np.zeros(row_count, dtype=np.int32)
    max_events_in_gene = np.zeros(row_count, dtype=np.int32)
    missing_count = np.zeros(row_count, dtype=np.int32)
    position_count = np.zeros(row_count, dtype=np.int32)
    position_sum = np.zeros(row_count, dtype=np.float64)
    position_maximum = np.zeros(row_count, dtype=np.int32)
    type_counts = {
        kind: np.zeros(row_count, dtype=np.int32) for kind in MUTATION_TYPES
    }

    for gene in genes:
        values = frame[gene].to_numpy(dtype=object)
        for row_index, value in enumerate(values):
            if pd.isna(value):
                missing_count[row_index] += 1
                continue
            if is_no_mutation(value):
                continue

            events = split_mutation_events(value)
            mutation_gene_count[row_index] += 1
            event_count[row_index] += len(events)
            multi_event_gene_count[row_index] += int(len(events) > 1)
            max_events_in_gene[row_index] = max(
                max_events_in_gene[row_index], len(events)
            )
            for event in events:
                type_counts[mutation_type(event)][row_index] += 1
                position = mutation_position(event)
                if position is not None:
                    position_count[row_index] += 1
                    position_sum[row_index] += position
                    position_maximum[row_index] = max(
                        position_maximum[row_index], position
                    )

    safe_event_count = np.maximum(event_count, 1)
    safe_position_count = np.maximum(position_count, 1)
    output: dict[str, np.ndarray] = {
        "mutation_gene_count": mutation_gene_count,
        "mutation_gene_count_log1p": np.log1p(mutation_gene_count),
        "mutation_gene_ratio": mutation_gene_count / max(len(genes), 1),
        "event_count": event_count,
        "event_count_log1p": np.log1p(event_count),
        "events_per_mutated_gene": event_count / np.maximum(mutation_gene_count, 1),
        "multi_event_gene_count": multi_event_gene_count,
        "multi_event_gene_ratio": multi_event_gene_count
        / np.maximum(mutation_gene_count, 1),
        "max_events_in_gene": max_events_in_gene,
        "missing_count": missing_count,
        "has_missing": (missing_count > 0).astype(np.int8),
        "position_parse_ratio": position_count / safe_event_count,
        "position_mean": position_sum / safe_position_count,
        "position_max": position_maximum,
    }
    for kind, counts in type_counts.items():
        output[f"type_{kind}_count"] = counts
        output[f"type_{kind}_ratio"] = counts / safe_event_count

    return pd.DataFrame(output, index=frame.index)


def make_hash_vectorizers(
    config: dict[str, object],
) -> tuple[HashingVectorizer, HashingVectorizer]:
    """Create stateless word and character hashing vectorizers.

    Hashing is fit-free, so the complete corpus can be transformed once without
    leaking validation labels or document-frequency statistics.  Fold-specific
    IDF is learned later with ``TfidfTransformer``.
    """

    word_range = tuple(config["word_ngram_range"])
    char_range = tuple(config["char_ngram_range"])
    word = HashingVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\S+",
        lowercase=False,
        ngram_range=word_range,
        n_features=int(config["word_hash_features"]),
        alternate_sign=False,
        norm=None,
        dtype=np.float32,
    )
    char = HashingVectorizer(
        analyzer="char_wb",
        lowercase=False,
        ngram_range=char_range,
        n_features=max(int(config["char_hash_features"]), 1),
        alternate_sign=False,
        norm=None,
        dtype=np.float32,
    )
    return word, char


def build_hashed_count_matrix(
    documents: list[str],
    vectorizer_config: dict[str, object],
):
    word, char = make_hash_vectorizers(vectorizer_config)
    if int(vectorizer_config["char_hash_features"]) <= 0:
        return word.transform(documents).tocsr()
    return hstack(
        [word.transform(documents), char.transform(documents)],
        format="csr",
    )


def _fold_tfidf(
    counts,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    *,
    sublinear_tf: bool,
):
    transformer = TfidfTransformer(
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=sublinear_tf,
    )
    train_matrix = transformer.fit_transform(counts[train_index])
    valid_matrix = transformer.transform(counts[valid_index])
    return train_matrix, valid_matrix


def _new_classifier(model_config: dict[str, object]) -> LinearSVC:
    return LinearSVC(
        C=float(model_config["C"]),
        class_weight=model_config.get("class_weight", "balanced"),
        max_iter=int(model_config.get("max_iter", 6000)),
        tol=float(model_config.get("tol", 1e-4)),
        dual="auto",
    )


def run_repeated_cv_ablation(
    raw_documents: list[str],
    split_documents: list[str],
    structural_features: pd.DataFrame,
    labels: pd.Series,
    *,
    n_splits: int,
    seeds: Iterable[int],
    vectorizer_config: dict[str, object],
    model_config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run leakage-safe repeated stratified CV for three preprocessing variants."""

    y = labels.astype(str).reset_index(drop=True)
    classes = np.asarray(sorted(y.unique()), dtype=object)
    fold_records: list[dict[str, object]] = []
    class_records: list[dict[str, object]] = []
    print("[CV] hashing raw-cell Word tokens once", flush=True)
    raw_counts = build_hashed_count_matrix(raw_documents, vectorizer_config)
    print("[CV] hashing split-event Word tokens once", flush=True)
    split_counts = build_hashed_count_matrix(split_documents, vectorizer_config)
    sublinear_tf = bool(vectorizer_config.get("sublinear_tf", True))

    for seed in seeds:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(seed),
        )
        for fold, (train_index, valid_index) in enumerate(
            splitter.split(np.zeros(len(y)), y),
            start=1,
        ):
            train_labels = y.iloc[train_index]
            valid_labels = y.iloc[valid_index]

            raw_train, raw_valid = _fold_tfidf(
                raw_counts,
                train_index,
                valid_index,
                sublinear_tf=sublinear_tf,
            )
            split_train, split_valid = _fold_tfidf(
                split_counts,
                train_index,
                valid_index,
                sublinear_tf=sublinear_tf,
            )

            scaler = StandardScaler()
            structural_train = scaler.fit_transform(
                structural_features.iloc[train_index]
            ).astype(np.float32)
            structural_valid = scaler.transform(
                structural_features.iloc[valid_index]
            ).astype(np.float32)
            combined_train = hstack(
                [split_train, csr_matrix(structural_train)],
                format="csr",
            )
            combined_valid = hstack(
                [split_valid, csr_matrix(structural_valid)],
                format="csr",
            )

            matrices = {
                "raw_cell_tfidf": (raw_train, raw_valid),
                "split_event_tfidf": (split_train, split_valid),
                "split_event_structural": (combined_train, combined_valid),
            }
            for model_name, (train_matrix, valid_matrix) in matrices.items():
                classifier = _new_classifier(model_config)
                classifier.fit(train_matrix, train_labels)
                predictions = classifier.predict(valid_matrix)
                macro_f1 = f1_score(
                    valid_labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
                fold_records.append(
                    {
                        "seed": int(seed),
                        "fold": fold,
                        "model": model_name,
                        "model_label": MODEL_LABELS[model_name],
                        "macro_f1": float(macro_f1),
                        "train_rows": len(train_index),
                        "valid_rows": len(valid_index),
                        "feature_count": int(train_matrix.shape[1]),
                    }
                )

                class_f1 = f1_score(
                    valid_labels,
                    predictions,
                    labels=classes,
                    average=None,
                    zero_division=0,
                )
                for class_name, score in zip(classes, class_f1):
                    class_records.append(
                        {
                            "seed": int(seed),
                            "fold": fold,
                            "model": model_name,
                            "model_label": MODEL_LABELS[model_name],
                            "SUBCLASS": str(class_name),
                            "f1": float(score),
                        }
                    )

            print(
                f"[CV] seed={seed} fold={fold}/{n_splits} complete",
                flush=True,
            )

    return pd.DataFrame(fold_records), pd.DataFrame(class_records)


def summarize_cv(
    fold_scores: pd.DataFrame,
    class_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        fold_scores.groupby(["model", "model_label"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
            evaluations=("macro_f1", "size"),
            feature_count_mean=("feature_count", "mean"),
        )
        .sort_values("macro_f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    per_class = (
        class_scores.groupby(
            ["model", "model_label", "SUBCLASS"],
            as_index=False,
        )
        .agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"))
    )
    raw = per_class.loc[
        per_class["model"].eq("raw_cell_tfidf"), ["SUBCLASS", "f1_mean"]
    ].rename(columns={"f1_mean": "raw_cell_f1"})
    best = per_class.loc[
        per_class["model"].eq("split_event_structural"),
        ["SUBCLASS", "f1_mean"],
    ].rename(columns={"f1_mean": "split_structural_f1"})
    delta = raw.merge(best, on="SUBCLASS", how="outer").fillna(0.0)
    delta["f1_delta"] = delta["split_structural_f1"] - delta["raw_cell_f1"]
    return summary, delta.sort_values("f1_delta", ascending=False)


def collect_preprocessing_audit(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict[str, dict[str, object]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preprocessor = MutationStringPreprocessor().fit(train)
    train_long = preprocessor.transform(train)
    test_long = preprocessor.transform(test)
    audit = {
        "train": preprocessor.audit(train, train_long),
        "test": preprocessor.audit(test, test_long),
    }

    examples = (
        pd.concat(
            [
                train_long.loc[
                    train_long["status"].eq("MUTATION"),
                    [
                        "sample_id",
                        "SUBCLASS",
                        "gene",
                        "raw_cell",
                        "raw_variant",
                        "normalized_variant",
                        "event_order",
                        "mutation_type",
                        "position",
                    ],
                ].assign(dataset="train"),
                test_long.loc[
                    test_long["status"].eq("MUTATION"),
                    [
                        "sample_id",
                        "gene",
                        "raw_cell",
                        "raw_variant",
                        "normalized_variant",
                        "event_order",
                        "mutation_type",
                        "position",
                    ],
                ].assign(dataset="test"),
            ],
            ignore_index=True,
        )
        .loc[lambda frame: frame["raw_cell"].astype(str).str.contains(" ")]
        .groupby(["dataset", "sample_id", "gene"], sort=False)
        .head(8)
        .head(80)
        .reset_index(drop=True)
    )
    return audit, examples, train_long, test_long


def count_unsplit_other(
    frame: pd.DataFrame,
    gene_columns: Iterable[str],
) -> int:
    count = 0
    for gene in gene_columns:
        for value in frame[gene]:
            if pd.isna(value) or is_no_mutation(value):
                continue
            count += int(mutation_type(_safe_token(value)) == "OTHER")
    return count


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def prepare_figure_data(
    audit: dict[str, dict[str, object]],
    train_long: pd.DataFrame,
    test_long: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce long event tables to small plotting frames before CV starts."""

    event_rows = []
    burden_rows = []
    for dataset, long_frame in (("Train", train_long), ("Test", test_long)):
        mutations = long_frame.loc[long_frame["status"].eq("MUTATION")]
        total = max(len(mutations), 1)
        for kind, count in mutations["mutation_type"].value_counts().items():
            event_rows.append(
                {
                    "dataset": dataset,
                    "mutation_type": kind,
                    "share": count / total,
                }
            )
        counts = mutations.groupby("row_position").size()
        sample_count = int(audit[dataset.lower()]["samples"])
        values = counts.reindex(range(sample_count), fill_value=0).to_numpy()
        burden_rows.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "log1p_event_count": np.log1p(values),
                }
            )
        )
    return pd.DataFrame(event_rows), pd.concat(burden_rows, ignore_index=True)


def create_figures(
    *,
    audit: dict[str, dict[str, object]],
    event_share: pd.DataFrame,
    burden: pd.DataFrame,
    cv_summary: pd.DataFrame,
    per_class_delta: pd.DataFrame,
    figure_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", font_scale=0.95)
    figure_dir.mkdir(parents=True, exist_ok=True)

    overview_rows = []
    for dataset in ("train", "test"):
        for metric, label in (
            ("mutation_cells", "Mutation cells"),
            ("mutation_events", "Split events"),
            ("multi_event_cells", "Multi-event cells"),
            ("missing_cells", "Missing cells"),
        ):
            overview_rows.append(
                {
                    "dataset": dataset.title(),
                    "metric": label,
                    "count": audit[dataset][metric],
                }
            )
    overview = pd.DataFrame(overview_rows)
    plt.figure(figsize=(10, 5.5))
    axis = sns.barplot(
        data=overview,
        x="metric",
        y="count",
        hue="dataset",
        palette=["#4C78A8", "#F58518"],
    )
    axis.set_yscale("symlog", linthresh=1)
    axis.set_title("Preprocessing audit: cells, split events, and missing values")
    axis.set_xlabel("")
    axis.set_ylabel("Count (symlog scale)")
    axis.tick_params(axis="x", rotation=15)
    _save_figure(figure_dir / "01_preprocessing_audit.png")

    plt.figure(figsize=(10, 5.5))
    axis = sns.barplot(
        data=event_share,
        x="mutation_type",
        y="share",
        hue="dataset",
        palette=["#4C78A8", "#F58518"],
    )
    axis.set_title("Mutation-type distribution after multi-event splitting")
    axis.set_xlabel("Mutation type")
    axis.set_ylabel("Share of mutation events")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    _save_figure(figure_dir / "02_mutation_type_share.png")

    plt.figure(figsize=(10, 5.5))
    axis = sns.histplot(
        data=burden,
        x="log1p_event_count",
        hue="dataset",
        stat="density",
        common_norm=False,
        bins=45,
        element="step",
        fill=False,
        palette=["#4C78A8", "#F58518"],
    )
    axis.set_title("Train–test mutation-event burden shift")
    axis.set_xlabel("log(1 + event count per sample)")
    axis.set_ylabel("Density")
    _save_figure(figure_dir / "03_event_burden_shift.png")

    order = cv_summary.sort_values("macro_f1_mean")["model_label"].tolist()
    plot_data = cv_summary.set_index("model_label").loc[order].reset_index()
    plt.figure(figsize=(9, 5.2))
    axis = sns.barplot(
        data=plot_data,
        y="model_label",
        x="macro_f1_mean",
        color="#4C78A8",
    )
    axis.errorbar(
        plot_data["macro_f1_mean"],
        np.arange(len(plot_data)),
        xerr=plot_data["macro_f1_std"],
        fmt="none",
        ecolor="#222222",
        capsize=4,
    )
    axis.set_title("Repeated CV preprocessing ablation")
    axis.set_xlabel("Macro F1 (mean ± SD)")
    axis.set_ylabel("")
    for index, value in enumerate(plot_data["macro_f1_mean"]):
        axis.text(value + 0.002, index, f"{value:.4f}", va="center")
    _save_figure(figure_dir / "04_cv_macro_f1.png")

    delta_plot = per_class_delta.sort_values("f1_delta")
    colors = np.where(delta_plot["f1_delta"] >= 0, "#54A24B", "#E45756")
    plt.figure(figsize=(10, 8))
    axis = plt.gca()
    axis.barh(delta_plot["SUBCLASS"], delta_plot["f1_delta"], color=colors)
    axis.axvline(0, color="#222222", linewidth=1)
    axis.set_title("Per-class F1 change: split-event + structural vs raw-cell")
    axis.set_xlabel("Mean F1 difference")
    axis.set_ylabel("SUBCLASS")
    _save_figure(figure_dir / "05_per_class_f1_delta.png")


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
