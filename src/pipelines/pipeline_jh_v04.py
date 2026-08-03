"""F0·F1·F3 변이 피처 생성 파이프라인입니다."""

from __future__ import annotations

from collections import Counter
import hashlib
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.pipelines.base import PreprocessingPipeline


CONSEQUENCE_TYPES = (
    "MISSENSE", "SYNONYMOUS", "STOP", "FRAMESHIFT",
    "COMPLEX", "OTHER", "UNPARSED",
)
NON_MUTATION_TOKENS = frozenset({"WT", "MISSING"})
AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY") + ("*", "X")
POSITION_BIN_EDGES = (-np.inf, 50, 100, 200, 400, 800, np.inf)
POSITION_BIN_LABELS = ("1-50", "51-100", "101-200", "201-400", "401-800", "801+")

STANDARD_AA_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")
SYNONYMOUS_EQUAL_PATTERN = re.compile(r"^([A-Z])(\d+)=$")
STOP_PATTERN = re.compile(r"^([A-Z])(\d+)\*$")
LEGACY_STOP_X_PATTERN = re.compile(r"^([A-Z])(\d+)X$")
POSITION_PATTERN = re.compile(r"(\d+)")


def normalize_event_token(token: object) -> str:
    normalized = re.sub(r"(?i)^p\.\s*", "", str(token).strip()).upper()
    normalized = normalized.replace("＊", "*")
    legacy_stop = LEGACY_STOP_X_PATTERN.fullmatch(normalized)
    if legacy_stop:
        return f"{legacy_stop.group(1)}{legacy_stop.group(2)}*"
    return normalized


def split_mutation_tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = re.sub(r"(?i)\bp\.\s*", "", str(value).strip())
    if not text:
        return []
    return list(dict.fromkeys(
        normalize_event_token(token)
        for token in re.split(r"\s+", text)
        if token.strip() and normalize_event_token(token) not in NON_MUTATION_TOKENS
    ))


def parse_event(token: str) -> dict[str, object]:
    exact_event = normalize_event_token(token)
    result: dict[str, object] = {
        "exact_event": exact_event,
        "consequence": "UNPARSED",
        "position": np.nan,
        "aa_from": pd.NA,
        "aa_to": pd.NA,
    }
    position_match = POSITION_PATTERN.search(exact_event)
    if position_match:
        result["position"] = int(position_match.group(1))

    if re.search(r"fs", exact_event, flags=re.IGNORECASE):
        result["consequence"] = "FRAMESHIFT"
        first_aa = re.match(r"^([A-Z])", exact_event)
        if first_aa:
            result["aa_from"] = first_aa.group(1)
        return result
    if re.search(r"delins|del|ins|dup", exact_event, flags=re.IGNORECASE):
        result["consequence"] = "COMPLEX"
        return result

    stop_match = STOP_PATTERN.fullmatch(exact_event)
    if stop_match:
        result.update(
            aa_from=stop_match.group(1),
            aa_to="*",
            position=int(stop_match.group(2)),
            consequence="STOP",
        )
        return result

    synonymous_match = SYNONYMOUS_EQUAL_PATTERN.fullmatch(exact_event)
    if synonymous_match:
        result.update(
            aa_from=synonymous_match.group(1),
            aa_to=synonymous_match.group(1),
            position=int(synonymous_match.group(2)),
            consequence="SYNONYMOUS",
        )
        return result

    aa_match = STANDARD_AA_PATTERN.fullmatch(exact_event)
    if aa_match:
        aa_from, position, aa_to = aa_match.groups()
        result.update(
            aa_from=aa_from,
            aa_to=aa_to,
            position=int(position),
            consequence="SYNONYMOUS" if aa_from == aa_to else "MISSENSE",
        )
        return result

    if position_match:
        result["consequence"] = "OTHER"
    return result


def parse_wide_mutations(features: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for gene in features.columns:
        values = features[gene]
        normalized = values.astype("string").str.strip().str.upper()
        candidate_positions = np.flatnonzero(
            (
                normalized.notna()
                & normalized.ne("")
                & ~normalized.isin(NON_MUTATION_TOKENS)
            ).to_numpy()
        )
        for row_index in candidate_positions:
            value = values.iloc[row_index]
            for token in split_mutation_tokens(value):
                records.append({"row_index": row_index, "gene": gene, **parse_event(token)})
    return pd.DataFrame.from_records(
        records,
        columns=("row_index", "gene", "exact_event", "consequence", "position", "aa_from", "aa_to"),
    )


def build_f0_matrix(features: pd.DataFrame) -> sparse.csr_matrix:
    columns = []
    for gene in features.columns:
        normalized = features[gene].astype("string").str.strip().str.upper()
        columns.append(
            (
                normalized.notna()
                & normalized.ne("")
                & ~normalized.isin(NON_MUTATION_TOKENS)
            ).to_numpy(dtype=np.float32)
        )
    return sparse.csr_matrix(np.column_stack(columns), dtype=np.float32)


def build_f1_matrix(events: pd.DataFrame, sample_count: int) -> tuple[np.ndarray, list[str]]:
    count_names = [
        "mutation_event_count", "mutated_gene_count", "multi_hit_gene_count",
        *(f"{value.lower()}_count" for value in CONSEQUENCE_TYPES),
    ]
    position_names = ["position_mean", "position_median", "position_max"]
    summary = pd.DataFrame(index=np.arange(sample_count))

    if not events.empty:
        summary = summary.join(events.groupby("row_index").agg(
            mutation_event_count=("exact_event", "size"),
            mutated_gene_count=("gene", "nunique"),
            position_mean=("position", "mean"),
            position_median=("position", "median"),
            position_max=("position", "max"),
        ))
        multi_hit = events.groupby(["row_index", "gene"]).size().gt(1).groupby(level=0).sum()
        summary = summary.join(multi_hit.rename("multi_hit_gene_count"))
        consequence = pd.crosstab(events["row_index"], events["consequence"]).reindex(
            columns=CONSEQUENCE_TYPES, fill_value=0,
        ).rename(columns=lambda value: f"{value.lower()}_count")
        summary = summary.join(consequence)

    summary[count_names + position_names] = summary.reindex(
        columns=count_names + position_names, fill_value=0,
    ).fillna(0).astype(float)
    summary["events_per_mutated_gene"] = (
        summary["mutation_event_count"] / summary["mutated_gene_count"].clip(lower=1)
    )
    denominator = summary["mutation_event_count"].clip(lower=1)
    ratio_names = []
    for consequence in CONSEQUENCE_TYPES:
        name = f"{consequence.lower()}_ratio"
        summary[name] = summary[f"{consequence.lower()}_count"] / denominator
        ratio_names.append(name)
    summary["is_all_wt"] = summary["mutation_event_count"].eq(0).astype(float)

    log_sources = count_names + position_names + ["events_per_mutated_gene"]
    log_names = []
    for name in log_sources:
        log_name = f"log1p_{name}"
        summary[log_name] = np.log1p(summary[name].clip(lower=0))
        log_names.append(log_name)
    feature_names = log_names + ratio_names + ["is_all_wt"]
    return summary[feature_names].to_numpy(dtype=np.float64), feature_names


def f3_feature_names() -> list[str]:
    return [
        *(f"AA_FROM={aa}" for aa in AMINO_ACIDS),
        *(f"AA_TO={aa}" for aa in AMINO_ACIDS),
        *(f"AA_CHANGE={left}>{right}" for left in AMINO_ACIDS for right in AMINO_ACIDS),
        *(f"POSITION_BIN={label}" for label in POSITION_BIN_LABELS),
    ]


def build_f3_matrix(events: pd.DataFrame, sample_count: int) -> tuple[sparse.csr_matrix, list[str]]:
    names = f3_feature_names()
    lookup = {name: index for index, name in enumerate(names)}
    valid_aa = set(AMINO_ACIDS)
    records: set[tuple[int, int]] = set()
    if not events.empty:
        position_bins = pd.cut(
            pd.to_numeric(events["position"], errors="coerce"),
            bins=POSITION_BIN_EDGES,
            labels=POSITION_BIN_LABELS,
            include_lowest=True,
        ).astype("string")
        for event, position_bin in zip(events.itertuples(index=False), position_bins):
            aa_from = str(event.aa_from) if pd.notna(event.aa_from) else None
            aa_to = str(event.aa_to) if pd.notna(event.aa_to) else None
            tokens = []
            if aa_from in valid_aa:
                tokens.append(f"AA_FROM={aa_from}")
            if aa_to in valid_aa:
                tokens.append(f"AA_TO={aa_to}")
            if aa_from in valid_aa and aa_to in valid_aa:
                tokens.append(f"AA_CHANGE={aa_from}>{aa_to}")
            if pd.notna(position_bin) and position_bin != "<NA>":
                tokens.append(f"POSITION_BIN={position_bin}")
            records.update((int(event.row_index), lookup[token]) for token in tokens)
    if not records:
        return sparse.csr_matrix((sample_count, len(names)), dtype=np.float32), names
    rows, columns = zip(*sorted(records))
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(sample_count, len(names)),
    )
    return matrix, names


def build_profile_groups(events: pd.DataFrame, sample_count: int) -> np.ndarray:
    profiles = np.full(sample_count, "ALL_WT", dtype=object)
    if not events.empty:
        tokens = events.assign(
            profile_token=events["gene"].astype(str) + ":" + events["exact_event"].astype(str)
        ).sort_values(["row_index", "profile_token"])
        joined = tokens.groupby("row_index")["profile_token"].agg("|".join)
        profiles[joined.index.to_numpy(dtype=int)] = joined.to_numpy(dtype=object)
    return np.asarray([
        hashlib.sha256(profile.encode("utf-8")).hexdigest() for profile in profiles
    ])


def make_class_burden_strata(
    labels: pd.Series,
    burden: np.ndarray,
    n_splits: int,
) -> tuple[pd.Series, pd.Series]:
    burden_bins = pd.cut(
        burden,
        bins=(-np.inf, 5, 9, 16, 33, np.inf),
        labels=("0-5", "6-9", "10-16", "17-33", "34+"),
        include_lowest=True,
    ).astype("string")
    labels = labels.reset_index(drop=True).astype("string")
    combined = "CLASS=" + labels + "|BURDEN=" + burden_bins
    rare_mask = combined.map(combined.value_counts()).lt(n_splits)
    rare_classes = set(labels.loc[rare_mask])
    strata = combined.copy()
    class_only = labels.isin(rare_classes)
    strata.loc[class_only] = "CLASS_ONLY=" + labels.loc[class_only]
    if strata.value_counts().lt(n_splits).any():
        raise ValueError("SGKF를 만들 수 없는 희소 계층이 있습니다.")
    return strata, burden_bins


class JHV04PreprocessingPipeline(PreprocessingPipeline):
    """학습 Fold에서 F0/F3 활성 열과 F1 scaler를 학습합니다."""

    name = "jh_v04"

    def __init__(self, **_: object) -> None:
        self.label_encoder = LabelEncoder()
        self.f1_scaler = StandardScaler()
        self.gene_columns: list[str] = []
        self.f0_active_mask: np.ndarray | None = None
        self.f3_active_mask: np.ndarray | None = None
        self.f1_feature_names: list[str] = []
        self.f3_all_names: list[str] = []

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "JHV04PreprocessingPipeline":
        self.gene_columns = features.columns.tolist()
        events = parse_wide_mutations(features)
        f0 = build_f0_matrix(features)
        f1, self.f1_feature_names = build_f1_matrix(events, len(features))
        f3, self.f3_all_names = build_f3_matrix(events, len(features))
        self.f0_active_mask = np.asarray(f0.getnnz(axis=0)).ravel() > 0
        self.f3_active_mask = np.asarray(f3.getnnz(axis=0)).ravel() > 0
        self.f1_scaler.fit(f1)
        self.label_encoder.fit(labels)
        return self

    def transform(self, features: pd.DataFrame) -> sparse.csr_matrix:
        if features.columns.tolist() != self.gene_columns:
            raise ValueError("학습 때와 유전자 열 또는 순서가 다릅니다.")
        events = parse_wide_mutations(features)
        f0 = build_f0_matrix(features)[:, self.f0_active_mask]
        f1, _ = build_f1_matrix(events, len(features))
        f3, _ = build_f3_matrix(events, len(features))
        return sparse.hstack(
            [f0, sparse.csr_matrix(self.f1_scaler.transform(f1).astype(np.float32)), f3[:, self.f3_active_mask]],
            format="csr",
            dtype=np.float32,
        )

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series) -> sparse.csr_matrix:
        return self.fit(features, labels).transform(features)

    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        return self.label_encoder.transform(labels)

    def decode_labels(self, labels: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(labels.astype(int))

    def summary(self) -> dict[str, int]:
        return {
            "f0_features": int(self.f0_active_mask.sum()),
            "f1_features": len(self.f1_feature_names),
            "f3_features": int(self.f3_active_mask.sum()),
            "remaining_features": int(self.f0_active_mask.sum()) + len(self.f1_feature_names) + int(self.f3_active_mask.sum()),
        }


# 기존 stash에서 사용한 이름과의 호환성
JHV1PreprocessingPipeline = JHV04PreprocessingPipeline
