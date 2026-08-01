"""JSJ v3: TEST_004_1의 문자열·LightGBM 피처를 재현합니다."""

import re

import numpy as np
import pandas as pd

from src.pipelines.pipeline_jsj_v1 import JSJV1PreprocessingPipeline


AMINO_ACIDS = tuple("ARNDCEQGHILKMFPSTWYV")
AMINO_PATTERN = re.compile(r"[ARNDCEQGHILKMFPSTWYVX*]+")


def safe_token(token: object) -> str:
    token = str(token).strip()
    if token in {"", "nan", "None", "WT"}:
        return "WT"
    if token.startswith("p."):
        token = token[2:]
    token = token.replace("?", "X")
    if "=" in token and token:
        token = token.replace("=", token[0])
    if any(value in token.lower() for value in ("splice", "ext", "dup")):
        return "WT"
    return token


def mutation_kind(token: str) -> str:
    if token == "WT":
        return "WT"
    lowered = token.lower()
    if any(value in lowered for value in ("fs", "ins", "del")):
        return "FS"
    if "*" in token:
        return "STOP"
    if ">" in token:
        return "MISSENSE"
    amino = AMINO_PATTERN.findall(token)
    if len(amino) >= 2:
        return "SILENT" if amino[0] == amino[-1] else "MISSENSE"
    if len(token) >= 2 and token[0] in AMINO_ACIDS and token[-1] in AMINO_ACIDS:
        return "SILENT" if token[0] == token[-1] else "MISSENSE"
    return "MISSENSE"


def cell_code(value: object) -> int:
    counts = [0, 0, 0, 0]
    for raw_token in str(value).split():
        kind = mutation_kind(safe_token(raw_token))
        if kind == "FS":
            counts[0] += 1
        elif kind == "STOP":
            counts[1] += 1
        elif kind == "MISSENSE":
            counts[2] += 1
        elif kind == "SILENT":
            counts[3] += 1
    counts = [min(9, count) for count in counts]
    return counts[0] * 1000 + counts[1] * 100 + counts[2] * 10 + counts[3]


def row_amino_features(row: pd.Series) -> pd.Series:
    result = {
        f"{kind}_{amino}": 0
        for kind in ("FS", "STOP", "MISSENSE", "SILENT", "TO")
        for amino in AMINO_ACIDS
    }
    for value in row:
        for raw_token in str(value).split():
            token = safe_token(raw_token)
            kind = mutation_kind(token)
            if kind == "WT":
                continue
            amino = AMINO_PATTERN.findall(token)
            if amino and amino[0] and amino[0][0] in AMINO_ACIDS:
                result[f"{kind}_{amino[0][0]}"] += 1
            if kind == "MISSENSE" and len(amino) >= 2 and amino[-1][0] in AMINO_ACIDS:
                result[f"TO_{amino[-1][0]}"] += 1
    return pd.Series(result, dtype=np.int16)


def make_tree_features(raw: pd.DataFrame) -> pd.DataFrame:
    """원 제출 노트북과 동일한 결정론적 LightGBM 피처를 생성합니다."""
    raw = raw.fillna("WT").astype(str)
    gene_codes = raw.apply(lambda column: column.map(cell_code)).astype(np.int16)
    gene_codes.columns = [f"GENE_{column}" for column in gene_codes.columns]
    amino = raw.apply(row_amino_features, axis=1).astype(np.int16)
    summary = pd.DataFrame(index=raw.index)
    summary["mutated_gene_count"] = (gene_codes != 0).sum(axis=1)
    summary["mutation_code_sum"] = gene_codes.sum(axis=1)
    summary["mutation_code_max"] = gene_codes.max(axis=1)
    summary["fs_gene_count"] = (gene_codes >= 1000).sum(axis=1)
    summary["stop_gene_count"] = (
        (gene_codes >= 100) & (gene_codes < 1000)
    ).sum(axis=1)
    summary["missense_gene_count"] = (
        (gene_codes >= 10) & (gene_codes < 100)
    ).sum(axis=1)
    summary["silent_gene_count"] = (
        (gene_codes >= 1) & (gene_codes < 10)
    ).sum(axis=1)
    event_count = (
        (gene_codes // 1000)
        + ((gene_codes // 100) % 10)
        + ((gene_codes // 10) % 10)
        + (gene_codes % 10)
    )
    summary["multi_event_gene_count"] = (event_count > 1).sum(axis=1)
    for low, high in (
        (1, 2), (2, 10), (10, 20), (20, 100),
        (100, 200), (200, 1000), (1000, 2000), (2000, 10000),
    ):
        summary[f"code_{low}_{high}"] = (
            (gene_codes >= low) & (gene_codes < high)
        ).sum(axis=1)
    features = pd.concat([gene_codes, amino, summary.astype(np.int32)], axis=1)
    features.replace([np.inf, -np.inf], 0, inplace=True)
    return features.reset_index(drop=True)


def add_e1_features(base_features: pd.DataFrame) -> pd.DataFrame:
    gene_names = [column for column in base_features if column.startswith("GENE_")]
    gene_codes = base_features[gene_names].to_numpy(dtype=np.int16, copy=False)
    fs = (gene_codes // 1000) > 0
    stop = ((gene_codes // 100) % 10) > 0
    missense = ((gene_codes // 10) % 10) > 0
    silent = (gene_codes % 10) > 0
    mutated = gene_codes > 0
    binary_parts = []
    for suffix, values in (
        ("MUT", mutated), ("FS", fs), ("STOP", stop),
        ("MISSENSE", missense), ("SILENT", silent),
    ):
        columns = [f"E1_{column}_{suffix}" for column in gene_names]
        binary_parts.append(pd.DataFrame(values, columns=columns, dtype=np.uint8))
    event_matrix = np.stack(
        (
            (gene_codes // 1000).sum(axis=1),
            ((gene_codes // 100) % 10).sum(axis=1),
            ((gene_codes // 10) % 10).sum(axis=1),
            (gene_codes % 10).sum(axis=1),
        ),
        axis=1,
    ).astype(np.float32)
    total_events = event_matrix.sum(axis=1)
    mutated_genes = mutated.sum(axis=1).astype(np.float32)
    proportions = event_matrix / np.maximum(total_events, 1.0)[:, None]
    entropy = -(proportions * np.log(proportions + 1e-12)).sum(axis=1)
    summary = pd.DataFrame(
        {
            "E1_total_event_count": total_events,
            "E1_log1p_total_event_count": np.log1p(total_events),
            "E1_log1p_mutated_gene_count": np.log1p(mutated_genes),
            "E1_mutated_gene_ratio": mutated_genes / gene_codes.shape[1],
            "E1_events_per_mutated_gene": total_events / np.maximum(mutated_genes, 1.0),
            "E1_fs_event_ratio": proportions[:, 0],
            "E1_stop_event_ratio": proportions[:, 1],
            "E1_missense_event_ratio": proportions[:, 2],
            "E1_silent_event_ratio": proportions[:, 3],
            "E1_mutation_type_entropy": entropy,
            "E1_dominant_type_ratio": proportions.max(axis=1),
            "E1_active_mutation_type_count": (event_matrix > 0).sum(axis=1),
            "E1_multi_type_gene_count": (
                fs.astype(np.uint8) + stop.astype(np.uint8)
                + missense.astype(np.uint8) + silent.astype(np.uint8) > 1
            ).sum(axis=1),
        },
        dtype=np.float32,
    )
    return pd.concat(
        [base_features.reset_index(drop=True), *binary_parts, summary], axis=1
    )


class JSJV3PreprocessingPipeline(JSJV1PreprocessingPipeline):
    """문자열 피처와 트리 피처를 함께 반환하는 기준 파이프라인."""

    name = "jsj_v3"

    def __init__(self, **kwargs):
        kwargs["return_bundle"] = True
        super().__init__(**kwargs)
