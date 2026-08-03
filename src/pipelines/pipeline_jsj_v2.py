"""JSJ v2: 유전자 변이 문자열을 XGBoost용 압축 구조 피처로 변환합니다."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


EVENT_TYPES = ("SUB", "SYN", "STOP", "FS", "DEL", "INS", "DELINS", "OTHER")
POSITION_BINS = ("001_050", "051_100", "101_250", "251_500", "501_PLUS")
POSITION_PATTERN = re.compile(r"(\d+)")
SUBSTITUTION_PATTERN = re.compile(r"^([A-Za-z*]+)(\d+)([A-Za-z*]+)$")


@dataclass(frozen=True)
class ParsedCell:
    """유전자 한 칸에서 추출한 결정론적 구조 정보입니다."""

    code: int
    mutated: int
    event_count: int
    multi_event: int
    type_counts: tuple[int, ...]
    position_counts: tuple[int, ...]


def _position_bin(position: int) -> int:
    if position <= 50:
        return 0
    if position <= 100:
        return 1
    if position <= 250:
        return 2
    if position <= 500:
        return 3
    return 4


def _mutation_type(event: str) -> str:
    lowered = event.lower()
    if "delins" in lowered:
        return "DELINS"
    if "fs" in lowered:
        return "FS"
    if "del" in lowered:
        return "DEL"
    if "ins" in lowered or "dup" in lowered:
        return "INS"
    if "*" in event or event.endswith("X") or "stop" in lowered:
        return "STOP"
    match = SUBSTITUTION_PATTERN.fullmatch(event)
    if match:
        reference, _, alternate = match.groups()
        return "SYN" if reference == alternate else "SUB"
    if ">" in event:
        return "SUB"
    return "OTHER"


@lru_cache(maxsize=250_000)
def _parse_cell(raw_value: str) -> ParsedCell:
    value = raw_value.strip()
    if not value or value.upper() in {"WT", "NAN", "NONE"}:
        return ParsedCell(0, 0, 0, 0, (0,) * len(EVENT_TYPES), (0,) * len(POSITION_BINS))

    events = value.split()
    type_counts = np.zeros(len(EVENT_TYPES), dtype=np.int16)
    position_counts = np.zeros(len(POSITION_BINS), dtype=np.int16)
    type_to_index = {name: index for index, name in enumerate(EVENT_TYPES)}

    for raw_event in events:
        event = raw_event[2:] if raw_event.startswith("p.") else raw_event
        event_type = _mutation_type(event)
        type_counts[type_to_index[event_type]] += 1
        position_match = POSITION_PATTERN.search(event)
        if position_match:
            position_counts[_position_bin(int(position_match.group(1)))] += 1

    disruptive = (
        type_counts[type_to_index["FS"]]
        + type_counts[type_to_index["DEL"]]
        + type_counts[type_to_index["INS"]]
        + type_counts[type_to_index["DELINS"]]
    )
    code = (
        min(9, int(disruptive)) * 1000
        + min(9, int(type_counts[type_to_index["STOP"]])) * 100
        + min(9, int(type_counts[type_to_index["SUB"]])) * 10
        + min(9, int(type_counts[type_to_index["SYN"]]))
    )
    if code == 0 and type_counts[type_to_index["OTHER"]] > 0:
        code = -1
    return ParsedCell(
        code=code,
        mutated=1,
        event_count=len(events),
        multi_event=int(len(events) > 1),
        type_counts=tuple(int(value) for value in type_counts),
        position_counts=tuple(int(value) for value in position_counts),
    )


class JSJV2PreprocessingPipeline:
    """유전자별 변이 코드와 환자 단위 구조 요약 피처를 생성합니다."""

    name = "jsj_v2"

    def __init__(self, **_: object) -> None:
        self.feature_columns: list[str] = []
        self.output_columns: list[str] = []
        self.dropped_constant_columns: list[str] = []
        self.label_encoder = LabelEncoder()
        self._fit_input_id: int | None = None
        self._fit_cache: pd.DataFrame | None = None

    def _validate_columns(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"누락된 피처가 있습니다: {sorted(missing)}")
        return features[self.feature_columns]

    def _build_features(self, features: pd.DataFrame) -> pd.DataFrame:
        row_count = len(features)
        gene_features: dict[str, np.ndarray] = {}
        mutated_gene_count = np.zeros(row_count, dtype=np.int32)
        total_event_count = np.zeros(row_count, dtype=np.int32)
        multi_event_gene_count = np.zeros(row_count, dtype=np.int32)
        missing_gene_count = np.zeros(row_count, dtype=np.int32)
        type_totals = np.zeros((row_count, len(EVENT_TYPES)), dtype=np.int32)
        position_totals = np.zeros((row_count, len(POSITION_BINS)), dtype=np.int32)

        for column in self.feature_columns:
            series = features[column]
            missing_gene_count += series.isna().to_numpy(dtype=np.int32)
            normalized = series.fillna("WT").astype(str)
            factor_codes, unique_values = pd.factorize(normalized, sort=False)
            parsed = [_parse_cell(str(value)) for value in unique_values]

            code_lookup = np.asarray([item.code for item in parsed], dtype=np.int16)
            mutated_lookup = np.asarray([item.mutated for item in parsed], dtype=np.int8)
            event_lookup = np.asarray([item.event_count for item in parsed], dtype=np.int16)
            multi_lookup = np.asarray([item.multi_event for item in parsed], dtype=np.int8)
            type_lookup = np.asarray([item.type_counts for item in parsed], dtype=np.int16)
            position_lookup = np.asarray(
                [item.position_counts for item in parsed], dtype=np.int16
            )

            gene_features[f"GENE_{column}"] = code_lookup[factor_codes]
            mutated_gene_count += mutated_lookup[factor_codes]
            total_event_count += event_lookup[factor_codes]
            multi_event_gene_count += multi_lookup[factor_codes]
            type_totals += type_lookup[factor_codes]
            position_totals += position_lookup[factor_codes]

        result = pd.DataFrame(gene_features, index=features.index)
        result["mutated_gene_count"] = mutated_gene_count
        result["total_event_count"] = total_event_count
        result["log1p_mutated_gene_count"] = np.log1p(mutated_gene_count).astype(np.float32)
        result["log1p_total_event_count"] = np.log1p(total_event_count).astype(np.float32)
        result["multi_event_gene_count"] = multi_event_gene_count
        result["missing_gene_count"] = missing_gene_count
        result["all_wt"] = (mutated_gene_count == 0).astype(np.int8)
        result["events_per_mutated_gene"] = (
            total_event_count / np.maximum(mutated_gene_count, 1)
        ).astype(np.float32)

        event_denominator = np.maximum(total_event_count, 1)
        for index, event_type in enumerate(EVENT_TYPES):
            result[f"type_{event_type}_count"] = type_totals[:, index]
            result[f"type_{event_type}_ratio"] = (
                type_totals[:, index] / event_denominator
            ).astype(np.float32)
        for index, position_name in enumerate(POSITION_BINS):
            result[f"position_{position_name}_count"] = position_totals[:, index]
            result[f"position_{position_name}_ratio"] = (
                position_totals[:, index] / event_denominator
            ).astype(np.float32)
        return result

    def fit(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> "JSJV2PreprocessingPipeline":
        self.feature_columns = features.columns.tolist()
        transformed = self._build_features(features)
        self.output_columns = transformed.columns[
            transformed.nunique(dropna=False) > 1
        ].tolist()
        self.dropped_constant_columns = [
            column for column in transformed.columns if column not in self.output_columns
        ]
        self.label_encoder.fit(labels)
        self._fit_input_id = id(features)
        self._fit_cache = transformed
        return self

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        self.fit(features, labels)
        transformed = self._fit_cache[self.output_columns].copy()
        self._fit_cache = None
        self._fit_input_id = None
        return transformed

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        selected = self._validate_columns(features)
        if self._fit_cache is not None and id(features) == self._fit_input_id:
            transformed = self._fit_cache
            self._fit_cache = None
            self._fit_input_id = None
        else:
            transformed = self._build_features(selected)
        return transformed[self.output_columns].copy()

    def summary(self) -> dict[str, int]:
        return {
            "remaining_features": len(self.output_columns),
            "dropped_constant_features": len(self.dropped_constant_columns),
        }

    def encode_labels(self, labels: pd.Series):
        return self.label_encoder.transform(labels)

    def decode_labels(self, labels):
        return self.label_encoder.inverse_transform(np.asarray(labels).astype(int))
