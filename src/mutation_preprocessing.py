"""암종 분류 데이터의 변이 문자열을 손실 없이 정규화합니다.

원본 데이터는 유전자별 wide 형식이며 한 셀에 여러 변이가 공백으로
구분되어 있을 수 있습니다. 이 모듈은 WT를 제외한 변이를 이벤트 단위
long 형식으로 변환하고, 결측은 WT와 구분하여 보존합니다.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


MUTATION_TYPES = ("SUB", "DELINS", "FS", "DEL", "INS", "DUP", "STOP", "OTHER")
POSITION_BINS = (
    ("001_050", 0, 50),
    ("051_100", 51, 100),
    ("101_250", 101, 250),
    ("251_500", 251, 500),
    ("501_PLUS", 501, np.inf),
)
NO_MUTATION_VALUES = frozenset({"", "WT", "NAN", "NONE", "NA", "N/A"})
PROTEIN_PREFIX = re.compile(r"^p\.", flags=re.IGNORECASE)
POSITION_PATTERN = re.compile(r"(\d+)")
SIMPLE_SUBSTITUTION = re.compile(r"^[A-Za-z*]+\d+[A-Za-z*]+$")


def is_missing_variant(value: object) -> bool:
    """원자료의 실제 결측 여부를 반환합니다."""
    return bool(pd.isna(value))


def is_no_mutation(value: object) -> bool:
    """WT와 기타 변이 없음 표기를 판정합니다. 실제 결측은 포함하지 않습니다."""
    if is_missing_variant(value):
        return False
    return str(value).strip().upper() in NO_MUTATION_VALUES


def split_mutation_events(value: object) -> tuple[str, ...]:
    """한 유전자 셀에 공백으로 나열된 변이를 이벤트 단위로 분리합니다."""
    if is_missing_variant(value) or is_no_mutation(value):
        return ()
    return tuple(token for token in str(value).strip().split() if token)


def normalize_variant(value: object) -> str:
    """단일 변이 이벤트의 접두사와 공백을 정규화합니다."""
    if is_missing_variant(value):
        return "MISSING"
    text = str(value).strip()
    if text.upper() in NO_MUTATION_VALUES:
        return "WT"
    return PROTEIN_PREFIX.sub("", text)


def mutation_type(value: object) -> str:
    """단일 변이 이벤트를 치환·결실·삽입 등으로 분류합니다."""
    text = normalize_variant(value)
    if text in {"WT", "MISSING"}:
        return text
    lowered = text.lower()
    if "delins" in lowered:
        return "DELINS"
    if "fs" in lowered:
        return "FS"
    if "del" in lowered:
        return "DEL"
    if "ins" in lowered:
        return "INS"
    if "dup" in lowered:
        return "DUP"
    if "*" in text or "stop" in lowered or "ter" in lowered:
        return "STOP"
    if ">" in text or SIMPLE_SUBSTITUTION.match(text):
        return "SUB"
    return "OTHER"


def mutation_position(value: object) -> int | None:
    """단일 변이 이벤트에서 첫 번째 위치 번호를 추출합니다."""
    text = normalize_variant(value)
    if text in {"WT", "MISSING"}:
        return None
    match = POSITION_PATTERN.search(text)
    return int(match.group(1)) if match else None


def position_bin(position: int | None) -> str | None:
    """변이 위치를 해석 가능한 고정 구간으로 변환합니다."""
    if position is None:
        return None
    for name, lower, upper in POSITION_BINS:
        if lower <= position <= upper:
            return name
    return None


class MutationStringPreprocessor(BaseEstimator, TransformerMixin):
    """wide 유전자 테이블을 변이 이벤트 단위 long 테이블로 변환합니다."""

    def __init__(
        self,
        id_column: str = "ID",
        target_column: str = "SUBCLASS",
        include_missing: bool = True,
    ) -> None:
        self.id_column = id_column
        self.target_column = target_column
        self.include_missing = include_missing

    def fit(self, frame: pd.DataFrame, labels: Iterable[object] | None = None):
        self._validate_frame(frame)
        excluded = {self.id_column, self.target_column}
        self.gene_columns_ = [
            column for column in frame.columns if column not in excluded
        ]
        if not self.gene_columns_:
            raise ValueError("전처리할 유전자 컬럼이 없습니다.")
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """WT를 제외하고 변이·결측 레코드를 long 형식으로 반환합니다."""
        check_is_fitted(self, attributes=["gene_columns_"])
        self._validate_frame(frame)
        missing_columns = sorted(set(self.gene_columns_) - set(frame.columns))
        if missing_columns:
            raise ValueError(f"필요한 유전자 컬럼이 없습니다: {missing_columns[:10]}")

        row_count = len(frame)
        sample_ids = (
            frame[self.id_column].astype(str).to_numpy(dtype=object)
            if self.id_column in frame
            else frame.index.astype(str).to_numpy(dtype=object)
        )
        targets = (
            frame[self.target_column].astype(str).to_numpy(dtype=object)
            if self.target_column in frame
            else None
        )
        records: dict[str, list[object]] = defaultdict(list)

        def append_record(
            row_position: int,
            gene: str,
            status: str,
            raw_cell: object,
            raw_event: object,
            event_order: int | None,
        ) -> None:
            normalized = (
                "MISSING" if status == "MISSING" else normalize_variant(raw_event)
            )
            event_position = (
                None if status == "MISSING" else mutation_position(raw_event)
            )
            records["row_position"].append(row_position)
            records["sample_id"].append(sample_ids[row_position])
            if targets is not None:
                records[self.target_column].append(targets[row_position])
            records["gene"].append(gene)
            records["status"].append(status)
            records["raw_cell"].append(raw_cell)
            records["raw_variant"].append(raw_event)
            records["normalized_variant"].append(normalized)
            records["event_order"].append(event_order)
            records["protein_prefix"].append(
                bool(
                    status == "MUTATION"
                    and PROTEIN_PREFIX.match(str(raw_event).strip())
                )
            )
            records["mutation_type"].append(
                "MISSING" if status == "MISSING" else mutation_type(raw_event)
            )
            records["position"].append(event_position)
            records["position_bin"].append(position_bin(event_position))

        for gene in self.gene_columns_:
            values = frame[gene].to_numpy(dtype=object)
            missing_mask = pd.isna(values)
            if self.include_missing:
                for row_position in np.flatnonzero(missing_mask):
                    value = values[row_position]
                    append_record(
                        row_position,
                        gene,
                        "MISSING",
                        value,
                        "MISSING",
                        None,
                    )

            mutation_candidates = np.flatnonzero(
                (~missing_mask) & (values != "WT")
            )
            for row_position in mutation_candidates:
                value = values[row_position]
                events = split_mutation_events(value)
                for event_order, event in enumerate(events, start=1):
                    append_record(
                        row_position,
                        gene,
                        "MUTATION",
                        value,
                        event,
                        event_order,
                    )

        columns = [
            "row_position",
            "sample_id",
            *([self.target_column] if targets is not None else []),
            "gene",
            "status",
            "raw_cell",
            "raw_variant",
            "normalized_variant",
            "event_order",
            "protein_prefix",
            "mutation_type",
            "position",
            "position_bin",
        ]
        result = pd.DataFrame(records)
        if result.empty:
            return pd.DataFrame(columns=columns)
        result["row_position"] = result["row_position"].astype(np.int32)
        result["event_order"] = result["event_order"].astype("Int16")
        result["position"] = result["position"].astype("Int32")
        return result[columns]

    def audit(
        self,
        frame: pd.DataFrame,
        long_frame: pd.DataFrame | None = None,
    ) -> dict[str, object]:
        """전처리 전후 건수와 파싱 품질을 JSON 직렬화 가능한 형태로 반환합니다."""
        check_is_fitted(self, attributes=["gene_columns_"])
        if long_frame is None:
            long_frame = self.transform(frame)

        mutation_rows = long_frame.loc[long_frame["status"].eq("MUTATION")]
        missing_rows = long_frame.loc[long_frame["status"].eq("MISSING")]
        mutation_cell_sizes = mutation_rows.groupby(
            ["row_position", "gene"], sort=False
        ).size()
        mutated_sample_counts = mutation_rows.groupby("row_position").size()
        mutation_cells = int(len(mutation_cell_sizes))
        total_cells = int(len(frame) * len(self.gene_columns_))
        missing_cells = int(len(missing_rows))
        mutation_events = int(len(mutation_rows))
        parsed_positions = int(mutation_rows["position"].notna().sum())

        return {
            "samples": int(len(frame)),
            "genes": int(len(self.gene_columns_)),
            "total_cells": total_cells,
            "wt_or_no_mutation_cells": total_cells - mutation_cells - missing_cells,
            "missing_cells": missing_cells,
            "samples_with_missing": int(missing_rows["row_position"].nunique()),
            "mutation_cells": mutation_cells,
            "mutation_events": mutation_events,
            "multi_event_cells": int(mutation_cell_sizes.gt(1).sum()),
            "max_events_in_cell": (
                int(mutation_cell_sizes.max()) if len(mutation_cell_sizes) else 0
            ),
            "samples_without_mutation": int(
                len(frame) - mutation_rows["row_position"].nunique()
            ),
            "mutation_events_per_sample_mean": (
                float(mutated_sample_counts.reindex(range(len(frame)), fill_value=0).mean())
                if len(frame)
                else 0.0
            ),
            "position_parse_rate": (
                parsed_positions / mutation_events if mutation_events else 0.0
            ),
            "protein_prefix_events": int(mutation_rows["protein_prefix"].sum()),
            "unique_raw_events": int(mutation_rows["raw_variant"].nunique()),
            "unique_normalized_events": int(
                mutation_rows["normalized_variant"].nunique()
            ),
            "mutation_type_counts": {
                str(key): int(value)
                for key, value in mutation_rows["mutation_type"]
                .value_counts()
                .to_dict()
                .items()
            },
        }

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame은 pandas DataFrame이어야 합니다.")
        if frame.columns.duplicated().any():
            duplicated = frame.columns[frame.columns.duplicated()].tolist()
            raise ValueError(f"중복 컬럼이 있습니다: {duplicated[:10]}")
