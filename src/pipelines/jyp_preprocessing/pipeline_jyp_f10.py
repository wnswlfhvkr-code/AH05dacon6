"""jyp_f10 고정 JYP 전처리 파이프라인입니다.

AGENTS.md의 버전 독립 규칙에 따라 이 파일이 사용하는 파싱, 피처 생성,
학습 및 변환 로직을 파일 안에 직접 보관합니다.

F9의 피처 순서를 그대로 유지하고, 환자별 표준 아미노산 치환 조성을
요약하는 42개 고정·타깃 독립 F10 피처를 마지막에 붙입니다.
"""
from __future__ import annotations
import math
import re
from functools import lru_cache
import time
from typing import Sequence
import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from dataclasses import dataclass, replace
from src.pipelines.base import PreprocessingPipeline
WT = 'WT'
MISSING = 'MISSING'
MISSING_TOKENS = {'', '.', '-', 'NA', 'N/A', 'NAN', 'NONE', 'NULL', MISSING}
CONSEQUENCE_TYPES = ('MISSENSE', 'SYNONYMOUS', 'STOP', 'FRAMESHIFT', 'COMPLEX', 'OTHER', 'UNPARSED')
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
POSITION_BIN_LABELS = ('pos_1_50', 'pos_51_100', 'pos_101_250', 'pos_251_500', 'pos_501_1000', 'pos_gt_1000')
AA_TRANSITIONS = tuple((f'{reference}>{alternate}' for reference in AMINO_ACIDS for alternate in f'{AMINO_ACIDS}*'))
F9_AA_SUBSTITUTIONS = tuple((f'{reference}>{alternate}' for reference in AMINO_ACIDS for alternate in AMINO_ACIDS if reference != alternate))
F9_AA_SUBSTITUTION_FEATURE_NAMES = tuple((f"F9__global_aa_pair_log1p__{transition.replace('>', '_to_')}" for transition in F9_AA_SUBSTITUTIONS))
F10_AA_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
F10_AA_FROM_INDICES = np.asarray([F10_AA_INDEX[transition[0]] for transition in F9_AA_SUBSTITUTIONS], dtype=np.int16)
F10_AA_TO_INDICES = np.asarray([F10_AA_INDEX[transition[2]] for transition in F9_AA_SUBSTITUTIONS], dtype=np.int16)
F10_AA_COMPOSITION_FEATURE_NAMES = (
    tuple((f'F10__global_aa_from_fraction__{amino_acid}' for amino_acid in AMINO_ACIDS))
    + tuple((f'F10__global_aa_to_fraction__{amino_acid}' for amino_acid in AMINO_ACIDS))
    + ('F10__global_aa_pair_entropy_normalized', 'F10__global_aa_pair_dominant_fraction')
)
SIMPLE_STOP_X_RE = re.compile(f'^([{AMINO_ACIDS}X])(\\d+)X$')
SUBSTITUTION_RE = re.compile(f'^([{AMINO_ACIDS}])(\\d+)([{AMINO_ACIDS}*])$')
SYNONYMOUS_EQUAL_RE = re.compile(f'^([{AMINO_ACIDS}])(\\d+)=$')

@dataclass(frozen=True)
class ParsedMutationCell:
    """한 유전자 셀의 정규화 결과입니다."""
    state: str
    events: tuple[str, ...]
    consequences: tuple[str, ...]

def normalize_event(token: str) -> str:
    """표기 차이만 통일하고 원래 생물학적 사건은 합치지 않습니다."""
    upper = token.strip().upper()
    if upper in MISSING_TOKENS:
        return MISSING
    if upper == WT:
        return WT
    normalized = upper
    normalized = re.sub('DELINS', 'delins', normalized)
    normalized = re.sub('DEL', 'del', normalized)
    normalized = re.sub('INS', 'ins', normalized)
    normalized = re.sub('DUP', 'dup', normalized)
    normalized = re.sub('FS', 'fs', normalized)
    stop_x = SIMPLE_STOP_X_RE.fullmatch(normalized.upper())
    if stop_x:
        return f'{stop_x.group(1)}{stop_x.group(2)}*'
    return normalized

def classify_event(event: str) -> str:
    """정규화된 사건을 단백질 변화 유형으로 분류합니다."""
    lowered = event.lower()
    upper = event.upper()
    if upper.endswith('X') and any((marker in lowered for marker in ('delins', 'del', 'ins', 'dup', 'fs'))):
        return 'COMPLEX'
    if 'fs' in lowered:
        return 'FRAMESHIFT'
    if any((marker in lowered for marker in ('delins', 'del', 'ins', 'dup'))):
        return 'COMPLEX'
    if re.fullmatch(f'[{AMINO_ACIDS}X]\\d+\\*', upper):
        return 'STOP'
    substitution = SUBSTITUTION_RE.fullmatch(upper)
    if substitution:
        return 'SYNONYMOUS' if substitution.group(1) == substitution.group(3) else 'MISSENSE'
    if SYNONYMOUS_EQUAL_RE.fullmatch(upper):
        return 'SYNONYMOUS'
    if any((marker in lowered for marker in ('splice', 'fusion', 'amp'))):
        return 'OTHER'
    return 'UNPARSED'

def extract_protein_position(event: str) -> int | None:
    """단백질 변이 표기에서 첫 위치 숫자를 추출합니다."""
    match = re.search('\\d+', event)
    return int(match.group(0)) if match else None

def protein_position_bin(position: int | None) -> int | None:
    """고정된 단백질 위치 구간의 인덱스를 반환합니다."""
    if position is None or position < 1:
        return None
    if position <= 50:
        return 0
    if position <= 100:
        return 1
    if position <= 250:
        return 2
    if position <= 500:
        return 3
    if position <= 1000:
        return 4
    return 5

def extract_amino_acid_transition(event: str) -> str | None:
    """단순 치환·동의·종결 event를 위치 없는 아미노산 변화로 일반화합니다."""
    upper = event.upper()
    substitution = SUBSTITUTION_RE.fullmatch(upper)
    if substitution:
        return f'{substitution.group(1)}>{substitution.group(3)}'
    synonymous = SYNONYMOUS_EQUAL_RE.fullmatch(upper)
    if synonymous:
        return f'{synonymous.group(1)}>{synonymous.group(1)}'
    return None

@lru_cache(maxsize=None)
def parse_text(text: str) -> ParsedMutationCell:
    """문자열 셀을 파싱하며 같은 원문은 프로세스 안에서 재사용합니다."""
    stripped = text.strip()
    if not stripped:
        return ParsedMutationCell(MISSING, (), ())
    tokens = tuple((normalize_event(token) for token in re.split('\\s+', stripped)))
    events = tuple(sorted({token for token in tokens if token not in {WT, MISSING}}))
    if events:
        return ParsedMutationCell(state='MUTATED', events=events, consequences=tuple((classify_event(event) for event in events)))
    if MISSING in tokens:
        return ParsedMutationCell(MISSING, (), ())
    return ParsedMutationCell(WT, (), ())

def parse_cell(value: object) -> ParsedMutationCell:
    """문자열·None·NaN 입력을 공통 파서로 연결합니다."""
    if value is None:
        return ParsedMutationCell(MISSING, (), ())
    if isinstance(value, float) and math.isnan(value):
        return ParsedMutationCell(MISSING, (), ())
    return parse_text(str(value))

@dataclass
class MutationScanResult:
    """F0~F4와 F9 계산에 함께 사용하는 타겟 독립 스캔 결과입니다."""
    f0_all_genes: sparse.csr_matrix
    f2_all_gene_consequences: sparse.csr_matrix
    f3_all_gene_position_bins: sparse.csr_matrix
    f3_all_gene_aa_transitions: sparse.csr_matrix
    f9_global_aa_substitution_counts: sparse.csr_matrix
    f4_exact_event_rows: np.ndarray | None
    f4_exact_event_gene_indices: np.ndarray | None
    f4_exact_events: tuple[str, ...] | None
    mutated_gene_count: np.ndarray
    unique_event_count: np.ndarray
    multi_hit_gene_count: np.ndarray
    missing_gene_count: np.ndarray
    consequence_counts: np.ndarray
    qc: dict[str, int | float]

def scan_mutation_frame(features: pd.DataFrame, *, stage: str, show_progress: bool=True, progress_interval: int=25000, collect_exact_events: bool=False) -> MutationScanResult:
    """WT가 아닌 후보 셀만 파싱해 F0와 환자별 원시 통계를 만듭니다."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError('features는 pandas DataFrame이어야 합니다.')
    if features.columns.has_duplicates:
        raise ValueError('중복된 유전자 열 이름이 있습니다.')
    if progress_interval < 1:
        raise ValueError('progress_interval은 1 이상이어야 합니다.')
    started = time.perf_counter()
    values = features.to_numpy(dtype=object, copy=False)
    missing_mask = pd.isna(values)
    wt_mask = values == WT
    candidate_mask = ~(missing_mask | wt_mask)
    candidate_rows, candidate_columns = np.nonzero(candidate_mask)
    total_candidates = len(candidate_rows)
    n_rows, n_genes = values.shape
    missing_gene_count = missing_mask.sum(axis=1).astype(np.int32)
    mutated_gene_count = np.zeros(n_rows, dtype=np.int32)
    unique_event_count = np.zeros(n_rows, dtype=np.int32)
    multi_hit_gene_count = np.zeros(n_rows, dtype=np.int32)
    consequence_counts = np.zeros((n_rows, len(CONSEQUENCE_TYPES)), dtype=np.int32)
    consequence_lookup = {name: index for index, name in enumerate(CONSEQUENCE_TYPES)}
    aa_transition_lookup = {name: index for index, name in enumerate(AA_TRANSITIONS)}
    f9_aa_substitution_lookup = {name: index for index, name in enumerate(F9_AA_SUBSTITUTIONS)}
    f0_rows: list[int] = []
    f0_columns: list[int] = []
    f2_rows: list[int] = []
    f2_columns: list[int] = []
    f3_position_rows: list[int] = []
    f3_position_columns: list[int] = []
    f3_aa_rows: list[int] = []
    f3_aa_columns: list[int] = []
    f9_aa_rows: list[int] = []
    f9_aa_columns: list[int] = []
    f4_exact_event_rows: list[int] = []
    f4_exact_event_gene_indices: list[int] = []
    f4_exact_events: list[str] = []
    if show_progress:
        print(f'[mutation_scan:F0] {stage} | {n_rows:,}명 × {n_genes:,}유전자 | 후보 셀 {total_candidates:,}개')
    for item_number, (row_index, column_index) in enumerate(zip(candidate_rows, candidate_columns), start=1):
        parsed = parse_cell(values[row_index, column_index])
        if parsed.state == MISSING:
            missing_gene_count[row_index] += 1
        elif parsed.state != WT:
            f0_rows.append(int(row_index))
            f0_columns.append(int(column_index))
            mutated_gene_count[row_index] += 1
            unique_event_count[row_index] += len(parsed.events)
            if len(parsed.events) >= 2:
                multi_hit_gene_count[row_index] += 1
            for consequence in parsed.consequences:
                consequence_counts[row_index, consequence_lookup[consequence]] += 1
            for consequence in set(parsed.consequences):
                f2_rows.append(int(row_index))
                f2_columns.append(int(column_index) * len(CONSEQUENCE_TYPES) + consequence_lookup[consequence])
            position_indices: set[int] = set()
            aa_indices: set[int] = set()
            for event in parsed.events:
                if collect_exact_events:
                    f4_exact_event_rows.append(int(row_index))
                    f4_exact_event_gene_indices.append(int(column_index))
                    f4_exact_events.append(event)
                position_index = protein_position_bin(extract_protein_position(event))
                if position_index is not None:
                    position_indices.add(position_index)
                transition = extract_amino_acid_transition(event)
                if transition is not None:
                    aa_indices.add(aa_transition_lookup[transition])
                    f9_aa_index = f9_aa_substitution_lookup.get(transition)
                    if f9_aa_index is not None:
                        f9_aa_rows.append(int(row_index))
                        f9_aa_columns.append(f9_aa_index)
            for position_index in position_indices:
                f3_position_rows.append(int(row_index))
                f3_position_columns.append(int(column_index) * len(POSITION_BIN_LABELS) + position_index)
            for aa_index in aa_indices:
                f3_aa_rows.append(int(row_index))
                f3_aa_columns.append(int(column_index) * len(AA_TRANSITIONS) + aa_index)
        if show_progress and (item_number % progress_interval == 0 or item_number == total_candidates):
            elapsed = time.perf_counter() - started
            rate = item_number / elapsed if elapsed else 0.0
            remaining = (total_candidates - item_number) / rate if rate else 0.0
            print(f'[mutation_scan:F0] {item_number:,}/{total_candidates:,} ({item_number / max(total_candidates, 1):.1%}) | 경과 {elapsed:.1f}s | 예상 {remaining:.1f}s')
    f0_all_genes = sparse.csr_matrix((np.ones(len(f0_rows), dtype=np.float32), (np.asarray(f0_rows, dtype=np.int32), np.asarray(f0_columns, dtype=np.int32))), shape=(n_rows, n_genes), dtype=np.float32)
    f2_all_gene_consequences = sparse.csr_matrix((np.ones(len(f2_rows), dtype=np.float32), (np.asarray(f2_rows, dtype=np.int32), np.asarray(f2_columns, dtype=np.int32))), shape=(n_rows, n_genes * len(CONSEQUENCE_TYPES)), dtype=np.float32)
    f3_all_gene_position_bins = sparse.csr_matrix((np.ones(len(f3_position_rows), dtype=np.float32), (np.asarray(f3_position_rows, dtype=np.int32), np.asarray(f3_position_columns, dtype=np.int32))), shape=(n_rows, n_genes * len(POSITION_BIN_LABELS)), dtype=np.float32)
    f3_all_gene_aa_transitions = sparse.csr_matrix((np.ones(len(f3_aa_rows), dtype=np.float32), (np.asarray(f3_aa_rows, dtype=np.int32), np.asarray(f3_aa_columns, dtype=np.int32))), shape=(n_rows, n_genes * len(AA_TRANSITIONS)), dtype=np.float32)
    f9_global_aa_substitution_counts = sparse.csr_matrix((np.ones(len(f9_aa_rows), dtype=np.float32), (np.asarray(f9_aa_rows, dtype=np.int32), np.asarray(f9_aa_columns, dtype=np.int32))), shape=(n_rows, len(F9_AA_SUBSTITUTIONS)), dtype=np.float32)
    return MutationScanResult(f0_all_genes=f0_all_genes, f2_all_gene_consequences=f2_all_gene_consequences, f3_all_gene_position_bins=f3_all_gene_position_bins, f3_all_gene_aa_transitions=f3_all_gene_aa_transitions, f9_global_aa_substitution_counts=f9_global_aa_substitution_counts, f4_exact_event_rows=np.asarray(f4_exact_event_rows, dtype=np.int32) if collect_exact_events else None, f4_exact_event_gene_indices=np.asarray(f4_exact_event_gene_indices, dtype=np.int32) if collect_exact_events else None, f4_exact_events=tuple(f4_exact_events) if collect_exact_events else None, mutated_gene_count=mutated_gene_count, unique_event_count=unique_event_count, multi_hit_gene_count=multi_hit_gene_count, missing_gene_count=missing_gene_count, consequence_counts=consequence_counts, qc={'rows': n_rows, 'genes': n_genes, 'candidate_cells': total_candidates, 'mutated_cells': len(f0_rows), 'exact_event_records': len(f4_exact_events), 'missing_cells': int(missing_gene_count.sum()), 'elapsed_seconds': round(time.perf_counter() - started, 3)})

def select_active_genes(scan: MutationScanResult, gene_columns: Sequence[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """Fold-Train에서 한 번 이상 변이인 유전자만 F0 스키마로 고정합니다."""
    support = np.asarray(scan.f0_all_genes.sum(axis=0)).ravel()
    active_indices = np.flatnonzero(support > 0)
    active_genes = [gene_columns[index] for index in active_indices]
    dropped_genes = [gene for gene, count in zip(gene_columns, support) if count == 0]
    return (active_indices, active_genes, dropped_genes)
F1_BASE_COLUMNS = ('mutated_gene_count', 'mutated_gene_count_log1p', 'mutated_gene_count_clipped', 'unique_event_count', 'multi_hit_gene_count', 'missing_gene_count', 'is_all_wt')
F1_COUNT_COLUMNS = tuple((f'{name.lower()}_event_count' for name in CONSEQUENCE_TYPES))
F1_RATIO_COLUMNS = tuple((f'{name.lower()}_event_ratio' for name in CONSEQUENCE_TYPES))
F1_DERIVED_COLUMNS = ('synonymous_nonsynonymous_ratio', 'stop_frameshift_event_count', 'stop_frameshift_event_ratio', 'consequence_entropy', 'dominant_consequence_fraction')
F1_COLUMNS = F1_BASE_COLUMNS + F1_COUNT_COLUMNS + F1_RATIO_COLUMNS + F1_DERIVED_COLUMNS

def fit_burden_clip(scan: MutationScanResult, quantile: float=0.99) -> float:
    """Fold-Train의 변이 유전자 수에서 clip 기준을 학습합니다."""
    if not 0.0 < quantile <= 1.0:
        raise ValueError('burden clip quantile은 0보다 크고 1 이하여야 합니다.')
    return float(max(1.0, np.quantile(scan.mutated_gene_count, quantile)))

def build_f1_matrix(scan: MutationScanResult, *, burden_clip_value: float) -> sparse.csr_matrix:
    """고정된 26개 F1 요약 피처를 CSR 행렬로 반환합니다."""
    counts = scan.consequence_counts
    event_total = counts.sum(axis=1)
    ratios = np.divide(counts, event_total[:, None], out=np.zeros_like(counts, dtype=np.float64), where=event_total[:, None] > 0)
    consequence_index = {name: index for index, name in enumerate(CONSEQUENCE_TYPES)}
    synonymous = counts[:, consequence_index['SYNONYMOUS']]
    non_synonymous = counts[:, [consequence_index['MISSENSE'], consequence_index['STOP'], consequence_index['FRAMESHIFT'], consequence_index['COMPLEX']]].sum(axis=1)
    stop_frameshift = counts[:, [consequence_index['STOP'], consequence_index['FRAMESHIFT']]].sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy_terms = np.where(ratios > 0, ratios * np.log(ratios), 0.0)
    entropy = -entropy_terms.sum(axis=1)
    dominant = ratios.max(axis=1)
    columns: list[np.ndarray] = [scan.mutated_gene_count.astype(float), np.log1p(scan.mutated_gene_count), np.minimum(scan.mutated_gene_count, burden_clip_value).astype(float), scan.unique_event_count.astype(float), scan.multi_hit_gene_count.astype(float), scan.missing_gene_count.astype(float), ((scan.mutated_gene_count == 0) & (scan.missing_gene_count == 0)).astype(float)]
    columns.extend((counts[:, index].astype(float) for index in range(len(CONSEQUENCE_TYPES))))
    columns.extend((ratios[:, index] for index in range(len(CONSEQUENCE_TYPES))))
    columns.extend([(synonymous + 1.0) / (non_synonymous + 1.0), stop_frameshift.astype(float), np.divide(stop_frameshift, event_total, out=np.zeros_like(stop_frameshift, dtype=float), where=event_total > 0), entropy, dominant])
    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    if matrix.shape[1] != len(F1_COLUMNS):
        raise RuntimeError('F1 피처 수가 고정 스키마와 일치하지 않습니다.')
    return sparse.csr_matrix(matrix, dtype=np.float32)

def build_f9_global_aa_substitution_matrix(scan: MutationScanResult) -> sparse.csr_matrix:
    """환자별 표준 아미노산 치환 event count에 log1p를 적용합니다."""
    matrix = scan.f9_global_aa_substitution_counts.copy()
    matrix.sum_duplicates()
    matrix.data = np.log1p(matrix.data).astype(np.float32, copy=False)
    if matrix.shape[1] != len(F9_AA_SUBSTITUTION_FEATURE_NAMES):
        raise RuntimeError('F9 아미노산 치환 피처 수가 고정 스키마와 일치하지 않습니다.')
    return matrix

def build_f10_global_aa_composition_matrix(scan: MutationScanResult) -> sparse.csr_matrix:
    """절대 치환량과 분리된 환자별 아미노산 치환 조성을 반환합니다."""
    counts = scan.f9_global_aa_substitution_counts.tocsr(copy=True)
    counts.sum_duplicates()
    if counts.shape[1] != len(F9_AA_SUBSTITUTIONS):
        raise RuntimeError('F10 입력 치환 피처 수가 고정 스키마와 일치하지 않습니다.')
    totals = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64, copy=False)
    inverse_totals = np.divide(
        1.0,
        totals,
        out=np.zeros_like(totals),
        where=totals > 0,
    )
    coordinates = counts.tocoo(copy=False)
    fractions = coordinates.data.astype(np.float64, copy=False) * inverse_totals[coordinates.row]
    from_fractions = np.zeros((counts.shape[0], len(AMINO_ACIDS)), dtype=np.float64)
    to_fractions = np.zeros((counts.shape[0], len(AMINO_ACIDS)), dtype=np.float64)
    np.add.at(from_fractions, (coordinates.row, F10_AA_FROM_INDICES[coordinates.col]), fractions)
    np.add.at(to_fractions, (coordinates.row, F10_AA_TO_INDICES[coordinates.col]), fractions)
    entropy = np.zeros(counts.shape[0], dtype=np.float64)
    positive = fractions > 0
    np.add.at(
        entropy,
        coordinates.row[positive],
        -(fractions[positive] * np.log(fractions[positive])),
    )
    entropy /= np.log(len(F9_AA_SUBSTITUTIONS))
    dominant_fraction = np.zeros(counts.shape[0], dtype=np.float64)
    np.maximum.at(dominant_fraction, coordinates.row, fractions)
    matrix = np.column_stack(
        (from_fractions, to_fractions, entropy, dominant_fraction)
    ).astype(np.float32, copy=False)
    if matrix.shape[1] != len(F10_AA_COMPOSITION_FEATURE_NAMES):
        raise RuntimeError('F10 조성 피처 수가 고정 스키마와 일치하지 않습니다.')
    if not np.isfinite(matrix).all():
        raise RuntimeError('F10 조성 피처에 유한하지 않은 값이 있습니다.')
    return sparse.csr_matrix(matrix, dtype=np.float32)

def select_active_gene_consequences(scan: MutationScanResult, gene_columns: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Fold-Train에서 관찰된 유전자×변이유형 열만 고정합니다."""
    support = np.asarray(scan.f2_all_gene_consequences.sum(axis=0)).ravel()
    active_indices = np.flatnonzero(support > 0)
    feature_names: list[str] = []
    type_count = len(CONSEQUENCE_TYPES)
    for index in active_indices:
        gene_index, consequence_index = divmod(int(index), type_count)
        feature_names.append(f'F2__gene_consequence__{gene_columns[gene_index]}__{CONSEQUENCE_TYPES[consequence_index].lower()}')
    return (active_indices, feature_names)

def _active_indices(matrix: sparse.csr_matrix, minimum_support: int) -> np.ndarray:
    if minimum_support < 1:
        raise ValueError('F3 minimum_support는 1 이상이어야 합니다.')
    support = np.asarray(matrix.sum(axis=0)).ravel()
    return np.flatnonzero(support >= minimum_support)

def select_active_gene_position_bins(scan: MutationScanResult, gene_columns: Sequence[str], *, minimum_support: int) -> tuple[np.ndarray, list[str]]:
    """Fold-Train support를 통과한 유전자×위치 구간 피처를 고정합니다."""
    active = _active_indices(scan.f3_all_gene_position_bins, minimum_support)
    names: list[str] = []
    bin_count = len(POSITION_BIN_LABELS)
    for index in active:
        gene_index, bin_index = divmod(int(index), bin_count)
        names.append(f'F3__gene_position__{gene_columns[gene_index]}__{POSITION_BIN_LABELS[bin_index]}')
    return (active, names)

def select_active_gene_aa_transitions(scan: MutationScanResult, gene_columns: Sequence[str], *, minimum_support: int) -> tuple[np.ndarray, list[str]]:
    """Fold-Train support를 통과한 유전자×아미노산 변화 피처를 고정합니다."""
    active = _active_indices(scan.f3_all_gene_aa_transitions, minimum_support)
    names: list[str] = []
    transition_count = len(AA_TRANSITIONS)
    for index in active:
        gene_index, transition_index = divmod(int(index), transition_count)
        transition = AA_TRANSITIONS[transition_index].replace('>', '_to_')
        names.append(f'F3__gene_aa__{gene_columns[gene_index]}__{transition}')
    return (active, names)
ExactHotspotKey = tuple[int, str]

def _exact_event_records(scan: MutationScanResult) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rows = scan.f4_exact_event_rows
    gene_indices = scan.f4_exact_event_gene_indices
    events = scan.f4_exact_events
    if rows is None or gene_indices is None or events is None:
        raise RuntimeError('F4 exact event 수집이 활성화된 스캔 결과가 필요합니다.')
    return (rows, gene_indices, events)

def select_exact_hotspot_vocabulary(scan: MutationScanResult, gene_columns: Sequence[str], *, minimum_support: int) -> tuple[tuple[ExactHotspotKey, ...], list[str]]:
    """Fold-Train 고유 환자 support를 통과한 exact hotspot만 고정합니다."""
    if minimum_support < 1:
        raise ValueError('F4 minimum_support는 1 이상이어야 합니다.')
    _, gene_indices, events = _exact_event_records(scan)
    support = Counter(((int(gene_index), event) for gene_index, event in zip(gene_indices, events)))
    vocabulary = tuple(sorted((key for key, count in support.items() if count >= minimum_support), key=lambda key: (key[0], key[1])))
    feature_names = [f'F4__exact_hotspot__{gene_columns[gene_index]}__{event}' for gene_index, event in vocabulary]
    return (vocabulary, feature_names)

def build_exact_hotspot_matrix(scan: MutationScanResult, vocabulary: Sequence[ExactHotspotKey]) -> sparse.csr_matrix:
    """학습 때 고정한 exact hotspot만 변환하며 미관찰 event는 무시합니다."""
    rows, gene_indices, events = _exact_event_records(scan)
    n_rows = scan.f0_all_genes.shape[0]
    if not vocabulary:
        return sparse.csr_matrix((n_rows, 0), dtype=np.float32)
    lookup = {key: index for index, key in enumerate(vocabulary)}
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    for row_index, gene_index, event in zip(rows, gene_indices, events):
        feature_index = lookup.get((int(gene_index), event))
        if feature_index is not None:
            matrix_rows.append(int(row_index))
            matrix_columns.append(feature_index)
    matrix = sparse.csr_matrix((np.ones(len(matrix_rows), dtype=np.float32), (np.asarray(matrix_rows, dtype=np.int32), np.asarray(matrix_columns, dtype=np.int32))), shape=(n_rows, len(vocabulary)), dtype=np.float32)
    matrix.sum_duplicates()
    if matrix.nnz:
        matrix.data.fill(1.0)
    return matrix

def build_missing_gene_matrix(features: pd.DataFrame) -> sparse.csr_matrix:
    """WT와 분리할 유전자별 결측 마스크를 생성합니다."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError('features는 pandas DataFrame이어야 합니다.')
    if features.columns.has_duplicates:
        raise ValueError('중복된 유전자 열 이름이 있습니다.')
    string_tokens = {variant for token in MISSING_TOKENS for variant in (token, token.lower(), token.title())}
    missing = features.isna().to_numpy() | features.isin(string_tokens).to_numpy()
    return sparse.csr_matrix(missing, dtype=np.float32)
F7_SCORE_NAMES = ('mutated_log_odds_mean', 'bernoulli_llr_mean')

@dataclass(frozen=True, order=True)
class F7OrderedPair:
    """An immutable contrast whose positive direction favors ``first_label``."""
    first_label: str
    second_label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, 'first_label', str(self.first_label))
        object.__setattr__(self, 'second_label', str(self.second_label))
        if self.first_label == self.second_label:
            raise ValueError('F7 ordered pair labels must be different.')

@dataclass(frozen=True)
class F7PairStatistics:
    """Immutable learned statistics for one ordered pair."""
    pair: F7OrderedPair
    selected_gene_indices: tuple[int, ...]
    selected_gene_names: tuple[str, ...]
    directions: tuple[int, ...]
    burden_adjusted_effects: tuple[float, ...]
    mutation_log_odds: tuple[float, ...]
    log_mutation_probability_ratio: tuple[float, ...]
    log_wt_probability_ratio: tuple[float, ...]
    mutation_support: tuple[int, ...]
    direction_consistency: tuple[int, ...]
    selection_frequency: tuple[int, ...]

@dataclass(frozen=True)
class F7PairContrastModel:
    """Fold-train state used unchanged for validation or test transformation."""
    ordered_pairs: tuple[F7OrderedPair, ...]
    pair_statistics: tuple[F7PairStatistics, ...]
    gene_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    top_k_per_direction: int
    minimum_pair_mutation_support: int
    laplace_alpha: float
    burden_quantiles: int
    stability_folds: int
    minimum_direction_consistency: int
    minimum_selection_frequency: int
    random_state: int
    oof_fit_instabilities: tuple[tuple[int, str, str, str], ...] = ()

def _normalize_pair(pair: F7OrderedPair | Sequence[object]) -> F7OrderedPair:
    if isinstance(pair, F7OrderedPair):
        return pair
    if len(pair) != 2:
        raise ValueError('Each F7 ordered pair must contain exactly two labels.')
    return F7OrderedPair(str(pair[0]), str(pair[1]))

def _normalize_inputs(matrix: sparse.spmatrix, missing_matrix: sparse.spmatrix, labels: Sequence[object] | np.ndarray, gene_names: Sequence[str]) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, tuple[str, ...]]:
    mutations = sparse.csr_matrix(matrix, dtype=np.float32, copy=True)
    missing = sparse.csr_matrix(missing_matrix, dtype=np.float32, copy=True)
    if mutations.shape != missing.shape:
        raise ValueError('F7 mutation and missing matrices must have the same shape.')
    if mutations.shape[0] != len(labels):
        raise ValueError('F7 matrix and labels row counts do not match.')
    names = tuple((str(name) for name in gene_names))
    if mutations.shape[1] != len(names):
        raise ValueError('F7 gene name count does not match the matrix columns.')
    if len(set(names)) != len(names):
        raise ValueError('F7 gene names must be unique.')
    for candidate, description in ((mutations, 'mutation'), (missing, 'missing')):
        candidate.sum_duplicates()
        candidate.eliminate_zeros()
        if candidate.nnz and (not np.isfinite(candidate.data).all() or np.any(candidate.data < 0)):
            raise ValueError(f'F7 {description} matrix must contain finite nonnegative values.')
        if candidate.nnz:
            candidate.data.fill(1.0)
    if mutations.multiply(missing).nnz:
        raise ValueError('F7 cells cannot be marked as both mutated and missing.')
    normalized_labels = np.asarray([str(label) for label in labels], dtype=str)
    return (mutations, missing, normalized_labels, names)

def _validate_parameters(*, top_k_per_direction: int, minimum_pair_mutation_support: int, laplace_alpha: float, burden_quantiles: int, stability_folds: int, minimum_direction_consistency: int, minimum_selection_frequency: int) -> None:
    if top_k_per_direction < 1:
        raise ValueError('F7 top_k_per_direction must be at least 1.')
    if minimum_pair_mutation_support < 1:
        raise ValueError('F7 minimum pair mutation support must be at least 1.')
    if laplace_alpha <= 0:
        raise ValueError('F7 Laplace alpha must be positive.')
    if burden_quantiles < 2:
        raise ValueError('F7 burden_quantiles must be at least 2.')
    if stability_folds < 4:
        raise ValueError('F7 stability_folds must be at least 4.')
    if not 4 <= minimum_direction_consistency <= stability_folds:
        raise ValueError('F7 direction consistency must be between 4 and stability_folds.')
    if not 3 <= minimum_selection_frequency <= stability_folds:
        raise ValueError('F7 selection frequency must be between 3 and stability_folds.')

def _burden_strata(matrix: sparse.csr_matrix, quantiles: int) -> np.ndarray:
    burden = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    boundaries = np.quantile(burden, np.arange(1, quantiles) / quantiles)
    return np.searchsorted(boundaries, burden, side='left').astype(np.int8)

def _pooled_statistics(matrix: sparse.csr_matrix, missing: sparse.csr_matrix, binary_labels: np.ndarray, *, laplace_alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first = binary_labels == 0
    second = ~first
    mutated_first = np.asarray(matrix[first].sum(axis=0)).ravel().astype(np.float64)
    mutated_second = np.asarray(matrix[second].sum(axis=0)).ravel().astype(np.float64)
    observed_first = first.sum() - np.asarray(missing[first].sum(axis=0)).ravel()
    observed_second = second.sum() - np.asarray(missing[second].sum(axis=0)).ravel()
    if np.any(mutated_first > observed_first) or np.any(mutated_second > observed_second):
        raise ValueError('F7 mutation counts exceed observed (non-missing) counts.')
    probability_first = (mutated_first + laplace_alpha) / (observed_first + 2.0 * laplace_alpha)
    probability_second = (mutated_second + laplace_alpha) / (observed_second + 2.0 * laplace_alpha)
    log_odds = np.log(probability_first) - np.log1p(-probability_first) - np.log(probability_second) + np.log1p(-probability_second)
    return (probability_first, probability_second, log_odds, mutated_first + mutated_second)

def _burden_adjusted_effect(matrix: sparse.csr_matrix, missing: sparse.csr_matrix, binary_labels: np.ndarray, *, laplace_alpha: float, burden_quantiles: int) -> np.ndarray:
    strata = _burden_strata(matrix, burden_quantiles)
    weighted_effect = np.zeros(matrix.shape[1], dtype=np.float64)
    total_weight = np.zeros(matrix.shape[1], dtype=np.float64)
    for stratum in range(burden_quantiles):
        first = (binary_labels == 0) & (strata == stratum)
        second = (binary_labels == 1) & (strata == stratum)
        if not first.any() or not second.any():
            continue
        mutated_first = np.asarray(matrix[first].sum(axis=0)).ravel()
        mutated_second = np.asarray(matrix[second].sum(axis=0)).ravel()
        observed_first = first.sum() - np.asarray(missing[first].sum(axis=0)).ravel()
        observed_second = second.sum() - np.asarray(missing[second].sum(axis=0)).ravel()
        valid = (observed_first > 0) & (observed_second > 0)
        probability_first = (mutated_first + laplace_alpha) / (observed_first + 2.0 * laplace_alpha)
        probability_second = (mutated_second + laplace_alpha) / (observed_second + 2.0 * laplace_alpha)
        effect = np.log(probability_first) - np.log1p(-probability_first) - np.log(probability_second) + np.log1p(-probability_second)
        weight = np.divide(observed_first * observed_second, observed_first + observed_second, out=np.zeros_like(effect), where=valid)
        weighted_effect += effect * weight
        total_weight += weight
    return np.divide(weighted_effect, total_weight, out=np.zeros_like(weighted_effect), where=total_weight > 0)

def _rank_direction(effect: np.ndarray, eligible: np.ndarray, gene_names: np.ndarray, *, direction: int, top_k: int) -> np.ndarray:
    candidates = np.flatnonzero(eligible & (effect * direction > 0))
    if not len(candidates):
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((gene_names[candidates], -(effect[candidates] * direction)))
    return candidates[order[:top_k]].astype(np.int64, copy=False)

def _fit_pair_statistics(matrix: sparse.csr_matrix, missing: sparse.csr_matrix, labels: np.ndarray, gene_names: tuple[str, ...], pair: F7OrderedPair, *, top_k_per_direction: int, minimum_pair_mutation_support: int, laplace_alpha: float, burden_quantiles: int, stability_folds: int, minimum_direction_consistency: int, minimum_selection_frequency: int, random_state: int) -> F7PairStatistics:
    pair_rows = (labels == pair.first_label) | (labels == pair.second_label)
    pair_matrix = matrix[pair_rows]
    pair_missing = missing[pair_rows]
    pair_labels = np.where(labels[pair_rows] == pair.first_label, 0, 1).astype(np.int8)
    class_sizes = np.bincount(pair_labels, minlength=2)
    if np.any(class_sizes < stability_folds):
        raise ValueError(f'F7 pair {pair.first_label!r} vs {pair.second_label!r} requires at least {stability_folds} samples per label for stability CV; found {class_sizes[0]} and {class_sizes[1]}.')
    probability_first, probability_second, log_odds, support = _pooled_statistics(pair_matrix, pair_missing, pair_labels, laplace_alpha=laplace_alpha)
    full_effect = _burden_adjusted_effect(pair_matrix, pair_missing, pair_labels, laplace_alpha=laplace_alpha, burden_quantiles=burden_quantiles)
    eligible = support >= minimum_pair_mutation_support
    direction_counts = np.zeros(matrix.shape[1], dtype=np.int16)
    selection_counts = np.zeros(matrix.shape[1], dtype=np.int16)
    splitter = StratifiedKFold(n_splits=stability_folds, shuffle=True, random_state=random_state)
    names_array = np.asarray(gene_names)
    full_direction = np.sign(full_effect)
    for train_indices, _ in splitter.split(np.zeros(len(pair_labels)), pair_labels):
        fold_matrix = pair_matrix[train_indices]
        fold_missing = pair_missing[train_indices]
        fold_labels = pair_labels[train_indices]
        fold_effect = _burden_adjusted_effect(fold_matrix, fold_missing, fold_labels, laplace_alpha=laplace_alpha, burden_quantiles=burden_quantiles)
        _, _, _, fold_support = _pooled_statistics(fold_matrix, fold_missing, fold_labels, laplace_alpha=laplace_alpha)
        fold_eligible = fold_support >= minimum_pair_mutation_support
        direction_counts += ((full_direction != 0) & (np.sign(fold_effect) == full_direction)).astype(np.int16)
        for direction in (1, -1):
            selected = _rank_direction(fold_effect, fold_eligible, names_array, direction=direction, top_k=top_k_per_direction)
            selection_counts[selected] += 1
    stable = eligible & (direction_counts >= minimum_direction_consistency) & (selection_counts >= minimum_selection_frequency)
    selected_parts = [_rank_direction(full_effect, stable, names_array, direction=direction, top_k=top_k_per_direction) for direction in (1, -1)]
    selected = np.concatenate(selected_parts)
    directions = np.sign(full_effect[selected]).astype(np.int8)
    return F7PairStatistics(pair=pair, selected_gene_indices=tuple((int(value) for value in selected)), selected_gene_names=tuple((gene_names[index] for index in selected)), directions=tuple((int(value) for value in directions)), burden_adjusted_effects=tuple((float(value) for value in full_effect[selected])), mutation_log_odds=tuple((float(value) for value in log_odds[selected])), log_mutation_probability_ratio=tuple((float(value) for value in np.log(probability_first[selected] / probability_second[selected]))), log_wt_probability_ratio=tuple((float(value) for value in np.log((1.0 - probability_first[selected]) / (1.0 - probability_second[selected])))), mutation_support=tuple((int(value) for value in support[selected])), direction_consistency=tuple((int(value) for value in direction_counts[selected])), selection_frequency=tuple((int(value) for value in selection_counts[selected])))

def build_f7_feature_names(ordered_pairs: Sequence[F7OrderedPair | Sequence[object]]) -> list[str]:
    """Return two deterministic feature names for every ordered pair."""
    pairs = tuple((_normalize_pair(pair) for pair in ordered_pairs))
    return [f"F7__{pair.first_label.replace(' ', '_')}_vs_{pair.second_label.replace(' ', '_')}__{score_name}" for pair in pairs for score_name in F7_SCORE_NAMES]

def fit_full(matrix: sparse.spmatrix, missing_matrix: sparse.spmatrix, labels: Sequence[object] | np.ndarray, gene_names: Sequence[str], ordered_pairs: Sequence[F7OrderedPair | Sequence[object]], *, top_k_per_direction: int=25, minimum_pair_mutation_support: int=10, laplace_alpha: float=4.0, burden_quantiles: int=5, stability_folds: int=5, minimum_direction_consistency: int=4, minimum_selection_frequency: int=3, random_state: int=42) -> F7PairContrastModel:
    """Fit stable pair contrasts on one training partition."""
    _validate_parameters(top_k_per_direction=top_k_per_direction, minimum_pair_mutation_support=minimum_pair_mutation_support, laplace_alpha=laplace_alpha, burden_quantiles=burden_quantiles, stability_folds=stability_folds, minimum_direction_consistency=minimum_direction_consistency, minimum_selection_frequency=minimum_selection_frequency)
    mutations, missing, normalized_labels, names = _normalize_inputs(matrix, missing_matrix, labels, gene_names)
    pairs = tuple((_normalize_pair(pair) for pair in ordered_pairs))
    if not pairs:
        raise ValueError('F7 requires at least one ordered label pair.')
    if len(set(pairs)) != len(pairs):
        raise ValueError('F7 ordered label pairs must be unique.')
    available = set(normalized_labels)
    absent = sorted({label for pair in pairs for label in (pair.first_label, pair.second_label)} - available)
    if absent:
        raise ValueError(f'F7 ordered pairs contain labels absent from training data: {absent}.')
    statistics = tuple((_fit_pair_statistics(mutations, missing, normalized_labels, names, pair, top_k_per_direction=top_k_per_direction, minimum_pair_mutation_support=minimum_pair_mutation_support, laplace_alpha=laplace_alpha, burden_quantiles=burden_quantiles, stability_folds=stability_folds, minimum_direction_consistency=minimum_direction_consistency, minimum_selection_frequency=minimum_selection_frequency, random_state=random_state) for pair in pairs))
    return F7PairContrastModel(ordered_pairs=pairs, pair_statistics=statistics, gene_names=names, feature_names=tuple(build_f7_feature_names(pairs)), top_k_per_direction=top_k_per_direction, minimum_pair_mutation_support=minimum_pair_mutation_support, laplace_alpha=laplace_alpha, burden_quantiles=burden_quantiles, stability_folds=stability_folds, minimum_direction_consistency=minimum_direction_consistency, minimum_selection_frequency=minimum_selection_frequency, random_state=random_state)

def build_f7_matrix(matrix: sparse.spmatrix, missing_matrix: sparse.spmatrix, model: F7PairContrastModel) -> sparse.csr_matrix:
    """Apply a fitted model; missing genes contribute to neither likelihood side."""
    dummy_labels = np.zeros(matrix.shape[0], dtype=np.int8)
    mutations, missing, _, names = _normalize_inputs(matrix, missing_matrix, dummy_labels, model.gene_names)
    if names != model.gene_names:
        raise ValueError('F7 transform gene schema differs from the fitted model.')
    output_columns: list[np.ndarray] = []
    for statistics in model.pair_statistics:
        indices = np.asarray(statistics.selected_gene_indices, dtype=np.int64)
        mutated = mutations[:, indices].toarray() > 0
        observed = ~(missing[:, indices].toarray() > 0)
        log_odds = np.asarray(statistics.mutation_log_odds, dtype=np.float64)
        mutated_count = mutated.sum(axis=1)
        mutated_sum = (mutated * log_odds[None, :]).sum(axis=1)
        mutated_mean = np.divide(mutated_sum, mutated_count, out=np.zeros(matrix.shape[0], dtype=np.float64), where=mutated_count > 0)
        mutation_ratio = np.asarray(statistics.log_mutation_probability_ratio, dtype=np.float64)
        wt_ratio = np.asarray(statistics.log_wt_probability_ratio, dtype=np.float64)
        likelihood_terms = np.where(mutated, mutation_ratio[None, :], wt_ratio[None, :])
        likelihood_sum = np.where(observed, likelihood_terms, 0.0).sum(axis=1)
        observed_count = observed.sum(axis=1)
        normalized_likelihood = np.divide(likelihood_sum, observed_count, out=np.zeros(matrix.shape[0], dtype=np.float64), where=observed_count > 0)
        output_columns.extend((mutated_mean, normalized_likelihood))
    output = np.column_stack(output_columns).astype(np.float32, copy=False)
    if output.shape[1] != len(model.feature_names):
        raise RuntimeError('F7 output columns do not match the fixed feature schema.')
    if not np.isfinite(output).all():
        raise RuntimeError('F7 output contains non-finite values.')
    return sparse.csr_matrix(output, dtype=np.float32)

def fit_with_oof(matrix: sparse.spmatrix, missing_matrix: sparse.spmatrix, labels: Sequence[object] | np.ndarray, gene_names: Sequence[str], ordered_pairs: Sequence[F7OrderedPair | Sequence[object]], *, oof_folds: int=5, top_k_per_direction: int=25, minimum_pair_mutation_support: int=10, laplace_alpha: float=4.0, burden_quantiles: int=5, stability_folds: int=5, minimum_direction_consistency: int=4, minimum_selection_frequency: int=3, random_state: int=42) -> tuple[F7PairContrastModel, sparse.csr_matrix]:
    """Return the full model and fold-local, leakage-safe training features."""
    mutations, missing, normalized_labels, names = _normalize_inputs(matrix, missing_matrix, labels, gene_names)
    if oof_folds < 2:
        raise ValueError('F7 oof_folds must be at least 2.')
    _, counts = np.unique(normalized_labels, return_counts=True)
    if not len(counts) or counts.min() < oof_folds:
        raise ValueError(f'F7 OOF requires at least {oof_folds} samples for every label.')
    fit_kwargs = dict(top_k_per_direction=top_k_per_direction, minimum_pair_mutation_support=minimum_pair_mutation_support, laplace_alpha=laplace_alpha, burden_quantiles=burden_quantiles, stability_folds=stability_folds, minimum_direction_consistency=minimum_direction_consistency, minimum_selection_frequency=minimum_selection_frequency, random_state=random_state)
    full_model = fit_full(mutations, missing, normalized_labels, names, ordered_pairs, **fit_kwargs)
    splitter = StratifiedKFold(n_splits=oof_folds, shuffle=True, random_state=random_state)
    oof = np.zeros((mutations.shape[0], len(full_model.feature_names)), dtype=np.float32)
    oof_fit_instabilities: list[tuple[int, str, str, str]] = []
    for fold_number, (train_indices, valid_indices) in enumerate(splitter.split(np.zeros(len(normalized_labels)), normalized_labels), start=1):
        fold_model = fit_full(mutations[train_indices], missing[train_indices], normalized_labels[train_indices], names, ordered_pairs, **{**fit_kwargs, 'random_state': random_state + fold_number})
        oof_fit_instabilities.extend(((fold_number, statistics.pair.first_label, statistics.pair.second_label, 'zero_stable_genes') for statistics in fold_model.pair_statistics if not statistics.selected_gene_indices))
        oof[valid_indices] = build_f7_matrix(mutations[valid_indices], missing[valid_indices], fold_model).toarray()
    full_model = replace(full_model, oof_fit_instabilities=tuple(oof_fit_instabilities))
    return (full_model, sparse.csr_matrix(oof, dtype=np.float32))

def build_f7_selected_gene_catalog(model: F7PairContrastModel) -> list[dict[str, object]]:
    """Return one metadata record for every selected pair/gene direction."""
    catalog: list[dict[str, object]] = []
    for pair_index, statistics in enumerate(model.pair_statistics):
        if not statistics.selected_gene_indices:
            catalog.append({'record_type': 'fit_instability', 'fit_scope': 'full_fit', 'oof_fold': None, 'reason': 'zero_stable_genes', 'pair_index': pair_index, 'first_label': statistics.pair.first_label, 'second_label': statistics.pair.second_label, 'selected_gene_count': 0})
        direction_ranks = {1: 0, -1: 0}
        for values in zip(statistics.selected_gene_indices, statistics.selected_gene_names, statistics.directions, statistics.burden_adjusted_effects, statistics.mutation_log_odds, statistics.mutation_support, statistics.direction_consistency, statistics.selection_frequency):
            gene_index, gene_name, direction, effect, log_odds, support, consistency, frequency = values
            direction_ranks[direction] += 1
            catalog.append({'record_type': 'selected_gene', 'fit_scope': 'full_fit', 'oof_fold': None, 'reason': None, 'pair_index': pair_index, 'first_label': statistics.pair.first_label, 'second_label': statistics.pair.second_label, 'gene_index': gene_index, 'gene_name': gene_name, 'direction': 'first' if direction > 0 else 'second', 'direction_rank': direction_ranks[direction], 'burden_adjusted_effect': effect, 'mutation_log_odds': log_odds, 'pair_mutation_support': support, 'direction_consistency': consistency, 'selection_frequency': frequency})
    pair_indices = {(pair.first_label, pair.second_label): pair_index for pair_index, pair in enumerate(model.ordered_pairs)}
    catalog.extend(({'record_type': 'fit_instability', 'fit_scope': 'inner_oof', 'oof_fold': int(fold_number), 'reason': reason, 'pair_index': pair_indices[first_label, second_label], 'first_label': first_label, 'second_label': second_label, 'selected_gene_count': 0} for fold_number, first_label, second_label, reason in model.oof_fit_instabilities))
    return catalog
class F10GlobalAACompositionNoRawPreprocessingPipeline(PreprocessingPipeline):
    """F9에 부담량 독립적인 전역 아미노산 치환 조성을 더한 F10 구현입니다."""
    name = 'jyp_f10'
    artifact_schema_version = 6

    def __init__(self, *, burden_clip_quantile: float=0.99, f3_position_min_support: int=2, f3_aa_min_support: int=2, f4_min_support: int=10, f7_pairs: Sequence[Sequence[str]]=(('KIRC', 'KIPAN'), ('LGG', 'GBMLGG')), f7_top_k_per_direction: int=5, f7_min_gene_support: int=10, f7_laplace_alpha: float=4.0, f7_burden_quantiles: int=5, f7_stability_folds: int=5, f7_min_direction_consistency: int=4, f7_min_selection_frequency: int=3, f7_random_state: int=42, show_progress: bool=True, progress_interval: int=25000) -> None:
        super().__init__()
        self.pipeline_name = self.name
        self.feature_blocks = ('f0', 'f1', 'f2', 'f3_position', 'f3_aa', 'f4', 'f7_paircontrast', 'f9_global_aa_pair_log1p', 'f10_global_aa_composition')
        self.steps = self.feature_blocks
        if progress_interval < 1:
            raise ValueError('progress_interval은 1 이상이어야 합니다.')
        self.show_progress = bool(show_progress)
        self.progress_interval = int(progress_interval)
        if not 0.0 < burden_clip_quantile <= 1.0:
            raise ValueError('burden_clip_quantile은 0보다 크고 1 이하여야 합니다.')
        self.burden_clip_quantile = float(burden_clip_quantile)
        if f3_position_min_support < 1:
            raise ValueError('F3 minimum support는 1 이상이어야 합니다.')
        self.f3_position_min_support = int(f3_position_min_support)
        if f3_aa_min_support < 1:
            raise ValueError('F3 minimum support는 1 이상이어야 합니다.')
        self.f3_aa_min_support = int(f3_aa_min_support)
        if f4_min_support < 1:
            raise ValueError('F4 minimum support는 1 이상이어야 합니다.')
        self.f4_min_support = int(f4_min_support)
        if isinstance(f7_pairs, (str, bytes)):
            raise TypeError('f7_pairs는 두 클래스씩 묶은 시퀀스여야 합니다.')
        normalized_f7_pairs = []
        for pair in f7_pairs:
            if isinstance(pair, (str, bytes)) or len(pair) != 2:
                raise ValueError('각 F7 pair는 정확히 두 클래스여야 합니다.')
            first, second = (str(pair[0]), str(pair[1]))
            if first == second:
                raise ValueError('F7 pair의 두 클래스는 달라야 합니다.')
            normalized_f7_pairs.append((first, second))
        if not normalized_f7_pairs:
            raise ValueError('f7_pairs는 한 쌍 이상이어야 합니다.')
        if len(set(normalized_f7_pairs)) != len(normalized_f7_pairs):
            raise ValueError('f7_pairs에 중복된 pair가 있습니다.')
        if f7_top_k_per_direction < 1 or f7_min_gene_support < 1:
            raise ValueError('F7 top-K와 minimum support는 1 이상이어야 합니다.')
        if f7_laplace_alpha <= 0:
            raise ValueError('F7 Laplace alpha는 0보다 커야 합니다.')
        if f7_burden_quantiles < 2 or f7_stability_folds < 4:
            raise ValueError('F7 burden quantiles는 2 이상, stability folds는 4 이상이어야 합니다.')
        if not 4 <= f7_min_direction_consistency <= f7_stability_folds:
            raise ValueError('F7 direction consistency는 4 이상이어야 합니다.')
        if not 3 <= f7_min_selection_frequency <= f7_stability_folds:
            raise ValueError('F7 selection frequency는 3 이상이어야 합니다.')
        self.f7_pairs = tuple(normalized_f7_pairs)
        self.f7_top_k_per_direction = int(f7_top_k_per_direction)
        self.f7_min_gene_support = int(f7_min_gene_support)
        self.f7_laplace_alpha = float(f7_laplace_alpha)
        self.f7_burden_quantiles = int(f7_burden_quantiles)
        self.f7_stability_folds = int(f7_stability_folds)
        self.f7_min_direction_consistency = int(f7_min_direction_consistency)
        self.f7_min_selection_frequency = int(f7_min_selection_frequency)
        self.f7_random_state = int(f7_random_state)

    def fit(self, features: pd.DataFrame, labels: pd.Series | Sequence[str]) -> 'F10GlobalAACompositionNoRawPreprocessingPipeline':
        frame = self._validate_frame(features)
        label_values = self._validate_labels(frame, labels)
        self.gene_columns_ = frame.columns.tolist()
        self.label_encoder.fit(label_values)
        scan = self._scan(frame, stage='fit')
        self.active_gene_indices_, self.active_gene_columns_, self.dropped_constant_columns = select_active_genes(scan, self.gene_columns_)
        self.burden_clip_value_ = fit_burden_clip(scan, quantile=self.burden_clip_quantile)
        self.active_f2_indices_, self.f2_feature_names_ = select_active_gene_consequences(scan, self.gene_columns_)
        self.active_f3_position_indices_, self.f3_position_feature_names_ = select_active_gene_position_bins(scan, self.gene_columns_, minimum_support=self.f3_position_min_support)
        self.active_f3_aa_indices_, self.f3_aa_feature_names_ = select_active_gene_aa_transitions(scan, self.gene_columns_, minimum_support=self.f3_aa_min_support)
        self.f4_hotspot_vocabulary_, self.f4_feature_names_ = select_exact_hotspot_vocabulary(scan, self.gene_columns_, minimum_support=self.f4_min_support)
        self.f7_pair_contrast_model_, self._fit_f7_oof_matrix_ = fit_with_oof(scan.f0_all_genes, build_missing_gene_matrix(frame), label_values, self.gene_columns_, self.f7_pairs, oof_folds=self.f7_stability_folds, top_k_per_direction=self.f7_top_k_per_direction, minimum_pair_mutation_support=self.f7_min_gene_support, laplace_alpha=self.f7_laplace_alpha, burden_quantiles=self.f7_burden_quantiles, stability_folds=self.f7_stability_folds, minimum_direction_consistency=self.f7_min_direction_consistency, minimum_selection_frequency=self.f7_min_selection_frequency, random_state=self.f7_random_state)
        self.f7_feature_names_ = list(self.f7_pair_contrast_model_.feature_names)
        selected_arrays = [np.asarray(statistics.selected_gene_indices, dtype=np.int64) for statistics in self.f7_pair_contrast_model_.pair_statistics if statistics.selected_gene_indices]
        self.f7_selected_gene_indices_ = np.unique(np.concatenate(selected_arrays)) if selected_arrays else np.asarray([], dtype=np.int64)
        self.f7_selected_gene_catalog_ = build_f7_selected_gene_catalog(self.f7_pair_contrast_model_)
        names: list[str] = []
        names.extend((f'F0__gene__{gene}' for gene in self.active_gene_columns_))
        names.extend((f'F1__patient__{name}' for name in F1_COLUMNS))
        names.extend(self.f2_feature_names_)
        names.extend(self.f3_position_feature_names_)
        names.extend(self.f3_aa_feature_names_)
        names.extend(self.f4_feature_names_)
        names.extend(self.f7_feature_names_)
        names.extend(F9_AA_SUBSTITUTION_FEATURE_NAMES)
        names.extend(F10_AA_COMPOSITION_FEATURE_NAMES)
        self.feature_names_out_ = np.asarray(names, dtype=object)
        self._fit_scan_ = scan
        self._log(f'fit 완료 | pipeline={self.pipeline_name}, RAW={0:,}, F0={len(self.active_gene_columns_):,}, F1={len(F1_COLUMNS)}, F2={len(self.f2_feature_names_):,}, F3_POS={len(self.f3_position_feature_names_):,}, F3_AA={len(self.f3_aa_feature_names_):,}, F4={len(self.f4_feature_names_):,}, F7={len(self.f7_feature_names_):,}, F9={len(F9_AA_SUBSTITUTION_FEATURE_NAMES):,}, F10={len(F10_AA_COMPOSITION_FEATURE_NAMES):,}')
        return self

    def transform(self, features: pd.DataFrame):
        self._require_fitted()
        frame = self._validate_transform_frame(features)
        scan = self._scan(frame, stage='transform')
        return self._build_matrix(frame, scan=scan, use_fit_oof=False)

    def _build_matrix(self, frame: pd.DataFrame, *, scan: MutationScanResult, use_fit_oof: bool) -> sparse.csr_matrix:
        """고정 스키마로 행렬을 만들고 target-aware OOF 사용 여부를 제어합니다."""
        blocks: list[sparse.spmatrix] = [
            scan.f0_all_genes[:, self.active_gene_indices_],
            build_f1_matrix(scan, burden_clip_value=self.burden_clip_value_),
            scan.f2_all_gene_consequences[:, self.active_f2_indices_],
            scan.f3_all_gene_position_bins[:, self.active_f3_position_indices_],
            scan.f3_all_gene_aa_transitions[:, self.active_f3_aa_indices_],
            build_exact_hotspot_matrix(scan, self.f4_hotspot_vocabulary_),
        ]
        if use_fit_oof:
            if self._fit_f7_oof_matrix_ is None:
                raise RuntimeError('F7 OOF 행렬이 준비되지 않았습니다.')
            if self._fit_f7_oof_matrix_.shape[0] != len(frame):
                raise RuntimeError('F7 OOF 행렬과 학습 입력의 행 수가 다릅니다.')
            blocks.append(self._fit_f7_oof_matrix_)
        else:
            blocks.append(build_f7_matrix(scan.f0_all_genes, build_missing_gene_matrix(frame), self.f7_pair_contrast_model_))
        blocks.append(build_f9_global_aa_substitution_matrix(scan))
        blocks.append(build_f10_global_aa_composition_matrix(scan))
        output = sparse.hstack(blocks, format='csr', dtype=np.float32)
        if output.shape[1] != len(self.feature_names_out_):
            raise RuntimeError('생성된 피처 수와 고정 스키마가 일치하지 않습니다.')
        return output

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series | Sequence[str]):
        self.fit(features, labels)
        frame = self._validate_transform_frame(features)
        scan = self._fit_scan_
        self._log('학습 행렬의 target-aware 블록은 내부 OOF 통계만 사용합니다.')
        return self._build_matrix(frame, scan=scan, use_fit_oof=True)

    def get_feature_names_out(self) -> np.ndarray:
        self._require_fitted()
        return self.feature_names_out_.copy()

    def summary(self) -> dict[str, object]:
        self._require_fitted()
        return {
            'pipeline_name': self.pipeline_name,
            'feature_blocks': list(self.feature_blocks),
            'remaining_features': len(self.feature_names_out_),
            'raw_features': 0,
            'includes_raw_ordinal': False,
            'f0_features': len(self.active_gene_columns_),
            'f1_features': len(F1_COLUMNS),
            'f2_features': len(self.f2_feature_names_),
            'f3_position_features': len(self.f3_position_feature_names_),
            'f3_aa_features': len(self.f3_aa_feature_names_),
            'f4_exact_hotspot_features': len(self.f4_feature_names_),
            'f5_signature_features': 0,
            'f5_selected_gene_union': 0,
            'f7_pair_count': len(self.f7_pairs),
            'f7_pair_contrast_features': len(self.f7_feature_names_),
            'f7_selected_gene_union': len(self.f7_selected_gene_indices_),
            'f9_global_aa_pair_features': len(F9_AA_SUBSTITUTION_FEATURE_NAMES),
            'f9_global_aa_pair_scale': 'log1p_event_count',
            'f10_global_aa_composition_features': len(F10_AA_COMPOSITION_FEATURE_NAMES),
            'f10_global_aa_composition_scale': 'row_fraction',
            'f5_missing_policy': 'legacy_treat_as_wt',
            'dropped_constant_features': len(self.dropped_constant_columns),
            'burden_clip_value': self.burden_clip_value_,
        }

    def get_diagnostics(self) -> dict[str, object]:
        """검증 코드가 private 속성에 의존하지 않도록 학습 상태를 복사해 반환합니다."""
        self._require_fitted()
        return {
            'pipeline_name': self.pipeline_name,
            'capabilities': sorted(self.capabilities),
            'summary': self.summary(),
            'f5_signature_model': None,
            'f5_selected_gene_indices': np.asarray([], dtype=np.int64),
            'f5_output_column_indices': np.asarray([], dtype=np.int64),
            'f5_fit_oof_matrix': None,
            'f7_pair_contrast_model': self.f7_pair_contrast_model_,
            'f7_selected_gene_indices': self.f7_selected_gene_indices_.copy(),
            'f7_selected_gene_catalog': [
                dict(item) for item in self.f7_selected_gene_catalog_
            ],
            'f7_fit_oof_matrix': self._fit_f7_oof_matrix_,
        }

    @property
    def capabilities(self) -> frozenset[str]:
        """이 파이프라인이 제공하는 고정 피처 블록 집합입니다."""
        return frozenset(self.feature_blocks)

    def encode_labels(self, labels: pd.Series | Sequence[str]) -> np.ndarray:
        self._require_fitted()
        return self.label_encoder.transform(np.asarray(labels))

    def decode_labels(self, labels: Sequence[int]) -> np.ndarray:
        self._require_fitted()
        return self.label_encoder.inverse_transform(np.asarray(labels, dtype=int))

    def _scan(self, features: pd.DataFrame, *, stage: str) -> MutationScanResult:
        return scan_mutation_frame(features, stage=stage, show_progress=self.show_progress, progress_interval=self.progress_interval, collect_exact_events=True)

    @staticmethod
    def _validate_frame(features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise TypeError('features는 pandas DataFrame이어야 합니다.')
        if features.columns.has_duplicates:
            raise ValueError('중복된 유전자 열 이름이 있습니다.')
        if not features.index.is_unique:
            raise ValueError('features 인덱스는 고유해야 합니다.')
        return features

    @staticmethod
    def _validate_labels(frame: pd.DataFrame, labels: pd.Series | Sequence[str]) -> np.ndarray:
        if isinstance(labels, pd.Series):
            if not labels.index.is_unique:
                raise ValueError('labels 인덱스는 고유해야 합니다.')
            if not frame.index.equals(labels.index):
                raise ValueError('features와 labels 인덱스 및 순서가 일치해야 합니다.')
            values = labels.to_numpy()
        else:
            if isinstance(labels, (str, bytes)):
                raise TypeError('labels는 문자열 하나가 아니라 1차원 시퀀스여야 합니다.')
            values = np.asarray(labels)
        if values.ndim != 1:
            raise ValueError('labels는 1차원이어야 합니다.')
        if len(values) != len(frame):
            raise ValueError('features와 labels의 행 수가 일치하지 않습니다.')
        if pd.isna(values).any():
            raise ValueError('labels에 결측값이 있습니다.')
        return values

    def _validate_transform_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        frame = self._validate_frame(features)
        expected = list(self.gene_columns_)
        actual = list(frame.columns)
        expected_set = set(expected)
        actual_set = set(actual)
        missing = [column for column in expected if column not in actual_set]
        unexpected = [column for column in actual if column not in expected_set]
        if missing or unexpected:
            details = []
            if missing:
                details.append(f'누락={missing}')
            if unexpected:
                details.append(f'예상 밖={unexpected}')
            raise ValueError('유전자 피처 스키마가 다릅니다: ' + ', '.join(details))
        if actual != expected:
            raise ValueError('유전자 피처 열 순서가 학습 시점과 다릅니다.')
        return frame

    def _require_fitted(self) -> None:
        if not hasattr(self, 'feature_names_out_'):
            raise RuntimeError('fit을 먼저 실행해야 합니다.')

    def __getstate__(self) -> dict[str, object]:
        """추론에 불필요한 학습 행 단위 캐시를 artifact에서 제외합니다."""
        state = self.__dict__.copy()
        state.pop('_fit_frame_reference_', None)
        state['_fit_scan_'] = None
        state['_fit_f7_oof_matrix_'] = None
        return state

    def _log(self, message: str) -> None:
        if self.show_progress:
            print(f'[{self.pipeline_name}] {message}')
