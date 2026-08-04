"""jyp_f5_selective_no_raw 고정 JYP 전처리 파이프라인입니다.

AGENTS.md의 버전 독립 규칙에 따라 이 파일이 사용하는 파싱, 피처 생성,
학습 및 변환 로직을 파일 안에 직접 보관합니다.
"""
from __future__ import annotations
import math
import re
from dataclasses import dataclass
from functools import lru_cache
import time
from typing import Sequence
import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from src.pipelines.base import PreprocessingPipeline
WT = 'WT'
MISSING = 'MISSING'
MISSING_TOKENS = {'', '.', '-', 'NA', 'N/A', 'NAN', 'NONE', 'NULL', MISSING}
CONSEQUENCE_TYPES = ('MISSENSE', 'SYNONYMOUS', 'STOP', 'FRAMESHIFT', 'COMPLEX', 'OTHER', 'UNPARSED')
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
POSITION_BIN_LABELS = ('pos_1_50', 'pos_51_100', 'pos_101_250', 'pos_251_500', 'pos_501_1000', 'pos_gt_1000')
AA_TRANSITIONS = tuple((f'{reference}>{alternate}' for reference in AMINO_ACIDS for alternate in f'{AMINO_ACIDS}*'))
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
    """F0~F4 계산에 함께 사용하는 타겟 독립 스캔 결과입니다."""
    f0_all_genes: sparse.csr_matrix
    f2_all_gene_consequences: sparse.csr_matrix
    f3_all_gene_position_bins: sparse.csr_matrix
    f3_all_gene_aa_transitions: sparse.csr_matrix
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
    f0_rows: list[int] = []
    f0_columns: list[int] = []
    f2_rows: list[int] = []
    f2_columns: list[int] = []
    f3_position_rows: list[int] = []
    f3_position_columns: list[int] = []
    f3_aa_rows: list[int] = []
    f3_aa_columns: list[int] = []
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
    return MutationScanResult(f0_all_genes=f0_all_genes, f2_all_gene_consequences=f2_all_gene_consequences, f3_all_gene_position_bins=f3_all_gene_position_bins, f3_all_gene_aa_transitions=f3_all_gene_aa_transitions, f4_exact_event_rows=np.asarray(f4_exact_event_rows, dtype=np.int32) if collect_exact_events else None, f4_exact_event_gene_indices=np.asarray(f4_exact_event_gene_indices, dtype=np.int32) if collect_exact_events else None, f4_exact_events=tuple(f4_exact_events) if collect_exact_events else None, mutated_gene_count=mutated_gene_count, unique_event_count=unique_event_count, multi_hit_gene_count=multi_hit_gene_count, missing_gene_count=missing_gene_count, consequence_counts=consequence_counts, qc={'rows': n_rows, 'genes': n_genes, 'candidate_cells': total_candidates, 'mutated_cells': len(f0_rows), 'exact_event_records': len(f4_exact_events), 'missing_cells': int(missing_gene_count.sum()), 'elapsed_seconds': round(time.perf_counter() - started, 3)})

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
F5_SCORE_NAMES = ('mutated_log_odds_sum', 'mutated_log_odds_mean', 'mutated_log_odds_max', 'bernoulli_log_likelihood')

@dataclass(frozen=True)
class F5SignatureModel:
    """Fold-Train에서 학습해 Validation/Test에 그대로 적용할 F5 통계입니다."""
    class_labels: tuple[str, ...]
    selected_gene_indices: tuple[np.ndarray, ...]
    mutation_log_odds: tuple[np.ndarray, ...]
    log_mutation_probabilities: tuple[np.ndarray, ...]
    log_wt_probabilities: tuple[np.ndarray, ...]
    direction_consistency: tuple[np.ndarray, ...]
    selection_frequency: tuple[np.ndarray, ...]
    effective_stability_folds: int

def build_missing_gene_matrix(features: pd.DataFrame) -> sparse.csr_matrix:
    """WT와 분리할 유전자별 결측 마스크를 생성합니다."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError('features는 pandas DataFrame이어야 합니다.')
    if features.columns.has_duplicates:
        raise ValueError('중복된 유전자 열 이름이 있습니다.')
    string_tokens = {variant for token in MISSING_TOKENS for variant in (token, token.lower(), token.title())}
    missing = features.isna().to_numpy() | features.isin(string_tokens).to_numpy()
    return sparse.csr_matrix(missing, dtype=np.float32)

def _normalize_missing_matrix(matrix: sparse.csr_matrix, missing_matrix: sparse.csr_matrix | None) -> sparse.csr_matrix | None:
    if missing_matrix is None:
        return None
    if missing_matrix.shape != matrix.shape:
        raise ValueError('F5 mutation matrix와 missing matrix의 shape이 일치하지 않습니다.')
    normalized = missing_matrix.tocsr().astype(np.float32, copy=True)
    normalized.sum_duplicates()
    if normalized.nnz:
        normalized.data.fill(1.0)
    if matrix.multiply(normalized).nnz:
        raise ValueError('같은 유전자 셀이 변이와 결측으로 동시에 표시됐습니다.')
    return normalized

def _validate_parameters(*, top_k_per_class: int, minimum_gene_support: int, laplace_alpha: float, stability_folds: int, minimum_direction_consistency: int, minimum_selection_frequency: int) -> None:
    if top_k_per_class < 1:
        raise ValueError('F5 top_k_per_class는 1 이상이어야 합니다.')
    if minimum_gene_support < 1:
        raise ValueError('F5 minimum_gene_support는 1 이상이어야 합니다.')
    if laplace_alpha <= 0:
        raise ValueError('F5 laplace_alpha는 0보다 커야 합니다.')
    if stability_folds < 2:
        raise ValueError('F5 stability_folds는 2 이상이어야 합니다.')
    if minimum_direction_consistency < 1:
        raise ValueError('F5 minimum_direction_consistency는 1 이상이어야 합니다.')
    if minimum_selection_frequency < 1:
        raise ValueError('F5 minimum_selection_frequency는 1 이상이어야 합니다.')

def _calculate_class_statistics(matrix: sparse.csr_matrix, labels: np.ndarray, *, missing_matrix: sparse.csr_matrix | None=None, class_count: int, laplace_alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """클래스별 one-vs-rest log-odds와 클래스 내 변이확률을 계산합니다."""
    total_support = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    total_observed = len(labels) - np.asarray(missing_matrix.sum(axis=0)).ravel().astype(np.float64) if missing_matrix is not None else None
    effects = np.zeros((class_count, matrix.shape[1]), dtype=np.float64)
    probabilities = np.zeros_like(effects)
    for class_index in range(class_count):
        class_mask = labels == class_index
        samples_in_class = int(class_mask.sum())
        samples_in_rest = len(labels) - samples_in_class
        if samples_in_class == 0 or samples_in_rest == 0:
            raise ValueError('F5 계산에는 각 클래스와 one-vs-rest 표본이 모두 필요합니다.')
        mutated_in_class = np.asarray(matrix[class_mask].sum(axis=0)).ravel()
        mutated_in_rest = total_support - mutated_in_class
        if missing_matrix is None:
            observed_in_class = samples_in_class
            observed_in_rest = samples_in_rest
        else:
            missing_in_class = np.asarray(missing_matrix[class_mask].sum(axis=0)).ravel()
            observed_in_class = samples_in_class - missing_in_class
            observed_in_rest = total_observed - observed_in_class
            if np.any(mutated_in_class > observed_in_class) or np.any(mutated_in_rest > observed_in_rest):
                raise ValueError('F5 변이 수가 관찰된 표본 수보다 큽니다.')
        class_probability = (mutated_in_class + laplace_alpha) / (observed_in_class + 2.0 * laplace_alpha)
        rest_probability = (mutated_in_rest + laplace_alpha) / (observed_in_rest + 2.0 * laplace_alpha)
        effects[class_index] = np.log(class_probability) - np.log1p(-class_probability) - np.log(rest_probability) + np.log1p(-rest_probability)
        probabilities[class_index] = class_probability
    return (effects, probabilities, total_support)

def _rank_top_genes(effect: np.ndarray, candidate_mask: np.ndarray, gene_names: np.ndarray, *, top_k: int) -> np.ndarray:
    """절대 효과크기 내림차순, 동점은 유전자명 사전순으로 고정합니다."""
    candidates = np.flatnonzero(candidate_mask)
    if not len(candidates):
        return np.asarray([], dtype=np.int64)
    order = np.lexsort((gene_names[candidates], -np.abs(effect[candidates])))
    return candidates[order[:top_k]].astype(np.int64, copy=False)

def _fit_signature_model(matrix: sparse.csr_matrix, labels: np.ndarray, gene_columns: Sequence[str], class_labels: Sequence[str], *, missing_matrix: sparse.csr_matrix | None, top_k_per_class: int, minimum_gene_support: int, laplace_alpha: float, stability_splits: Sequence[tuple[np.ndarray, np.ndarray]] | None, minimum_direction_consistency: int, minimum_selection_frequency: int, show_progress: bool) -> F5SignatureModel:
    class_count = len(class_labels)
    gene_names = np.asarray(gene_columns, dtype=str)
    effects, probabilities, support = _calculate_class_statistics(matrix, labels, missing_matrix=missing_matrix, class_count=class_count, laplace_alpha=laplace_alpha)
    eligible = support >= minimum_gene_support
    direction_counts = np.zeros_like(effects, dtype=np.int16)
    selection_counts = np.zeros_like(effects, dtype=np.int16)
    if stability_splits is not None:
        for fold_number, (train_indices, _) in enumerate(stability_splits, start=1):
            fold_effects, _, fold_support = _calculate_class_statistics(matrix[train_indices], labels[train_indices], missing_matrix=missing_matrix[train_indices] if missing_matrix is not None else None, class_count=class_count, laplace_alpha=laplace_alpha)
            fold_eligible = fold_support >= minimum_gene_support
            for class_index in range(class_count):
                direction_counts[class_index] += (np.sign(fold_effects[class_index]) == np.sign(effects[class_index])) & (np.sign(effects[class_index]) != 0)
                fold_top = _rank_top_genes(fold_effects[class_index], fold_eligible, gene_names, top_k=top_k_per_class)
                selection_counts[class_index, fold_top] += 1
            if show_progress:
                print(f'[f5_signature] 안정성 Fold {fold_number}/{len(stability_splits)} 완료')
    selected_gene_indices: list[np.ndarray] = []
    selected_log_odds: list[np.ndarray] = []
    selected_log_mutation: list[np.ndarray] = []
    selected_log_wt: list[np.ndarray] = []
    selected_direction: list[np.ndarray] = []
    selected_frequency: list[np.ndarray] = []
    effective_folds = len(stability_splits) if stability_splits is not None else 0
    required_direction = min(minimum_direction_consistency, effective_folds)
    required_frequency = min(minimum_selection_frequency, effective_folds)
    for class_index in range(class_count):
        candidate_mask = eligible.copy()
        if stability_splits is not None:
            candidate_mask &= direction_counts[class_index] >= required_direction
            candidate_mask &= selection_counts[class_index] >= required_frequency
        selected = _rank_top_genes(effects[class_index], candidate_mask, gene_names, top_k=top_k_per_class)
        class_probability = probabilities[class_index, selected]
        selected_gene_indices.append(selected)
        selected_log_odds.append(effects[class_index, selected].astype(np.float32))
        selected_log_mutation.append(np.log(class_probability).astype(np.float32))
        selected_log_wt.append(np.log1p(-class_probability).astype(np.float32))
        selected_direction.append(direction_counts[class_index, selected].copy())
        selected_frequency.append(selection_counts[class_index, selected].copy())
    return F5SignatureModel(class_labels=tuple((str(label) for label in class_labels)), selected_gene_indices=tuple(selected_gene_indices), mutation_log_odds=tuple(selected_log_odds), log_mutation_probabilities=tuple(selected_log_mutation), log_wt_probabilities=tuple(selected_log_wt), direction_consistency=tuple(selected_direction), selection_frequency=tuple(selected_frequency), effective_stability_folds=effective_folds)

def build_f5_signature_matrix(matrix: sparse.csr_matrix, model: F5SignatureModel, *, missing_matrix: sparse.csr_matrix | None=None) -> sparse.csr_matrix:
    """저장된 통계만 적용하며 결측 유전자는 WT likelihood에서 제외합니다."""
    missing_matrix = _normalize_missing_matrix(matrix, missing_matrix)
    columns: list[np.ndarray] = []
    row_count = matrix.shape[0]
    for gene_indices, log_odds, log_mutation, log_wt in zip(model.selected_gene_indices, model.mutation_log_odds, model.log_mutation_probabilities, model.log_wt_probabilities):
        if not len(gene_indices):
            columns.extend((np.zeros(row_count, dtype=np.float32) for _ in F5_SCORE_NAMES))
            continue
        mutated = matrix[:, gene_indices].toarray() > 0
        selected_missing = missing_matrix[:, gene_indices].toarray() > 0 if missing_matrix is not None else None
        if selected_missing is not None and np.any(mutated & selected_missing):
            raise ValueError('F5 선택 유전자에 변이·결측 상태 충돌이 있습니다.')
        mutated_count = mutated.sum(axis=1)
        weighted = np.where(mutated, log_odds[None, :], 0.0)
        score_sum = weighted.sum(axis=1)
        score_mean = np.divide(score_sum, mutated_count, out=np.zeros(row_count, dtype=np.float64), where=mutated_count > 0)
        max_candidates = np.where(mutated, log_odds[None, :], -np.inf)
        score_max = max_candidates.max(axis=1)
        score_max[mutated_count == 0] = 0.0
        bernoulli_terms = np.where(mutated, log_mutation[None, :], log_wt[None, :])
        if selected_missing is not None:
            bernoulli_terms = np.where(selected_missing, 0.0, bernoulli_terms)
        bernoulli = bernoulli_terms.sum(axis=1)
        columns.extend((score_sum, score_mean, score_max, bernoulli))
    output = np.column_stack(columns).astype(np.float32, copy=False)
    return sparse.csr_matrix(output, dtype=np.float32)

def build_f5_feature_names(class_labels: Sequence[str]) -> list[str]:
    """클래스 순서×4종 점수의 고정 피처명을 반환합니다."""
    return [f"F5__class_signature__{str(class_label).replace(' ', '_')}__{score_name}" for class_label in class_labels for score_name in F5_SCORE_NAMES]

def fit_f5_signature_features(matrix: sparse.csr_matrix, labels: np.ndarray, gene_columns: Sequence[str], class_labels: Sequence[str], *, missing_matrix: sparse.csr_matrix | None=None, top_k_per_class: int=25, minimum_gene_support: int=5, laplace_alpha: float=1.0, stability_folds: int=5, minimum_direction_consistency: int=4, minimum_selection_frequency: int=3, random_state: int=42, show_progress: bool=True) -> tuple[F5SignatureModel, sparse.csr_matrix]:
    """전체 Fold-Train 모델과 학습 행용 OOF signature를 함께 생성합니다."""
    _validate_parameters(top_k_per_class=top_k_per_class, minimum_gene_support=minimum_gene_support, laplace_alpha=laplace_alpha, stability_folds=stability_folds, minimum_direction_consistency=minimum_direction_consistency, minimum_selection_frequency=minimum_selection_frequency)
    encoded_labels = np.asarray(labels, dtype=np.int64)
    missing_matrix = _normalize_missing_matrix(matrix, missing_matrix)
    class_count = len(class_labels)
    if len(encoded_labels) != matrix.shape[0]:
        raise ValueError('F5 matrix와 labels의 행 수가 일치하지 않습니다.')
    class_sizes = np.bincount(encoded_labels, minlength=class_count)
    if np.any(class_sizes == 0):
        raise ValueError('F5 Fold-Train에 표본이 없는 클래스가 있습니다.')
    effective_folds = min(stability_folds, int(class_sizes.min()))
    if effective_folds < 2:
        raise ValueError('F5 OOF 생성을 위해 클래스마다 최소 2개 표본이 필요합니다.')
    splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
    splits = list(splitter.split(np.zeros(len(encoded_labels)), encoded_labels))
    full_model = _fit_signature_model(matrix, encoded_labels, gene_columns, class_labels, missing_matrix=missing_matrix, top_k_per_class=top_k_per_class, minimum_gene_support=minimum_gene_support, laplace_alpha=laplace_alpha, stability_splits=splits, minimum_direction_consistency=minimum_direction_consistency, minimum_selection_frequency=minimum_selection_frequency, show_progress=show_progress)
    oof = np.zeros((matrix.shape[0], class_count * len(F5_SCORE_NAMES)), dtype=np.float32)
    for fold_number, (train_indices, valid_indices) in enumerate(splits, start=1):
        fold_model = _fit_signature_model(matrix[train_indices], encoded_labels[train_indices], gene_columns, class_labels, missing_matrix=missing_matrix[train_indices] if missing_matrix is not None else None, top_k_per_class=top_k_per_class, minimum_gene_support=minimum_gene_support, laplace_alpha=laplace_alpha, stability_splits=None, minimum_direction_consistency=minimum_direction_consistency, minimum_selection_frequency=minimum_selection_frequency, show_progress=False)
        oof[valid_indices] = build_f5_signature_matrix(matrix[valid_indices], fold_model, missing_matrix=missing_matrix[valid_indices] if missing_matrix is not None else None).toarray()
        if show_progress:
            print(f'[f5_signature] OOF Fold {fold_number}/{effective_folds} 완료')
    return (full_model, sparse.csr_matrix(oof, dtype=np.float32))

class F5SelectiveNoRawPreprocessingPipeline(PreprocessingPipeline):
    """F5SelectiveNoRawPreprocessingPipeline의 단계 고정 전처리 구현입니다."""
    name = 'jyp_f5_selective_no_raw'
    artifact_schema_version = 4

    def __init__(
        self,
        *,
        burden_clip_quantile: float = 0.99,
        f3_position_min_support: int = 3,
        f3_aa_min_support: int = 3,
        f4_min_support: int = 5,
        f5_top_k_per_class: int = 10,
        f5_min_gene_support: int = 3,
        f5_laplace_alpha: float = 1.0,
        f5_stability_folds: int = 5,
        f5_min_direction_consistency: int = 3,
        f5_min_selection_frequency: int = 2,
        f5_random_state: int = 42,
        f5_output_rare_class_count: int | None = 10,
        f5_output_score_names: Sequence[str] | None = None,
        f5_output_class_labels: Sequence[str] | None = None,
        show_progress: bool = True,
        progress_interval: int = 25000,
    ) -> None:
        super().__init__()
        self.pipeline_name = self.name
        self.feature_blocks = ('f0', 'f1', 'f2', 'f3_position', 'f3_aa', 'f4', 'f5_selective', 'missing_mask')
        self.steps = self.feature_blocks
        if progress_interval < 1:
            raise ValueError('progress_interval은 1 이상이어야 합니다.')
        self.show_progress = bool(show_progress)
        self.progress_interval = int(progress_interval)
        self.label_encoder = LabelEncoder()
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
        if f5_top_k_per_class < 1 or f5_min_gene_support < 1:
            raise ValueError('F5 top-K와 minimum support는 1 이상이어야 합니다.')
        if f5_laplace_alpha <= 0:
            raise ValueError('F5 Laplace alpha는 0보다 커야 합니다.')
        if f5_stability_folds < 2:
            raise ValueError('F5 stability folds는 2 이상이어야 합니다.')
        if f5_min_direction_consistency < 1 or f5_min_selection_frequency < 1:
            raise ValueError('F5 안정성 기준은 1 이상이어야 합니다.')
        if f5_output_rare_class_count is not None and (isinstance(f5_output_rare_class_count, (bool, np.bool_)) or not isinstance(f5_output_rare_class_count, (int, np.integer)) or f5_output_rare_class_count < 1):
            raise ValueError('f5_output_rare_class_count는 None 또는 양의 정수여야 합니다.')
        if isinstance(f5_output_score_names, str):
            raise TypeError('f5_output_score_names는 문자열 시퀀스여야 합니다.')
        normalized_output_scores = None if f5_output_score_names is None else tuple(f5_output_score_names)
        if normalized_output_scores is not None:
            unknown_scores = set(normalized_output_scores) - set(F5_SCORE_NAMES)
            if unknown_scores:
                raise ValueError(f'지원하지 않는 F5 출력 점수가 있습니다: {sorted(unknown_scores)}')
            if len(set(normalized_output_scores)) != len(normalized_output_scores):
                raise ValueError('f5_output_score_names에 중복된 점수가 있습니다.')
            normalized_output_scores = tuple((score_name for score_name in F5_SCORE_NAMES if score_name in normalized_output_scores))
        if isinstance(f5_output_class_labels, str):
            raise TypeError('f5_output_class_labels는 클래스 라벨 시퀀스여야 합니다.')
        normalized_output_class_labels = None if f5_output_class_labels is None else tuple((str(label) for label in f5_output_class_labels))
        if normalized_output_class_labels is not None and len(set(normalized_output_class_labels)) != len(normalized_output_class_labels):
            raise ValueError('f5_output_class_labels에 중복된 클래스가 있습니다.')
        self.f5_top_k_per_class = int(f5_top_k_per_class)
        self.f5_min_gene_support = int(f5_min_gene_support)
        self.f5_laplace_alpha = float(f5_laplace_alpha)
        self.f5_stability_folds = int(f5_stability_folds)
        self.f5_min_direction_consistency = int(f5_min_direction_consistency)
        self.f5_min_selection_frequency = int(f5_min_selection_frequency)
        self.f5_random_state = int(f5_random_state)
        self.f5_output_rare_class_count = int(f5_output_rare_class_count) if f5_output_rare_class_count is not None else None
        self.f5_output_score_names = normalized_output_scores
        self.f5_output_class_labels = normalized_output_class_labels

    def fit(self, features: pd.DataFrame, labels: pd.Series | Sequence[str]) -> 'F5SelectiveNoRawPreprocessingPipeline':
        frame = self._validate_frame(features)
        label_values = self._validate_labels(frame, labels)
        self.gene_columns_ = frame.columns.tolist()
        self.label_encoder.fit(label_values)
        self.raw_wt_codes_ = np.asarray([], dtype=np.float32)
        scan = self._scan(frame, stage='fit')
        self.active_gene_indices_, self.active_gene_columns_, self.dropped_constant_columns = select_active_genes(scan, self.gene_columns_)
        self.burden_clip_value_ = fit_burden_clip(scan, quantile=self.burden_clip_quantile)
        self.active_f2_indices_, self.f2_feature_names_ = select_active_gene_consequences(scan, self.gene_columns_)
        self.active_f3_position_indices_, self.f3_position_feature_names_ = select_active_gene_position_bins(scan, self.gene_columns_, minimum_support=self.f3_position_min_support)
        self.active_f3_aa_indices_, self.f3_aa_feature_names_ = select_active_gene_aa_transitions(scan, self.gene_columns_, minimum_support=self.f3_aa_min_support)
        self.f4_hotspot_vocabulary_, self.f4_feature_names_ = select_exact_hotspot_vocabulary(scan, self.gene_columns_, minimum_support=self.f4_min_support)
        encoded_labels = self.label_encoder.transform(label_values)
        self.f5_signature_model_, full_f5_oof_matrix = fit_f5_signature_features(scan.f0_all_genes, encoded_labels, self.gene_columns_, self.label_encoder.classes_, missing_matrix=build_missing_gene_matrix(frame), top_k_per_class=self.f5_top_k_per_class, minimum_gene_support=self.f5_min_gene_support, laplace_alpha=self.f5_laplace_alpha, stability_folds=self.f5_stability_folds, minimum_direction_consistency=self.f5_min_direction_consistency, minimum_selection_frequency=self.f5_min_selection_frequency, random_state=self.f5_random_state, show_progress=self.show_progress)
        full_f5_feature_names = build_f5_feature_names(self.f5_signature_model_.class_labels)
        self._configure_f5_output(encoded_labels)
        self._fit_f5_oof_matrix_ = full_f5_oof_matrix[:, self.f5_output_column_indices_]
        self.f5_feature_names_ = [full_f5_feature_names[index] for index in self.f5_output_column_indices_]
        selected_arrays = [indices for class_index, indices in enumerate(self.f5_signature_model_.selected_gene_indices) if class_index in self.f5_output_class_indices_ if len(indices)]
        self.f5_selected_gene_indices_ = np.unique(np.concatenate(selected_arrays)) if selected_arrays else np.asarray([], dtype=np.int64)
        self.f7_pair_contrast_model_ = None
        self._fit_f7_oof_matrix_ = None
        self.f7_feature_names_ = []
        self.f7_selected_gene_indices_ = np.asarray([], dtype=np.int64)
        self.f7_selected_gene_catalog_ = []
        names: list[str] = []
        names.extend((f'F0__gene__{gene}' for gene in self.active_gene_columns_))
        names.extend((f'F1__patient__{name}' for name in F1_COLUMNS))
        names.extend(self.f2_feature_names_)
        names.extend(self.f3_position_feature_names_)
        names.extend(self.f3_aa_feature_names_)
        names.extend(self.f4_feature_names_)
        names.extend(self.f5_feature_names_)
        self.feature_names_out_ = np.asarray(names, dtype=object)
        self._fit_scan_ = scan
        self._log(f'fit 완료 | pipeline={self.pipeline_name}, RAW={0:,}, F0={len(self.active_gene_columns_):,}, F1={len(F1_COLUMNS)}, F2={len(self.f2_feature_names_):,}, F3_POS={len(self.f3_position_feature_names_):,}, F3_AA={len(self.f3_aa_feature_names_):,}, F4={len(self.f4_feature_names_):,}, F5={len(self.f5_feature_names_):,}, F7={len(self.f7_feature_names_):,}')
        return self

    def transform(self, features: pd.DataFrame):
        self._require_fitted()
        frame = self._validate_transform_frame(features)
        scan = self._scan(frame, stage='transform')
        return self._build_matrix(frame, scan=scan, use_fit_oof=False)

    def _build_matrix(self, frame: pd.DataFrame, *, scan: MutationScanResult | None, use_fit_oof: bool) -> sparse.csr_matrix:
        """고정 스키마로 행렬을 만들고 target-aware OOF 사용 여부를 제어합니다."""
        raw = None
        if scan is None:
            raise RuntimeError('RAW 이후 단계에는 변이 스캔 결과가 필요합니다.')
        blocks: list[sparse.spmatrix] = []
        if raw is not None:
            blocks.append(raw)
        blocks.append(scan.f0_all_genes[:, self.active_gene_indices_])
        blocks.append(build_f1_matrix(scan, burden_clip_value=self.burden_clip_value_))
        blocks.append(scan.f2_all_gene_consequences[:, self.active_f2_indices_])
        blocks.append(scan.f3_all_gene_position_bins[:, self.active_f3_position_indices_])
        blocks.append(scan.f3_all_gene_aa_transitions[:, self.active_f3_aa_indices_])
        blocks.append(build_exact_hotspot_matrix(scan, self.f4_hotspot_vocabulary_))
        if use_fit_oof:
            if self._fit_f5_oof_matrix_ is None:
                raise RuntimeError('F5 OOF 행렬이 준비되지 않았습니다.')
            if self._fit_f5_oof_matrix_.shape[0] != len(frame):
                raise RuntimeError('F5 OOF 행렬과 학습 입력의 행 수가 다릅니다.')
            blocks.append(self._fit_f5_oof_matrix_)
        else:
            f5_matrix = build_f5_signature_matrix(scan.f0_all_genes, self.f5_signature_model_, missing_matrix=build_missing_gene_matrix(frame))
            blocks.append(f5_matrix[:, self.f5_output_column_indices_])
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
        result: dict[str, object] = {'pipeline_name': self.pipeline_name, 'feature_blocks': list(self.feature_blocks), 'remaining_features': len(self.feature_names_out_), 'raw_features': 0, 'includes_raw_ordinal': False, 'f0_features': len(self.active_gene_columns_), 'f1_features': len(F1_COLUMNS), 'f2_features': len(self.f2_feature_names_), 'f3_position_features': len(self.f3_position_feature_names_), 'f3_aa_features': len(self.f3_aa_feature_names_), 'f4_exact_hotspot_features': len(self.f4_feature_names_), 'f5_signature_features': len(self.f5_feature_names_), 'f5_selected_gene_union': len(self.f5_selected_gene_indices_), 'f7_pair_count': 0, 'f7_pair_contrast_features': len(self.f7_feature_names_), 'f7_selected_gene_union': len(self.f7_selected_gene_indices_), 'f5_missing_policy': 'exclude_from_probability_and_wt_likelihood', 'dropped_constant_features': len(self.dropped_constant_columns), 'burden_clip_value': self.burden_clip_value_}
        result.update({'f5_output_class_count': len(self.f5_output_class_labels_), 'f5_output_score_count': len(self.f5_output_score_names_)})
        f6_model = getattr(self, 'f6_compression_model_', None)
        if f6_model is not None:
            result.update({'f6_min_document_frequency': f6_model.minimum_document_frequency, 'f6_selected_token_features': len(f6_model.selected_token_indices), 'f6_requested_components': f6_model.requested_components, 'f6_actual_components': f6_model.actual_components, 'f6_explained_variance_ratio_sum': f6_model.explained_variance_ratio_sum})
        return result

    def get_diagnostics(self) -> dict[str, object]:
        """검증 코드가 private 속성에 의존하지 않도록 학습 상태를 복사해 반환합니다."""
        self._require_fitted()
        f5_model = getattr(self, 'f5_signature_model_', None)
        return {'pipeline_name': self.pipeline_name, 'capabilities': sorted(self.capabilities), 'summary': self.summary(), 'f5_signature_model': f5_model, 'f5_selected_gene_indices': self.f5_selected_gene_indices_.copy(), 'f5_output_column_indices': self.f5_output_column_indices_.copy(), 'f5_fit_oof_matrix': self._fit_f5_oof_matrix_, 'f7_pair_contrast_model': getattr(self, 'f7_pair_contrast_model_', None), 'f7_selected_gene_indices': self.f7_selected_gene_indices_.copy(), 'f7_selected_gene_catalog': [dict(item) for item in self.f7_selected_gene_catalog_], 'f7_fit_oof_matrix': self._fit_f7_oof_matrix_}

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

    def _configure_f5_output(self, encoded_labels: np.ndarray) -> None:
        """F5 전체 출력에서 노출할 class-major/score-minor 열을 고정합니다."""
        class_labels = self.f5_signature_model_.class_labels
        if self.f5_output_class_labels is not None:
            requested_labels = set(self.f5_output_class_labels)
            unknown_labels = requested_labels - set(class_labels)
            if unknown_labels:
                raise ValueError(f'학습 데이터에 없는 F5 출력 클래스가 있습니다: {sorted(unknown_labels)}')
            selected_class_indices = [class_index for class_index, class_label in enumerate(class_labels) if class_label in requested_labels]
        elif self.f5_output_rare_class_count is not None:
            if self.f5_output_rare_class_count > len(class_labels):
                raise ValueError('f5_output_rare_class_count가 학습 클래스 수보다 큽니다.')
            class_sizes = np.bincount(encoded_labels, minlength=len(class_labels))
            rare_rank = sorted(range(len(class_labels)), key=lambda class_index: (int(class_sizes[class_index]), class_labels[class_index]))
            selected_class_set = set(rare_rank[:self.f5_output_rare_class_count])
            selected_class_indices = [class_index for class_index in range(len(class_labels)) if class_index in selected_class_set]
        else:
            selected_class_indices = list(range(len(class_labels)))
        selected_score_names = self.f5_output_score_names if self.f5_output_score_names is not None else F5_SCORE_NAMES
        score_indices = [F5_SCORE_NAMES.index(name) for name in selected_score_names]
        self.f5_output_class_labels_ = tuple((class_labels[index] for index in selected_class_indices))
        self.f5_output_class_indices_ = np.asarray(selected_class_indices, dtype=np.int64)
        self.f5_output_score_names_ = tuple(selected_score_names)
        self.f5_output_column_indices_ = np.asarray([class_index * len(F5_SCORE_NAMES) + score_index for class_index in selected_class_indices for score_index in score_indices], dtype=np.int64)

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
        state['_fit_f5_oof_matrix_'] = None
        return state

    def _log(self, message: str) -> None:
        if self.show_progress:
            print(f'[{self.pipeline_name}] {message}')
