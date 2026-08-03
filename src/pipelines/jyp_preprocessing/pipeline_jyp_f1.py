"""jyp_f1 고정 JYP 전처리 파이프라인입니다.

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
RAW_MISSING_TOKEN = '__JYP_RAW_MISSING__'

class F1PreprocessingPipeline(PreprocessingPipeline):
    """F1PreprocessingPipeline의 단계 고정 전처리 구현입니다."""
    name = 'jyp_f1'
    artifact_schema_version = 4

    def __init__(self, *, burden_clip_quantile: float=0.99, show_progress: bool=True, progress_interval: int=25000) -> None:
        super().__init__()
        self.pipeline_name = self.name
        self.feature_blocks = ('raw', 'f0', 'f1')
        self.steps = self.feature_blocks
        if progress_interval < 1:
            raise ValueError('progress_interval은 1 이상이어야 합니다.')
        self.show_progress = bool(show_progress)
        self.progress_interval = int(progress_interval)
        self.label_encoder = LabelEncoder()
        self.raw_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.float32)
        if not 0.0 < burden_clip_quantile <= 1.0:
            raise ValueError('burden_clip_quantile은 0보다 크고 1 이하여야 합니다.')
        self.burden_clip_quantile = float(burden_clip_quantile)

    def fit(self, features: pd.DataFrame, labels: pd.Series | Sequence[str]) -> 'F1PreprocessingPipeline':
        frame = self._validate_frame(features)
        label_values = self._validate_labels(frame, labels)
        self._fit_frame_reference_ = frame
        self.gene_columns_ = frame.columns.tolist()
        self.label_encoder.fit(label_values)
        self.raw_encoder.fit(self._prepare_raw(frame))
        self.raw_wt_codes_ = np.asarray([float(np.flatnonzero(categories == 'WT')[0]) if np.any(categories == 'WT') else -2.0 for categories in self.raw_encoder.categories_], dtype=np.float32)
        scan = self._scan(frame, stage='fit')
        self.active_gene_indices_, self.active_gene_columns_, self.dropped_constant_columns = select_active_genes(scan, self.gene_columns_)
        self.burden_clip_value_ = fit_burden_clip(scan, quantile=self.burden_clip_quantile)
        self.active_f2_indices_ = np.asarray([], dtype=np.int64)
        self.f2_feature_names_ = []
        self.active_f3_position_indices_ = np.asarray([], dtype=np.int64)
        self.f3_position_feature_names_ = []
        self.active_f3_aa_indices_ = np.asarray([], dtype=np.int64)
        self.f3_aa_feature_names_ = []
        self.f4_hotspot_vocabulary_ = ()
        self.f4_feature_names_ = []
        self.f5_signature_model_ = None
        self._fit_f5_oof_matrix_ = None
        self.f5_feature_names_ = []
        self.f5_output_class_labels_ = ()
        self.f5_output_class_indices_ = np.asarray([], dtype=np.int64)
        self.f5_output_score_names_ = ()
        self.f5_output_column_indices_ = np.asarray([], dtype=np.int64)
        self.f5_selected_gene_indices_ = np.asarray([], dtype=np.int64)
        self.f7_pair_contrast_model_ = None
        self._fit_f7_oof_matrix_ = None
        self.f7_feature_names_ = []
        self.f7_selected_gene_indices_ = np.asarray([], dtype=np.int64)
        self.f7_selected_gene_catalog_ = []
        names: list[str] = []
        names.extend((f'RAW__gene__{gene}' for gene in self.gene_columns_))
        names.extend((f'F0__gene__{gene}' for gene in self.active_gene_columns_))
        names.extend((f'F1__patient__{name}' for name in F1_COLUMNS))
        self.feature_names_out_ = np.asarray(names, dtype=object)
        self._fit_scan_ = scan
        self._log(f'fit 완료 | pipeline={self.pipeline_name}, RAW={len(self.gene_columns_):,}, F0={len(self.active_gene_columns_):,}, F1={len(F1_COLUMNS)}, F2={len(self.f2_feature_names_):,}, F3_POS={len(self.f3_position_feature_names_):,}, F3_AA={len(self.f3_aa_feature_names_):,}, F4={len(self.f4_feature_names_):,}, F5={len(self.f5_feature_names_):,}, F7={len(self.f7_feature_names_):,}')
        return self

    def transform(self, features: pd.DataFrame):
        self._require_fitted()
        frame = self._validate_transform_frame(features)
        use_fit_oof = getattr(self, '_fit_frame_reference_', None) is features
        scan = self._fit_scan_ if use_fit_oof else self._scan(frame, stage='transform')
        return self._build_matrix(frame, scan=scan, use_fit_oof=use_fit_oof)

    def _build_matrix(self, frame: pd.DataFrame, *, scan: MutationScanResult | None, use_fit_oof: bool) -> sparse.csr_matrix:
        """고정 스키마로 행렬을 만들고 target-aware OOF 사용 여부를 제어합니다."""
        raw = self._build_raw_matrix(frame)
        if scan is None:
            raise RuntimeError('RAW 이후 단계에는 변이 스캔 결과가 필요합니다.')
        blocks: list[sparse.spmatrix] = []
        if raw is not None:
            blocks.append(raw)
        blocks.append(scan.f0_all_genes[:, self.active_gene_indices_])
        blocks.append(build_f1_matrix(scan, burden_clip_value=self.burden_clip_value_))
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
        result: dict[str, object] = {'pipeline_name': self.pipeline_name, 'feature_blocks': list(self.feature_blocks), 'remaining_features': len(self.feature_names_out_), 'raw_features': len(self.gene_columns_), 'includes_raw_ordinal': True, 'f0_features': len(self.active_gene_columns_), 'f1_features': len(F1_COLUMNS), 'f2_features': len(self.f2_feature_names_), 'f3_position_features': len(self.f3_position_feature_names_), 'f3_aa_features': len(self.f3_aa_feature_names_), 'f4_exact_hotspot_features': len(self.f4_feature_names_), 'f5_signature_features': len(self.f5_feature_names_), 'f5_selected_gene_union': len(self.f5_selected_gene_indices_), 'f7_pair_count': 0, 'f7_pair_contrast_features': len(self.f7_feature_names_), 'f7_selected_gene_union': len(self.f7_selected_gene_indices_), 'f5_missing_policy': 'legacy_treat_as_wt', 'dropped_constant_features': len(self.dropped_constant_columns), 'burden_clip_value': self.burden_clip_value_}
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

    def _scan(self, features: pd.DataFrame, *, stage: str) -> MutationScanResult:
        return scan_mutation_frame(features, stage=stage, show_progress=self.show_progress, progress_interval=self.progress_interval, collect_exact_events=False)

    @staticmethod
    def _prepare_raw(features: pd.DataFrame) -> pd.DataFrame:
        """도메인 가공 없이 결측만 명시한 exact-cell 범주를 보존합니다."""
        return features.fillna(RAW_MISSING_TOKEN).astype(str)

    def _build_raw_matrix(self, features: pd.DataFrame) -> sparse.csr_matrix:
        """exact 원문 범주를 유지하되 WT가 0이 되도록 이동해 희소 행렬로 만듭니다."""
        encoded = self.raw_encoder.transform(self._prepare_raw(features)).astype(np.float32, copy=False)
        encoded -= self.raw_wt_codes_[None, :]
        return sparse.csr_matrix(encoded, dtype=np.float32)

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
        """학습 DataFrame 참조를 artifact에 중복 저장하지 않습니다."""
        state = self.__dict__.copy()
        state['_fit_frame_reference_'] = None
        return state

    def _log(self, message: str) -> None:
        if self.show_progress:
            print(f'[{self.pipeline_name}] {message}')
