# test_002_e8_lsvc_grid_C_0p05

| 항목 | 결과 |
| --- | --- |
| 실행 시작 | 2026-08-05T21:46:36+09:00 |
| 실행 종료 | 2026-08-05T21:52:03+09:00 |
| 총 실행 시간 | 0:05:27.644379 |
| 모델 | linear_svc |
| 전처리 파이프라인 | jh_v09 |
| 검증 | StratifiedGroupKFold 3-Fold × 1 seeds |
| 학습 데이터 행 수 | 6201 |
| Fold 피처 수 | 평균 4793.3, 범위 4776~4812 |
| OOF Macro F1 | 0.391315 ± nan |
| Fold Train Macro F1 평균 | 0.744606 |
| Fold Validation Macro F1 평균 | 0.390768 |
| 평균 generalization gap | +0.353838 |
| 최대 generalization gap | +0.382977 |
| 과적합 경고 Fold (gap > 0.10) | 3/3 |
| 설정 파일 | `data/processed/test_002_e8_lsvc_grid/generated_configs/C_0p05.yaml` |
| 제출 파일 | `data/processed/test_002_e8_lsvc_grid/C_0p05/submission_test_002_e8_lsvc_grid_C_0p05.csv` |
| 요약 JSON | `data/processed/test_002_e8_lsvc_grid/C_0p05/jh_v09_sgkf_summary.json` |

## Seed별 OOF

| seed | oof_macro_f1 |
| --- | --- |
| 42 | 0.391315 |

## Fold별 결과

| seed | fold | feature_count | train_macro_f1 | validation_macro_f1 | generalization_gap | converged | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0 | 4812 | 0.747042 | 0.364065 | 0.382977 | True | 102.576336 |
| 42 | 1 | 4776 | 0.742475 | 0.391600 | 0.350874 | True | 100.548790 |
| 42 | 2 | 4792 | 0.744302 | 0.416640 | 0.327662 | True | 102.060568 |

## 하이퍼파라미터

```yaml
name: linear_svc
class_weight: balanced
max_iter: 20000
tol: 0.0001
dual: auto
seed_plus_fold: true
C: 0.05
```

## 전처리 설정

```yaml
name: jh_v09
parser_version: jh_parser_v1
f4_min_support: 5
features:
  f0_gene_mutation_presence: true
  f1_sample_summary: true
  f2_gene_consequence: false
  f3_global_amino_acid_position: true
  f3_gene_aware_amino_acid_position: false
  f4_exact_hotspot: true
f3_position_bins:
- 1-50
- 51-100
- 101-200
- 201-400
- 401-800
- 801+
f1_scaling: fold_local_standard_scaler
tfidf_sublinear_tf: true
tfidf_norm: l2
```
