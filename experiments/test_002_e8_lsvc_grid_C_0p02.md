# test_002_e8_lsvc_grid_C_0p02

| 항목 | 결과 |
| --- | --- |
| 실행 시작 | 2026-08-05T21:52:05+09:00 |
| 실행 종료 | 2026-08-05T21:57:43+09:00 |
| 총 실행 시간 | 0:05:37.166959 |
| 모델 | linear_svc |
| 전처리 파이프라인 | jh_v09 |
| 검증 | StratifiedGroupKFold 3-Fold × 1 seeds |
| 학습 데이터 행 수 | 6201 |
| Fold 피처 수 | 평균 4793.3, 범위 4776~4812 |
| OOF Macro F1 | 0.326368 ± nan |
| Fold Train Macro F1 평균 | 0.581309 |
| Fold Validation Macro F1 평균 | 0.324882 |
| 평균 generalization gap | +0.256427 |
| 최대 generalization gap | +0.261957 |
| 과적합 경고 Fold (gap > 0.10) | 3/3 |
| 설정 파일 | `data/processed/test_002_e8_lsvc_grid/generated_configs/C_0p02.yaml` |
| 제출 파일 | `data/processed/test_002_e8_lsvc_grid/C_0p02/submission_test_002_e8_lsvc_grid_C_0p02.csv` |
| 요약 JSON | `data/processed/test_002_e8_lsvc_grid/C_0p02/jh_v09_sgkf_summary.json` |

## Seed별 OOF

| seed | oof_macro_f1 |
| --- | --- |
| 42 | 0.326368 |

## Fold별 결과

| seed | fold | feature_count | train_macro_f1 | validation_macro_f1 | generalization_gap | converged | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0 | 4812 | 0.579304 | 0.317347 | 0.261957 | True | 104.726694 |
| 42 | 1 | 4776 | 0.579396 | 0.320323 | 0.259073 | True | 106.171181 |
| 42 | 2 | 4792 | 0.585226 | 0.336975 | 0.248251 | True | 103.097792 |

## 하이퍼파라미터

```yaml
name: linear_svc
class_weight: balanced
max_iter: 20000
tol: 0.0001
dual: auto
seed_plus_fold: true
C: 0.02
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
