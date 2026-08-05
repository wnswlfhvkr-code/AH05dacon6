# test_002_v08_overfit_grid_C_0p1

| 항목 | 결과 |
| --- | --- |
| 실행 시작 | 2026-08-05T19:36:10+09:00 |
| 실행 종료 | 2026-08-05T19:49:29+09:00 |
| 총 실행 시간 | 0:13:19.732842 |
| 모델 | logistic_regression |
| 전처리 파이프라인 | jh_v08 |
| 검증 | StratifiedGroupKFold 3-Fold × 3 seeds |
| 학습 데이터 행 수 | 6201 |
| Fold 피처 수 | 평균 4787.1, 범위 4774~4812 |
| OOF Macro F1 | 0.453399 ± 0.007268 |
| Fold Train Macro F1 평균 | 0.796432 |
| Fold Validation Macro F1 평균 | 0.451799 |
| 평균 generalization gap | +0.344633 |
| 최대 generalization gap | +0.361912 |
| 과적합 경고 Fold (gap > 0.10) | 9/9 |
| 설정 파일 | `data/processed/test_002_v08_overfit_grid/generated_configs/C_0p1.yaml` |
| 제출 파일 | `data/processed/test_002_v08_overfit_grid/C_0p1/submission_test_002_v08_overfit_grid_C_0p1.csv` |
| 요약 JSON | `data/processed/test_002_v08_overfit_grid/C_0p1/jh_v08_sgkf_summary.json` |

## Seed별 OOF

| seed | oof_macro_f1 |
| --- | --- |
| 42 | 0.448957 |
| 2026 | 0.449453 |
| 777 | 0.461786 |

## Fold별 결과

| seed | fold | feature_count | train_macro_f1 | validation_macro_f1 | generalization_gap | converged | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0 | 4812 | 0.797161 | 0.435248 | 0.361912 | True | 85.015922 |
| 42 | 1 | 4776 | 0.802180 | 0.453723 | 0.348458 | True | 85.347813 |
| 42 | 2 | 4792 | 0.787086 | 0.456209 | 0.330877 | True | 88.361295 |
| 2026 | 0 | 4782 | 0.804596 | 0.444410 | 0.360187 | True | 85.482700 |
| 2026 | 1 | 4790 | 0.794852 | 0.444629 | 0.350223 | True | 84.863856 |
| 2026 | 2 | 4788 | 0.794928 | 0.456288 | 0.338640 | True | 91.350788 |
| 777 | 0 | 4783 | 0.795147 | 0.457392 | 0.337755 | True | 86.816287 |
| 777 | 1 | 4774 | 0.788310 | 0.460266 | 0.328044 | True | 85.671576 |
| 777 | 2 | 4787 | 0.803630 | 0.458030 | 0.345600 | True | 84.443784 |

## 하이퍼파라미터

```yaml
name: logistic_regression
C: 0.1
class_weight: balanced
solver: saga
penalty: l2
max_iter: 3000
tol: 0.0001
n_jobs: -1
seed_plus_fold: true
```

## 전처리 설정

```yaml
name: jh_v08
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
```
