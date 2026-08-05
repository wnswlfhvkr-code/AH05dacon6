# test_002_v08_overfit_grid_C_0p3

| 항목 | 결과 |
| --- | --- |
| 실행 시작 | 2026-08-05T19:21:36+09:00 |
| 실행 종료 | 2026-08-05T19:36:07+09:00 |
| 총 실행 시간 | 0:14:31.364980 |
| 모델 | logistic_regression |
| 전처리 파이프라인 | jh_v08 |
| 검증 | StratifiedGroupKFold 3-Fold × 3 seeds |
| 학습 데이터 행 수 | 6201 |
| Fold 피처 수 | 평균 4787.1, 범위 4774~4812 |
| OOF Macro F1 | 0.447685 ± 0.009197 |
| Fold Train Macro F1 평균 | 0.862549 |
| Fold Validation Macro F1 평균 | 0.446148 |
| 평균 generalization gap | +0.416401 |
| 최대 generalization gap | +0.440295 |
| 과적합 경고 Fold (gap > 0.10) | 9/9 |
| 설정 파일 | `data/processed/test_002_v08_overfit_grid/generated_configs/C_0p3.yaml` |
| 제출 파일 | `data/processed/test_002_v08_overfit_grid/C_0p3/submission_test_002_v08_overfit_grid_C_0p3.csv` |
| 요약 JSON | `data/processed/test_002_v08_overfit_grid/C_0p3/jh_v08_sgkf_summary.json` |

## Seed별 OOF

| seed | oof_macro_f1 |
| --- | --- |
| 42 | 0.445081 |
| 2026 | 0.440072 |
| 777 | 0.457903 |

## Fold별 결과

| seed | fold | feature_count | train_macro_f1 | validation_macro_f1 | generalization_gap | converged | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0 | 4812 | 0.872154 | 0.431859 | 0.440295 | True | 93.458426 |
| 42 | 1 | 4776 | 0.864263 | 0.452769 | 0.411494 | True | 94.070685 |
| 42 | 2 | 4792 | 0.855420 | 0.450135 | 0.405285 | True | 97.400828 |
| 2026 | 0 | 4782 | 0.858782 | 0.434296 | 0.424486 | True | 92.383553 |
| 2026 | 1 | 4790 | 0.870766 | 0.435758 | 0.435008 | True | 93.098546 |
| 2026 | 2 | 4788 | 0.858387 | 0.447854 | 0.410534 | True | 99.519635 |
| 777 | 0 | 4783 | 0.857727 | 0.455182 | 0.402546 | True | 93.825351 |
| 777 | 1 | 4774 | 0.859538 | 0.455995 | 0.403544 | True | 93.048130 |
| 777 | 2 | 4787 | 0.865899 | 0.451487 | 0.414413 | True | 91.391225 |

## 하이퍼파라미터

```yaml
name: logistic_regression
C: 0.3
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
