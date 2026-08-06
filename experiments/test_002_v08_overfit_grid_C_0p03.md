# test_002_v08_overfit_grid_C_0p03

| 항목 | 결과 |
| --- | --- |
| 실행 시작 | 2026-08-05T19:49:32+09:00 |
| 실행 종료 | 2026-08-05T20:02:28+09:00 |
| 총 실행 시간 | 0:12:55.974644 |
| 모델 | logistic_regression |
| 전처리 파이프라인 | jh_v08 |
| 검증 | StratifiedGroupKFold 3-Fold × 3 seeds |
| 학습 데이터 행 수 | 6201 |
| Fold 피처 수 | 평균 4787.1, 범위 4774~4812 |
| OOF Macro F1 | 0.442654 ± 0.007709 |
| Fold Train Macro F1 평균 | 0.686180 |
| Fold Validation Macro F1 평균 | 0.440443 |
| 평균 generalization gap | +0.245738 |
| 최대 generalization gap | +0.268327 |
| 과적합 경고 Fold (gap > 0.10) | 9/9 |
| 설정 파일 | `data/processed/test_002_v08_overfit_grid/generated_configs/C_0p03.yaml` |
| 제출 파일 | `data/processed/test_002_v08_overfit_grid/C_0p03/submission_test_002_v08_overfit_grid_C_0p03.csv` |
| 요약 JSON | `data/processed/test_002_v08_overfit_grid/C_0p03/jh_v08_sgkf_summary.json` |

## Seed별 OOF

| seed | oof_macro_f1 |
| --- | --- |
| 42 | 0.436795 |
| 2026 | 0.439780 |
| 777 | 0.451387 |

## Fold별 결과

| seed | fold | feature_count | train_macro_f1 | validation_macro_f1 | generalization_gap | converged | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0 | 4812 | 0.687587 | 0.419260 | 0.268327 | True | 81.213670 |
| 42 | 1 | 4776 | 0.686716 | 0.443957 | 0.242760 | True | 81.583511 |
| 42 | 2 | 4792 | 0.683465 | 0.444127 | 0.239338 | True | 84.014579 |
| 2026 | 0 | 4782 | 0.691258 | 0.438699 | 0.252560 | True | 81.918286 |
| 2026 | 1 | 4790 | 0.684368 | 0.439477 | 0.244891 | True | 81.229458 |
| 2026 | 2 | 4788 | 0.686318 | 0.436815 | 0.249503 | True | 83.569786 |
| 777 | 0 | 4783 | 0.686079 | 0.439787 | 0.246292 | True | 84.737538 |
| 777 | 1 | 4774 | 0.683016 | 0.460089 | 0.222927 | True | 87.184596 |
| 777 | 2 | 4787 | 0.686814 | 0.441773 | 0.245041 | True | 87.772552 |

## 하이퍼파라미터

```yaml
name: logistic_regression
C: 0.03
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
