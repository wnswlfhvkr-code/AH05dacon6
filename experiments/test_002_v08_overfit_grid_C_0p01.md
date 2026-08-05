# test_002_v08_overfit_grid_C_0p01

| 항목 | 결과 |
| --- | --- |
| 실행 시작 | 2026-08-05T20:02:30+09:00 |
| 실행 종료 | 2026-08-05T20:14:49+09:00 |
| 총 실행 시간 | 0:12:18.882255 |
| 모델 | logistic_regression |
| 전처리 파이프라인 | jh_v08 |
| 검증 | StratifiedGroupKFold 3-Fold × 3 seeds |
| 학습 데이터 행 수 | 6201 |
| Fold 피처 수 | 평균 4787.1, 범위 4774~4812 |
| OOF Macro F1 | 0.401609 ± 0.005899 |
| Fold Train Macro F1 평균 | 0.579148 |
| Fold Validation Macro F1 평균 | 0.398882 |
| 평균 generalization gap | +0.180266 |
| 최대 generalization gap | +0.197281 |
| 과적합 경고 Fold (gap > 0.10) | 9/9 |
| 설정 파일 | `data/processed/test_002_v08_overfit_grid/generated_configs/C_0p01.yaml` |
| 제출 파일 | `data/processed/test_002_v08_overfit_grid/C_0p01/submission_test_002_v08_overfit_grid_C_0p01.csv` |
| 요약 JSON | `data/processed/test_002_v08_overfit_grid/C_0p01/jh_v08_sgkf_summary.json` |

## Seed별 OOF

| seed | oof_macro_f1 |
| --- | --- |
| 42 | 0.396212 |
| 2026 | 0.400709 |
| 777 | 0.407906 |

## Fold별 결과

| seed | fold | feature_count | train_macro_f1 | validation_macro_f1 | generalization_gap | converged | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0 | 4812 | 0.573991 | 0.378907 | 0.195084 | True | 79.479562 |
| 42 | 1 | 4776 | 0.581143 | 0.395319 | 0.185824 | True | 81.769384 |
| 42 | 2 | 4792 | 0.579591 | 0.412886 | 0.166705 | True | 79.970461 |
| 2026 | 0 | 4782 | 0.583406 | 0.397604 | 0.185802 | True | 79.609811 |
| 2026 | 1 | 4790 | 0.571311 | 0.400133 | 0.171178 | True | 78.099660 |
| 2026 | 2 | 4788 | 0.589031 | 0.391750 | 0.197281 | True | 79.217493 |
| 777 | 0 | 4783 | 0.579868 | 0.405868 | 0.174001 | True | 78.772760 |
| 777 | 1 | 4774 | 0.581352 | 0.408312 | 0.173040 | True | 78.426651 |
| 777 | 2 | 4787 | 0.572637 | 0.399161 | 0.173476 | True | 78.280652 |

## 하이퍼파라미터

```yaml
name: logistic_regression
C: 0.01
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
