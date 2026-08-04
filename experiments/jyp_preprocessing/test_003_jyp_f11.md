# test_003_jyp_f11

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-03T16:41:06+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | jyp_f11 |
| 선택된 최소 변이 횟수 | - |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 55013 |
| 검증 Macro F1 | 0.425416 |
| 설정 파일 | `configs\test_003.yaml` |
| 제출 파일 | `data\processed\test_003_jyp_f11_submission.csv` |
| 모델 아티팩트 | `models\test_003_jyp_f11.pkl` |

## 하이퍼파라미터

```yaml
name: xgboost
n_estimators: 100
learning_rate: 0.1
max_depth: 6
n_jobs: -1
eval_metric: mlogloss
tree_method: hist
```

## 전처리 설정

```yaml
name: jyp_f11
burden_clip_quantile: 0.99
f3_position_min_support: 3
f3_aa_min_support: 3
f4_min_support: 5
f7_pairs:
- - KIRC
  - KIPAN
- - LGG
  - GBMLGG
f7_top_k_per_direction: 3
f7_min_gene_support: 10
f7_laplace_alpha: 4.0
f7_burden_quantiles: 5
f7_stability_folds: 5
f7_min_direction_consistency: 4
f7_min_selection_frequency: 3
f7_random_state: 42
show_progress: true
progress_interval: 25000
```
