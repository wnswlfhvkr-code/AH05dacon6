# test_006_pipeComb_em_v1_002

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-04T21:57:14+09:00 |
| 모델 | pipecomb_weighted_soft_voting |
| 전처리 파이프라인 | pipeComb_em_v1_001 |
| 선택된 최소 변이 횟수 | 5 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 248437 |
| 최종 Macro F1 | 0.355000 |
| 80% 학습 Macro F1 | 0.420078 |
| 20% 검증 Macro F1 | 0.346294 |
| 과적합 격차 | 0.073784 |
| 과적합 여부 | False |
| 설정 파일 | `configs/test_006_pipeComb_em_v1_002.yaml` |
| 제출 파일 | `data/processed/test_006_pipeComb_em_v1_002_pipecomb_weighted_soft_voting_submission.csv` |
| 모델 아티팩트 | `models/test_006_pipeComb_em_v1_002.pkl` |

## 하이퍼파라미터

```yaml
name: pipecomb_weighted_soft_voting
voting_weights:
  broad_ensemble: 0.35
  regularized_e14: 0.65
broad_ensemble:
  weights:
    linear_svc: 0.35
    xgboost: 0.4
    lightgbm: 0.25
  linear_temperature: 1.1
  linear_svc:
    C: 0.08
    class_weight: balanced
    max_iter: 20000
    tol: 0.0001
    dual: auto
  xgboost:
    n_estimators: 280
    learning_rate: 0.025
    max_depth: 2
    min_child_weight: 12.0
    subsample: 0.7
    colsample_bytree: 0.5
    reg_alpha: 2.0
    reg_lambda: 18.0
    eval_metric: mlogloss
    tree_method: hist
    n_jobs: -1
  lightgbm:
    objective: multiclass
    n_estimators: 240
    learning_rate: 0.025
    num_leaves: 11
    max_depth: 4
    min_child_samples: 45
    subsample: 0.7
    subsample_freq: 1
    colsample_bytree: 0.5
    reg_alpha: 2.0
    reg_lambda: 18.0
    n_jobs: -1
    verbosity: -1
  pair_specialists:
  - labels:
    - KIRC
    - KIPAN
    blend_weight: 0.05
    minimum_activation_probability: 0.7
    temperature: 1.8
    model:
      C: 0.05
  - labels:
    - LGG
    - GBMLGG
    blend_weight: 0.05
    minimum_activation_probability: 0.7
    temperature: 1.8
    model:
      C: 0.05
regularized_e14:
  n_estimators: 260
  learning_rate: 0.025
  max_depth: 2
  min_child_weight: 12.0
  subsample: 0.7
  colsample_bytree: 0.5
  reg_alpha: 2.0
  reg_lambda: 20.0
  eval_metric: mlogloss
  tree_method: hist
  n_jobs: -1
```

## 전처리 설정

```yaml
name: pipeComb_em_v1_001
min_mutation_count: 5
min_functional_mutation_count: 5
min_hotspot_count: 5
max_hotspots: 256
top_genes_per_class: 16
inner_signature_folds: 5
signature_random_state: 42
word_ngram_range:
- 1
- 2
word_min_df: 3
word_max_features: 180000
char_ngram_range:
- 3
- 5
char_min_df: 4
char_max_features: 120000
char_weight: 0.4
split_multi_event: false
e14_parameters:
  min_mutation_count: 8
  top_genes_per_class: 12
  min_hotspot_count: 6
  max_hotspots: 192
  inner_signature_folds: 5
  random_state: 42
  smoothing: 1.0
  max_log2_odds: 6.0
  shrinkage: 20.0
```
