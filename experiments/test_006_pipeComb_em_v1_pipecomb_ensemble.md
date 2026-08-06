# test_006_pipeComb_em_v1

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-04T19:41:28+09:00 |
| 모델 | pipecomb_ensemble |
| 전처리 파이프라인 | pipeComb_em_v1 |
| 선택된 최소 변이 횟수 | 2 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 434002 |
| 최종 Macro F1 | 0.427222 |
| 80% 학습 Macro F1 | 0.649330 |
| 20% 검증 Macro F1 | 0.410967 |
| 과적합 격차 | 0.238364 |
| 과적합 여부 | True |
| 설정 파일 | `configs/test_006_pipeComb_em_v1.yaml` |
| 제출 파일 | `data/processed/test_006_pipeComb_em_v1_pipecomb_ensemble_submission.csv` |
| 모델 아티팩트 | `models/test_006_pipeComb_em_v1.pkl` |

## 하이퍼파라미터

```yaml
name: pipecomb_ensemble
weights:
  linear_svc: 0.5
  xgboost: 0.3
  lightgbm: 0.2
linear_temperature: 0.8
linear_svc:
  C: 0.15
  class_weight: balanced
  max_iter: 20000
  tol: 0.0001
  dual: auto
xgboost:
  n_estimators: 350
  learning_rate: 0.03
  max_depth: 3
  min_child_weight: 8.0
  subsample: 0.75
  colsample_bytree: 0.55
  reg_alpha: 1.0
  reg_lambda: 12.0
  eval_metric: mlogloss
  tree_method: hist
  n_jobs: -1
lightgbm:
  objective: multiclass
  n_estimators: 300
  learning_rate: 0.03
  num_leaves: 15
  max_depth: 5
  min_child_samples: 35
  subsample: 0.75
  subsample_freq: 1
  colsample_bytree: 0.55
  reg_alpha: 1.0
  reg_lambda: 12.0
  n_jobs: -1
  verbosity: -1
pair_specialists:
- labels:
  - KIRC
  - KIPAN
  blend_weight: 0.15
  minimum_activation_probability: 0.55
  temperature: 1.5
  model:
    C: 0.08
- labels:
  - LGG
  - GBMLGG
  blend_weight: 0.15
  minimum_activation_probability: 0.55
  temperature: 1.5
  model:
    C: 0.08
```

## 전처리 설정

```yaml
name: pipeComb_em_v1
min_mutation_count: 2
min_functional_mutation_count: 2
min_hotspot_count: 3
max_hotspots: 300
top_genes_per_class: 20
inner_signature_folds: 5
signature_random_state: 42
word_ngram_range:
- 1
- 2
word_min_df: 2
word_max_features: 250000
char_ngram_range:
- 3
- 5
char_min_df: 3
char_max_features: 180000
char_weight: 0.5
split_multi_event: false
```
