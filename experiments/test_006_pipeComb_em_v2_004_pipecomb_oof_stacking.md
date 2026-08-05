# test_006_pipeComb_em_v2_004

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-05T19:15:50+09:00 |
| 모델 | pipecomb_oof_stacking |
| 전처리 파이프라인 | pipeComb_em_v2 |
| 선택된 최소 변이 횟수 | - |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 262598 |
| 최종 Macro F1 | 0.559778 |
| 80% 학습 Macro F1 | 0.827595 |
| 20% 검증 Macro F1 | 0.559778 |
| 과적합 격차 | 0.267817 |
| 과적합 여부 | True |
| 설정 파일 | `configs/test_006_pipeComb_em_v2_004.yaml` |
| 제출 파일 | `data/processed/test_006_pipeComb_em_v2_004_pipecomb_oof_stacking_submission.csv` |
| 모델 아티팩트 | `models/test_006_pipeComb_em_v2_004.pkl` |

## 하이퍼파라미터

```yaml
name: pipecomb_oof_stacking
stacking_folds: 5
class_weight_mode: sqrt_balanced
class_weight_clip:
- 0.75
- 2.5
early_stopping_fraction: 0.15
text_temperature: 1.2
xgboost_temperature: 1.15
lightgbm_temperature: 1.15
text_linear_svc:
  C: 0.1
  class_weight: balanced
  max_iter: 20000
  tol: 0.0001
  dual: auto
emv46_xgboost:
  objective: multi:softprob
  n_estimators: 350
  learning_rate: 0.025
  max_depth: 3
  min_child_weight: 8.0
  subsample: 0.75
  colsample_bytree: 0.55
  reg_alpha: 1.5
  reg_lambda: 15.0
  max_delta_step: 1.0
  eval_metric: mlogloss
  tree_method: hist
  n_jobs: -1
  early_stopping_rounds: 30
emv46_lightgbm:
  objective: multiclass
  n_estimators: 280
  learning_rate: 0.025
  num_leaves: 11
  max_depth: 4
  min_child_samples: 45
  subsample: 0.75
  subsample_freq: 1
  colsample_bytree: 0.5
  reg_alpha: 2.0
  reg_lambda: 18.0
  n_jobs: -1
  verbosity: -1
  early_stopping_rounds: 30
meta_learner:
  C: 0.03
  solver: lbfgs
  class_weight: balanced
  max_iter: 3000
```

## 전처리 설정

```yaml
name: pipeComb_em_v2
text_parameters:
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
emv46_parameters:
  min_mutation_count: 5
  min_functional_mutation_count: 5
  min_feature_support: 2
  top_genes_per_class: 20
  smoothing: 0.5
  max_log2_odds: 8.0
  shrinkage: 10.0
  min_hotspot_count: 5
  max_hotspots: 384
  inner_signature_folds: 5
  signature_temperature: 1.0
  signature_random_state: 42
  random_state: 42
  feature_parameters:
    F05:
      min_hotspot_count: 5
      max_hotspots: 384
    F10:
      min_feature_support: 2
    F12:
      min_hotspot_count: 5
      max_hotspots: 384
    F16:
      top_genes_per_class: 20
    F17:
      top_genes_per_class: 20
    F18:
      top_genes_per_class: 20
    F19:
      top_genes_per_class: 20
```
