# test_006_pipeComb_em_v3_011

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-06T11:11:03+09:00 |
| 모델 | pipecomb_tabpfn_subset_stacking |
| 전처리 파이프라인 | pipeComb_em_v3 |
| 선택된 최소 변이 횟수 | - |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 258591 |
| 최종 Macro F1 | 0.570758 |
| 80% 학습 Macro F1 | 0.808047 |
| 20% 검증 Macro F1 | 0.570758 |
| 과적합 격차 | 0.237289 |
| 과적합 여부 | True |
| 설정 파일 | `configs/test_006_pipeComb_em_v3_011.yaml` |
| 제출 파일 | `data/processed/test_006_pipeComb_em_v3_011_pipecomb_tabpfn_subset_stacking_submission.csv` |
| 모델 아티팩트 | `models/test_006_pipeComb_em_v3_011.pkl` |

## 하이퍼파라미터

```yaml
name: pipecomb_tabpfn_subset_stacking
stacking_folds: 5
class_weight_mode: sqrt_balanced
class_weight_power: 0.35
class_weight_clip:
- 0.85
- 1.8
early_stopping_fraction: 0.15
text_temperature: 1.2
xgboost_temperature: 1.15
lightgbm_temperature: 1.15
text_linear_svc:
  C: 0.07
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
  C: 0.015
  solver: lbfgs
  class_weight: balanced
  max_iter: 3000
subset_selection_folds: 5
subset_score_tolerance: 0.001
minimum_selected_models: 1
tabpfn:
  api_key_name: tabpfn_api_key
  env_file: .env
  model_path: auto
  max_features: 512
  n_estimators: 4
  softmax_temperature: 1.0
  temperature: 1.1
  balance_probabilities: false
  ignore_pretraining_limits: false
  thinking_mode: false
```

## 전처리 설정

```yaml
name: pipeComb_em_v3
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
jyp9_parameters:
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
  f7_oof_folds: 5
  f7_random_state: 42
  f9_min_transition_support: 5
  f9_max_transitions: 128
  f9_burden_normalize: true
correlation_filter_parameters:
  enabled: true
  correlation_method: pearson
  correlation_threshold: 0.98
  correlation_max_features: 5000
  correlation_block_size: 256
  min_active_count: 5
  active_epsilon: 1.0e-08
```
