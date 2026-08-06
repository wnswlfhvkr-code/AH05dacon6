# test_006_pipeComb_em_v4_009

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-06T08:49:51+09:00 |
| 모델 | pipecomb_oof_stacking |
| 전처리 파이프라인 | pipeComb_em_v4 |
| 선택된 최소 변이 횟수 | - |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 262794 |
| 최종 Macro F1 | 0.567189 |
| 80% 학습 Macro F1 | 0.794230 |
| 20% 검증 Macro F1 | 0.567189 |
| 과적합 격차 | 0.227041 |
| 과적합 여부 | True |
| 설정 파일 | `configs/test_006_pipeComb_em_v4_009.yaml` |
| 제출 파일 | `data/processed/test_006_pipeComb_em_v4_009_pipecomb_oof_stacking_submission.csv` |
| 모델 아티팩트 | `models/test_006_pipeComb_em_v4_009.pkl` |

## 하이퍼파라미터

```yaml
name: pipecomb_oof_stacking
stacking_folds: 5
class_weight_mode: sqrt_balanced
class_weight_power: 0.3
class_weight_clip:
- 0.9
- 1.7
early_stopping_fraction: 0.15
text_temperature: 1.25
xgboost_temperature: 1.2
lightgbm_temperature: 1.2
text_linear_svc:
  C: 0.06
  class_weight: balanced
  max_iter: 20000
  tol: 0.0001
  dual: auto
emv46_xgboost:
  objective: multi:softprob
  n_estimators: 400
  learning_rate: 0.02
  max_depth: 3
  min_child_weight: 10.0
  subsample: 0.65
  colsample_bytree: 0.5
  reg_alpha: 2.0
  reg_lambda: 20.0
  max_delta_step: 1.0
  eval_metric: mlogloss
  tree_method: hist
  n_jobs: -1
  early_stopping_rounds: 40
emv46_lightgbm:
  objective: multiclass
  n_estimators: 320
  learning_rate: 0.02
  num_leaves: 9
  max_depth: 4
  min_child_samples: 55
  subsample: 0.65
  subsample_freq: 1
  colsample_bytree: 0.45
  reg_alpha: 2.5
  reg_lambda: 22.0
  n_jobs: -1
  verbosity: -1
  early_stopping_rounds: 40
meta_learner:
  C: 0.018
  solver: lbfgs
  class_weight: balanced
  max_iter: 3000
```

## 전처리 설정

```yaml
name: pipeComb_em_v4
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
jh10_parameters:
  svd_components: 64
  svd_n_iter: 5
  svd_random_state: 42
tree_view_parameters:
  asymmetric_svd_routing: true
```
