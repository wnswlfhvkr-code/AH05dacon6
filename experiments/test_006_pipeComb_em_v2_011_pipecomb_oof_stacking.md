# test_006_pipeComb_em_v2_011

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-05T22:02:28+09:00 |
| 모델 | pipecomb_oof_stacking |
| 전처리 파이프라인 | pipeComb_em_v2 |
| 선택된 최소 변이 횟수 | - |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 204386 |
| 최종 Macro F1 | 0.551666 |
| 80% 학습 Macro F1 | 0.780691 |
| 20% 검증 Macro F1 | 0.551666 |
| 과적합 격차 | 0.229025 |
| 과적합 여부 | True |
| 설정 파일 | `configs/test_006_pipeComb_em_v2_011.yaml` |
| 제출 파일 | `data/processed/test_006_pipeComb_em_v2_011_pipecomb_oof_stacking_submission.csv` |
| 모델 아티팩트 | `models/test_006_pipeComb_em_v2_011.pkl` |

## 하이퍼파라미터

```yaml
name: pipecomb_oof_stacking
stacking_folds: 5
class_weight_mode: sqrt_balanced
class_weight_power: 0.25
class_weight_clip:
- 0.9
- 1.6
early_stopping_fraction: 0.2
text_temperature: 1.35
xgboost_temperature: 1.2
lightgbm_temperature: 1.2
text_linear_svc:
  C: 0.05
  class_weight: balanced
  max_iter: 20000
  tol: 0.0001
  dual: auto
emv46_xgboost:
  objective: multi:softprob
  n_estimators: 450
  learning_rate: 0.02
  max_depth: 3
  min_child_weight: 12.0
  subsample: 0.7
  colsample_bytree: 0.45
  gamma: 0.05
  reg_alpha: 2.5
  reg_lambda: 22.0
  max_delta_step: 1.0
  eval_metric: mlogloss
  tree_method: hist
  n_jobs: -1
  early_stopping_rounds: 40
emv46_lightgbm:
  objective: multiclass
  n_estimators: 360
  learning_rate: 0.02
  num_leaves: 9
  max_depth: 3
  min_child_samples: 60
  subsample: 0.7
  subsample_freq: 1
  colsample_bytree: 0.45
  min_split_gain: 0.02
  reg_alpha: 3.0
  reg_lambda: 25.0
  n_jobs: -1
  verbosity: -1
  early_stopping_rounds: 40
meta_learner:
  C: 0.015
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
  word_min_df: 4
  word_max_features: 160000
  char_ngram_range:
  - 3
  - 5
  char_min_df: 6
  char_max_features: 100000
  char_weight: 0.3
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
