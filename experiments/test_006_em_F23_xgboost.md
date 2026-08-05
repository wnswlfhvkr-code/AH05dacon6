# test_006_em_F23

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-05T08:30:25+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | em_F20 |
| 선택된 최소 변이 횟수 | 5 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 23059 |
| 최종 Macro F1 | 0.460325 |
| 80% 학습 Macro F1 | 0.659748 |
| 20% 검증 Macro F1 | 0.434517 |
| 과적합 격차 | 0.225231 |
| 과적합 여부 | True |
| 설정 파일 | `configs/test_006_em_F23.yaml` |
| 제출 파일 | `data/processed/test_006_em_F23_xgboost_submission.csv` |
| 모델 아티팩트 | `models/test_006_em_F23.pkl` |

## 하이퍼파라미터

```yaml
name: xgboost
objective: multi:softprob
n_estimators: 500
learning_rate: 0.03
max_depth: 3
min_child_weight: 7.0
subsample: 0.7
colsample_bytree: 0.5
reg_alpha: 1.5
reg_lambda: 15.0
early_stopping_rounds: 30
eval_metric: mlogloss
tree_method: hist
n_jobs: -1
```

## 전처리 설정

```yaml
name: em_F20
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
