# test_006_E7

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-04T17:05:03+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | em_E7 |
| 선택된 최소 변이 횟수 | 5 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 4623 |
| 최종 Macro F1 | 0.401137 |
| 80% 학습 Macro F1 | 0.588549 |
| 20% 검증 Macro F1 | 0.398875 |
| 과적합 격차 | 0.189674 |
| 과적합 여부 | True |
| 설정 파일 | `configs/test_006_E7_h1.yaml` |
| 제출 파일 | `data/processed/test_006_E7_xgboost_submission.csv` |
| 모델 아티팩트 | `models/test_006_E7.pkl` |

## 하이퍼파라미터

```yaml
name: xgboost
objective: multi:softprob
n_estimators: 500
learning_rate: 0.03
max_depth: 3
min_child_weight: 5.0
subsample: 0.8
colsample_bytree: 0.65
reg_alpha: 1.0
reg_lambda: 10.0
early_stopping_rounds: 25
eval_metric: mlogloss
tree_method: hist
n_jobs: -1
```

## 전처리 설정

```yaml
name: em_E7
min_mutation_count: 5
top_genes_per_class: 20
smoothing: 0.5
max_log2_odds: 8.0
shrinkage: 10.0
min_hotspot_count: 5
max_hotspots: 384
inner_signature_folds: 5
random_state: 42
```
