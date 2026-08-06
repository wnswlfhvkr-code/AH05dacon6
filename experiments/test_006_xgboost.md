# test_006

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-03T23:41:35+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | em_v16 |
| 선택된 최소 변이 횟수 | 5 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 4616 |
| 최종 Macro F1 | 0.394757 |
| 80% 학습 Macro F1 | 0.539745 |
| 20% 검증 Macro F1 | 0.412001 |
| 과적합 격차 | 0.127743 |
| 과적합 여부 | True |
| 설정 파일 | `configs/test_006_m1.yaml` |
| 제출 파일 | `data/processed/test_006_xgboost_submission.csv` |
| 모델 아티팩트 | `models/test_006.pkl` |

## 하이퍼파라미터

```yaml
name: xgboost
n_estimators: 500
learning_rate: 0.03
max_depth: 3
min_child_weight: 5.0
subsample: 0.75
colsample_bytree: 0.6
reg_alpha: 0.5
reg_lambda: 10.0
early_stopping_rounds: 30
n_jobs: -1
eval_metric: mlogloss
tree_method: hist
```

## 전처리 설정

```yaml
name: em_v16
min_mutation_count: 5
top_genes_per_class: 20
smoothing: 0.5
max_log2_odds: 8.0
shrinkage: 10.0
min_hotspot_count: 5
max_hotspots: 384
inner_signature_folds: 5
```
