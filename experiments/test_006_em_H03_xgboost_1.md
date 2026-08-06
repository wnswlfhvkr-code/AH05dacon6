# test_006_em_H03

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-05T15:53:54+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | em_H03 |
| 선택된 최소 변이 횟수 | 5 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 1500 |
| 최종 Macro F1 | 0.432328 |
| 80% 학습 Macro F1 | 0.574172 |
| 20% 검증 Macro F1 | 0.423621 |
| 과적합 격차 | 0.150551 |
| 과적합 여부 | True |
| 설정 파일 | `configs/test_006_em_H03.yaml` |
| 제출 파일 | `data/processed/test_006_em_H03_xgboost_submission.csv` |
| 모델 아티팩트 | `models/test_006_em_H03.pkl` |

## 하이퍼파라미터

```yaml
name: xgboost
objective: multi:softprob
n_estimators: 450
learning_rate: 0.025
max_depth: 3
min_child_weight: 10.0
subsample: 0.7
colsample_bytree: 0.45
reg_alpha: 2.0
reg_lambda: 20.0
early_stopping_rounds: 30
eval_metric: mlogloss
tree_method: hist
n_jobs: -1
```

## 전처리 설정

```yaml
name: em_H03
min_mutation_count: 5
k_features: 1500
random_state: 42
```
