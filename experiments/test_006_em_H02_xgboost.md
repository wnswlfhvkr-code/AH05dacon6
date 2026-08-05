# test_006_em_H02

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-05T18:10:22+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | em_H02 |
| 선택된 최소 변이 횟수 | 5 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 1500 |
| 최종 Macro F1 | 0.438885 |
| 80% 학습 Macro F1 | 0.596963 |
| 20% 검증 Macro F1 | 0.452294 |
| 과적합 격차 | 0.144670 |
| 과적합 여부 | True |
| 설정 파일 | `configs/test_006_em_H02.yaml` |
| 제출 파일 | `data/processed/test_006_em_H02_xgboost_submission.csv` |
| 모델 아티팩트 | `models/test_006_em_H02.pkl` |

## 하이퍼파라미터

```yaml
name: xgboost
objective: multi:softprob
n_estimators: 500
learning_rate: 0.03
max_depth: 3
min_child_weight: 7.0
subsample: 0.75
colsample_bytree: 0.5
reg_alpha: 1.5
reg_lambda: 17.0
early_stopping_rounds: 30
eval_metric: mlogloss
tree_method: hist
n_jobs: -1
```

## 전처리 설정

```yaml
name: em_H02
min_mutation_count: 5
k_features: 1500
```
