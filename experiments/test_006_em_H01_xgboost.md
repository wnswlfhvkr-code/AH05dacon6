# test_006_em_H01

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-06T12:42:48+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | em_H01 |
| 선택된 최소 변이 횟수 | 2 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 23113 |
| 최종 Macro F1 | 0.459220 |
| 80% 학습 Macro F1 | 0.656120 |
| 20% 검증 Macro F1 | 0.442654 |
| 과적합 격차 | 0.213466 |
| 과적합 여부 | True |
| 설정 파일 | `configs/test_006_em_H01.yaml` |
| 제출 파일 | `data/processed/test_006_em_H01_xgboost_submission.csv` |
| 모델 아티팩트 | `models/test_006_em_H01.pkl` |

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
name: em_H01
min_mutation_count: 2
```
