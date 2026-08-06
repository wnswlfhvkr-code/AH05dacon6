# test_006_em_H06

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-05T15:31:24+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | em_H06 |
| 선택된 최소 변이 횟수 | 5 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 256 |
| 최종 Macro F1 | 0.380729 |
| 80% 학습 Macro F1 | 0.687519 |
| 20% 검증 Macro F1 | 0.375760 |
| 과적합 격차 | 0.311759 |
| 과적합 여부 | True |
| 설정 파일 | `configs/test_006_em_H06.yaml` |
| 제출 파일 | `data/processed/test_006_em_H06_xgboost_submission.csv` |
| 모델 아티팩트 | `models/test_006_em_H06.pkl` |

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
name: em_H06
min_mutation_count: 5
n_components: 256
svd_n_iter: 7
random_state: 42
```
