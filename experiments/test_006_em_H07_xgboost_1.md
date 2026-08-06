# test_006_em_H07

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-05T15:35:24+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | em_H07 |
| 선택된 최소 변이 횟수 | 5 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 4559 |
| 최종 Macro F1 | 0.299801 |
| 80% 학습 Macro F1 | 0.349542 |
| 20% 검증 Macro F1 | 0.272062 |
| 과적합 격차 | 0.077480 |
| 과적합 여부 | False |
| 설정 파일 | `configs/test_006_em_H07.yaml` |
| 제출 파일 | `data/processed/test_006_em_H07_xgboost_submission.csv` |
| 모델 아티팩트 | `models/test_006_em_H07.pkl` |

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
name: em_H07
min_mutation_count: 5
min_hotspot_count: 5
max_hotspots: 384
```
