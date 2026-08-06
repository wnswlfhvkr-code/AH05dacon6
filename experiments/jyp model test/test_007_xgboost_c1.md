# test_007_xgboost_c1

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-04T20:01:36+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | pipeComb_v3 |
| 선택된 최소 변이 횟수 | - |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 55105 |
| 최종 Macro F1 | 0.448779 |
| 80% 학습 Macro F1 | 0.909785 |
| 20% 검증 Macro F1 | 0.466205 |
| 과적합 격차 | 0.443581 |
| 과적합 여부 | True |
| 설정 파일 | `data\backup\yaml\test_007_xgboost_c1.yaml` |
| 제출 파일 | `data\processed\test_007_xgboost_c1_submission.csv` |
| 모델 아티팩트 | `models\test_007_xgboost_c1.pkl` |

## 하이퍼파라미터

```yaml
name: xgboost
n_estimators: 100
learning_rate: 0.1
max_depth: 6
n_jobs: -1
eval_metric: mlogloss
tree_method: hist
device: cuda
```

## 전처리 설정

```yaml
name: pipeComb_v3
```
