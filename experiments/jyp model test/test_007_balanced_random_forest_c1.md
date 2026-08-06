# test_007_balanced_random_forest_c1

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-05T00:51:37+09:00 |
| 모델 | balanced_random_forest |
| 전처리 파이프라인 | pipeComb_v3 |
| 선택된 최소 변이 횟수 | - |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 55105 |
| 최종 Macro F1 | 0.395469 |
| 80% 학습 Macro F1 | 0.573789 |
| 20% 검증 Macro F1 | 0.383127 |
| 과적합 격차 | 0.190662 |
| 과적합 여부 | True |
| 설정 파일 | `data\backup\yaml\test_007_balanced_random_forest_c1.yaml` |
| 제출 파일 | `data\processed\test_007_balanced_random_forest_c1_submission.csv` |
| 모델 아티팩트 | `models\test_007_balanced_random_forest_c1.pkl` |

## 하이퍼파라미터

```yaml
name: balanced_random_forest
n_estimators: 500
criterion: gini
max_depth: 28
min_samples_split: 4
min_samples_leaf: 2
max_features: sqrt
sampling_strategy: all
replacement: true
bootstrap: false
class_weight: null
n_jobs: -1
```

## 전처리 설정

```yaml
name: pipeComb_v3
```
