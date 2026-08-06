# test_007_lightgbm_c1

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-04T20:30:36+09:00 |
| 모델 | lightgbm |
| 전처리 파이프라인 | pipeComb_v3 |
| 선택된 최소 변이 횟수 | - |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 55105 |
| 최종 Macro F1 | 0.474520 |
| 80% 학습 Macro F1 | 0.955640 |
| 20% 검증 Macro F1 | 0.471884 |
| 과적합 격차 | 0.483756 |
| 과적합 여부 | True |
| 설정 파일 | `data\backup\yaml\test_007_lightgbm_c1.yaml` |
| 제출 파일 | `data\processed\test_007_lightgbm_c1_submission.csv` |
| 모델 아티팩트 | `models\test_007_lightgbm_c1.pkl` |

## 하이퍼파라미터

```yaml
name: lightgbm
n_estimators: 500
learning_rate: 0.03
num_leaves: 31
min_child_samples: 20
subsample: 0.8
subsample_freq: 1
colsample_bytree: 0.8
reg_alpha: 0.1
reg_lambda: 5.0
class_weight: balanced
device_type: cpu
n_jobs: -1
verbosity: -1
```

## 전처리 설정

```yaml
name: pipeComb_v3
```
