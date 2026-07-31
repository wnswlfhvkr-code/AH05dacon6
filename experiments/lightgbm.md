# lightgbm

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-07-31T16:45:10+09:00 |
| 모델 | lightgbm |
| 전처리 파이프라인 | baseline |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 4230 |
| 검증 Macro F1 | 0.262962 |
| 설정 파일 | `configs\lightgbm.yaml` |
| 제출 파일 | `data\processed\lightgbm_submission.csv` |
| 모델 아티팩트 | `models\lightgbm.pkl` |

## 하이퍼파라미터

```yaml
name: lightgbm
objective: multiclass
n_estimators: 300
learning_rate: 0.05
num_leaves: 31
max_depth: -1
min_child_samples: 20
subsample: 1.0
colsample_bytree: 1.0
reg_alpha: 0.0
reg_lambda: 0.0
n_jobs: -1
verbosity: -1
```
