# xgboost_baseline

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-07-31T13:37:40+09:00 |
| 모델 | xgboost |
| 전처리 버전 | v1 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 4230 |
| 검증 Macro F1 | 0.292181 |
| 설정 파일 | `configs/xgboost_baseline.yaml` |
| 제출 파일 | `data/processed/xgboost_baseline_submission.csv` |
| 모델 아티팩트 | `models/xgboost_baseline.pkl` |

## 하이퍼파라미터

```yaml
name: xgboost
n_estimators: 100
learning_rate: 0.1
max_depth: 6
n_jobs: -1
```
