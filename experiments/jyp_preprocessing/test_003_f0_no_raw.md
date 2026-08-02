# test_003_f0_no_raw

| 항목 | 결과 |
| --- | --- |
| 실행 ID | `20260801T205021123187+0900-d0633f46` |
| 실행 시작 | 2026-08-01T20:50:21+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | jyp_f0_no_raw |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 4230 |
| 검증 Macro F1 | 0.314610 |
| 설정 파일 | `configs\test_003.yaml` |
| 제출 파일 | `data/processed/test_003_f0_no_raw_submission.csv` (Git 제외) |
| 모델 아티팩트 | `models/test_003_f0_no_raw.pkl` (Git 제외) |

## 하이퍼파라미터

```yaml
name: xgboost
n_estimators: 100
learning_rate: 0.1
max_depth: 6
n_jobs: -1
eval_metric: mlogloss
tree_method: hist
```

## 적용 전처리 파라미터

- RAW 원본 피처: `제외`
- 확인 근거: 현재 독립 파이프라인의 `name` 기준으로 정규화해 표시했습니다.

```yaml
name: jyp_f0_no_raw
show_progress: true
progress_interval: 25000
```

## 결과 해석

- `f0` 대비 ΔF1은 `+0.028213`이며 피처는 4,384개 줄었습니다.
- Test 예측의 `44.894%`가 바뀌어 RAW 제거가 모델 결정 경계를 크게 바꿨습니다.
- 동일 단계 RAW 포함/제외 비교 중 가장 큰 개선으로, 이후 no-raw 방향의 근거가 됩니다.
