# test_003_f4_no_raw_hot5

| 항목 | 결과 |
| --- | --- |
| 실행 ID | `20260801T200833406407+0900-47b8f16e` |
| 실행 시작 | 2026-08-01T20:08:33+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | jyp_f4_no_raw |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 54617 |
| 검증 Macro F1 | 0.398243 |
| 사용자 제공 리더보드 점수 | 0.2878895927 |
| 설정 파일 | `configs/test_003.yaml` |
| 제출 파일 | `data/processed/test_003_f4_no_raw_hot5_submission.csv` |
| 모델 아티팩트 | `models/test_003_f4_no_raw_hot5.pkl` |

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

## 후속 검증

`test_003_f7_paircontrast_no_raw_kidney_glioma_k3_a4`는 3-seed 평균 Macro F1이 `+0.002347` 올라 승격했습니다. F4는 pair-contrast 검증의 비교 기준 전처리로 유지합니다.

## 적용 전처리 파라미터

- RAW 원본 피처: `제외`
- 확인 근거: 현재 독립 파이프라인의 `preprocessing.name`과 실행 기록을 대조했습니다.

```yaml
name: jyp_f4_no_raw
burden_clip_quantile: 0.99
f3_position_min_support: 3
f3_aa_min_support: 3
f4_min_support: 5
show_progress: true
progress_interval: 25000
```

## 결과 해석

- `f4_s5_no_raw`와 점수는 같고 Test 예측은 `99.961%` 일치합니다.
- F3 support를 3으로 정리하면서 피처를 82,051개에서 54,617개로 27,434개(33.44%) 줄였습니다.
- 사용자 제공 리더보드 점수는 `0.2878895927`로 holdout보다 크게 낮아 단일 분할 낙관성을 보여줍니다.
- 동일 성능의 더 간결한 F4 기준으로 유지할 근거가 있습니다.
