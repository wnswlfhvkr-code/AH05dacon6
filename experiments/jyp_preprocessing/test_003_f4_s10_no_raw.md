# test_003_f4_s10_no_raw

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-01T02:17:46+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | jyp_f4_no_raw |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 81742 |
| 검증 Macro F1 | 0.395643 |
| 설정 파일 | `configs\test_003.yaml` |
| 제출 파일 | `data\processed\test_003_f4_s10_no_raw_submission.csv` |
| 모델 아티팩트 | `models\test_003_f4_s10_no_raw.pkl` |

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
name: jyp_f4_no_raw
burden_clip_quantile: 0.99
f3_position_min_support: 2
f3_aa_min_support: 2
f4_min_support: 10
show_progress: true
progress_interval: 25000
```

## 결과 해석

- F3 no-raw 대비 ΔF1은 `+0.000911`로 거의 중립입니다.
- 동일 support 10의 RAW 포함 실행보다는 `+0.016793` 높아 no-raw 효과는 다시 확인됐습니다.
- Test 예측 변화율은 F3 no-raw 대비 `6.952%`입니다.
