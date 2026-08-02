# test_003_4_s5_no_raw

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-01T02:34:18+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | jyp_f4_no_raw |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 82051 |
| 검증 Macro F1 | 0.398243 |
| 설정 파일 | `configs\test_003.yaml` |
| 제출 파일 | `data\processed\test_003_4_s5_no_raw_submission.csv` |
| 모델 아티팩트 | `models\test_003_4_s5_no_raw.pkl` |

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
f4_min_support: 5
show_progress: true
progress_interval: 25000
```

## 결과 해석

- support 10 no-raw 대비 ΔF1은 `+0.002600`입니다.
- hotspot 최소 support를 5로 낮추면서 희소하지만 유효한 exact event가 추가됐습니다.
- Test 예측 변화율은 `6.127%`로, 작은 파라미터 변경이 일부 환자 분류를 바꿨습니다.
