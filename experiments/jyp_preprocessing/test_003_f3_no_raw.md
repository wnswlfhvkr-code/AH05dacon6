# test_003_f3_no_raw

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-01T02:46:48+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | jyp_f3_no_raw |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 81622 |
| 검증 Macro F1 | 0.394732 |
| 설정 파일 | `configs\test_003.yaml` |
| 제출 파일 | `data\processed\test_003_f3_no_raw_submission.csv` |
| 모델 아티팩트 | `models\test_003_f3_no_raw.pkl` |

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
name: jyp_f3_no_raw
burden_clip_quantile: 0.99
f3_position_min_support: 2
f3_aa_min_support: 2
show_progress: true
progress_interval: 25000
```

## 결과 해석

- RAW 포함 F3 대비 ΔF1은 `+0.019146`입니다.
- Test 예측의 `21.838%`가 달라졌으며, 고차원 변이 피처에서도 RAW 제거 효과가 반복됐습니다.
- 이후 F4/F5 비교의 no-raw 기준점으로 사용하기 적절합니다.
