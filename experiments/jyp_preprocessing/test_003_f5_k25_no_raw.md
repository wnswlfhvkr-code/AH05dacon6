# test_003_f5_k25_no_raw

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-01T02:20:03+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | jyp_f5_no_raw |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 81846 |
| 검증 Macro F1 | 0.382693 |
| 설정 파일 | `configs\test_003.yaml` |
| 제출 파일 | `data\processed\test_003_f5_k25_no_raw_submission.csv` |
| 모델 아티팩트 | `models\test_003_f5_k25_no_raw.pkl` |

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
- 결측 처리: `legacy_treat_as_wt`
- 확인 근거: 현재 독립 파이프라인의 `name` 기준으로 정규화해 표시했습니다.

```yaml
name: jyp_f5_no_raw
burden_clip_quantile: 0.99
f3_position_min_support: 2
f3_aa_min_support: 2
f4_min_support: 10
f5_top_k_per_class: 25
f5_min_gene_support: 5
f5_laplace_alpha: 1.0
f5_stability_folds: 5
f5_min_direction_consistency: 4
f5_min_selection_frequency: 3
f5_random_state: 42
show_progress: true
progress_interval: 25000
```

## 결과 해석

- F4 support 10 no-raw 대비 ΔF1은 `-0.012950`입니다.
- RAW 포함 F5보다는 `+0.017923` 개선됐지만 F4 기준을 회복하지 못했습니다.
- Test 예측 변화율은 대응 F4 대비 `32.561%`로 target signature의 영향이 컸습니다.
