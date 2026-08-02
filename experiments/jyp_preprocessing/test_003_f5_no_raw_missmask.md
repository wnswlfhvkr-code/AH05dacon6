# test_003_f5_no_raw_missmask

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-01T03:52:29+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | jyp_f5_no_raw_missmask |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 54721 |
| 검증 Macro F1 | 0.415031 |
| 설정 파일 | `configs\test_003_f5_no_raw_missmask.yaml` |
| 제출 파일 | `data\processed\test_003_f5_no_raw_missmask_submission.csv` |
| 모델 아티팩트 | `models\test_003_f5_no_raw_missmask.pkl` |

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
- 결측 처리: `exclude_from_probability_and_wt_likelihood`
- 확인 근거: 현재 독립 파이프라인의 `name` 기준으로 정규화해 표시했습니다.

```yaml
name: jyp_f5_no_raw_missmask
burden_clip_quantile: 0.99
f3_position_min_support: 3
f3_aa_min_support: 3
f4_min_support: 5
f5_top_k_per_class: 10
f5_min_gene_support: 3
f5_laplace_alpha: 1.0
f5_stability_folds: 5
f5_min_direction_consistency: 3
f5_min_selection_frequency: 2
f5_random_state: 42
show_progress: true
progress_interval: 25000
```

## 결과 해석

- 최종 F4 대비 피처 104개 추가로 ΔF1 `+0.016788`을 기록해 단일 holdout 최고점입니다.
- Test 예측의 `25.373%`가 달라져 작은 저차원 블록이 모델 결정에 큰 영향을 줬습니다.
- optimizer의 non-missmask 최고점도 동일한 `0.415031`이므로, 이 holdout에서는 missmask 자체의 추가 점수 이득은 관찰되지 않았습니다.
- Target-aware 탐색 결과이며 Git 내부에 독립 다중 seed 검증 수치가 없어 최종 운영 승격값으로 단정하지 않습니다.
