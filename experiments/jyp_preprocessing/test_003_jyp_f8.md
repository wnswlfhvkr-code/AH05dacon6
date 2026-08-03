# test_003_jyp_f8

| 항목                  | 결과                                                                       |
| --------------------- | -------------------------------------------------------------------------- |
| 실행 시각             | 2026-08-03T11:47:39+09:00                                                  |
| 모델                  | xgboost                                                                    |
| 전처리 파이프라인     | jyp_f8                                                                     |
| 선택된 최소 변이 횟수 | -                                                                          |
| 학습 데이터 행 수     | 6201                                                                       |
| 피처 수               | 54661                                                                      |
| 검증 Macro F1         | 0.408919                                                                   |
| 설정 파일             | `configs\test_003.yaml`                                                  |
| 제출 파일             | `data\processed\test_003_jyp_f8_submission.csv`                          |
| 모델 아티팩트         | `models\test_003_jyp_f8.pkl`                                             |

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

## 전처리 설정

```yaml
name: jyp_f8
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
f5_output_rare_class_count: 10
f7_pairs:
- - KIRC
  - KIPAN
- - LGG
  - GBMLGG
f7_top_k_per_direction: 3
f7_min_gene_support: 10
f7_laplace_alpha: 4.0
f7_burden_quantiles: 5
f7_stability_folds: 5
f7_min_direction_consistency: 4
f7_min_selection_frequency: 3
f7_random_state: 42
show_progress: true
progress_interval: 25000
```

## 비교와 판단

| 비교 기준 | Macro F1 변화 |
|---|---:|
| F5 selective N=10 `0.408814` 대비 | +0.000105 |
| F7 단일 holdout `0.399189` 대비 | +0.009730 |
| F4 no-raw hot5 `0.398243` 대비 | +0.010676 |
| 전체 F5 출력 `0.415031` 대비 | -0.006112 |

- F7 4개를 N=10에 추가한 직접 효과는 `+0.000105`로 작지만 양수입니다.
- 피처 수는 예상대로 54,657개에서 54,661개로 4개 증가했습니다.
- 현재 판정은 `안정성 비교 전 보류`입니다. 점수 부족으로 미승격한 것이 아닙니다.
- N=10과 같은 분할·seed·Fold에서 paired 검증한 뒤 안정성이 비슷하고 평균 Macro F1 변화가 `0보다 크면`, 상승 폭과 관계없이 F8을 승격합니다.
- seed별 방향 반전, Fold 성능 분산, 클래스별 급락 또는 분포 drift가 기준안보다 명백히 나빠지면 보류합니다.
- 검증 전 운영 후보는 다중 Fold·리더보드 근거가 있는 `jyp_f7`을 유지합니다.

## 승격 기준

- 고정 최소 개선 폭 없음
- `안정성 유사 + ΔMacro F1 > 0`: 승격
- `안정성 명백히 악화`: 보류
- CI와 PSI는 안정성 판단 자료이며 단독 자동 탈락 기준이 아님
