# test_003_f7_paircontrast_no_raw

| 항목              | 결과                                                                         |
| ----------------- | ---------------------------------------------------------------------------- |
| 실행 ID           | `20260802T035715388542+0900-e3b5b79a`                                      |
| 실행 시작         | 2026-08-02T03:57:15+09:00                                                    |
| 모델              | xgboost                                                                      |
| 전처리 파이프라인 | jyp_f7                                                                       |
| 학습 데이터 행 수 | 6201                                                                         |
| 피처 수           | 54621                                                                        |
| 검증 Macro F1     | 0.399189                                                                     |
| 리더보드 Macro F1 | 0.3034169097                                                                 |
| 설정 파일         | `configs\test_003.yaml`                                                    |
| 제출 파일         | `data/processed/test_003_f7_paircontrast_no_raw_submission.csv` (Git 제외) |
| 모델 아티팩트     | `models/test_003_f7_paircontrast_no_raw.pkl` (Git 제외)                    |

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
- 결측 처리: `pair 점수 계산에서 제외`
- 확인 근거: 저장된 모델 아티팩트에서 확인한 실제 실행값입니다.

```yaml
name: jyp_f7
burden_clip_quantile: 0.99
f3_position_min_support: 3
f3_aa_min_support: 3
f4_min_support: 5
f7_pairs:
  - [KIRC, KIPAN]
  - [LGG, GBMLGG]
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

## 결과 해석

- 최종 F4에 4개 피처만 추가해 ΔF1 `+0.000946`을 기록했습니다.
- Test 예측 변화율은 `10.291%`로 점수 개선에 비해 예측 변화가 큽니다.
- 후속 3-seed 평균 개선은 `+0.002347`로 0보다 커 F7을 승격합니다.
- 리더보드 `0.3034169097`로 F4 제출 `0.2878895927`보다 `+0.0155273170` 개선됐습니다.
- 높은 PSI와 일부 안정성 실패는 승격을 막지 않으며, Test 전달 위험을 확인하는 진단 정보로 남깁니다.
