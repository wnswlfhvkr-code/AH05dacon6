# test_003_jyp_f9

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-03T15:50:30+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | jyp_f9 |
| 선택된 최소 변이 횟수 | - |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 55001 |
| 검증 Macro F1 | 0.436640 |
| 설정 파일 | `configs\test_003.yaml` |
| 제출 파일 | `data\processed\test_003_jyp_f9_submission.csv` |
| 모델 아티팩트 | `models\test_003_jyp_f9.pkl` |

## 3-seed Fold 과적합 측정

집계 원본: `data/processed/test_003_jypF9_EM16_severity_signature_3seed_scores.csv`의 `name == f9`인 15개 Fold

| 구분 | Fold 수 | Train Macro F1 평균 | Validation Macro F1 평균 | Train-Validation Gap 평균 | Gap > 0.1 | 판정 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 전체 | 15 | 0.893350205 | 0.428489695 | 0.464860510 | 15/15 | 과적합 |
| seed 101 | 5 | 0.892744757 | 0.432234539 | 0.460510218 | 5/5 | 과적합 |
| seed 2027 | 5 | 0.894252182 | 0.429735059 | 0.464517123 | 5/5 | 과적합 |
| seed 7301 | 5 | 0.893053675 | 0.423499486 | 0.469554189 | 5/5 | 과적합 |

임계값 0.1을 15개 Fold 모두 초과했으므로 F9는 과적합으로 판정한다.

- Gap 범위: 0.437265594 ~ 0.487361031
- 이번 3-seed 평균 OOF Macro F1: 0.430298941
- 문서 상단의 0.436640은 별도 단일 Holdout 실행 결과이다.

3-seed 평균 OOF Macro F1(0.430298941)과 15개 Fold Macro F1 평균(0.428489695)의 차이는 Macro F1의 비선형 집계 특성 때문이다.

해석: 학습 데이터 적합도에 비해 새 Fold 일반화 성능이 크게 낮으므로, 이후 후보 선택은 Train 점수가 아니라 OOF·리더보드 성능을 기준으로 한다.

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
name: jyp_f9
burden_clip_quantile: 0.99
f3_position_min_support: 3
f3_aa_min_support: 3
f4_min_support: 5
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
