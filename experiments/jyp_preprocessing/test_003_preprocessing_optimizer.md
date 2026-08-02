# test_003 전처리 파라미터 최적화

모델과 80:20 분할은 `test_003.yaml` 그대로 고정하고, RAW를 제외한 F3→F4→F5만 순차 탐색했습니다.

| 단계 | 최고 Validation Macro F1 | 이전 채택안 대비 Δ | 판정 |
| --- | ---: | ---: | --- |
| F3 no-raw | 0.394732 | 기준 | 채택 |
| F4 no-raw | 0.398243 | +0.003511 | 채택 |
| F5 no-raw | 0.415031 | +0.016788 | 채택 |

- 단일 holdout 탐색 후보 기준: Macro F1이 이전 채택안보다 최소 `0.003` 개선
- 3-seed × 5-Fold 확정 승격 기준: 평균 Macro F1 변화 `> 0`
- 최종 추천 단계: `jyp_f5_no_raw`
- 최종 추천 점수: `0.415031`
- 현재 공유 설정 파일: `configs/test_003.yaml` (F7 최종 후보)

## 최종 추천 전처리 파라미터

| 파라미터 | 값 |
| --- | ---: |
| `f3_position_min_support` | 3 |
| `f3_aa_min_support` | 3 |
| `f4_min_support` | 5 |
| `f5_top_k_per_class` | 10 |
| `f5_min_gene_support` | 3 |
| `f5_laplace_alpha` | 1.0 |
| `f5_min_direction_consistency` | 3 |
| `f5_min_selection_frequency` | 2 |

## 주의

이 결과는 기존 ablation과 같은 단일 80:20 split의 1차 탐색 기록입니다.
현재 최종 공유 설정은 F7이며, 아래 F5 값은 과거 후보의 재현을 위한 실험 기록으로만 사용합니다.

## 적용 전처리 파라미터

- RAW 원본 피처: `제외`
- 결측 처리: `legacy_treat_as_wt`
- 확인 근거: optimizer CSV의 최고 점수 trial에서 확인한 값입니다.

```yaml
name: jyp_f5_no_raw
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

- 같은 holdout을 반복 탐색한 결과 F5 no-raw가 `0.415031`로 최고였습니다.
- F3→F4는 `+0.003511`, F4→F5는 `+0.016788` 개선됐습니다.
- 이 결과는 adaptive single-split 탐색이므로 후보 발굴에는 유효하지만 최종 일반화 근거는 아닙니다.
- 독립 seed·paired bootstrap 결과가 없는 상태에서는 F5를 운영 기준으로 자동 승격하지 않습니다.
