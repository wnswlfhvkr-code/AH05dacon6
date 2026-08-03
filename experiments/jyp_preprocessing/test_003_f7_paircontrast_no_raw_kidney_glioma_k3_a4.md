# test_003_f7_paircontrast_no_raw_kidney_glioma_k3_a4

| 항목 | 결과 |
| --- | --- |
| 기준 전처리 | `test_003_f4_no_raw_hot5` |
| 모델 | XGBoost 고정 설정 |
| 비교 pair | KIRC/KIPAN, LGG/GBMLGG |
| 추가 피처 | pair당 2개, 총 4개 |
| 방향별 top-K | 3 |
| Laplace α | 4 |
| 최종 판정 | 채택, F7 승격 |
| 리더보드 Macro F1 | `0.3034169097` |
| F4 리더보드 대비 | `+0.0155273170` |

## 완료 실행 식별정보

| 항목 | 값 |
| --- | --- |
| 완료 상태 | `complete` |
| 완료 시각 | `2026-08-02 04:52:51 KST` |
| 선택 후보 | `f7_both_k3` |
| 검증기 판정 | `adopt_f7` |
| 계약 SHA-256 | `93ea56bad5c1c9e28f287b98eeff942246fd671add6cad5180a8b0b047804ba3` |
| 공유용 최종 결과 | 현재 문서 |

당시 검증은 동일한 `test_003` 설정과 고정된 paired CV 계산식으로 수행했습니다.
이 문서에는 확정된 결과와 적용 파라미터만 기록합니다.

## 전처리

- F0 유전자 변이 유무에서 두 클래스의 Fold-Train 통계만 계산
- 결측 유전자는 WT로 계산하지 않고 pair 점수에서 제외
- 변이 부담 5분위별 효과로 passenger association 조정
- pair support 10 이상
- 내부 5-Fold 방향 일치 4/5 이상
- 내부 5-Fold top-K 선택 빈도 3/5 이상
- Train 피처는 내부 OOF, Validation/Test는 Fold-Train full-fit 통계 사용
- 안정 유전자가 0개인 pair는 기준을 낮추지 않고 두 피처를 0으로 생성한 뒤 불안정 기록으로 저장

## 결과

| 검증 | F4 | F7 | 변화 |
| --- | ---: | ---: | ---: |
| seed 42 탐색 5-Fold | 0.394433 | 0.395841 | +0.001408 |
| 3-seed 확정 평균 | 0.387828 | 0.390175 | +0.002347 |
| exact genotype Group 5-Fold | 0.408361 | 0.408819 | +0.000457 |

| 확정 seed | F4 | F7 | 변화 | Fold 승리 |
| ---: | ---: | ---: | ---: | ---: |
| 101 | 0.389736 | 0.392882 | +0.003146 | 3/5 |
| 2027 | 0.387425 | 0.390741 | +0.003316 | 4/5 |
| 7301 | 0.386325 | 0.386903 | +0.000578 | 4/5 |

- Fold 승리: 11/15
- paired bootstrap 95% CI: `[-0.001644, +0.006332]`
- bootstrap에서 ΔF1>0 확률: `0.8738`
- 신장 pair 평균 F1 변화: `+0.014830`
- 뇌종양 pair 평균 F1 변화: `+0.034556`
- KIRC/KIPAN zero-stable-gene 불안정: 2건
- F7 OOF Train/Test 최대 PSI: `7.427574`
- F7 full-fit Train/Test 최대 PSI: `0.071995`
- Test 범위 이탈률 최댓값: `0.014140`
- Validation-only sentinel: 통과
- Target shuffle 진단: 통과

## 승격 판정 및 진단

현재 검증기에서 승격을 차단하는 기준은 `3-seed 평균 ΔF1 > 0` 하나이며 통과했습니다.
나머지는 위험을 확인하는 진단 항목으로 저장됩니다.

| 검사 항목 | 결과 |
| --- | --- |
| 3-seed 평균 ΔF1 > 0 | 통과 |
| 모든 seed 개선 | 통과 |
| Bootstrap CI 하한 > 0 | 실패 |
| 15개 Fold 중 10개 이상 승리 | 통과 |
| 두 pair 각각 평균 F1 +0.03 이상 | 실패 |
| 대상 4개 클래스 개별 하락 0.01 이하 | 통과 |
| 기타 암종 최악 하락 0.015 이하 | 실패 |
| 희소 암종 최악 하락 0.015 이하 | 실패 |
| 고변이 상위 10% 하락 0.01 이하 | 통과 |
| 확정 검증 zero-stable-gene 0건 | 실패 |
| Group CV ΔF1 -0.005 이상 | 통과 |
| Group CV zero-stable-gene 0건 | 통과 |
| F7 피처별 OOF Train/Test PSI 0.25 미만 | 실패 |
| Validation-only sentinel | 통과 |
| Target shuffle 진단 | 통과 |

- 검증기 공식 판정: `adopt_f7`
- 실제 제출 결과: 리더보드 `0.3034169097`, F4 대비 `+0.0155273170`
- 주요 위험: bootstrap CI가 0을 포함하며 OOF 방식 Train과 full-fit Test 사이 PSI가 큼
- 해석 주의: full-fit Train/Test PSI는 최대 `0.071995`로 낮아서, 높은 OOF PSI에는 OOF와 full-fit 생성 방식의 차이도 포함됨

## 실행

```bash
python -m src.train --config configs/test_003.yaml
```

`configs/test_003.yaml`은 동일한 모델·데이터 조건과 확정된 F7 파라미터를 사용합니다.
F7 학습 피처는 파이프라인 내부에서 OOF로 생성되며, Validation/Test는 학습 데이터에서
학습한 full-fit 통계만 사용합니다.

## 적용 전처리 파라미터

- RAW 원본 피처: `제외`
- 결측 처리: `pair 점수 계산에서 제외`
- 확인 근거: 검증기의 Fold별 F4 기준 및 F7 후보 recipe에서 확인한 값입니다.

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
show_progress: false
progress_interval: 25000
```

## 결과 해석

- 15개 Fold 중 11개에서 F7이 이겼고 3-seed 평균 ΔF1도 `+0.002347`로 양수여서 승격합니다.
- Bootstrap 95% CI가 0을 포함하고 Group CV 개선은 `+0.000457`로 작습니다.
- 최대 PSI `7.427574`와 zero-stable-gene 사례는 승격 후에도 확인해야 할 Test 전달 위험입니다.
- 최종 판정은 F7 채택이며 F4는 비교 기준으로 유지합니다.
- 리더보드에서도 F4보다 `+0.0155273170` 높아 최종 제출 후보로 유지합니다.
