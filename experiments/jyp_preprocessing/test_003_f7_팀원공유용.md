# test_003 F7 팀원 공유용

## 한 줄 결론

`F4 no-raw hot5`에 두 혼동 암종 pair의 비교 점수 4개만 추가했습니다. 모델은 기존 XGBoost를
유지했고, 리더보드 Macro F1은 `0.2878895927 → 0.3034169097`로 `+0.0155273170` 개선됐습니다.

## 적용 구성

### 사용

- `+` RAW exact-cell 문자열 제외
- `+` F0 유전자별 변이 유무
- `+` F1 환자별 변이 부담·변이 유형 요약
- `+` F2 유전자×변이 유형
- `+` F3 변이 위치 구간·아미노산 치환
- `+` F4 exact hotspot, support 5 이상
- `+` F7 KIRC/KIPAN 비교 점수 2개
- `+` F7 LGG/GBMLGG 비교 점수 2개
- `+` 결측 유전자는 WT로 보지 않고 pair 점수 계산에서 제외
- `+` 학습 피처는 내부 OOF 통계, Validation/Test는 학습 Fold의 full-fit 통계 사용

### 사용하지 않음

- `-` RAW OrdinalEncoder: 동일 단계의 no-raw 비교에서 반복적으로 성능이 낮아 제외
- `-` F5 signature: 단일 holdout 최고점은 높았지만 반복 탐색된 target-aware 후보라 최종 F7에는 미포함
- `-` F6 TF-IDF·SVD: F4 대비 Macro F1 `-0.051182`로 하락해 미채택
- `-` 추가 co-mutation·exact hotspot: F4 hotspot을 유지하고 F7에서는 비교 점수 4개만 추가

## 추가된 F7 피처

```text
F7__KIRC_vs_KIPAN__mutated_log_odds_mean
F7__KIRC_vs_KIPAN__bernoulli_llr_mean
F7__LGG_vs_GBMLGG__mutated_log_odds_mean
F7__LGG_vs_GBMLGG__bernoulli_llr_mean
```

- 양수: 피처명 앞쪽 암종 지지
- 음수: 뒤쪽 암종 지지
- `mutated_log_odds_mean`: 변이된 선택 유전자들의 평균 상대 점수
- `bernoulli_llr_mean`: 변이와 WT를 함께 반영한 평균 상대 가능도

## 점수 변화

| 비교 | F4 | F7 | 변화 |
| --- | ---: | ---: | ---: |
| 단일 holdout | 0.398243094 | 0.399188637 | +0.000945543 |
| seed 42 탐색 5-Fold | 0.394433 | 0.395841 | +0.001408 |
| 3-seed 확정 평균 | 0.387828 | 0.390175 | +0.002347 |
| exact genotype Group 5-Fold | 0.408361 | 0.408819 | +0.000457 |
| 리더보드 | 0.2878895927 | 0.3034169097 | +0.0155273170 |

| 항목 | F4 | F7 | 변화 |
| --- | ---: | ---: | ---: |
| 전체 피처 수 | 54,617 | 54,621 | +4 |
| Test 예측 변화율 | - | 10.291% | - |

## 핵심 파라미터

| 파라미터 | 값 |
| --- | --- |
| 전처리 이름 | `jyp_f7` |
| 비교 pair | KIRC/KIPAN, LGG/GBMLGG |
| 방향별 top-K | 3 |
| pair 변이 support | 10 이상 |
| Laplace α | 4.0 |
| 변이 부담 구간 | 5분위 |
| 안정성 Fold | 5 |
| 효과 방향 일치 | 4/5 이상 |
| top-K 선택 빈도 | 3/5 이상 |
| F4 hotspot support | 5 이상 |
| RAW 원본 피처 | 제외 |

## 채택 판단

- `+` 3개 확정 seed 모두 개선
- `+` 15개 Fold 중 11개에서 F7 승리
- `+` 뇌종양 pair 평균 F1 `+0.034556`
- `+` 실제 리더보드에서 F4 대비 `+0.0155273170`
- `주의` bootstrap 95% CI `[-0.001644, +0.006332]`로 0 포함
- `주의` 신장 pair 평균 F1 `+0.014830`으로 목표 `+0.03` 미달
- `주의` OOF Train/Test 최대 PSI `7.427574`
- `참고` full-fit Train/Test 최대 PSI는 `0.071995`

최종적으로 F7을 제출 후보로 채택하되, 개선 폭과 분포 안정성 위험을 함께 공유합니다.

## 실행

```bash
python -m src.train --config configs/test_003.yaml
```

- 모델: 기존 XGBoost 1개
- 설정: `configs/test_003.yaml`
- 상세 근거: `experiments/jyp_preprocessing/test_003_f7_paircontrast_no_raw_kidney_glioma_k3_a4.md`
