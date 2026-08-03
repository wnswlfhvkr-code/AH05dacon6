# test_003 F5 selective no-raw 결과

## 동일 모델·분할 비교

| 실행 | F5 출력 | 총 피처 수 | 검증 Macro F1 | 판정 |
|---|---:|---:|---:|---|
| `test_003_jyp_f5_selective_no_raw` | 실제 26개 클래스 × 4점수 = 104개 | 54,721 | 0.415031 | 전체 F5 출력 참고값 |
| `test_003_jyp_f5_selective_n5` | 희소 5개 클래스 × 4점수 = 20개 | 54,637 | 0.397690 | F4와 거의 같아 미채택 |
| `test_003_jyp_f5_selective_n10` | 희소 10개 클래스 × 4점수 = 40개 | 54,657 | 0.408814 | F7 결합 후보로 선택 |
| `test_003_jyp_f8` | N10 40개 + F7 4개 | 54,661 | 0.408919 | 안정성 비교 전 보류 |

## N=10 비교

| 비교 기준 | Macro F1 변화 |
|---|---:|
| N=5 대비 | +0.011124 |
| F4 no-raw hot5 `0.398243` 대비 | +0.010571 |
| F7 단일 holdout `0.399189` 대비 | +0.009625 |
| 전체 F5 출력 `0.415031` 대비 | -0.006217 |

## 해석

- N=5는 signature를 지나치게 줄여 F4보다도 소폭 낮았습니다.
- N=10은 N=5보다 높고 F4·F7 단일 holdout보다도 높았습니다.
- 전체 26개 클래스 출력보다는 낮지만 F5 피처를 104개에서 40개로 줄였습니다.
- 모두 동일한 단일 holdout 결과이므로 독립 다중 seed 일반화 성능으로 해석하지 않습니다.
- F8은 N=10보다 `+0.000105` 높아 점수 방향은 양수입니다.
- 개선 폭이 작다는 이유로 탈락시키지 않으며, 현재 상태는 `안정성 비교 전 보류`입니다.
- N=10과 같은 반복 검증에서 안정성이 비슷하고 평균 Macro F1 변화가 양수로 유지되면 F8을 승격합니다. 안정성이 명백히 나빠지면 보류합니다.

## 근거 파일

- `data/processed/test_003_jyp_f5_selective_no_raw_metrics.json`
- `data/processed/test_003_jyp_f5_selective_n5_metrics.json`
- `data/processed/test_003_jyp_f5_selective_n10_metrics.json`
- `data/processed/test_003_jyp_f8_metrics.json`
- `experiments/jyp_preprocessing/test_003_jyp_f5_selective_n5.md`
- `experiments/jyp_preprocessing/test_003_jyp_f5_selective_n10.md`
- `experiments/jyp_preprocessing/test_003_jyp_f8.md`
