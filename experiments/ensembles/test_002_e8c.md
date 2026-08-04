# test_002_e8c — E8A + EM16 aligned SGKF 앙상블

## 목적

E8B의 개선이 서로 다른 OOF 분할을 결합한 효과인지 확인하기 위해 EM16도 E8A와 동일한 고정 SGKF 3-Fold × 3 seeds로 재학습했다.

## 검증과 결과

- 동일 변이 프로필 환자는 같은 Fold에 유지
- EM16 전처리는 각 Fold-Train에서만 학습
- EM16 aligned OOF: seed 42 `0.361767`, seed 2026 `0.377680`, seed 777 `0.375016`
- E8A: `0.467274 ± 0.006753`
- E8C: `0.473037 ± 0.016776`
- E8C - E8A: `+0.005763`

| seed | E8A | E8C | 변화 |
| ---: | ---: | ---: | ---: |
| 42 | 0.463521 | 0.473594 | +0.010073 |
| 2026 | 0.463232 | 0.455990 | -0.007242 |
| 777 | 0.475070 | 0.489527 | +0.014457 |

## 판단

평균 점수는 E8A보다 높았지만 seed 2026이 하락했고 표준편차가 커졌다. 원 노트북은 bootstrap과 Test 예측 전에 종료되었으므로 최종 채택 근거가 없다. 후속 Public 결과에서 EM 계열의 분포 이동 위험도 확인되어 최종 전략에서 제외했다.

## 입력 파일

- `em16_sgkf_oof_probability_by_seed.npy`: `(3, 6201, 26)`
- `em16_sgkf_test_probability_by_seed_fold.npy`: `(3, 3, 2546, 26)`

## 재현 명령

```bash
python -m src.ensembles.train_jh_e8c \
  --config configs/ensembles/test_002_e8c.yaml
```
