# TEST_003 JYP F9 + EM24 XGBoost 확률 결합

> 후속 4개 가중치 비교에서 점수 우선 후보는 `F9 0.60 + EM24 0.40`으로 변경됐다. 이 문서는 w70 후보의 최초 확정 검증 기록이며, 최종 비교는 `test_003_jypF9_EM24_weight_comparison.md`를 따른다.

## 최종 조건

- 모델: 모든 멤버에 동일한 XGBoost 사용
  - `n_estimators=100`
  - `learning_rate=0.1`
  - `max_depth=6`
  - `tree_method=hist`
- 기준 전처리: `jyp_f9`
- 결합 전처리: `em_v24`
- 결합 방식: 26개 원본 `SUBCLASS` 확률의 가중 평균
  - `jyp_f9=0.70`
  - `em_v24=0.30`
- 검증: 동일한 `StratifiedKFold`를 사용한 5-Fold OOF
- 확정 seed: `101`, `2027`, `7301`
- Test와 리더보드 점수는 후보 선택에 사용하지 않음

## 1차 선별 결과

seed 42, 5-Fold에서 네 전처리를 Fold당 한 번만 학습했다.

| 후보 | OOF Macro F1 | F9 대비 | Fold 승리 |
|---|---:|---:|---:|
| F9 단독 | 0.426805 | 기준 | - |
| F9 0.60 + EM24 0.40 | 0.448592 | +0.021787 | 5/5 |
| F9 0.65 + EM20 0.35 | 0.447036 | +0.020232 | 5/5 |
| F9 0.65 + EM19 0.35 | 0.447036 | +0.020232 | 5/5 |

- EM20과 EM19의 OOF 확률은 이 조건에서 완전히 동일했다.
- 저장된 OOF로 가중치만 비교한 결과, EM24 조합은 F9 비중 0.70에서 `0.448765`였다.
- F9 0.70은 F9 0.60보다 점수가 소폭 높고 최악 클래스 하락도 작아 확정 후보로 선택했다.

## 3-seed 확정 결과

| Seed | F9 | F9 0.70 + EM24 0.30 | 변화 |
|---:|---:|---:|---:|
| 101 | 0.433925 | 0.450033 | +0.016108 |
| 2027 | 0.430998 | 0.448547 | +0.017549 |
| 7301 | 0.425973 | 0.447580 | +0.021607 |
| 평균 | 0.430299 | 0.448720 | +0.018421 |

- seed 승리: `3/3`
- Fold 승리: `15/15`
- seed 표준편차: F9 `0.003284` → 결합 `0.001009`
- 환자 단위 층화 paired bootstrap 10,000회:
  - 평균 변화 `+0.018393`
  - 95% CI `[+0.012780, +0.024097]`
  - 변화가 양수인 bootstrap 비율 `100%`

## 클래스 영향

- 26개 중 평균 F1 개선 22개, 하락 4개
- 모든 seed에서 개선 12개
- 모든 seed에서 하락한 클래스는 SKCM 1개
- 평균 하락이 큰 클래스:
  - TGCT `-0.011414`
  - SKCM `-0.010757`
- 평균 개선이 큰 클래스:
  - DLBC `+0.178453`
  - ACC `+0.061108`
  - BLCA `+0.036140`
- KIRC 평균 변화 `-0.002397`, KIPAN `+0.005847`
- LGG 평균 변화 `+0.006363`, GBMLGG `+0.007998`

## 당시 채택 판단

- 3-seed 검증 통과: `jyp_f9 0.70 + em_v24 0.30`
- 근거:
  - 평균 Macro F1이 `+0.018421` 개선됨
  - 3개 seed와 15개 Fold에서 모두 F9보다 높음
  - seed 변동성이 감소함
  - bootstrap 95% CI 하한이 0보다 큼
- Test 예측은 F9 단독 대비 197건(`7.74%`) 변경됨
- 로컬 제출 파일의 2,546개 ID 순서, 라벨 집합, 결측 여부 검증 완료

## 실행 및 산출물

```bash
python -m src.validate_preprocessing_stability --config configs/test_003_xgb_blend.yaml
```

- 설정: `configs/test_003_xgb_blend.yaml`
- 결과: `data/processed/test_003_jypF9_EM24_w70_3seed.json`
- Fold 점수: `data/processed/test_003_jypF9_EM24_w70_3seed_scores.csv`
- OOF/Test 확률: `data/processed/test_003_jypF9_EM24_w70_3seed_probabilities.npz`
- 제출 후보: `data/processed/test_003_jypF9_EM24_w70_3seed_jypF9_EM24_w70_submission.csv`

제출 파일은 로컬에서만 생성했으며 외부 리더보드에는 자동 제출하지 않았다.
