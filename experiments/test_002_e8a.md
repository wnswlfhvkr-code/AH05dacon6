# test_002_e8a — E7·E8 cross-fit 보정 앙상블

## 실험 목적

E7 Balanced Logistic Regression의 안정성과 E8 TF-IDF·LinearSVC의 높은 성능을 결합했다. LinearSVC decision score를 temperature softmax로 보정한 후 E7 확률과 가중 결합했다.

## 검증 방식

- StratifiedGroupKFold 3-Fold × seeds `42, 2026, 777`
- 평가 Fold를 제외한 나머지 2개 Fold에서 temperature와 E8 weight 선택
- 선택한 파라미터를 제외된 평가 Fold에만 적용
- Test·Leaderboard 정보는 파라미터 선택에 미사용

## OOF 성능

| 모델 | OOF Macro F1 |
| --- | ---: |
| E7 | 0.439016 |
| E8 | 0.454945 |
| E8A | **0.467274 ± 0.006753** |

- E8A - E8: `+0.012330`
- E8A - E7: `+0.028258`

## Paired stratified bootstrap

| 비교 | 평균 변화 | 95% CI | P(변화 > 0) |
| --- | ---: | ---: | ---: |
| E8A - E8 | +0.012231 | [+0.007670, +0.017011] | 1.0000 |
| E8A - E7 | +0.028296 | [+0.020843, +0.035471] | 1.0000 |

## 로컬 재현 검증

GitHub 독립 실행기에서 Kaggle E7 확률 배열과 Kaggle E8 decision score 배열을 로드해 E8A를 다시 검증했다.

| 검증 | OOF Macro F1 |
| --- | ---: |
| Kaggle 고정 Fold E8A | 0.467274 ± 0.006753 |
| 로컬 재생성 SGKF E8A | 0.467164 ± 0.007186 |
| 로컬 - Kaggle | -0.000110 |

로컬 paired stratified bootstrap 결과는 다음과 같다.

| 비교 | 평균 변화 | 95% CI | P(변화 > 0) |
| --- | ---: | ---: | ---: |
| E8A - E8 | +0.012081 | [+0.007100, +0.017133] | 1.0000 |
| E8A - E7 | +0.028147 | [+0.020817, +0.035316] | 1.0000 |

Kaggle은 저장된 고정 Fold를 사용했고, 로컬 실행은 SGKF를 다시 생성했다. 로컬 Fold는 모든 fold가 `2,067명`이었지만 Kaggle 고정 Fold는 seed별 fold 크기가 달랐다. 따라서 두 점수의 미세한 차이는 모델 오류가 아니라 분할 차이로 해석한다.

서로 다른 SGKF 분할에서 OOF 성능 차이가 `0.000110`에 그쳤고, 두 검증 모두 E8·E7 대비 bootstrap 95% CI가 완전히 양수였다. 이를 통해 E8A 개선이 특정 seed·fold 분할에만 의존하지 않음을 확인했다.

## Test 예측 점검

- Test `2,546개`, 26개 암종 모두 예측
- E8 대비 예측 변화율: `0.080911`
- E7 대비 예측 변화율: `0.250196`
- 예측이 0개인 암종 없음

## 결론

E8A는 세 seed에서 E8보다 높은 OOF 성능을 기록했고, paired bootstrap 신뢰구간도 완전히 양수였다. Kaggle 고정 Fold와 로컬 재생성 SGKF에서 개선이 모두 재현되었다. Test에서는 E8 예측의 약 8.09%만 변경하여 E8의 특성을 대부분 유지했다. 따라서 E8A를 최종 앙상블 후보로 채택한다.

## 재현 명령

```bash
python -m src.train_ensemble_jh_e8a --config configs/test_002_e8a.yaml
```

E7 확률 배열과 E8 decision score 배열은 `data/processed` 하위에 필요하며, 해당 결과 파일은 `.gitignore` 대상이다.
