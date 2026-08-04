# test_006 E1~E10 최종 권장 전처리 실험 설계

모든 실험은 동일한 XGBoost 설정을 사용해 전처리 차이만 비교한다. 모든 상태는
각 학습 fold에서만 fit하며 test는 transform과 predict에만 사용한다. 타깃은
원본 `SUBCLASS` 26개를 유지하고 `GBMLGG`, `KIPAN`, `STES`를 분해하지 않는다.

| 실험 | 파이프라인 | EMV16에 결합하는 고유 처리 | 기대 효과 | 주요 위험 |
| --- | --- | --- | --- | --- |
| E1 | `em_E1` | 없음: severity·burden·OOF signature·hotspot 기준선 | 후속 실험 비교 기준 | 고차원 희소 피처 과적합 |
| E2 | `em_E2` | 동일 원시 변이 패턴의 train support와 재현 여부 | 반복 패턴의 신뢰도 표현 | test 고유 패턴에는 효과 없음 |
| E3 | `em_E3` | token 수·multi-hit 유전자 수·초과 token·비율 | 복합 변이와 다중 타격 구분 | 문자열 표기 잡음 민감성 |
| E4 | `em_E4` | train에서 support와 lift를 만족한 co-mutation pair | XGBoost가 희소 상호작용을 쉽게 학습 | 후보 수가 많으면 과적합 |
| E5 | `em_E5` | 여러 train 내부 fold에서 재현된 hotspot만 유지 | 불안정한 희귀 hotspot 축소 | 엄격한 기준에서 신호 손실 |
| E6 | `em_E6` | KIRC–KIPAN, LGG–GBMLGG signature 차이 | 중첩 레이블 경계에 직접적인 대비 신호 | 두 쌍 이외 클래스에는 제한적 |
| E7 | `em_E7` | 동일 패턴 레이블 support·entropy·최대확률·충돌 여부를 OOF 생성 | 레이블 충돌 표본의 불확실성 표현 | 중복이 적은 fold에서는 희소함 |
| E8 | `em_E8` | 클래스 판별 유전자 합집합과 빈도 상위 유전자로 raw gene cap | 차원·분산·학습시간 감소 | cap이 작으면 희귀 클래스 신호 손실 |
| E9 | `em_E9` | train에서 완전히 같은 변이 벡터의 유전자 열 하나만 유지 | 동일 정보 중복 제거 | 완전 중복이 적으면 변화가 작음 |
| E10 | `em_E10` | E2·E3·E4·E5·E6·E8·E9를 중복 없이 compact 통합 | 재현성·상호작용·안정성과 차원 축소 동시 검증 | 복합 변경이라 단일 효과 해석 어려움 |

## 공통 XGBoost 기준

- `objective=multi:softprob`
- `n_estimators=600`, `learning_rate=0.03`, `max_depth=3`
- `min_child_weight=5`, `subsample=0.80`, `colsample_bytree=0.65`
- `reg_alpha=1`, `reg_lambda=10`, `tree_method=hist`
- 평가 지표는 Macro F1이며, 설정 생성 단계에서는 학습·평가를 실행하지 않는다.

## 중복 제거 원칙

- E5의 안정 hotspot은 EMV16 hotspot을 추가로 복제하지 않고 선택 방식을 교체한다.
- E6은 기존 signature를 복제하지 않고 두 signature의 차이만 추가한다.
- E8은 선택된 유전자에 같은 raw severity 열을 두 번 만들지 않는다.
- E9는 완전 중복 유전자 중 첫 열만 유지한다.
- E10은 동일 기능을 한 번씩만 적용하며 E7의 label-posterior 피처는 과적합 위험 때문에 제외한다.
