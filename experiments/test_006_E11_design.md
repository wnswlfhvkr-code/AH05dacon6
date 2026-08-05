# test_006 E11~E13 설계

세 파이프라인 모두 EMV16의 consequence severity·burden·inner-fold OOF class
signature와 E7의 동일 패턴 레이블 불확실성을 한 번만 생성한다.

| 실험 | 결합 | 고유 추가 피처 | 중복 제거 방식 |
| --- | --- | --- | --- |
| E11 | E7 + E3 | token event·multi-hit·token excess·multi-hit ratio | EMV16 burden을 다시 만들지 않고 token 복잡도만 추가 |
| E12 | E7 + E6 | KIRC–KIPAN, LGG–GBMLGG signature 차이 | 기존 signature를 복제하지 않고 차이만 추가 |
| E13 | E7 + E5 | 여러 fold에서 재현된 hotspot | EMV16 hotspot과 별도 추가하지 않고 안정 선택으로 교체 |

E7의 support·entropy·최대 클래스 확률·충돌 여부는 학습 행 자신의 레이블을
직접 보지 않도록 inner-fold OOF로 생성한다. validation/test에는 각 외부 학습
fold에서 고정된 lookup만 적용한다.

기준 모델은 이전 E 실험과 동일한 XGBoost이며 설계 단계에서는 학습·평가하지 않는다.
