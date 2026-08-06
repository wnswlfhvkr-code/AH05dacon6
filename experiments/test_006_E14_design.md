# test_006 E14 설계

E14는 EMV16에 E7, E3, E6의 고유 피처를 결합한다.

| 구성 | 생성 피처 | 중복 제거 |
| --- | --- | --- |
| EMV16 | consequence severity·burden·OOF class signature·hotspot | 공통 피처는 한 번만 생성 |
| E7 | 동일 패턴 label support·entropy·최대확률·충돌 여부 | 학습 행은 inner-fold OOF, validation/test는 고정 lookup |
| E3 | token event·multi-hit 유전자·token excess·최대 token·비율 | 기존 mutation burden을 다시 생성하지 않음 |
| E6 | KIRC–KIPAN, LGG–GBMLGG weighted/count signature 차이 | 기존 class signature를 복제하지 않음 |

기준 모델은 이전 E 실험과 동일한 XGBoost다. 이 단계에서는 데이터 전처리 fit,
모델 학습, OOF 평가와 submission 생성을 실행하지 않는다.

