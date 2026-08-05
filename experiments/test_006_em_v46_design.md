# EMV46: EMV45 + F22 중복 제거 결합 설계

`test_006_em_F22`는 별도의 `em_F22` 전처리 파이프라인이 아니라 `em_F20`을
사용한다. EMF20에는 EMV45가 이미 포함되어 있으므로 EMV45와 EMF20 전체를
그대로 결합하면 EMV45 피처가 두 번 생성된다.

EMV46은 다음처럼 구성한다.

```text
EMV45 1회
  +
F22가 사용한 EMF20의 F01~F19 고유 파생 피처 1회
```

- EMV45 원본 피처 중복: 제거
- F01~F15 비레이블 파생 피처: 유지
- F16~F19 레이블 기반 signature: inner-fold OOF 방식 유지
- F22 XGBoost 설정: `reg_lambda=17.0` 반영
- GBMLGG, KIPAN, STES: 원본 SUBCLASS 유지
- 외부 데이터와 test-fit: 사용하지 않음

구현과 설정만 갱신했으며 학습, 평가, submission 생성은 수행하지 않았다.
