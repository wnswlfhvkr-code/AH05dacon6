# pipeComb_em_v1 전처리·앙상블 설계

## 결합 결과

`pipeComb_em_v1`은 EMV45와 JSJ V9를 단순 열 결합하지 않는다. JSJ V9의
Word·Char TF-IDF는 텍스트 뷰로 유지하고, 트리 입력은 EMV45 피처만 사용한다.

| 입력 뷰 | 유지 내용 | 제거한 중복 |
|---|---|---|
| Text | gene, gene-consequence, 구체 변이, 아미노산 치환, 위치 구간 TF-IDF | 없음 |
| Tree | EMV45 severity, burden, consequence, hotspot, dual OOF signature, pattern uncertainty, pair contrast | JSJ의 변이 여부 행렬과 12개 단순 구조 요약 |
| Conflict | KIRC–KIPAN, LGG–GBMLGG 원본 레이블 쌍 | JSJ V9의 EM16·EM24 재앙상블(EMV45가 V24를 이미 포함) |

`GBMLGG`, `KIPAN`, `STES`를 분해하거나 병합하지 않으며 원본 `SUBCLASS`를
그대로 예측한다. 결측 유전자 셀은 WT와 동일하게 취급한다.

## 앙상블

| 구성원 | 입력 | 역할 | 초기 가중치 |
|---|---|---|---:|
| LinearSVC | TF-IDF | 희소한 구체 변이 토큰과 선형 암종 signature | 0.50 |
| XGBoost | EMV45 | 제한된 깊이의 비선형 피처 상호작용 | 0.30 |
| LightGBM | EMV45 | XGBoost와 다른 분할 방식으로 오차 다양성 확보 | 0.20 |

두 암종 쌍 전문가는 전역 앙상블의 top-2가 정확히 해당 쌍이고 두 클래스의
확률 합이 0.55 이상인 경우만 활성화한다. 이진 전문가 혼합 비율은 0.15로
제한하여 소수 클래스 구간의 과도한 보정을 막는다.

## 과적합 방지 규칙

1. 모든 TF-IDF vocabulary, 빈도 필터, hotspot, class signature, pattern lookup은
   outer fold의 train 부분에서만 학습한다.
2. EMV45의 학습 행 signature와 pattern 피처는 inner-fold OOF 값만 사용한다.
3. 트리는 얕은 depth, 작은 leaf 수, 높은 min-child, row/column subsampling,
   L1/L2 규제를 사용한다.
4. 전역 앙상블은 서로 다른 표현과 모델 계열을 섞고 한 모델의 가중치를
   0.60 이하로 제한한다.
5. 향후 가중치나 pair 임계값을 바꿀 때는 동일한 outer OOF 예측만 사용한다.
   test 예측 또는 제출 점수로 선택하지 않는다.
6. 후보 구성원의 OOF 오류 상관이 0.98 이상이면 더 낮은 Macro F1 구성원을
   제거하고, 앙상블이 best single model보다 개선되지 않으면 단일 모델을 유지한다.

## 실행 상태

- 파이프라인·모델·설정 파일만 생성했다.
- 학습, 교차검증 평가, test 예측, submission 생성은 수행하지 않았다.
- 초기 가중치는 검증 결과가 아니라 실행 가능한 설계 시작값이다.
