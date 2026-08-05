# pipeComb_em_v1_001 Weighted Soft Voting 설계

## 변경 목적

Train Macro F1이 과도하게 높았던 고용량 서브 모델의 영향력을 낮추고,
희귀 피처 제한과 강한 모델 규제를 적용한 E14 전용 모델의 확률 비중을 높인다.
학습 피처를 단일 행렬에 반복 결합하지 않고 모델별 뷰로 분리한다.

## 최종 유효 가중치

상위 soft voting에서 broad ensemble 0.40, regularized E14 0.60을 사용한다.
broad 내부 가중치를 반영한 실제 최종 비율은 다음과 같다.

| 서브 모델 | 피처 뷰 | 유효 가중치 |
|---|---|---:|
| TF-IDF LinearSVC | JSJ 텍스트 | 0.14 |
| XGBoost | EMV45 | 0.16 |
| LightGBM | EMV45 | 0.10 |
| 강한 규제 XGBoost | E14 전용 | 0.60 |

각 모델의 `predict_proba`를 동일한 클래스 순서로 정렬한 뒤 위 비율로 더하고,
행별 합이 1이 되도록 다시 정규화한다. hard voting은 사용하지 않는다.

## 과적합 억제

- TF-IDF 최소 문서 빈도를 높이고 최대 vocabulary 수를 축소한다.
- broad 트리의 depth·leaf·tree 수를 줄이고 min-child와 L1/L2를 높인다.
- E14는 `min_mutation_count=8`, class별 signature 유전자 12개로 제한한다.
- E14 signature는 smoothing과 shrinkage를 높이고 log2 odds 상한을 낮춘다.
- pair specialist는 확률 합 0.70 이상에서만 활성화하고 보정 비율을 0.05로 제한한다.
- 모든 전처리 학습은 outer fold의 train 부분에서만 수행한다.
- `GBMLGG`, `KIPAN`, `STES`는 분해하거나 병합하지 않는다.

## 중복 제거 기준

- 기존 pipeComb 구현을 composition으로 재사용하여 TF-IDF·EMV45 코드를 복사하지 않는다.
- E14 피처는 broad 행렬에 이어 붙이지 않고 E14 모델에만 전달한다.
- 기존 EM16·EM24 팀 재앙상블은 추가하지 않는다.
- 같은 피처 행렬 안의 중복 열은 0개다.

## 실행 상태

- 파이프라인, 모델, 설정, 설계 문서만 생성했다.
- 학습, OOF 평가, test 예측, submission 생성은 수행하지 않았다.
- 가중치는 검증 결과가 아닌 다음 실험을 위한 사전 고정값이다.
