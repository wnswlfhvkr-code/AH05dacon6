# em_v38: em_v15 + em_v16 OOF 구조 분석

## OOF 적용 위치

| 구분 | em_v15 | em_v16 / em_v38 |
|---|---|---|
| 바깥쪽 모델 평가 | `evaluation_folds=5` | `evaluation_folds=5` |
| 평가 fold별 전처리 재학습 | 적용 | 적용 |
| 학습 표본 signature 교차 적합 | 미적용 | `fit_transform()`의 inner fold에서 적용 |
| validation/test signature | outer/full train 가중치 사용 | outer/full train 가중치 사용 |

`em_v15`의 OOF는 모델 일반화 성능을 추정하기 위한 outer OOF다. 각 outer
validation fold는 학습에 사용되지 않으므로 평가 누수는 없다. 다만 outer train
표본의 signature는 같은 outer train 전체 레이블로 만들어져 모델 학습 피처에
자기 레이블 영향이 남을 수 있다.

`em_v16`은 outer OOF 정책을 유지하면서 학습 표본의 signature만 inner-fold로
교차 적합한다. 각 학습 표본은 자신을 제외한 inner train에서 학습한 유전자
가중치로 signature를 받는다. outer validation과 test는 해당 학습 데이터 전체로
학습한 가중치로 변환한다.

## em_v38 중복 제거 결정

- outer OOF는 `evaluation_folds=5` 한 번만 선언한다.
- signature inner OOF는 `fit_transform()`에서 한 번만 생성한다.
- severity, consequence summary, mutation burden, hotspot과 signature 열은 각각 한
  세트만 유지한다.
- `em_v15`와 `em_v16`을 상속하거나 import하지 않고 `PreprocessingPipeline`을 직접
  상속한다.
- 원본 `SUBCLASS`는 변경하지 않는다.

학습, 평가 및 submission 생성은 수행하지 않았다.
