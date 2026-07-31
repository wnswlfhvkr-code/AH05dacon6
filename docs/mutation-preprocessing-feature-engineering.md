# 변이 문자열 전처리·피처 엔지니어링 제안

## 1. 작업 목적

이번 작업의 범위는 **모델 학습 전 단계인 변이 문자열 전처리와 구조적
피처 생성**까지입니다. 예측 모델과 제출 후처리는 포함하지 않습니다.

어제 진행한 세 번의 Public 리더보드 실험 결과는 다음과 같습니다.

| 구성 | OOF Macro F1 | Public Macro F1 |
| --- | ---: | ---: |
| 강한 확률 조정·클래스 보정 | 0.504178 | 0.366947 |
| 보정 없는 SVM·트리 앙상블 | 0.482546 | 0.357333 |
| 완만한 확률 조정·클래스 보정 | 0.505595 | **0.371498** |

유전자명과 변이 문자열만으로도 분류 신호가 있었지만, OOF와 Public 사이에
약 0.12~0.14의 차이가 나타났습니다. 따라서 같은 문자열 피처의 가중치만
더 조정하기보다, 표기를 안정적으로 통일하고 샘플의 전체 변이 구조를
설명하는 피처를 추가할 필요가 있습니다.

## 2. 팀원 분석에서 반영한 내용

팀원 분석은 6,201개 샘플, 26개 `SUBCLASS`, 4,384개 유전자 피처에서
`WT`가 아닌 값을 변이로 정의했습니다. 특징 유전자 후보 기준은 다음과
같습니다.

- 해당 암종 내부 변이율 10% 이상
- 해당 암종 내부 변이 건수 5건 이상
- 다른 암종 대비 변이율 차이 5%p 이상
- Lift 1.5 이상
- 빈도를 함께 고려한 log2 OR 순위

`log2 OR`은 특정 유전자 변이가 해당 암종에서 다른 암종보다 얼마나
특징적인지를 나타내는 로그 오즈비입니다. `log2 OR=1`이면 오즈가 약
2배, `log2 OR=2`이면 약 4배라는 의미입니다.

분석에서 확인된 대표 조합은 다음과 같습니다.

- `COAD(결장선암)-APC`
- `UCEC(자궁내막암)-PTEN·PIK3CA·CTNNB1`
- `LGG(저등급 신경교종)/GBMLGG(교모세포종·저등급 신경교종 통합군)-IDH1·ATRX`
- `KIRC(신장 투명세포암)/KIPAN(범신장암군)-VHL`
- `THCA(갑상선암)/SKCM(피부 흑색종)-BRAF`
- `LAML(급성 골수성 백혈병)-NPM1·IDH2`

이 목록은 고정된 정답표로 사용하지 않습니다. 실제 교차검증에서는
각 학습 fold에서 변이율, Lift와 log2 OR을 다시 계산하고 검증 fold에는
학습된 값만 적용합니다.

## 3. 전처리

### 표기 통일

- `WT`, 빈 문자열, `NA`, `N/A`를 변이 없음으로 통일
- 실제 결측은 `WT`와 합치지 않고 `MISSING`으로 보존
- 앞뒤 공백 제거
- `p.V600E`의 `p.`는 모델용 문자열에서 제거하되 단백질 표기 개수는 별도 보존
- 원본 데이터는 수정하지 않고 변환 결과만 새 피처로 생성

현재 데이터에는 `p.` 접두사와 앞뒤 공백이 발견되지 않았습니다. 다만 한
유전자 셀에 여러 변이가 공백으로 나열된 경우가 train 22,026건, test
63,178건 확인되었습니다. 따라서 내부 공백을 제거하지 않고 변이 이벤트
구분자로 사용합니다.

### 변이 문자열 분해

| 파생 정보 | 예시 |
| --- | --- |
| 이벤트 분리 | `L26V L24V → L26V`, `L24V` |
| 변이 유형 | `SUB`, `DEL`, `INS`, `DELINS`, `FS`, `DUP`, `STOP`, `OTHER` |
| 변이 위치 | `p.V600E → 600` |
| 위치 구간 | `001_050`, `051_100`, `101_250`, `251_500`, `501_PLUS` |
| 파싱 실패 | 삭제하지 않고 `unparsed_mutation_count`로 보존 |

## 4. 생성 피처

### 샘플 단위 변이 통계

- 전체 변이 수와 `WT` 수
- 전체 유전자 중 변이 비율
- 변이 유형별 개수와 비율
- 위치가 파싱된 변이와 파싱되지 않은 변이의 개수
- 변이 위치 평균·최댓값
- 위치 구간별 변이 개수와 비율

### 암종별 marker 피처

각 학습 fold에서 선별된 특징 유전자로 다음 피처를 생성합니다.

- 암종별 `marker_score`: 변이가 존재하는 유전자의 log2 OR 합
- 암종별 `marker_hits`: 해당 암종 marker가 적중한 개수
- 가장 높은 marker 점수
- marker 1위와 2위 점수 차이
- 선택된 marker 유전자 중 변이된 유전자 수

### 유전자 결합 피처

- `gene_APC_mutated`: APC 변이 존재 여부
- `gene_type_APC_STOP`: APC 종결 변이 존재 여부
- `gene_type_PTEN_SUB`: PTEN 치환 변이 존재 여부

유전자×변이 유형 피처는 모든 유전자에 만들지 않고, 학습 fold에서 선택된
marker 유전자에만 생성하여 차원 증가를 제한합니다.

## 5. 우선 배제하는 피처

- 변이 건수가 3~5건뿐인데 Lift만 높은 유전자
- 전체 데이터에서 한 번만 계산한 log2 OR
- 모든 유전자 쌍의 조합
- 지지도가 낮은 동시 변이
- 여러 암종에서 흔한 `TP53` 하나만으로 만든 암종 결정 규칙
- Public 점수에 맞춰 수동으로 조정한 클래스별 배율
- 규정 확인이 끝나지 않은 외부 유전자·경로 데이터

## 6. 누수 없는 사용 방법

```python
from sklearn.model_selection import StratifiedKFold
from src.mutation_features import MutationFeatureEngineer

splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, valid_index in splitter.split(features, labels):
    engineer = MutationFeatureEngineer()

    train_features = engineer.fit_transform(
        features.iloc[train_index],
        labels.iloc[train_index],
    )
    valid_features = engineer.transform(features.iloc[valid_index])

    # train_features로 모델 학습
    # valid_features로 검증
```

전체 데이터에서 `fit`한 뒤 CV를 나누면 검증 fold의 라벨 정보가 marker
선정에 들어가므로 사용하면 안 됩니다.

## 7. 검증 및 반영 기준

1. 3-fold로 피처 블록을 빠르게 선별합니다.
2. 개선된 피처만 `5-fold × 3 seeds`로 다시 검증합니다.
3. 평균 Macro F1이 0.003 이상 개선되는지 확인합니다.
4. 세 seed 중 두 개 이상에서 같은 방향으로 개선되는지 확인합니다.
5. 평균뿐 아니라 클래스별 F1과 fold별 표준편차를 함께 확인합니다.

반복 검증에서 개선되지 않는 피처는 제외하고, 안정적으로 개선되는
피처만 이후 `LinearSVC`, `LightGBM`, `XGBoost` 모델 실험에 전달합니다.
