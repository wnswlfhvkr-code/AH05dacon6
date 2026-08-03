# NCI GDC TCGA 코드 기반 전처리 케이스: em_v26~em_v27

## 1. 참고한 공식 자료

- [NCI GDC TCGA Resources](https://gdc.cancer.gov/about-data/gdc-data-processing/resources-tcga-users)
- [NCI GDC TCGA Study Abbreviations](https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations)

GDC는 TCGA Study Abbreviation이 일반적으로 암종별로 구성된 개별 TCGA 연구를 식별한다고 설명한다. TCGA barcode는 biospecimen의 주요 식별자이고 batch·sample·center 정보를 해석하는 데 사용되지만, 현재 데이터의 `ID`는 익명화되어 있어 barcode 파생 피처를 만들 근거가 없다.

## 2. train.csv에서 확인한 사항

| 항목 | 값 |
| --- | ---: |
| 학습 표본 | 6,201 |
| 원본 유전자 피처 | 4,384 |
| SUBCLASS | 26 |
| 비-WT 셀 | 218,893 |
| 복수 토큰 셀 | 22,026 |
| 최소 클래스 | DLBC 38 |
| 최대 클래스 | BRCA 786 |

26개 중 23개는 GDC Study Abbreviation 표의 단일 study code와 일치한다. `GBMLGG`, `KIPAN`, `STES`는 GDC 표의 단일 study code가 아니므로 외부 정의를 이용해 하위 암종으로 추정 분해하지 않는다.

## 3. 공통 레이블 원칙

- 모델 정답은 원본 26개 flat `SUBCLASS`다.
- `GBMLGG`, `KIPAN`, `STES`는 각각 독립 클래스다.
- 보조 연구군은 signature 생성에만 사용하며 `LabelEncoder` 입력은 변경하지 않는다.
- supervised signature는 실제 학습 시 inner-fold OOF로 생성하도록 구현했다.
- 이번 작업에서는 실제 `train.csv` 학습과 성능 평가를 실행하지 않았다.

## 4. em_v26: TCGA study-family coarse signature

공식 study name에서 추론 가능한 인접 해부학 연구군을 보조 타깃으로 사용한다.

예시:

- central nervous system: `GBMLGG`, `LGG`
- kidney: `KIPAN`, `KIRC`
- thoracic: `LUAD`, `LUSC`, `THYM`
- digestive: `COAD`, `LIHC`, `PAAD`, `STES`
- hematolymphoid: `DLBC`, `LAML`
- endocrine/neuroendocrine: `ACC`, `PCPG`, `THCA`

단일 SUBCLASS만 포함하는 연구군은 기존 one-vs-rest class signature와 같아 별도 family signature를 만들지 않는다.

생성 피처:

- 최소 변이 빈도와 상한을 적용한 원시 유전자 mutation presence
- `mutation_burden_log1p`
- 연구군별 안정화 log2 odds weighted signature
- 연구군별 burden-normalized match rate

목적은 먼저 넓은 조직 기원 신호를 제공해 서로 멀리 떨어진 암종 사이의 혼동을 줄이는 것이다.

## 5. em_v27: related-study fine contrast

같은 보조 연구군 안에서만 각 원본 클래스를 sibling class와 비교한다.

예시:

- `GBMLGG` 대 `LGG`
- `KIPAN` 대 `KIRC`
- `LUAD` 대 `LUSC`·`THYM`
- `STES` 대 `COAD`·`LIHC`·`PAAD`

생성 피처:

- 최소 변이 빈도와 상한을 적용한 원시 유전자 mutation presence
- `mutation_burden_log1p`
- 원본 클래스별 one-vs-sibling weighted signature
- 원본 클래스별 burden-normalized sibling match rate

목적은 one-vs-all signature가 놓칠 수 있는 해부학적으로 가까운 연구 코호트 사이의 작은 차이를 강조하는 것이다. `GBMLGG`, `KIPAN`, `STES`의 이름과 정답 값은 그대로 유지된다.

## 6. 구현 파일과 상태

| 버전 | 파일 | 등록 | 실제 학습·평가 |
| --- | --- | --- | --- |
| em_v26 | `src/pipelines/pipeline_em_v26.py` | 완료 | 수행하지 않음 |
| em_v27 | `src/pipelines/pipeline_em_v27.py` | 완료 | 수행하지 않음 |

두 버전은 다른 버전 파이프라인을 상속하지 않고 `PreprocessingPipeline`만 직접 상속한다. 합성 데이터 테스트에서 protected composite label의 encode/decode 보존과 OOF 피처 컬럼 구성을 확인한다.
