# Project workspace rules

- 이 저장소와 관련된 모든 작업의 프로젝트 루트는 `/Users/admin/project/oz/AH05dacon6`이다.
- 모든 셸 명령, Python 실행, 테스트 및 Git 명령은 위 프로젝트 루트를 작업 디렉터리로 사용한다.
- 생성하거나 수정하는 소스 코드, 설정, 문서, 모델 및 결과 파일은 모두 위 프로젝트 폴더 하위에 저장한다.
- 외부 경로의 원본 파일은 사용자가 명시적으로 요청하지 않는 한 직접 수정하지 않는다. 필요한 경우 프로젝트 폴더 하위로 복사한 뒤 작업한다.
- 임시 파일이 필요한 경우에만 시스템 임시 디렉터리를 사용할 수 있으며, 최종 산출물은 반드시 프로젝트 폴더 하위에 반영한다.

## Scope

이 파일은 현재 디렉터리와 모든 하위 디렉터리에 적용된다.

## Preprocessing pipeline inheritance

- 모든 버전별 전처리 파이프라인은 `PreprocessingPipeline` 공통 기반 클래스만 직접 상속한다.
- `EMV2PreprocessingPipeline`이 `EMV1PreprocessingPipeline`을 상속하는 것처럼 버전별 파이프라인끼리 상속하지 않는다.
- 버전별 구현은 해당 `pipeline_em_vN.py` 파일 안에서 독립적으로 이해하고 실행할 수 있어야 한다.
- `base.py`에는 상수 열 제거, 범주형 인코딩, 레이블 인코딩처럼 모든 버전에 적용되는 기반 기능만 둔다.
- 변이 행렬, 변이 빈도 선택, burden, consequence, signature처럼 실험 버전별 기능은 여러 버전에서 사용하더라도 해당 `pipeline_em_vN.py` 안에 각각 기록한다.

## SUBCLASS label integrity

- 모든 전처리·학습 파이프라인은 `train.csv`에 제공된 `SUBCLASS` 값을 서로 배타적인 원본 클래스 그대로 유지한다.
- `GBMLGG`, `KIPAN`, `STES`는 각각 하나의 독립 클래스이며 하위 암종으로 분해하지 않는다.
- `GBMLGG`를 `GBM`·`LGG`로, `KIPAN`을 `KICH`·`KIRC`·`KIRP`로, `STES`를 `STAD`·`ESCA`로 재매핑하거나 추정 분할하지 않는다.
- 사용자가 명시적으로 레이블 체계 변경을 요청하지 않는 한 클래스 병합, 계층화, 이름 변경 또는 외부 임상 분류에 따른 타깃 재구성을 수행하지 않는다.

## Competition evaluation and participation rules

- 공식 평가 지표는 `Macro F1 Score`다. 전체 클래스의 F1을 동일한 비중으로 평가하므로 전체 정확도보다 클래스별 성능과 소수 클래스 성능을 우선 확인한다.
- Public 평가는 test 데이터 100%를 사용한다. 별도의 공개/비공개 test 분할을 가정하지 않는다.
- 개인 또는 팀으로 참여할 수 있으며, 참여 형태는 모델·전처리 구현 규칙을 변경하지 않는다.

## External data and pretrained models

- 외부 데이터는 사용할 수 없다. 공개 여부와 무관하게 대회가 제공하지 않은 환자 데이터, 임상 메타데이터, 유전자 주석값, 단백질 서열, ESM 유전자 임베딩, pathway·분자 상호작용 네트워크 등에서 새로운 입력 피처를 만들지 않는다.
- 사전학습 모델은 사용할 수 있다. 다만 모델 입력은 대회가 제공한 train/test 데이터에서만 만들어야 하며, 별도의 외부 참조 데이터를 결합하지 않는다.
- 사전학습 모델의 가중치 사용 가능과 외부 데이터 기반 파생 피처 사용 가능을 동일하게 해석하지 않는다.
- OncoBERT처럼 외부 단백질 서열로 생성한 ESM 임베딩이 추가로 필요한 구성과 MutationProjector처럼 외부 분자 네트워크가 필요한 구성은 외부 데이터 금지 조건에 맞지 않는 것으로 취급한다.
- 사전학습 모델 사용을 제안하거나 구현할 때는 모델 가중치 외에 외부 데이터·외부 임베딩·외부 네트워크가 필요한지 먼저 점검하고 결과를 사용자에게 알린다.

## Test data leakage prevention

- test 데이터는 최종 예측을 위한 `transform`과 `predict`에만 사용한다. 모델, 전처리기, 통계량, 임계값 또는 후처리 규칙을 학습·선택하는 데 사용하지 않는다.
- label encoding은 train의 레이블만으로 fit한다.
- one-hot/ordinal encoding과 범주 사전은 train에만 fit하고 test에는 동일 객체로 transform한다. test에 `pd.get_dummies()`를 별도로 적용하지 않는다.
- 결측치 대체값, 스케일링 평균·표준편차, clipping 경계와 같은 통계량은 train에서만 계산한다.
- 최소 변이 빈도, 상수 열, 상관관계 제거, hotspot, consequence 사전, class signature, 피처 선택, 차원 축소와 같은 모든 전처리 상태는 train에서만 결정한다.
- OOF 평가에서는 위 전처리 상태를 각 fold의 학습 부분에만 fit하고 validation 부분에는 transform만 적용한다.
- 최종 제출에서는 선택이 끝난 고정 파이프라인을 전체 train에 다시 fit한 뒤 test에 한 번 transform/predict한다.
- test 전체의 패턴 빈도, 클래스 수 추정, 분포 또는 ID 순서를 이용해 레이블을 배분하거나 임계값을 조정하는 transductive 후처리를 사용하지 않는다.
- train/test를 합쳐 인코더, 스케일러, 결측치 처리기, 비지도 차원 축소 또는 피처 선택기를 fit하지 않는다.
