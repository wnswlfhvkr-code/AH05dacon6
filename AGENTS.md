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
