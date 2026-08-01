# AH05dacon6

암종 분류 모델 개발 프로젝트입니다. GitHub에는 코드, 설정, 테스트, 실험 조건을 기록하고 원본 데이터와 대용량 모델은 별도 저장소에서 관리합니다.

## 프로젝트 구조

- `src/`: 전처리·학습·평가·추론 코드
- `configs/`: 재현 가능한 모델 및 학습 설정
- `data/`: 원본·전처리 데이터의 저장 위치와 안내 문서
- `notebooks/`: 탐색적 분석 및 빠른 검증
- `tests/`: 전처리와 모델 입출력 검증
- `experiments/`: 실험 조건과 결과 기록
- `models/`: 모델 버전 및 외부 저장 위치 기록
- `docs/`: 데이터 사전과 프로젝트 문서
- `.github/workflows/`: Pull Request 자동 검사

## 작업 순서

1. 원본 데이터는 `data/raw/`에 준비하고 GitHub에는 커밋하지 않습니다.
2. `configs/baseline.yaml`에 베이스라인의 데이터 분할·시드·모델 조건을 기록합니다.
3. 전처리·학습·평가 코드를 `src/`에 구현합니다.
4. 결과를 `experiments/baseline.md`에 기록합니다.
5. 기능별 브랜치에서 작업하고 Pull Request로 검토합니다.

## 베이스라인 실행

데이터 파일 `train.csv`, `test.csv`, `sample_submission.csv`을 `data/raw/`에 둔 뒤 아래 명령을 실행합니다.

```bash
python -m src.train --config configs/baseline.yaml
```

검증 Macro F1은 화면과 `data/processed/baseline_metrics.json`에 기록되며, 제출 파일은 `data/processed/baseline_submission.csv`에 생성됩니다.

새 모델은 `src/models/`에 생성 함수를 추가하고
`src/models/__init__.py`의 `MODEL_BUILDERS`에 등록합니다. 팀원별 실험
설정은 `configs/test_001.yaml`부터 `test_004.yaml`까지 분리하여 사용합니다.

현재 등록된 모델은 `xgboost`, `lightgbm`, `linear_svc`,
`wc_tfidf_lsvc_lgbm`입니다. 정세준 실험은 `test_004.yaml`에서 관리합니다.

```bash
python -m src.train --config configs/test_004.yaml
```

`wc_tfidf_lsvc_lgbm`은 기존 Public 0.3714981583 제출의 핵심 구성인
Word+Char TF-IDF, LinearSVC 95%, 트리 모델 5%, 클래스 보정을
협업 저장소에서 다시 학습할 수 있도록 구성한 버전입니다. 기존 제출은
여러 OOF 산출물을 결합했으므로 새 실행 결과가 기존 제출 파일과 완전히
같다고 가정하지 않으며, 동일한 검증 조건에서 다시 비교해야 합니다.

## 데이터 품질 점검

```bash
python -m src.data_quality --config configs/baseline.yaml
```

## 전처리 파이프라인

각 팀원 설정 파일의 `preprocessing.name`에서 전처리 파이프라인을
선택합니다. 현재 `baseline`은 상수 피처 제거, 범주형 순서 인코딩,
타깃 레이블 인코딩을 적용합니다. 새 파이프라인은 `src/pipelines/`에
추가하고 레지스트리에 등록한 뒤, `preprocessing.name`만 바꿔 같은
모델 조건에서 비교합니다.

사용 가능한 파이프라인은 `baseline`, `em_v1`, `jsj_v1`~`jsj_v7`입니다.
현재 `configs/test_004.yaml`은 XGBoost용 압축 구조 피처인 `jsj_v2`를 사용합니다.

## TEST_004 제출 재현

정세준의 제출 5건은 `configs/test_004_1.yaml`부터
`configs/test_004_5.yaml`까지 분리되어 있습니다.

```bash
python -m src.reproduce_test_004 --config configs/test_004_5.yaml
```

모델 사양은 `src/models/baseline/TEST_004_n.py`, 전처리는
`src/pipelines/pipeline_jsj_v3.py`부터 `pipeline_jsj_v7.py`에서 확인합니다.
전체 실행 매핑은 `docs/test-004-reproduction.md`에 정리되어 있습니다.

## 실행 환경

Python 3.14를 기준으로 개발과 자동 테스트를 실행합니다.
