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
2. `configs/xgboost_baseline.yaml`에 베이스라인의 데이터 분할·시드·모델 조건을 기록합니다.
3. 전처리·학습·평가 코드를 `src/`에 구현합니다.
4. 결과를 `experiments/baseline.md`에 기록합니다.
5. 기능별 브랜치에서 작업하고 Pull Request로 검토합니다.

## 베이스라인 실행

데이터 파일 `train.csv`, `test.csv`, `sample_submission.csv`을 `data/raw/`에 둔 뒤 아래 명령을 실행합니다.

```bash
python -m src.train --config configs/xgboost_baseline.yaml
```

검증 Macro F1은 화면과 `data/processed/xgboost_baseline_metrics.json`에 기록되며, 제출 파일은 `data/processed/xgboost_baseline_submission.csv`에 생성됩니다.

새 모델은 `src/models/`에 생성 함수를 추가하고, `src/models/__init__.py`의 `MODEL_BUILDERS`에 등록한 뒤 `configs/`에 해당 모델의 설정 파일을 추가합니다.

## 실행 환경

Python 3.14를 기준으로 개발과 자동 테스트를 실행합니다.
