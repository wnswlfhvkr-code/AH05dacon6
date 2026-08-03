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

새 모델은 `src/models/`에 생성 함수를 추가하고, `src/models/__init__.py`의 `MODEL_BUILDERS`에 등록한 뒤 `configs/`에 해당 모델의 설정 파일을 추가합니다.

## 데이터 품질 점검

```bash
python -m src.data_quality --config configs/baseline.yaml
```

## 전처리 파이프라인

설정 파일의 `preprocessing.name`에서 전처리 파이프라인을 선택합니다. 현재 `baseline`은 상수 피처 제거, 범주형 순서 인코딩, 타깃 레이블 인코딩을 적용합니다. 순서 인코더는 학습 데이터에 없는 값을 `-1`로 변환합니다. 새 파이프라인은 `src/pipelines/`에 추가한 뒤 설정 파일의 이름만 바꿔 같은 모델 조건에서 비교합니다.

연구 파이프라인은 `em_v1~em_v28`이 등록되어 있습니다. `em_v15`는 5-fold OOF 평가를, `em_v16`은 학습 표본의 자기 정답 영향을 줄이는 inner-fold OOF signature를 적용합니다. `em_v17`은 셀 문자열 분리와 중복 토큰 제거 후 유전자별 mutation count, consequence, recurrent hotspot, multi-hit 피처를 생성합니다. `em_v18`은 동의 변이를 기능 변이 기반 유전자 선택·signature·hotspot에서 분리하며, `em_v19`는 em_v16 E4와 em_v17의 비중복 token 피처를 결합합니다. `em_v20`은 기능 변이 지지도 5/8/10을 fold 내부에서 선택하고 안정 유전자에만 consequence one-hot을 적용합니다. `em_v21`은 고차원 원시 변이와 불안정 hotspot을 축소합니다. `em_v22`는 em_v16에 em_v26의 TCGA study-family coarse signature를, `em_v23`은 em_v16에 em_v27의 관련 study 내부 fine contrast를 중복 없이 결합합니다. `em_v24`는 em_v16의 전체 변이 신호와 em_v18의 기능 변이 신호를 단일 consequence 처리와 OOF 분할 안에서 분리해 결합합니다. `em_v25`는 em_v16의 OOF 암종 signature와 em_v17의 token count·multi-hit·hotspot을 단일 토큰 처리 흐름에서 결합하고 구조적으로 중복된 고상관 피처를 제거합니다. `em_v28`은 em_v16의 OOF signature와 em_v20의 기능 변이 지지도·안정 유전자 consequence one-hot을 중복 없는 단일 흐름으로 결합합니다. `em_v26`과 `em_v27`을 포함한 모든 TCGA 기반 파이프라인은 원본 26개 레이블을 변경하지 않습니다. 사용할 버전은 설정 YAML의 `preprocessing.name`으로 선택합니다.

`em_v15` 이후 학습 결과에는 5-fold OOF 최종 Macro F1과 fold 평균·표준편차가 기록됩니다. 모든 버전에서 동일한 80% 학습 점수, 20% 검증 점수, 두 점수의 차이와 과적합 여부도 함께 출력됩니다.

선정된 `em_v16` E4 실험 조건은 다음 설정으로 실행합니다.

```bash
# E4: OOF Macro F1과 fold 안정성이 가장 좋은 조건
python -m src.train --config configs/test_005.yaml
```

`test_005.yaml`은 OOF와 80/20 검증에서 조기 종료를 적용하고, 전체 데이터 최종 학습에서는 fold별 최적 트리 수의 중앙값으로 모델을 다시 학습합니다.

토큰 기반 `em_v17`은 다음 설정으로 실행하며, 결과는 `experiments/em_v17_token_feature_experiments.md`에 기록됩니다.

```bash
python -m src.train --config configs/test_007.yaml
```

동의 변이와 기능 변이를 분리하는 `em_v18`은 다음 설정으로 실행합니다.

```bash
python -m src.train --config configs/test_008.yaml
```

토큰 분포, OOF·과적합 결과와 생식세포 해석 제한은
`experiments/em_v18_functional_consequence_experiment.md`에 기록됩니다.

지지도 fold 내부 선택, 안정 유전자 one-hot과 E4형 조기 종료를 적용한 `em_v20`은
다음 설정으로 실행합니다.

```bash
python -m src.train --config configs/test_010.yaml
```

em_v16 E4와 em_v17 결합 실험은 다음 설정으로 실행하며, 결과는
`experiments/em_v19_em_v16_e4_plus_em_v17_experiments.md`에 기록됩니다.

```bash
python -m src.train --config configs/test_009.yaml
```

## 실행 환경

Python 3.14를 기준으로 개발과 자동 테스트를 실행합니다.
