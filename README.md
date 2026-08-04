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

모든 XGBoost 실행은 `device: cuda`를 명시하며, 중앙 모델 팩토리도 CUDA를
기본값으로 사용합니다. CPU 실행이 꼭 필요한 예외만 설정에서 `device: cpu`로
명시적으로 덮어씁니다.

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

기본·EM·JSJ 파이프라인은 `baseline`, `em_v1`~`em_v14`, `jsj_v1`~`jsj_v8`입니다.
현재 `configs/test_004.yaml`은 XGBoost용 압축 구조 피처인 `jsj_v2`를 사용합니다.

`em_v14`는 기능 결과·암종 signature에 학습 Fold에서 반복 관측된 hotspot
변이를 추가합니다. `configs/test_001.yaml`에서 선택할 수 있습니다.

### JYP test_003

`configs/test_003.yaml`은 모델 `xgboost`와 최종 전처리
`pipeComb_v4`를 사용합니다. JYP 전처리는
`src/pipelines/jyp_preprocessing/`에 구성되며 설정의
`preprocessing.name`으로 선택합니다. 단계별 JYP 파이프라인은 서로 독립이고,
보존하는 결합 파이프라인은 `pipeComb_v3`, `pipeComb_v4` 두 개입니다.

- `jyp_raw`, `jyp_f0`, `jyp_f0_no_raw`, `jyp_f1`, `jyp_f2`
- `jyp_f3_position`, `jyp_f3`, `jyp_f3_no_raw`, `jyp_f4`, `jyp_f4_no_raw`
- `jyp_f5`, `jyp_f5_no_raw`, `jyp_f5_no_raw_missmask`, `jyp_f5_selective_no_raw`
- `jyp_f6`, `jyp_f7`, `jyp_f8`, `jyp_f9`, `jyp_f10`, `jyp_f11`
- `pipeComb_v3`: `jyp_f9` + EM24 전체·기능 변이 dual signature 104개
- `pipeComb_v4`: `jyp_f9` + EM24 전체·기능 변이 weighted signature 52개

`pipeComb_v3`는 weighted와 match-count를 모두 보존한 D104 결과이고,
`pipeComb_v4`는 두 채널의 weighted만 남긴 W52 그룹 안전 검증 결과입니다. 둘 다
F9와 EM24의 피처를 내부 결합하며 확률 앙상블은 포함하지 않습니다.
v4는 입력 변이 프로필에서 그룹을 자동 생성해 F9와 EM24의 내부 OOF를
같은 그룹 경계로 분리합니다. 외부에서 `groups`를 전달하면 해당 값을 우선하며,
`auto_profile_groups: false`로 기존 행 단위 내부 OOF를 명시적으로 복원할 수 있습니다.
Public LB는 v3 `0.3390097657`, v4 `0.3342990633`으로 v4가
`-0.0047107024` 낮아 최종 제출 우선순위는 v3로 유지합니다.
v3 재현 설정은 `configs/test_003_pipecomb_v3.yaml`에 따로 보존합니다.
상세 실험 기록은 `experiments/jyp_preprocessing/test_003_pipeComb_v3.md`와
`experiments/jyp_preprocessing/test_003_pipeComb_v4.md`에 있습니다.

별도 실행기 없이 공용 학습 명령을 사용합니다.

```bash
python -m src.train --config configs/test_003.yaml
```

## TEST_004 제출 재현

정세준의 제출 6건은 `configs/test_004_1.yaml`부터
`configs/test_004_6.yaml`까지 분리되어 있습니다.

```bash
python -m src.reproduce_test_004 --config configs/test_004_6.yaml
```

연구 파이프라인은 `em_v1~em_v43`이 등록되어 있습니다. `em_v15`는 5-fold OOF 평가를, `em_v16`은 학습 표본의 자기 정답 영향을 줄이는 inner-fold OOF signature를 적용합니다. `em_v17`은 셀 문자열 분리와 중복 토큰 제거 후 유전자별 mutation count, consequence, recurrent hotspot, multi-hit 피처를 생성합니다. `em_v18`은 동의 변이를 기능 변이 기반 유전자 선택·signature·hotspot에서 분리하며, `em_v19`는 em_v16 OOF signature와 안정 consequence를 결합합니다. `em_v20`은 기능 변이 지지도를 fold 내부에서 선택하고 안정 유전자에만 consequence one-hot을 적용합니다. `em_v21`은 고차원 원시 변이와 불안정 hotspot을 축소합니다. `em_v22`는 em_v16에 em_v26의 TCGA study-family coarse signature를, `em_v23`은 em_v16에 em_v27의 관련 study 내부 fine contrast를 중복 없이 결합합니다. `em_v24`는 em_v16의 전체 변이 신호와 em_v18의 기능 변이 신호를 단일 consequence 처리와 OOF 분할 안에서 분리해 결합합니다. `em_v25`는 em_v16의 OOF 암종 signature와 em_v17의 token count·multi-hit·hotspot을 단일 토큰 처리 흐름에서 결합하고 구조적으로 중복된 고상관 피처를 제거합니다. `em_v28`은 em_v16의 OOF signature에 em_v21의 원시 severity 상한·fold 안정 hotspot·비율 중심 compact summary를 중복 없이 결합합니다. `em_v29`는 em_v19에 이미 포함된 em_v16의 OOF·burden·severity·hotspot을 다시 만들지 않고 기능 변이 signature와 안정 consequence 흐름을 한 번만 유지합니다. `em_v30`은 em_v16에 em_v5의 전체 입력 유전자 burden total·rate만 추가합니다. `em_v31`은 em_v6 기능이 이미 포함된 em_v16 피처를 한 세트만 유지하고, `em_v32`는 em_v9 signature를 중복 생성하지 않고 em_v16의 OOF 개선형만 유지합니다. `em_v33`과 `em_v34`는 각각 em_v10·em_v11에서 중복되지 않는 전체 입력 burden total·rate만 추가합니다. `em_v35`는 em_v12의 피처를 중복 생성하지 않고 em_v16 OOF·hotspot 개선형을 유지하며, `em_v36`은 em_v13에서 중복되지 않는 전체 입력 burden total·rate만 추가합니다. `em_v37`은 em_v14의 severity·consequence·signature·hotspot을 다시 만들지 않고 em_v16의 inner-fold OOF 개선형을 한 세트만 유지합니다. `em_v38`은 em_v15의 outer 모델 평가 OOF와 em_v16의 inner signature OOF를 서로 다른 단계에 각각 한 번만 적용합니다. `em_v39`는 em_v6의 consequence·burden 중복을 제외하고 v24의 세분화된 dual signature 흐름만 유지하며, `em_v40`은 em_v9 signature를 중복 생성하지 않습니다. `em_v41`은 em_v12 severity·summary·signature를 다시 만들지 않습니다. `em_v42`와 `em_v43`은 em_v14·em_v15에서 v24와 겹치는 기능 hotspot을 제외하고 동의 변이 recurrent hotspot만 추가하며, em_v15의 outer OOF는 v24의 정책과 하나로 통합합니다. `em_v26`과 `em_v27`을 포함한 모든 TCGA 기반 파이프라인은 원본 26개 레이블을 변경하지 않습니다. 사용할 버전은 설정 YAML의 `preprocessing.name`으로 선택합니다.

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
