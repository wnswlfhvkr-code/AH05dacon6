# TEST_004 제출 6건 재현 가이드

## 공통 준비

`train.csv`, `test.csv`, `sample_submission.csv`를 `data/raw/`에 둡니다.

```bash
python -m pip install -r requirements.txt
```

빠른 구조 검증은 `--quick`, 기록용 5-Fold 재학습은 옵션 없이 실행합니다.

## 실행 매핑

| 모델명 | 설정 파일 | baseline 사양 | pipeline | 역사적 OOF | Public |
| --- | --- | --- | --- | ---: | ---: |
| TEST_004_1 | `configs/test_004_1.yaml` | `src/models/baseline/TEST_004_1.py` | `src/pipelines/pipeline_jsj_v3.py` | 0.505595 | 0.371498 |
| TEST_004_2 | `configs/test_004_2.yaml` | `src/models/baseline/TEST_004_2.py` | `src/pipelines/pipeline_jsj_v4.py` | 0.507241 | 0.374100 |
| TEST_004_3 | `configs/test_004_3.yaml` | `src/models/baseline/TEST_004_3.py` | `src/pipelines/pipeline_jsj_v5.py` | 0.508134 | 0.375613 |
| TEST_004_4 | `configs/test_004_4.yaml` | `src/models/baseline/TEST_004_4.py` | `src/pipelines/pipeline_jsj_v6.py` | 0.508400 | 0.376826 |
| TEST_004_5 | `configs/test_004_5.yaml` | `src/models/baseline/TEST_004_5.py` | `src/pipelines/pipeline_jsj_v7.py` | 0.514965 | 0.386336 |
| TEST_004_6 | `configs/test_004_6.yaml` | `src/models/baseline/TEST_004_6.py` | `src/pipelines/pipeline_jsj_v8.py` | 0.518823 | 0.389342 |

```bash
python -m src.reproduce_test_004 --config configs/test_004_1.yaml
python -m src.reproduce_test_004 --config configs/test_004_2.yaml
python -m src.reproduce_test_004 --config configs/test_004_3.yaml
python -m src.reproduce_test_004 --config configs/test_004_4.yaml
python -m src.reproduce_test_004 --config configs/test_004_5.yaml
python -m src.reproduce_test_004 --config configs/test_004_6.yaml
```

생성 결과는 각각 다음 위치에 저장됩니다.

- `data/processed/test_004_n_submission.csv`
- `data/processed/test_004_n_metrics.json`

## 단계별 차이

- `TEST_004_1`: Word+Char TF-IDF LinearSVC 95%와 구조 피처 LightGBM 5%를 결합합니다.
- `TEST_004_2`: KIRC–KIPAN, LGG–GBMLGG 확률을 전문 이진 LinearSVC로 전면 보정합니다.
- `TEST_004_3`: 전문 모델의 결과가 반드시 해당 충돌 쌍 내부에서만 바뀌도록 제한합니다.
- `TEST_004_4`: LGG–GBMLGG는 100%, KIRC–KIPAN은 10%만 보정합니다.
- `TEST_004_5`: LGG–GBMLGG 전문 모델의 C를 0.10으로 낮추고 GBMLGG 방향 결정값에 0.10 오프셋을 적용합니다.
- `TEST_004_6`: `em_v14`의 consequence·암종 signature·hotspot 피처를 LightGBM으로 학습하고 KIRC–KIPAN 전용 전문가로 30% 결합합니다.

## 재현 범위

- 외부 데이터와 사전학습 모델을 사용하지 않습니다.
- TF-IDF와 모델은 각 Fold의 학습 부분에서만 fit합니다.
- `em_v14`의 signature와 hotspot도 각 Fold의 학습 부분에서만 선택합니다.
- test 데이터는 transform과 predict에만 사용합니다.
- Public 점수는 Dacon 정답이 있어야 계산되므로 config의 Public 값은 제출 당시 기록입니다.
- 패키지 버전이나 LightGBM 실행 환경이 달라지면 재학습 예측이 소수 달라질 수 있습니다.
