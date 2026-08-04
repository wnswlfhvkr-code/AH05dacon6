# v01 — F0 + F1 + F3 Balanced Logistic Regression

## 목적

동일 변이 프로필 누출을 차단한 StratifiedGroupKFold 조건에서 변이 feature 블록의 효과를 단계적으로 비교한다. 리더보드 제출용 최종 후보는 로컬 OOF Macro F1뿐 아니라 seed 안정성, 부담 구간별 성능, Test 예측 분포를 함께 평가한다.

## 데이터 및 검증

- Train 6,201개, Test 2,546개, 암종 26개
- 전체 유전자 4,384개 중 Train 활성 유전자 4,230개
- `StratifiedGroupKFold`: 3 folds × seeds 42, 2026, 777
- 그룹: 정규화한 exact mutation profile SHA-256
- 계층: `SUBCLASS × 변이 부담 구간`; 희소 조합은 클래스 단위로 통합
- 주 지표: OOF Macro F1

## Feature 정의

| 블록 | 정의 | 수 |
| --- | --- | ---: |
| F0 | 유전자별 변이 유무 | 4,230 |
| F1 | event·변이 유전자·multi-hit·consequence 개수/비율·위치 요약·All-WT | 22 |
| F2 | 유전자 × consequence multi-hot | v01에서 제외 |
| F3 | AA_FROM, AA_TO, AA_CHANGE, 위치 구간 multi-hot | 활성 320 |

F1의 count와 위치 값에는 `log1p`를 적용하고, `StandardScaler`는 각 학습 Fold에만 fit한다. F3는 유전자와 결합하지 않아 희소 feature 폭증을 억제한다.

## 모델

- Balanced Logistic Regression
- `solver=saga`, `penalty=l2`, `C=1.0`
- `max_iter=3000`, `tol=1e-4`
- 9개 Fold 모델의 Test 확률 평균으로 제출 예측 생성

## 단계별 결과

| 단계 | Feature | OOF Macro F1 | 표준편차 | 판정 |
| --- | --- | ---: | ---: | --- |
| E3 | F0 | 0.342970 | 0.000526 | 기준선 |
| E4 | F0 + F1 | 0.393076 | 0.007553 | F1 채택 |
| E5 | F0 + F1 + F2 | 0.391355 | 0.008840 | F2 전체 보류 |
| v01 (Kaggle 노트북) | F0 + F1 + F3 | **0.432786** | 0.008196 | 노트북 기준 최고 |

### v01 seed별 OOF

| Seed | Macro F1 |
| ---: | ---: |
| 42 | 0.428485 |
| 2026 | 0.427637 |
| 777 | 0.442237 |

모든 seed에서 E4보다 상승했고, v01 최저 fold `0.411439`도 E4 최고 fold `0.404648`보다 높았다.

## 부담 구간 변화

| 변이 유전자 수 | E4 | v01 | 변화 |
| --- | ---: | ---: | ---: |
| 0–5 | 0.158171 | 0.163131 | +0.004960 |
| 6–9 | 0.231090 | 0.230424 | -0.000666 |
| 10–16 | 0.250679 | 0.271074 | +0.020395 |
| 17–33 | 0.319296 | 0.321683 | +0.002386 |
| 34+ | 0.251713 | 0.321608 | **+0.069894** |

## 리더보드 제출

| 항목 | 결과 |
| --- | ---: |
| Kaggle 노트북 OOF Macro F1 | 0.432786 |
| Public Score | 0.309881672 |
| OOF–Public 차이 | -0.122904 |

Test 예측은 SKCM 20.42%, ACC 6.95%, COAD 11.70%로 Train 비율보다 높았다. Test 정답 분포를 알 수 없으므로 오예측으로 단정할 수 없지만, Train/Test의 변이 부담 차이와 `class_weight=balanced`에 의한 소수 클래스 강화 가능성을 일반화 위험으로 기록한다.

## 결론

- F0, F1 채택
- F2 전체는 보류
- F3는 로컬 기준 채택하되 Public 일반화 추가 검증 필요
- 현재 Kaggle 기준점: `0.309881672`
- 다음 제출은 OOF, seed 일관성, `34+` 성능, Test 예측 쏠림을 모두 확인한 후보에만 사용


## 실행

```bash
python -m src.train_sgkf --config configs/test_002.yaml
```

기존 `src.train_v01`은 `src.train_sgkf`로 통합되었다. `configs/test_002.yaml`은 최신 실험 설정을 가리키므로 이 과거 실험을 재현할 때는 보고서의 v01 전처리·모델 설정으로 변경해야 한다. 산출물은 `data/processed/`에 저장되며 `.gitignore`에 의해 GitHub에는 포함되지 않는다.

## 재실행 결과

동일한 v01 파이프라인을 로컬 환경에서 재실행한 결과,
OOF Macro F1은 `0.429780 ± 0.004209`로 측정되었다.

| Seed | OOF Macro F1 |
|---:|---:|
| 42 | 0.432862 |
| 2026 | 0.431493 |
| 777 | 0.424984 |

기존 Kaggle 노트북 실험 결과인 `0.432786 ± 0.008196`보다
평균 Macro F1이 `0.003006` 낮았다.

특히 seed 777의 결과 차이가 컸으므로, 실행 도구인 VS Code 자체의
차이라기보다는 SGKF 분할, 입력 데이터 및 전처리 결과, 라이브러리 버전,
모델 설정의 차이 여부를 추가로 확인할 필요가 있다.

본 문서와 `data/processed/v01/`에는 재현 가능한 코드로 다시 실행한
로컬 결과를 최종 결과로 기록한다.
