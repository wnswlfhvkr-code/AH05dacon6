# em_v19: em_v16 E4 + em_v17 결합 실험 결과

## 1. 실험 목적

기존 최고 파이프라인인 `em_v16`의 E4 조건에 `em_v17`의 토큰 기반 mutation count와 multi-hit 정보를 추가해 암종 다중분류 성능과 일반화 변화를 검증했다.

```text
em_v16
  최소 변이 빈도 + consequence severity
  inner-fold OOF class signature
  recurrent exact hotspot

em_v17 추가 정보
  셀 문자열 분리 + 중복 토큰 제거
  유전자별 unique mutation_count + multi-hit

em_v19
  공통 consequence/hotspot은 한 번만 생성
  기존 em_v16 피처와 고상관인 v17 추가 피처 제거
```

모든 파이프라인 버전 독립 구현 규칙에 따라 `em_v19`는 `em_v16`이나 `em_v17`을 상속하지 않고 `PreprocessingPipeline`만 직접 상속한다.

## 2. 중복 제거와 누수 방지

- v16과 v17에 모두 존재하는 consequence와 recurrent hotspot은 한 번만 생성했다.
- v16의 severity·signature·hotspot을 먼저 보존했다.
- v17의 mutation_count와 multi-hit가 동일 유전자의 기존 v16 피처와 `|Pearson r| >= 0.90`이면 추가 피처를 제거했다.
- 유전자 선택, hotspot 선택, class signature 가중치와 상관 제거는 각 outer-fold 학습부에서만 적합했다.
- 학습 표본의 class signature는 다시 inner 5-fold OOF로 생성해 자기 정답의 직접 영향을 차단했다.

## 3. E4 학습 조건

```yaml
model:
  name: xgboost
  n_estimators: 500
  learning_rate: 0.03
  max_depth: 3
  min_child_weight: 5.0
  subsample: 0.75
  colsample_bytree: 0.6
  reg_alpha: 0.5
  reg_lambda: 10.0
  early_stopping_rounds: 30
  eval_metric: mlogloss
  tree_method: hist

preprocessing:
  name: em_v19
  min_mutation_count: 5
  top_genes_per_class: 20
  min_hotspot_count: 5
  max_hotspots: 384
  inner_signature_folds: 5
  max_token_genes: 3000
  correlation_threshold: 0.9
```

- 평가: `StratifiedKFold` 5-fold OOF Macro F1
- 보조 평가: seed 42의 80% 학습/20% 검증
- 과적합 기준: `80% 학습 F1 - 20% 검증 F1 > 0.1`

## 4. 전체 결과와 순위

| 순위 | 파이프라인 | 모델 | OOF F1 | fold 평균 | fold 표준편차 | 학습 F1 | 검증 F1 | 과적합 격차 | 과적합 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `em_v16` | E4 | **0.394757** | **0.393625** | **0.002167** | 0.539745 | **0.412001** | **0.127743** | true |
| 2 | `em_v19` | E4 | 0.393759 | 0.392377 | 0.006007 | **0.543230** | 0.394566 | 0.148664 | true |
| 3 | `em_v17` | E1 | 0.360916 | 0.359305 | 0.005490 | 0.472547 | 0.340009 | 0.132538 | true |

em_v19는 단독 em_v17보다 OOF가 0.032843 높아 v16 signature가 결합 모델의 주된 성능을 유지했다. 그러나 em_v16 E4와 비교하면 OOF가 0.000998 낮고, fold 표준편차는 0.003839 증가했으며, 과적합 격차도 0.020921 증가했다.

## 5. fold별 OOF Macro F1

| 파이프라인 | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `em_v16` E4 | 0.395840 | 0.390792 | 0.394793 | 0.391224 | 0.395477 |
| `em_v17` E1 | 0.350297 | 0.361772 | 0.362346 | 0.365958 | 0.356150 |
| `em_v19` E4 | 0.384112 | 0.396118 | 0.388470 | 0.391729 | **0.401455** |

em_v19는 fold 5에서 가장 높은 0.401455를 기록했지만 fold 1은 0.384112로 하락해 em_v16 E4보다 분할 민감도가 커졌다.

## 6. 과적합 분석

| 비교 항목 | em_v16 E4 | em_v19 E4 | 변화 |
| --- | ---: | ---: | ---: |
| 80% 학습 F1 | 0.539745 | 0.543230 | +0.003485 |
| 20% 검증 F1 | 0.412001 | 0.394566 | -0.017435 |
| 과적합 격차 | 0.127743 | 0.148664 | **+0.020921** |

추가 피처가 학습 점수는 소폭 높였지만 검증 점수를 낮췄다. 따라서 v17의 count/multi-hit 정보가 일부 학습 표본 특이 패턴을 더 잘 설명했으나 일반화 신호로 이어지지 않은 것으로 해석한다.

## 7. 최종 피처 구성

전체 6,201개 학습 표본으로 최종 적합한 결과다.

| 피처 유형 | 생성/선택 | 고상관·상수 제거 후 유지 |
| --- | ---: | ---: |
| v16 consequence severity 유전자 | 4,171 | 4,171 |
| v16 OOF/full signature | 52 | 52 |
| 공통 recurrent hotspot | 384 | 384 |
| v16 consequence·burden 요약 | 9 | 9 |
| v17 mutation_count | 3,000 | 1,461 |
| v17 multi-hit | 3,000 | 2,750 |
| 합계 | 10,616 | **8,827** |

- v17 추가 피처 중 고상관 제거: 1,540개
  - mutation_count 1,539개
  - multi-hit 1개
- 상수 피처 제거: 249개
- 제거 전 검사된 v17→v16 최대 절대 상관: 0.992502
- 희귀 원본 유전자 제거: 213개

mutation_count의 약 절반이 기존 consequence severity와 거의 같은 변이 존재 정보를 표현해 제거됐다. 최종 증가분은 mutation_count 1,461개와 multi-hit 2,750개로, em_v16보다 차원이 크게 늘었다.

## 8. 조기 종료 결과

| 구분 | 사용 트리 수 |
| --- | --- |
| fold 1~5 | 500, 499, 499, 500, 500 |
| 전체 최종 모델 | 500 |

조기 종료가 거의 작동하지 않았고 최종 모델도 fold 중앙값 500개 트리로 학습됐다. em_v19의 성능 차이는 학습 조기 종료보다 추가 피처 구성의 영향으로 보는 것이 타당하다.

## 9. submission 검증

- 파일: `data/processed/test_009_submission.csv`
- 크기: 2,546행 × 2열
- 컬럼: `ID`, `SUBCLASS`
- sample submission ID 순서 일치: true
- 결측 예측: 0개
- 예측 암종 수: 26개
- SHA-256: `398384ca1869fa28eb4f7824a46e3045892682e5e551987fd5fc2c7eb8bc7b43`

em_v16 E4의 `test_005_submission.csv`와 비교하면 2,546건 중 2,058건(80.83%)이 같고 488건(19.17%)이 다르다. 예측 다양성은 존재하지만 정답이 없는 test 차이이므로 앙상블 향상을 보장하지는 않는다.

## 10. 결론

- em_v16 E4와 em_v17을 중복 없이 결합한 em_v19를 구현하고 실행했다.
- em_v19 OOF F1은 0.393759로 기존 em_v16 E4와 매우 가깝지만 0.000998 낮다.
- fold 안정성과 과적합 격차도 em_v16 E4보다 나빠 최종 단독 모델 교체 근거는 없다.
- 최종 우선순위는 em_v16 E4 유지다. em_v19는 test 예측이 19.17% 달라 다양성은 있으므로, OOF probability blending으로 개선이 확인될 때만 앙상블 후보로 사용한다.
- 다음 ablation은 4,211개 count/multi-hit 피처 전체가 아니라 class별 기여가 안정적인 multi-hit 소수만 fold 내부에서 선택해야 한다.
