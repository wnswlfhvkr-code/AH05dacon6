# EM F01~F19 파생 피처 설계

## 공통 원칙

- 각 파이프라인은 EMV45 전체 피처를 공통 베이스로 사용하고 요청한 F번호
  피처 그룹을 추가해, 같은 베이스 모델 조건에서 증분 효과를 비교할 수 있게 한다.
- 추가 피처에는 `F번호__` 접두사를 붙여 EMV45 컬럼명과 충돌하지 않게 한다.
- 결측값, 빈 문자열, `WT`, `<NA>`는 변이 없음으로 동일하게 처리한다.
- 한 셀에 같은 변이 토큰이 반복되면 한 번만 계산한다.
- 빈도 필터와 hotspot 목록은 현재 fold의 train 부분에서만 결정한다.
- `SUBCLASS` 원본 체계를 유지하며 `GBMLGG`, `KIPAN`, `STES`를 분해하지 않는다.
- 외부 유전자 주석이나 임상 데이터는 사용하지 않는다.

## 표본 단위

| 번호 | 파이프라인 | 생성 피처 | 정의 |
|---|---|---|---|
| F01 | `em_F01` | `mutated_gene_count` | 하나 이상의 변이가 있는 유전자 수 |
| F02 | `em_F02` | `mutation_event_count` | 셀 내부 중복 제거 후 전체 변이 토큰 수 |
| F03 | `em_F03` | `mutation_burden_log1p` | `log1p(mutation_event_count)` |
| F04 | `em_F04` | consequence별 이벤트 수 4개 | missense, nonsense, frameshift, splice를 상호 배타적으로 집계 |
| F05 | `em_F05` | `hotspot_mutation_count` | fold-train에서 일정 횟수 이상 반복된 gene-token 보유 수 |
| F06 | `em_F06` | `multi_hit_gene_count` | 서로 다른 변이 토큰이 2개 이상인 유전자 수 |
| F07 | `em_F07` | `no_mutation_sample` | 전체 변이 이벤트가 0이면 1 |
| F08 | `em_F08` | 집중도·다양성 3개 | 최대 유전자 이벤트 비중, 정규화 entropy, 변이 유전자 비율 |

## 유전자 단위

| 번호 | 파이프라인 | 생성 피처 | 일반화 처리 |
|---|---|---|---|
| F09 | `em_F09` | `gene_mutated__{gene}` | fold-train 최소 지지도 이후 이진화 |
| F10 | `em_F10` | `gene_mutation_count__{gene}` | 중복 제거된 유전자별 이벤트 수 |
| F11 | `em_F11` | `gene_multi_hit__{gene}` | multi-hit 지지도를 만족한 유전자만 유지 |
| F12 | `em_F12` | `gene_hotspot__{gene}` | fold-train recurrent hotspot을 하나 이상 포함하면 1 |
| F13 | `em_F13` | `gene_truncating__{gene}` | nonsense 또는 frameshift 변이가 있으면 1 |
| F14 | `em_F14` | `gene_missense__{gene}` | 아미노산 시작·종료 문자가 다른 치환이면 1 |
| F15 | `em_F15` | `gene_splice__{gene}` | splice, IVS, `+/-` 위치 패턴을 포함하면 1 |

## 암종 signature

F16~F19는 fold-train의 유전자 변이 여부로 암종별 stabilized log-odds와
mutation prevalence profile을 학습한다. smoothing, support shrinkage, log-odds
clipping, class별 상위 유전자 제한을 적용한다.

| 번호 | 파이프라인 | 생성 피처 | 정의 |
|---|---|---|---|
| F16 | `em_F16` | `signature_weighted__{class}` | 표본 변이 벡터와 암종별 양의 stabilized log-odds 내적 |
| F17 | `em_F17` | `signature_similarity__{class}` | 표본 변이 벡터와 암종 prevalence profile의 cosine 유사도 |
| F18 | `em_F18` | `signature_top2_margin` | 가장 높은 signature 유사도와 두 번째 유사도의 차이 |
| F19 | `em_F19` | `signature_entropy` | 암종 유사도 softmax 분포의 정규화 entropy |

학습 행의 F16~F19는 inner `StratifiedKFold` OOF 값으로 생성한다. validation과
test에는 outer fold의 train 부분에서 학습한 signature만 `transform`한다.

## 실행 상태

- F01~F19 소스, 레지스트리와 실행용 YAML 19개를 생성했다.
- 실행 명령은 `python -m src.train --config configs/test_006_em_F번호.yaml`이다.
- 이번 작업에서는 학습, 교차검증 평가, train/test 데이터 변환, submission 생성을
  수행하지 않았다.

## F20 전체 결합

`em_F20`은 EMV45를 한 번만 생성한 뒤 F01~F19의 전용 파생 피처를 모두
추가한다. F01~F19 공개 파이프라인 전체를 이어 붙이지 않으므로 각 파이프라인에
포함된 V45 베이스가 반복 생성되지 않는다. F16~F19는 학습 행에서 각각 기존
inner-fold OOF 값을 사용한다.
