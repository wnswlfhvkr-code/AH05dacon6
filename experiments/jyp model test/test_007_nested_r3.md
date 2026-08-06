# TEST_007 완전 Nested OOF와 r3 결과

## 현재 판정

| 항목 | 결과 |
|---|---|
| 고정 전처리 | `pipeComb_v3` |
| 평가 구조 | 3 seeds × 5 outer folds, outer-train 내부 4-fold 선택 |
| 모델 선택·튜닝에 Test 사용 | 없음 |
| r3 Nested OOF Macro F1 | `0.501254` |
| r3 Public LB | `0.374393` |
| 팀 기준 `JSJ_VoteSelect_v1.csv` Public LB | `0.466060` |
| 최종 판정 | **r3 실전 제출 후보 탈락** |

r3는 Train-only Nested OOF에서는 기존 주력보다 소폭 개선됐지만 실제
리더보드로 전이되지 않았다. 현재 점수만으로 r3의 가중치나 후처리를 다시
튜닝하지 않으며, 제출 재생 무결성과 검증 분포 대표성을 먼저 감사한다.

## 목표와 평가 원칙

목표는 Test를 모델 선택·튜닝에 사용하지 않고 동일한 `pipeComb_v3`, fold,
seed 조건에서 평균 Macro F1을 최대화하는 것이었다. 평균 점수 외에도 seed,
암종, mutation burden, novelty 구간에서 반복적인 성능 붕괴가 없어야 한다.
gap은 목표함수가 아니라 1-SE 범위의 동률 후보를 고를 때만 후순위 기준으로
사용했다.

## 진행 타임라인

| 단계 | 목적 | 진행 내용 | 결과 |
|---:|---|---|---|
| 1 | 모델 후보 확장 | TabM, ModernNCA, RealMLP, RealTabR, xRFM과 기존 트리·TFM 후보를 동일 콤3에서 실행 | global·collision expert 후보군 구성 |
| 2 | 사전 OOF 선별 | Macro F1, fold 변동성, 오류 다양성 비교 | Nested 탐색 모델 풀 확정 |
| 3 | 완전 Nested OOF | 각 outer fold의 조합·가중치·temperature를 outer-train의 inner OOF에서만 선택 | 주력·강건형·전문형 역할 생성 |
| 4 | 최초 주력 확정 | 3 seeds × 5 folds의 선택 확률을 결합 | 평균 Macro F1 `0.500910` |
| 5 | 동결 추론 | 선택을 먼저 동결하고 Test 캐시로 제출 확률 생성 | 주력·강건형·전문형 제출 생성 |
| 6 | RealTabR 감사 | 구형 번들의 predict 시 내부 Train context 복원 문제 확인 | 엄격한 no-post-Test-fit 계약 미통과 발견 |
| 7 | RealTabR 재생성 | classifier와 retrieval context를 직렬화한 strict bundle을 3 seeds에서 재생성 | 기존 OOF와 byte-identical, Test 접근 없음 |
| 8 | r3 보정 | 모델과 앙상블을 고정하고 inner OOF에서 방향성 암종 충돌만 보정 | 주력 OOF `0.501254` |
| 9 | 안정성 판정 | seed·암종·burden·novelty 반복 붕괴 검사 | 주력 통과, 두 challenger 미통과 |
| 10 | 리더보드 확인 | 동결된 r3 제출을 1회 평가 | `0.374393`, 실전 후보 탈락 |

## r3 앙상블 구조

r3는 하나의 고정 앙상블이 아니다. 각 seed·outer fold의 inner OOF에서 선택된
3~5개 모델의 확률 앙상블 15개를 최종 평균한다. r3 단계에서는 이 모델 조합과
가중치·temperature를 변경하지 않았다.

| 모델 | 15개 fold 중 선택 횟수 | 역할 |
|---|---:|---|
| CatBoost | 13 | 주력 트리 축 |
| TabICL global | 12 | 26-class 전역 TFM 축 |
| RealMLP | 10 | 신경망 오류 다양성 |
| TabFM expert | 10 | 암종 충돌 전문 보완 |
| ExtraTrees | 4 | 무작위 트리 다양성 |
| TabM | 3 | MLP 내부 앙상블 보완 |
| TabPFN expert | 2 | 충돌군 보완 |
| TabICL expert | 2 | 충돌군 보완 |
| Balanced Random Forest | 1 | 클래스 불균형 보완 |
| RealTabR expert | 1 | retrieval 기반 충돌 보완 |
| ModernNCA expert | 1 | 거리 기반 충돌 보완 |
| LightGBM | 1 | 트리 다양성 |
| GPU Logistic Regression | 1 | 선형 오류 다양성 |
| xRFM | 1 | 커널·metric 기반 다양성 |

핵심 축은 `CatBoost + TabICL global + RealMLP + TabFM expert`이며 나머지는
특정 seed·fold의 inner OOF에서 선택됐을 때만 포함됐다.

## r3 pair-bias 보정

r3는 기존 앙상블에서 두 암종이 확률 1·2위로 근접한 행만 보정한다.
`source ← target` 규칙에서 target이 1위, source가 2위이고 두 확률의 차이가
`margin` 이하이면 `alpha × target 확률`을 source로 이동한다. 조건을 만족하지
않는 행은 변경하지 않는다.

| Seed | Fold | 보정 방향 | Margin | Alpha |
|---:|---:|---|---:|---:|
| 42 | 0 | LUAD ← LUSC | 0.150 | 0.200 |
| 42 | 1 | KIPAN ← TGCT | 0.050 | 0.075 |
| 42 | 2 | LIHC ← LUSC | 0.150 | 0.300 |
| 42 | 3 | LIHC ← UCEC | 0.100 | 0.300 |
| 42 | 4 | CESC ← BLCA | 0.150 | 0.150 |
| 2026 | 0 | PAAD ← HNSC | 0.050 | 0.150 |
| 2026 | 1 | SARC ← PRAD | 0.075 | 0.200 |
| 2026 | 2 | PRAD ← PCPG | 0.025 | 0.150 |
| 2026 | 3 | SARC ← BRCA | 0.050 | 0.100 |
| 2026 | 4 | DLBC ← KIPAN | 0.050 | 0.150 |
| 777 | 0 | SARC ← BRCA | 0.100 | 0.300 |
| 777 | 1 | SARC ← PRAD | 0.200 | 0.300 |
| 777 | 2 | CESC ← BLCA | 0.300 | 0.300 |
| 777 | 3 | PAAD ← STES | 0.150 | 0.300 |
| 777 | 4 | PAAD ← STES | 0.100 | 0.200 |

암종쌍·margin·alpha는 각각의 outer-train 내부 OOF에서만 선택했고, 숨겨둔
outer validation은 선택이 끝난 다음 평가에만 사용했다.

## Nested OOF 결과

| 역할 | Seed 42 | Seed 2026 | Seed 777 | 평균 | 표준편차 | 판정 |
|---|---:|---:|---:|---:|---:|---|
| r3 주력 | 0.499776 | 0.500819 | 0.503167 | **0.501254** | 0.001737 | Train-only 제약 통과 |
| r3 강건형 | 0.499776 | 0.501273 | 0.502638 | 0.501229 | 0.001432 | `burden 1` 반복 하락 |
| r3 전문형 | 0.499575 | 0.500359 | 0.503453 | 0.501129 | 0.002050 | 충돌 보완 방향 일관성 부족 |

| 비교 | Macro F1 | 변화 |
|---|---:|---:|
| 기존 Nested 주력 | 0.500910 | 기준 |
| r3 주력 | **0.501254** | **+0.000343** |

수치 향상은 작으며 새로운 모델 우위가 아니라 기존 앙상블의 제한적인 오류
교정으로 해석한다. 강건형과 전문형은 평균 점수가 높더라도 반복 붕괴 제약을
통과하지 못해 채택하지 않았다.

## gap 해석

r3에는 일반적인 학습 행 재예측 점수가 없다. Nested 평가에서 train 측 대용치는
inner OOF이며, validation은 outer OOF다.

| 지표 | Macro F1 |
|---|---:|
| 주력 inner OOF 평균 | 0.490631 |
| 주력 outer-fold Macro F1 평균 | 0.500121 |
| inner − outer 평균 gap | **-0.009490** |
| fold별 절대 gap 평균 | 0.015813 |
| seed 단위 결합 OOF 평균 | 0.501254 |

내부에서는 전형적인 `train > validation` 형태가 아니었다. 그러나 실제 전이
격차는 `Nested OOF 0.501254 → Public LB 0.374393`, 즉 `-0.126861`이었다.
따라서 일반적인 train–validation gap보다 제출 재생과 validation 대표성 문제가
더 중요하다.

## 제출 감사와 리더보드 결과

| 항목 | 결과 |
|---|---:|
| 제출 행 수 | 2,546 |
| 고유 ID | 2,546 |
| 출력 클래스 수 | 26 |
| 기존 동결 primary 대비 r3 최종 라벨 변경 | **3행** |
| strict RealTabR 대체 전후 argmax 변경 | **0행** |
| r3 CSV SHA-256 | `d5134a007425ffe65ee9872c01135a651d647eb8f00d0475d5ea0bc5092e4e0f` |
| r3 Public LB | **0.3743927897** |
| `JSJ_VoteSelect_v1.csv` Public LB | **0.4660597084** |
| LB 차이 | **-0.0916669187** |

r3 후처리로 최종 라벨이 3행만 바뀌었고 strict RealTabR 교체도 argmax를 바꾸지
않았다. 따라서 낮은 LB를 pair-bias나 RealTabR 직렬화 수정만의 문제로 설명할
수 없다. 기존 동결 primary와 거의 같은 예측을 만든 복잡한 fold별 확률
앙상블의 전이 실패, 또는 그 이전 단계의 재생·검증 대표성 문제를 우선
조사해야 한다.

리더보드 수치는 사용자 제공 기록이다. 이를 근거로 새 모델·가중치·threshold를
튜닝하지 않고 사후 진단과 후보 탈락 판정에만 사용한다.

## 최종 인사이트와 다음 방향

1. r3 주력은 Train-only Nested OOF 절차는 통과했지만 실전 성능 기준으로 탈락한다.
2. r3 pair-bias는 최종 라벨 3개만 바꿨으므로 LB 붕괴의 주원인이 아니다.
3. fold마다 모델·가중치가 크게 달라지는 15개 확률 앙상블은 inner OOF 선택
   잡음을 증폭했을 가능성이 있다.
4. `JSJ_VoteSelect_v1`의 단순 투표 구조를 동일 콤3·fold·seed의 Nested OOF로
   역검증하는 것을 다음 기준 실험으로 삼는다.
5. 리더보드 점수는 진단에만 사용하며 JSJ 구조의 세부 선택도 Train-only
   inner OOF에서 다시 결정한다.

## 근거 산출물

대용량 OOF와 제출 산출물은 Git에 복사하지 않고 workspace 외부 `.omx`에
보존한다.

| 산출물 | SHA-256 |
|---|---|
| `refined_cv_metrics.json` | `84aaa08d6bbb0ada135185c17708ed7e2b8deeb01ab16c6f95aebd33efb46a24` |
| `refined_selection_records.json` | `8c644b3b2993a2af83701e7450f3001ef3809d17ba0c68a26f97019a3ad2374b` |
| `refined_manifest.json` | `7b5087dc2faaeb2e6d18838781f95dea7b0d41db1c1984c4616d47645b29a2d5` |
| r3 제출 CSV | `d5134a007425ffe65ee9872c01135a651d647eb8f00d0475d5ea0bc5092e4e0f` |

기계 판독용 요약은 `experiments/test_007_nested_r3_metrics.csv`에 기록한다.
