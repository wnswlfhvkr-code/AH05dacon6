# pipeComb_v3 — JYP F9 + EM24 내부 피처 결합

## 결론

- 공식 파이프라인 이름: `pipeComb_v3`
- 구성: `F9 + EM24 dual signature 104개`
- 단일 XGBoost OOF Macro F1: `0.426805 → 0.446768` (`+0.019964`)
- Fold 승리: `5/5`
- paired bootstrap 10,000회 95% CI: `[+0.008490, +0.031469]`
- EM24 summary 16개는 dual signature에 추가하지 않음
- Public Leaderboard Macro F1: `0.3390097657`
- 당시 로컬 검증은 seed 42 한 번이므로 다중 seed 안정성은 별도 확인 대상

## 목적

- 확률 앙상블이 아닌 하나의 XGBoost 내부에서 F9와 EM24 정보를 결합
- 모델 차이 없이 추가 피처 자체의 효과만 확인
- F9와 중복이 큰 EM24 블록은 제외하고 저차원 후보만 비교

## 공통 조건

- 모델: XGBoost 1개
- 파라미터: `n_estimators=100`, `learning_rate=0.1`, `max_depth=6`, `tree_method=hist`
- 검증: seed 42, 동일 `StratifiedKFold` 5-Fold
- Train: F9·EM24 모두 `fit_transform()` 사용
- Validation/Test: 해당 Outer Fold Train으로 학습된 전처리기의 `transform()`만 사용
- EM24 target-aware signature의 Train 값은 내부 OOF로 생성
- Fold마다 F9와 EM24를 한 번씩만 학습하고 네 후보가 결과를 공유
- Test는 점수 계산과 후보 선택에 사용하지 않음

## 후보 구성

| 후보 | F9 뒤에 추가한 EM24 블록 | 추가 피처 수 |
|---|---|---:|
| `f9` | 없음 | 0 |
| `f9_em24_summary` | 변이 부담·유형 요약 | 16 |
| `f9_em24_dual_signature` | 전체 변이 signature 52 + 기능 변이 signature 52 | 104 |
| `f9_em24_dual_signature_summary` | summary + dual signature | 120 |

dual signature는 26개 암종 각각에 대해 다음 정보를 만든다.

- 전체 변이 기준: class-weighted score, match count
- 기능 변이 기준: class-weighted score, match count

EM24 gene severity와 hotspot은 첫 비교에서 제외했다.

- EM24 선택 유전자의 변이 유무는 F9 F0와 현재 데이터에서 전부 일치
- EM24 hotspot 369개는 F9 F4 hotspot에 369/369 포함
- 유전자별 consequence 정보도 F9 F2와 중복도가 높음

## 5-Fold 결과

| 후보 | OOF Macro F1 | F9 대비 | Fold 승리 | 피처 수 범위 |
|---|---:|---:|---:|---:|
| F9 | 0.426805 | 기준 | - | 46,900–48,829 |
| F9 + summary | 0.430377 | +0.003573 | 5/5 | 46,916–48,845 |
| **F9 + dual signature** | **0.446768** | **+0.019964** | **5/5** | **47,004–48,933** |
| F9 + dual signature + summary | 0.445304 | +0.018499 | 5/5 | 47,020–48,949 |

Fold별 dual signature 변화는 `+0.013773`, `+0.030987`, `+0.024779`, `+0.013621`, `+0.020193`으로 모두 양수였다.

## Train-Validation 과적합 격차

seed 42의 동일 5-Fold에서 계산한 값이다. Validation은 Fold Macro F1 평균이며, 문서 위 OOF Macro F1과는 집계 방식이 달라 소폭 차이가 난다.

| 후보 | Train Macro F1 평균 | Validation Macro F1 평균 | Gap 평균 | Gap 범위 | Gap > 0.1 |
|---|---:|---:|---:|---:|---:|
| F9 | 0.893585 | 0.423865 | 0.469720 | 0.462300–0.492727 | 5/5 |
| **F9 + EM24 dual signature** | **0.907617** | **0.444536** | **0.463081** | **0.453532–0.483890** | **5/5** |

- F9 대비 Gap 변화: `-0.006639`
- 내부 결합으로 격차가 조금 줄었지만 모든 Fold가 임계값 `0.1`을 초과하므로 과적합 판정은 유지한다.

## 통계 확인

- bootstrap seed: `20260803`
- 원본 `SUBCLASS`별 표본 수를 유지한 채 클래스 안에서 환자를 복원추출
- 각 재표본에서 두 후보의 Macro F1을 다시 계산한 paired bootstrap 10,000회

| 비교 | 평균 변화 | paired bootstrap 95% CI | 변화가 양수인 비율 |
|---|---:|---:|---:|
| dual signature - F9 | +0.019959 | `[+0.008490, +0.031469]` | 99.98% |
| summary - F9 | +0.003528 | `[-0.003697, +0.011188]` | 82.33% |
| summary 포함 조합 - dual signature | -0.001500 | `[-0.007873, +0.004727]` | 32.26% |

- dual signature 효과는 seed 42의 환자 단위 층화 재표본화에서 0보다 안정적으로 높았다.
- summary 단독 효과는 CI가 0을 포함한다.
- summary를 dual signature에 추가할 근거가 없고 실제 점수도 `-0.001465` 낮다.

## 암종별 변화

| 암종 | 표본 수 | F9 | F9+dual signature | 변화 |
|---|---:|---:|---:|---:|
| ACC | 72 | 0.834646 | 0.834646 | +0.000000 |
| BLCA | 104 | 0.445714 | 0.444444 | -0.001270 |
| BRCA | 786 | 0.519117 | 0.518310 | -0.000807 |
| CESC | 155 | 0.389706 | 0.358974 | -0.030732 |
| COAD | 223 | 0.750605 | 0.765550 | +0.014945 |
| DLBC | 38 | 0.139535 | 0.423077 | +0.283542 |
| GBMLGG | 461 | 0.352941 | 0.391775 | +0.038834 |
| HNSC | 223 | 0.318302 | 0.311558 | -0.006745 |
| KIPAN | 515 | 0.248773 | 0.291910 | +0.043137 |
| KIRC | 334 | 0.072100 | 0.133748 | +0.061648 |
| LAML | 158 | 0.528428 | 0.581940 | +0.053512 |
| LGG | 229 | 0.253275 | 0.289474 | +0.036199 |
| LIHC | 158 | 0.352459 | 0.362140 | +0.009681 |
| LUAD | 184 | 0.402516 | 0.403846 | +0.001330 |
| LUSC | 178 | 0.539945 | 0.543353 | +0.003408 |
| OV | 253 | 0.401421 | 0.395062 | -0.006359 |
| PAAD | 120 | 0.319588 | 0.329897 | +0.010309 |
| PCPG | 147 | 0.279570 | 0.292517 | +0.012947 |
| PRAD | 266 | 0.324409 | 0.320493 | -0.003916 |
| SARC | 198 | 0.120301 | 0.147368 | +0.027068 |
| SKCM | 276 | 0.872381 | 0.867562 | -0.004819 |
| STES | 379 | 0.531863 | 0.530120 | -0.001742 |
| TGCT | 124 | 0.492891 | 0.479638 | -0.013253 |
| THCA | 324 | 0.647934 | 0.660033 | +0.012099 |
| THYM | 98 | 0.331797 | 0.306383 | -0.025414 |
| UCEC | 198 | 0.626703 | 0.632153 | +0.005450 |

- 개선/하락/동일 클래스: `15/10/1`
- 큰 개선: DLBC `+0.283542`, KIRC `+0.061648`, LAML `+0.053512`, KIPAN `+0.043137`
- 확인 필요 하락: CESC `-0.030732`, THYM `-0.025414`, TGCT `-0.013253`
- DLBC는 표본이 38개뿐이므로 큰 개선 폭이 다른 seed에서도 유지되는지 반드시 확인

## 기존 혼동쌍 변화

| 실제 → 예측 | F9 오분류 | F9+dual signature 오분류 | 감소 |
|---|---:|---:|---:|
| KIRC → KIPAN | 252 | 230 | 22 |
| KIPAN → KIRC | 222 | 203 | 19 |
| LGG → GBMLGG | 151 | 144 | 7 |
| GBMLGG → LGG | 161 | 152 | 9 |

전체 26-class signature가 F7에서 집중했던 신장·뇌종양 혼동도 함께 줄였다.

## 확률 앙상블과 비교

동일 seed 42·동일 Fold 기준이다.

| 방식 | 모델 수 | OOF Macro F1 | F9 대비 |
|---|---:|---:|---:|
| F9 | 1 | 0.426805 | 기준 |
| F9 + EM24 dual signature 내부 결합 | 1 | 0.446768 | +0.019964 |
| F9 0.60 + EM24 0.40 확률 결합 | 2 | 0.448592 | +0.021787 |

- 내부 결합은 확률 결합 개선 폭의 약 `91.6%`를 단일 모델에서 재현했다.
- 내부 결합은 확률 결합보다 `-0.001824` 낮다.
- 내부 결합 - 확률 결합 bootstrap 95% CI는 `[-0.011102, +0.007824]`로 차이를 확정할 수 없다.
- 따라서 EM24의 104개 signature가 앙상블 이득의 대부분을 설명하지만, EM24 모델 자체의 표현·규제 차이도 일부 남아 있다.

OOF 예측 변경은 F9 대비 1,338건(`21.58%`), Test 최종 예측 변경은 349건(`13.71%`)이다. 내부 결합과 확률 결합도 Test에서 374건(`14.69%`) 달라 두 방식은 완전히 같은 후보가 아니다.

## 채택 판단과 남은 검증

- 프로젝트 전처리 이름으로 승격: `pipeComb_v3`
- 제외: summary 단독 및 summary 추가 조합
- 후속 안정성 확인 항목:
  1. 승격 후보와 F9만 seed `101`, `2027`, `7301`의 5-Fold로 paired 재검증
  2. seed별·15개 Fold별 승리 수와 클래스 하락 안정성 확인
  3. 104개 연속 signature의 Train/Test PSI 및 범위 이탈 확인
  4. 확률 결합과 내부 결합의 3-seed 직접 비교
  5. seed별 희소 암종 F1 하락 여부 확인

## 실행 및 산출물

독립 파이프라인 구현과 공용 실행 설정은 아래 파일에 연결했다.

- 파이프라인: `src/pipelines/jyp_preprocessing/pipeline_pipe_comb_v3.py`
- 공용 실행 설정: `configs/test_003.yaml`
- 레지스트리 이름: `pipeComb_v3`

`configs/test_003_internal_fusion.yaml`은 후속 V16 검증에 재사용되었다. 당시 EM24의 정확한 설정은 아래 결과 JSON에 함께 저장되어 있다.

- 결과: `data/processed/test_003_jypF9_EM24_internal_seed42.json`
- Fold 점수: `data/processed/test_003_jypF9_EM24_internal_seed42_scores.csv`
- OOF/Test 확률: `data/processed/test_003_jypF9_EM24_internal_seed42_probabilities.npz`
- 리더보드 제출 원본: `data/processed/test_003_jypF9_EM24_internal_seed42_f9_em24_dual_signature_submission.csv`
- 공식 이름 복사본: `data/processed/pipeComb_v3_submission.csv`

리더보드 제출 원본은 Outer 5-Fold Test 확률 평균으로 만들었다. 공용
`src.train`에서 `configs/test_003.yaml`을 실행하면 같은 전처리 조합을
full-fit 모델로 다시 학습하므로 제출 예측은 기존 제출 원본과 달라질 수 있다.
