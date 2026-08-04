# pipeComb_v4 — Group-safe F9 + EM24 weighted W52 검증 채택 / LB 미승격

작성일: 2026-08-04

## 핵심 결론

- 공식 파이프라인 이름: `pipeComb_v4`
- 상태: 그룹 안전 로컬 검증 채택, Public LB 기준 v3 미대체
- 구성: 그룹 F9 + EM24 전체 변이 weighted 26개 + 기능 변이 weighted 26개
- 최종 OOF Macro F1: `0.4507880142`
- 평균 학습-검증 gap: `0.4388958499`
- 최대 단일 fold gap: `0.4716459489`
- v4 Public Leaderboard Macro F1: `0.3342990633`
- v3 Public LB 대비: `-0.0047107024`

`pipeComb_v4`는 D104의 match-count 52개를 제거하고, 두 EM24 채널의 weighted
암종 서명 52개만 유지한 W52 파이프라인이다. OOF F1과 평균 gap이 함께
개선돼 로컬 검증에서는 승격했지만, Public LB는 v3보다 낮아 최종 제출
우선순위를 대체하지 못했다. 과적합 자체를 해결한 모델로도 해석하지 않는다.

## 파이프라인 구조

1. `jyp_f9`의 그룹 안전 특징을 생성한다.
2. EM24에서 전체 변이와 기능 변이의 암종별 weighted signature를 학습한다.
3. `signature_all_*_weighted` 26개를 붙인다.
4. `signature_functional_*_weighted` 26개를 붙인다.
5. D104에 있던 두 채널의 `match_count` 52개는 제외한다.

최종 W52 순서는 전체 변이 26개 다음 기능 변이 26개인 channel-major 순서로
고정한다. F9와 EM24의 label class 순서, 출력 열 이름·순서·개수, NaN/Inf와
중복 열을 모두 검사한다.

계단식 GPU 검증 당시 설정의 내부 파이프라인 이름은 D104와 W52를 함께
물질화하는 `pipeComb_v8`이었고, 후보 identity는 `F9+W52`였다. 여기서 승격한
weighted-only W52 구성을 전용 래퍼와 레지스트리 이름으로 고정한 것이
`pipeComb_v4`다. 따라서 frozen 결과의 experiment ID는
`test_003_fusion_stair2_weighted_gpu`로 남아 있다.

## 고정 검증 조건

- 데이터: 학습 데이터 `6,201`행만 사용
- 모델: 단일 XGBoost
- 모델 설정: `n_estimators=100`, `learning_rate=0.1`, `max_depth=6`,
  `tree_method=hist`
- 실제 장치: NVIDIA GeForce RTX 5070 Ti `cuda:0`
- outer validation: seeds `(42, 2026, 777)` × 3-fold
  `StratifiedGroupKFold`
- 그룹: normalized exact mutation profile
- 외부 fold 수: 총 9개
- 내부 OOF: F9 F7과 EM24 signature 모두 5-fold group-safe
- 승격 조건: 기준안보다 평균 OOF F1이 높고 평균 gap이 낮을 것

외부 fold와 내부 F9·EM24 OOF에서 동일 mutation-profile group의
학습·검증 중복은 모두 `0`이었다.

## V24 특징 축소 계단식 검증

| 차수 | 기준안 → 후보       | V24 특징 |                             OOF Macro F1 |                                 평균 gap | 판정      |
| ---- | -------------------- | -------: | ---------------------------------------: | ---------------------------------------: | --------- |
| 1차  | 그룹 안전 F9 → D104 |      104 | `0.436011 → 0.448506` (`+0.012495`) | `0.444819 → 0.441571` (`-0.003248`) | D104 승격 |
| 2차  | D104 → W52          |       52 | `0.448506 → 0.450788` (`+0.002282`) | `0.441571 → 0.438896` (`-0.002675`) | W52 승격  |
| 3차  | W52 → FW26          |       26 | `0.450788 → 0.448259` (`-0.002529`) | `0.438896 → 0.439903` (`+0.001007`) | FW26 탈락 |

해석은 명확하다.

- 단순 match-count 52개는 제거하는 편이 점수와 gap 모두에 유리했다.
- 전체 변이 weighted 26개까지 제거하면 정보 손실이 발생했다.
- 따라서 전체 변이와 기능 변이 weighted 채널을 모두 유지한 W52가 최종안이다.

## D104 대비 seed별 W52 결과

| Seed | D104 OOF F1 | W52 OOF F1 |     ΔOOF | D104 gap |  W52 gap |     Δgap | W52 승리 fold |
| ---: | ----------: | ---------: | --------: | -------: | -------: | --------: | ------------: |
|   42 |    0.450503 |   0.453519 | +0.003016 | 0.441241 | 0.437240 | -0.004002 |           2/3 |
| 2026 |    0.446430 |   0.448903 | +0.002473 | 0.440834 | 0.439074 | -0.001760 |           1/3 |
|  777 |    0.448585 |   0.449942 | +0.001357 | 0.442637 | 0.440374 | -0.002263 |           1/3 |

세 seed 모두 OOF F1이 상승하고 평균 gap이 감소했다. 다만 개별 fold 기준으로는
W52가 4/9 fold에서만 F1 우세였고 gap 감소도 3/9 fold였다. 평균 개선은
재현됐지만 모든 분할에서 일관된 우위는 아니다.

## 최종 W52 지표

| 지표                     |                     값 |
| ------------------------ | ---------------------: |
| 전체 OOF Macro F1        |     0.4507880141783693 |
| fold train Macro F1 평균 |     0.8881052665628162 |
| fold valid Macro F1 평균 |     0.4492094166734317 |
| 평균 train-valid gap     |    0.43889584988938457 |
| 최대 fold gap            |     0.4716459489417417 |
| OOF 예측 행 수           |                 18,603 |
| D104 대비 OOF 변화       | +0.0022818753311159936 |
| D104 대비 gap 변화       | -0.0026748359995847903 |

## v3 기록과의 관계

`test_003_pipeComb_v3.md`의 OOF `0.446768`과 Public Leaderboard
`0.3390097657`은 seed 42 기반의 이전 v3 실행 기록이다. 위 계단식 비교의
D104 `0.448506`은 3 seeds × 3 grouped folds로 다시 평가한 canonical v3
기준선이므로 두 숫자를 같은 검증 결과로 섞지 않는다.

v4 W52 제출의 Public Leaderboard Macro F1은 `0.3342990633`이다. v3의
`0.3390097657`보다 `0.0047107024` 낮다. 따라서 그룹 안전 OOF에서 확인한
W52의 우위가 Public LB 개선으로 이어지지는 않았다.

## 무결성 감사

- 외부 OOF 예측: `6,201 × 3 seeds = 18,603`행 완전 생성
- 외부 group overlap: 9/9 fold 모두 `0`
- F9 F7·EM24 내부 OOF group overlap: 모든 감사 항목 `0`
- D104 기준 예측 lineage: 이전 단계 18,603행과 완전 일치
- 모델·전처리·스키마·학습 데이터·split·소스 hash 계약: 통과
- 결과 JSON SHA-256:
  `05db66975cfdae3c959153bb7a0b82b7c146128089a98cc4d5d889953a568ed8`

## 냉정한 판단

W52는 현재 그룹 안전 로컬 검증 조건에서 가장 좋은 전처리다. 그러나 평균 gap
`0.438896`은 여전히 매우 크고 9/9 fold가 과적합 임계값 `0.1`을 넘었다.
Public LB도 v3보다 `-0.0047107024` 낮다. 따라서 v4의 의미는 “과적합 해결”이나
“최종 제출 승격”이 아니라 다음으로 한정한다.

> D104에서 정보량이 낮은 match-count 52개를 제거해 OOF F1을 높이고 평균
> gap을 소폭 줄인 그룹 안전 전처리. 최종 제출 우선순위는 v3로 유지한다.

같은 9개 개발 fold로 D104·W52·FW26을 순차 선택했으므로 독립 외부 홀드아웃
재현으로 표현하지 않는다.

## 재현 및 산출물

- 최종 실행 설정: `configs/test_003.yaml`
- v3 보존 설정: `configs/test_003_pipecomb_v3.yaml`
- v4 파이프라인: `src/pipelines/jyp_preprocessing/pipeline_pipe_comb_v4.py`
- v4 테스트: `tests/test_pipe_comb_v4.py`
- Public LB 제출: `data/processed/test_003_pipeComb_v4_1_submission.csv`
- Public LB 실행 기록: `experiments/jyp_preprocessing/test_003_pipeComb_v4_1.md`
- frozen 결과:
  `data/processed/test_003_fusion_stair2_weighted_gpu/test_003_fusion_stair2_weighted_gpu/result.json`
- fold 지표: 같은 디렉터리의 `fold_metrics.csv`
- seed 지표: 같은 디렉터리의 `seed_metrics.csv`
- class 지표: 같은 디렉터리의 `class_metrics.csv`
- OOF 예측: 같은 디렉터리의 `oof_predictions.csv`

기본 학습 명령은 다음과 같다.

```bash
python -m src.train --config configs/test_003.yaml
```
