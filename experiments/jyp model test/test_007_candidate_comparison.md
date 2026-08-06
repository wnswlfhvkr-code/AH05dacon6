# TEST_007 유지 후보 비교

## 결론

현재 유지하는 신규 후보는 TabM, ModernNCA, RealMLP, RealTabR, xRFM이다.
모든 후보는 동일한 `pipeComb_v3`와 seed 42 조건에서 비교했다.

1. 신규 global 후보 중에는 TabM이 OOF `0.457402`, gap `0.086284`로 가장 균형이 좋다.
2. ModernNCA collision expert는 global 대비 `+0.010379` 개선됐지만 절대 점수가 낮다.
3. RealTabR collision expert는 XGBoost 기준 대비 `+0.004394` 개선됐으나 CPU 비용이 크다.
4. RealMLP는 TabM보다 점수·안정성·gap이 모두 약해 오류 다양성이 확인될 때만 사용한다.
5. xRFM은 OOF `0.292941`로 현재 설정에서 추가 실험 우선순위가 낮다.

## 전체 결과

| 번호 | 후보와 구조 | 장치 | OOF Macro F1 | Holdout F1 | Train F1 | Gap | 기준 대비 변화 | 판정 |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | TabM global | GPU | **0.457402** | 0.451831 | 0.538115 | **0.086284** | - | 유지 |
| 2 | ModernNCA global | GPU | 0.372220 | 0.371888 | 0.635221 | 0.263333 | 기준 | 중단 |
| 2 | ModernNCA collision expert | GPU | 0.382598 | 0.377614 | 0.631361 | 0.253747 | **+0.010379** | 보류 |
| 3 | RealMLP global | GPU | 0.434804 | 0.444948 | 0.550559 | 0.105611 | - | 보류 |
| 4 | RealTabR collision expert | GPU base + CPU expert | 0.453173 | 0.466211 | 0.833972 | 0.367762 | **+0.004394** | 보류 |
| 5 | xRFM global | CPU | 0.292941 | 0.282050 | 0.729065 | 0.447015 | - | 중단 |

비교 기준은 동일 콤3 XGBoost OOF `0.448779`, LightGBM OOF `0.474520`이다.

## 후보별 인사이트

### TabM

- 내부 앙상블이 일반 MLP보다 과적합을 잘 억제했다.
- 단독 채택보다 LightGBM·CatBoost와의 OOF 오류 상관을 확인하는 것이 우선이다.

### ModernNCA

- 충돌 expert 구조의 방향은 유효했지만 현재 SVD 표현에서 절대 성능이 부족했다.
- 재시험한다면 모델 내부에서 암종 서명·빈도 블록을 별도로 보존해야 한다.

### RealMLP

- gap은 비교적 작지만 OOF와 fold 안정성이 TabM보다 낮다.
- 앙상블 오류 다양성이 충분할 때만 후보로 유지한다.

### RealTabR

- collision expert 효과는 양수지만 Windows 환경에서 retrieval은 CPU로 실행된다.
- 비용 대비 개선 폭이 작아 단독 제출보다 제한적인 전문 모델로 본다.

### xRFM

- 현재 SVD와 smoke budget에서 점수와 일반화가 모두 약했다.
- 같은 자원은 TabM 또는 기존 트리 앙상블 검증에 사용하는 편이 낫다.

## 구현 구조

| 항목 | 결과 |
|---|---|
| 공통 전처리 | `pipeComb_v3` 고정 |
| 차원 축소 | 각 학습 fold 내부에서만 fit |
| 충돌 expert context | 해당 fold의 충돌 pair 행만 사용 |
| 확률 결합 | 두 충돌 클래스의 기존 확률 질량 안에서 재분배 |
| `pipeComb_v3_2` | 생성하지 않음 |

기계 판독용 표는 `experiments/jyp model test/test_007_candidate_metrics.csv`에 저장한다.
