# pipeComb_em_v1_006 교차 적합 스태킹 설계

## 기존 결과 분석

| 버전 | 결합 방식 | OOF Macro F1 | Train F1 | Holdout F1 | 과적합 격차 |
| --- | --- | ---: | ---: | ---: | ---: |
| v1 | LinearSVC·XGBoost·LightGBM soft voting | 0.427222 | 0.649330 | 0.410967 | 0.238364 |
| v1_001 | broad 40% + E14 60% | 0.357068 | 0.424490 | 0.347075 | 0.077415 |
| v1_002 | broad 35% + E14 65% | 0.355000 | 0.420078 | 0.346294 | 0.073784 |
| v1_003 | broad 30% + E14 70% | 0.354283 | 0.415824 | 0.342142 | 0.073682 |
| v1_004 | broad 25% + E14 75% | 0.352362 | 0.413144 | 0.341252 | 0.071891 |
| v1_005 | broad 2.44% + E14 97.56% | 0.346251 | 0.400301 | 0.329216 | 0.071085 |

E14 비율을 높일수록 격차는 0.00633만 감소했지만 OOF F1은 0.01082
하락했다. 따라서 v1_001~005는 안정적인 최적점이라기보다 강한 규제로
인한 언더피팅 구간이다. 반대로 v1은 OOF 성능은 높지만 Train-Validation
격차가 0.23836으로 크다.

## v1_006 방법: cross-fitted regularized stacking

Weighted Soft Voting은 모든 표본과 암종에 같은 가중치를 사용한다. v1_006은
다음 세 base learner의 **outer-train 내부 OOF 확률**로 다중 로지스틱 meta
learner를 학습한다.

1. JSJ9 Word/Char TF-IDF → LinearSVC
2. EMV45 mutation 피처 → 규제형 XGBoost
3. 동일 EMV45 피처 → 규제형 LightGBM

각 확률에는 temperature scaling을 적용한다. meta 입력은 모델별 26개
log-probability, entropy, top-1/top-2 margin으로 총 `3 × (26 + 2) = 84개`다.
StandardScaler 뒤에 `C=0.05`, L2 기본 규제, class-balanced multinomial
LogisticRegression을 사용한다. 고정 가중치와 달리 암종별 신뢰도와 모델별
오류 패턴을 학습하면서 작은 meta 공간과 강한 L2 규제로 과적합을 제어한다.

## 하이퍼파라미터 근거

- XGBoost: v1의 350 trees를 유지하되 learning rate 0.025,
  min_child_weight 8, alpha 1.5, lambda 15로 규제를 강화한다.
- LightGBM: leaves 11, depth 4, min_child_samples 45, lambda 18로 복잡도를
  제한한다.
- 두 트리 모델 모두 행·열 subsampling으로 상관된 유전자 피처에 대한
  의존도를 줄인다.
- LinearSVC: `C=0.10`으로 v1의 0.15보다 낮춰 희소 n-gram 과적합을 줄인다.
- 확률 temperature 1.15~1.20으로 base learner의 과신을 완화한다.
- meta learner: `C=0.05`로 class-specific 조합은 허용하되 계수 폭증을 막는다.

## 누수 방지 실험 절차

1. 기존과 동일한 seed 42, outer Stratified 5-fold를 고정한다.
2. 각 outer fold마다 전처리를 outer-train으로만 fit하고 outer-valid/test에는
   transform만 적용한다.
3. outer-train 안에서 다시 Stratified 5-fold를 만들고 base learner의 inner
   OOF 확률만으로 meta learner를 fit한다.
4. base learner를 전체 outer-train에 다시 fit한 뒤 outer-valid를 예측한다.
5. 하이퍼파라미터 선택에는 test를 사용하지 않는다. test의 통계, 인코딩,
   결측치 정보도 fit에 사용하지 않는다.
6. SUBCLASS는 그대로 유지하며 GBMLGG, KIPAN, STES를 하위 암종으로
   분해하지 않는다.

## 판정 기준

- 1차: OOF Macro F1이 v1의 0.427222 이상
- 2차: Train-Validation gap이 v1의 0.238364보다 작고 목표값 0.10에 접근
- 안정성: fold 표준편차가 v1의 약 0.00935보다 크게 악화되지 않을 것
- 위 세 조건을 동시에 확인하며 holdout 한 번의 결과만으로 선택하지 않는다.

현재 파일은 설계 및 실행 가능 상태만 구성했다. 학습, 평가, submission 생성은
수행하지 않았다.
