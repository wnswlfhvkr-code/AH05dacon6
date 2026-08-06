# pipeComb_em_v1_007 Meta-Learner L2 규제 강화 설계

## 목적

v006의 피처와 base learner 조합을 그대로 유지하면서 최종 분류기만 미세
조정한다. Train Macro F1은 0.70 이하를 관리 목표로 두고, OOF Macro F1의
손실을 최소화하면서 Train-Validation 격차를 줄이는 것이 목적이다.

## v006 대비 변경값

| 항목 | v006 | v007 |
| --- | ---: | ---: |
| Meta-Learner | L2 LogisticRegression | L2 LogisticRegression |
| `meta_learner.C` | 0.05 | 0.03 |
| Train Macro F1 관리 목표 | 미지정 | 0.70 이하 |

`C`는 규제 강도의 역수이므로 0.05에서 0.03으로 낮추면 L2 규제가
강해진다. Lasso로 바꾸면 meta 계수가 급격히 0이 되면서 소수 암종 정보가
손실될 수 있어 이번 버전은 Ridge 성격의 L2 LogisticRegression을 유지한다.

## 유지되는 구성

- JSJ9 Word/Char TF-IDF → LinearSVC
- EMV45 → XGBoost
- EMV45 → LightGBM
- inner Stratified 5-fold OOF probability stacking
- 모델별 log-probability·entropy·top-1/top-2 margin
- StandardScaler와 class-balanced meta learner
- base learner의 모든 하이퍼파라미터와 temperature
- 원본 SUBCLASS 유지 및 GBMLGG·KIPAN·STES 비분해

## 평가 방법

1. seed 42와 동일 outer Stratified 5-fold를 사용한다.
2. 전처리와 inner stacking은 각 outer-train 안에서만 fit한다.
3. v006과 v007의 OOF Macro F1, fold 표준편차, Train F1, 과적합 격차를
   같은 fold로 비교한다.
4. Train F1이 0.70 이하여도 OOF F1이 유의하게 하락하면 선택하지 않는다.
5. test 데이터는 최종 선택 후 transform과 predict에만 사용한다.

현재 버전은 구현과 설정만 생성했으며 학습·평가·submission 생성은 수행하지
않았다. Train F1 0.70 이하는 목표값이며 실행 전 보장값이 아니다.
