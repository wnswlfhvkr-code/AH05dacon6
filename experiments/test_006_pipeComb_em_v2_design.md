# pipeComb_em_v2: EMV46 + JSJ9 중복 제거 설계

```text
JSJ9 Word/Char TF-IDF → LinearSVC ─┐
                                    ├→ inner-OOF 확률 → L2 Meta-Learner
EMV46 수치 피처 → XGBoost ─────────┤
EMV46 수치 피처 → LightGBM ────────┘
```

JSJ9의 tree 뷰에는 유전자 변이 여부와 변이 구조 요약이 포함된다. 이 정보는
EMV46의 mutation presence, burden, consequence, multi-hit 및 F01~F19 파생
피처와 겹치므로 결합에서 제외한다. EMV46은 한 번만 생성하며 JSJ9에서는
희소 텍스트 표현만 추가한다. 두 피처군을 한 행렬로 섞지 않고 LinearSVC에는
text 뷰, XGBoost와 LightGBM에는 EMV46 tree 뷰만 전달한다.

모델 값은 최고 조합으로 제공된 pipeComb v1_007 설정을 기준으로 유지한다.

- LinearSVC: C 0.10, temperature 1.20
- XGBoost: 350 trees, depth 3, learning rate 0.025, temperature 1.15
- LightGBM: 280 trees, leaves 11, depth 4, temperature 1.15
- Meta-Learner: L2 LogisticRegression, C 0.03
- Stacking: 학습 데이터 내부 5-fold OOF 확률만 사용

## 누수 방지 Early Stopping

각 Stacking fold는 다음 세 부분으로 분리한다.

```text
Inner-train 85% → XGBoost·LightGBM 학습
Inner-train 15% → early stopping 전용
Inner-valid      → Meta-Learner용 OOF 확률 생성
```

Meta-Learner용 Inner-valid 레이블은 early stopping에 사용하지 않는다. XGBoost와
LightGBM 모두 patience 30을 적용하고, 다섯 fold의 best iteration 중앙값으로
전체 학습용 Base Model을 다시 fit한다. LinearSVC는 boosting 모델이 아니므로
early stopping 대상에서 제외한다.

## 중첩 교차검증 제거

기존 구조는 train.py의 Outer 5-fold 각각에서 Stacking의 Inner 5-fold를 다시
실행했다. v2는 파이프라인의 `evaluation_folds=1`로 outer OOF를 끄고 다음
두 번만 Stacking을 수행한다.

```text
1. 80% train: Inner 5-fold OOF stacking → 20% holdout 평가
2. 전체 train: Inner 5-fold OOF stacking → test 예측
```

Stacking 한 번은 `5 folds × 3 base models + 전체 재학습 3개 = 18회`의 base
학습을 사용한다. 따라서 전체 base 학습은 기존 126회에서 36회로 줄어든다.
Outer OOF Macro F1 대신 고정 seed의 stratified 20% holdout Macro F1이 해당
실행의 평가값이 된다.

- 학습 fold에서만 EMV46과 TF-IDF를 fit한다.
- validation/test에는 동일 객체의 transform만 적용한다.
- GBMLGG, KIPAN, STES를 하위 암종으로 분해하지 않는다.
- 외부 데이터는 사용하지 않는다.

구현과 설정만 생성했으며 학습, 평가, submission 생성 및 결과 분석은 수행하지
않았다.
