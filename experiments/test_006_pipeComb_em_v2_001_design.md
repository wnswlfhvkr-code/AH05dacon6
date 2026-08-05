# test_006_pipeComb_em_v2_001 설계

## 결합 구조

- `pipeComb_em_v1_007`의 JSJ9 Word/Char TF-IDF text view를 유지한다.
- 기존 EMV45 tree view를 EMV46 tree view로 교체한다.
- EMV46은 EMV45와 F01~F19 고유 피처를 이미 중복 없이 포함하므로 EMV45를 별도로 결합하지 않는다.
- JSJ9의 mutation/structure tree는 EMV46과 중복되므로 제외한다.
- 모델 및 규제 파라미터는 `pipeComb_em_v1_007`과 동일한 LinearSVC + XGBoost + LightGBM OOF stacking을 사용한다.
- XGBoost와 LightGBM은 각 inner-fold 학습 구간에서 다시 분리한 15% 구간으로 early stopping을 수행한다. OOF validation 구간은 early stopping에 사용하지 않는다.

## 실행량과 예상시간

- Outer evaluation 5-fold, 별도 holdout 1회, 최종 전체 학습 1회로 stacking `fit`은 총 7회다.
- stacking 1회마다 inner 5-fold의 3개 base model 15회와 전체 데이터 refit 3회가 필요하다.
- 예상 base model 학습 횟수는 `7 × (5 × 3 + 3) = 126회`다.
- EMV46은 EMV45 베이스와 F01~F19 고유 파생 피처를 한 번씩만 생성한다. 실제 피처 수는 각 fold의 학습 데이터에서 결정한다.
- XGBoost와 LightGBM의 patience는 각각 30이며, fold별 최적 반복 수의 중앙값으로 전체 refit 반복 수를 결정한다.
- 동일 장비 기준 예상 소요시간은 v1_007 실측시간의 약 1.2~2.3배다. 절대 계획 범위는 CPU 환경에서 약 1.5~4.5시간으로 잡는다.
- 조기 종료가 최대 반복 수 전에 발생하지 않으면 최악 조건은 기존 예상과 같은 약 2~6시간이다.
- 기존 결과 파일에는 시작/종료 소요시간이 기록되지 않아 절대시간은 참고값이며, CPU 코어 수·메모리·LightGBM 설치 상태에 따라 달라진다.

## 실행 상태

- 설계 및 정적 검증만 수행한다.
- 학습, 평가, submission 생성은 수행하지 않는다.
