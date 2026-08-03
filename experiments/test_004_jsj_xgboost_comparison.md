# JSJ v1~v8 XGBoost 고정 비교

## 평가 조건

- 분할: Stratified holdout 80/20
- seed: 42
- 모델: XGBoost
- n_estimators: 10
- learning_rate: 0.1
- max_depth: 4
- tree_method: hist
- 평가지표: Macro F1
- 비교 baseline Macro F1: 0.237390
- TF-IDF 상한: Word 2,000 + Char 2,000

기본 TF-IDF 상한(Word 250,000 + Char 180,000)은 XGBoost 실행이
15분 내 완료되지 않았고, Word/Char 각 20,000 설정도 v1이 7분 내
완료되지 않아 전체 버전 비교에는 경량 상한을 사용했습니다.

## 결과

괄호 안은 해당 파이프라인을 앙상블에 적용한 OOF Macro F1입니다.

| GitHub 파일명 | 요약 | Macro F1 | 채택 여부 | 생성자 | 메모 |
| --- | --- | ---: | --- | --- | --- |
| `src/pipelines/pipeline_jsj_v1.py` | Word+Char TF-IDF | 0.333524 | 조건부 채택 | 정세준 | 문자열 피처 기반, 성능은 높지만 실행이 느림 |
| `src/pipelines/pipeline_jsj_v2.py` | 숫자 구조 피처 | 0.313571 | 채택 | 정세준 | 빠르고 XGBoost에 적합, 정식 100-tree 결과 0.382563 |
| `src/pipelines/pipeline_jsj_v3.py` | TF-IDF+구조 피처 | 0.330637 (0.505595) | 채택 | 정세준 | 기본 앙상블 |
| `src/pipelines/pipeline_jsj_v4.py` | Soft 충돌 보정 | 0.330637 (0.507241) | 채택 | 정세준 | 충돌 암종 확률 보정 |
| `src/pipelines/pipeline_jsj_v5.py` | Strict 충돌 보정 | 0.330637 (0.508134) | 채택 | 정세준 | 충돌 쌍 내부 재판정 |
| `src/pipelines/pipeline_jsj_v6.py` | 비대칭 충돌 보정 | 0.330637 (0.508400) | 채택 | 정세준 | 암종 쌍별 강도 조절 |
| `src/pipelines/pipeline_jsj_v7.py` | 전문가 설정 조정 | 0.330637 (0.514965) | 채택 | 정세준 | C·가중치·offset 적용 |
| `src/pipelines/pipeline_jsj_v8.py` | EM v14 전문가 결합 | 0.330637 (0.518823) | 최종 채택 | 정세준 | 앙상블 Public 0.389342 |

## 해석

- XGBoost 단독 경량 비교 최고점은 `jsj_v1`입니다.
- 실행 효율과 정식 100-tree 재현 결과까지 고려하면 XGBoost 전용 파이프라인은 `jsj_v2`를 채택합니다.
- `jsj_v4`~`jsj_v8`의 차이는 피처 전처리가 아니라 충돌 쌍 후처리·전문가 결합 설정이므로 XGBoost 단독 점수는 `jsj_v3`와 같습니다.
- 각 버전의 앙상블 OOF/Public 점수는 `docs/test-004-reproduction.md`에서 별도로 관리합니다.

## 앙상블 비교

| GitHub 파일명 | 앙상블 요약 | OOF Macro F1 | Public Macro F1 | 채택 여부 | 메모 |
| --- | --- | ---: | ---: | --- | --- |
| `src/pipelines/pipeline_jsj_v1.py` | Word+Char TF-IDF 기반 | - | - | 기반 피처 | 단독 제출 기록 없음 |
| `src/pipelines/pipeline_jsj_v2.py` | XGBoost 숫자 구조 피처 | - | - | 보조 피처 | XGBoost 단독 검증 0.382563 |
| `src/pipelines/pipeline_jsj_v3.py` | TF-IDF+LightGBM 기본 앙상블 | 0.505595 | 0.371498 | 채택 | TEST_004_1 |
| `src/pipelines/pipeline_jsj_v4.py` | 두 충돌 암종 쌍 Soft 보정 | 0.507241 | 0.374100 | 채택 | TEST_004_2 |
| `src/pipelines/pipeline_jsj_v5.py` | 충돌 쌍 내부 Strict 재판정 | 0.508134 | 0.375613 | 채택 | TEST_004_3 |
| `src/pipelines/pipeline_jsj_v6.py` | LGG 100%·KIRC 10% 비대칭 보정 | 0.508400 | 0.376826 | 채택 | TEST_004_4 |
| `src/pipelines/pipeline_jsj_v7.py` | 전문가 C·가중치·offset 조정 | 0.514965 | 0.386336 | 채택 | TEST_004_5 |
| `src/pipelines/pipeline_jsj_v8.py` | EM v14 KIRC 전문가 결합 | 0.518823 | 0.389342 | 최종 채택 | TEST_004_6 |
