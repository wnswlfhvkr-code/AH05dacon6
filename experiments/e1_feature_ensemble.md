# E1 변이 파생 피처 및 OOF 앙상블 실험

## 목적

제공된 유전체 변이 문자열에서 변이 유형과 환자 단위 집계 피처를 생성하고, 서로 다른 표현을 사용하는 모델의 OOF 확률을 결합했을 때의 변화를 확인한다.

## 실행 파일

```text
notebooks/e1_feature_ensemble_oof.ipynb
```

VS Code 또는 Jupyter에서 노트북을 열고 전체 셀을 순서대로 실행한다.

```bash
jupyter notebook notebooks/e1_feature_ensemble_oof.ipynb
```

원본 데이터는 다음 위치에 준비한다.

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

실행 결과는 Git 관리 대상이 아닌 `data/processed/`에 생성된다.

## 피처 구성

### 기본 변이 피처

- 유전자별 변이 조합 코드
- 환자별 변이 유전자 수
- Frameshift, Stop, Missense, Silent 개수
- 시작·종료 아미노산별 집계
- 다중 변이 유전자 수

### E1 파생 피처

- 유전자별 `MUT`, `FS`, `STOP`, `MISSENSE`, `SILENT` 이진 피처
- 전체 변이 이벤트 수와 로그 변환
- 변이 유전자 비율
- 변이 유전자당 이벤트 수
- 변이 유형별 비율
- 변이 유형 엔트로피
- 지배적인 변이 유형 비율
- 여러 변이 유형이 함께 나타나는 유전자 수

## 모델 및 검증

- Stratified 5-Fold
- Seed: 42, 2026
- 기본 LightGBM
- E1 LightGBM
- 유전자-변이 토큰 TF-IDF + Linear SVM
- OOF 확률 기반 가중 평균
- 평가 지표: Macro F1

최종 확률 조합:

```text
Linear SVM 90% + 기본 LightGBM 5% + E1 LightGBM 5%
```

## 실험 결과

| 구성 | OOF Macro F1 |
| --- | ---: |
| 기본 LightGBM | 0.44203 |
| E1 LightGBM 단독 | 0.43672 |
| 기존 보정 앙상블 | 0.48916 |
| E1 LightGBM 완전 교체 | 0.48519 |
| SVM 90% + 기본 LGB 5% + E1 LGB 5% | 0.49177 |

E1 LightGBM은 단독 성능이 기준 모델보다 낮았으므로 대체 모델로 사용하지 않았다. 기존 모델과 다른 오류 패턴을 제공하는 다양성 모델로 5%만 결합했다.

## 데이터 및 규정 확인

- 제공된 Train/Test 데이터만 사용
- 외부 환자 데이터 사용 없음
- 외부 유전자·단백질·pathway 데이터 사용 없음
- 피처 선택과 라벨 인코딩은 Train 데이터에서 수행
- Test 데이터는 학습, 피처 선택, 클래스 보정에 사용하지 않음
- 최종 모델 선택은 OOF 검증 결과를 기준으로 수행

## 생성 파일

노트북 실행 시 `data/processed/`에 다음 파일이 생성된다.

```text
submission_01_lgb_raw.csv
submission_02_sparse_blend_raw.csv
submission_04_sparse_blend_aggressive.csv
submission_05_e1_feature_blend_corrected.csv
run_summary.json
```

`data/processed/`의 결과 파일과 모델 확률 배열은 재생성 가능한 산출물이므로 Git에 커밋하지 않는다.

## 재현성 점검

- 노트북 코드 셀 Python 문법 검사
- Train/Test 유전자 열과 순서 일치 검사
- 예측 행 수와 제출 양식 검사
- 예측 라벨이 Train의 26개 클래스에 포함되는지 검사
- 결측 예측 여부 검사

## 관련 이슈

Closes #2
