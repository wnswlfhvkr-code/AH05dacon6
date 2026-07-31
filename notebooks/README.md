# Notebooks

탐색적 데이터 분석과 빠른 가설 검증에 사용합니다. 검증된 로직은 `src/`의 파이썬 모듈로 옮깁니다.

## 원본 베이스라인의 프로젝트 대응

| 노트북 단계 | 프로젝트 구현 |
| --- | --- |
| CSV 로딩 | `src/train.py` |
| `ID`, `SUBCLASS` 분리 | `src/train.py` |
| 타깃 LabelEncoder | `src/pipelines/base.py` |
| 문자열 피처 처리 | `src/pipelines/base.py`의 `OrdinalEncoder` |
| XGBClassifier 생성 | `src/models/xgboost_model.py` |
| 학습·검증·전체 재학습 | `src/train.py` |
| 제출 파일 생성 | `src/train.py` |

기본 실험 조건은 `configs/baseline.yaml`에서 관리합니다. 원본 노트북과 동일하게 `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`를 사용합니다.
