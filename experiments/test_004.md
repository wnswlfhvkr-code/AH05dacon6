# test_004

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-07-31T20:54:11+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | jsj_v2 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 4257 |
| 검증 Macro F1 | 0.382563 |
| 설정 파일 | `configs\test_004.yaml` |
| 제출 파일 | `data\processed\test_004_submission.csv` |
| 모델 아티팩트 | `models\test_004.pkl` |

## 하이퍼파라미터

```yaml
name: xgboost
n_estimators: 100
learning_rate: 0.1
max_depth: 6
n_jobs: -1
eval_metric: mlogloss
tree_method: hist
```

## 비교 결과

- `test_001` (`em_v1` + XGBoost): Macro F1 `0.328795`
- `test_004` (`jsj_v2` + XGBoost): Macro F1 `0.382563`
- 개선 폭: `+0.053768`
- 전체 실행 시간: 약 109초

`jsj_v2`는 복합 변이를 개별 이벤트로 분리한 뒤 유전자별 변이 코드와
변이 유형·위치·부담 요약을 4,257개 숫자 피처로 압축했습니다.
