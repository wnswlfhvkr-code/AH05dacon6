# test_003_raw

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-07-31T22:17:22+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | jyp_raw |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 4384 |
| 검증 Macro F1 | 0.286397 |
| 설정 파일 | `configs\test_003.yaml` |
| 제출 파일 | `data\processed\test_003_raw_submission.csv` |
| 모델 아티팩트 | `models\test_003_raw.pkl` |

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

## 적용 전처리 파라미터

- RAW 원본 피처: `포함`
- 확인 근거: 현재 독립 파이프라인의 `name` 기준으로 정규화해 표시했습니다.

```yaml
name: jyp_raw
show_progress: true
progress_interval: 25000
```

## 결과 해석

- 전체 전처리 비교의 기준점은 Macro F1 `0.286397`입니다.
- F0를 RAW와 함께 추가한 실행과 점수 및 Test 예측이 완전히 같아, RAW가 있는 상태에서는 F0의 추가 효과가 관찰되지 않았습니다.
- 이 결과는 단일 holdout 기준이며 일반화 성능을 의미하지 않습니다.
