# test_003_f2

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-07-31T22:26:50+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | jyp_f2 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 23213 |
| 검증 Macro F1 | 0.365969 |
| 설정 파일 | `configs\test_003.yaml` |
| 제출 파일 | `data\processed\test_003_f2_submission.csv` |
| 모델 아티팩트 | `models\test_003_f2.pkl` |

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
name: jyp_f2
burden_clip_quantile: 0.99
show_progress: true
progress_interval: 25000
```

## 결과 해석

- F1 대비 ΔF1은 `+0.019050`입니다.
- 유전자별 변이 유무뿐 아니라 같은 유전자에서 어떤 변이 유형이 발생했는지가 추가 정보를 제공했습니다.
- Test 예측의 `23.016%`가 바뀌어 실질적인 분류 변화가 확인됐습니다.
