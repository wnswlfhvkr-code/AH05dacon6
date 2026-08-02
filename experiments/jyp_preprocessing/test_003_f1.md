# test_003_f1

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-07-31T22:23:18+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | jyp_f1 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 8640 |
| 검증 Macro F1 | 0.346919 |
| 설정 파일 | `configs\test_003.yaml` |
| 제출 파일 | `data\processed\test_003_f1_submission.csv` |
| 모델 아티팩트 | `models\test_003_f1.pkl` |

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
name: jyp_f1
burden_clip_quantile: 0.99
show_progress: true
progress_interval: 25000
```

## 결과 해석

- `f0` 대비 피처는 26개만 늘었지만 ΔF1은 `+0.060522`로 전체 단계 중 가장 큰 단일 개선입니다.
- 총 변이 부담과 변이 유형 구성비가 암종 분류에 강한 환자 단위 신호임을 보여줍니다.
- Test 예측 변화율은 `61.508%`로 모델의 분류 기준이 크게 달라졌습니다.
