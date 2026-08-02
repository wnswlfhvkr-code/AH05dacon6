# test_003_f3_position

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-07-31T23:08:34+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | jyp_f3_position |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 39537 |
| 검증 Macro F1 | 0.361163 |
| 설정 파일 | `configs\test_003.yaml` |
| 제출 파일 | `data\processed\test_003_f3_position_submission.csv` |
| 모델 아티팩트 | `models\test_003_f3_position.pkl` |

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
name: jyp_f3_position
burden_clip_quantile: 0.99
f3_position_min_support: 2
show_progress: true
progress_interval: 25000
```

## 결과 해석

- F2 대비 ΔF1은 `-0.004806`입니다.
- 위치 구간 피처 16,324개가 추가됐지만 성능이 하락해, 위치 정보만으로는 희소성과 잡음이 더 컸습니다.
- 단독 위치 블록은 채택 근거가 부족합니다.
