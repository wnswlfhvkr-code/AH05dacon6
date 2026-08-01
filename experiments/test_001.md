# test_001

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-01T16:17:04+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | em_v4 |
| 선택된 최소 변이 횟수 | 5 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 4171 |
| 검증 Macro F1 | 0.314610 |
| 설정 파일 | `configs/test_001.yaml` |
| 제출 파일 | `data/processed/test_001_submission.csv` |
| 모델 아티팩트 | `models/test_001.pkl` |

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

## 전처리 설정

```yaml
name: em_v4
min_mutation_count: 5
top_genes_per_class: 20
smoothing: 0.5
max_log2_odds: 8.0
shrinkage: 10.0
min_hotspot_count: 5
max_hotspots: 384
```
