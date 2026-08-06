# test_006_em_G01

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-06T10:40:38+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | em_G01 |
| 선택된 최소 변이 횟수 | 5 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 11685 |
| 최종 Macro F1 | 0.446019 |
| 80% 학습 Macro F1 | 0.620425 |
| 20% 검증 Macro F1 | 0.436697 |
| 과적합 격차 | 0.183728 |
| 과적합 여부 | True |
| 설정 파일 | `configs/test_006_em_G01.yaml` |
| 제출 파일 | `data/processed/test_006_em_G01_xgboost_submission.csv` |
| 모델 아티팩트 | `models/test_006_em_G01.pkl` |

## 하이퍼파라미터

```yaml
name: xgboost
objective: multi:softprob
n_estimators: 500
learning_rate: 0.03
max_depth: 3
min_child_weight: 7.0
subsample: 0.75
colsample_bytree: 0.5
reg_alpha: 1.5
reg_lambda: 17.0
early_stopping_rounds: 30
eval_metric: mlogloss
tree_method: hist
n_jobs: -1
```

## 전처리 설정

```yaml
name: em_G01
min_active_count: 20
correlation_threshold: 0.9
correlation_max_features: 3000
correlation_block_size: 256
stability_folds: 5
stability_top_fraction: 0.2
active_epsilon: 1.0e-08
random_state: 42
min_mutation_count: 5
min_functional_mutation_count: 5
min_hotspot_count: 5
max_hotspots: 384
top_genes_per_class: 20
inner_signature_folds: 5
```
