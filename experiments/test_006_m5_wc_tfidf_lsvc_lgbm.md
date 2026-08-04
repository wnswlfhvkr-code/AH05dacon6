# test_006_m5

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-03T23:36:56+09:00 |
| 모델 | wc_tfidf_lsvc_lgbm |
| 전처리 파이프라인 | jsj_v1 |
| 선택된 최소 변이 횟수 | - |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 434242 |
| 최종 Macro F1 | 0.469640 |
| 80% 학습 Macro F1 | 0.942840 |
| 20% 검증 Macro F1 | 0.469640 |
| 과적합 격차 | 0.473200 |
| 과적합 여부 | True |
| 설정 파일 | `configs/test_006_m5.yaml` |
| 제출 파일 | `data/processed/test_006_m5_wc_tfidf_lsvc_lgbm_submission.csv` |
| 모델 아티팩트 | `models/test_006_m5.pkl` |

## 하이퍼파라미터

```yaml
name: wc_tfidf_lsvc_lgbm
linear_weight: 0.95
tree_weight: 0.05
temperature: 0.5
linear:
  C: 0.2
  class_weight: balanced
  max_iter: 20000
  tol: 0.0001
tree:
  objective: multiclass
  n_estimators: 300
  learning_rate: 0.05
  num_leaves: 31
  max_depth: -1
  min_child_samples: 20
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 0.5
  reg_lambda: 5.0
  class_weight: balanced
  n_jobs: -1
  verbosity: -1
```

## 전처리 설정

```yaml
name: jsj_v1
word_ngram_range:
- 1
- 2
word_min_df: 2
word_max_features: 250000
char_ngram_range:
- 3
- 5
char_min_df: 3
char_max_features: 180000
char_weight: 0.5
sublinear_tf: true
split_multi_event: true
return_bundle: true
```
