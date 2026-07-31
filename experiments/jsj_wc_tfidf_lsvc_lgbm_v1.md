# jsj_wc_tfidf_lsvc_lgbm_v1

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-07-31T16:54:36+09:00 |
| 모델 | wc_tfidf_lsvc_lgbm |
| 전처리 파이프라인 | jsj_v1 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 434242 |
| 검증 Macro F1 | 0.445791 |
| 설정 파일 | `configs\jsj_wc_tfidf_lsvc_lgbm_v1.yaml` |
| 제출 파일 | `data\processed\jsj_wc_tfidf_lsvc_lgbm_v1_submission.csv` |
| 모델 아티팩트 | `models\jsj_wc_tfidf_lsvc_lgbm_v1.pkl` |

## 하이퍼파라미터

```yaml
name: wc_tfidf_lsvc_lgbm
linear_weight: 0.95
tree_weight: 0.05
temperature: 0.5
class_multipliers:
- 0.65
- 1.14
- 1.14
- 1.0
- 0.94
- 1.0
- 1.24
- 1.0
- 1.0
- 1.14
- 0.8
- 1.36
- 1.06
- 1.06
- 0.88
- 1.06
- 1.14
- 1.14
- 1.14
- 1.24
- 0.65
- 1.0
- 0.88
- 0.65
- 0.88
- 0.88
linear:
  C: 0.2
  class_weight: balanced
  max_iter: 10000
  tol: 0.0001
tree:
  objective: multiclass
  n_estimators: 300
  learning_rate: 0.05
  num_leaves: 31
  max_depth: -1
  min_child_samples: 20
  subsample: 1.0
  colsample_bytree: 1.0
  reg_alpha: 0.0
  reg_lambda: 0.0
  n_jobs: -1
  verbosity: -1
```
