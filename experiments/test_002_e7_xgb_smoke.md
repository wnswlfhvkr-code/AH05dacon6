# test_002_e7_xgb_smoke: Raw E7과 TF-IDF E7 + XGBoost 성능비교

| 항목 | 결과 |
| --- | --- |
| 설명 | E7 Raw/TF-IDF 피처의 XGBoost seed 42 예비 비교 |
| 실행 시작 | 2026-08-05T20:48:18+09:00 |
| 실행 종료 | 2026-08-05T21:05:26+09:00 |
| 총 실행 시간 | 0:17:08.513058 |
| 기준 모델 | E7_logistic_C_0p1 |
| 기준 OOF Macro F1 | 0.448957 |
| Seed | [42] |
| Fold 수 | 3 |
| 설정 파일 | `configs/test_002_e7_xgb_smoke.yaml` |
| 결과 폴더 | `data/processed/test_002_e7_xgb_smoke` |

## 후보별 결과

| candidate | pipeline | oof_macro_f1 | reference_oof_macro_f1 | delta_vs_reference | train_macro_f1_mean | validation_macro_f1_mean | generalization_gap_mean | generalization_gap_max | feature_count_mean | all_folds_converged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_e7_xgb | jh_v08 | 0.434117 | 0.448957 | -0.014840 | 0.665063 | 0.430076 | 0.234987 | 0.245296 | 4793.333333 | True |
| tfidf_e7_xgb | jh_v09 | 0.424476 | 0.448957 | -0.024481 | 0.689669 | 0.421199 | 0.268470 | 0.269857 | 4793.333333 | True |

## Fold별 결과

| candidate | pipeline | seed | fold | train_size | validation_size | feature_count | best_iteration | train_macro_f1 | validation_macro_f1 | generalization_gap | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_e7_xgb | jh_v08 | 42 | 0 | 4107 | 2094 | 4812 | 815 | 0.647249 | 0.408734 | 0.238515 | 114.765025 |
| raw_e7_xgb | jh_v08 | 42 | 1 | 4149 | 2052 | 4776 | 1202 | 0.689717 | 0.444420 | 0.245296 | 136.209749 |
| raw_e7_xgb | jh_v08 | 42 | 2 | 4146 | 2055 | 4792 | 913 | 0.658223 | 0.437074 | 0.221149 | 119.422643 |
| tfidf_e7_xgb | jh_v09 | 42 | 0 | 4107 | 2094 | 4812 | 693 | 0.676388 | 0.406531 | 0.269857 | 206.366485 |
| tfidf_e7_xgb | jh_v09 | 42 | 1 | 4149 | 2052 | 4776 | 859 | 0.698981 | 0.430318 | 0.268663 | 232.191047 |
| tfidf_e7_xgb | jh_v09 | 42 | 2 | 4146 | 2055 | 4792 | 758 | 0.693638 | 0.426749 | 0.266890 | 217.276464 |

## 설정

```yaml
project:
  name: AH05dacon6
  experiment_name: test_002_e7_xgb_smoke
  description: E7 Raw/TF-IDF 피처의 XGBoost seed 42 예비 비교
  task_type: model_smoke_comparison
  parent_experiment: test_002_v08_overfit_grid_C_0p1
data:
  raw_dir: data/raw
  processed_dir: data/processed/test_002_e7_xgb_smoke
  train_file: train.csv
  test_file: test.csv
  submission_file: sample_submission.csv
  target_column: SUBCLASS
  id_column: ID
validation:
  fixed_split_file: data/processed/fixed_stratified_group_cv_splits.csv
  n_splits: 3
  seeds:
  - 42
reference:
  name: E7_logistic_C_0p1
  seed: 42
  oof_macro_f1: 0.448957
candidates:
- name: raw_e7_xgb
  preprocessing:
    name: jh_v08
    parser_version: jh_parser_v1
    f4_min_support: 5
    features:
      f0_gene_mutation_presence: true
      f1_sample_summary: true
      f2_gene_consequence: false
      f3_global_amino_acid_position: true
      f3_gene_aware_amino_acid_position: false
      f4_exact_hotspot: true
    f3_position_bins:
    - 1-50
    - 51-100
    - 101-200
    - 201-400
    - 401-800
    - 801+
    f1_scaling: fold_local_standard_scaler
- name: tfidf_e7_xgb
  preprocessing:
    name: jh_v09
    parser_version: jh_parser_v1
    f4_min_support: 5
    features:
      f0_gene_mutation_presence: true
      f1_sample_summary: true
      f2_gene_consequence: false
      f3_global_amino_acid_position: true
      f3_gene_aware_amino_acid_position: false
      f4_exact_hotspot: true
    f3_position_bins:
    - 1-50
    - 51-100
    - 101-200
    - 201-400
    - 401-800
    - 801+
    f1_scaling: fold_local_standard_scaler
    tfidf_sublinear_tf: true
    tfidf_norm: l2
model:
  name: xgboost
  objective: multi:softprob
  eval_metric: mlogloss
  n_estimators: 1500
  learning_rate: 0.02
  max_depth: 3
  min_child_weight: 10
  gamma: 0.1
  subsample: 0.8
  colsample_bytree: 0.6
  reg_alpha: 1.0
  reg_lambda: 10.0
  max_delta_step: 1
  early_stopping_rounds: 100
  tree_method: hist
  device: cpu
  n_jobs: -1
  verbosity: 0
```

## 결과 

|후보|	OOF F1|	C=0.1 Logistic 대비|	Validation 평균|	Gap	|시간|
|---|---|---|---|---|---|
|C=0.1 Logistic|	0.448957|	기준|	약 0.448|	약 0.347|	—|
|Raw E7 XGBoost|	0.434117|	-0.014840|	0.430076|	0.234987|	약 370초|
|TF-IDF E7 XGBoost|	0.424476|	-0.024481|	0.421199|	0.268470|	약 656초|

## 판정
tfidf_e7_xgb는 Raw XGBoost보다 OOF가 -0.009641 낮음
TF-IDF 적용 후 Train F1과 gap은 오히려 증가
실행 시간은 약 1.8배 증가
기준 C=0.1 Logistic보다 -0.024481
따라서 TF-IDF는 XGBoost에 도움이 되지 않았고 명확한 미채택입니다. 나머지 seed도 실행할 필요가 없습니다.
