# test_002_e8_lsvc_grid

| 항목 | 결과 |
| --- | --- |
| 설명 | E11: E7에서 채택한 피처 + TF-IDF + Balanced LinearSVC vs E7 Logistic Regression C=0.1 |
| 실행 시작 | 2026-08-05T21:35:09+09:00 |
| 실행 종료 | 2026-08-05T21:57:43+09:00 |
| 총 실행 시간 | 0:22:33.926195 |
| 기준 모델 | E7_logistic_regression_C_0p1 |
| 기준 OOF Macro F1 | 0.448957 |
| 비교 C | [0.2, 0.1, 0.05, 0.02] |
| Seed | [42] |
| Fold 수 | 3 |
| 최상위 후보 | C_0p2 |
| 최상위 C | 0.2 |
| 최상위 OOF Macro F1 | 0.444679 |
| 설정 파일 | `configs/test_002_e8_lsvc_grid.yaml` |
| 결과 폴더 | `data/processed/test_002_e8_lsvc_grid` |

## 후보별 결과

| rank | candidate | C | oof_macro_f1 | reference_oof_macro_f1 | delta_vs_reference | train_macro_f1_mean | validation_macro_f1_mean | generalization_gap_mean | generalization_gap_min | generalization_gap_max | feature_count_mean | all_folds_converged | passes_reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | C_0p2 | 0.200000 | 0.444679 | 0.448957 | -0.004278 | 0.893770 | 0.444189 | 0.449581 | 0.422062 | 0.484362 | 4793.333333 | True | False |
| 2 | C_0p1 | 0.100000 | 0.427048 | 0.448957 | -0.021909 | 0.834760 | 0.426697 | 0.408063 | 0.384623 | 0.431226 | 4793.333333 | True | False |
| 3 | C_0p05 | 0.050000 | 0.391315 | 0.448957 | -0.057642 | 0.744606 | 0.390768 | 0.353838 | 0.327662 | 0.382977 | 4793.333333 | True | False |
| 4 | C_0p02 | 0.020000 | 0.326368 | 0.448957 | -0.122589 | 0.581309 | 0.324882 | 0.256427 | 0.248251 | 0.261957 | 4793.333333 | True | False |

## Fold별 결과

| candidate | C | seed | fold | feature_count | train_macro_f1 | validation_macro_f1 | generalization_gap | converged | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C_0p2 | 0.200000 | 42 | 0 | 4812 | 0.904337 | 0.419976 | 0.484362 | True | 108.105273 |
| C_0p2 | 0.200000 | 42 | 1 | 4776 | 0.889663 | 0.447345 | 0.442318 | True | 106.345434 |
| C_0p2 | 0.200000 | 42 | 2 | 4792 | 0.887310 | 0.465248 | 0.422062 | True | 107.501161 |
| C_0p1 | 0.100000 | 42 | 0 | 4812 | 0.838904 | 0.407678 | 0.431226 | True | 104.785361 |
| C_0p1 | 0.100000 | 42 | 1 | 4776 | 0.832934 | 0.424594 | 0.408340 | True | 103.688144 |
| C_0p1 | 0.100000 | 42 | 2 | 4792 | 0.832442 | 0.447819 | 0.384623 | True | 104.624358 |
| C_0p05 | 0.050000 | 42 | 0 | 4812 | 0.747042 | 0.364065 | 0.382977 | True | 102.576336 |
| C_0p05 | 0.050000 | 42 | 1 | 4776 | 0.742475 | 0.391600 | 0.350874 | True | 100.548790 |
| C_0p05 | 0.050000 | 42 | 2 | 4792 | 0.744302 | 0.416640 | 0.327662 | True | 102.060568 |
| C_0p02 | 0.020000 | 42 | 0 | 4812 | 0.579304 | 0.317347 | 0.261957 | True | 104.726694 |
| C_0p02 | 0.020000 | 42 | 1 | 4776 | 0.579396 | 0.320323 | 0.259073 | True | 106.171181 |
| C_0p02 | 0.020000 | 42 | 2 | 4792 | 0.585226 | 0.336975 | 0.248251 | True | 103.097792 |

## 설정

```yaml
project:
  name: AH05dacon6
  experiment_name: test_002_e8_lsvc_grid
  description: 'E11: E7에서 채택한 피처 + TF-IDF + Balanced LinearSVC vs E7 Logistic Regression
    C=0.1'
  seed: 42
  python_version: '3.14'
  task_type: linear_svc_regularization_grid
  parent_experiment: test_002_v08_overfit_grid_C_0p1
data:
  raw_dir: data/raw
  processed_dir: data/processed/test_002_e8_lsvc_grid
  train_file: train.csv
  test_file: test.csv
  submission_file: sample_submission.csv
  target_column: SUBCLASS
  id_column: ID
  wt_token: WT
checks:
  expected_train_rows: 6201
  expected_test_rows: 2546
  expected_gene_count: 4384
  expected_class_count: 26
  required_unique_id: true
  required_same_gene_order: true
validation:
  strategy: fixed_stratified_group_class_burden
  fixed_split_file: data/processed/fixed_stratified_group_cv_splits.csv
  n_splits: 3
  seeds:
  - 42
  group: normalized_exact_mutation_profile
  burden_bins:
    boundaries:
    - 0
    - 6
    - 10
    - 17
    - 34
    labels:
    - 0-5
    - 6-9
    - 10-16
    - 17-33
    - 34+
reference:
  name: E7_logistic_regression_C_0p1
  seed: 42
  oof_macro_f1: 0.448957
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
  name: linear_svc
  class_weight: balanced
  max_iter: 20000
  tol: 0.0001
  dual: auto
  seed_plus_fold: true
grid:
  C_values:
  - 0.2
  - 0.1
  - 0.05
  - 0.02
  reuse_completed: true
training:
  metric: macro_f1
  diagnostic:
    train_macro_f1: true
    validation_macro_f1: true
    generalization_gap: true
    overfit_warning_threshold: 0.1
  test_prediction: false
```

## 결과
Balanced LinearSVC는 단독 모델로 LR C=0.1을 대체하지 못했습니다.

|LinearSVC C|	OOF F1|	LR C=0.1 대비|	Validation|	Gap|
|---|---|---|---|---|
|0.2|	0.444679|	-0.004278|	0.444189|	0.449581|
|0.1|	0.427048|	-0.021909|	0.426697|	0.408063|
|0.05|	0.391315|	-0.057642|	0.390768|	0.353838|
|0.02|	0.326368|	-0.122589|	0.324882|	0.256427|


## 해석 
LinearSVC에서는 규제를 강화할수록:
- Train F1 감소
- gap 감소
- Validation과 OOF가 더 큰 폭으로 감소

즉, LR에서 발견된 C=0.1 같은 좋은 균형점은 나오지 않았습니다. LinearSVC 후보 중에서는 기존 설정인 C=0.2가 가장 낫습니다.

LR C=0.1과 Fold별로 비교하면:

|Fold|	LR C=0.1|	LinearSVC C=0.2|	변화|
|---|---|---|---|
|0|	0.435248|	0.419976|	-0.015272|
|1|	0.453723|	0.447345|	-0.006378|
|2|	0.456209|	0.465248|	+0.009039|

LinearSVC가 Fold 2에서는 LR보다 높기 때문에 결정경계가 완전히 동일하지는 않습니다. 따라서:
- 단독 모델: 미채택
- 나머지 seed 단독 확장: 아직 불필요
- LR과의 앙상블 다양성 후보: 유지