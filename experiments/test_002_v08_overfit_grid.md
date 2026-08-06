# test_002_v08_overfit_grid

| 항목 | 결과 |
| --- | --- |
| 설명 | E7 Balanced Logistic Regression L2 C 규제 강도 비교 |
| 실행 시작 | 2026-08-05T19:21:34+09:00 |
| 실행 종료 | 2026-08-05T20:14:50+09:00 |
| 총 실행 시간 | 0:53:15.373505 |
| 기준 C | 1 |
| 비교 C | [0.3, 0.1, 0.03, 0.01] |
| OOF 하락 허용값 | 0.003000 |
| 최소 gap 감소값 | 0.050000 |
| 설정 파일 | `configs/test_002_v08_overfit_grid.yaml` |
| 결과 폴더 | `data/processed/test_002_v08_overfit_grid` |

## 후보별 결과

| candidate | C | train_macro_f1_mean | validation_macro_f1_mean | generalization_gap_mean | generalization_gap_min | generalization_gap_max | oof_macro_f1_mean | oof_macro_f1_std | oof_delta | gap_reduction | improved_seed_count | converged_fold_count | total_fold_count | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_C_1 | 1.000000 | 0.904401 | 0.437303 | 0.467097 | 0.447944 | 0.501696 | 0.439016 | 0.007888 | 0.000000 | 0.000000 | 0 | 9 | 9 | False |
| C_0p3 | 0.300000 | 0.862549 | 0.446148 | 0.416401 | 0.402546 | 0.440295 | 0.447685 | 0.009197 | 0.008669 | 0.050697 | 3 | 9 | 9 | True |
| C_0p1 | 0.100000 | 0.796432 | 0.451799 | 0.344633 | 0.328044 | 0.361912 | 0.453399 | 0.007268 | 0.014382 | 0.122465 | 3 | 9 | 9 | True |
| C_0p03 | 0.030000 | 0.686180 | 0.440443 | 0.245738 | 0.222927 | 0.268327 | 0.442654 | 0.007709 | 0.003637 | 0.221360 | 3 | 9 | 9 | True |
| C_0p01 | 0.010000 | 0.579148 | 0.398882 | 0.180266 | 0.166705 | 0.197281 | 0.401609 | 0.005899 | -0.037407 | 0.286832 | 0 | 9 | 9 | False |

## Seed별 결과

| seed | baseline_oof_macro_f1 | candidate | C | candidate_oof_macro_f1 | delta |
| --- | --- | --- | --- | --- | --- |
| 42 | 0.434502 | C_0p3 | 0.300000 | 0.445081 | 0.010579 |
| 2026 | 0.434423 | C_0p3 | 0.300000 | 0.440072 | 0.005648 |
| 777 | 0.448124 | C_0p3 | 0.300000 | 0.457903 | 0.009779 |
| 42 | 0.434502 | C_0p1 | 0.100000 | 0.448957 | 0.014456 |
| 2026 | 0.434423 | C_0p1 | 0.100000 | 0.449453 | 0.015029 |
| 777 | 0.448124 | C_0p1 | 0.100000 | 0.461786 | 0.013661 |
| 42 | 0.434502 | C_0p03 | 0.030000 | 0.436795 | 0.002293 |
| 2026 | 0.434423 | C_0p03 | 0.030000 | 0.439780 | 0.005356 |
| 777 | 0.448124 | C_0p03 | 0.030000 | 0.451387 | 0.003263 |
| 42 | 0.434502 | C_0p01 | 0.010000 | 0.396212 | -0.038290 |
| 2026 | 0.434423 | C_0p01 | 0.010000 | 0.400709 | -0.033714 |
| 777 | 0.448124 | C_0p01 | 0.010000 | 0.407906 | -0.040218 |

## 선택 결과

| 항목 | 결과 |
| --- | --- |
| 추천 후보 | C_0p1 |
| 추천 C | 0.1 |
| 추천 OOF Macro F1 | 0.453399 |
| 추천 평균 generalization gap | 0.344633 |

## 설정

```yaml
project:
  name: AH05dacon6
  experiment_name: test_002_v08_overfit_grid
  description: E7 Balanced Logistic Regression L2 C 규제 강도 비교
  task_type: overfit_regularization_grid
  parent_experiment: test_002_v08_overfit
grid:
  base_config: configs/test_002_v08_overfit.yaml
  baseline:
    C: 1.0
    result_dir: data/processed/test_002_v08_overfit
  C_values:
  - 0.3
  - 0.1
  - 0.03
  - 0.01
  reuse_completed: true
  selection:
    oof_drop_tolerance: 0.003
    min_gap_reduction: 0.05
output:
  processed_dir: data/processed/test_002_v08_overfit_grid
  report_file: experiments/test_002_v08_overfit_grid.md

## 설정


