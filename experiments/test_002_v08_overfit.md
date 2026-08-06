# test_002_v08_overfit

| 항목 | 결과 |
| --- | --- |
| 실행 시작 | 2026-08-05T18:07:32+09:00 |
| 실행 종료 | 2026-08-05T18:24:18+09:00 |
| 총 실행 시간 | 0:16:46.027702 |
| 모델 | logistic_regression |
| 전처리 파이프라인 | jh_v08 |
| 검증 | StratifiedGroupKFold 3-Fold × 3 seeds |
| 학습 데이터 행 수 | 6201 |
| Fold 피처 수 | 평균 4787.1, 범위 4774~4812 |
| OOF Macro F1 | 0.439016 ± 0.007888 |
| Fold Train Macro F1 평균 | 0.904401 |
| Fold Validation Macro F1 평균 | 0.437303 |
| 평균 generalization gap | +0.467097 |
| 최대 generalization gap | +0.501696 |
| 과적합 경고 Fold (gap > 0.10) | 9/9 |
| 설정 파일 | `configs/test_002_v08_overfit.yaml` |
| 제출 파일 | `data/processed/test_002_v08_overfit/submission_test_002_v08_overfit.csv` |
| 요약 JSON | `data/processed/test_002_v08_overfit/jh_v08_sgkf_summary.json` |

## Seed별 OOF

| seed | oof_macro_f1 |
| --- | --- |
| 42 | 0.434502 |
| 2026 | 0.434423 |
| 777 | 0.448124 |

## Fold별 결과

| seed | fold | feature_count | train_macro_f1 | validation_macro_f1 | generalization_gap | converged | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0 | 4812 | 0.920162 | 0.418466 | 0.501696 | True | 105.152300 |
| 42 | 1 | 4776 | 0.906814 | 0.444431 | 0.462383 | True | 110.931344 |
| 42 | 2 | 4792 | 0.889201 | 0.439561 | 0.449640 | True | 112.146728 |
| 2026 | 0 | 4782 | 0.904080 | 0.433622 | 0.470458 | True | 105.833484 |
| 2026 | 1 | 4790 | 0.914226 | 0.425776 | 0.488451 | True | 108.625996 |
| 2026 | 2 | 4788 | 0.893118 | 0.441946 | 0.451172 | True | 112.131052 |
| 777 | 0 | 4783 | 0.900641 | 0.441523 | 0.459117 | True | 109.724820 |
| 777 | 1 | 4774 | 0.896460 | 0.448516 | 0.447944 | True | 109.074257 |
| 777 | 2 | 4787 | 0.914906 | 0.441889 | 0.473016 | True | 108.336922 |

## 하이퍼파라미터

```yaml
name: logistic_regression
C: 1.0
class_weight: balanced
solver: saga
penalty: l2
max_iter: 3000
tol: 0.0001
n_jobs: -1
seed_plus_fold: true
```

## 전처리 설정

```yaml
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
```
최종 비교

|C	|Validation| F1|	OOF F1|	Gap|	판정|
|---|---|---|---|---|---|
|1.0|	0.437303|	0.439016|	0.467097|	과적합 기준|
|0.3|	0.446148|	0.447685|	0.416401|	개선|
|0.1|	0.451799|	0.453399|	0.344633|	최종 추천|
|0.03|	0.440443|	0.442654|	0.245738|	규제가 다소 강함|
|0.01|	0.398882|	0.401609|	0.180266|	과도한 규제|


C=0.1은 기존 E7 대비:
- OOF: +0.014382
- Validation: +0.014496
- 평균 gap: -0.122465
- Train F1: 0.904401 → 0.796432
- OOF 표준편차: 0.007888 → 0.007268
- 세 seed 모두 개선
- 9개 Fold 모두 수렴

따라서 단순히 gap만 줄인 것이 아니라 과적합을 완화하면서 검증 성능과 안정성까지 개선한 후보입니다.

다만 gap 0.344633은 여전히 크고 9개 Fold 모두 경고 기준을 넘으므로, “과적합 해결”이 아니라 현재 피처 표현에서 가장 적절한 규제점 발견으로 표현하는 것이 정확합니다.