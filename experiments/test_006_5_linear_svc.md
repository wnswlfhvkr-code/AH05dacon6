# test_006_5

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-04T13:47:32+09:00 |
| 모델 | linear_svc |
| 전처리 파이프라인 | em_v16 |
| 선택된 최소 변이 횟수 | 5 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 4616 |
| 최종 Macro F1 | 0.333131 |
| 80% 학습 Macro F1 | 0.957377 |
| 20% 검증 Macro F1 | 0.324455 |
| 과적합 격차 | 0.632923 |
| 과적합 여부 | True |
| 설정 파일 | `configs/test_006_5.yaml` |
| 제출 파일 | `data/processed/test_006_5_linear_svc_submission.csv` |
| 모델 아티팩트 | `models/test_006_5.pkl` |

## 하이퍼파라미터

```yaml
name: linear_svc
C: 0.15
class_weight: balanced
max_iter: 20000
tol: 0.0001
dual: auto
```

## 전처리 설정

```yaml
name: em_v16
min_functional_mutation_count: 5
max_raw_gene_features: 1500
top_genes_per_class: 15
smoothing: 0.5
max_log2_odds: 8.0
shrinkage: 10.0
min_hotspot_count: 5
stable_hotspot_folds: 5
min_stable_hotspot_folds: 4
max_stable_hotspots: 128
inner_signature_folds: 5
random_state: 42
```
