# test_006_m3

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-03T23:36:50+09:00 |
| 모델 | linear_svc |
| 전처리 파이프라인 | em_v16 |
| 선택된 최소 변이 횟수 | 5 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 4616 |
| 최종 Macro F1 | 0.326267 |
| 80% 학습 Macro F1 | 0.959890 |
| 20% 검증 Macro F1 | 0.315065 |
| 과적합 격차 | 0.644825 |
| 과적합 여부 | True |
| 설정 파일 | `configs/test_006_m3.yaml` |
| 제출 파일 | `data/processed/test_006_m3_linear_svc_submission.csv` |
| 모델 아티팩트 | `models/test_006_m3.pkl` |

## 하이퍼파라미터

```yaml
name: linear_svc
C: 0.2
class_weight: balanced
max_iter: 20000
tol: 0.0001
dual: auto
```

## 전처리 설정

```yaml
name: em_v16
min_mutation_count: 5
top_genes_per_class: 20
smoothing: 0.5
max_log2_odds: 8.0
shrinkage: 10.0
min_hotspot_count: 5
max_hotspots: 384
inner_signature_folds: 5
signature_random_state: 42
```
