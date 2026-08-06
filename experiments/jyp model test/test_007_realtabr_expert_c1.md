# test_007_realtabr_expert_c1

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-05T03:04:59+09:00 |
| 모델 | realtabr_collision_expert |
| 전처리 파이프라인 | pipeComb_v3 |
| 선택된 최소 변이 횟수 | - |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 55105 |
| 최종 Macro F1 | 0.453173 |
| 80% 학습 Macro F1 | 0.833972 |
| 20% 검증 Macro F1 | 0.466211 |
| 과적합 격차 | 0.367762 |
| 과적합 여부 | True |
| 설정 파일 | `data\backup\yaml\test_007_realtabr_expert_c1.yaml` |
| 제출 파일 | `data\processed\test_007_realtabr_expert_c1_submission.csv` |
| 모델 아티팩트 | `models\test_007_realtabr_expert_c1.pkl` |

## 하이퍼파라미터

```yaml
name: realtabr_collision_expert
class_names:
- ACC
- BLCA
- BRCA
- CESC
- COAD
- DLBC
- GBMLGG
- HNSC
- KIPAN
- KIRC
- LAML
- LGG
- LIHC
- LUAD
- LUSC
- OV
- PAAD
- PCPG
- PRAD
- SARC
- SKCM
- STES
- TGCT
- THCA
- THYM
- UCEC
collision_pairs:
- - GBMLGG
  - LGG
- - KIPAN
  - KIRC
base_model:
  n_estimators: 100
  learning_rate: 0.1
  max_depth: 6
  n_jobs: -1
  eval_metric: mlogloss
  tree_method: hist
  device: cuda
svd_components: 128
expert_device: cpu
n_epochs: 4
batch_size: 64
eval_batch_size: 256
context_size: 32
d_main: 64
patience: 2
val_fraction: 0.15
verbosity: 0
```

## 전처리 설정

```yaml
name: pipeComb_v3
```
