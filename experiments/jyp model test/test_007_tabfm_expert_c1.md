# test_007_tabfm_expert_c1

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-04T23:41:48+09:00 |
| 모델 | tabfm_collision_expert |
| 전처리 파이프라인 | pipeComb_v3 |
| 선택된 최소 변이 횟수 | - |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 55105 |
| 최종 Macro F1 | 0.473452 |
| 80% 학습 Macro F1 | 0.868998 |
| 20% 검증 Macro F1 | 0.494729 |
| 과적합 격차 | 0.374269 |
| 과적합 여부 | True |
| 설정 파일 | `data\backup\yaml\test_007_tabfm_expert_c1.yaml` |
| 제출 파일 | `data\processed\test_007_tabfm_expert_c1_submission.csv` |
| 모델 아티팩트 | `models\test_007_tabfm_expert_c1.pkl` |

## 하이퍼파라미터

```yaml
name: tabfm_collision_expert
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
ensemble_size: 4
batch_size: 1
expert_device: cuda
```

## 전처리 설정

```yaml
name: pipeComb_v3
```
