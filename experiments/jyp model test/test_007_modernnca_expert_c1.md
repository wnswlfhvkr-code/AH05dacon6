# test_007_modernnca_expert_c1

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-05T02:57:11+09:00 |
| 모델 | modernnca_collision_expert |
| 전처리 파이프라인 | pipeComb_v3 |
| 선택된 최소 변이 횟수 | - |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 55105 |
| 최종 Macro F1 | 0.382598 |
| 80% 학습 Macro F1 | 0.631361 |
| 20% 검증 Macro F1 | 0.377614 |
| 과적합 격차 | 0.253747 |
| 과적합 여부 | True |
| 설정 파일 | `data\backup\yaml\test_007_modernnca_expert_c1.yaml` |
| 제출 파일 | `data\processed\test_007_modernnca_expert_c1_submission.csv` |
| 모델 아티팩트 | `models\test_007_modernnca_expert_c1.pkl` |

## 하이퍼파라미터

```yaml
name: modernnca_collision_expert
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
- - KIRC
  - KIPAN
- - LGG
  - GBMLGG
base_model:
  svd_components: 128
  dim: 64
  d_block: 128
  n_blocks: 1
  epochs: 5
  batch_size: 128
  predict_batch_size: 512
  class_weight: balanced
  device: cuda
expert_model:
  svd_components: 64
  dim: 32
  d_block: 128
  n_blocks: 1
  epochs: 5
  batch_size: 32
  predict_batch_size: 256
  class_weight: balanced
  device: cuda
```

## 전처리 설정

```yaml
name: pipeComb_v3
```
