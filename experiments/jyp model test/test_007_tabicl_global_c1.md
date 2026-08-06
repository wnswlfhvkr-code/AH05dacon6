# test_007_tabicl_global_c1

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-04T23:17:27+09:00 |
| 모델 | tabicl |
| 전처리 파이프라인 | pipeComb_v3 |
| 선택된 최소 변이 횟수 | - |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 55105 |
| 최종 Macro F1 | 0.458556 |
| 80% 학습 Macro F1 | 0.777365 |
| 20% 검증 Macro F1 | 0.483711 |
| 과적합 격차 | 0.293654 |
| 과적합 여부 | True |
| 설정 파일 | `data\backup\yaml\test_007_tabicl_global_c1.yaml` |
| 제출 파일 | `data\processed\test_007_tabicl_global_c1_submission.csv` |
| 모델 아티팩트 | `models\test_007_tabicl_global_c1.pkl` |

## 하이퍼파라미터

```yaml
name: tabicl
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
svd_components: 128
ensemble_size: 4
batch_size: 1
support_many_classes: true
checkpoint_version: tabicl-classifier-v2-20260212.ckpt
device: cuda
use_amp: true
offload_mode: auto
verbose: false
```

## 전처리 설정

```yaml
name: pipeComb_v3
```
