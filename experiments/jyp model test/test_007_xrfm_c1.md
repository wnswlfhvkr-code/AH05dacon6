# test_007_xrfm_c1

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-05T02:55:51+09:00 |
| 모델 | xrfm |
| 전처리 파이프라인 | pipeComb_v3 |
| 선택된 최소 변이 횟수 | - |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 55105 |
| 최종 Macro F1 | 0.292941 |
| 80% 학습 Macro F1 | 0.729065 |
| 20% 검증 Macro F1 | 0.282050 |
| 과적합 격차 | 0.447015 |
| 과적합 여부 | True |
| 설정 파일 | `data\backup\yaml\test_007_xrfm_c1.yaml` |
| 제출 파일 | `data\processed\test_007_xrfm_c1_submission.csv` |
| 모델 아티팩트 | `models\test_007_xrfm_c1.pkl` |

## 하이퍼파라미터

```yaml
name: xrfm
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
device: cpu
validation_fraction: 0.15
iterations: 1
max_leaf_size: 512
matrix_batch_size: 128
regularization: 0.001
bandwidth: 10.0
time_limit_seconds: 300
verbose: false
```

## 전처리 설정

```yaml
name: pipeComb_v3
```
