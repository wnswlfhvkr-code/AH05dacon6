# test_006_m7

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-03T23:49:15+09:00 |
| 모델 | muat |
| 전처리 파이프라인 | jyp_raw |
| 선택된 최소 변이 횟수 | - |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 4384 |
| 최종 Macro F1 | 0.274130 |
| 80% 학습 Macro F1 | 0.529214 |
| 20% 검증 Macro F1 | 0.274130 |
| 과적합 격차 | 0.255084 |
| 과적합 여부 | True |
| 설정 파일 | `configs/test_006_m7.yaml` |
| 제출 파일 | `data/processed/test_006_m7_muat_submission.csv` |
| 모델 아티팩트 | `models/test_006_m7.pkl` |

## 하이퍼파라미터

```yaml
name: muat
input_mode: protein_proxy
allow_proxy_mode: true
mutation_sampling_size: 128
variant_hash_size: 65536
gene_embedding_dim: 64
variant_embedding_dim: 32
model_dim: 128
num_layers: 2
num_heads: 4
hidden_dim: 64
dropout: 0.25
epochs: 30
learning_rate: 0.0003
weight_decay: 0.0001
batch_size: 64
num_workers: 0
gradient_clip_norm: 1.0
label_smoothing: 0.05
class_weight: balanced
early_stopping_rounds: 5
device: auto
```

## 전처리 설정

```yaml
name: jyp_raw
show_progress: true
progress_interval: 25000
```
