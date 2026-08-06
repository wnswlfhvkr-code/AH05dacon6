# test_007_torch_mlp_c1

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-04T20:49:17+09:00 |
| 모델 | torch_mlp |
| 전처리 파이프라인 | pipeComb_v3 |
| 선택된 최소 변이 횟수 | - |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 55105 |
| 최종 Macro F1 | 0.369818 |
| 80% 학습 Macro F1 | 0.964022 |
| 20% 검증 Macro F1 | 0.366140 |
| 과적합 격차 | 0.597883 |
| 과적합 여부 | True |
| 설정 파일 | `data\backup\yaml\test_007_torch_mlp_c1.yaml` |
| 제출 파일 | `data\processed\test_007_torch_mlp_c1_submission.csv` |
| 모델 아티팩트 | `models\test_007_torch_mlp_c1.pkl` |

## 하이퍼파라미터

```yaml
name: torch_mlp
hidden_dims:
- 256
- 128
learning_rate: 0.001
epochs: 50
batch_size: 256
weight_decay: 0.0001
dropout: 0.2
class_weight: balanced
label_smoothing: 0.02
gradient_clip_norm: 5.0
device: cuda
verbose: false
```

## 전처리 설정

```yaml
name: pipeComb_v3
```
