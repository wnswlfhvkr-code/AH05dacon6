# linear_svc

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-07-31T16:50:48+09:00 |
| 모델 | linear_svc |
| 전처리 파이프라인 | jsj_v1 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 430000 |
| 검증 Macro F1 | 0.423998 |
| 설정 파일 | `configs\linear_svc.yaml` |
| 제출 파일 | `data\processed\linear_svc_submission.csv` |
| 모델 아티팩트 | `models\linear_svc.pkl` |

## 하이퍼파라미터

```yaml
name: linear_svc
C: 0.2
class_weight: balanced
max_iter: 10000
tol: 0.0001
```
