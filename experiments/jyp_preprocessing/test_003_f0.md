# test_003_f0

| 항목 | 결과 |
| --- | --- |
| 실행 ID | `20260801T205132919580+0900-83aa0821` |
| 실행 시작 | 2026-08-01T20:51:32+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | jyp_f0 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 8614 |
| 검증 Macro F1 | 0.286397 |
| 설정 파일 | `configs\test_003.yaml` |
| 제출 파일 | `data/processed/test_003_f0_submission.csv` (Git 제외) |
| 모델 아티팩트 | `models/test_003_f0.pkl` (Git 제외) |

## 하이퍼파라미터

```yaml
name: xgboost
n_estimators: 100
learning_rate: 0.1
max_depth: 6
n_jobs: -1
eval_metric: mlogloss
tree_method: hist
```

## 적용 전처리 파라미터

- RAW 원본 피처: `포함`
- 확인 근거: 현재 독립 파이프라인의 `name` 기준으로 정규화해 표시했습니다.

```yaml
name: jyp_f0
show_progress: true
progress_interval: 25000
```

## 결과 해석

- RAW 기준 대비 ΔF1은 `+0.000000`이고 Test 예측 변화율도 `0.000%`입니다.
- 유전자 변이 유무 F0가 exact RAW 인코딩과 함께 있을 때는 완전히 중복된 결과를 만들었습니다.
- F0의 효과는 RAW를 제외한 `f0_no_raw`에서 따로 판단해야 합니다.
