# test_003_f6_no_raw

| 항목 | 결과 |
| --- | --- |
| 실행 ID | `20260802T034550912252+0900-93863c5e` |
| 실행 시작 | 2026-08-02T03:45:50+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | jyp_f6 |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 154 |
| 검증 Macro F1 | 0.347061 |
| 설정 파일 | `configs\test_003.yaml` |
| 제출 파일 | `data/processed/test_003_f6_no_raw_submission.csv` (Git 제외) |
| 모델 아티팩트 | `models/test_003_f6_no_raw.pkl` (Git 제외) |

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

- RAW 원본 피처: `제외`
- 확인 근거: 저장된 모델 아티팩트에서 확인한 실제 실행값입니다.

```yaml
name: jyp_f6
burden_clip_quantile: 0.99
f3_position_min_support: 3
f3_aa_min_support: 3
f4_min_support: 5
f6_min_document_frequency: 2
f6_max_components: 128
f6_random_state: 42
show_progress: true
progress_interval: 25000
```

## 결과 해석

- 최종 F4 대비 피처를 54,463개 줄여 154개만 사용했지만 ΔF1은 `-0.051182`입니다.
- 피처 감소율은 `99.72%`, Test 예측 변화율은 `41.948%`입니다.
- F1 점수 `0.346919`와 F6 점수 `0.347061`이 거의 같아, 압축이 변이 부담 정보는 보존했지만 암종 특이 희소 신호는 충분히 보존하지 못한 것으로 해석됩니다.
