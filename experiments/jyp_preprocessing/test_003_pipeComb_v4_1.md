# test_003_pipeComb_v4_1

| 항목 | 결과 |
| --- | --- |
| 실행 시각 | 2026-08-04T16:39:32+09:00 |
| 모델 | xgboost |
| 전처리 파이프라인 | pipeComb_v4 |
| 선택된 최소 변이 횟수 | - |
| 학습 데이터 행 수 | 6201 |
| 피처 수 | 55053 |
| 최종 Macro F1 | 0.442491 |
| 80% 학습 Macro F1 | 0.885711 |
| 20% 검증 Macro F1 | 0.464864 |
| 과적합 격차 | 0.420848 |
| 과적합 여부 | True |
| Public Leaderboard Macro F1 | 0.3342990633 |
| pipeComb_v3 LB 대비 | -0.0047107024 |
| 설정 파일 | `configs\test_003.yaml` |
| 제출 파일 | `data\processed\test_003_pipeComb_v4_1_submission.csv` |
| 모델 아티팩트 | `models\test_003_pipeComb_v4_1.pkl` |

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

## 전처리 설정

```yaml
name: pipeComb_v4
auto_profile_groups: true
```

## 리더보드 판정

- `pipeComb_v3` Public LB: `0.3390097657`
- `pipeComb_v4` Public LB: `0.3342990633`
- v3 대비 변화: `-0.0047107024`
- 판정: v4는 그룹 안전 검증 후보로 보존하지만, 최종 제출 우선순위는 v3로 유지한다.
