# test_007_catboost_c1

| 항목                  | 결과                                                   |
| --------------------- | ------------------------------------------------------ |
| 실행 시각             | 2026-08-04T19:15:25+09:00                              |
| 모델                  | catboost                                               |
| 전처리 파이프라인     | pipeComb_v3                                            |
| 선택된 최소 변이 횟수 | -                                                      |
| 학습 데이터 행 수     | 6201                                                   |
| 피처 수               | 55105                                                  |
| 최종 Macro F1         | 0.460525                                               |
| 80% 학습 Macro F1     | 0.547480                                               |
| 20% 검증 Macro F1     | 0.459183                                               |
| 과적합 격차           | 0.088297                                               |
| 과적합 여부           | False                                                  |
| 설정 파일             | `configs\test_007.yaml`                              |
| 제출 파일             | `data\processed\test_007_catboost_c1_submission.csv` |
| 모델 아티팩트         | `models\test_007_catboost_c1.pkl`                    |

## 하이퍼파라미터

```yaml
name: catboost
n_estimators: 500
learning_rate: 0.03
depth: 6
l2_leaf_reg: 10.0
random_strength: 1.0
bootstrap_type: Bayesian
bagging_temperature: 1.0
loss_function: MultiClass
eval_metric: MultiClass
auto_class_weights: Balanced
task_type: GPU
devices: '0'
thread_count: -1
verbose: false
```

## 전처리 설정

```yaml
name: pipeComb_v3
```
