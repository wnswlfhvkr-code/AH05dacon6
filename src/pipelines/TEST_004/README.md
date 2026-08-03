# TEST_004 pipelines

## EM_SJ_pipeComb_v1

- 기반: Word+Char TF-IDF·LinearSVC 중심 `jsj_v8`
- 구조 전문가: XGBoost E4 기반 `em_v16` 40% + `em_v24` 60%
- 적용 구간: `KIRC↔KIPAN`, `LGG↔GBMLGG`
- 5-Fold OOF Macro F1: `0.5285696598`
- 외부 데이터 및 test 학습 사용: 없음

재현 및 제출 CSV 생성:

```bash
python -m src.reproduce_test_004_7 --config configs/test_004_7.yaml
```

결과: `data/processed/EM_SJ_pipeComb_v1.csv`
