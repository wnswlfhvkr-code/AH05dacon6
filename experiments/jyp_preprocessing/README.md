# JYP 전처리 실험 기록

`test_003`의 JYP 전처리 단계별 결과와 파라미터 탐색 기록을 모아 둔 폴더입니다.

- `test_003_raw.md`~`test_003_f5_*.md`: 단계별 ablation 결과
- `test_003_preprocessing_optimizer.csv/.md`: 전처리 파라미터 탐색 결과
- `test_003_f5_no_raw_missmask.md`: F5 no-raw missmask 결과
- `test_003_f4_no_raw_hot5.md`: paired 5-Fold 검증 후 채택한 F4 최종 결과
- `test_003_f6_no_raw.md`: F6 TF-IDF·SVD 단일 실행 결과
- `test_003_f7_paircontrast_no_raw.md`: F7 pair-contrast 단일 실행 결과
- `test_003_f7_paircontrast_no_raw_kidney_glioma_k3_a4.md`: F7 다중 Fold 검증, 승격 근거와 진단
- `test_003_전체결과_인사이트.md`: 전체 단계 비교, submission 변화율, 채택 판단과 종합 인사이트

각 결과 Markdown의 `적용 전처리 파라미터`에는 실제 실행에서 확인한 값만 기록합니다.
JYP 전처리 16개는 `src/pipelines/jyp_preprocessing/`의 각 `pipeline_jyp_*.py`
파일에 독립적으로 구현되어 있으며, `configs/test_003.yaml`의
`preprocessing.name`으로 선택합니다. 별도 실행기는 사용하지 않습니다.
과거 결과 파일명과 실행 ID는 재현성을 위해 유지하고, 현재 등록 이름은
`jyp_raw`, `jyp_f0`~`jyp_f7` 형식으로 표기합니다.

현재 결과의 핵심 결론은 RAW exact-cell 인코딩 제외, F1 변이 부담과 F2 변이 유형 채택, F4 no-raw hot5를 기준으로 한 F7 pair-contrast 승격입니다. F5는 단일 holdout 최고점이지만 독립 다중 seed 근거가 부족하고, F6는 성능이 하락했습니다.

F7 전체 검증은 `2026-08-02 04:52:51 KST`에 완료됐으며 선택 후보는 `f7_both_k3`, 검증기 판정은 `adopt_f7`입니다. 상세 수치와 통과·실패 진단은 `test_003_f7_paircontrast_no_raw_kidney_glioma_k3_a4.md`에 고정해 두었습니다.

F7 both K3 제출 리더보드 점수는 `0.3034169097`이며 F4 제출 `0.2878895927`보다 `+0.0155273170` 높습니다.

최종 실행 설정은 `configs/test_003.yaml`이며, 이 폴더에는 팀 공유용으로 확정한 단계별 결과와
인사이트만 버전 관리합니다. 학습 산출물과 자동 생성 보고서는 공용 `src.train`의 기본 규칙을
따르며 Git에 포함하지 않습니다.

```bash
python -m src.train --config configs/test_003.yaml
```
