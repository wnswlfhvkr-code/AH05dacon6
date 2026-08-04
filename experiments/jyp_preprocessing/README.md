# JYP 전처리 실험 기록

`test_003`의 JYP 전처리 단계별 결과와 파라미터 탐색 기록을 모아 둔 폴더입니다.

- `test_003_raw.md`~`test_003_f5_*.md`: 단계별 ablation 결과
- `test_003_preprocessing_optimizer.csv/.md`: 전처리 파라미터 탐색 결과
- `test_003_f5_no_raw_missmask.md`: F5 no-raw missmask 결과
- `test_003_f5_selective_no_raw_summary.md`: F5 selective 전체·N5·N10 비교와 N10 채택 근거
- `test_003_jyp_f8.md`: F5 selective N10+F7 결합 실행 결과
- `test_003_f4_no_raw_hot5.md`: paired 5-Fold 검증 후 채택한 F4 최종 결과
- `test_003_f6_no_raw.md`: F6 TF-IDF·SVD 단일 실행 결과
- `test_003_f7_paircontrast_no_raw.md`: F7 pair-contrast 단일 실행 결과
- `test_003_f7_paircontrast_no_raw_kidney_glioma_k3_a4.md`: F7 다중 Fold 검증, 승격 근거와 진단
- `test_003_pipeComb_v3.md`: `pipeComb_v3` 내부 피처 결합, 로컬 검증과 리더보드 결과
- `test_003_pipeComb_v4.md`: `pipeComb_v4` weighted W52 승격 근거, GPU 반복 검증과 한계
- `test_003_전체결과_인사이트.md`: 전체 단계 비교, submission 변화율, 채택 판단과 종합 인사이트

각 결과 Markdown의 `적용 전처리 파라미터`에는 실제 실행에서 확인한 값만 기록합니다.
JYP 단계 전처리는 `src/pipelines/jyp_preprocessing/`의 각 `pipeline_jyp_*.py`
파일에 독립적으로 구현되어 있으며, `configs/test_003.yaml`의
`preprocessing.name`으로 선택합니다. 별도 실행기는 사용하지 않습니다.
과거 F0~F11 단계별 결과 파일명과 실행 ID는 재현성을 위해 유지하고, 현재 등록
이름은 `jyp_raw`, `jyp_f0`~`jyp_f11` 형식으로 표기합니다. 확정 결합 문서는
공식 파이프라인 이름으로 통일합니다. 보존하는 확정 조합은
`pipeComb_v3`(dual D104)와 `pipeComb_v4`(weighted W52)뿐이며, 각각
`pipeline_pipe_comb_v3.py`, `pipeline_pipe_comb_v4.py`에서 F9와 EM24를
직접 조립합니다.

현재 실행용 `test_003` 계열 YAML의 JYP 매핑에는 등록 이름만 기록합니다.
채택된 support, top-K, 안정성 Fold, smoothing, random seed 등은 각 JYP
파이프라인 Python 파일의 생성자 기본값으로 고정합니다. 과거 결과 문서의
파라미터 표와 다른 팀원 companion의 비교 조건은 재현 근거이므로 삭제하지
않습니다.

현재 결과의 핵심 결론은 RAW exact-cell 인코딩 제외, F1 변이 부담과 F2 변이 유형 채택, F4 no-raw hot5를 기준으로 한 F7 pair-contrast 승격입니다. F5는 단일 holdout 최고점이지만 독립 다중 seed 근거가 부족하고, F6는 성능이 하락했습니다.

## 전처리 승격 규칙

- 기준안과 후보를 같은 분할·seed·Fold에서 paired 비교합니다.
- 후보의 안정성이 기준안보다 명백히 나빠지면 점수가 올라도 `보류`합니다.
- 안정성이 기준안과 비슷하고 평균 Macro F1 변화가 `0보다 크면` 개선 폭이 작아도 `승격`합니다.
- `+0.003`처럼 별도의 최소 점수 상승 폭은 두지 않습니다.
- bootstrap CI, Fold 승률, seed별 방향, 클래스별 급락, PSI는 안정성을 판단하는 진단값이며, 한 지표만으로 자동 탈락시키지 않습니다.

현재 F8은 N=10보다 단일 holdout Macro F1이 `+0.000105` 높으므로 미채택이 아니라 `안정성 비교 전 보류` 상태입니다. 동일 조건의 반복 검증에서 안정성이 비슷하면 승격합니다.

F7 전체 검증은 `2026-08-02 04:52:51 KST`에 완료됐으며 선택 후보는 `f7_both_k3`, 검증기 판정은 `adopt_f7`입니다. 상세 수치와 통과·실패 진단은 `test_003_f7_paircontrast_no_raw_kidney_glioma_k3_a4.md`에 고정해 두었습니다.

F7 both K3 제출 리더보드 점수는 `0.3034169097`이며 F4 제출 `0.2878895927`보다 `+0.0155273170` 높습니다.

`pipeComb_v3`는 F9에 EM24 dual signature 104개를 내부 결합한 보존 기준이며,
해당 제출의 Public Leaderboard Macro F1은 `0.3390097657`입니다. 현재 최종
파이프라인은 match-count를 제외하고 weighted 52개만 남긴 `pipeComb_v4`입니다.
두 버전의 전용 기록은 각각 `test_003_pipeComb_v3.md`,
`test_003_pipeComb_v4.md`에서 확인합니다.

최종 v4 실행 설정은 `configs/test_003.yaml`, v3 보존 설정은
`configs/test_003_pipecomb_v3.yaml`입니다. 이 폴더에는 팀 공유용으로 확정한 단계별 결과와
인사이트만 버전 관리합니다. 학습 산출물과 자동 생성 보고서는 공용 `src.train`의 기본 규칙을
따르며 Git에 포함하지 않습니다.

```bash
python -m src.train --config configs/test_003.yaml
```
