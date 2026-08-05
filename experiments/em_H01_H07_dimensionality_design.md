# EM H01~H07 차원 축소 실험 설계

모든 실험은 `EMV46`을 공통 베이스로 한 번 생성한 뒤 그 수치 피처에
차원 축소를 적용한다. EMV46 자체와 이후 선택·분해 모델은 모두 학습 fold에서
fit한다. 기준 실험 H01의 활성 빈도 후보
`1,2,3,5,10,20`은 학습 fold 내부 3-fold Macro F1으로만 선택한다.
H02~H07은 비교 변수를 한 가지로 제한하기 위해 최소 빈도를 5로 고정한다.

| 파이프라인 | 출력 | 기본 차원 | 지도 정보 |
| --- | --- | ---: | --- |
| em_H01 | 최소 빈도 통과 EMV46 피처 | 데이터 의존 | 없음 |
| em_H02 | Chi-square 상위 EMV46 피처 | 1,500 | fold-train 레이블 |
| em_H03 | Mutual information 상위 EMV46 피처 | 1,500 | fold-train 레이블 |
| em_H04 | 암종별 enrichment 상위 유전자 합집합 | 클래스당 20 | fold-train 레이블 |
| em_H05 | MiniBatch NMF 변이 모듈 | 128 | 없음 |
| em_H06 | Truncated SVD 잠재 피처 | 256 | 없음 |
| em_H07 | 원본 유전자 + burden + recurrent hotspot | 데이터 의존 | 없음 |

H02/H03은 `1,000/1,500/2,000`, H05는 `64/128/256`, H06은
`128/256/512`를 별도 설정으로 비교한다. 차원 후보 자체를 같은 validation
fold에서 반복 선택하지 않고 outer 학습 fold 안쪽에서만 결정해야 한다.

Chi-square와 NMF에는 비음수 조건을 만족하도록 EMV46 입력을 0 아래에서
clip한다. MI와 SVD는 EMV46의 원래 수치 범위를 유지한다.

H04는 EMV46 출력 중 원본 유전자 이름과 일치하는 피처를 대상으로 각 암종 대
나머지 암종의 smoothed log-odds에 빈도 신뢰도 shrinkage를 곱하고 상위
유전자의 합집합을 만든다. GBMLGG, KIPAN, STES는 분해하지 않는다.

H07은 EMV46에서 유지된 원본 유전자 피처를 사용한다. burden은 전체 원본 입력
유전자로 계산하고, hotspot 목록은 학습 fold에서
최소 5개 표본에 반복된 유전자-변이 토큰만 선택한다. validation/test는 학습
fold에서 확정된 유전자·hotspot·분해 모델로 transform만 수행한다.

외부 데이터는 사용하지 않으며 현재 학습, 평가, submission 생성은 수행하지 않는다.
