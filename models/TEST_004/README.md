# TEST_004_8 / TEST_004_9 probability bundle

`vote_select_inputs.npz`는 두 VoteSelect 제출물을 정확히 재현하는 데 필요한
최소 OOF·테스트 확률만 담은 압축 모델 아티팩트입니다.

- 원천: 대회에서 제공한 `train.csv`, `test.csv`만으로 생성한 팀 모델의 확률
- 외부 데이터: 사용하지 않음
- 테스트 정답: 사용하지 않음
- 역할: 학습된 기반 모델들의 예측을 다시 학습하는 대신 동일한 후처리 선택기를 재현
- 무결성: `vote_select_inputs.json`의 SHA-256과 원본 데이터 ID 해시를 실행 시 검증

원본 중간 캐시가 모두 있는 개발 환경에서는 다음 명령으로 번들을 다시 만들 수 있습니다.

```bash
python -m src.build_test_004_vote_bundle
```

GitHub에서는 `.npz` 파일을 Git LFS로 관리합니다.
