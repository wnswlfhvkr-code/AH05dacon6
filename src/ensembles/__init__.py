"""OOF·Test 출력을 결합하는 앙상블 실행기 패키지.

독립적인 피처 생성과 모델 학습은 ``src.pipelines``와
공통 학습기에서 담당한다. 이 패키지는 이미 생성된
OOF 예측과 Test 예측을 cross-fit 방식으로 결합하는
실험만 관리한다.
"""

__all__ = [
    "train_jh_e8a",
    "train_jh_e8b",
    "train_jh_e8c",
]
