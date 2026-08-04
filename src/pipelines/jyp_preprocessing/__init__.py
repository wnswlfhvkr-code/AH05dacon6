"""파일별로 완결된 JYP 전처리 파이프라인 패키지입니다.

각 ``pipeline_jyp_*.py``는 ``PreprocessingPipeline``만 직접 상속하며,
해당 버전에서 사용하는 변이 파싱과 피처 생성 로직을 파일 안에 보관합니다.
``pipeline_pipe_comb_v3.py``와 ``pipeline_pipe_comb_v4.py``는 보존하는
F9·EM24 결합 파이프라인이며 역시 ``PreprocessingPipeline``만 직접 상속합니다.
실행용 YAML은 등록 이름만 선택하고, 채택된 전처리 파라미터는 각 클래스의
생성자 기본값을 단일 기준으로 사용합니다.
"""
