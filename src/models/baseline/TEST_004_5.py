"""TEST_004_5: 충돌 쌍별 C와 임계 오프셋을 조정한 최종 제출 모델."""

from src.models.baseline.TEST_004_1 import SPEC as BASE_SPEC

SPEC = {
    **BASE_SPEC,
    "name": "TEST_004_5",
    "postprocessing": {
        "mode": "soft",
        "KIRC_KIPAN": 0.10,
        "LGG_GBMLGG": 1.0,
        "specialist_c": {
            "KIRC_KIPAN": 0.20,
            "LGG_GBMLGG": 0.10,
        },
        "right_offset": {
            "KIRC_KIPAN": 0.0,
            "LGG_GBMLGG": 0.10,
        },
    },
    "historical_oof_macro_f1": 0.5149647043026445,
    "historical_public_macro_f1": 0.3863356794,
}
