"""TEST_004_3: 충돌 쌍 내부에서만 엄격하게 재판정하는 모델."""

from src.models.baseline.TEST_004_1 import SPEC as BASE_SPEC

SPEC = {
    **BASE_SPEC,
    "name": "TEST_004_3",
    "postprocessing": {
        "mode": "strict",
        "KIRC_KIPAN": 1.0,
        "LGG_GBMLGG": 1.0,
    },
    "historical_oof_macro_f1": 0.5081343300,
    "historical_public_macro_f1": 0.3756133965,
}

