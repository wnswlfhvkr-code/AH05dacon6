"""TEST_004_2: 두 충돌 쌍의 확률을 전면 재분배하는 모델."""

from src.models.baseline.TEST_004_1 import SPEC as BASE_SPEC

SPEC = {
    **BASE_SPEC,
    "name": "TEST_004_2",
    "postprocessing": {
        "mode": "soft",
        "KIRC_KIPAN": 1.0,
        "LGG_GBMLGG": 1.0,
    },
    "historical_oof_macro_f1": 0.5072405912,
    "historical_public_macro_f1": 0.3741003172,
}

