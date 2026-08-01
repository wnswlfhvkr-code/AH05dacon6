"""TEST_004_4: LGG 계열은 전면, KIRC 계열은 10%만 보정하는 모델."""

from src.models.baseline.TEST_004_1 import SPEC as BASE_SPEC

SPEC = {
    **BASE_SPEC,
    "name": "TEST_004_4",
    "postprocessing": {
        "mode": "soft",
        "KIRC_KIPAN": 0.10,
        "LGG_GBMLGG": 1.0,
    },
    "historical_oof_macro_f1": 0.5084001917,
    "historical_public_macro_f1": 0.3768261636,
}
