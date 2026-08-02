"""TEST_004_6: EM v14를 KIRC-KIPAN 전용 전문가로 결합한 제출 모델."""

from src.models.baseline.TEST_004_1 import SPEC as BASE_SPEC

SPEC = {
    **BASE_SPEC,
    "name": "TEST_004_6",
    "postprocessing": {
        "mode": "soft",
        "KIRC_KIPAN": 0.10,
        "LGG_GBMLGG": 1.0,
        "specialist_c": {
            "KIRC_KIPAN": 0.15,
            "LGG_GBMLGG": 0.06,
        },
        "right_offset": {
            "KIRC_KIPAN": -0.05,
            "LGG_GBMLGG": 0.06,
        },
        "auxiliary": {
            "name": "em_v14_kirc_specialist",
            "pair": "KIRC_KIPAN",
            "weight": 0.30,
            "temperature": 0.50,
            "right_offset": 0.30,
        },
    },
    "historical_oof_macro_f1": 0.5188229895662899,
    "historical_public_macro_f1": 0.3893423841,
}
