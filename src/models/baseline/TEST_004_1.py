"""TEST_004_1: Word+Char TF-IDF 중심 기준 앙상블."""

SPEC = {
    "name": "TEST_004_1",
    "linear_weight": 0.95,
    "base_tree_weight": 0.025,
    "e1_tree_weight": 0.025,
    "temperature": 0.5,
    "linear_c": 0.2,
    "class_multipliers": [
        0.65, 1.14, 1.14, 1.00, 0.94, 1.00, 1.24, 1.00, 1.00,
        1.14, 0.80, 1.36, 1.06, 1.06, 0.88, 1.06, 1.14, 1.14,
        1.14, 1.24, 0.65, 1.00, 0.88, 0.65, 0.88, 0.88,
    ],
    "postprocessing": {
        "mode": "none",
        "KIRC_KIPAN": 0.0,
        "LGG_GBMLGG": 0.0,
    },
    "historical_oof_macro_f1": 0.5055950606,
    "historical_public_macro_f1": 0.3714981583,
}

