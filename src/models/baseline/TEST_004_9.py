"""TEST_004_9: JSJ VoteSelect v2 historical public submission."""

SPEC = {
    "name": "TEST_004_9",
    "pipeline_name": "jsj_v11",
    "pipeline_file": "pipeline_jsj_v11.py",
    "selector_version": "v2",
    "full_meta_version": "v7",
    "selector": {"min_count": 3, "shrinkage": 0.5},
    "historical_oof_macro_f1": 0.6487595723798856,
    "historical_public_macro_f1": 0.4535576457,
    "historical_submission_sha256": "245e2dc030f066bef1fcc6286d8ca1338aecb3fd40b0682b56483e62a3cd00dc",
}
