"""TEST_004_8: JSJ VoteSelect v1 historical public submission."""

SPEC = {
    "name": "TEST_004_8",
    "pipeline_name": "jsj_v10",
    "pipeline_file": "pipeline_jsj_v10.py",
    "selector_version": "v1",
    "full_meta_version": "v6",
    "selector": {"min_count": 3, "shrinkage": 1.0},
    "historical_oof_macro_f1": 0.6401862208553689,
    "historical_public_macro_f1": 0.4660597084,
    "historical_submission_sha256": "a76dc0f35ce94f48ec3a42674e003df4d6627aba10b432f9930be609831c5157",
}
