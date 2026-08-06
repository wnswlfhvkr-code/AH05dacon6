from src.models.baseline.TEST_004_8 import SPEC as SPEC_8
from src.models.baseline.TEST_004_9 import SPEC as SPEC_9
from src.pipelines.preprocessing_registry import create_preprocessing_pipeline


def test_test_004_8_linkage():
    pipeline = create_preprocessing_pipeline({"name": "jsj_v10"})
    assert pipeline.name == SPEC_8["pipeline_name"]
    assert pipeline.selector_version == SPEC_8["selector_version"]
    assert pipeline.selector_min_count == SPEC_8["selector"]["min_count"]
    assert pipeline.selector_shrinkage == SPEC_8["selector"]["shrinkage"]


def test_test_004_9_linkage():
    pipeline = create_preprocessing_pipeline({"name": "jsj_v11"})
    assert pipeline.name == SPEC_9["pipeline_name"]
    assert pipeline.selector_version == SPEC_9["selector_version"]
    assert pipeline.selector_min_count == SPEC_9["selector"]["min_count"]
    assert pipeline.selector_shrinkage == SPEC_9["selector"]["shrinkage"]
