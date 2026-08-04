import inspect

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import src.pipelines.jyp_preprocessing.pipeline_pipe_comb_v4 as module
from src.pipelines.base import PreprocessingPipeline
from src.pipelines.jyp_preprocessing.pipeline_pipe_comb_v4 import (
    PipeCombV4PreprocessingPipeline,
    build_exact_profile_groups,
    build_em24_signature_columns,
    build_em24_weighted_signature_columns,
)
from src.pipelines.preprocessing_registry import create_preprocessing_pipeline


CLASSES = np.asarray([f"C{index:02d}" for index in range(26)], dtype=object)
F9_NAMES = np.asarray(
    ["F0__gene__G1", "F7__pair_contrast__C00__C01", "F9__global_aa_pair__A_to_V"],
    dtype=object,
)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {"marker": np.arange(26, dtype=np.float32)},
        index=[f"p{index:02d}" for index in range(26)],
    )


def test_exact_profile_groups_are_normalized_deterministic_and_order_independent() -> None:
    features = pd.DataFrame(
        {
            "G1": ["WT", "A1V", " a1v ", "WT", "WT", None, "NA"],
            "G2": ["WT", "WT", "WT", "A2X", "A2*", "WT", "WT"],
        }
    )

    first = build_exact_profile_groups(features)
    second = build_exact_profile_groups(features.loc[:, ["G2", "G1"]])

    np.testing.assert_array_equal(first, second)
    assert first[0] == first[5]
    assert first[1] == first[2]
    assert first[3] != first[4]
    assert first[0] != first[6]
    assert len(set(first)) == 5


class _FakeEncoder:
    def __init__(self) -> None:
        self.classes_ = CLASSES.copy()

    def transform(self, labels: object) -> np.ndarray:
        lookup = {label: index for index, label in enumerate(self.classes_)}
        return np.asarray([lookup[label] for label in labels], dtype=int)

    def inverse_transform(self, labels: object) -> np.ndarray:
        return self.classes_[np.asarray(labels, dtype=int)]


class _FakeF9:
    def __init__(self, **parameters: object) -> None:
        self.parameters = parameters
        self.label_encoder = _FakeEncoder()
        self.fit_calls = 0
        self.fit_transform_calls = 0
        self.groups = None
        self.last_fit_matrix = None

    @staticmethod
    def _matrix(features: pd.DataFrame) -> sparse.csr_matrix:
        marker = features["marker"].to_numpy(dtype=np.float32)[:, None]
        return sparse.coo_matrix(marker * 100.0 + np.arange(3, dtype=np.float32))

    def fit(self, features: pd.DataFrame, labels: object, *, groups: object = None):
        self.fit_calls += 1
        self.groups = groups
        return self

    def fit_transform(
        self, features: pd.DataFrame, labels: object, *, groups: object = None
    ) -> sparse.csr_matrix:
        self.fit_transform_calls += 1
        self.groups = groups
        self.last_fit_matrix = self._matrix(features)
        return self.last_fit_matrix

    def transform(self, features: pd.DataFrame) -> sparse.csr_matrix:
        return self._matrix(features)

    def get_feature_names_out(self) -> np.ndarray:
        return F9_NAMES.copy()

    def summary(self) -> dict[str, object]:
        return {
            "remaining_features": 3,
            "f7_oof_group_safe": True,
            "f7_oof_group_count": 13,
            "f7_oof_fold_count": 5,
        }

    def get_diagnostics(self) -> dict[str, object]:
        return {"f7_oof_fold_audits": [{"group_overlap_free": True}]}


class _FakeEM24:
    def __init__(self, **parameters: object) -> None:
        self.parameters = parameters
        self.label_encoder = _FakeEncoder()
        self.fit_calls = 0
        self.fit_transform_calls = 0
        self.groups = None
        self.reverse_transform_columns = False

    @staticmethod
    def _frame(features: pd.DataFrame) -> pd.DataFrame:
        output: dict[str, np.ndarray] = {
            "G1": np.full(len(features), 999.0, dtype=np.float32),
            "mutation_count": np.full(len(features), 998.0, dtype=np.float32),
        }
        marker = features["marker"].to_numpy(dtype=np.float32)
        for class_index, class_name in enumerate(CLASSES):
            output[f"signature_all_{class_name}_weighted"] = marker + class_index
            output[f"signature_all_{class_name}_match_count"] = (
                marker + 100 + class_index
            )
        output["hotspot_G1_x"] = np.full(len(features), 997.0, dtype=np.float32)
        for class_index, class_name in enumerate(CLASSES):
            output[f"signature_functional_{class_name}_weighted"] = (
                marker + 200 + class_index
            )
            output[f"signature_functional_{class_name}_match_count"] = (
                marker + 300 + class_index
            )
        return pd.DataFrame(output, index=features.index, dtype=np.float32)

    def fit(
        self, features: pd.DataFrame, labels: object, *, groups: object = None
    ):
        self.fit_calls += 1
        self.groups = groups
        return self

    def fit_transform(
        self, features: pd.DataFrame, labels: object, *, groups: object = None
    ) -> pd.DataFrame:
        self.fit_transform_calls += 1
        self.groups = groups
        return self._frame(features)

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        frame = self._frame(features)
        if self.reverse_transform_columns:
            frame = frame.loc[:, frame.columns[::-1]]
        return frame

    def summary(self) -> dict[str, object]:
        return {
            "remaining_features": 107,
            "signature_oof_group_safe": True,
            "signature_oof_group_count": 13,
            "signature_oof_fold_count": 5,
        }

    def get_diagnostics(self) -> dict[str, object]:
        return {
            "signature_oof_group_safe": True,
            "signature_oof_fold_audits": [{"group_overlap_free": True}],
        }


def _patch_components(monkeypatch) -> None:
    monkeypatch.setattr(module, "F9GlobalAAPairNoRawPreprocessingPipeline", _FakeF9)
    monkeypatch.setattr(module, "EMV24PreprocessingPipeline", _FakeEM24)


def test_direct_inheritance_name_registry_and_keyword_only_groups() -> None:
    assert PipeCombV4PreprocessingPipeline.__bases__ == (PreprocessingPipeline,)
    assert PipeCombV4PreprocessingPipeline.name == "pipeComb_v4"
    assert isinstance(
        create_preprocessing_pipeline({"name": "pipeComb_v4"}),
        PipeCombV4PreprocessingPipeline,
    )
    for method_name in ("fit", "fit_transform"):
        parameter = inspect.signature(
            getattr(PipeCombV4PreprocessingPipeline, method_name)
        ).parameters["groups"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_weighted_schema_is_exact_channel_major_w52() -> None:
    full_columns = build_em24_signature_columns(CLASSES)
    weighted_columns = build_em24_weighted_signature_columns(CLASSES)

    assert len(full_columns) == 104
    assert len(weighted_columns) == 52
    assert weighted_columns[:2] == (
        "signature_all_C00_weighted",
        "signature_all_C01_weighted",
    )
    assert weighted_columns[25:28] == (
        "signature_all_C25_weighted",
        "signature_functional_C00_weighted",
        "signature_functional_C01_weighted",
    )
    assert weighted_columns[-1] == "signature_functional_C25_weighted"


@pytest.mark.parametrize("class_count", [25, 27])
def test_weighted_schema_rejects_non_26_classes(class_count: int) -> None:
    classes = [f"C{index:02d}" for index in range(class_count)]
    with pytest.raises(ValueError, match="requires exactly 26 fitted classes"):
        build_em24_weighted_signature_columns(classes)


def test_fit_transform_fits_components_once_and_emits_only_weighted_w52(
    monkeypatch,
) -> None:
    _patch_components(monkeypatch)
    features = _features()
    labels = pd.Series(CLASSES, index=features.index)
    groups = np.asarray([f"g{index // 2}" for index in range(26)], dtype=object)

    pipeline = PipeCombV4PreprocessingPipeline()
    output = pipeline.fit_transform(features, labels, groups=groups)
    em24_frame = pipeline.em24_pipeline_._frame(features)
    expected_weighted = em24_frame.loc[
        :, list(build_em24_weighted_signature_columns(CLASSES))
    ]

    assert pipeline.f9_pipeline_.fit_calls == 0
    assert pipeline.f9_pipeline_.fit_transform_calls == 1
    assert pipeline.em24_pipeline_.fit_calls == 0
    assert pipeline.em24_pipeline_.fit_transform_calls == 1
    assert pipeline.f9_pipeline_.groups is groups
    assert pipeline.em24_pipeline_.groups is groups
    assert pipeline.profile_group_source_ == "provided"
    assert sparse.isspmatrix_csr(output)
    assert output.dtype == np.float32
    assert output.has_canonical_format
    assert np.isfinite(output.data).all()
    np.testing.assert_array_equal(
        output[:, : len(F9_NAMES)].toarray(),
        pipeline.f9_pipeline_.last_fit_matrix.toarray(),
    )
    np.testing.assert_array_equal(
        output[:, len(F9_NAMES) :].toarray(), expected_weighted.to_numpy()
    )

    names = pipeline.get_feature_names_out().tolist()
    em24_names = names[len(F9_NAMES) :]
    assert em24_names == [
        f"EM24__{name}" for name in build_em24_weighted_signature_columns(CLASSES)
    ]
    assert not any("match_count" in name for name in names)
    assert not any("mutation_count" in name or "hotspot" in name for name in names)


def test_fit_transform_derives_and_propagates_profile_groups_when_omitted(
    monkeypatch,
) -> None:
    _patch_components(monkeypatch)
    features = _features()
    features.loc[features.index[1], "marker"] = features.loc[
        features.index[0], "marker"
    ]
    labels = pd.Series(CLASSES, index=features.index)

    pipeline = PipeCombV4PreprocessingPipeline()
    pipeline.fit_transform(features, labels)
    derived_groups = pipeline.f9_pipeline_.groups

    assert derived_groups is pipeline.em24_pipeline_.groups
    assert pipeline.profile_group_source_ == "derived"
    assert len(derived_groups) == len(features)
    assert derived_groups[0] == derived_groups[1]
    assert len(set(derived_groups)) == len(features) - 1


def test_auto_profile_groups_can_be_disabled_without_affecting_other_pipelines(
    monkeypatch,
) -> None:
    _patch_components(monkeypatch)
    features = _features()
    labels = pd.Series(CLASSES, index=features.index)

    pipeline = PipeCombV4PreprocessingPipeline(auto_profile_groups=False)
    pipeline.fit_transform(features, labels)

    assert pipeline.f9_pipeline_.groups is None
    assert pipeline.em24_pipeline_.groups is None
    assert pipeline.profile_group_source_ == "disabled"


def test_provided_group_series_must_match_feature_index_exactly(monkeypatch) -> None:
    _patch_components(monkeypatch)
    features = _features()
    labels = pd.Series(CLASSES, index=features.index)
    aligned = pd.Series(
        [f"g{index // 2}" for index in range(len(features))],
        index=features.index,
    )

    pipeline = PipeCombV4PreprocessingPipeline()
    pipeline.fit_transform(features, labels, groups=aligned)
    assert pipeline.f9_pipeline_.groups is aligned
    assert pipeline.em24_pipeline_.groups is aligned

    with pytest.raises(ValueError, match="index and order must exactly match"):
        pipeline.fit_transform(features, labels, groups=aligned.iloc[::-1])


def test_repeated_fit_resets_components_and_group_source(monkeypatch) -> None:
    _patch_components(monkeypatch)
    features = _features()
    labels = pd.Series(CLASSES, index=features.index)
    groups = np.asarray([f"g{index // 2}" for index in range(26)], dtype=object)
    pipeline = PipeCombV4PreprocessingPipeline()

    pipeline.fit_transform(features, labels, groups=groups)
    first_f9 = pipeline.f9_pipeline_
    first_em24 = pipeline.em24_pipeline_
    assert pipeline.profile_group_source_ == "provided"

    pipeline.fit_transform(features, labels)
    assert pipeline.f9_pipeline_ is not first_f9
    assert pipeline.em24_pipeline_ is not first_em24
    assert pipeline.profile_group_source_ == "derived"


def _wrong_class(columns: list[str]) -> list[str]:
    index = columns.index("signature_all_C00_weighted")
    columns[index] = "signature_all_WRONG_weighted"
    return columns


def _wrong_suffix(columns: list[str]) -> list[str]:
    index = columns.index("signature_functional_C25_match_count")
    columns[index] = "signature_functional_C25_score"
    return columns


def _reordered(columns: list[str]) -> list[str]:
    first = columns.index("signature_all_C00_weighted")
    second = columns.index("signature_all_C00_match_count")
    columns[first], columns[second] = columns[second], columns[first]
    return columns


@pytest.mark.parametrize("corrupt_columns", [_wrong_class, _wrong_suffix, _reordered])
def test_rejects_noncanonical_full_em24_schema(monkeypatch, corrupt_columns) -> None:
    _patch_components(monkeypatch)
    original_frame = _FakeEM24._frame

    def corrupted_frame(features: pd.DataFrame) -> pd.DataFrame:
        frame = original_frame(features)
        return frame.set_axis(corrupt_columns(frame.columns.tolist()), axis="columns")

    monkeypatch.setattr(_FakeEM24, "_frame", staticmethod(corrupted_frame))
    features = _features()
    labels = pd.Series(CLASSES, index=features.index)

    with pytest.raises(RuntimeError, match="canonical fitted-class schema or order"):
        PipeCombV4PreprocessingPipeline().fit_transform(features, labels)


@pytest.mark.parametrize("class_count", [25, 27])
def test_pipeline_rejects_non_26_fitted_classes(monkeypatch, class_count: int) -> None:
    _patch_components(monkeypatch)
    pipeline = PipeCombV4PreprocessingPipeline()
    classes = np.asarray([f"C{index:02d}" for index in range(class_count)])
    pipeline.f9_pipeline_.label_encoder.classes_ = classes
    pipeline.em24_pipeline_.label_encoder.classes_ = classes.copy()

    with pytest.raises(ValueError, match="requires exactly 26 fitted classes"):
        pipeline._finalize_schema(_FakeEM24._frame(_features()))


def test_fit_then_transform_preserves_fitted_schema_and_row_order(monkeypatch) -> None:
    _patch_components(monkeypatch)
    features = _features()
    labels = pd.Series(CLASSES, index=features.index)
    groups = np.asarray([f"g{index // 2}" for index in range(26)], dtype=object)
    pipeline = PipeCombV4PreprocessingPipeline().fit(features, labels, groups=groups)

    assert pipeline.f9_pipeline_.fit_calls == 1
    assert pipeline.em24_pipeline_.fit_calls == 1
    assert pipeline.f9_pipeline_.groups is groups
    assert pipeline.em24_pipeline_.groups is groups

    pipeline.em24_pipeline_.reverse_transform_columns = True
    reordered = features.iloc[::-1]
    output = pipeline.transform(reordered)
    expected_weighted = pipeline.em24_pipeline_._frame(reordered).loc[
        :, list(pipeline.em24_weighted_signature_columns_)
    ]
    np.testing.assert_array_equal(
        output[:, : len(F9_NAMES)].toarray(),
        pipeline.f9_pipeline_.transform(reordered).toarray(),
    )
    np.testing.assert_array_equal(
        output[:, len(F9_NAMES) :].toarray(), expected_weighted.to_numpy()
    )


def test_summary_diagnostics_and_label_api(monkeypatch) -> None:
    _patch_components(monkeypatch)
    features = _features()
    pipeline = PipeCombV4PreprocessingPipeline()
    pipeline.fit_transform(features, pd.Series(CLASSES, index=features.index))
    summary = pipeline.summary()
    diagnostics = pipeline.get_diagnostics()

    assert summary["f9_prefix_features"] == len(F9_NAMES)
    assert summary["em24_weighted_signature_features"] == 52
    assert summary["em24_signature_all_weighted_features"] == 26
    assert summary["em24_signature_functional_weighted_features"] == 26
    assert summary["em24_match_count_features"] == 0
    assert summary["class_count"] == 26
    assert summary["auto_profile_groups"] is True
    assert summary["profile_group_source"] == "derived"
    assert summary["oof_group_safe"] is True
    assert diagnostics["f9_oof_fold_audits"][0]["group_overlap_free"] is True
    assert diagnostics["em24_oof_fold_audits"][0]["group_overlap_free"] is True
    assert len(diagnostics["em24_weighted_signature_columns"]) == 52
    assert len(diagnostics["output_column_indices"]) == len(F9_NAMES) + 52
    np.testing.assert_array_equal(pipeline.encode_labels(CLASSES), np.arange(26))
    np.testing.assert_array_equal(pipeline.decode_labels([0, 25]), ["C00", "C25"])
