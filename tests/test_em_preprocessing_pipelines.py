import pandas as pd
import pytest

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.pipeline_em_v1 import EMV1PreprocessingPipeline
from src.pipelines.pipeline_em_v2 import EMV2PreprocessingPipeline
from src.pipelines.pipeline_em_v3 import EMV3PreprocessingPipeline
from src.pipelines.pipeline_em_v4 import EMV4PreprocessingPipeline
from src.pipelines.pipeline_em_v5 import EMV5PreprocessingPipeline
from src.pipelines.pipeline_em_v6 import EMV6PreprocessingPipeline
from src.pipelines.pipeline_em_v7 import EMV7PreprocessingPipeline
from src.pipelines.pipeline_em_v8 import EMV8PreprocessingPipeline
from src.pipelines.pipeline_em_v9 import EMV9PreprocessingPipeline
from src.pipelines.pipeline_em_v10 import EMV10PreprocessingPipeline
from src.pipelines.pipeline_em_v11 import EMV11PreprocessingPipeline
from src.pipelines.pipeline_em_v12 import EMV12PreprocessingPipeline
from src.pipelines.pipeline_em_v13 import EMV13PreprocessingPipeline
from src.pipelines.pipeline_em_v14 import EMV14PreprocessingPipeline
from src.pipelines.pipeline_em_v15 import EMV15PreprocessingPipeline
from src.pipelines.pipeline_em_v16 import EMV16PreprocessingPipeline
from src.pipelines.pipeline_em_v17 import EMV17PreprocessingPipeline
from src.pipelines.pipeline_em_v18 import (
    CONSEQUENCE_FRAMESHIFT,
    CONSEQUENCE_INFRAME,
    CONSEQUENCE_MISSENSE,
    CONSEQUENCE_NONSENSE,
    CONSEQUENCE_SYNONYMOUS,
    EMV18PreprocessingPipeline,
    classify_mutation_token,
)
from src.pipelines.pipeline_em_v19 import EMV19PreprocessingPipeline
from src.pipelines.pipeline_em_v20 import EMV20PreprocessingPipeline
from src.pipelines.pipeline_em_v21 import EMV21PreprocessingPipeline
from src.pipelines.pipeline_em_v22 import EMV22PreprocessingPipeline
from src.pipelines.pipeline_em_v23 import EMV23PreprocessingPipeline
from src.pipelines.pipeline_em_v24 import EMV24PreprocessingPipeline
from src.pipelines.pipeline_em_v25 import EMV25PreprocessingPipeline
from src.pipelines.pipeline_em_v26 import EMV26PreprocessingPipeline
from src.pipelines.pipeline_em_v27 import EMV27PreprocessingPipeline
from src.pipelines.pipeline_em_v28 import EMV28PreprocessingPipeline
from src.pipelines.preprocessing_registry import create_preprocessing_pipeline


def make_pathway_features() -> tuple[pd.DataFrame, pd.Series]:
    genes = (
        "CDKN2A", "NF2", "MYC", "NOTCH1", "NFE2L2", "PIK3CA", "EGFR",
        "TGFBR1", "TP53", "APC", "OTHER",
    )
    features = pd.DataFrame({
        gene: ["MUT" if (row + index) % (index % 3 + 2) == 0 else "WT" for row in range(18)]
        for index, gene in enumerate(genes)
    })
    labels = pd.Series(["A"] * 6 + ["B"] * 6 + ["C"] * 6)
    return features, labels


def test_pipeline_factory_creates_em_v1() -> None:
    pipeline = create_preprocessing_pipeline({"name": "em_v1"})

    assert isinstance(pipeline, EMV1PreprocessingPipeline)
    assert pipeline.name == "em_v1"
    assert pipeline.steps == (
        "유전자 변이 여부 이진 피처 추가",
        "상수 열 제거",
        "범주형 순서 인코딩",
    )


def test_em_v1_adds_binary_mutation_features_and_keeps_originals() -> None:
    features = pd.DataFrame({
        gene: ["WT", f"{gene}_variant"]
        for gene in EMV1PreprocessingPipeline.mutation_genes
    })
    features.loc[0, "APC"] = None
    labels = pd.Series(["A", "B"])
    pipeline = EMV1PreprocessingPipeline()

    transformed = pipeline.fit_transform(features, labels)

    for gene in pipeline.mutation_genes:
        mutation_feature = f"{gene}_mutated"
        assert gene in transformed.columns
        assert mutation_feature in transformed.columns
        assert transformed[mutation_feature].dtype.name == "int8"
        assert transformed.loc[0, mutation_feature] == 0
        assert transformed.loc[1, mutation_feature] == 1


def test_pipeline_factory_creates_em_v2() -> None:
    pipeline = create_preprocessing_pipeline({"name": "em_v2"})

    assert isinstance(pipeline, EMV2PreprocessingPipeline)
    assert pipeline.name == "em_v2"
    assert len(pipeline.mutation_genes) == 35
    assert len(set(pipeline.mutation_genes)) == 35


def test_em_v2_adds_all_binary_mutation_features() -> None:
    features = pd.DataFrame({
        gene: ["WT", f"{gene}_variant"]
        for gene in EMV2PreprocessingPipeline.mutation_genes
    })
    labels = pd.Series(["A", "B"])
    pipeline = EMV2PreprocessingPipeline()

    transformed = pipeline.fit_transform(features, labels)

    mutation_features = [f"{gene}_mutated" for gene in pipeline.mutation_genes]
    assert all(feature in transformed.columns for feature in mutation_features)
    assert all(gene not in transformed.columns for gene in pipeline.mutation_genes)
    assert transformed[mutation_features].dtypes.eq("int8").all()
    assert transformed.loc[0, mutation_features].sum() == 0
    assert transformed.loc[1, mutation_features].sum() == len(mutation_features)
    assert pipeline.summary()["dropped_original_mutation_features"] == len(
        pipeline.mutation_genes
    )
    assert transformed.shape[1] == pipeline.summary()["remaining_features"]


def test_em_v3_filters_rare_features_and_preserves_class_marker() -> None:
    features = pd.DataFrame({
        "class_marker": ["MUT", "MUT", "WT", "WT", "WT", "WT"],
        "rare_noise": ["WT", "WT", "WT", "MUT", "WT", "WT"],
        "frequent": ["MUT", "WT", "MUT", "WT", "MUT", "WT"],
        "always_wt": ["WT"] * 6,
    })
    labels = pd.Series(["A", "A", "A", "B", "B", "B"])
    pipeline = EMV3PreprocessingPipeline(
        min_mutation_count=3,
        min_class_mutation_count=2,
    )

    transformed = pipeline.fit_transform(features, labels)

    assert pipeline.dropped_rare_columns == ["rare_noise", "always_wt"]
    assert transformed.columns.tolist() == ["class_marker", "frequent"]
    assert pipeline.summary()["dropped_rare_features"] == 2


def test_em_v4_replaces_strings_with_binary_gene_features() -> None:
    features, labels = make_pathway_features()
    pipeline = EMV4PreprocessingPipeline(min_mutation_count=2)

    transformed = pipeline.fit_transform(features, labels)

    assert transformed.shape[1] == pipeline.summary()["remaining_features"]
    assert set(transformed.stack().unique()) <= {0.0, 1.0}
    assert transformed.dtypes.eq("float32").all()

def test_em_v5_adds_sample_mutation_burden() -> None:
    features, labels = make_pathway_features()
    pipeline = EMV5PreprocessingPipeline(min_mutation_count=2)

    transformed = pipeline.fit_transform(features, labels)
    expected_burden = features.ne("WT").sum(axis=1).astype("float32")

    assert transformed["mutation_burden_total"].equals(expected_burden)
    assert set(pipeline.burden_columns) <= set(transformed.columns)


def test_em_v6_encodes_protein_consequence_groups() -> None:
    features = pd.DataFrame({
        "GENE1": ["WT", "A1A", "A1V", "K2FS", "R3*", "483_484MP>IA"],
        "GENE2": ["WT", "B2C", "WT", "B2B", "WT", "B2C"],
    })
    labels = pd.Series(["A", "A", "B", "B", "C", "C"])
    pipeline = EMV6PreprocessingPipeline(min_mutation_count=1)

    transformed = pipeline.fit_transform(features, labels)

    assert transformed["GENE1"].tolist() == [0.0, 1.0, 2.0, 4.0, 4.0, 3.0]
    assert pipeline.summary()["consequence_summary_features"] == 9


def test_em_v7_adds_ten_pathway_groups() -> None:
    features, labels = make_pathway_features()
    pipeline = EMV7PreprocessingPipeline(min_mutation_count=2)

    transformed = pipeline.fit_transform(features, labels)
    pathway_columns = [
        column for column in transformed.columns if column.startswith("pathway_")
    ]

    assert len(pathway_columns) == 30
    assert pipeline.summary()["pathway_groups"] == 10


def test_em_v8_creates_nonnegative_mutation_modules() -> None:
    features, labels = make_pathway_features()
    pipeline = EMV8PreprocessingPipeline(
        min_mutation_count=2,
        n_components=3,
        max_iter=200,
        batch_size=6,
    )

    transformed = pipeline.fit_transform(features, labels)

    assert transformed.shape == (len(features), 4)
    assert transformed.ge(0).all().all()
    assert pipeline.summary()["mutation_modules"] == 3


def test_em_v9_creates_class_signature_scores_without_raw_genes() -> None:
    features, labels = make_pathway_features()
    pipeline = EMV9PreprocessingPipeline(
        min_mutation_count=2,
        top_genes_per_class=2,
    )

    transformed = pipeline.fit_transform(features, labels)

    assert transformed.shape[1] == 2 * labels.nunique() + 1
    assert all(column.startswith("signature_") or column == "mutation_burden_log1p"
               for column in transformed.columns)
    assert set(pipeline.class_gene_weights_) == set(labels.unique())


def test_em_v10_combines_burden_and_consequence_without_binary_duplicates() -> None:
    features, labels = make_pathway_features()
    pipeline = EMV10PreprocessingPipeline(min_mutation_count=2)

    transformed = pipeline.fit_transform(features, labels)

    assert transformed.columns.is_unique
    assert transformed.columns.tolist().count("mutation_burden_log1p") == 1
    assert "mutation_burden_total" in transformed
    assert "consequence_count_missense" in transformed
    assert set(transformed[pipeline.selected_gene_columns].stack().unique()) <= {0.0, 2.0}


def test_em_v11_combines_binary_burden_and_signatures_without_log_duplicate() -> None:
    features, labels = make_pathway_features()
    pipeline = EMV11PreprocessingPipeline(min_mutation_count=2, top_genes_per_class=2)

    transformed = pipeline.fit_transform(features, labels)

    assert transformed.columns.is_unique
    assert transformed.columns.tolist().count("mutation_burden_log1p") == 1
    assert len([column for column in transformed if column.startswith("signature_")]) == 6
    assert set(transformed[pipeline.selected_gene_columns].stack().unique()) <= {0.0, 1.0}


def test_em_v12_combines_consequence_and_signatures_without_burden_duplicate() -> None:
    features, labels = make_pathway_features()
    pipeline = EMV12PreprocessingPipeline(min_mutation_count=2, top_genes_per_class=2)

    transformed = pipeline.fit_transform(features, labels)

    assert transformed.columns.is_unique
    assert transformed.columns.tolist().count("mutation_burden_log1p") == 1
    assert "mutation_burden_total" not in transformed
    assert "consequence_count_missense" in transformed
    assert len([column for column in transformed if column.startswith("signature_")]) == 6


def test_em_v13_combines_all_features_without_duplicates() -> None:
    features, labels = make_pathway_features()
    pipeline = EMV13PreprocessingPipeline(min_mutation_count=2, top_genes_per_class=2)

    transformed = pipeline.fit_transform(features, labels)

    assert transformed.columns.is_unique
    assert transformed.columns.tolist().count("mutation_burden_log1p") == 1
    assert "mutation_burden_total" in transformed
    assert "mutation_burden_rate" in transformed
    assert "consequence_count_missense" in transformed
    assert len([column for column in transformed if column.startswith("signature_")]) == 6


def test_em_v14_adds_fold_learned_recurrent_hotspots() -> None:
    features = pd.DataFrame({
        "GENE1": ["A1V", "A1V", "WT", "A1V B2C", "WT", "B2C"] * 3,
        "GENE2": ["WT", "C3*", "C3*", "WT", "C3*", "WT"] * 3,
        "GENE3": ["WT", "WT", "D4D", "D4D", "WT", "WT"] * 3,
    })
    labels = pd.Series(["A"] * 6 + ["B"] * 6 + ["C"] * 6)
    pipeline = EMV14PreprocessingPipeline(
        min_mutation_count=1,
        top_genes_per_class=2,
        min_hotspot_count=3,
        max_hotspots=4,
    )

    transformed = pipeline.fit_transform(features, labels)
    hotspot_columns = [
        column for column in transformed if column.startswith("hotspot_")
    ]

    assert 0 < len(hotspot_columns) <= 4
    assert transformed[hotspot_columns].dtypes.eq("float32").all()
    assert set(transformed[hotspot_columns].stack().unique()) <= {0.0, 1.0}
    assert pipeline.summary()["hotspot_features"] == len(hotspot_columns)


NEW_PIPELINES = (
    EMV15PreprocessingPipeline,
    EMV16PreprocessingPipeline,
    EMV17PreprocessingPipeline,
    EMV18PreprocessingPipeline,
    EMV19PreprocessingPipeline,
    EMV20PreprocessingPipeline,
    EMV21PreprocessingPipeline,
    EMV22PreprocessingPipeline,
    EMV23PreprocessingPipeline,
    EMV24PreprocessingPipeline,
    EMV25PreprocessingPipeline,
    EMV26PreprocessingPipeline,
    EMV27PreprocessingPipeline,
    EMV28PreprocessingPipeline,
)


@pytest.mark.parametrize("pipeline_class", NEW_PIPELINES)
def test_new_em_pipelines_directly_inherit_common_base(pipeline_class) -> None:
    assert pipeline_class.__bases__ == (PreprocessingPipeline,)
    assert pipeline_class.evaluation_folds == 5


def test_em_v16_applies_oof_signature_features() -> None:
    features, labels = make_pathway_features()
    pipeline = EMV16PreprocessingPipeline(
        min_mutation_count=2,
        top_genes_per_class=2,
        min_hotspot_count=2,
        max_hotspots=4,
        inner_signature_folds=3,
    )

    transformed = pipeline.fit_transform(features, labels)

    rate_columns = [column for column in transformed if column.endswith("_match_rate")]
    count_columns = [column for column in transformed if column.startswith("consequence_count_")]
    assert not rate_columns
    assert count_columns
    assert transformed.dtypes.eq("float32").all()


def test_em_v17_splits_deduplicates_and_creates_gene_level_features() -> None:
    features = pd.DataFrame({
        "GENE1": ["WT", "A1V", "A1V A1V", "B2C", "A1V B2C", "C3*", "D4D"],
        "GENE2": ["WT", "WT", "K5FS", "WT", "K5FS", "WT", "WT"],
    })
    labels = pd.Series(["A", "A", "A", "B", "B", "C", "C"])
    pipeline = EMV17PreprocessingPipeline(
        min_gene_mutation_count=1,
        max_gene_features=2,
        min_hotspot_count=2,
        max_hotspots=4,
        correlation_threshold=0.99,
    )

    transformed = pipeline.fit_transform(features, labels)

    assert transformed["gene_GENE1_mutation_count"].tolist() == [0, 1, 1, 1, 2, 1, 1]
    assert transformed["gene_GENE1_multi_hit"].tolist() == [0, 0, 0, 0, 1, 0, 0]
    assert transformed["gene_GENE1_consequence_max"].tolist() == [0, 2, 2, 2, 2, 4, 1]
    hotspot_columns = [column for column in transformed if column.startswith("hotspot_")]
    assert hotspot_columns
    assert pipeline.summary()["hotspot_features_created"] == 3
    assert transformed.columns.is_unique


def test_em_v18_classifies_protein_consequences() -> None:
    assert classify_mutation_token("L42L") == CONSEQUENCE_SYNONYMOUS
    assert classify_mutation_token("A427V") == CONSEQUENCE_MISSENSE
    assert classify_mutation_token("R586*") == CONSEQUENCE_NONSENSE
    assert classify_mutation_token("Q456FS") == CONSEQUENCE_FRAMESHIFT
    assert classify_mutation_token("483_484MP>IA") == CONSEQUENCE_INFRAME


def test_em_v18_excludes_synonymous_variants_from_functional_signatures() -> None:
    features = pd.DataFrame({
        "GENE1": ["WT", "L42L", "A42V", "A42V L42L", "R10*", "Q11FS"] * 3,
        "GENE2": ["WT", "S7S", "WT", "K8N", "WT", "10_11AA>VV"] * 3,
    })
    labels = pd.Series(["A"] * 6 + ["B"] * 6 + ["C"] * 6)
    pipeline = EMV18PreprocessingPipeline(
        min_functional_mutation_count=1,
        top_genes_per_class=1,
        min_hotspot_count=2,
        max_hotspots=4,
        inner_signature_folds=2,
    )

    transformed = pipeline.fit_transform(features, labels)

    assert transformed.loc[1, "GENE1"] == CONSEQUENCE_SYNONYMOUS
    assert transformed.loc[2, "GENE1"] == CONSEQUENCE_MISSENSE
    assert transformed.loc[3, "GENE1"] == CONSEQUENCE_MISSENSE
    assert transformed.loc[4, "GENE1"] == CONSEQUENCE_NONSENSE
    assert transformed.loc[5, "GENE1"] == CONSEQUENCE_FRAMESHIFT
    assert transformed.loc[5, "GENE2"] == CONSEQUENCE_INFRAME
    assert all("L42L" not in name and "S7S" not in name for name in pipeline.hotspots_)
    assert transformed.loc[1, "functional_mutation_count_log1p"] == 0.0
    assert transformed.loc[1, "consequence_synonymous_ratio"] == 1.0
    assert transformed.loc[3, "multi_variant_gene_count_log1p"] > 0.0
    assert transformed.dtypes.eq("float32").all()


def test_em_v19_combines_em_v16_and_em_v20_without_duplicate_features() -> None:
    features = pd.DataFrame({
        "GENE1": ["A1V", "A1V", "WT", "A1V B2C", "WT", "B2C"] * 3,
        "GENE2": ["WT", "C3*", "C3*", "WT", "C3*", "WT"] * 3,
        "GENE3": ["WT", "WT", "D4D", "D4D", "WT", "WT"] * 3,
    })
    labels = pd.Series(["A"] * 6 + ["B"] * 6 + ["C"] * 6)
    pipeline = EMV19PreprocessingPipeline(
        min_mutation_count=1,
        top_genes_per_class=2,
        min_hotspot_count=2,
        max_hotspots=4,
        inner_signature_folds=3,
        stable_gene_folds=3,
        min_stable_gene_folds=2,
        onehot_genes_per_class=2,
        max_stable_genes=3,
    )

    transformed = pipeline.fit_transform(features, labels)

    assert len([column for column in transformed if column.startswith("signature_")]) == 6
    assert len([column for column in transformed if column.startswith("hotspot_")]) > 0
    assert any(column.startswith("stable_") for column in transformed)
    assert "multi_variant_gene_count_log1p" in transformed
    assert transformed.columns.tolist().count("mutation_burden_log1p") == 1
    assert transformed.columns.is_unique
    assert pipeline.__class__.__bases__ == (PreprocessingPipeline,)


def test_em_v20_limits_consequence_onehot_to_stable_genes() -> None:
    labels = pd.Series(["A"] * 6 + ["B"] * 6 + ["C"] * 6)
    features = pd.DataFrame({
        "GENE_A": ["A1V", "A1V", "A1V L2L", "R3*", "A1V", "WT"] + ["WT"] * 12,
        "GENE_B": ["WT"] * 6 + ["B2C", "B2C", "B2C", "Q4FS", "B2C", "WT"] + ["WT"] * 6,
        "GENE_C": ["WT"] * 12 + ["C3D", "C3D", "C3D", "10_11AA>VV", "C3D", "WT"],
        "NOISE": ["N1K" if index % 5 == 0 else "WT" for index in range(18)],
    })
    pipeline = EMV20PreprocessingPipeline(
        min_functional_mutation_count=1,
        top_genes_per_class=1,
        min_hotspot_count=2,
        max_hotspots=4,
        inner_signature_folds=2,
        stable_gene_folds=3,
        min_stable_gene_folds=2,
        onehot_genes_per_class=1,
        max_stable_genes=3,
    )

    transformed = pipeline.fit_transform(features, labels)

    assert set(pipeline.stable_genes_) == {"GENE_A", "GENE_B", "GENE_C"}
    for gene in pipeline.stable_genes_:
        assert gene not in transformed
        assert any(column.startswith(f"stable_{gene}_") for column in transformed)
    assert "stable_GENE_A_synonymous" in transformed
    assert transformed.loc[2, "stable_GENE_A_synonymous"] == 1.0
    assert pipeline.summary()["stable_onehot_genes"] == 3
    assert transformed.columns.is_unique
    assert transformed.dtypes.eq("float32").all()


def test_em_v21_reduces_raw_capacity_and_keeps_only_stable_hotspots() -> None:
    labels = pd.Series(["A"] * 6 + ["B"] * 6 + ["C"] * 6)
    features = pd.DataFrame({
        "GENE_A": ["A1V"] * 5 + ["WT"] + ["WT"] * 12,
        "GENE_B": ["WT"] * 6 + ["B2C"] * 5 + ["WT"] + ["WT"] * 6,
        "GENE_C": ["WT"] * 12 + ["C3D"] * 5 + ["WT"],
        "GENE_D": ["D4E" if index % 4 == 0 else "WT" for index in range(18)],
    })
    pipeline = EMV21PreprocessingPipeline(
        min_functional_mutation_count=1,
        max_raw_gene_features=2,
        top_genes_per_class=1,
        min_hotspot_count=3,
        stable_hotspot_folds=3,
        min_stable_hotspot_folds=2,
        max_stable_hotspots=2,
        inner_signature_folds=2,
    )

    transformed = pipeline.fit_transform(features, labels)

    assert len(pipeline.eligible_gene_columns) == 4
    assert len(pipeline.raw_gene_columns) == 2
    assert len(pipeline.hotspots_) <= 2
    assert pipeline.hotspots_
    assert not any(column.startswith("stable_GENE") for column in transformed)
    assert len([column for column in transformed if column.endswith("_match_rate")]) == 3
    assert not any(column.endswith("_match_count") for column in transformed)
    assert pipeline.summary()["consequence_onehot_features"] == 0
    assert transformed.dtypes.eq("float32").all()


def make_tcga_label_integrity_features() -> tuple[pd.DataFrame, pd.Series]:
    labels = pd.Series(
        ["GBMLGG"] * 2 + ["LGG"] * 2
        + ["KIPAN"] * 2 + ["KIRC"] * 2
        + ["STES"] * 2 + ["COAD"] * 2
    )
    features = pd.DataFrame({
        f"GENE_{class_name}": [
            "MUT" if label == class_name else "WT" for label in labels
        ]
        for class_name in ("GBMLGG", "LGG", "KIPAN", "KIRC", "STES", "COAD")
    })
    return features, labels


def test_em_v26_adds_tcga_family_signatures_without_changing_labels() -> None:
    features, labels = make_tcga_label_integrity_features()
    pipeline = EMV26PreprocessingPipeline(
        min_mutation_count=1,
        max_raw_gene_features=6,
        top_genes_per_family=1,
        inner_family_folds=2,
    )

    transformed = pipeline.fit_transform(features, labels)

    family_columns = [column for column in transformed if column.startswith("tcga_family_")]
    assert len(family_columns) == 6
    assert pipeline.label_encoder.classes_.tolist() == sorted(labels.unique())
    assert {"GBMLGG", "KIPAN", "STES"} <= set(pipeline.label_encoder.classes_)
    assert not {"GBM", "KICH", "KIRP", "STAD", "ESCA"} & set(
        pipeline.label_encoder.classes_
    )


def test_em_v27_adds_sibling_contrasts_without_splitting_composites() -> None:
    features, labels = make_tcga_label_integrity_features()
    pipeline = EMV27PreprocessingPipeline(
        min_mutation_count=1,
        max_raw_gene_features=6,
        top_genes_per_class=1,
        inner_contrast_folds=2,
    )

    transformed = pipeline.fit_transform(features, labels)

    contrast_columns = [column for column in transformed if column.startswith("tcga_sibling_")]
    assert len(contrast_columns) == 12
    assert "tcga_sibling_GBMLGG_weighted" in transformed
    assert "tcga_sibling_KIPAN_weighted" in transformed
    assert "tcga_sibling_STES_weighted" in transformed
    assert pipeline.label_encoder.inverse_transform(
        pipeline.label_encoder.transform(labels)
    ).tolist() == labels.tolist()


def test_em_v22_combines_v16_and_family_features_without_duplicates() -> None:
    features, labels = make_tcga_label_integrity_features()
    pipeline = EMV22PreprocessingPipeline(
        min_mutation_count=1,
        max_raw_gene_features=6,
        top_genes_per_class=1,
        top_genes_per_family=1,
        min_hotspot_count=1,
        max_hotspots=6,
        inner_signature_folds=2,
    )

    transformed = pipeline.fit_transform(features, labels)

    assert transformed.columns.tolist().count("mutation_burden_log1p") == 1
    assert any(column.startswith("signature_GBMLGG_") for column in transformed)
    assert any(column.startswith("tcga_family_") for column in transformed)
    assert pipeline.label_encoder.classes_.tolist() == sorted(labels.unique())


def test_em_v23_combines_v16_and_sibling_features_without_duplicates() -> None:
    features, labels = make_tcga_label_integrity_features()
    pipeline = EMV23PreprocessingPipeline(
        min_mutation_count=1,
        max_raw_gene_features=6,
        top_genes_per_class=1,
        top_genes_per_sibling=1,
        min_hotspot_count=1,
        max_hotspots=6,
        inner_signature_folds=2,
    )

    transformed = pipeline.fit_transform(features, labels)

    assert transformed.columns.tolist().count("mutation_burden_log1p") == 1
    assert any(column.startswith("signature_GBMLGG_") for column in transformed)
    assert "tcga_sibling_GBMLGG_weighted" in transformed
    assert pipeline.label_encoder.inverse_transform(
        pipeline.label_encoder.transform(labels)
    ).tolist() == labels.tolist()


def test_em_v24_combines_all_and_functional_signatures_without_duplicates() -> None:
    labels = pd.Series(["A"] * 6 + ["B"] * 6 + ["C"] * 6)
    features = pd.DataFrame({
        "GENE_A": ["A1V"] * 6 + ["WT"] * 12,
        "GENE_B": ["WT"] * 6 + ["B2C"] * 6 + ["WT"] * 6,
        "GENE_C": ["WT"] * 12 + ["C3D"] * 6,
        "SYN_A": ["L4L"] * 6 + ["WT"] * 12,
    })
    pipeline = EMV24PreprocessingPipeline(
        min_mutation_count=1,
        min_functional_mutation_count=1,
        top_genes_per_class=1,
        min_hotspot_count=2,
        max_hotspots=4,
        inner_signature_folds=2,
    )

    transformed = pipeline.fit_transform(features, labels)

    assert transformed.columns.is_unique
    assert transformed.columns.tolist().count("mutated_gene_count_log1p") == 1
    assert any(column.startswith("signature_all_A_") for column in transformed)
    assert any(column.startswith("signature_functional_A_") for column in transformed)
    assert "SYN_A" in pipeline.mutation_signature_genes_
    assert "SYN_A" not in pipeline.functional_signature_genes_
    assert pipeline.__class__.__bases__ == (PreprocessingPipeline,)


def test_em_v25_combines_oof_signature_and_token_features_without_duplicates() -> None:
    labels = pd.Series(["A"] * 6 + ["B"] * 6 + ["C"] * 6)
    features = pd.DataFrame({
        "GENE_A": ["A1V", "A1V A1V", "A1V B2C", "A1V", "A1V", "WT"] + ["WT"] * 12,
        "GENE_B": ["WT"] * 6 + ["B2C", "B2C", "B2C C3*", "B2C", "B2C", "WT"] + ["WT"] * 6,
        "GENE_C": ["WT"] * 12 + ["C3D", "C3D", "C3D D4D", "C3D", "C3D", "WT"],
    })
    pipeline = EMV25PreprocessingPipeline(
        min_mutation_count=1,
        max_gene_features=3,
        top_genes_per_class=1,
        min_hotspot_count=2,
        max_hotspots=6,
        correlation_threshold=0.99,
        inner_signature_folds=2,
    )

    transformed = pipeline.fit_transform(features, labels)

    assert transformed.columns.is_unique
    assert transformed.columns.tolist().count("mutation_burden_log1p") <= 1
    assert "gene_GENE_A_mutation_count" in transformed
    assert "gene_GENE_A_multi_hit" in transformed
    assert any(column.startswith("signature_A_") for column in transformed)
    assert pipeline.__class__.__bases__ == (PreprocessingPipeline,)


def test_em_v28_combines_oof_signature_and_stable_consequence_without_duplicates() -> None:
    labels = pd.Series(["A"] * 6 + ["B"] * 6 + ["C"] * 6)
    features = pd.DataFrame({
        "GENE_A": ["A1V", "A1V", "A1V L2L", "R3*", "A1V", "WT"] + ["WT"] * 12,
        "GENE_B": ["WT"] * 6 + ["B2C", "B2C", "B2C", "Q4FS", "B2C", "WT"] + ["WT"] * 6,
        "GENE_C": ["WT"] * 12 + ["C3D", "C3D", "C3D", "10_11AA>VV", "C3D", "WT"],
    })
    pipeline = EMV28PreprocessingPipeline(
        min_mutation_count=1,
        top_genes_per_class=1,
        min_hotspot_count=2,
        max_hotspots=4,
        inner_signature_folds=2,
        stable_gene_folds=3,
        min_stable_gene_folds=2,
        onehot_genes_per_class=1,
        max_stable_genes=3,
    )

    transformed = pipeline.fit_transform(features, labels)

    assert transformed.columns.is_unique
    assert transformed.columns.tolist().count("mutation_burden_log1p") == 1
    assert any(column.startswith("signature_A_") for column in transformed)
    assert any(column.startswith("stable_GENE_A_") for column in transformed)
    assert pipeline.__class__.__bases__ == (PreprocessingPipeline,)
