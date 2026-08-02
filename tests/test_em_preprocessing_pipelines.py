import pandas as pd

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
