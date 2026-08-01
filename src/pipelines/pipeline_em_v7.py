"""TCGA canonical pathway 그룹을 추가하는 EM 실험 버전 7입니다."""

from __future__ import annotations

import pandas as pd

from src.pipelines.base import PreprocessingPipeline


def create_mutation_presence_matrix(
    features: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """v7 입력을 WT=0, 변이=1인 이진 행렬로 변환합니다."""
    selected_columns = features.columns.tolist() if columns is None else columns
    missing_columns = set(selected_columns) - set(features.columns)
    if missing_columns:
        raise ValueError(f"변이 행렬 생성에 필요한 피처가 없습니다: {sorted(missing_columns)}")
    mutation_columns = {
        column: features[column].astype("string").str.strip().str.upper()
        .ne("WT").fillna(False).astype("int8")
        for column in selected_columns
    }
    return pd.DataFrame(mutation_columns, index=features.index)


def select_genes_by_mutation_count(
    features: pd.DataFrame,
    minimum_count: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    """v7 학습 데이터에서 최소 변이 횟수를 만족하는 유전자를 선택합니다."""
    mutation_counts = (
        create_mutation_presence_matrix(features).sum(axis=0).astype(int).to_dict()
    )
    selected = [column for column in features.columns if mutation_counts[column] >= minimum_count]
    dropped = [column for column in features.columns if mutation_counts[column] < minimum_count]
    return selected, dropped, mutation_counts


class EMV7PreprocessingPipeline(PreprocessingPipeline):
    """이진 유전자 피처에 10개 canonical cancer pathway 집계를 추가합니다."""

    name = "em_v7"
    pathway_source = "Sanchez-Vega et al., Cell 2018, doi:10.1016/j.cell.2018.03.035"
    pathway_genes = {
        "cell_cycle": (
            "CDKN2A", "CDKN2B", "CCND1", "CCND2", "CCND3", "CDK4", "CDK6",
            "RB1", "RBL1", "RBL2", "E2F1", "E2F2", "E2F3", "E2F4", "E2F5",
        ),
        "hippo": (
            "NF2", "LATS1", "LATS2", "STK3", "STK4", "SAV1", "YAP1", "WWTR1",
            "FAT1", "FAT2", "FAT3", "FAT4", "TEAD1", "TEAD2", "TEAD3", "TEAD4",
        ),
        "myc": ("MYC", "MYCN", "MYCL", "MAX", "MGA", "MNT", "MXD1", "FBXW7"),
        "notch": (
            "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "FBXW7", "EP300", "CREBBP",
            "NCOR1", "NCOR2", "SPEN", "MAML1", "MAML2", "MAML3",
        ),
        "nrf2": ("NFE2L2", "KEAP1", "CUL3"),
        "pi3k_akt": (
            "PIK3CA", "PIK3CB", "PIK3R1", "PIK3R2", "PTEN", "AKT1", "AKT2",
            "AKT3", "MTOR", "TSC1", "TSC2", "STK11", "RHEB", "RICTOR", "INPP4B",
        ),
        "rtk_ras": (
            "EGFR", "ERBB2", "ERBB3", "ERBB4", "KRAS", "NRAS", "HRAS", "BRAF",
            "RAF1", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "NF1", "RASA1", "ALK",
            "ROS1", "RET", "MET", "FGFR1", "FGFR2", "FGFR3", "FGFR4", "KIT",
            "PDGFRA", "PDGFRB",
        ),
        "tgf_beta": (
            "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SMAD4", "ACVR1B", "ACVR2A",
            "BMPR2",
        ),
        "p53": ("TP53", "MDM2", "MDM4", "ATM", "CHEK1", "CHEK2", "CDKN2A"),
        "wnt": ("APC", "CTNNB1", "AXIN1", "AXIN2", "RNF43", "ZNRF3", "TCF7L2", "AMER1"),
    }

    def __init__(self, min_mutation_count: int = 5, **parameters: object) -> None:
        super().__init__(**parameters)
        if (
            isinstance(min_mutation_count, bool)
            or not isinstance(min_mutation_count, int)
            or min_mutation_count < 1
        ):
            raise ValueError("min_mutation_count는 1 이상의 정수여야 합니다.")

        self.min_mutation_count = min_mutation_count
        self.selected_gene_columns: list[str] = []
        self.dropped_rare_columns: list[str] = []
        self.mutation_counts_: dict[str, int] = {}
        self.available_pathway_genes: dict[str, list[str]] = {}
        self.required_gene_columns: list[str] = []
        self.steps = (
            "최소 변이 빈도 필터",
            "전체 유전자 이진 변이 표현",
            "TCGA canonical pathway 집계 피처 추가",
        )

    def _build_features(self, features: pd.DataFrame) -> pd.DataFrame:
        mutation_matrix = create_mutation_presence_matrix(
            features,
            self.required_gene_columns,
        )
        engineered = mutation_matrix[self.selected_gene_columns].copy()
        for pathway, genes in self.available_pathway_genes.items():
            count = mutation_matrix[genes].sum(axis=1).astype("float32")
            engineered[f"pathway_{pathway}_count"] = count
            engineered[f"pathway_{pathway}_any"] = count.gt(0).astype("int8")
            engineered[f"pathway_{pathway}_rate"] = (count / len(genes)).astype(
                "float32"
            )
        return engineered

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "EMV7PreprocessingPipeline":
        (
            self.selected_gene_columns,
            self.dropped_rare_columns,
            self.mutation_counts_,
        ) = select_genes_by_mutation_count(features, self.min_mutation_count)
        available_columns = set(features.columns)
        self.available_pathway_genes = {
            pathway: [gene for gene in genes if gene in available_columns]
            for pathway, genes in self.pathway_genes.items()
        }
        empty_pathways = [
            pathway for pathway, genes in self.available_pathway_genes.items() if not genes
        ]
        if empty_pathways:
            raise ValueError(f"데이터에 유전자가 없는 pathway가 있습니다: {empty_pathways}")

        pathway_union = {
            gene for genes in self.available_pathway_genes.values() for gene in genes
        }
        self.required_gene_columns = [
            column for column in features.columns
            if column in set(self.selected_gene_columns) | pathway_union
        ]
        super().fit(self._build_features(features), labels)
        print(f"[{self.name}] 생성된 pathway 집계 피처: {3 * len(self.pathway_genes)}개")
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return super().transform(self._build_features(features)).astype("float32")

    def summary(self) -> dict[str, int]:
        summary = super().summary()
        summary["dropped_rare_features"] = len(self.dropped_rare_columns)
        summary["pathway_groups"] = len(self.pathway_genes)
        summary["pathway_features"] = 3 * len(self.pathway_genes)
        return summary
