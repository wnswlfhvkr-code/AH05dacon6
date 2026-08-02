"""암종별 특징 유전자 전체를 사용하는 EM 실험 버전 2 파이프라인입니다."""

import pandas as pd

from src.pipelines.base import PreprocessingPipeline


class EMV2PreprocessingPipeline(PreprocessingPipeline):
    """암종별 특징 유전자의 변이 여부를 이진 피처로 추가합니다."""

    name = "em_v2"
    mutation_genes = (
        "APC",
        "ATRX",
        "BRAF",
        "BTG1",
        "BTG2",
        "CDKN1A",
        "CDKN2A",
        "CMPK2",
        "CTNNB1",
        "DST",
        "EGFR",
        "HRAS",
        "IDH1",
        "IDH2",
        "KIT",
        "KMT2D",
        "MXRA5",
        "NCOR2",
        "NLRP3",
        "NOTCH1",
        "NPM1",
        "PCLO",
        "PIK3CA",
        "PIM1",
        "PKD1",
        "PLEC",
        "PLXNB2",
        "PTEN",
        "RYR2",
        "SOWAHC",
        "SPOP",
        "SPTA1",
        "SYNE1",
        "TP53",
        "VHL",
    )

    def __init__(self, **parameters: object) -> None:
        super().__init__(**parameters)
        self.dropped_original_mutation_columns: list[str] = []
        self.steps = (
            "유전자 변이 여부 이진 피처 추가",
            *self.steps,
            "원본 특징 유전자 피처 제거",
        )

    @classmethod
    def _add_mutation_features(cls, features: pd.DataFrame) -> pd.DataFrame:
        missing_genes = [gene for gene in cls.mutation_genes if gene not in features.columns]
        if missing_genes:
            raise ValueError(f"변이 피처 생성에 필요한 유전자 컬럼이 없습니다: {missing_genes}")

        enriched = features.copy()
        for gene in cls.mutation_genes:
            normalized = enriched[gene].astype("string").str.strip().str.upper()
            enriched[f"{gene}_mutated"] = normalized.ne("WT").fillna(False).astype("int8")
        return enriched

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "EMV2PreprocessingPipeline":
        super().fit(self._add_mutation_features(features), labels)
        self.dropped_original_mutation_columns = [
            gene for gene in self.mutation_genes if gene in self.feature_columns
        ]
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        transformed = super().transform(self._add_mutation_features(features))
        return transformed.drop(
            columns=self.dropped_original_mutation_columns,
            errors="ignore",
        )

    def summary(self) -> dict[str, int]:
        summary = super().summary()
        dropped_count = len(self.dropped_original_mutation_columns)
        summary["dropped_original_mutation_features"] = dropped_count
        summary["remaining_features"] -= dropped_count
        return summary
