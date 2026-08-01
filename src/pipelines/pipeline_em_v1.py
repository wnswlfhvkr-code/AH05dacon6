"""EM 실험 버전 1용 전처리 파이프라인입니다."""

import pandas as pd

from src.pipelines.base import PreprocessingPipeline


class EMV1PreprocessingPipeline(PreprocessingPipeline):
    """특징 유전자의 변이 여부를 추가한 EM v1 전처리를 적용합니다."""

    name = "em_v1"
    mutation_genes = (
        "IDH1",
        "APC",
        "PTEN",
        "BRAF",
        "TP53",
        "PLEC",
        "VHL",
        "PCLO",
        "PIK3CA",
        "MXRA5",
    )

    def __init__(self, **parameters: object) -> None:
        super().__init__(**parameters)
        self.steps = ("유전자 변이 여부 이진 피처 추가", *self.steps)

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

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "EMV1PreprocessingPipeline":
        super().fit(self._add_mutation_features(features), labels)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return super().transform(self._add_mutation_features(features))
