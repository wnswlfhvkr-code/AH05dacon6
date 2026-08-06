"""EMV45 수치 피처와 JSJ V9 텍스트 피처를 중복 없이 결합합니다."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from src.pipelines.base import PreprocessingPipeline
from src.pipelines.pipeline_em_v45 import EMV45PreprocessingPipeline
from src.pipelines.pipeline_jsj_v1 import TextTreeFeatureBundle


def _mutation_type(value: str) -> str:
    lowered = value.lower()
    if "delins" in lowered:
        return "DELINS"
    if "fs" in lowered:
        return "FS"
    if "del" in lowered:
        return "DEL"
    if "ins" in lowered:
        return "INS"
    if "dup" in lowered:
        return "DUP"
    if "*" in value or "stop" in lowered:
        return "STOP"
    if ">" in value:
        return "SUB"
    return "OTHER"


def _position_bin(position: int) -> str:
    if position <= 50:
        return "001_050"
    if position <= 100:
        return "051_100"
    if position <= 250:
        return "101_250"
    if position <= 500:
        return "251_500"
    return "501_PLUS"


class PipeCombEMV1PreprocessingPipeline(PreprocessingPipeline):
    """서로 다른 모델에 전달할 텍스트·수치 피처 묶음을 만듭니다.

    ``text``에는 JSJ V9의 Word/Char TF-IDF만 담고, ``tree``에는 EMV45의
    severity, burden, consequence, hotspot, OOF signature 피처만 담습니다.
    JSJ의 단순 변이 여부·구조 요약과 V9의 EM16/EM24 재앙상블은 EMV45와
    중복되므로 생성하지 않습니다.
    """

    name = "pipeComb_em_v1"
    evaluation_folds = 5
    _event_pattern = re.compile(r"^([A-Za-z*]+)(\d+)([A-Za-z*]+)$")
    conflict_pairs = (("KIRC", "KIPAN"), ("LGG", "GBMLGG"))

    def __init__(
        self,
        word_ngram_range: list[int] | tuple[int, int] = (1, 2),
        word_min_df: int = 2,
        word_max_features: int = 250_000,
        char_ngram_range: list[int] | tuple[int, int] = (3, 5),
        char_min_df: int = 3,
        char_max_features: int = 180_000,
        char_weight: float = 0.5,
        sublinear_tf: bool = True,
        split_multi_event: bool = False,
        **parameters: object,
    ) -> None:
        super().__init__()
        self.word_ngram_range = tuple(word_ngram_range)
        self.char_ngram_range = tuple(char_ngram_range)
        self.char_weight = float(char_weight)
        self.split_multi_event = bool(split_multi_event)
        self.word_vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            ngram_range=self.word_ngram_range,
            min_df=word_min_df,
            max_features=word_max_features,
            sublinear_tf=sublinear_tf,
            dtype=np.float32,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.char_ngram_range,
            min_df=char_min_df,
            max_features=char_max_features,
            sublinear_tf=sublinear_tf,
            dtype=np.float32,
        )
        self.em_pipeline = EMV45PreprocessingPipeline(**parameters)
        self.text_feature_count_ = 0
        self.tree_feature_count_ = 0
        self.steps = (
            "결측값과 WT 동일 처리",
            "JSJ Word·Char TF-IDF 텍스트 뷰",
            "EMV45 수치 뷰",
            "중복 JSJ 구조 피처 및 EM16·EM24 재앙상블 제거",
            "원본 SUBCLASS 유지",
        )

    def _validate_columns(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"변환에 필요한 유전자 컬럼이 없습니다: {sorted(missing)}")
        return features[self.feature_columns]

    def _documents(self, features: pd.DataFrame) -> np.ndarray:
        values = features.astype("string").fillna("WT").to_numpy()
        documents: list[str] = []
        for row in values:
            tokens: list[str] = []
            for gene_index, raw_value in enumerate(row):
                value = str(raw_value).strip()
                if not value or value.upper() in {"WT", "<NA>"}:
                    continue
                raw_events = value.split()
                events = raw_events if self.split_multi_event else [value.replace(" ", "")]
                gene = self.feature_columns[gene_index]
                for event in events:
                    if event.startswith("p."):
                        event = event[2:]
                    event_type = _mutation_type(event)
                    tokens.extend((
                        f"G={gene}",
                        f"GT={gene}:{event_type}",
                        f"E={gene}:{event}",
                        f"T={event_type}",
                    ))
                    match = self._event_pattern.match(event)
                    if match:
                        reference, position, alternate = match.groups()
                        position_group = _position_bin(int(position))
                        tokens.extend((
                            f"AA={gene}:{reference}>{alternate}",
                            f"AAG={reference}>{alternate}",
                            f"POS={gene}:{position_group}",
                            f"POSG={position_group}",
                        ))
            documents.append(" ".join(tokens) if tokens else "NO_MUTATION")
        return np.asarray(documents, dtype=object)

    def _fit_text(self, features: pd.DataFrame) -> None:
        documents = self._documents(features)
        self.word_vectorizer.fit(documents)
        self.char_vectorizer.fit(documents)
        self.text_feature_count_ = (
            len(self.word_vectorizer.vocabulary_)
            + len(self.char_vectorizer.vocabulary_)
        )

    def _transform_text(self, features: pd.DataFrame) -> csr_matrix:
        documents = self._documents(features)
        word = self.word_vectorizer.transform(documents)
        char = self.char_vectorizer.transform(documents)
        return hstack((word, char * self.char_weight), format="csr")

    @staticmethod
    def _as_tree_matrix(features: pd.DataFrame) -> csr_matrix:
        return csr_matrix(features.to_numpy(dtype=np.float32, copy=False))

    def fit(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> "PipeCombEMV1PreprocessingPipeline":
        self.feature_columns = features.columns.tolist()
        selected = self._validate_columns(features)
        self._fit_text(selected)
        self.em_pipeline.fit(selected, labels)
        self.tree_feature_count_ = int(
            self.em_pipeline.summary()["remaining_features"]
        )
        self.label_encoder.fit(labels)
        return self

    def fit_transform(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> TextTreeFeatureBundle:
        self.feature_columns = features.columns.tolist()
        selected = self._validate_columns(features)
        self._fit_text(selected)
        tree = self.em_pipeline.fit_transform(selected, labels)
        self.tree_feature_count_ = tree.shape[1]
        self.label_encoder.fit(labels)
        return TextTreeFeatureBundle(
            text=self._transform_text(selected),
            tree=self._as_tree_matrix(tree),
        )

    def transform(self, features: pd.DataFrame) -> TextTreeFeatureBundle:
        selected = self._validate_columns(features)
        tree = self.em_pipeline.transform(selected)
        return TextTreeFeatureBundle(
            text=self._transform_text(selected),
            tree=self._as_tree_matrix(tree),
        )

    def summary(self) -> dict[str, int]:
        em_summary = self.em_pipeline.summary()
        return {
            "remaining_features": self.text_feature_count_ + self.tree_feature_count_,
            "text_features": self.text_feature_count_,
            "tree_features": self.tree_feature_count_,
            "dropped_constant_features": int(
                em_summary.get("dropped_constant_features", 0)
            ),
            "removed_duplicate_jsj_tree_branches": 1,
            "removed_duplicate_jsj_fixed_structural_features": 12,
            "removed_duplicate_em_team_branches": 2,
            "conflict_pairs_preserved_for_model": len(self.conflict_pairs),
        }
