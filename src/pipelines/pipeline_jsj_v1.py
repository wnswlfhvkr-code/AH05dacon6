"""JSJ v1: 유전자 변이를 Word+Char TF-IDF와 구조 피처로 변환합니다."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


@dataclass
class TextTreeFeatureBundle:
    """문자열 모델과 트리 모델에 전달할 피처 묶음입니다."""

    text: csr_matrix
    tree: csr_matrix


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


class JSJV1PreprocessingPipeline:
    """JSJ v1 실험용 Word+Char TF-IDF 전처리 파이프라인입니다."""

    name = "jsj_v1"
    _event_pattern = re.compile(r"^([A-Za-z*]+)(\d+)([A-Za-z*]+)$")
    _event_types = ("SUB", "STOP", "FS", "DEL", "INS", "DELINS", "DUP", "OTHER")

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
        return_bundle: bool = False,
        **_: object,
    ) -> None:
        self.word_ngram_range = tuple(word_ngram_range)
        self.char_ngram_range = tuple(char_ngram_range)
        self.char_weight = float(char_weight)
        self.split_multi_event = bool(split_multi_event)
        self.return_bundle = bool(return_bundle)
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
        self.label_encoder = LabelEncoder()
        self.feature_columns: list[str] = []
        self.tree_gene_columns: list[str] = []
        self._tree_feature_count = 0

    def _validate_columns(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"누락된 피처가 있습니다: {sorted(missing)}")
        return features[self.feature_columns]

    def _documents_and_structural(
        self, features: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = features.fillna("WT").astype(str).to_numpy()
        missing_flags = features.isna().to_numpy()
        documents: list[str] = []
        gene_mutated = np.zeros(values.shape, dtype=np.float32)
        structural = np.zeros(
            (len(features), 4 + len(self._event_types)), dtype=np.float32
        )
        type_to_index = {
            event_type: index for index, event_type in enumerate(self._event_types)
        }

        for row_index, row in enumerate(values):
            tokens: list[str] = []
            event_count = 0
            multi_event_genes = 0
            for gene_index, raw_value in enumerate(row):
                value = raw_value.strip()
                if not value or value.upper() == "WT":
                    continue
                gene_mutated[row_index, gene_index] = 1.0
                raw_events = value.split()
                if len(raw_events) > 1:
                    multi_event_genes += 1
                events = raw_events if self.split_multi_event else [value.replace(" ", "")]
                for event in events:
                    if event.startswith("p."):
                        event = event[2:]
                    event_count += 1
                    event_type = _mutation_type(event)
                    structural[
                        row_index, 4 + type_to_index[event_type]
                    ] += 1.0
                    tokens.extend(
                        (
                            f"G={self.feature_columns[gene_index]}",
                            f"GT={self.feature_columns[gene_index]}:{event_type}",
                            f"E={self.feature_columns[gene_index]}:{event}",
                            f"T={event_type}",
                        )
                    )
                    match = self._event_pattern.match(event)
                    if match:
                        reference, position, alternate = match.groups()
                        bin_name = _position_bin(int(position))
                        tokens.extend(
                            (
                                f"AA={self.feature_columns[gene_index]}:{reference}>{alternate}",
                                f"AAG={reference}>{alternate}",
                                f"POS={self.feature_columns[gene_index]}:{bin_name}",
                                f"POSG={bin_name}",
                            )
                        )
            structural[row_index, 0] = gene_mutated[row_index].sum()
            structural[row_index, 1] = event_count
            structural[row_index, 2] = multi_event_genes
            structural[row_index, 3] = missing_flags[row_index].sum()
            documents.append(" ".join(tokens) if tokens else "NO_MUTATION")

        return (
            np.asarray(documents, dtype=object),
            gene_mutated,
            structural,
        )

    def fit(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> "JSJV1PreprocessingPipeline":
        self.feature_columns = features.columns.tolist()
        documents, gene_mutated, structural = self._documents_and_structural(features)
        self.word_vectorizer.fit(documents)
        self.char_vectorizer.fit(documents)
        variable_mask = np.ptp(gene_mutated, axis=0) > 0
        self.tree_gene_columns = list(
            np.asarray(self.feature_columns, dtype=object)[variable_mask]
        )
        self._tree_feature_count = len(self.tree_gene_columns) + structural.shape[1]
        self.label_encoder.fit(labels)
        return self

    def fit_transform(self, features: pd.DataFrame, labels: pd.Series):
        return self.fit(features, labels).transform(features)

    def transform(self, features: pd.DataFrame):
        selected = self._validate_columns(features)
        documents, gene_mutated, structural = self._documents_and_structural(selected)
        word = self.word_vectorizer.transform(documents)
        char = self.char_vectorizer.transform(documents)
        text = hstack((word, char * self.char_weight), format="csr")
        if not self.return_bundle:
            return text

        gene_indices = [self.feature_columns.index(column) for column in self.tree_gene_columns]
        tree = hstack(
            (
                csr_matrix(gene_mutated[:, gene_indices]),
                csr_matrix(structural),
            ),
            format="csr",
        )
        return TextTreeFeatureBundle(text=text, tree=tree)

    def summary(self) -> dict[str, int]:
        text_count = len(self.word_vectorizer.vocabulary_) + len(
            self.char_vectorizer.vocabulary_
        )
        return {
            "remaining_features": text_count + (
                self._tree_feature_count if self.return_bundle else 0
            ),
            "dropped_constant_features": len(self.feature_columns)
            - len(self.tree_gene_columns),
        }

    def encode_labels(self, labels: pd.Series):
        return self.label_encoder.transform(labels)

    def decode_labels(self, labels):
        return self.label_encoder.inverse_transform(np.asarray(labels).astype(int))
