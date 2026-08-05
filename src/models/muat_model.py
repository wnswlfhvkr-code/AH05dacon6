"""Mutation-Attention(MuAt) 방식의 현재 CSV용 암종 분류 모델.

정식 MuAt는 MAF/VCF에서 얻은 염기서열 motif, genomic position, strand와
genomic annotation을 개별 변이 토큰으로 사용합니다. 현재 train.csv에는
유전자별 단백질 변화 문자열만 있으므로 이 파일은 존재하지 않는 DNA 정보를
추정하지 않습니다.

대신 ``jyp_raw`` 전처리가 만든 유전자별 exact-cell 범주 행렬의 각 non-zero
항목을 하나의 변이 사건으로 취급하는 ``protein_proxy`` 모드를 제공합니다.
이는 MuAt의 mutation-level attention 학습 아이디어를 적용한 신규 모델이며,
공식 MuAt 체크포인트와 입력 의미가 같지 않습니다. MAF/VCF를 확보한 경우에는
공식 ``muat`` 패키지의 motif+position+GES 모델을 사용해야 합니다.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy import sparse
from sklearn.metrics import f1_score

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as error:  # pragma: no cover - 의존성 미설치 환경
    raise ImportError(
        "MuAt 모델에는 PyTorch가 필요합니다. "
        "`pip install -r requirements.txt`를 실행하세요."
    ) from error


class MuAtProteinProxyNetwork(nn.Module):
    """유전자와 exact 단백질 변이 범주를 mutation token으로 처리합니다."""

    def __init__(
        self,
        num_genes: int,
        variant_hash_size: int,
        gene_embedding_dim: int,
        variant_embedding_dim: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gene_embedding = nn.Embedding(
            num_genes + 1, gene_embedding_dim, padding_idx=0
        )
        self.variant_embedding = nn.Embedding(
            variant_hash_size + 1, variant_embedding_dim, padding_idx=0
        )
        self.input_projection = nn.Linear(
            gene_embedding_dim + variant_embedding_dim, model_dim
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(model_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        gene_tokens: torch.Tensor,
        variant_tokens: torch.Tensor,
        log_burden: torch.Tensor,
    ) -> torch.Tensor:
        embedded = torch.cat(
            (
                self.gene_embedding(gene_tokens),
                self.variant_embedding(variant_tokens),
            ),
            dim=-1,
        )
        embedded = self.input_projection(embedded)
        cls = self.cls_token.expand(gene_tokens.shape[0], -1, -1)
        embedded = torch.cat((cls, embedded), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros(
                    (gene_tokens.shape[0], 1),
                    dtype=torch.bool,
                    device=gene_tokens.device,
                ),
                gene_tokens.eq(0),
            ),
            dim=1,
        )
        encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)
        representation = torch.cat((encoded[:, 0], log_burden), dim=1)
        return self.classifier(representation)


class MuAtClassifier:
    """프로젝트 학습 루프와 호환되는 mutation-level attention 분류기."""

    def __init__(self, model_config: dict, seed: int) -> None:
        self.seed = int(seed)
        self.input_mode = str(model_config.get("input_mode", "protein_proxy"))
        self.allow_proxy_mode = bool(model_config.get("allow_proxy_mode", False))
        self.mutation_sampling_size = int(
            model_config.get("mutation_sampling_size", 128)
        )
        self.variant_hash_size = int(model_config.get("variant_hash_size", 65_536))
        self.gene_embedding_dim = int(model_config.get("gene_embedding_dim", 64))
        self.variant_embedding_dim = int(
            model_config.get("variant_embedding_dim", 32)
        )
        self.model_dim = int(model_config.get("model_dim", 128))
        self.num_layers = int(model_config.get("num_layers", 2))
        self.num_heads = int(model_config.get("num_heads", 4))
        self.hidden_dim = int(model_config.get("hidden_dim", 64))
        self.dropout = float(model_config.get("dropout", 0.25))
        self.epochs = int(model_config.get("epochs", 30))
        if "n_estimators" in model_config:
            self.epochs = int(model_config["n_estimators"])
        self.learning_rate = float(model_config.get("learning_rate", 3e-4))
        self.weight_decay = float(model_config.get("weight_decay", 1e-4))
        self.batch_size = int(model_config.get("batch_size", 64))
        self.num_workers = int(model_config.get("num_workers", 0))
        self.gradient_clip_norm = float(
            model_config.get("gradient_clip_norm", 1.0)
        )
        self.label_smoothing = float(model_config.get("label_smoothing", 0.05))
        self.class_weight = model_config.get("class_weight", "balanced")
        self.early_stopping_rounds = model_config.get("early_stopping_rounds")
        self.device_name = str(model_config.get("device", "auto"))

        self._validate_config()
        self.classes_: np.ndarray | None = None
        self.model_: MuAtProteinProxyNetwork | None = None
        self.device_: torch.device | None = None
        self.num_genes_: int | None = None
        self.best_iteration: int | None = None
        self.history_: list[dict[str, float | int]] = []
        self.n_features_in_: int | None = None

    def _validate_config(self) -> None:
        if self.input_mode != "protein_proxy":
            raise ValueError(
                "이 프로젝트의 train.csv 학습 루프는 input_mode=protein_proxy만 "
                "지원합니다. MAF/VCF 기반 정식 MuAt는 공식 muat 패키지를 사용하세요."
            )
        if not self.allow_proxy_mode:
            raise ValueError(
                "현재 CSV 모드는 공식 MuAt 입력이 아닙니다. 이를 인지하고 실행하려면 "
                "allow_proxy_mode=true를 명시하세요."
            )
        positive = {
            "mutation_sampling_size": self.mutation_sampling_size,
            "variant_hash_size": self.variant_hash_size,
            "model_dim": self.model_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }
        if any(value < 1 for value in positive.values()):
            raise ValueError(f"1 이상이어야 하는 설정이 있습니다: {positive}")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim은 num_heads로 나누어떨어져야 합니다.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout은 0 이상 1 미만이어야 합니다.")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing은 0 이상 1 미만이어야 합니다.")
        if self.early_stopping_rounds is not None and int(
            self.early_stopping_rounds
        ) < 1:
            raise ValueError("early_stopping_rounds는 1 이상이어야 합니다.")

    @staticmethod
    def _require_sparse_matrix(features: object) -> sparse.csr_matrix:
        if not sparse.issparse(features):
            raise TypeError(
                "MuAt protein_proxy 입력에는 preprocessing.name=jyp_raw가 "
                "생성한 희소 행렬이 필요합니다."
            )
        matrix = sparse.csr_matrix(features, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("MuAt 입력 피처는 2차원이어야 합니다.")
        if matrix.data.size and not np.isfinite(matrix.data).all():
            raise ValueError("MuAt 입력에 NaN 또는 무한대가 있습니다.")
        rounded = np.rint(matrix.data)
        if matrix.data.size and not np.allclose(matrix.data, rounded, atol=1e-5):
            raise ValueError(
                "protein_proxy는 jyp_raw의 정수형 exact-cell 범주 코드가 필요합니다."
            )
        matrix.data = rounded.astype(np.float32, copy=False)
        matrix.eliminate_zeros()
        return matrix

    def _tokenize(
        self,
        features: sparse.csr_matrix,
        *,
        seed_offset: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample_count = features.shape[0]
        genes = np.zeros(
            (sample_count, self.mutation_sampling_size), dtype=np.int64
        )
        variants = np.zeros_like(genes)
        burden = np.zeros((sample_count, 1), dtype=np.float32)
        rng = np.random.default_rng(self.seed + seed_offset)

        for row_index in range(sample_count):
            start, stop = features.indptr[row_index : row_index + 2]
            gene_indices = features.indices[start:stop]
            category_codes = features.data[start:stop].astype(np.int64, copy=False)
            event_count = len(gene_indices)
            burden[row_index, 0] = np.log1p(event_count)
            if event_count == 0:
                continue
            if event_count > self.mutation_sampling_size:
                selected = np.sort(
                    rng.choice(
                        event_count,
                        size=self.mutation_sampling_size,
                        replace=False,
                    )
                )
                gene_indices = gene_indices[selected]
                category_codes = category_codes[selected]
            length = len(gene_indices)
            gene_tokens = gene_indices.astype(np.int64) + 1
            # 부호 있는 ordinal code를 음이 아닌 정수로 일대일 변환합니다.
            zigzag = np.where(
                category_codes >= 0,
                category_codes * 2,
                -category_codes * 2 - 1,
            )
            # Python hash 대신 고정된 정수식을 사용해 실행 간 토큰을 재현합니다.
            variant_tokens = 1 + (
                (gene_indices.astype(np.int64) * 1_000_003 + zigzag * 9_176)
                % self.variant_hash_size
            )
            genes[row_index, :length] = gene_tokens
            variants[row_index, :length] = variant_tokens
        return genes, variants, burden

    def _resolve_device(self) -> torch.device:
        if self.device_name != "auto":
            return torch.device(self.device_name)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _make_loader(
        self,
        tokens: tuple[np.ndarray, np.ndarray, np.ndarray],
        labels: np.ndarray | None,
        *,
        shuffle: bool,
        seed_offset: int,
    ) -> DataLoader:
        tensors: list[torch.Tensor] = [
            torch.as_tensor(tokens[0], dtype=torch.long),
            torch.as_tensor(tokens[1], dtype=torch.long),
            torch.as_tensor(tokens[2], dtype=torch.float32),
        ]
        if labels is not None:
            tensors.append(torch.as_tensor(labels, dtype=torch.long))
        generator = torch.Generator().manual_seed(self.seed + seed_offset)
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            generator=generator,
        )

    def _class_weights(self, labels: np.ndarray) -> torch.Tensor | None:
        if self.class_weight is None:
            return None
        if self.class_weight == "balanced":
            counts = np.bincount(labels, minlength=len(self.classes_))
            values = len(labels) / (len(counts) * counts)
        elif isinstance(self.class_weight, (list, tuple)):
            values = np.asarray(self.class_weight, dtype=np.float32)
            if len(values) != len(self.classes_):
                raise ValueError("class_weight 길이는 클래스 수와 같아야 합니다.")
        else:
            raise ValueError("class_weight는 balanced, null 또는 숫자 목록이어야 합니다.")
        return torch.as_tensor(values, dtype=torch.float32, device=self.device_)

    def _validation_metrics(self, loader: DataLoader) -> tuple[float, float]:
        if self.model_ is None or self.device_ is None:
            raise RuntimeError("모델이 초기화되지 않았습니다.")
        self.model_.eval()
        criterion = nn.CrossEntropyLoss()
        losses: list[float] = []
        truths: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for genes, variants, burden, labels in loader:
                logits = self.model_(
                    genes.to(self.device_),
                    variants.to(self.device_),
                    burden.to(self.device_),
                )
                losses.append(criterion(logits, labels.to(self.device_)).item())
                truths.append(labels.numpy())
                predictions.append(logits.argmax(dim=1).cpu().numpy())
        macro_f1 = f1_score(
            np.concatenate(truths), np.concatenate(predictions), average="macro"
        )
        return float(np.mean(losses)), float(macro_f1)

    def fit(
        self,
        features,
        labels: np.ndarray,
        eval_set: list[tuple[object, np.ndarray]] | None = None,
        verbose: bool = True,
    ) -> "MuAtClassifier":
        """개별 유전자-단백질 변이 범주 token으로 암종 분류기를 학습합니다."""
        matrix = self._require_sparse_matrix(features)
        labels = np.asarray(labels)
        if len(labels) != matrix.shape[0]:
            raise ValueError("features와 labels의 행 수가 다릅니다.")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.classes_, encoded_labels = np.unique(labels, return_inverse=True)
        self.num_genes_ = matrix.shape[1]
        self.n_features_in_ = matrix.shape[1]
        self.device_ = self._resolve_device()
        self.model_ = MuAtProteinProxyNetwork(
            num_genes=self.num_genes_,
            variant_hash_size=self.variant_hash_size,
            gene_embedding_dim=self.gene_embedding_dim,
            variant_embedding_dim=self.variant_embedding_dim,
            model_dim=self.model_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            num_classes=len(self.classes_),
            dropout=self.dropout,
        ).to(self.device_)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        criterion = nn.CrossEntropyLoss(
            weight=self._class_weights(encoded_labels),
            label_smoothing=self.label_smoothing,
        )

        validation_matrix = None
        encoded_valid = None
        if eval_set:
            validation_matrix = self._require_sparse_matrix(eval_set[0][0])
            if validation_matrix.shape[1] != self.num_genes_:
                raise ValueError("학습과 검증 데이터의 유전자 피처 수가 다릅니다.")
            class_to_index = {label: index for index, label in enumerate(self.classes_)}
            try:
                encoded_valid = np.asarray(
                    [class_to_index[label] for label in np.asarray(eval_set[0][1])]
                )
            except KeyError as error:
                raise ValueError(f"검증 데이터에 학습에 없는 클래스가 있습니다: {error}") from error

        self.history_ = []
        best_score = -np.inf
        best_state = None
        stale_epochs = 0
        for epoch in range(self.epochs):
            train_tokens = self._tokenize(matrix, seed_offset=epoch)
            train_loader = self._make_loader(
                train_tokens,
                encoded_labels,
                shuffle=True,
                seed_offset=epoch,
            )
            self.model_.train()
            train_losses: list[float] = []
            for genes, variants, burden, batch_labels in train_loader:
                optimizer.zero_grad()
                logits = self.model_(
                    genes.to(self.device_),
                    variants.to(self.device_),
                    burden.to(self.device_),
                )
                loss = criterion(logits, batch_labels.to(self.device_))
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model_.parameters(), self.gradient_clip_norm
                )
                optimizer.step()
                train_losses.append(loss.item())

            record: dict[str, float | int] = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
            }
            if validation_matrix is not None and encoded_valid is not None:
                validation_loader = self._make_loader(
                    self._tokenize(validation_matrix, seed_offset=0),
                    encoded_valid,
                    shuffle=False,
                    seed_offset=0,
                )
                valid_loss, valid_f1 = self._validation_metrics(validation_loader)
                record["validation_loss"] = valid_loss
                record["validation_macro_f1"] = valid_f1
                monitored_score = valid_f1
            else:
                monitored_score = -record["train_loss"]
            self.history_.append(record)
            if verbose:
                print(f"[muat:protein_proxy] {record}")

            if monitored_score > best_score + 1e-12:
                best_score = monitored_score
                self.best_iteration = epoch
                best_state = deepcopy(
                    {
                        key: value.detach().cpu().clone()
                        for key, value in self.model_.state_dict().items()
                    }
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
            if (
                validation_matrix is not None
                and self.early_stopping_rounds is not None
                and stale_epochs >= int(self.early_stopping_rounds)
            ):
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
            self.model_.to(self.device_)
        return self

    def predict_proba(self, features) -> np.ndarray:
        """SUBCLASS별 예측 확률을 반환합니다."""
        if self.model_ is None or self.classes_ is None or self.device_ is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        matrix = self._require_sparse_matrix(features)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("학습과 예측 데이터의 유전자 피처 수가 다릅니다.")
        loader = self._make_loader(
            self._tokenize(matrix, seed_offset=0),
            labels=None,
            shuffle=False,
            seed_offset=0,
        )
        probabilities: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for genes, variants, burden in loader:
                logits = self.model_(
                    genes.to(self.device_),
                    variants.to(self.device_),
                    burden.to(self.device_),
                )
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probabilities)

    def predict(self, features) -> np.ndarray:
        """가장 높은 확률의 인코딩된 SUBCLASS를 반환합니다."""
        if self.classes_ is None:
            raise RuntimeError("예측 전에 모델을 fit해야 합니다.")
        return self.classes_[self.predict_proba(features).argmax(axis=1)]

    def get_params(self, deep: bool = True) -> dict[str, int]:
        del deep
        return {"n_estimators": self.epochs}


def create_model(model_config: dict, seed: int) -> MuAtClassifier:
    """현재 CSV용 MuAt protein proxy 분류기를 생성합니다."""
    return MuAtClassifier(model_config, seed)
