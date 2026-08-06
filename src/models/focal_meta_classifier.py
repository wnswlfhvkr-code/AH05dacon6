"""OOF stacking 확률을 multiclass focal loss로 결합하는 규제형 메타 분류기."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


class FocalMetaClassifier:
    """희소 클래스와 어려운 OOF 표본에 집중하는 작은 PyTorch MLP입니다."""

    def __init__(self, parameters: dict, seed: int) -> None:
        self.seed = int(seed)
        self.hidden_dim = int(parameters.get("hidden_dim", 48))
        self.dropout = float(parameters.get("dropout", 0.30))
        self.gamma = float(parameters.get("focal_gamma", 1.5))
        self.alpha_beta = float(parameters.get("alpha_effective_beta", 0.99))
        alpha_clip = parameters.get("alpha_clip", (0.75, 2.25))
        self.alpha_clip = (float(alpha_clip[0]), float(alpha_clip[1]))
        self.learning_rate = float(parameters.get("learning_rate", 7e-4))
        self.weight_decay = float(parameters.get("weight_decay", 1.5e-2))
        self.batch_size = int(parameters.get("batch_size", 256))
        self.max_epochs = int(parameters.get("max_epochs", 300))
        self.patience = int(parameters.get("patience", 30))
        self.validation_fraction = float(parameters.get("validation_fraction", 0.15))
        self.gradient_clip_norm = float(parameters.get("gradient_clip_norm", 1.0))
        self.minimum_delta = float(parameters.get("minimum_delta", 1e-5))
        self.device_name = str(parameters.get("device", "cpu"))
        if self.hidden_dim < 1 or not 0 <= self.dropout < 1:
            raise ValueError("hidden_dim은 1 이상, dropout은 0 이상 1 미만이어야 합니다.")
        if self.gamma < 0 or not 0 < self.alpha_beta < 1:
            raise ValueError("focal_gamma는 음수가 아니고 alpha_effective_beta는 0~1이어야 합니다.")
        if (
            len(self.alpha_clip) != 2
            or self.alpha_clip[0] <= 0
            or self.alpha_clip[0] > self.alpha_clip[1]
        ):
            raise ValueError("alpha_clip은 양수 [최솟값, 최댓값]이어야 합니다.")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, max_epochs, patience는 1 이상이어야 합니다.")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction은 0보다 크고 0.5보다 작아야 합니다.")

        self.classes_: np.ndarray | None = None
        self.model_ = None
        self.alpha_: np.ndarray | None = None
        self.input_features_ = 0
        self.best_epoch_ = 0
        self.best_validation_loss_: float | None = None

    @staticmethod
    def _torch():
        try:
            import torch
        except ImportError as error:
            raise ImportError(
                "Focal meta learner에는 PyTorch가 필요합니다. "
                "`pip install -r requirements.txt`를 실행하세요."
            ) from error
        return torch

    def _device(self, torch):
        if self.device_name == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device_name)

    def _alpha(self, encoded_labels: np.ndarray, class_count: int) -> np.ndarray:
        counts = np.bincount(encoded_labels, minlength=class_count).astype("float64")
        raw = (1.0 - self.alpha_beta) / (
            1.0 - np.power(self.alpha_beta, counts)
        )
        raw /= np.average(raw, weights=counts)
        return np.clip(raw, *self.alpha_clip).astype("float32")

    def fit(self, features, labels):
        torch = self._torch()
        values = np.asarray(features, dtype="float32")
        y = np.asarray(labels)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Focal meta 입력은 유한한 2차원 수치 행렬이어야 합니다.")
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.input_features_ = values.shape[1]
        class_count = len(self.classes_)
        self.alpha_ = self._alpha(encoded, class_count)
        train_index, valid_index = train_test_split(
            np.arange(len(y)),
            test_size=self.validation_fraction,
            random_state=self.seed,
            stratify=encoded,
        )

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        device = self._device(torch)
        model = torch.nn.Sequential(
            torch.nn.Linear(self.input_features_, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_dim, class_count),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        alpha_tensor = torch.as_tensor(self.alpha_, dtype=torch.float32, device=device)
        x_train = torch.as_tensor(values[train_index], dtype=torch.float32)
        y_train = torch.as_tensor(encoded[train_index], dtype=torch.long)
        x_valid = torch.as_tensor(values[valid_index], dtype=torch.float32, device=device)
        y_valid = torch.as_tensor(encoded[valid_index], dtype=torch.long, device=device)
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        generator = torch.Generator().manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )

        def focal_loss(logits, targets):
            log_probabilities = torch.nn.functional.log_softmax(logits, dim=1)
            target_log_probability = log_probabilities.gather(
                1, targets[:, None]
            ).squeeze(1)
            target_probability = target_log_probability.exp()
            return (
                -alpha_tensor[targets]
                * torch.pow(1.0 - target_probability, self.gamma)
                * target_log_probability
            ).mean()

        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = focal_loss(model(batch_features), batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.gradient_clip_norm
                )
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    focal_loss(model(x_valid), y_valid).detach().cpu()
                )
            if validation_loss < best_loss - self.minimum_delta:
                best_loss = validation_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                self.best_epoch_ = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("Focal meta learner가 유효한 상태를 학습하지 못했습니다.")
        model.load_state_dict(best_state)
        self.model_ = model.cpu().eval()
        self.best_validation_loss_ = best_loss
        return self

    def predict_proba(self, features) -> np.ndarray:
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("예측 전에 Focal meta learner를 fit해야 합니다.")
        torch = self._torch()
        values = np.asarray(features, dtype="float32")
        if values.ndim != 2 or values.shape[1] != self.input_features_:
            raise ValueError("Focal meta 학습과 예측의 피처 수가 다릅니다.")
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(values), self.batch_size):
                batch = torch.as_tensor(
                    values[start : start + self.batch_size], dtype=torch.float32
                )
                probabilities = torch.softmax(self.model_(batch), dim=1)
                outputs.append(probabilities.cpu().numpy())
        return np.vstack(outputs).astype("float64", copy=False)

    def predict(self, features) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("예측 전에 Focal meta learner를 fit해야 합니다.")
        return self.classes_[self.predict_proba(features).argmax(axis=1)]

