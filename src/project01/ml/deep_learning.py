from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - depends on optional extra
    raise ImportError(
        "Deep Learning yêu cầu PyTorch. Cài bằng: pip install -e '.[ml]'"
    ) from exc

from .preprocessing import PreparedData


@dataclass
class DeepLearningOutput:
    model: "TabularMLP"
    preprocessor: Any
    classes: list[Any]
    validation_score: float
    validation_metric: str
    best_epoch: int
    epochs_ran: int
    stopped_early: bool
    history: dict[str, list[float]]
    test_predictions: np.ndarray
    test_probabilities: np.ndarray

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.preprocessor.transform(frame)
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        values = np.asarray(matrix, dtype=np.float32)
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(values).to(device)
            return torch.softmax(self.model(tensor), dim=1).cpu().numpy()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(frame)
        return np.asarray(self.classes)[np.argmax(probabilities, axis=1)]


class TabularMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_layers: tuple[int, ...] = (64, 32),
        dropout: float = 0.20,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_layers:
            layers.extend([nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)])
            previous = hidden
        layers.append(nn.Linear(previous, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, values):
        return self.network(values)


def _dense_float32(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _encode_target(values, classes: list[Any]) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(classes)}
    return np.asarray([mapping[value] for value in np.asarray(values)], dtype=np.int64)


def _validation_metric(y_true, probabilities, classes, task: str) -> tuple[float, str]:
    from sklearn.metrics import f1_score, roc_auc_score

    predictions = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    if task == "binary":
        positive_index = len(classes) - 1
        positive = classes[positive_index]
        score = roc_auc_score(
            (np.asarray(y_true) == positive).astype(int), probabilities[:, positive_index]
        )
        return float(score), "roc_auc"
    return float(f1_score(y_true, predictions, average="macro", zero_division=0)), "f1_macro"


def train_deep_learning_model(
    prepared: PreparedData,
    hidden_layers: tuple[int, ...] = (64, 32),
    dropout: float = 0.20,
    learning_rate: float = 1e-3,
    batch_size: int = 4096,
    max_epochs: int = 100,
    patience: int = 10,
    min_delta: float = 1e-4,
    scheduler_patience: int = 3,
    random_state: int = 42,
) -> DeepLearningOutput:
    """Train an MLP with class weights, validation checkpointing and early stopping."""
    if max_epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("max_epochs, patience và batch_size phải lớn hơn 0.")
    classes = list(prepared.profile["classes"])
    if len(classes) < 2:
        raise ValueError("MLP cần ít nhất hai lớp.")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    validation_preprocessor = clone(prepared.preprocessor)
    X_train = _dense_float32(validation_preprocessor.fit_transform(prepared.X_train))
    X_validation = _dense_float32(validation_preprocessor.transform(prepared.X_validation))
    y_train = _encode_target(prepared.y_train, classes)
    y_validation = prepared.y_validation.to_numpy()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    counts = np.bincount(y_train, minlength=len(classes)).astype(np.float32)
    weights = len(y_train) / (len(classes) * np.maximum(counts, 1))
    model = TabularMLP(X_train.shape[1], len(classes), hidden_layers, dropout).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=scheduler_patience, min_lr=1e-5
    )

    history = {"train_loss": [], "validation_score": [], "learning_rate": []}
    best_score, best_epoch, best_state = -np.inf, 0, None
    stale_epochs = 0
    stopped_early = False

    validation_tensor = torch.from_numpy(X_validation).to(device)
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            validation_probabilities = torch.softmax(
                model(validation_tensor), dim=1
            ).cpu().numpy()
        score, metric = _validation_metric(
            y_validation, validation_probabilities, classes, prepared.profile["task_kind"]
        )
        history["train_loss"].append(float(np.mean(losses)))
        history["validation_score"].append(score)
        scheduler.step(score)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))

        if score > best_score + min_delta:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                stopped_early = True
                break

    if best_state is None:
        raise RuntimeError("Không tạo được checkpoint hợp lệ trong quá trình huấn luyện MLP.")
    model.load_state_dict(best_state)
    epochs_ran = len(history["train_loss"])

    # Refit preprocessing and network on train+validation for best_epoch only.
    final_preprocessor = clone(prepared.preprocessor)
    X_train_validation = _dense_float32(
        final_preprocessor.fit_transform(prepared.X_train_validation)
    )
    y_train_validation = _encode_target(prepared.y_train_validation, classes)
    torch.manual_seed(random_state)
    final_model = TabularMLP(
        X_train_validation.shape[1], len(classes), hidden_layers, dropout
    ).to(device)
    final_counts = np.bincount(y_train_validation, minlength=len(classes)).astype(np.float32)
    final_weights = len(y_train_validation) / (
        len(classes) * np.maximum(final_counts, 1)
    )
    final_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(final_weights, dtype=torch.float32, device=device)
    )
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    final_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train_validation),
            torch.from_numpy(y_train_validation),
        ),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    for _ in range(best_epoch):
        final_model.train()
        for batch_x, batch_y in final_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            final_optimizer.zero_grad(set_to_none=True)
            loss = final_criterion(final_model(batch_x), batch_y)
            loss.backward()
            final_optimizer.step()

    output = DeepLearningOutput(
        model=final_model,
        preprocessor=final_preprocessor,
        classes=classes,
        validation_score=float(best_score),
        validation_metric=metric,
        best_epoch=best_epoch,
        epochs_ran=epochs_ran,
        stopped_early=stopped_early,
        history=history,
        test_predictions=np.array([], dtype=object),
        test_probabilities=np.empty((0, len(classes))),
    )
    output.test_probabilities = output.predict_proba(prepared.X_test)
    output.test_predictions = np.asarray(output.classes)[
        np.argmax(output.test_probabilities, axis=1)
    ]
    return output
