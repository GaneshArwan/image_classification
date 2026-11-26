"""Model evaluation metrics and confusion matrix visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch
from torch.utils.data import DataLoader

from src.schemas import EvaluationReport


def _compute_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> float | None:
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        return None
    try:
        if num_classes == 2:
            return float(roc_auc_score(y_true, y_prob[:, 1]))
        if len(unique_labels) == num_classes:
            return float(
                roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
            )
        return None
    except ValueError:
        return None


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader[Any],
    class_names: list[str],
    device: torch.device,
    architecture: str,
    augmented: bool,
) -> EvaluationReport:
    """Evaluate a trained classifier on a test/validation DataLoader."""
    model.eval()
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.numpy().tolist())

    if not all_probs:
        raise ValueError("Cannot evaluate on an empty dataloader.")

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_preds)
    y_prob = np.vstack(all_probs)
    if y_prob.shape[1] != len(class_names):
        raise ValueError(
            f"Model output classes ({y_prob.shape[1]}) do not match "
            f"class_names length ({len(class_names)})."
        )

    label_indices = list(range(len(class_names)))
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_indices,
        average=None,
        zero_division=0,
    )
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(per_class_f1))
    macro_roc_auc = _compute_roc_auc(y_true, y_prob, len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=label_indices).tolist()

    return EvaluationReport(
        architecture=architecture,
        augmented=augmented,
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        macro_roc_auc=macro_roc_auc,
        per_class_f1={
            name: float(score) for name, score in zip(class_names, per_class_f1)
        },
        confusion_matrix=cm,
        num_samples=len(y_true),
    )


def save_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str],
    output_path: Path,
) -> Path:
    """Render and save a labeled confusion matrix PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_size = max(5, min(14, len(class_names) + 3))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
    sns.heatmap(
        np.asarray(cm),
        annot=len(class_names) <= 15,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
