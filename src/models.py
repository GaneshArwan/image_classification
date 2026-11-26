"""Model architectures for baseline CNN and transfer learning backbones."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torchvision import models

from src.schemas import ArchitectureName


class BaselineCNN(nn.Module):
    """Lightweight 3-block convolutional baseline trained from scratch."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for an input batch."""
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def _freeze_backbone_params(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def build_model(
    architecture: ArchitectureName,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Instantiate a baseline CNN or torchvision transfer-learning model.

    Args:
        architecture: One of ``baseline_cnn``, ``resnet18``, ``efficientnet_b0``.
        num_classes: Number of target output classes.
        pretrained: Whether to load ImageNet pretrained backbone weights.
        freeze_backbone: Whether to freeze backbone parameters before head swap.

    Returns:
        Configured ``torch.nn.Module``.

    Raises:
        ValueError: If ``num_classes < 2`` or ``architecture`` is unsupported.
        RuntimeError: If downloading pretrained weights fails in offline mode.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}.")

    if architecture == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)

    try:
        if architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            if freeze_backbone:
                _freeze_backbone_params(model)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            return model

        if architecture == "efficientnet_b0":
            weights = (
                models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            model = models.efficientnet_b0(weights=weights)
            if freeze_backbone:
                _freeze_backbone_params(model)
            head = cast(nn.Linear, model.classifier[1])
            model.classifier[1] = nn.Linear(head.in_features, num_classes)
            return model
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load pretrained weights for '{architecture}'. "
            "Check network access or pass --no-pretrained."
        ) from exc

    raise ValueError(f"Unsupported architecture: {architecture}")


def get_target_conv_layer(
    model: nn.Module,
    architecture: ArchitectureName,
) -> nn.Module:
    """Return the final convolutional block used for Grad-CAM."""
    if architecture in ("baseline_cnn", "efficientnet_b0"):
        features = getattr(model, "features", None)
        if isinstance(features, nn.Sequential) and len(features) > 0:
            return features[-1]
    elif architecture == "resnet18":
        layer4 = getattr(model, "layer4", None)
        if isinstance(layer4, nn.Sequential) and len(layer4) > 0:
            return layer4[-1]
    raise ValueError(
        f"Cannot resolve target conv layer for architecture '{architecture}'."
    )
