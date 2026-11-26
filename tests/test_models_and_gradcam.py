"""Tests for model architectures and Grad-CAM attribution."""

from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import torch

from src.gradcam import GradCAM, overlay_heatmap
from src.models import build_model, get_target_conv_layer
from src.schemas import ArchitectureName


@pytest.mark.parametrize(
    "arch", ["baseline_cnn", "resnet18", "efficientnet_b0"]
)
def test_build_model_and_gradcam(arch: ArchitectureName) -> None:
    model = build_model(
        architecture=arch,
        num_classes=3,
        pretrained=False,
        freeze_backbone=True,
    )
    x = torch.randn(2, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (2, 3)

    target_layer = get_target_conv_layer(model, arch)
    with GradCAM(model, target_layer) as cam:
        with torch.no_grad():
            heatmap = cam.generate(x[:1])
        with pytest.raises(ValueError, match="range"):
            cam.generate(x[:1], class_idx=-1)
        with pytest.raises(ValueError, match="range"):
            cam.generate(x[:1], class_idx=99)

    assert heatmap.shape == (64, 64)
    assert heatmap.dtype == np.float32
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0
    assert len(target_layer._forward_hooks) == 0

    base_img = Image.new("RGB", (64, 64), color=(50, 150, 50))
    blended = overlay_heatmap(base_img, heatmap)
    assert blended.mode == "RGB"
    assert blended.size == (64, 64)
