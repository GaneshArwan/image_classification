"""Grad-CAM visual attribution and heatmap overlay utilities."""

from __future__ import annotations

import matplotlib
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle

matplotlib.use("Agg")


class GradCAM:
    """Gradient-weighted Class Activation Mapping with automatic hook cleanup."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._grad_handle: RemovableHandle | None = None
        self._fwd_handle = target_layer.register_forward_hook(
            self._save_activation
        )

    def _save_activation(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self._activations = output.detach()
        if self._grad_handle is not None:
            self._grad_handle.remove()
            self._grad_handle = None
        if output.requires_grad:
            self._grad_handle = output.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self._gradients = grad.detach()

    def close(self) -> None:
        """Remove registered forward and tensor gradient hooks."""
        self._fwd_handle.remove()
        if self._grad_handle is not None:
            self._grad_handle.remove()
            self._grad_handle = None

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> np.ndarray:
        """Compute a normalized [0, 1] Grad-CAM heatmap for a single image tensor.

        Args:
            input_tensor: Tensor of shape ``(1, C, H, W)``.
            class_idx: Target class index; defaults to top predicted class.

        Returns:
            2D float32 numpy array with values in ``[0.0, 1.0]``.

        Raises:
            ValueError: If input shape is invalid or ``class_idx`` is out of range.
        """
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                "Expected input shape (1, C, H, W), "
                f"got {tuple(input_tensor.shape)}."
            )

        was_training = self.model.training
        self.model.eval()
        self._activations = None
        self._gradients = None
        try:
            with torch.enable_grad():
                self.model.zero_grad(set_to_none=True)
                work_tensor = input_tensor.detach().requires_grad_(True)
                logits = self.model(work_tensor)
                num_classes = int(logits.shape[1])
                target_idx = (
                    int(logits.argmax(dim=1).item())
                    if class_idx is None
                    else int(class_idx)
                )
                if not 0 <= target_idx < num_classes:
                    raise ValueError(
                        f"class_idx {target_idx} out of range [0, {num_classes})."
                    )
                score = logits[0, target_idx]
                score.backward()

            if self._activations is None or self._gradients is None:
                raise RuntimeError("Grad-CAM hooks did not capture activations.")

            weights = self._gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self._activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(
                cam,
                size=work_tensor.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            cam_np = cam[0, 0].cpu().numpy().astype(np.float32)
            cam_np = np.nan_to_num(cam_np, nan=0.0, posinf=1.0, neginf=0.0)
            max_val = float(cam_np.max())
            if max_val > 1e-8:
                cam_np = cam_np / max_val
            else:
                cam_np = np.zeros_like(cam_np, dtype=np.float32)
            return np.clip(cam_np, 0.0, 1.0)
        finally:
            if self._grad_handle is not None:
                self._grad_handle.remove()
                self._grad_handle = None
            self.model.train(was_training)


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Blend a normalized Grad-CAM heatmap over an RGB image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be within [0, 1], got {alpha}.")
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape {heatmap.shape}.")
    safe_map = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
    rgb_image = image.convert("RGB")
    cmap = matplotlib.colormaps["jet"]
    colored = (cmap(np.clip(safe_map, 0.0, 1.0))[..., :3] * 255).astype(
        np.uint8
    )
    heatmap_img = Image.fromarray(colored, mode="RGB").resize(
        rgb_image.size, resample=Image.Resampling.BILINEAR
    )
    return Image.blend(rgb_image, heatmap_img, alpha=alpha)
