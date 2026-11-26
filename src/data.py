"""Dataset preparation, stratified splitting, and augmentation transforms."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_SYNTHETIC_CLASSES = (
    "Tomato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Yellow_Leaf_Curl",
)


class TransformSubset(Dataset[tuple[Any, int]]):
    """Subset wrapper that applies an independent transform per split."""

    def __init__(
        self,
        base_dataset: datasets.ImageFolder,
        indices: list[int],
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def sample_path(self, idx: int) -> Path:
        """Return the filesystem path for a subset item."""
        path, _ = self.base_dataset.samples[self.indices[idx]]
        return Path(path)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        path, target = self.base_dataset.samples[self.indices[idx]]
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
        result: Any = (
            self.transform(rgb_img) if self.transform is not None else rgb_img
        )
        return result, target


def build_transforms(
    image_size: int = 224,
    augment: bool = True,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Build training and evaluation torchvision transform pipelines.

    Args:
        image_size: Target spatial height and width.
        augment: Whether to apply training data augmentation.

    Returns:
        Tuple of ``(train_transform, eval_transform)``.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.15
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = eval_transform

    return train_transform, eval_transform


def generate_synthetic_leaf_dataset(
    output_dir: Path,
    classes: tuple[str, ...] = DEFAULT_SYNTHETIC_CLASSES,
    samples_per_class: int = 18,
    image_size: int = 64,
    seed: int = 42,
) -> Path:
    """Generate a deterministic synthetic leaf dataset for local verification.

    Args:
        output_dir: Destination folder containing per-class subdirectories.
        classes: Tuple of class directory names.
        samples_per_class: Number of images generated per class.
        image_size: Width and height in pixels (>= 16).
        seed: Random seed for reproducible textures.

    Returns:
        Path to the generated dataset root.

    Raises:
        ValueError: If ``image_size < 16`` or ``samples_per_class < 3``.
    """
    if image_size < 16:
        raise ValueError(f"image_size must be >= 16, got {image_size}.")
    if samples_per_class < 3:
        raise ValueError(
            f"samples_per_class must be >= 3, got {samples_per_class}."
        )

    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_idx, class_name in enumerate(classes):
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for sample_idx in range(samples_per_class):
            bg_val = int(rng.integers(210, 240))
            canvas = np.full(
                (image_size, image_size, 3), bg_val, dtype=np.uint8
            )
            img = Image.fromarray(canvas, mode="RGB")
            draw = ImageDraw.Draw(img)

            margin = max(6, image_size // 8)
            bbox = (
                margin + int(rng.integers(-2, 3)),
                margin + int(rng.integers(-2, 3)),
                image_size - margin + int(rng.integers(-2, 3)),
                image_size - margin + int(rng.integers(-2, 3)),
            )
            if class_idx == 0:
                leaf_color = (34, int(rng.integers(135, 175)), 48)
            elif class_idx == 1:
                leaf_color = (62, int(rng.integers(120, 150)), 44)
            else:
                leaf_color = (
                    int(rng.integers(185, 225)),
                    int(rng.integers(175, 210)),
                    45,
                )

            draw.ellipse(bbox, fill=leaf_color)
            draw.line(
                [(image_size // 2, bbox[1]), (image_size // 2, bbox[3])],
                fill=(45, 95, 35),
                width=max(1, image_size // 32),
            )

            low_x = bbox[0] + 4
            high_x = max(low_x + 1, bbox[2] - 4)
            low_y = bbox[1] + 4
            high_y = max(low_y + 1, bbox[3] - 4)
            if class_idx == 1:
                for _ in range(5):
                    cx = int(rng.integers(low_x, high_x))
                    cy = int(rng.integers(low_y, high_y))
                    r = max(2, image_size // 16)
                    draw.ellipse(
                        (cx - r, cy - r, cx + r, cy + r), fill=(85, 48, 28)
                    )
            elif class_idx == 2:
                for _ in range(4):
                    cx = int(rng.integers(low_x, high_x))
                    cy = int(rng.integers(low_y, high_y))
                    r = max(2, image_size // 14)
                    draw.ellipse(
                        (cx - r, cy - r, cx + r, cy + r), fill=(235, 220, 60)
                    )

            img.save(class_dir / f"sample_{sample_idx:03d}.png", format="PNG")

    return output_dir


def _stratified_split_indices(
    targets: list[int],
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Split dataset indices into stratified 70% train, 15% val, 15% test."""
    indices = np.arange(len(targets))
    y = np.asarray(targets)

    split_outer = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=seed
    )
    train_pos, temp_pos = next(split_outer.split(indices, y))
    train_idx = indices[train_pos].tolist()
    temp_idx = indices[temp_pos]
    temp_y = y[temp_pos]

    split_inner = StratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=seed
    )
    val_pos, test_pos = next(split_inner.split(temp_idx, temp_y))
    val_idx = temp_idx[val_pos].tolist()
    test_idx = temp_idx[test_pos].tolist()
    return train_idx, val_idx, test_idx


def load_dataloaders(
    data_dir: Path,
    image_size: int = 224,
    batch_size: int = 16,
    augment: bool = True,
    seed: int = 42,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], list[str]]:
    """Load stratified train/val/test DataLoaders from an ImageFolder layout.

    Args:
        data_dir: Root directory containing per-class image folders.
        image_size: Spatial resize dimension.
        batch_size: Mini-batch size.
        augment: Whether to apply training augmentations.
        seed: Random seed for reproducible splitting and shuffling.

    Returns:
        Tuple ``(train_loader, val_loader, test_loader, class_names)``.

    Raises:
        ValueError: If fewer than 2 classes or insufficient per-class samples
            exist for a 70/15/15 stratified split.
    """
    base_dataset = datasets.ImageFolder(root=str(data_dir))
    num_classes = len(base_dataset.classes)
    if num_classes < 2:
        raise ValueError(
            f"Expected at least 2 classes in '{data_dir}', found "
            f"{num_classes}."
        )

    class_counts = Counter(base_dataset.targets)
    min_class_count = min(class_counts.values())
    temp_total = int(np.ceil(len(base_dataset) * 0.30))
    if min_class_count < 7 or (temp_total // 2) < num_classes:
        raise ValueError(
            f"Dataset in '{data_dir}' needs at least 7 samples per class "
            f"(min found: {min_class_count}) for a 70/15/15 stratified split."
        )

    train_idx, val_idx, test_idx = _stratified_split_indices(
        base_dataset.targets, seed=seed
    )
    train_tf, eval_tf = build_transforms(image_size=image_size, augment=augment)

    train_ds = TransformSubset(base_dataset, train_idx, transform=train_tf)
    val_ds = TransformSubset(base_dataset, val_idx, transform=eval_tf)
    test_ds = TransformSubset(base_dataset, test_idx, transform=eval_tf)

    generator = torch.Generator().manual_seed(seed)
    drop_last = len(train_ds) > batch_size and (len(train_ds) % batch_size == 1)
    train_loader: DataLoader[Any] = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        generator=generator,
        num_workers=0,
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader: DataLoader[Any] = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader, list(base_dataset.classes)
