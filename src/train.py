"""Training CLI, augmentation comparison, and safe checkpoint persistence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import (
    TransformSubset,
    build_transforms,
    generate_synthetic_leaf_dataset,
    load_dataloaders,
)
from src.evaluate import evaluate_model, save_confusion_matrix
from src.gradcam import GradCAM, overlay_heatmap
from src.i18n import Translator
from src.models import build_model, get_target_conv_layer
from src.schemas import (
    LOW_CONFIDENCE_THRESHOLD,
    ArchitectureName,
    CheckpointMetadata,
    EvaluationReport,
    PredictionItem,
    PredictionResult,
    TrainConfig,
    validate_safe_path,
)


def set_seed(seed: int) -> None:
    """Set deterministic RNG seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy model parameters to CPU tensors to avoid VRAM duplication."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if is_train and optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += batch_size

    return total_loss / max(1, total), correct / max(1, total)


def save_checkpoint_safely(
    model: nn.Module,
    checkpoint_dir: Path,
    architecture: ArchitectureName,
    class_names: list[str],
    image_size: int,
    report: EvaluationReport,
) -> tuple[Path, Path]:
    """Save weights as a pure state_dict plus JSON metadata."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_path = checkpoint_dir / "model_state.pt"
    metadata_path = checkpoint_dir / "metadata.json"

    torch.save(model.state_dict(), weights_path)
    validated_meta = CheckpointMetadata(
        architecture=architecture,
        class_names=class_names,
        image_size=image_size,
    )
    metadata = {
        **validated_meta.model_dump(),
        "evaluation": report.model_dump(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return weights_path, metadata_path


def load_checkpoint_safely(
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, CheckpointMetadata]:
    """Load model state_dict safely with weights_only=True and schema validation."""
    safe_dir = validate_safe_path(checkpoint_dir)
    weights_path = safe_dir / "model_state.pt"
    metadata_path = safe_dir / "metadata.json"
    if not weights_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing model_state.pt or metadata.json in '{safe_dir}'."
        )

    meta = CheckpointMetadata.model_validate_json(
        metadata_path.read_text(encoding="utf-8")
    )
    model = build_model(
        architecture=meta.architecture,
        num_classes=len(meta.class_names),
        pretrained=False,
    )
    state_dict = torch.load(
        weights_path, map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, meta


def predict_with_gradcam(
    model: nn.Module,
    architecture: ArchitectureName,
    image: Image.Image,
    class_names: list[str],
    image_size: int,
    device: torch.device,
    top_k: int = 3,
) -> tuple[PredictionResult, Image.Image]:
    """Run top-K inference and generate a Grad-CAM overlay for an RGB image."""
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")

    _, eval_tf = build_transforms(image_size=image_size, augment=False)
    tensor = eval_tf(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    if probs.shape[0] != len(class_names):
        raise ValueError(
            f"Model output classes ({probs.shape[0]}) do not match "
            f"class_names ({len(class_names)})."
        )

    k = min(top_k, len(class_names))
    top_probs, top_indices = torch.topk(probs, k=k)
    items = [
        PredictionItem(
            label=class_names[int(idx)],
            probability=float(prob),
        )
        for prob, idx in zip(top_probs, top_indices)
    ]

    top_item = items[0]
    result = PredictionResult(
        predictions=items,
        top_label=top_item.label,
        top_probability=top_item.probability,
        low_confidence=top_item.probability < LOW_CONFIDENCE_THRESHOLD,
    )

    target_layer = get_target_conv_layer(model, architecture)
    with GradCAM(model, target_layer) as cam:
        heatmap = cam.generate(tensor, class_idx=int(top_indices[0]))
    overlay = overlay_heatmap(image, heatmap)
    return result, overlay


def train_single_run(
    config: TrainConfig,
    translator: Translator,
) -> EvaluationReport:
    """Train one model configuration, evaluate on test split, and save artifacts."""
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, class_names = load_dataloaders(
        data_dir=config.data_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        augment=config.use_augmentation,
        seed=config.seed,
    )

    model = build_model(
        architecture=config.architecture,
        num_classes=len(class_names),
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=config.learning_rate, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    print(
        translator.t(
            "train_start",
            arch=config.architecture,
            aug=config.use_augmentation,
            epochs=config.epochs,
            device=str(device),
        )
    )

    best_val_loss = float("inf")
    best_state = _cpu_state_dict(model)

    for epoch in range(1, config.epochs + 1):
        train_loss, _ = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)
        if math.isfinite(val_loss) and val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_state = _cpu_state_dict(model)
        print(
            translator.t(
                "epoch_log",
                epoch=epoch,
                epochs=config.epochs,
                train_loss=translator.dec(train_loss),
                val_loss=translator.dec(val_loss),
                val_acc=translator.pct(val_acc),
            )
        )

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    report = evaluate_model(
        model=model,
        dataloader=test_loader,
        class_names=class_names,
        device=device,
        architecture=config.architecture,
        augmented=config.use_augmentation,
    )

    aug_tag = "aug" if config.use_augmentation else "no_aug"
    run_dir = config.output_dir / f"{config.architecture}_{aug_tag}"
    save_checkpoint_safely(
        model=model,
        checkpoint_dir=run_dir,
        architecture=config.architecture,
        class_names=class_names,
        image_size=config.image_size,
        report=report,
    )
    save_confusion_matrix(
        cm=report.confusion_matrix,
        class_names=class_names,
        output_path=run_dir / "confusion_matrix.png",
    )

    if isinstance(test_loader.dataset, TransformSubset) and len(test_loader.dataset) > 0:
        sample_path = test_loader.dataset.sample_path(0)
        with Image.open(sample_path) as sample_img:
            _, cam_overlay = predict_with_gradcam(
                model=model,
                architecture=config.architecture,
                image=sample_img.convert("RGB"),
                class_names=class_names,
                image_size=config.image_size,
                device=device,
            )
            cam_overlay.save(run_dir / "gradcam_sample.png", format="PNG")

    auc_str = (
        translator.dec(report.macro_roc_auc)
        if report.macro_roc_auc is not None
        else "N/A"
    )
    print(
        translator.t(
            "eval_summary",
            arch=report.architecture,
            aug=report.augmented,
            acc=translator.pct(report.accuracy),
            f1=translator.dec(report.macro_f1),
            auc=auc_str,
            samples=report.num_samples,
        )
    )
    print(translator.t("saved_artifacts", path=str(run_dir)))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for training and comparison runs."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate plant disease image classifiers."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/plantvillage"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--architecture",
        choices=("baseline_cnn", "resnet18", "efficientnet_b0"),
        default="resnet18",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--locale", choices=("en", "id"), default="en")
    parser.add_argument("--synthetic-demo", action="store_true")
    parser.add_argument("--compare-all", action="store_true")
    return parser.parse_args(argv)


def _build_train_config(
    args: argparse.Namespace,
    data_dir: Path,
    output_dir: Path,
    architecture: ArchitectureName,
    use_augmentation: bool,
) -> TrainConfig:
    return TrainConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        architecture=architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        use_augmentation=use_augmentation,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        seed=args.seed,
        locale=args.locale,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    output_dir = validate_safe_path(args.output_dir)
    data_dir = validate_safe_path(args.data_dir)
    # Validate all scalar hyperparameters before any disk/dataset generation.
    base_cfg = _build_train_config(
        args,
        data_dir=data_dir,
        output_dir=output_dir,
        architecture=args.architecture,
        use_augmentation=not args.no_augment,
    )

    if args.synthetic_demo:
        data_dir = generate_synthetic_leaf_dataset(
            output_dir=output_dir / "synthetic_data",
            image_size=base_cfg.image_size,
            seed=base_cfg.seed,
        )

    translator = Translator(locale=base_cfg.locale)
    if args.compare_all:
        runs: list[tuple[ArchitectureName, bool]] = [
            ("baseline_cnn", False),
            ("baseline_cnn", True),
            ("resnet18", False),
            ("resnet18", True),
        ]
        reports = [
            train_single_run(
                _build_train_config(
                    args,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    architecture=arch,
                    use_augmentation=aug,
                ),
                translator,
            ).model_dump()
            for arch, aug in runs
        ]
        summary_path = output_dir / "comparison_summary.json"
        summary_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        return 0

    train_single_run(
        base_cfg.model_copy(update={"data_dir": data_dir}), translator
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
