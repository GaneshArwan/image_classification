"""End-to-end training, evaluation, CLI, app fallback, and checkpoint tests."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from app import _resolve_model
from src.data import generate_synthetic_leaf_dataset, load_dataloaders
from src.evaluate import evaluate_model
from src.i18n import Translator
from src.models import build_model
from src.schemas import TrainConfig
from src.train import (
    load_checkpoint_safely,
    main as train_main,
    predict_with_gradcam,
    train_single_run,
)


def test_load_dataloaders_rejects_insufficient_samples(
    tmp_path: Path,
) -> None:
    data_dir = generate_synthetic_leaf_dataset(
        output_dir=tmp_path / "tiny",
        samples_per_class=4,
        image_size=64,
        seed=3,
    )
    with pytest.raises(ValueError, match="at least 7 samples"):
        load_dataloaders(data_dir=data_dir, image_size=64, batch_size=4)


def test_evaluate_model_rejects_empty_dataloader() -> None:
    empty_ds = TensorDataset(
        torch.empty(0, 3, 64, 64), torch.empty(0, dtype=torch.long)
    )
    loader = DataLoader(empty_ds, batch_size=4)
    model = build_model("baseline_cnn", num_classes=2, pretrained=False)
    with pytest.raises(ValueError, match="empty dataloader"):
        evaluate_model(
            model=model,
            dataloader=loader,
            class_names=["A", "B"],
            device=torch.device("cpu"),
            architecture="baseline_cnn",
            augmented=False,
        )


def test_end_to_end_training_and_safe_reload(tmp_path: Path) -> None:
    data_dir = generate_synthetic_leaf_dataset(
        output_dir=tmp_path / "data",
        samples_per_class=12,
        image_size=64,
        seed=7,
    )
    train_loader, val_loader, test_loader, classes = load_dataloaders(
        data_dir=data_dir,
        image_size=64,
        batch_size=8,
        augment=True,
        seed=7,
    )
    assert len(classes) == 3
    assert len(train_loader.dataset) > len(val_loader.dataset)  # type: ignore[arg-type]
    assert len(test_loader.dataset) >= 3  # type: ignore[arg-type]

    out_dir = tmp_path / "artifacts"
    cfg = TrainConfig(
        data_dir=data_dir,
        output_dir=out_dir,
        architecture="baseline_cnn",
        epochs=1,
        batch_size=8,
        image_size=64,
        use_augmentation=True,
        pretrained=False,
        seed=7,
    )
    report = train_single_run(cfg, Translator("en"))
    assert 0.0 <= report.accuracy <= 1.0
    assert report.num_samples == len(test_loader.dataset)  # type: ignore[arg-type]

    ckpt_dir = out_dir / "baseline_cnn_aug"
    model, meta = load_checkpoint_safely(ckpt_dir, torch.device("cpu"))
    assert meta.architecture == "baseline_cnn"

    sample_img = Image.new("RGB", (64, 64), color=(35, 160, 50))
    pred, cam_img = predict_with_gradcam(
        model=model,
        architecture="baseline_cnn",
        image=sample_img,
        class_names=meta.class_names,
        image_size=meta.image_size,
        device=torch.device("cpu"),
    )
    assert len(pred.predictions) == 3
    assert cam_img.size == (64, 64)

    # Tampered metadata fails Pydantic validation.
    (ckpt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "architecture": "baseline_cnn",
                "class_names": ["OnlyOne"],
                "image_size": 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_checkpoint_safely(ckpt_dir, torch.device("cpu"))


def test_cli_synthetic_demo_and_app_demo_fallback(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_out"
    rc = train_main(
        [
            "--synthetic-demo",
            "--architecture",
            "baseline_cnn",
            "--epochs",
            "1",
            "--image-size",
            "64",
            "--no-pretrained",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "baseline_cnn_aug" / "model_state.pt").is_file()

    # App resolves saved checkpoint cleanly and falls back to demo mode when absent.
    _resolve_model.clear()
    _, arch, _, image_size, is_demo = _resolve_model(
        str(out_dir / "baseline_cnn_aug"), "cpu"
    )
    assert arch == "baseline_cnn"
    assert image_size == 64
    assert is_demo is False

    _, demo_arch, _, _, fallback_demo = _resolve_model(
        str(tmp_path / "missing_ckpt"), "cpu"
    )
    assert demo_arch == "resnet18"
    assert fallback_demo is True
