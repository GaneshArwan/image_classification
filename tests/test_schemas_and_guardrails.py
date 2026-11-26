"""Tests for Pydantic schemas, path/image guardrails, and Babel i18n."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest

from src.i18n import Translator
from src.schemas import (
    CheckpointMetadata,
    TrainConfig,
    validate_image_bytes,
    validate_safe_path,
)


def _make_image_bytes(
    fmt: str = "PNG",
    size: tuple[int, int] = (32, 32),
) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color=(40, 140, 60)).save(buf, format=fmt)
    return buf.getvalue()


def test_train_config_defaults_and_bounds() -> None:
    cfg = TrainConfig()
    assert cfg.architecture == "resnet18"
    assert cfg.epochs == 5
    with pytest.raises(ValueError):
        TrainConfig(epochs=0)
    with pytest.raises(ValueError):
        TrainConfig(learning_rate=2.0)
    with pytest.raises(ValueError):
        TrainConfig(image_size=32)


def test_checkpoint_metadata_validation() -> None:
    meta = CheckpointMetadata(
        architecture="resnet18",
        class_names=["Healthy", "Diseased"],
        image_size=224,
    )
    assert meta.architecture == "resnet18"
    with pytest.raises(ValueError):
        CheckpointMetadata(
            architecture="resnet18",
            class_names=["OnlyOne"],
            image_size=224,
        )
    with pytest.raises(ValueError):
        CheckpointMetadata(
            architecture="resnet18",
            class_names=["A", " "],
            image_size=224,
        )


def test_validate_safe_path_blocks_empty_traversal_and_null(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "safe"
    allowed.mkdir()
    inside = allowed / "model.pt"
    assert validate_safe_path(inside, base_dir=allowed) == inside.resolve()

    with pytest.raises(ValueError, match="non-empty"):
        validate_safe_path("")

    with pytest.raises(ValueError, match="null"):
        validate_safe_path("bad\x00path")

    with pytest.raises(ValueError, match="escapes") as exc_info:
        validate_safe_path(allowed / ".." / "outside", base_dir=allowed)
    assert str(allowed) not in str(exc_info.value)


def test_validate_image_bytes_guardrails() -> None:
    img = validate_image_bytes(_make_image_bytes("PNG", (32, 32)))
    assert img.mode == "RGB"
    assert img.size == (32, 32)

    with pytest.raises(ValueError, match="empty"):
        validate_image_bytes(b"")

    with pytest.raises(ValueError, match="maximum"):
        validate_image_bytes(b"x" * 100, max_bytes=50)

    with pytest.raises(ValueError, match="Unsupported"):
        validate_image_bytes(_make_image_bytes("BMP", (16, 16)))

    with pytest.raises(ValueError, match="corrupted|Invalid"):
        validate_image_bytes(b"not-an-image")

    with pytest.raises(ValueError, match="bounds|Decompression|corrupted"):
        validate_image_bytes(
            _make_image_bytes("PNG", (20, 20)), max_pixels=100
        )


def test_translator_en_and_id() -> None:
    tr_en = Translator("en")
    tr_id = Translator("id")
    assert "Plant Disease" in tr_en.t("app_title")
    assert "Penyakit Daun" in tr_id.t("app_title")
    assert "%" in tr_en.pct(0.875)
    assert "0" in tr_id.dec(0.1234)
    with pytest.raises(ValueError):
        Translator("fr")  # type: ignore[arg-type]
