"""Pydantic validation schemas and input guardrails."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import threading
from typing import Literal

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator

ArchitectureName = Literal["baseline_cnn", "resnet18", "efficientnet_b0"]
LocaleCode = Literal["en", "id"]

ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 25_000_000
LOW_CONFIDENCE_THRESHOLD = 0.50

_PIL_LIMIT_LOCK = threading.Lock()


def validate_safe_path(
    path: str | Path,
    base_dir: Path | None = None,
) -> Path:
    """Validate and resolve a filesystem path against traversal and null bytes.

    Args:
        path: Candidate filesystem path or string.
        base_dir: Optional root directory that the resolved path must stay under.

    Returns:
        Resolved path object.

    Raises:
        ValueError: If the path is empty, contains null bytes, or escapes
            ``base_dir``.
    """
    raw = str(path).strip()
    if not raw or "\x00" in raw:
        raise ValueError("Path must be non-empty and free of null bytes.")
    resolved = Path(raw).expanduser().resolve()
    if base_dir is not None:
        base_resolved = base_dir.expanduser().resolve()
        if not resolved.is_relative_to(base_resolved):
            raise ValueError("Path escapes the allowed base directory.")
    return resolved


def validate_image_bytes(
    data: bytes,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> Image.Image:
    """Validate uploaded image bytes against size, format, and pixel limits.

    Args:
        data: Raw image byte payload.
        max_bytes: Maximum allowed payload size in bytes.
        max_pixels: Maximum allowed image pixel count (width * height).

    Returns:
        Decoded RGB ``PIL.Image.Image`` instance.

    Raises:
        ValueError: If payload is empty, oversized, unsupported, or corrupted.
    """
    if not data:
        raise ValueError("Image payload is empty.")
    if len(data) > max_bytes:
        raise ValueError(
            f"Image exceeds maximum allowed size of {max_bytes} bytes."
        )

    with _PIL_LIMIT_LOCK:
        previous_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = max_pixels
        try:
            with Image.open(BytesIO(data)) as probe:
                img_format = (probe.format or "").upper()
                if img_format not in ALLOWED_IMAGE_FORMATS:
                    allowed = ", ".join(sorted(ALLOWED_IMAGE_FORMATS))
                    raise ValueError(
                        f"Unsupported image format '{img_format}'. "
                        f"Allowed: {allowed}."
                    )
                width, height = probe.size
                if (
                    width <= 0
                    or height <= 0
                    or (width * height) > max_pixels
                ):
                    raise ValueError(
                        f"Image dimensions ({width}x{height}) exceed safety bounds."
                    )
                probe.verify()

            with Image.open(BytesIO(data)) as verified:
                return verified.convert("RGB")
        except (
            UnidentifiedImageError,
            Image.DecompressionBombError,
            OSError,
            SyntaxError,
            EOFError,
        ) as exc:
            raise ValueError(f"Invalid or corrupted image file: {exc}") from exc
        finally:
            Image.MAX_IMAGE_PIXELS = previous_limit


class TrainConfig(BaseModel):
    """Validated training and evaluation configuration."""

    data_dir: Path = Field(default=Path("data/plantvillage"))
    output_dir: Path = Field(default=Path("artifacts"))
    architecture: ArchitectureName = "resnet18"
    epochs: int = Field(default=5, ge=1, le=200)
    batch_size: int = Field(default=16, ge=1, le=512)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)
    image_size: int = Field(default=224, ge=64, le=1024)
    use_augmentation: bool = True
    pretrained: bool = True
    freeze_backbone: bool = False
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    locale: LocaleCode = "en"

    @field_validator("data_dir", "output_dir")
    @classmethod
    def _check_safe_path(cls, value: Path) -> Path:
        return validate_safe_path(value)


class CheckpointMetadata(BaseModel):
    """Validated metadata schema loaded alongside model_state.pt."""

    architecture: ArchitectureName
    class_names: list[str] = Field(min_length=2)
    image_size: int = Field(ge=64, le=1024)

    @field_validator("class_names")
    @classmethod
    def _check_non_empty_classes(cls, value: list[str]) -> list[str]:
        if any(not name.strip() for name in value):
            raise ValueError("Class names must be non-empty strings.")
        return value


class PredictionItem(BaseModel):
    """Single class prediction with probability."""

    label: str = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResult(BaseModel):
    """Top-K inference result with low-confidence guardrail."""

    predictions: list[PredictionItem] = Field(min_length=1)
    top_label: str = Field(min_length=1)
    top_probability: float = Field(ge=0.0, le=1.0)
    low_confidence: bool = False


class EvaluationReport(BaseModel):
    """Aggregated test-set evaluation metrics."""

    architecture: str = Field(min_length=1)
    augmented: bool
    accuracy: float = Field(ge=0.0, le=1.0)
    macro_precision: float = Field(ge=0.0, le=1.0)
    macro_recall: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    macro_roc_auc: float | None = Field(default=None, ge=0.0, le=1.0)
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    num_samples: int = Field(ge=1)
