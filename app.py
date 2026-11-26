"""Streamlit web application for plant leaf disease inference and Grad-CAM."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import streamlit as st
import torch

from src.data import DEFAULT_SYNTHETIC_CLASSES
from src.i18n import Translator
from src.models import build_model
from src.schemas import (
    DEFAULT_MAX_IMAGE_BYTES,
    ArchitectureName,
    validate_image_bytes,
    validate_safe_path,
)
from src.train import load_checkpoint_safely, predict_with_gradcam


@st.cache_resource(show_spinner=False)
def _resolve_model(
    checkpoint_dir_str: str,
    device_str: str,
) -> tuple[torch.nn.Module, ArchitectureName, list[str], int, bool]:
    """Load a saved checkpoint or fall back to an untrained demo model."""
    device = torch.device(device_str)
    try:
        model, meta = load_checkpoint_safely(Path(checkpoint_dir_str), device)
        return (
            model,
            meta.architecture,
            list(meta.class_names),
            meta.image_size,
            False,
        )
    except FileNotFoundError:
        demo_model = build_model(
            architecture="resnet18",
            num_classes=len(DEFAULT_SYNTHETIC_CLASSES),
            pretrained=False,
        ).to(device)
        demo_model.eval()
        return (
            demo_model,
            "resnet18",
            list(DEFAULT_SYNTHETIC_CLASSES),
            224,
            True,
        )


def main() -> None:
    """Render the Streamlit inference and Grad-CAM inspection UI."""
    st.set_page_config(page_title="Plant Disease Classifier", layout="wide")

    locale_choice = st.sidebar.selectbox(
        "Language / Bahasa", options=("en", "id"), index=0
    )
    tr = Translator(locale=locale_choice)

    st.title(tr.t("app_title"))
    st.caption(tr.t("app_subtitle"))

    raw_dir = st.sidebar.text_input(
        tr.t("model_source_label"), value="artifacts/resnet18_aug"
    ).strip()
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    if not raw_dir:
        st.error(tr.t("checkpoint_load_error"))
        return

    try:
        safe_dir = validate_safe_path(raw_dir, base_dir=Path.cwd())
        model, arch, class_names, image_size, is_demo = _resolve_model(
            str(safe_dir), device_str
        )
    except (
        ValueError,
        ValidationError,
        RuntimeError,
        KeyError,
        OSError,
        EOFError,
        TypeError,
    ):
        st.error(tr.t("checkpoint_load_error"))
        return

    if is_demo:
        st.warning(tr.t("demo_mode_banner"))
    else:
        st.info(
            tr.t("checkpoint_loaded", arch=arch, classes=len(class_names))
        )

    uploaded = st.file_uploader(
        tr.t("upload_label"), type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded is None:
        return

    if uploaded.size > DEFAULT_MAX_IMAGE_BYTES:
        st.error(
            tr.t(
                "invalid_image_error",
                err=(
                    f"Image exceeds maximum allowed size of "
                    f"{DEFAULT_MAX_IMAGE_BYTES} bytes."
                ),
            )
        )
        return

    try:
        image = validate_image_bytes(uploaded.getvalue())
    except ValueError as exc:
        st.error(tr.t("invalid_image_error", err=str(exc)))
        return

    try:
        result, overlay = predict_with_gradcam(
            model=model,
            architecture=arch,
            image=image,
            class_names=class_names,
            image_size=image_size,
            device=torch.device(device_str),
        )
    except (ValueError, RuntimeError, TypeError):
        st.error(tr.t("inference_error"))
        return

    if result.low_confidence:
        st.warning(
            tr.t(
                "low_confidence_warning",
                prob=tr.pct(result.top_probability),
            )
        )

    st.subheader(tr.t("predictions_header"))
    for item in result.predictions:
        st.write(f"- **{item.label}**: {tr.pct(item.probability)}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.image(image, caption=tr.t("original_image"), use_container_width=True)
    with col_right:
        st.image(
            overlay, caption=tr.t("gradcam_image"), use_container_width=True
        )


if __name__ == "__main__":
    main()
