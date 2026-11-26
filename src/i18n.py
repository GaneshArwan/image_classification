"""Internationalization helpers powered by Babel."""

from __future__ import annotations

from babel import Locale
from babel.numbers import format_decimal, format_percent

from src.schemas import LocaleCode

_CATALOG: dict[LocaleCode, dict[str, str]] = {
    "en": {
        "app_title": "Plant Disease Image Classifier",
        "app_subtitle": (
            "Transfer learning inference with Grad-CAM visual attribution."
        ),
        "model_source_label": "Checkpoint directory",
        "upload_label": "Upload a leaf image (JPG, PNG, WEBP)",
        "demo_mode_banner": (
            "No trained checkpoint found; running untrained demo weights. "
            "Train a model first for calibrated plant disease predictions."
        ),
        "checkpoint_loaded": "Loaded checkpoint: {arch} ({classes} classes).",
        "checkpoint_load_error": (
            "Unable to load checkpoint from the specified directory. "
            "Verify the directory path and checkpoint metadata."
        ),
        "inference_error": (
            "Inference failed for the uploaded image. "
            "Verify checkpoint compatibility and image integrity."
        ),
        "low_confidence_warning": (
            "Low confidence ({prob}): verify image lighting, leaf focus, "
            "or domain alignment before acting on this prediction."
        ),
        "predictions_header": "Top Predictions",
        "original_image": "Uploaded Image",
        "gradcam_image": "Grad-CAM Attribution",
        "invalid_image_error": "Image rejected by validation guardrail: {err}",
        "train_start": (
            "Starting training: arch={arch}, augmented={aug}, epochs={epochs}, "
            "device={device}."
        ),
        "epoch_log": (
            "Epoch {epoch}/{epochs} | train_loss={train_loss} | "
            "val_loss={val_loss} | val_acc={val_acc}"
        ),
        "eval_summary": (
            "[{arch} | aug={aug}] Acc={acc}, F1={f1}, AUC={auc}, "
            "Samples={samples}"
        ),
        "saved_artifacts": "Saved artifacts to {path}.",
    },
    "id": {
        "app_title": "Klasifikasi Penyakit Daun Tanaman",
        "app_subtitle": (
            "Inferensi transfer learning dengan visualisasi Grad-CAM."
        ),
        "model_source_label": "Direktori checkpoint",
        "upload_label": "Unggah gambar daun (JPG, PNG, WEBP)",
        "demo_mode_banner": (
            "Checkpoint terlatih tidak ditemukan; menjalankan bobot demo. "
            "Latih model terlebih dahulu untuk prediksi yang terkalibrasi."
        ),
        "checkpoint_loaded": "Checkpoint dimuat: {arch} ({classes} kelas).",
        "checkpoint_load_error": (
            "Gagal memuat checkpoint dari direktori tersebut. "
            "Periksa jalur direktori dan metadata checkpoint."
        ),
        "inference_error": (
            "Inferensi gagal untuk gambar yang diunggah. "
            "Periksa kecocokan checkpoint dan integritas gambar."
        ),
        "low_confidence_warning": (
            "Keyakinan rendah ({prob}): periksa pencahayaan, fokus daun, "
            "atau kecocokan domain sebelum mengambil tindakan."
        ),
        "predictions_header": "Prediksi Teratas",
        "original_image": "Gambar Diunggah",
        "gradcam_image": "Atribusi Grad-CAM",
        "invalid_image_error": "Gambar ditolak oleh validasi keamanan: {err}",
        "train_start": (
            "Memulai pelatihan: arch={arch}, augmentasi={aug}, "
            "epoch={epochs}, perangkat={device}."
        ),
        "epoch_log": (
            "Epoch {epoch}/{epochs} | train_loss={train_loss} | "
            "val_loss={val_loss} | val_acc={val_acc}"
        ),
        "eval_summary": (
            "[{arch} | aug={aug}] Akurasi={acc}, F1={f1}, AUC={auc}, "
            "Sampel={samples}"
        ),
        "saved_artifacts": "Artefak disimpan ke {path}.",
    },
}


class Translator:
    """Locale-aware string and number formatter using Babel."""

    def __init__(self, locale: LocaleCode = "en") -> None:
        if locale not in _CATALOG:
            raise ValueError(f"Unsupported locale: {locale}")
        self.locale_code: LocaleCode = locale
        self.locale: Locale = Locale.parse(locale)
        self._messages = _CATALOG[locale]

    def t(self, key: str, **kwargs: object) -> str:
        """Translate a catalog key and format placeholders."""
        template = self._messages.get(key, _CATALOG["en"].get(key, key))
        return template.format(**kwargs) if kwargs else template

    def pct(self, value: float) -> str:
        """Format a [0, 1] ratio as a locale-aware percentage string."""
        return format_percent(value, format="#,##0.0%", locale=self.locale)

    def dec(self, value: float) -> str:
        """Format a decimal number with four fractional digits."""
        return format_decimal(value, format="#,##0.0000", locale=self.locale)
