# Plant Disease Image Classification with Transfer Learning

## Problem Statement
Identify plant leaf diseases from RGB leaf imagery using convolutional neural networks and transfer learning, while providing visual attribution (Grad-CAM) and runtime input guardrails for safe inference.

## Dataset
- **Target Dataset:** PlantVillage Dataset (Kaggle) — 54,000+ RGB images of healthy and diseased crop leaves across 38 classes arranged in `ImageFolder` class directories under `data/plantvillage/`.
- **Local Verification Dataset:** Deterministic 3-class synthetic leaf generator (`Tomato_Healthy`, `Tomato_Early_Blight`, `Tomato_Yellow_Leaf_Curl`, 18 samples/class, 54 total images, stratified 70/15/15 split: 37 train / 8 val / 9 test) built into `src/data.py` so the full training, evaluation, and Grad-CAM pipeline runs on CPU without requiring an external download.

## Skills & Features Implemented
- **Architectures (`src/models.py`):** 3-block `BaselineCNN` trained from scratch, `ResNet18`, and `EfficientNet-B0` transfer learning backbones with optional backbone freezing.
- **Data Pipeline & Augmentation (`src/data.py`):** Deterministic stratified train/val/test split (`StratifiedShuffleSplit`), ImageNet normalization, and training augmentations (random resized crop, horizontal/vertical flips, rotation, color jitter).
- **Evaluation (`src/evaluate.py`):** Accuracy, macro precision, macro recall, macro F1, macro OvR ROC-AUC, per-class F1, and labeled confusion matrix plots.
- **Explainability (`src/gradcam.py`):** Grad-CAM visual attribution with automatic hook cleanup and heatmap overlay rendering.
- **Guardrails & i18n (`src/schemas.py`, `src/i18n.py`):** Pydantic v2 config/checkpoint/result validation, path-traversal checks, thread-safe upload byte/pixel/format checks (`JPEG`, `PNG`, `WEBP`, decompression-bomb guard), `weights_only=True` checkpoint loading, low-confidence warnings (`< 50%`), and English/Indonesian (`en`/`id`) Babel translations.
- **Interactive Demo (`app.py`):** Streamlit app displaying Top-3 predictions, confidence warnings, and side-by-side Grad-CAM overlays.

## CPU Smoke Benchmark (3-Class Synthetic Dataset, 2 Epochs, 64x64, `pretrained=False`)
Verified via `python -m src.train --synthetic-demo --compare-all --epochs 2 --image-size 64 --no-pretrained --output-dir artifacts/smoke_run`:

| Architecture | Augmented | Accuracy | Macro Precision | Macro Recall | Macro F1 | Macro ROC-AUC | Test Samples |
|--------------|-----------|----------|-----------------|--------------|----------|---------------|--------------|
| `baseline_cnn` | `False` | 0.667 | 0.500 | 0.667 | 0.556 | 1.000 | 9 |
| `baseline_cnn` | `True` | 0.667 | 0.500 | 0.667 | 0.556 | 1.000 | 9 |
| `resnet18` | `False` | 0.333 | 0.111 | 0.333 | 0.167 | 0.889 | 9 |
| `resnet18` | `True` | 0.667 | 0.500 | 0.667 | 0.556 | 0.907 | 9 |

## Key Findings
1. Under a 2-epoch CPU smoke budget from random initialization (`--no-pretrained`) on 37 training images, the lightweight `baseline_cnn` converged quickly (`0.667` accuracy, `0.556` macro F1, `1.000` macro OvR ROC-AUC), whereas unpretrained `resnet18` without augmentation collapsed (`0.333` accuracy, `0.167` macro F1).
2. Adding data augmentation stabilized `resnet18` from `0.333` to `0.667` accuracy (`0.167` to `0.556` macro F1; `0.889` to `0.907` ROC-AUC); for full 38-class PlantVillage training, pretrained ImageNet weights (`--pretrained` default) and longer schedules are recommended.
3. Checkpoint persistence separates pure `state_dict` weights (`model_state.pt`, loaded with `weights_only=True`) from Pydantic-validated JSON metadata (`metadata.json`), preventing unsafe pickle deserialization during inference.

## How to Run
```bash
pip install -r requirements.txt

# 1. Run unit and end-to-end tests
pytest -v

# 2. Run local CPU comparison demo (synthetic dataset)
python -m src.train --synthetic-demo --compare-all --epochs 2 --image-size 64 --no-pretrained

# 3. Train on full PlantVillage dataset (placed under data/plantvillage/<class_name>/)
python -m src.train --data-dir data/plantvillage --architecture resnet18 --epochs 10

# 4. Launch Streamlit inference demo
streamlit run app.py
```

## Project Structure
- `src/schemas.py` — Pydantic v2 schemas and path/image validation guardrails
- `src/i18n.py` — Babel English/Indonesian localization
- `src/data.py` — Transforms, stratified splitting, and synthetic dataset generator
- `src/models.py` — `BaselineCNN`, `ResNet18`, and `EfficientNet-B0` builders
- `src/gradcam.py` — Grad-CAM hook manager and heatmap overlay
- `src/evaluate.py` — Metrics computation and confusion matrix plotting
- `src/train.py` — Training CLI, comparison runner, and safe checkpoint I/O
- `app.py` — Streamlit interactive classifier and Grad-CAM viewer
- `notebooks/01_eda.ipynb` — Exploratory data analysis notebook
- `notebooks/02_modeling.ipynb` — Modeling comparison notebook
- `tests/` — Unit and end-to-end test suite

## Methodology & Limitations
The pipeline splits datasets into stratified 70% training, 15% validation, and 15% test partitions, selects the best validation-loss checkpoint, and reports multi-metric test performance alongside Grad-CAM spatial attribution. The local benchmark numbers above reflect the built-in 3-class synthetic leaf dataset on CPU (`pretrained=False`) rather than the 38-class PlantVillage corpus; real-world field images with cluttered backgrounds, variable lighting, or unseen crop species require training and calibration on representative field data.
