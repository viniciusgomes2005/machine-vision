# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computer vision pipeline for morphological analysis of tree seedlings (*Eucalyptus* and *Pine*), measuring height, stem diameter, leaf area, and leaf count from controlled JPEG images. Academic project (MVISIA challenge). All variable names, comments, and docs are in **Portuguese**.

## Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Dependencies: `numpy`, `opencv-python`, `matplotlib`, `jupyterlab`, `ipykernel`.

## Running the Code

```bash
# Main analysis notebook
jupyter lab LuizVini_Projeto1.ipynb

# Standalone exploration scripts
python explora_pinheiro_23.py
python explora_area_foliar_familias.py

# Development/debug scripts (output goes to _dev_debug/)
python _dev_pipeline.py
python _dev_measures.py
```

There is no formal test suite — correctness is validated visually by inspecting output images in `_dev_debug/` and `Resultados_LuizVini_Projeto1/`.

## Architecture

The pipeline processes images through these sequential stages:

### 1. Image Loading & Color Spaces
Images are loaded as BGR via OpenCV, then converted to RGB and HSV. The controlled setup uses a **blue background**, **black pot (tubete)**, and **white cylinder** as a diameter reference.

### 2. Scene Segmentation (`SegmentaCena` / `segmenta_base`)
Returns four binary masks:
- `mask_fundo` — blue background (HSV H: 90–135)
- `mask_tubete` — black pot (HSV V < 80, lower 40% of image)
- `mask_cilindro` — white reference cylinder (HSV V > 170)
- `mask_planta` — plant tissue (everything else after morphological cleanup)

### 3. Species-Specific Stem Detection
- **Pine:** vertical morphological opening with a tall narrow kernel `(3, 41)`
- **Eucalyptus:** distance transform from plant mask centroid

### 4. Fine Leaf Segmentation (`mascara_fina`)
Color refinement on the plant mask to separate green/yellowish leaves from stem, including red-tinted new shoots.

### 5. Measurement Extraction
Eight+ metrics written to `resultado_medidas.csv`:

| Column | Description |
|--------|-------------|
| `altura_basica_px` | Topmost point to pot top |
| `comprimento_basico_px` | Euclidean stem length |
| `comprimento_total_px` | Total path length |
| `comprimento_vertical_px` | Pure vertical component |
| `diametro_coleto_px` | Stem diameter at collar |
| `area_foliar_px2` | Total leaf area (pixels²) |
| `numero_folhas` | Leaf count |
| `confianca_folhas` | Confidence enum (`"components"`, etc.) |

### Key Files

| File | Role |
|------|------|
| `LuizVini_Projeto1.ipynb` | Full integrated pipeline notebook |
| `explora_pinheiro_23.py` | Most complete modular implementation (reference) |
| `explora_area_foliar_familias.py` | Leaf area analysis by species/family |
| `_dev_pipeline.py` | Segmentation sandbox / debug helper |
| `_dev_measures.py` | Measurement function sandbox |

### Data Layout

```
Dataset_Projeto1/
├── _Eucalipto_Escolhidos1/   # ~10 JPEG images
└── _Pinheiro_Escolhidos1/    # ~6 JPEG images

Resultados_LuizVini_Projeto1/
├── resultado_medidas.csv
├── anotadas/                 # Annotated output images
├── debug/                    # Per-stage segmentation debug images
└── figuras/                  # Plots
```

## Key Implementation Details

- Morphological kernel sizes: `(3,3)` ellipse for fine work, `(5,5)` medium, `(25,25)` for dilation
- Minimum component area threshold: 120–200 px (filters noise)
- Pixel-to-mm calibration is done via the white cylinder's known diameter
- `_maior_componente()` and `_filtra_area_min()` are utility helpers used throughout all segmentation stages
- Debug images are written with `cv2.imwrite` inside a `Debug()` helper; pass a flag to enable/disable
