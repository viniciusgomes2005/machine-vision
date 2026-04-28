# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Python/OpenCV pipeline for measuring seedling morphology from controlled images. Core reusable helpers live in `reutilizaveis.py`. Evaluation logic and reference comparisons live in `avaliar_resultados.py`. Step-based computer vision scripts are in `scripts_cv/`, with numbered stages such as `01_tratamentos_iniciais.py`, `03_diametro_coleto.py`, and `06_pipeline_csv.py`.

Input images are under `Dataset_Projeto1/`, split into `_Eucalipto_Escolhidos1/` and `_Pinheiro_Escolhidos1/`. Generated CSVs belong in `resultados/`; visual debug artifacts belong in `debug_saida/` or `_dev_debug/`. Project instructions and notebooks are stored in `Instrucoes/`, while notes belong in `docs/`.

## Build, Test, and Development Commands

Create and activate a local environment before running scripts:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy opencv-python matplotlib jupyterlab ipykernel
```

Run the main CSV pipeline:

```bash
python3 scripts_cv/06_pipeline_csv.py \
  --input-dir Dataset_Projeto1/_Eucalipto_Escolhidos1 \
  --output-csv resultados/resultado.csv
```

Debug a single image and write masks/overlays:

```bash
python3 scripts_cv/testar_uma_imagem.py \
  --img Dataset_Projeto1/_Eucalipto_Escolhidos1/Eucalipto1.jpg \
  --out-dir debug_saida/euc1_diag
```

Evaluate result CSVs against references:

```bash
python3 avaliar_resultados.py --help
```

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation. Keep the repository's Portuguese naming style for variables, functions, comments, and documentation. Prefer `pathlib.Path` for paths and keep image masks as explicit `uint8` OpenCV-compatible arrays. Preserve the numbered script order in `scripts_cv/` when adding pipeline stages.

## Testing Guidelines

There is no formal automated test suite. Validate changes by running `06_pipeline_csv.py`, comparing generated CSV values, and inspecting overlays/masks in `debug_saida/`. For segmentation changes, include at least one eucalyptus and one pine image when practical. Avoid committing large temporary debug outputs unless they are intentionally part of the investigation.

## Commit & Pull Request Guidelines

Recent history uses short imperative messages such as `Add image processing scripts for eucalyptus analysis`. Prefer concise, descriptive commits in that style, for example `Refine leaf mask cleanup for eucalyptus`.

Pull requests should describe the image set tested, commands run, CSV outputs changed, and any visual debug evidence. Link related tasks or notes when available. Include screenshots or paths to representative overlays when behavior changes are visual.

## Agent-Specific Instructions

Do not overwrite datasets or reference CSVs without a clear reason. Keep generated artifacts in `debug_saida/` or `resultados/`, and keep code changes focused on the pipeline stage being modified.
