# Machine Vision Seedling Inspection

Classical computer-vision pipeline for morphology inspection of eucalyptus and pine seedlings imaged over a blue background with a black pot and white cylindrical support.

## What it does
- Reads all images from `--input_dir`
- Prints and saves initial image inspection stats (dimensions + color means)
- Segments background, plant, pot, and white support
- Detects pot top and collar/base reference region
- Measures:
  - `total_length_px_basic` (vertical distance collar -> top)
  - `total_length_px_advanced` (skeleton geodesic from collar)
  - `collar_diameter_px` (median local diameter above pot from distance transform)
  - `leaf_area_px2` (species-aware foliar mask)
  - `leaf_count` + confidence/note
- Saves annotated and debug images
- Exports `outputs/results.csv`

## Run
```bash
python main.py --input_dir Dataset_Projeto1 --output_dir outputs
```

With optional metric calibration from white cylinder diameter:
```bash
python main.py \
  --input_dir Dataset_Projeto1 \
  --output_dir outputs \
  --reference_object white_cylinder \
  --reference_diameter_mm 50
```

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

## Calibration behavior
- Default: pixel units only (`px`, `px²`), reported in CSV with `calibration_status=not_calibrated_px_only`.
- If calibration is enabled and reference is detected, pipeline adds:
  - `total_length_mm_basic`, `total_length_mm_advanced`
  - `collar_diameter_mm`
  - `leaf_area_mm2`
- If calibration is requested but reference detection fails, CSV notes this explicitly.

## Species-specific handling
- Eucalyptus: attempts stem suppression before leaf-area and leaf-count estimation; leaf count uses watershed splitting with shape/area filtering.
- Pine: dense needles are treated as foliar mass for area; `leaf_count` is a low-confidence cluster proxy and flagged in notes.

## Assumptions and limitations
- Camera geometry and setup are assumed stable and centered.
- Measurements are image-plane values (no perspective correction).
- Collar diameter is an estimate near the base; heavy occlusion can bias it.
- Leaf counting is approximate, especially for overlapped broad leaves (eucalyptus) and needle clusters (pine).
- No deep learning is used; this is an explainable deterministic baseline.
