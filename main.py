"""CLI entrypoint for seedling morphology inspection."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

from config import AppConfig
from plant_inspection.measurements import compute_measurements
from plant_inspection.preprocessing import inspect_images, save_preview
from plant_inspection.segmentation import segment_scene
from plant_inspection.utils import (
    add_calibrated_measurements,
    compute_scale_mm_per_px,
    ensure_dir,
    infer_species,
    list_images,
    load_bgr_image,
)
from plant_inspection.visualization import render_final_annotation, save_debug_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classical CV pipeline for seedling morphology inspection.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing input images.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output folder for CSV and images.")
    parser.add_argument(
        "--reference_object",
        type=str,
        default=None,
        choices=["white_cylinder"],
        help="Reference object to use for calibration.",
    )
    parser.add_argument(
        "--reference_diameter_mm",
        type=float,
        default=None,
        help="Known real-world diameter of the reference object in mm.",
    )
    parser.add_argument("--no_debug", action="store_true", help="Disable debug image export.")
    return parser.parse_args()


def _write_csv(rows: List[Dict[str, object]], output_csv: Path) -> None:
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_inspection_table(stats: List[Dict[str, object]]) -> None:
    if not stats:
        return

    print("[INFO] Image inventory and basic color statistics")
    headers = ["image_name", "height", "width", "b_mean", "g_mean", "r_mean", "h_mean", "s_mean", "v_mean"]
    print(" | ".join(headers))
    for row in stats:
        vals = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        print(" | ".join(vals))


def inspect_and_log(image_paths: List[Path], output_dir: Path) -> List[Dict[str, object]]:
    stats = inspect_images(image_paths)
    _print_inspection_table(stats)
    _write_csv(stats, output_dir / "image_inspection_stats.csv")
    save_preview(image_paths, output_dir / "debug" / "dataset_preview.png")
    return stats


def calibrate_for_image(config: AppConfig, reference_diameter_px: Optional[float]) -> Optional[float]:
    if not config.calibration.enabled:
        return None
    if config.calibration.reference_object != "white_cylinder":
        return None
    return compute_scale_mm_per_px(config.calibration.reference_diameter_mm, reference_diameter_px)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    debug_dir = output_dir / "debug"
    annotated_dir = output_dir / "annotated"

    ensure_dir(output_dir)
    ensure_dir(debug_dir)
    ensure_dir(annotated_dir)

    config = AppConfig()
    config.save_debug = not args.no_debug

    if args.reference_object and args.reference_diameter_mm is not None:
        config.calibration.enabled = True
        config.calibration.reference_object = args.reference_object
        config.calibration.reference_diameter_mm = float(args.reference_diameter_mm)

    image_paths = list_images(input_dir)
    if not image_paths:
        raise SystemExit(f"No images found under {input_dir}")

    inspect_and_log(image_paths, output_dir)

    rows: List[Dict[str, object]] = []

    for image_path in image_paths:
        image = load_bgr_image(image_path)
        species = infer_species(image_path.name)

        seg = segment_scene(image, config.segmentation)
        meas = compute_measurements(seg, species, config.measurement)

        mm_per_px = calibrate_for_image(config, seg.reference_diameter_px)

        row: Dict[str, object] = {
            "image_name": image_path.name,
            "species": species,
            "total_length_px_basic": meas.total_length_px_basic,
            "total_length_px_advanced": meas.total_length_px_advanced,
            "collar_diameter_px": meas.collar_diameter_px,
            "leaf_area_px2": meas.leaf_area_px2,
            "leaf_count": meas.leaf_count,
            "leaf_count_confidence": meas.leaf_count_confidence,
            "pot_top_y_px": seg.pot_top_y,
            "reference_diameter_px": seg.reference_diameter_px,
            "notes": meas.notes,
        }

        row = add_calibrated_measurements(row, mm_per_px)

        if config.calibration.enabled and mm_per_px is None:
            extra_note = "calibration_requested_but_reference_not_found"
            row["notes"] = f"{row['notes']};{extra_note}" if row["notes"] else extra_note

        if config.save_debug:
            save_debug_images(image, image_path.name, debug_dir, seg, meas)

        if config.save_annotated:
            render_final_annotation(image, image_path.name, species, annotated_dir, seg, meas, row)

        rows.append(row)

    _write_csv(rows, output_dir / "results.csv")

    print(f"[INFO] Processed {len(rows)} images")
    print(f"[INFO] Results CSV: {output_dir / 'results.csv'}")
    print(f"[INFO] Annotated images: {annotated_dir}")
    if config.save_debug:
        print(f"[INFO] Debug images: {debug_dir}")


if __name__ == "__main__":
    main()
