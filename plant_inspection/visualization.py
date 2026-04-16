"""Visualization and annotation output for inspection results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from .measurements import MeasurementResult
from .segmentation import SegmentationResult
from .utils import save_image


def _overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = image_bgr.copy()
    overlay = np.zeros_like(out)
    overlay[mask] = color
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def save_debug_images(
    image_bgr: np.ndarray,
    image_name: str,
    output_debug_dir: Path,
    seg: SegmentationResult,
    meas: MeasurementResult,
) -> None:
    """Save required intermediate masks and overlays."""
    stem = image_name.rsplit(".", 1)[0]

    bg = _overlay_mask(image_bgr, seg.background_mask, (255, 0, 0), alpha=0.25)
    save_image(output_debug_dir / f"{stem}_01_background_mask.png", bg)

    plant = _overlay_mask(image_bgr, seg.plant_mask, (0, 255, 0), alpha=0.4)
    save_image(output_debug_dir / f"{stem}_02_plant_mask.png", plant)

    objects = image_bgr.copy()
    objects = _overlay_mask(objects, seg.pot_mask, (20, 20, 20), alpha=0.55)
    objects = _overlay_mask(objects, seg.white_cylinder_mask, (255, 255, 255), alpha=0.55)
    x0, y0, x1, y1 = seg.pot_bbox
    cv2.rectangle(objects, (x0, y0), (x1, y1), (0, 165, 255), 2)
    cv2.line(objects, (0, seg.pot_top_y), (objects.shape[1] - 1, seg.pot_top_y), (0, 165, 255), 1)
    cv2.circle(objects, seg.collar_point, 8, (0, 0, 255), -1)
    save_image(output_debug_dir / f"{stem}_03_pot_collar_detection.png", objects)

    skeleton = image_bgr.copy()
    skeleton[meas.skeleton_mask] = (0, 0, 255)
    cv2.circle(skeleton, seg.collar_point, 8, (0, 255, 255), -1)
    save_image(output_debug_dir / f"{stem}_04_skeleton.png", skeleton)

    leaf = image_bgr.copy()
    leaf = _overlay_mask(leaf, meas.leaf_mask, (0, 255, 255), alpha=0.45)
    leaf = _overlay_mask(leaf, meas.stem_mask, (255, 0, 255), alpha=0.5)
    save_image(output_debug_dir / f"{stem}_05_leaf_segmentation.png", leaf)


def render_final_annotation(
    image_bgr: np.ndarray,
    image_name: str,
    species: str,
    output_annotated_dir: Path,
    seg: SegmentationResult,
    meas: MeasurementResult,
    calibrated_fields: Dict[str, object],
) -> None:
    """Draw final metrics and geometric references."""
    out = image_bgr.copy()

    out = _overlay_mask(out, seg.plant_above_pot_mask, (40, 220, 40), alpha=0.28)

    x0, y0, x1, y1 = seg.pot_bbox
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 165, 255), 2)
    cv2.line(out, (0, seg.pot_top_y), (out.shape[1] - 1, seg.pot_top_y), (0, 165, 255), 1)
    cv2.circle(out, seg.collar_point, 8, (0, 0, 255), -1)

    if np.any(seg.plant_above_pot_mask):
        top_y = int(np.where(seg.plant_above_pot_mask)[0].min())
        top_x_candidates = np.where(seg.plant_above_pot_mask[top_y])[0]
        if len(top_x_candidates) > 0:
            top_x = int(np.median(top_x_candidates))
            cv2.circle(out, (top_x, top_y), 7, (255, 255, 0), -1)
            cv2.line(out, seg.collar_point, (top_x, top_y), (255, 255, 0), 2)

    text_lines = [
        f"image: {image_name}",
        f"species: {species}",
        f"length_basic_px: {meas.total_length_px_basic:.1f}",
        f"length_adv_px: {meas.total_length_px_advanced:.1f}" if meas.total_length_px_advanced is not None else "length_adv_px: n/a",
        f"collar_diam_px: {meas.collar_diameter_px:.1f}",
        f"leaf_area_px2: {meas.leaf_area_px2:.0f}",
        f"leaf_count: {meas.leaf_count}",
        f"leaf_count_conf: {meas.leaf_count_confidence}",
    ]

    if calibrated_fields.get("calibration_status") == "calibrated":
        text_lines.append(f"length_basic_mm: {calibrated_fields.get('total_length_mm_basic', 0):.1f}")
        text_lines.append(f"collar_diam_mm: {calibrated_fields.get('collar_diameter_mm', 0):.2f}")
        text_lines.append(f"leaf_area_mm2: {calibrated_fields.get('leaf_area_mm2', 0):.1f}")

    y = 36
    for line in text_lines:
        cv2.putText(out, line, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 3)
        cv2.putText(out, line, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 1)
        y += 34

    stem = image_name.rsplit(".", 1)[0]
    save_image(output_annotated_dir / f"{stem}_annotated.png", out)
