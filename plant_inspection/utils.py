"""Utility functions for IO, calibration, and species inference."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(input_dir: Path) -> List[Path]:
    images = [p for p in input_dir.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS]
    return sorted(images)


def infer_species(image_name: str) -> str:
    lower = image_name.lower()
    if "eucalipto" in lower or "eucalyptus" in lower:
        return "eucalyptus"
    if "pinheiro" in lower or "pine" in lower:
        return "pine"
    return "unknown"


def load_bgr_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def image_color_stats(image_bgr: np.ndarray) -> Dict[str, float]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    bgr_mean = image_bgr.reshape(-1, 3).mean(axis=0)
    hsv_mean = hsv.reshape(-1, 3).mean(axis=0)
    return {
        "b_mean": float(bgr_mean[0]),
        "g_mean": float(bgr_mean[1]),
        "r_mean": float(bgr_mean[2]),
        "h_mean": float(hsv_mean[0]),
        "s_mean": float(hsv_mean[1]),
        "v_mean": float(hsv_mean[2]),
    }


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)


def find_largest_component(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    best_idx = -1
    best_area = -1
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area > best_area and area >= min_area:
            best_area = area
            best_idx = idx
    if best_idx < 0:
        return np.zeros_like(mask, dtype=bool)
    return labels == best_idx


def clean_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            out |= labels == idx
    return out


def compute_scale_mm_per_px(reference_diameter_mm: Optional[float], reference_diameter_px: Optional[float]) -> Optional[float]:
    if reference_diameter_mm is None or reference_diameter_px is None:
        return None
    if reference_diameter_mm <= 0 or reference_diameter_px <= 0:
        return None
    return reference_diameter_mm / reference_diameter_px


def add_calibrated_measurements(row: Dict[str, object], mm_per_px: Optional[float]) -> Dict[str, object]:
    if mm_per_px is None:
        row["calibration_status"] = "not_calibrated_px_only"
        return row

    row["total_length_mm_basic"] = row["total_length_px_basic"] * mm_per_px
    if row.get("total_length_px_advanced") is not None:
        row["total_length_mm_advanced"] = row["total_length_px_advanced"] * mm_per_px
    row["collar_diameter_mm"] = row["collar_diameter_px"] * mm_per_px
    row["leaf_area_mm2"] = row["leaf_area_px2"] * (mm_per_px**2)
    row["calibration_status"] = "calibrated"
    return row


def save_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image)


def config_to_dict(config_obj: object) -> Dict[str, object]:
    return asdict(config_obj)
