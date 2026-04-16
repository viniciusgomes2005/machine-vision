"""Segmentation routines for plant, pot, and reference object masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config import SegmentationConfig
from .utils import clean_small_components, find_largest_component


@dataclass
class SegmentationResult:
    background_mask: np.ndarray
    foreground_mask: np.ndarray
    plant_mask: np.ndarray
    pot_mask: np.ndarray
    white_cylinder_mask: np.ndarray
    plant_above_pot_mask: np.ndarray
    pot_top_y: int
    pot_bbox: Tuple[int, int, int, int]
    pot_center: Tuple[int, int]
    collar_point: Tuple[int, int]
    reference_diameter_px: Optional[float]


def _morph(mask: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    out = mask.astype(np.uint8)
    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out.astype(bool)


def _largest_mask_in_roi(mask: np.ndarray, y_min: int, x_center: int, x_half_window: int) -> np.ndarray:
    h, w = mask.shape
    roi = np.zeros_like(mask, dtype=bool)
    x0 = max(0, x_center - x_half_window)
    x1 = min(w, x_center + x_half_window)
    roi[y_min:, x0:x1] = True
    candidate = mask & roi
    return find_largest_component(candidate)


def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1, y1)


def _pot_features(pot_mask: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[int, Tuple[int, int, int, int], Tuple[int, int]]:
    h, w = image_shape
    if not np.any(pot_mask):
        return int(0.62 * h), (0, 0, 0, 0), (w // 2, int(0.62 * h))

    bbox = _mask_bbox(pot_mask)
    x0, y0, x1, y1 = bbox
    pot_top_y = y0
    center = ((x0 + x1) // 2, (y0 + y1) // 2)
    return pot_top_y, bbox, center


def _find_collar_point(plant_mask: np.ndarray, pot_top_y: int, pot_center_x: int, search_above: int = 100, search_below: int = 20, center_half_window: int = 140) -> Tuple[int, int]:
    h, w = plant_mask.shape
    y0 = max(0, pot_top_y - search_above)
    y1 = min(h, pot_top_y + search_below)
    x0 = max(0, pot_center_x - center_half_window)
    x1 = min(w, pot_center_x + center_half_window)

    ys, xs = np.where(plant_mask[y0:y1, x0:x1])
    if len(xs) == 0:
        return (pot_center_x, pot_top_y)

    ys = ys + y0
    xs = xs + x0

    # Closest plant support point to the pot top and center line.
    score = np.abs(ys - pot_top_y) * 2.0 + np.abs(xs - pot_center_x)
    idx = int(np.argmin(score))
    return (int(xs[idx]), int(ys[idx]))


def _estimate_reference_diameter_px(white_mask: np.ndarray) -> Optional[float]:
    if not np.any(white_mask):
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask.astype(np.uint8), connectivity=8)
    best_idx = -1
    best_score = -1.0
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        if area < 1000:
            continue
        score = area * (min(w, h) / (max(w, h) + 1e-6))
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx < 0:
        return None

    x = stats[best_idx, cv2.CC_STAT_LEFT]
    y = stats[best_idx, cv2.CC_STAT_TOP]
    w = stats[best_idx, cv2.CC_STAT_WIDTH]
    h = stats[best_idx, cv2.CC_STAT_HEIGHT]
    component = labels[y : y + h, x : x + w] == best_idx

    ys, xs = np.where(component)
    if len(xs) < 10:
        return None

    diam_x = xs.max() - xs.min() + 1
    diam_y = ys.max() - ys.min() + 1
    return float((diam_x + diam_y) / 2.0)


def segment_scene(image_bgr: np.ndarray, config: SegmentationConfig) -> SegmentationResult:
    """Segment plant and support objects using deterministic color logic."""
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    blue_bg = (
        (H >= config.blue_hue_range[0])
        & (H <= config.blue_hue_range[1])
        & (S >= config.blue_s_min)
        & (V >= config.blue_v_min)
    )
    foreground = ~blue_bg
    foreground = _morph(foreground, config.open_kernel, config.close_kernel)
    foreground = clean_small_components(foreground, config.min_component_area_px)

    black_mask = (V <= config.black_v_max) & (S <= config.black_s_max)
    black_mask &= foreground
    black_mask = _morph(black_mask, 3, 5)

    white_mask = (V >= config.white_v_min) & (S <= config.white_s_max)
    white_mask &= foreground
    white_mask = _morph(white_mask, 3, 5)

    lower_start = int(0.45 * h)
    pot_mask = _largest_mask_in_roi(black_mask, y_min=lower_start, x_center=w // 2, x_half_window=int(w * 0.26))

    white_candidate = _largest_mask_in_roi(white_mask, y_min=int(0.50 * h), x_center=w // 2, x_half_window=int(w * 0.30))

    plant_color = (
        (H >= config.plant_hue_range[0])
        & (H <= config.plant_hue_range[1])
        & (S >= config.plant_s_min)
        & (V >= config.plant_v_min)
    )

    plant_mask = foreground & (~pot_mask) & (~white_candidate)
    plant_mask &= (plant_color | (S >= 35))
    plant_mask = _morph(plant_mask, config.open_kernel, config.close_kernel)
    plant_mask = clean_small_components(plant_mask, config.min_component_area_px)

    pot_top_y, pot_bbox, pot_center = _pot_features(pot_mask, (h, w))

    plant_above = plant_mask.copy()
    plant_above[pot_top_y + 4 :, :] = False
    plant_above = clean_small_components(plant_above, config.min_component_area_px)

    collar_point = _find_collar_point(plant_mask, pot_top_y, pot_center[0])

    ref_diameter = _estimate_reference_diameter_px(white_candidate)

    return SegmentationResult(
        background_mask=blue_bg,
        foreground_mask=foreground,
        plant_mask=plant_mask,
        pot_mask=pot_mask,
        white_cylinder_mask=white_candidate,
        plant_above_pot_mask=plant_above,
        pot_top_y=pot_top_y,
        pot_bbox=pot_bbox,
        pot_center=pot_center,
        collar_point=collar_point,
        reference_diameter_px=ref_diameter,
    )
