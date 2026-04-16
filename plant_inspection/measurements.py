"""Morphological measurements extracted from segmented masks."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import MeasurementConfig
from .segmentation import SegmentationResult
from .utils import clean_small_components


@dataclass
class MeasurementResult:
    total_length_px_basic: float
    total_length_px_advanced: Optional[float]
    collar_diameter_px: float
    leaf_area_px2: float
    leaf_count: Optional[int]
    leaf_count_confidence: str
    notes: str
    stem_mask: np.ndarray
    skeleton_mask: np.ndarray
    leaf_mask: np.ndarray


def _skeletonize(binary_mask: np.ndarray) -> np.ndarray:
    """Morphological skeletonization using iterative erosion/opening."""
    img = (binary_mask.astype(np.uint8) * 255).copy()
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break

    return skel > 0


def _nearest_skeleton_point(skeleton: np.ndarray, point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    ys, xs = np.where(skeleton)
    if len(xs) == 0:
        return None
    px, py = point
    d2 = (xs - px) ** 2 + (ys - py) ** 2
    idx = int(np.argmin(d2))
    return int(xs[idx]), int(ys[idx])


def _geodesic_longest_path_length(skeleton: np.ndarray, start_xy: Tuple[int, int]) -> Optional[float]:
    coords = np.argwhere(skeleton)
    if len(coords) == 0:
        return None

    start = (start_xy[1], start_xy[0])
    if not skeleton[start[0], start[1]]:
        nearest = _nearest_skeleton_point(skeleton, start_xy)
        if nearest is None:
            return None
        start = (nearest[1], nearest[0])

    dist = np.full(skeleton.shape, np.inf, dtype=np.float64)
    dist[start] = 0.0

    pq = [(0.0, start[0], start[1])]
    neighbors = [
        (-1, -1, np.sqrt(2.0)),
        (-1, 0, 1.0),
        (-1, 1, np.sqrt(2.0)),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, np.sqrt(2.0)),
        (1, 0, 1.0),
        (1, 1, np.sqrt(2.0)),
    ]

    h, w = skeleton.shape
    while pq:
        d, r, c = heappop(pq)
        if d > dist[r, c]:
            continue
        for dr, dc, weight in neighbors:
            rr, cc = r + dr, c + dc
            if rr < 0 or cc < 0 or rr >= h or cc >= w or not skeleton[rr, cc]:
                continue
            nd = d + weight
            if nd < dist[rr, cc]:
                dist[rr, cc] = nd
                heappush(pq, (nd, rr, cc))

    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return None
    return float(finite.max())


def _basic_length(mask: np.ndarray, collar_point: Tuple[int, int]) -> float:
    ys = np.where(mask)[0]
    if len(ys) == 0:
        return 0.0
    return max(0.0, float(collar_point[1] - ys.min()))


def _build_stem_mask(plant_mask: np.ndarray, collar_point: Tuple[int, int], band_height: int) -> np.ndarray:
    """Approximate stem by retaining narrow connected structures near base."""
    if not np.any(plant_mask):
        return np.zeros_like(plant_mask, dtype=bool)

    dist = cv2.distanceTransform(plant_mask.astype(np.uint8), cv2.DIST_L2, 3)
    narrow = (dist <= 6.0) & plant_mask

    h, w = plant_mask.shape
    x, y = collar_point
    y0 = max(0, y - band_height)
    y1 = min(h, y + 20)
    x0 = max(0, x - 90)
    x1 = min(w, x + 90)

    seed = np.zeros_like(plant_mask, dtype=bool)
    seed[y0:y1, x0:x1] = True
    stem_seed = narrow & seed

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(narrow.astype(np.uint8), connectivity=8)
    stem = np.zeros_like(plant_mask, dtype=bool)
    if num_labels <= 1:
        return stem

    valid_ids = np.unique(labels[stem_seed])
    for idx in valid_ids:
        if idx == 0:
            continue
        if stats[idx, cv2.CC_STAT_AREA] >= 80:
            stem |= labels == idx

    stem = cv2.dilate(stem.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    return stem


def _collar_diameter(plant_mask: np.ndarray, collar_point: Tuple[int, int], cfg: MeasurementConfig) -> float:
    if not np.any(plant_mask):
        return 0.0

    skel = _skeletonize(plant_mask)
    dist = cv2.distanceTransform(plant_mask.astype(np.uint8), cv2.DIST_L2, 3)

    x, y = collar_point
    y_top = max(0, y - cfg.collar_band_height_px)
    y_bottom = max(0, y - cfg.collar_band_offset_px)
    x0 = max(0, x - 90)
    x1 = min(plant_mask.shape[1], x + 90)

    band = np.zeros_like(plant_mask, dtype=bool)
    band[y_top:y_bottom, x0:x1] = True
    band_skel = skel & band

    ys, xs = np.where(band_skel)
    if len(xs) == 0:
        local = dist[y_top:y_bottom, x0:x1]
        vals = local[local > 0]
        if len(vals) == 0:
            return 0.0
        return float(np.median(vals) * 2.0)

    diameters = 2.0 * dist[ys, xs]
    diameters = diameters[diameters > 0]
    if len(diameters) == 0:
        return 0.0

    if len(diameters) >= cfg.collar_samples_min:
        return float(np.median(diameters))
    return float(np.mean(diameters))


def _leaf_segmentation(plant_mask: np.ndarray, stem_mask: np.ndarray, species: str, cfg: MeasurementConfig) -> np.ndarray:
    if species == "eucalyptus":
        leaf = plant_mask & (~stem_mask)
        leaf = clean_small_components(leaf, cfg.eucalyptus_leaf_min_area_px)
    elif species == "pine":
        leaf = plant_mask.copy()
        leaf = clean_small_components(leaf, cfg.pine_leaf_min_area_px)
    else:
        leaf = plant_mask.copy()
    return leaf


def _count_eucalyptus_leaves(leaf_mask: np.ndarray, cfg: MeasurementConfig) -> Tuple[int, str]:
    if not np.any(leaf_mask):
        return 0, "low"

    mask_u8 = (leaf_mask.astype(np.uint8) * 255)

    # Split touching leaves with a moderate opening before connected components.
    sep_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    separated = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, sep_kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(separated, connectivity=8)
    areas = []
    for idx in range(1, n_labels):
        a = stats[idx, cv2.CC_STAT_AREA]
        if a >= cfg.eucalyptus_leaf_min_area_px:
            areas.append(a)

    count = len(areas)

    # Peak-based fallback to reduce undercount when leaves touch strongly.
    if count <= 1:
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
        if dist.max() > 0:
            dilated = cv2.dilate(dist, np.ones((7, 7), np.uint8))
            peaks = (dist == dilated) & (dist > 0.28 * dist.max())
            peak_labels, _, _, _ = cv2.connectedComponentsWithStats(peaks.astype(np.uint8), connectivity=8)
            peak_count = max(0, peak_labels - 1)
            if peak_count > count:
                count = peak_count

    confidence = "medium" if count > 0 else "low"
    if 3 <= count <= 10:
        confidence = "high"
    return count, confidence


def _count_pine_needles_proxy(leaf_mask: np.ndarray) -> Tuple[Optional[int], str, str]:
    if not np.any(leaf_mask):
        return 0, "low", "pine_mask_empty"

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(leaf_mask.astype(np.uint8), connectivity=8)
    count = 0
    for idx in range(1, n_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area >= 200:
            count += 1

    note = "pine_leaf_count_is_cluster_proxy_due_to_needle_overlap"
    return count, "low", note


def compute_measurements(seg: SegmentationResult, species: str, cfg: MeasurementConfig) -> MeasurementResult:
    """Compute all requested morphology metrics from segmentation outputs."""
    plant = seg.plant_above_pot_mask
    if not np.any(plant):
        empty = np.zeros_like(plant, dtype=bool)
        return MeasurementResult(
            total_length_px_basic=0.0,
            total_length_px_advanced=None,
            collar_diameter_px=0.0,
            leaf_area_px2=0.0,
            leaf_count=None,
            leaf_count_confidence="low",
            notes="plant_mask_empty",
            stem_mask=empty,
            skeleton_mask=empty,
            leaf_mask=empty,
        )

    skeleton = _skeletonize(plant)

    length_basic = _basic_length(plant, seg.collar_point)
    length_adv = _geodesic_longest_path_length(skeleton, seg.collar_point)
    collar_diam = _collar_diameter(seg.plant_mask, seg.collar_point, cfg)

    stem_mask = _build_stem_mask(plant, seg.collar_point, cfg.collar_band_height_px)
    leaf_mask = _leaf_segmentation(plant, stem_mask, species, cfg)
    leaf_area = float(np.count_nonzero(leaf_mask))

    note_parts: List[str] = []

    if species == "eucalyptus":
        leaf_count, conf = _count_eucalyptus_leaves(leaf_mask, cfg)
    elif species == "pine":
        leaf_count, conf, note = _count_pine_needles_proxy(leaf_mask)
        note_parts.append(note)
    else:
        cc_count, _, _, _ = cv2.connectedComponentsWithStats(leaf_mask.astype(np.uint8), connectivity=8)
        leaf_count = max(0, cc_count - 1)
        conf = "low"
        note_parts.append("unknown_species_leaf_strategy")

    if length_adv is None:
        note_parts.append("advanced_length_unavailable")
    elif length_basic > 0 and length_adv < 0.25 * length_basic:
        length_adv = float(length_basic)
        note_parts.append("advanced_length_fallback_to_basic")

    if collar_diam <= 0:
        note_parts.append("collar_diameter_unreliable")

    notes = ";".join(note_parts) if note_parts else "ok"

    return MeasurementResult(
        total_length_px_basic=float(length_basic),
        total_length_px_advanced=float(length_adv) if length_adv is not None else None,
        collar_diameter_px=float(collar_diam),
        leaf_area_px2=leaf_area,
        leaf_count=leaf_count,
        leaf_count_confidence=conf,
        notes=notes,
        stem_mask=stem_mask,
        skeleton_mask=skeleton,
        leaf_mask=leaf_mask,
    )
