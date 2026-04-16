"""Preprocessing helpers, including input inspection and normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .utils import image_color_stats, load_bgr_image, save_image


def inspect_images(image_paths: List[Path]) -> List[Dict[str, object]]:
    """Return per-image metadata and color statistics."""
    stats = []
    for path in image_paths:
        image = load_bgr_image(path)
        h, w = image.shape[:2]
        color = image_color_stats(image)
        stats.append(
            {
                "image_name": path.name,
                "height": h,
                "width": w,
                **color,
            }
        )
    return stats


def build_preview_mosaic(image_paths: List[Path], max_images: int = 12, tile_size: int = 320) -> np.ndarray:
    """Create a quick visual contact sheet for sanity checks."""
    images = []
    for path in image_paths[:max_images]:
        img = load_bgr_image(path)
        img = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        label = path.name
        cv2.putText(img, label, (10, tile_size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(img, label, (10, tile_size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        images.append(img)

    if not images:
        return np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

    n = len(images)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))

    blank = np.zeros_like(images[0])
    grid_rows = []
    idx = 0
    for _ in range(rows):
        row_tiles = []
        for _ in range(cols):
            if idx < n:
                row_tiles.append(images[idx])
            else:
                row_tiles.append(blank)
            idx += 1
        grid_rows.append(np.hstack(row_tiles))
    return np.vstack(grid_rows)


def save_preview(image_paths: List[Path], output_path: Path) -> None:
    preview = build_preview_mosaic(image_paths)
    save_image(output_path, preview)
