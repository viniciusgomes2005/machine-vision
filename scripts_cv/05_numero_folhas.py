from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

AREA_MIN_FOLHA_COUNT = 250


def contar_folhas_debug(mask_folhas: np.ndarray) -> Tuple[int, Dict[str, np.ndarray]]:
    mask_proc = cv2.morphologyEx(
        mask_folhas,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_proc, connectivity=8)
    validos = np.zeros_like(mask_proc)
    contagem = 0
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < AREA_MIN_FOLHA_COUNT:
            continue
        contagem += 1
        validos[labels == label] = 255

    return int(contagem), {"mask_processada": mask_proc, "mask_componentes_validos": validos}


def contar_folhas(mask_folhas: np.ndarray) -> int:
    contagem, _ = contar_folhas_debug(mask_folhas)
    return int(contagem)
