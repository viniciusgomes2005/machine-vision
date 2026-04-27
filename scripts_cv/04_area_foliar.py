from __future__ import annotations

import cv2
import numpy as np

AREA_MIN_FOLHA = 40


def _kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)


def _remover_componentes_pequenos(mask: np.ndarray, area_min: int) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= area_min:
            out[labels == label] = 255
    return out


def obter_mascara_folhas(mask_planta_acima: np.ndarray, mask_caule: np.ndarray) -> np.ndarray:
    sem_caule = cv2.bitwise_and(mask_planta_acima, cv2.bitwise_not(mask_caule))

    mask_folhas = cv2.morphologyEx(sem_caule, cv2.MORPH_OPEN, _kernel((3, 3)))
    mask_folhas = cv2.morphologyEx(mask_folhas, cv2.MORPH_CLOSE, _kernel((7, 7)))
    mask_folhas = _remover_componentes_pequenos(mask_folhas, AREA_MIN_FOLHA)

    contours, _ = cv2.findContours(mask_folhas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    preenchida = np.zeros_like(mask_folhas)
    if contours:
        cv2.drawContours(preenchida, contours, -1, 255, cv2.FILLED)
    return preenchida


def medir_area_foliar(mask_folhas: np.ndarray) -> int:
    return int(np.count_nonzero(mask_folhas))
