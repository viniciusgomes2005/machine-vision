from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

AREA_MIN_FOLHA_COUNT = 250
AREA_MIN_FOLHA_FRACA = 150
AREA_MUDA_PEQUENA = 35000
AREA_MUDA_MUITO_PEQUENA = 22000
AREA_MUDA_MEDIA = 80000
AREA_MUDA_GRANDE = 100000
COUNT_MIN_CONFIAVEL = 10
RAZAO_COMPONENTE_DOMINANTE = 0.75
AREA_MIN_COMPONENTE_BILOBADO = 3000
DEFECT_PROFUNDIDADE_BILOBADO = 18.0


def _abrir(mask_folhas: np.ndarray, kernel_size: int) -> np.ndarray:
    return cv2.morphologyEx(
        mask_folhas,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    )


def _contar_componentes(mask_proc: np.ndarray, area_min: int) -> Tuple[int, np.ndarray, int]:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_proc, connectivity=8)
    validos = np.zeros_like(mask_proc)
    contagem = 0
    maior_area = 0
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        maior_area = max(maior_area, area)
        if area < area_min:
            continue
        contagem += 1
        validos[labels == label] = 255

    return int(contagem), validos, int(maior_area)


def _selecionar_mascara_contagem(mask_folhas: np.ndarray) -> Tuple[np.ndarray, int]:
    mask_base = _abrir(mask_folhas, 3)
    contagem_base, _, maior_area = _contar_componentes(mask_base, AREA_MIN_FOLHA_COUNT)
    area_total = int(np.count_nonzero(mask_base))
    razao_dominante = float(maior_area) / float(area_total) if area_total > 0 else 0.0

    if contagem_base >= COUNT_MIN_CONFIAVEL:
        return mask_base, AREA_MIN_FOLHA_COUNT

    if area_total <= AREA_MUDA_PEQUENA and contagem_base <= 5:
        return _abrir(mask_folhas, 15), AREA_MIN_FOLHA_COUNT

    if area_total >= AREA_MUDA_GRANDE and razao_dominante >= RAZAO_COMPONENTE_DOMINANTE:
        return _abrir(mask_folhas, 13), AREA_MIN_FOLHA_COUNT

    if area_total >= AREA_MUDA_GRANDE and contagem_base < COUNT_MIN_CONFIAVEL:
        return _abrir(mask_folhas, 13), AREA_MIN_FOLHA_COUNT

    if area_total <= AREA_MUDA_MEDIA and contagem_base < COUNT_MIN_CONFIAVEL:
        return _abrir(mask_folhas, 9), AREA_MIN_FOLHA_FRACA

    return mask_base, AREA_MIN_FOLHA_COUNT


def _estimar_bilobos_muda_pequena(mask_proc: np.ndarray, contagem: int) -> int:
    area_total = int(np.count_nonzero(mask_proc))
    if area_total > AREA_MUDA_MUITO_PEQUENA or contagem != 2:
        return int(contagem)

    contours, _ = cv2.findContours(mask_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    estimada = 0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < AREA_MIN_FOLHA_COUNT:
            continue

        folhas_no_contorno = 1
        if area >= AREA_MIN_COMPONENTE_BILOBADO:
            hull = cv2.convexHull(contour, returnPoints=False)
            if hull is not None and len(hull) > 3:
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None:
                    profundos = [
                        float(depth) / 256.0
                        for _, _, _, depth in defects[:, 0, :]
                        if float(depth) / 256.0 >= DEFECT_PROFUNDIDADE_BILOBADO
                    ]
                    if profundos:
                        folhas_no_contorno = 2

        estimada += folhas_no_contorno

    return max(int(contagem), int(estimada))


def contar_folhas_debug(mask_folhas: np.ndarray) -> Tuple[int, Dict[str, np.ndarray]]:
    mask_proc, area_min = _selecionar_mascara_contagem(mask_folhas)
    contagem, validos, _ = _contar_componentes(mask_proc, area_min)
    contagem = _estimar_bilobos_muda_pequena(_abrir(mask_folhas, 3), contagem)

    return int(contagem), {"mask_processada": mask_proc, "mask_componentes_validos": validos}


def contar_folhas(mask_folhas: np.ndarray) -> int:
    contagem, _ = contar_folhas_debug(mask_folhas)
    return int(contagem)
