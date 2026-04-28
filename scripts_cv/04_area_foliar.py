"""
Etapa 4 - area foliar.

A área foliar vem da mascara da planta sem o caule, claro. Primeiro há reforço da máscara
do caule com o skeleton, porque o caule pode ficar fino ou quebrado. Depois tiramos
esse caule da planta, limpamos componentes muito pequenos, fechamos pequenas falhas e
preenchemos contornos das folhas. A área final é simplesmente a contagem de pixels
positivos da mascara de folhas.
Visualmente ainda pegamos o caule. Em iterações anteriores, no trabalho que descartamos, tirávamos bem melhor. Entretanto,
acho que mediamos a imagem distorcida de alguma forma e ficávamos muito longe de sua referência, Dinho. O atual pega o caule visualmente, porém fica muito mais próximo do 
que devia!
"""

from __future__ import annotations

import cv2
import numpy as np

AREA_MIN_FOLHA = 40  # menor componente preservado como folha
AREA_MIN_CONTORNO = 30  # menor contorno preenchido na mascara final
CAULE_DILATE_KERNEL = (7, 7)  # engrossa o caule para remove-lo da area foliar
CAULE_CLOSE_VERTICAL_KERNEL = (5, 25)  # fecha falhas verticais do caule
SKELETON_DILATE_KERNEL = (5, 5)  # tolerancia para juntar skeleton e caule estimado
PLANTA_ESPARSAR_AREA_MAX = 55000  # area maxima para tratar muda alta e esparsa
PLANTA_ESPARSAR_ALTURA_MIN = 780  # altura minima da muda alta e esparsa
CAULE_ALTURA_FRAC_MIN = 0.20  # caule menor que isso e considerado curto demais


def _kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)


def _remover_componentes_pequenos(mask: np.ndarray, area_min: int) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= area_min:
            out[labels == label] = 255
    return out


def _normalizar_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    bin_mask = _normalizar_mask(mask)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(bin_mask)

    skel = np.zeros_like(bin_mask)
    element = _kernel((3, 3))
    img = bin_mask.copy()

    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(eroded, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break

    return skel


def _mascara_caule_por_skeleton(
    mask_planta_acima: np.ndarray,
    mask_caule: np.ndarray,
    skeleton: np.ndarray | None,
) -> np.ndarray:
    planta = _normalizar_mask(mask_planta_acima)
    caule_base = _normalizar_mask(mask_caule)
    skel = _normalizar_mask(skeleton) if skeleton is not None else _skeletonize(planta)

    caule_ref = cv2.dilate(caule_base, _kernel(SKELETON_DILATE_KERNEL), iterations=1)
    skel_caule = cv2.bitwise_and(skel, caule_ref)

    if cv2.countNonZero(skel_caule) == 0:
        skel_caule = cv2.bitwise_and(skel, cv2.dilate(caule_base, _kernel((13, 13)), iterations=1))

    caule = cv2.bitwise_or(caule_base, skel_caule)
    caule = cv2.morphologyEx(caule, cv2.MORPH_CLOSE, _kernel(CAULE_CLOSE_VERTICAL_KERNEL))
    caule = cv2.dilate(caule, _kernel(CAULE_DILATE_KERNEL), iterations=1)
    caule = cv2.bitwise_and(caule, planta)

    ys_planta, _ = np.where(planta > 0)
    ys_caule, _ = np.where(caule_base > 0)
    if ys_planta.size > 0:
        altura_planta = int(ys_planta.max() - ys_planta.min() + 1)
        area_planta = int(np.count_nonzero(planta))
        altura_caule = int(ys_caule.max() - ys_caule.min() + 1) if ys_caule.size > 0 else 0
        caule_curto = float(altura_caule) < CAULE_ALTURA_FRAC_MIN * float(max(1, altura_planta))
        planta_alta_esparsa = area_planta <= PLANTA_ESPARSAR_AREA_MAX and altura_planta >= PLANTA_ESPARSAR_ALTURA_MIN
        if planta_alta_esparsa and caule_curto:
            caule = cv2.bitwise_or(caule, cv2.bitwise_and(skel, planta))

    return caule


def _preencher_por_contornos(mask_folhas: np.ndarray, mask_planta_acima: np.ndarray, mask_caule_removido: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask_folhas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    preenchida = np.zeros_like(mask_folhas)
    if contours:
        for contour in contours:
            if cv2.contourArea(contour) < AREA_MIN_CONTORNO:
                continue
            cv2.drawContours(preenchida, [contour], -1, 255, cv2.FILLED)

    preenchida = cv2.bitwise_and(preenchida, _normalizar_mask(mask_planta_acima))
    preenchida = cv2.bitwise_and(preenchida, cv2.bitwise_not(mask_caule_removido))
    return preenchida


def obter_mascara_folhas_debug(
    mask_planta_acima: np.ndarray,
    mask_caule: np.ndarray,
    skeleton: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    planta = _normalizar_mask(mask_planta_acima)
    caule_removido = _mascara_caule_por_skeleton(planta, mask_caule, skeleton)
    sem_caule = cv2.bitwise_and(planta, cv2.bitwise_not(caule_removido))

    mask_folhas = cv2.morphologyEx(sem_caule, cv2.MORPH_OPEN, _kernel((3, 3)))
    mask_folhas = cv2.morphologyEx(mask_folhas, cv2.MORPH_CLOSE, _kernel((7, 7)))
    mask_folhas = _remover_componentes_pequenos(mask_folhas, AREA_MIN_FOLHA)
    preenchida = _preencher_por_contornos(mask_folhas, planta, caule_removido)

    debug = {
        "mask_caule_removido_area": caule_removido,
        "mask_folhas_sem_caule": sem_caule,
        "mask_folhas_morfologia": mask_folhas,
        "mask_folhas_contornos": preenchida,
    }
    return preenchida, debug


def obter_mascara_folhas(
    mask_planta_acima: np.ndarray,
    mask_caule: np.ndarray,
    skeleton: np.ndarray | None = None,
) -> np.ndarray:
    mask_folhas, _ = obter_mascara_folhas_debug(mask_planta_acima, mask_caule, skeleton=skeleton)
    return mask_folhas


def medir_area_foliar(mask_folhas: np.ndarray) -> int:
    return int(np.count_nonzero(mask_folhas))
