from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

JANELA_DEFAULT = (-14, -8)
LARGURA_MAX_FRAC = 0.08
DIST_MAX_FRAC = 0.10
EXPANSAO_MAX = 20
HALF_WINDOW_X = 10


def _segmentos_horizontais(row_bool: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.where(row_bool)[0]
    if idx.size == 0:
        return []

    segmentos = []
    inicio = int(idx[0])
    fim = int(idx[0])
    for x in idx[1:]:
        x = int(x)
        if x == fim + 1:
            fim = x
        else:
            segmentos.append((inicio, fim))
            inicio = x
            fim = x
    segmentos.append((inicio, fim))
    return segmentos


def _medir_larguras_validas(
    mask_planta_acima: np.ndarray,
    y_min: int,
    y_max: int,
    x_ref: int,
    mask_caule: Optional[np.ndarray] = None,
) -> Tuple[List[int], List[Dict[str, int]]]:
    h, w = mask_planta_acima.shape
    larguras = []
    detalhes = []
    largura_max = max(4, int(w * LARGURA_MAX_FRAC))
    dist_max = max(20, int(w * DIST_MAX_FRAC))

    for y in range(max(0, y_min), min(h - 1, y_max) + 1):
        row = mask_planta_acima[y, :] > 0
        segmentos = _segmentos_horizontais(row)
        if not segmentos:
            continue

        x_row_ref = x_ref
        if mask_caule is not None:
            xs_caule = np.where(mask_caule[y, :] > 0)[0]
            if xs_caule.size > 0:
                x_row_ref = int(np.median(xs_caule))

        melhor = None
        melhor_score = 10**9
        for x0, x1 in segmentos:
            wx0 = max(x0, x_row_ref - HALF_WINDOW_X)
            wx1 = min(x1, x_row_ref + HALF_WINDOW_X)
            if wx0 > wx1:
                continue

            largura = wx1 - wx0 + 1
            if largura > largura_max:
                continue
            x_centro = (wx0 + wx1) // 2
            dist = abs(x_centro - x_row_ref)
            if dist > dist_max:
                continue
            # Se houver referencia por mask_caule, prioriza segmento que contem esse x.
            if mask_caule is not None and (wx0 <= x_row_ref <= wx1):
                score = 0.1 * float(largura)
            else:
                score = float(dist) + 0.2 * float(largura)
            if score < melhor_score:
                melhor_score = score
                melhor = (wx0, wx1, largura)

        if melhor is None:
            continue

        x0, x1, largura = melhor
        larguras.append(int(largura))
        detalhes.append({"y": int(y), "x0": int(x0), "x1": int(x1), "largura": int(largura)})

    return larguras, detalhes


def _estimar_por_mascara(mask_base: np.ndarray, y_topo_tubete: int, x_ref: int):
    y_min = y_topo_tubete + JANELA_DEFAULT[0]
    y_max = y_topo_tubete + JANELA_DEFAULT[1]

    larguras, detalhes = _medir_larguras_validas(mask_base, y_min, y_max, x_ref)
    expansao = 0
    while not larguras and expansao <= EXPANSAO_MAX:
        expansao += 4
        larguras, detalhes = _medir_larguras_validas(mask_base, y_min - expansao, y_max + expansao, x_ref)

    if not larguras:
        return 0, None

    diametro = int(np.median(np.array(larguras, dtype=np.float32)))
    linha = min(detalhes, key=lambda d: abs(d["largura"] - diametro))
    return diametro, linha


def medir_diametro_coleto_debug(
    mask_planta_acima: np.ndarray,
    y_topo_tubete: int,
    x_base_ou_x_ref: int,
    mask_caule: Optional[np.ndarray] = None,
) -> Tuple[int, Dict[str, Optional[int]]]:
    # Primeiro tenta uma versao erodida para reduzir interferencia de folhas largas.
    stem_hint = cv2.erode(
        mask_planta_acima,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    y_min = y_topo_tubete + JANELA_DEFAULT[0]
    y_max = y_topo_tubete + JANELA_DEFAULT[1]
    diametro, linha = 0, None
    if mask_caule is not None:
        larguras, detalhes = _medir_larguras_validas(
            stem_hint, y_min, y_max, x_base_ou_x_ref, mask_caule=mask_caule
        )
        if larguras:
            diametro = int(np.median(np.array(larguras, dtype=np.float32)))
            linha = min(detalhes, key=lambda d: abs(d["largura"] - diametro))

    if diametro <= 0:
        diametro, linha = _estimar_por_mascara(stem_hint, y_topo_tubete, x_base_ou_x_ref)
    if diametro <= 0:
        diametro, linha = _estimar_por_mascara(mask_planta_acima, y_topo_tubete, x_base_ou_x_ref)

    if diametro <= 0 or linha is None:
        return 0, {"y": None, "x0": None, "x1": None, "largura": 0}
    return diametro, linha


def medir_diametro_coleto(
    mask_planta_acima: np.ndarray,
    y_topo_tubete: int,
    x_base_ou_x_ref: int,
    mask_caule: Optional[np.ndarray] = None,
) -> int:
    diametro, _ = medir_diametro_coleto_debug(
        mask_planta_acima, y_topo_tubete, x_base_ou_x_ref, mask_caule=mask_caule
    )
    return int(diametro)
