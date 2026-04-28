from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

BASE_SCAN_UP = 40
BASE_SCAN_DOWN = 0
MARGEM_SEGURA_VASO = 5
MAX_BASE_WIDTH = 32
SEED_HALF_W = 8
SEED_HALF_H = 4
AREA_MIN_CONNECTED = 35
MAX_CANDIDATOS = 80
BASE_TRACK_HALF_W = 4
BASE_RESIDUO_GAP_MAX = 8
BASE_RESIDUO_W_MAX = 2
BASE_RESIDUO_TOPO_DELTA_MIN = 90
BASE_RESIDUO_TOPO_DELTA_MAX = 260
BASE_RESIDUO_SUBIR_FRAC = 0.042
BASE_RESIDUO_SUBIR_MIN = 18
BASE_RESIDUO_SUBIR_MAX = 42
STEM_INVISIBLE_GAP = 10
HSV_BLUE_LOW = np.array([75, 40, 40], dtype=np.uint8)
HSV_BLUE_HIGH = np.array([140, 255, 255], dtype=np.uint8)
TOP_RAW_HALF_W = 90
TOP_RAW_AREA_MIN = 12


def _segmentos_horizontais(row_bool: np.ndarray):
    xs = np.where(row_bool)[0]
    if xs.size == 0:
        return []

    segs = []
    ini = int(xs[0])
    fim = int(xs[0])
    for x in xs[1:]:
        x = int(x)
        if x == fim + 1:
            fim = x
        else:
            segs.append((ini, fim))
            ini = x
            fim = x
    segs.append((ini, fim))
    return segs


def _coletar_candidatos_base(
    mask_planta_acima: np.ndarray,
    y_topo_tubete: int,
) -> list[Tuple[int, int, int]]:
    h, _ = mask_planta_acima.shape
    y_bottom = max(0, min(h - 1, int(y_topo_tubete) - MARGEM_SEGURA_VASO))
    y_top = max(0, y_bottom - BASE_SCAN_UP)

    candidatos: list[Tuple[int, int, int]] = []
    for y in range(y_bottom, y_top - 1, -1):
        row = mask_planta_acima[y, :] > 0
        segs = _segmentos_horizontais(row)
        for x0, x1 in segs:
            largura = x1 - x0 + 1
            if largura > MAX_BASE_WIDTH:
                continue
            xc = int((x0 + x1) / 2)
            candidatos.append((xc, int(y), int(largura)))
        if len(candidatos) >= MAX_CANDIDATOS:
            break

    return candidatos[:MAX_CANDIDATOS]


def _componente_conectado_da_seed(mask: np.ndarray, x_seed: int, y_seed: int) -> np.ndarray:
    h, w = mask.shape
    seed = np.zeros_like(mask)

    x0 = max(0, x_seed - SEED_HALF_W)
    x1 = min(w - 1, x_seed + SEED_HALF_W)
    y0 = max(0, y_seed - SEED_HALF_H)
    y1 = min(h - 1, y_seed + SEED_HALF_H)
    seed[y0 : y1 + 1, x0 : x1 + 1] = mask[y0 : y1 + 1, x0 : x1 + 1]

    if np.count_nonzero(seed) == 0:
        return np.zeros_like(mask)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    seed_bool = seed > 0

    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < AREA_MIN_CONNECTED:
            continue
        comp = labels == label
        if np.any(comp & seed_bool):
            out[comp] = 255

    return out


def _obter_topo_conectado(
    mask_source: np.ndarray,
    x_base: int,
    y_base: int,
) -> Optional[int]:
    comp = _componente_conectado_da_seed(mask_source, int(x_base), int(y_base))
    ys, _ = np.where(comp > 0)
    if ys.size == 0:
        return None
    return int(ys.min())


def _tem_suporte_vertical(mask: np.ndarray, x: int, y: int, altura: int = 34, meia_largura: int = 2, min_pixels: int = 10) -> bool:
    h, w = mask.shape
    y0 = max(0, int(y) - int(altura))
    y1 = max(0, int(y) - 1)
    if y1 < y0:
        return False
    x0 = max(0, int(x) - int(meia_largura))
    x1 = min(w - 1, int(x) + int(meia_largura))
    return int(np.count_nonzero(mask[y0 : y1 + 1, x0 : x1 + 1])) >= int(min_pixels)


def _topo_raw_local(
    img_bgr: np.ndarray,
    x_base: int,
    y_topo_tubete: int,
) -> Optional[int]:
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_bg = cv2.inRange(hsv, HSV_BLUE_LOW, HSV_BLUE_HIGH)
    mask_obj = cv2.bitwise_not(mask_bg)
    y_lim = max(0, int(y_topo_tubete) - MARGEM_SEGURA_VASO)
    mask_obj[y_lim:, :] = 0

    x0 = max(0, int(x_base) - TOP_RAW_HALF_W)
    x1 = min(w - 1, int(x_base) + TOP_RAW_HALF_W)
    roi = mask_obj[:, x0 : x1 + 1]

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    y_top = None
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < TOP_RAW_AREA_MIN:
            continue
        y = int(stats[label, cv2.CC_STAT_TOP])
        if y_top is None or y < y_top:
            y_top = y
    return y_top


def obter_pontos_altura_vertical(
    mask_planta_acima: np.ndarray,
    ponto_base: Tuple[int, int],
    mask_objetos: Optional[np.ndarray] = None,
    y_topo_tubete: Optional[int] = None,
    img_bgr: Optional[np.ndarray] = None,
    x_centro_tubete: Optional[int] = None,
    mask_caule: Optional[np.ndarray] = None,
) -> Tuple[int, int, int]:
    _ = img_bgr, mask_caule

    mask_planta_local = mask_planta_acima.copy()
    if y_topo_tubete is not None:
        y_limite = max(0, int(y_topo_tubete) - MARGEM_SEGURA_VASO)
        mask_planta_local[y_limite:, :] = 0
    else:
        y_limite = None

    ys_planta, _ = np.where(mask_planta_local > 0)
    if ys_planta.size == 0:
        return int(ponto_base[0]), int(ponto_base[1]), int(ponto_base[1])

    y_top_global = int(ys_planta.min())

    if y_topo_tubete is None:
        x_base = int(ponto_base[0])
        y_base = int(ponto_base[1])
        return x_base, y_base, y_top_global

    h, _ = mask_planta_local.shape
    y_base_lim = max(0, min(h - 1, int(y_topo_tubete) - MARGEM_SEGURA_VASO))
    altura_total = int(max(1, y_base_lim - y_top_global))

    if mask_objetos is not None:
        mask_source = mask_objetos.copy()
        mask_source[y_base_lim:, :] = 0
    else:
        mask_source = mask_planta_local.copy()
        mask_source[y_base_lim:, :] = 0

    candidatos = _coletar_candidatos_base(mask_planta_local, int(y_topo_tubete))
    if not candidatos:
        x_ref = int(x_centro_tubete if x_centro_tubete is not None else ponto_base[0])
        return x_ref, y_base_lim, y_top_global

    x_pref = int(ponto_base[0])
    if x_centro_tubete is not None:
        x_pref = int(0.65 * int(x_centro_tubete) + 0.35 * int(ponto_base[0]))

    melhor = None
    melhor_score = -1e12

    for x_base, y_base, largura in candidatos:
        if int(y_base) >= int(y_base_lim):
            continue
        y_top_conn = _obter_topo_conectado(mask_source, x_base, y_base)
        if y_top_conn is None:
            continue
        span = float(y_base - y_top_conn)
        dist_pref = abs(int(x_base) - x_pref)
        score = span - 0.06 * float(dist_pref) - 0.20 * float(largura)
        if score > melhor_score:
            melhor_score = score
            melhor = (int(x_base), int(y_base), int(y_top_conn), int(largura), int(span), int(dist_pref))

    if melhor is None:
        x_ref = int(x_centro_tubete if x_centro_tubete is not None else ponto_base[0])
        return x_ref, y_base_lim, y_top_global

    x_base, y_base, y_top_conn, largura_base, _span, _dist_pref = melhor
    # Refina base: desce seguindo o mesmo eixo x do caule para pegar
    # o primeiro ponto real do caule, mesmo com fusao local da mascara.
    y_max_track = min(h - 1, int(y_base_lim))
    for y in range(int(y_base) + 1, y_max_track + 1):
        x0 = max(0, int(x_base) - BASE_TRACK_HALF_W)
        x1 = min(mask_planta_local.shape[1] - 1, int(x_base) + BASE_TRACK_HALF_W)
        if np.count_nonzero(mask_planta_local[y, x0 : x1 + 1]) > 0:
            y_base = int(y)
        else:
            break

    # Residuo na base: quando a base fica colada ao topo do vaso com segmento fino,
    # corrige para a emergencia da muda alguns pixels acima.
    gap_vaso = int(y_base_lim) - int(y_base)
    topo_delta = int(y_top_conn) - int(y_top_global)
    if (
        gap_vaso <= BASE_RESIDUO_GAP_MAX
        and int(largura_base) <= BASE_RESIDUO_W_MAX
        and topo_delta >= BASE_RESIDUO_TOPO_DELTA_MIN
        and topo_delta <= BASE_RESIDUO_TOPO_DELTA_MAX
    ):
        subir = int(max(BASE_RESIDUO_SUBIR_MIN, min(BASE_RESIDUO_SUBIR_MAX, BASE_RESIDUO_SUBIR_FRAC * altura_total)))
        y_base = max(0, int(y_base) - subir)

    y_base = min(int(y_base), y_base_lim)
    y_topo = min(int(y_top_global), int(y_top_conn))
    if img_bgr is not None:
        y_top_raw = _topo_raw_local(img_bgr, int(x_base), int(y_topo_tubete))
        if y_top_raw is not None:
            y_topo = min(int(y_topo), int(y_top_raw))

    return int(x_base), int(y_base), int(y_topo)


def medir_comprimento_vertical(
    mask_planta_acima: np.ndarray,
    ponto_base: Tuple[int, int],
    mask_objetos: Optional[np.ndarray] = None,
    y_topo_tubete: Optional[int] = None,
    img_bgr: Optional[np.ndarray] = None,
    x_centro_tubete: Optional[int] = None,
    mask_caule: Optional[np.ndarray] = None,
) -> int:
    _ = mask_caule
    _, y_base, y_topo = obter_pontos_altura_vertical(
        mask_planta_acima=mask_planta_acima,
        ponto_base=ponto_base,
        mask_objetos=mask_objetos,
        y_topo_tubete=y_topo_tubete,
        img_bgr=img_bgr,
        x_centro_tubete=x_centro_tubete,
    )
    return int(max(0, y_base - y_topo))
