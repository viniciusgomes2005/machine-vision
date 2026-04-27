from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

HSV_BLUE_LOW = np.array([75, 40, 40], dtype=np.uint8)
HSV_BLUE_HIGH = np.array([140, 255, 255], dtype=np.uint8)
K_OPEN = (3, 3)
K_CLOSE = (7, 7)
AREA_MIN_OBJ = 40
AREA_MIN_TUBETE = 1500
MARGEM_ACIMA_TUBETE = 2
MARGEM_SEGURA_VASO = 5
BASE_BAND_UP = 22
BASE_BAND_DOWN = 2
BASE_HALF_WIDTH_KEEP = 16
MAX_LARGURA_BASE_COLETO = 30
BOCA_PIX_ACIMA = 26
BOCA_PIX_ABAIXO = 16
BOCA_MEIA_LARGURA = 180
HSV_VASO_S_MAX = 120
HSV_VASO_V_MAX = 145
MIN_SUPORTE_VERTICAL = 12
ALTURA_SUPORTE_VERTICAL = 35
MEIA_LARGURA_SUPORTE = 4
ROI_INICIO_TUBETE_FRAC = 0.45
LARGURA_MIN_TUBETE_FRAC = 0.08
LINHAS_CONSECUTIVAS_TOPO = 8
OFFSET_Y_BORDA_TRASEIRA = 2
PERSISTENCIA_BORDA_TRASEIRA = 4
OFFSET_Y_MEDIA_CONFIANCA = 6
OFFSET_Y_BAIXA_CONFIANCA = 2


def _odd(v: int) -> int:
    return v if v % 2 == 1 else v + 1


def ler_imagem(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Nao foi possivel ler imagem: {path}")
    return img_bgr


def _kernel(size: Tuple[int, int]) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)


def _limpar_componentes_pequenos(mask: np.ndarray, area_min: int) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    saida = np.zeros_like(mask)
    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= area_min:
            saida[labels == label] = 255
    return saida


def segmentar_fundo_azul(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_fundo = cv2.inRange(hsv, HSV_BLUE_LOW, HSV_BLUE_HIGH)
    mask_fundo = cv2.morphologyEx(mask_fundo, cv2.MORPH_OPEN, _kernel(K_OPEN))
    mask_fundo = cv2.morphologyEx(mask_fundo, cv2.MORPH_CLOSE, _kernel(K_CLOSE))
    return mask_fundo


def obter_mask_objetos(img_bgr: np.ndarray) -> np.ndarray:
    mask_fundo = segmentar_fundo_azul(img_bgr)
    mask_objetos = cv2.bitwise_not(mask_fundo)
    mask_objetos = cv2.morphologyEx(mask_objetos, cv2.MORPH_OPEN, _kernel(K_OPEN))
    mask_objetos = cv2.morphologyEx(mask_objetos, cv2.MORPH_CLOSE, _kernel((5, 5)))
    return _limpar_componentes_pequenos(mask_objetos, AREA_MIN_OBJ)


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1))


def _segmento_principal_linha(mask: np.ndarray, y: int, x_ref: Optional[int] = None) -> Optional[Tuple[int, int]]:
    segs = _segmentos_horizontais(mask[y, :])
    if not segs:
        return None
    if x_ref is None:
        return max(segs, key=lambda s: (s[1] - s[0] + 1))

    melhor = None
    melhor_score = 1e12
    for x0, x1 in segs:
        xc = 0.5 * (x0 + x1)
        largura = (x1 - x0 + 1)
        score = abs(xc - x_ref) - 0.2 * largura
        if score < melhor_score:
            melhor_score = score
            melhor = (int(x0), int(x1))
    return melhor


def _mascara_candidata_vaso(
    img_bgr: np.ndarray,
    mask_objetos: np.ndarray,
    roi_inicio_y: int,
    area_min_tubete: int,
) -> np.ndarray:
    _ = img_bgr
    h, w = mask_objetos.shape
    roi_obj = np.zeros_like(mask_objetos)
    roi_obj[roi_inicio_y:, :] = mask_objetos[roi_inicio_y:, :]

    # Estratégia principal: vaso é corpo grande/regular no rodapé.
    # Abertura elíptica grande remove estruturas finas (caule/ramificações).
    k_open = _odd(max(13, int(w * 0.028)))
    k_close = _odd(max(23, int(w * 0.050)))
    k_refine = _odd(max(7, int(w * 0.014)))

    candidato = cv2.morphologyEx(roi_obj, cv2.MORPH_OPEN, _kernel((k_open, k_open)))
    candidato = cv2.morphologyEx(candidato, cv2.MORPH_CLOSE, _kernel((k_close, k_close)))
    candidato = cv2.morphologyEx(candidato, cv2.MORPH_CLOSE, _kernel((k_refine, k_refine)))
    candidato = cv2.morphologyEx(candidato, cv2.MORPH_OPEN, _kernel((5, 5)))
    candidato = _limpar_componentes_pequenos(candidato, max(120, area_min_tubete // 8))

    # Fallback se abertura agressiva zerar demais.
    if np.count_nonzero(candidato) < area_min_tubete:
        candidato = cv2.morphologyEx(roi_obj, cv2.MORPH_CLOSE, _kernel((k_refine, k_refine)))
        candidato = cv2.morphologyEx(candidato, cv2.MORPH_OPEN, _kernel((7, 7)))
        candidato = _limpar_componentes_pequenos(candidato, max(120, area_min_tubete // 10))

    return candidato


def _escolher_corpo_principal_vaso(mask_vaso_candidata: np.ndarray, roi_inicio_y: int) -> Tuple[np.ndarray, bool, str]:
    h, w = mask_vaso_candidata.shape
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_vaso_candidata, connectivity=8)

    melhor_label = 0
    melhor_score = -1e12
    for label in range(1, n_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 120:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        y_bottom = y + bh - 1
        if y_bottom < int(h * 0.70):
            continue

        fill = area / max(1.0, float(bw * bh))
        compact = min(bw, bh) / max(1.0, float(max(bw, bh)))
        score = area + 2.2 * y_bottom + 200.0 * fill + 60.0 * compact
        if score > melhor_score:
            melhor_score = score
            melhor_label = label

    if melhor_label == 0:
        out = np.zeros_like(mask_vaso_candidata)
        out[roi_inicio_y:, :] = mask_vaso_candidata[roi_inicio_y:, :]
        return out, False, 'fallback_roi_inferior'

    out = np.zeros_like(mask_vaso_candidata)
    out[labels == melhor_label] = 255
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, _kernel((5, 5)))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, _kernel((3, 3)))
    return out, True, 'corpo_principal'


def _expandir_corpo_com_objetos(mask_corpo: np.ndarray, mask_objetos: np.ndarray, roi_inicio_y: int) -> np.ndarray:
    roi_obj = mask_objetos.copy()
    roi_obj[:roi_inicio_y, :] = 0
    seed = cv2.dilate(mask_corpo, _kernel((5, 5)), iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi_obj, connectivity=8)
    out = np.zeros_like(roi_obj)
    seed_bool = seed > 0
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 100:
            continue
        comp = labels == label
        if np.any(comp & seed_bool):
            out[comp] = 255

    if np.count_nonzero(out) == 0:
        return mask_corpo
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, _kernel((5, 5)))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, _kernel((3, 3)))
    return out


def _estimar_topo_funcional_vaso(
    mask_corpo: np.ndarray,
    roi_inicio_y: int,
    largura_min_frac: float,
    linhas_consecutivas: int,
) -> Tuple[Optional[int], Optional[int], Dict[str, object]]:
    h, w = mask_corpo.shape
    mask_profile = cv2.morphologyEx(mask_corpo, cv2.MORPH_CLOSE, _kernel((5, 13)))
    bbox = _bbox_from_mask(mask_profile)
    body_w = int(bbox[2]) if bbox[2] > 0 else int(w * 0.2)
    x_ref = int(bbox[0] + body_w / 2) if bbox[2] > 0 else None

    valid = np.zeros(h, dtype=np.uint8)
    larguras = np.zeros(h, dtype=np.int32)
    centros = np.full(h, -1, dtype=np.int32)
    xesq = np.full(h, -1, dtype=np.int32)
    xdir = np.full(h, -1, dtype=np.int32)

    for y in range(max(0, roi_inicio_y), h):
        seg = _segmento_principal_linha(mask_profile, y, x_ref=x_ref)
        if seg is None:
            continue
        x0, x1 = seg
        largura = x1 - x0 + 1
        xc = int((x0 + x1) / 2)
        larguras[y] = int(largura)
        centros[y] = int(xc)
        xesq[y] = int(x0)
        xdir[y] = int(x1)

    nonzero_widths = larguras[larguras > 0]
    if nonzero_widths.size == 0:
        debug = {
            'largura_min_px': 0,
            'largura_target_px': 0,
            'perfil_larguras': larguras,
            'perfil_centros_x': centros,
            'linhas_validas': valid,
            'linhas_consecutivas_validas': 0,
            'x_esquerda_por_linha': xesq,
            'x_direita_por_linha': xdir,
        }
        return None, None, debug

    # Limiares físicos do corpo do vaso (evita confundir com caule fino).
    largura_floor = int(max(28, body_w * 0.12))
    largura_target = int(max(largura_floor + 8, body_w * 0.18))
    valid[larguras >= largura_floor] = 1

    y_top = None
    x_top_center = None
    melhor_valid = 0
    x_tol = max(18, int(w * 0.07))
    y_start = max(0, roi_inicio_y)
    y_end = max(y_start, h - linhas_consecutivas)

    for y in range(y_start, y_end + 1):
        win = slice(y, y + linhas_consecutivas)
        win_valid_idx = np.where(valid[win] > 0)[0]
        if win_valid_idx.size < max(4, linhas_consecutivas - 2):
            continue
        ys_win = y + win_valid_idx
        xs_win = centros[ys_win]
        xs_win = xs_win[xs_win >= 0]
        if xs_win.size == 0:
            continue
        x_med = int(np.median(xs_win))
        if np.max(np.abs(xs_win - x_med)) > x_tol:
            continue

        widths_win = larguras[y : y + linhas_consecutivas]
        if int(np.median(widths_win)) < largura_target:
            continue

        if y_top is None or y < y_top:
            y_top = int(y)
            x_top_center = int(x_med)
            melhor_valid = int(win_valid_idx.size)

    # Refinamento: sobe para o primeiro y ainda "largo de vaso", com centro coerente.
    if y_top is not None and x_top_center is not None:
        x_tol_relax = max(20, int(w * 0.08))
        largura_relax = int(max(largura_floor, 0.75 * largura_target))
        y_from = max(roi_inicio_y, y_top - 36)
        y_to = y_top
        y_cand = y_top
        gaps = 0
        for y in range(y_from, y_to + 1):
            if larguras[y] < largura_relax:
                gaps += 1
                if gaps > 2:
                    break
                continue
            xc = centros[y]
            if xc < 0:
                continue
            if abs(int(xc) - int(x_top_center)) <= x_tol_relax:
                y_cand = y
                break
        y_top = int(y_cand)

    # Cena 3D: o topo funcional deve tangenciar a BORDA SUPERIOR do vaso.
    # Busca uma "frente de borda" acima do corpo, com largura menor que o corpo
    # mas ainda consistente e centrada.
    if y_top is not None and x_top_center is not None:
        largura_rim = int(max(10, 0.28 * largura_target))
        y_from = max(roi_inicio_y, y_top - 90)
        y_to = y_top
        y_rim = y_top
        # procura o primeiro y (mais alto) com persistencia local
        for y in range(y_from, y_to + 1):
            hits = 0
            for yy in range(y, min(y_to + 1, y + 6)):
                if larguras[yy] < largura_rim:
                    continue
                xc = centros[yy]
                if xc < 0:
                    continue
                if abs(int(xc) - int(x_top_center)) <= x_tol_relax:
                    hits += 1
            if hits >= 3:
                y_rim = y
                break
        y_top = int(y_rim)

    debug = {
        'largura_min_px': int(largura_floor),
        'largura_target_px': int(largura_target),
        'perfil_larguras': larguras,
        'perfil_centros_x': centros,
        'linhas_validas': valid,
        'linhas_consecutivas_validas': int(melhor_valid),
        'x_esquerda_por_linha': xesq,
        'x_direita_por_linha': xdir,
    }
    return y_top, x_top_center, debug


def _refinar_topo_borda_traseira(
    mask_objetos: np.ndarray,
    y_topo_frente: int,
    x_centro_ref: int,
    bbox_corpo: Tuple[int, int, int, int],
    persistencia: int = PERSISTENCIA_BORDA_TRASEIRA,
    offset_y: int = OFFSET_Y_BORDA_TRASEIRA,
) -> Tuple[int, Dict[str, int]]:
    h, w = mask_objetos.shape
    _, _, bw, _ = bbox_corpo
    bw = int(max(40, bw))

    # Regiao da boca (parte traseira) costuma ser mais estreita que a parede frontal.
    w_min = int(max(16, 0.10 * bw))
    w_max = int(max(w_min + 8, 0.72 * bw))
    x_tol = int(max(22, 0.10 * bw))

    y_from = max(0, int(y_topo_frente) - 140)
    y_to = max(0, int(y_topo_frente) - 4)
    if y_from > y_to:
        return int(y_topo_frente), {
            "y_topo_frente": int(y_topo_frente),
            "y_topo_borda_traseira": int(y_topo_frente),
            "hits_borda_traseira": 0,
        }

    # Fecha pequenas falhas para deixar a borda superior mais continua.
    mask_ref = cv2.morphologyEx(mask_objetos, cv2.MORPH_CLOSE, _kernel((5, 3)))

    melhor_y = int(y_topo_frente)
    melhor_hits = 0
    janela = max(persistencia + 2, 8)

    for y in range(y_from, y_to + 1):
        hits = 0
        for yy in range(y, min(y_to + 1, y + janela)):
            seg = _segmento_principal_linha(mask_ref, yy, x_ref=int(x_centro_ref))
            if seg is None:
                continue
            x0, x1 = seg
            largura = int(x1 - x0 + 1)
            xc = int((x0 + x1) / 2)
            if largura < w_min or largura > w_max:
                continue
            if abs(xc - int(x_centro_ref)) > x_tol:
                continue
            hits += 1

        if hits >= persistencia:
            melhor_y = int(y)
            melhor_hits = int(hits)
            break
        if hits > melhor_hits:
            melhor_hits = int(hits)

    # Ajuste adaptativo por confianca da evidência de borda traseira:
    # - alta: sobe pouco (encaixe fino)
    # - media: sobe mais (quando a borda aparece mas com baixa persistencia)
    # - baixa: nao sobe; desloca levemente para baixo para evitar falso topo alto.
    limiar_medio = max(2, int(persistencia) - 2)
    modo = 0  # 2=alta, 1=media, 0=baixa
    if melhor_hits >= int(persistencia):
        y_refinado = max(0, int(melhor_y) - int(max(0, offset_y)))
        y_refinado = min(int(y_topo_frente), int(y_refinado))
        modo = 2
    elif melhor_hits >= limiar_medio:
        y_refinado = max(0, int(melhor_y) - int(max(0, OFFSET_Y_MEDIA_CONFIANCA)))
        y_refinado = min(int(y_topo_frente), int(y_refinado))
        modo = 1
    else:
        y_refinado = min(h - 1, int(y_topo_frente) + int(max(0, OFFSET_Y_BAIXA_CONFIANCA)))
        modo = 0

    return int(y_refinado), {
        "y_topo_frente": int(y_topo_frente),
        "y_topo_borda_traseira": int(y_refinado),
        "hits_borda_traseira": int(melhor_hits),
        "modo_borda_traseira": int(modo),
    }


def detectar_tubete_melhorado(
    img_bgr: np.ndarray,
    mask_objetos: np.ndarray,
    roi_inicio_frac: float = ROI_INICIO_TUBETE_FRAC,
    largura_min_frac: float = LARGURA_MIN_TUBETE_FRAC,
    linhas_consecutivas: int = LINHAS_CONSECUTIVAS_TOPO,
    area_min_tubete: int = AREA_MIN_TUBETE,
) -> Dict[str, object]:
    h, w = mask_objetos.shape
    roi_inicio_y = int(max(0, min(h - 1, h * roi_inicio_frac)))

    mask_vaso_candidata = _mascara_candidata_vaso(
        img_bgr=img_bgr,
        mask_objetos=mask_objetos,
        roi_inicio_y=roi_inicio_y,
        area_min_tubete=area_min_tubete,
    )
    mask_corpo_raw, ok_corpo, metodo = _escolher_corpo_principal_vaso(mask_vaso_candidata, roi_inicio_y)
    mask_corpo = _expandir_corpo_com_objetos(mask_corpo_raw, mask_objetos, roi_inicio_y)

    y_topo_func, x_center_func, dbg_perfil = _estimar_topo_funcional_vaso(
        mask_corpo=mask_corpo,
        roi_inicio_y=roi_inicio_y,
        largura_min_frac=largura_min_frac,
        linhas_consecutivas=linhas_consecutivas,
    )

    confianca = bool(ok_corpo and y_topo_func is not None and x_center_func is not None)
    if not confianca:
        metodo = 'fallback_conservador'
        y_topo_func = int(max(roi_inicio_y, int(h * 0.72)))
        x_center_func = int(w // 2)

    bbox_corpo = _bbox_from_mask(mask_corpo)
    dbg_borda = {
        "y_topo_frente": int(y_topo_func),
        "y_topo_borda_traseira": int(y_topo_func),
        "hits_borda_traseira": 0,
    }
    if confianca:
        y_refinado, dbg_borda = _refinar_topo_borda_traseira(
            mask_objetos=mask_objetos,
            y_topo_frente=int(y_topo_func),
            x_centro_ref=int(x_center_func),
            bbox_corpo=bbox_corpo,
        )
        y_topo_func = int(y_refinado)

    mask_tubete = mask_corpo.copy()
    mask_tubete[: int(y_topo_func), :] = 0
    mask_tubete = cv2.morphologyEx(mask_tubete, cv2.MORPH_CLOSE, _kernel((5, 5)))
    mask_tubete = cv2.morphologyEx(mask_tubete, cv2.MORPH_OPEN, _kernel((3, 3)))
    mask_tubete = _limpar_componentes_pequenos(mask_tubete, max(120, area_min_tubete // 8))

    faixa0 = int(y_topo_func)
    faixa1 = min(h - 1, faixa0 + max(3, linhas_consecutivas - 1))
    xs_faixa = []
    for y in range(faixa0, faixa1 + 1):
        seg = _segmento_principal_linha(mask_tubete, y, x_ref=int(x_center_func))
        if seg is None:
            continue
        x0, x1 = seg
        xs_faixa.extend([x0, x1])

    if xs_faixa:
        x_centro_tubete = int(0.5 * (min(xs_faixa) + max(xs_faixa)))
    else:
        x_centro_tubete = int(x_center_func)

    bbox_tubete = _bbox_from_mask(mask_tubete)
    if bbox_tubete[2] == 0 or bbox_tubete[3] == 0:
        y0 = int(max(roi_inicio_y, int(h * 0.75)))
        mask_tubete = np.zeros_like(mask_objetos)
        mask_tubete[y0:, :] = mask_objetos[y0:, :]
        mask_tubete = cv2.morphologyEx(mask_tubete, cv2.MORPH_CLOSE, _kernel((5, 5)))
        mask_tubete = _limpar_componentes_pequenos(mask_tubete, max(120, area_min_tubete // 10))
        bbox_tubete = _bbox_from_mask(mask_tubete)
        y_topo_func = int(bbox_tubete[1] if bbox_tubete[3] > 0 else y0)
        x_centro_tubete = int(bbox_tubete[0] + bbox_tubete[2] / 2) if bbox_tubete[2] > 0 else int(w // 2)
        confianca = False
        metodo = 'fallback_bbox_inferior'

    debug_tubete = {
        'mask_vaso_candidata': mask_vaso_candidata,
        'mask_corpo_vaso_raw': mask_corpo_raw,
        'mask_corpo_vaso': mask_corpo,
        'roi_inicio_y': int(roi_inicio_y),
        'y_topo_tubete': int(y_topo_func),
        'x_centro_tubete': int(x_centro_tubete),
        'largura_topo_detectada_px': int(max(0, bbox_tubete[2])),
        'metodo_tubete': metodo,
        'confianca_tubete': bool(confianca),
        **dbg_perfil,
        **dbg_borda,
    }

    return {
        'mask_tubete': mask_tubete,
        'y_topo_tubete': int(y_topo_func),
        'bbox_tubete': bbox_tubete,
        'x_centro_tubete': int(x_centro_tubete),
        'confianca_tubete': bool(confianca),
        'debug_tubete': debug_tubete,
        'metodo_tubete': metodo,
    }


def detectar_tubete(img_bgr: np.ndarray, mask_objetos: np.ndarray) -> Dict[str, object]:
    return detectar_tubete_melhorado(img_bgr, mask_objetos)


def separar_planta_acima_vaso(
    mask_objetos: np.ndarray,
    mask_tubete: np.ndarray,
    y_topo_tubete: int,
    x_centro_tubete: int,
) -> np.ndarray:
    sem_tubete = cv2.bitwise_and(mask_objetos, cv2.bitwise_not(mask_tubete))

    mask_planta = sem_tubete.copy()
    y_limite_planta = max(0, y_topo_tubete - MARGEM_SEGURA_VASO)
    mask_planta[y_limite_planta:, :] = 0

    # Seleciona componentes primeiro; dilatacao aqui pode colar caule com tampa do vaso.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_planta, connectivity=8)

    h, w = mask_objetos.shape
    ys, xs = np.where(mask_tubete > 0)
    x_centro = int(np.median(xs)) if xs.size else w // 2

    saida = np.zeros_like(mask_objetos)
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < AREA_MIN_OBJ:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])

        y_bottom = y + bh
        perto_base = y_bottom >= (y_topo_tubete - 45)
        cruza_faixa_central = (x <= x_centro <= x + bw) or (abs((x + bw // 2) - x_centro) < int(w * 0.12))

        if perto_base or cruza_faixa_central or area > 5000:
            saida[labels == label] = 255

    saida = cv2.morphologyEx(saida, cv2.MORPH_CLOSE, _kernel((5, 5)))
    # Fronteira rigida: planta nunca pode invadir a regiao do vaso/tampa.
    saida[y_limite_planta:, :] = 0
    saida = _podar_tampa_vaso(saida, y_topo_tubete, x_centro_tubete)
    saida[y_limite_planta:, :] = 0
    return saida


def _segmentos_horizontais(row: np.ndarray):
    xs = np.where(row > 0)[0]
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


def _segmento_contendo_x(row: np.ndarray, x_ref: int):
    for x0, x1 in _segmentos_horizontais(row):
        if x0 <= x_ref <= x1:
            return x0, x1
    return None


def _podar_tampa_vaso(mask: np.ndarray, y_topo_tubete: int, x_centro_tubete: int) -> np.ndarray:
    h, w = mask.shape
    y0 = max(0, y_topo_tubete - BASE_BAND_UP)
    y1 = min(h - 1, y_topo_tubete + BASE_BAND_DOWN)

    # Busca faixa mais estreita que cruza o centro para estimar regiao de caule.
    x_stem = int(x_centro_tubete)
    melhor_w = 10**9
    for y in range(y1, y0 - 1, -1):
        seg = _segmento_contendo_x(mask[y, :], x_stem)
        if seg is None:
            continue
        x0, x1 = seg
        ww = x1 - x0 + 1
        if ww < melhor_w:
            melhor_w = ww
            x_stem = int((x0 + x1) / 2)

    out = mask.copy()
    for y in range(y0, y1 + 1):
        seg = _segmento_contendo_x(out[y, :], x_stem)
        if seg is None:
            continue
        x0, x1 = seg
        ww = x1 - x0 + 1
        # Se a faixa estiver larga demais perto do topo do tubete, poda lateralmente.
        if ww > MAX_LARGURA_BASE_COLETO:
            keep0 = max(0, x_stem - BASE_HALF_WIDTH_KEEP)
            keep1 = min(w - 1, x_stem + BASE_HALF_WIDTH_KEEP)
            out[y, :keep0] = 0
            out[y, keep1 + 1 :] = 0
    return out


def remover_boca_vaso(
    img_bgr: np.ndarray,
    mask_planta_acima: np.ndarray,
    y_topo_tubete: int,
    x_centro_tubete: int,
    mask_tubete: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = mask_planta_acima.shape
    y0 = max(0, int(y_topo_tubete) - BOCA_PIX_ACIMA)
    y1 = min(h - 1, int(y_topo_tubete) + BOCA_PIX_ABAIXO)
    x0 = max(0, int(x_centro_tubete) - BOCA_MEIA_LARGURA)
    x1 = min(w - 1, int(x_centro_tubete) + BOCA_MEIA_LARGURA)

    region = np.zeros_like(mask_planta_acima)
    region[y0 : y1 + 1, x0 : x1 + 1] = 255

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    dark_low_sat = ((s <= HSV_VASO_S_MAX) & (v <= HSV_VASO_V_MAX)).astype(np.uint8) * 255

    prox_tubete = cv2.dilate(mask_tubete, _kernel((9, 9)), iterations=1)

    # Segmentos largos/irregulares na boca (tampa/terra) dentro da regiao central.
    mask_segmento_largo = np.zeros_like(mask_planta_acima)
    for y in range(y0, y1 + 1):
        seg = _segmento_principal_linha(mask_planta_acima, y, x_ref=int(x_centro_tubete))
        if seg is None:
            continue
        sx0, sx1 = seg
        largura = sx1 - sx0 + 1
        if largura >= int(max(MAX_LARGURA_BASE_COLETO + 4, 0.22 * (x1 - x0 + 1))):
            mask_segmento_largo[y, sx0 : sx1 + 1] = 255

    mask_remover = np.zeros_like(mask_planta_acima)
    cond_vaso = cv2.bitwise_and(dark_low_sat, prox_tubete)
    cond_vaso = cv2.bitwise_or(cond_vaso, mask_segmento_largo)
    cond_vaso = cv2.bitwise_and(cond_vaso, region)
    cond_vaso = cv2.bitwise_and(cond_vaso, mask_planta_acima)
    mask_remover[cond_vaso > 0] = 255

    out = mask_planta_acima.copy()
    out[mask_remover > 0] = 0

    overlay = img_bgr.copy()
    overlay_mask = np.zeros_like(img_bgr)
    overlay_mask[:, :] = (0, 0, 255)
    alpha = 0.35
    blend = cv2.addWeighted(overlay, 1.0, overlay_mask, alpha, 0)
    m = mask_remover > 0
    overlay[m] = blend[m]
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 2)
    cv2.putText(
        overlay,
        "boca vaso removida",
        (max(10, x0), max(25, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    return out, mask_remover, overlay


def tem_suporte_vertical(
    mask: np.ndarray,
    x: int,
    y: int,
    altura: int = ALTURA_SUPORTE_VERTICAL,
    meia_largura: int = MEIA_LARGURA_SUPORTE,
    min_pixels: int = MIN_SUPORTE_VERTICAL,
) -> bool:
    h, w = mask.shape
    y0 = max(0, int(y) - int(altura))
    y1 = max(0, int(y) - 1)
    if y1 < y0:
        return False
    x0 = max(0, int(x) - int(meia_largura))
    x1 = min(w - 1, int(x) + int(meia_largura))
    roi = mask[y0 : y1 + 1, x0 : x1 + 1]
    return int(np.count_nonzero(roi)) >= int(min_pixels)


def achar_base_coleto(
    mask_planta_acima: np.ndarray,
    y_topo_tubete: int,
    x_centro_tubete: int,
    mask_tubete: Optional[np.ndarray] = None,
    mask_boca_removida: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    h, w = mask_planta_acima.shape
    # Linha de tangencia horizontal do vaso.
    y_tangencia = int(y_topo_tubete)
    y_limite = max(0, min(h - 1, y_tangencia - MARGEM_SEGURA_VASO))

    # Busca o primeiro ponto de caule acima da tangencia.
    faixa_x = int(max(35, w * 0.10))
    x_min = max(0, int(x_centro_tubete) - faixa_x)
    x_max = min(w - 1, int(x_centro_tubete) + faixa_x)
    y_top_busca = max(0, y_limite - 140)

    for y in range(y_limite, y_top_busca - 1, -1):
        segs = _segmentos_horizontais(mask_planta_acima[y, :])
        if not segs:
            continue

        # Segmento mais plausivel de caule: estreito e perto do centro do vaso.
        melhor = None
        melhor_score = 1e12
        for x0, x1 in segs:
            # restringe ao corredor central do vaso
            if x1 < x_min or x0 > x_max:
                continue
            largura = int(x1 - x0 + 1)
            if largura > MAX_LARGURA_BASE_COLETO:
                continue
            xc = int((x0 + x1) / 2)
            dist_centro = abs(int(xc) - int(x_centro_tubete))
            score = float(dist_centro) + 0.25 * float(largura)
            if score < melhor_score:
                melhor_score = score
                melhor = (xc, largura)

        if melhor is None:
            continue

        x_base, _ = melhor
        if not tem_suporte_vertical(mask_planta_acima, x_base, int(y)):
            continue
        if mask_tubete is not None and 0 <= int(y) < h and 0 <= int(x_base) < w:
            if mask_tubete[int(y), int(x_base)] > 0:
                continue
        if mask_boca_removida is not None and 0 <= int(y) < h and 0 <= int(x_base) < w:
            if mask_boca_removida[int(y), int(x_base)] > 0:
                continue

        return int(x_base), int(y)

    ys_all, xs_all = np.where(mask_planta_acima > 0)
    if ys_all.size == 0:
        return int(x_centro_tubete), int(y_limite)

    # Fallback conservador: ponto mais proximo do centro na linha limite.
    alvo_y = int(y_limite)
    dist2 = (xs_all - x_centro_tubete) ** 2 + (ys_all - alvo_y) ** 2
    idx = int(np.argmin(dist2))
    return int(xs_all[idx]), int(ys_all[idx])


def skeletonize(mask: np.ndarray) -> np.ndarray:
    bin_mask = (mask > 0).astype(np.uint8) * 255

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


def _vizinhos8(y: int, x: int, h: int, w: int):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx


def _ajustar_para_skeleton(skel: np.ndarray, ponto_base: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    x_base, y_base = ponto_base
    ys, xs = np.where(skel > 0)
    if ys.size == 0:
        return None

    dist2 = (xs - x_base) ** 2 + (ys - y_base) ** 2
    idx = int(np.argmin(dist2))
    return int(xs[idx]), int(ys[idx])


def estimar_eixo_caule(mask_planta_acima: np.ndarray, ponto_base: Tuple[int, int]) -> np.ndarray:
    skel = skeletonize(mask_planta_acima)
    h, w = skel.shape

    ponto_inicio = _ajustar_para_skeleton(skel, ponto_base)
    if ponto_inicio is None:
        return np.zeros_like(mask_planta_acima)

    x0, y0 = ponto_inicio
    visitado = np.zeros_like(skel, dtype=bool)
    prev_x = np.full_like(skel, -1, dtype=np.int32)
    prev_y = np.full_like(skel, -1, dtype=np.int32)

    fila = [(y0, x0)]
    visitado[y0, x0] = True

    while fila:
        y, x = fila.pop(0)
        for ny, nx in _vizinhos8(y, x, h, w):
            if skel[ny, nx] == 0 or visitado[ny, nx]:
                continue
            visitado[ny, nx] = True
            prev_x[ny, nx] = x
            prev_y[ny, nx] = y
            fila.append((ny, nx))

    ys, xs = np.where(visitado)
    if ys.size == 0:
        return np.zeros_like(mask_planta_acima)

    idx_topo = int(np.argmin(ys))
    y_topo = int(ys[idx_topo])
    x_topo = int(xs[idx_topo])

    path_mask = np.zeros_like(mask_planta_acima)
    cx, cy = x_topo, y_topo
    path_mask[cy, cx] = 255
    while not (cx == x0 and cy == y0):
        px = prev_x[cy, cx]
        py = prev_y[cy, cx]
        if px < 0 or py < 0:
            break
        cx, cy = int(px), int(py)
        path_mask[cy, cx] = 255

    mask_caule = cv2.dilate(path_mask, _kernel((5, 5)), iterations=1)
    mask_caule = cv2.bitwise_and(mask_caule, mask_planta_acima)
    return mask_caule


def _salvar_debug_masks(debug_dir: Path, masks: Dict[str, np.ndarray]) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    for nome, mask in masks.items():
        cv2.imwrite(str(debug_dir / f"{nome}.png"), mask)


def _salvar_debug_fronteira_vaso(
    debug_dir: Path,
    img_bgr: np.ndarray,
    y_topo_tubete: int,
    y_limite_planta: int,
    y_topo_frente: Optional[int] = None,
) -> None:
    overlay = img_bgr.copy()
    h, w = overlay.shape[:2]
    y_top = int(max(0, min(h - 1, y_topo_tubete)))
    y_lim = int(max(0, min(h - 1, y_limite_planta)))
    if y_topo_frente is not None:
        y_front = int(max(0, min(h - 1, int(y_topo_frente))))
        cv2.line(overlay, (0, y_front), (w - 1, y_front), (255, 0, 255), 1)
        cv2.putText(
            overlay,
            f"y_topo_frente={y_front}",
            (20, max(24, y_front - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 255),
            2,
        )
    cv2.line(overlay, (0, y_top), (w - 1, y_top), (0, 0, 255), 2)
    cv2.line(overlay, (0, y_lim), (w - 1, y_lim), (0, 255, 255), 2)
    cv2.putText(
        overlay,
        f"y_topo_tubete={y_top}",
        (20, max(30, y_top - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        overlay,
        f"y_limite_planta={y_lim}",
        (20, min(h - 20, y_lim + 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.imwrite(str(debug_dir / "overlay_fronteira_vaso.png"), overlay)


def processar_tratamentos_iniciais(path: str, debug_dir: Optional[str] = None) -> Dict[str, object]:
    img_bgr = ler_imagem(path)
    mask_fundo = segmentar_fundo_azul(img_bgr)
    mask_objetos = obter_mask_objetos(img_bgr)

    info_tubete = detectar_tubete(img_bgr, mask_objetos)
    mask_tubete = info_tubete["mask_tubete"]
    y_topo_tubete = int(info_tubete["y_topo_tubete"])
    bbox_tubete = info_tubete["bbox_tubete"]
    x_centro_tubete = int(info_tubete["x_centro_tubete"])
    confianca_tubete = bool(info_tubete.get("confianca_tubete", False))
    metodo_tubete = str(info_tubete.get("metodo_tubete", "desconhecido"))
    debug_tubete = info_tubete.get("debug_tubete", {})

    mask_planta_acima = separar_planta_acima_vaso(
        mask_objetos, mask_tubete, y_topo_tubete, x_centro_tubete
    )
    mask_planta_antes_remover_boca = mask_planta_acima.copy()

    mask_planta_acima, mask_boca_removida, overlay_boca_removida = remover_boca_vaso(
        img_bgr=img_bgr,
        mask_planta_acima=mask_planta_acima,
        y_topo_tubete=y_topo_tubete,
        x_centro_tubete=x_centro_tubete,
        mask_tubete=mask_tubete,
    )

    y_limite_planta = max(0, y_topo_tubete - MARGEM_SEGURA_VASO)
    mask_planta_acima[y_limite_planta:, :] = 0
    ponto_base = achar_base_coleto(
        mask_planta_acima,
        y_topo_tubete,
        x_centro_tubete,
        mask_tubete=mask_tubete,
        mask_boca_removida=mask_boca_removida,
    )

    skeleton = skeletonize(mask_planta_acima)
    skeleton[y_limite_planta:, :] = 0
    mask_caule = estimar_eixo_caule(mask_planta_acima, ponto_base)
    mask_caule[y_limite_planta:, :] = 0

    if debug_dir is not None:
        debug_masks = {
            "mask_fundo": mask_fundo,
            "mask_objetos": mask_objetos,
            "mask_tubete": mask_tubete,
            "mask_planta_acima": mask_planta_acima,
            "mask_planta_antes_remover_boca_vaso": mask_planta_antes_remover_boca,
            "mask_planta_sem_boca_vaso": mask_planta_acima,
            "skeleton": skeleton,
            "mask_caule": mask_caule,
        }
        if isinstance(debug_tubete, dict):
            if "mask_vaso_candidata" in debug_tubete:
                debug_masks["mask_vaso_candidata"] = debug_tubete["mask_vaso_candidata"]
            if "mask_corpo_vaso_raw" in debug_tubete:
                debug_masks["mask_corpo_vaso_raw"] = debug_tubete["mask_corpo_vaso_raw"]
            if "mask_corpo_vaso" in debug_tubete:
                debug_masks["mask_corpo_vaso"] = debug_tubete["mask_corpo_vaso"]
        debug_masks["mask_planta_acima_sem_vaso"] = mask_planta_acima
        _salvar_debug_masks(
            Path(debug_dir),
            debug_masks,
        )
        cv2.imwrite(str(Path(debug_dir) / "overlay_boca_vaso_removida.png"), overlay_boca_removida)
        _salvar_debug_fronteira_vaso(
            Path(debug_dir),
            img_bgr=img_bgr,
            y_topo_tubete=y_topo_tubete,
            y_limite_planta=y_limite_planta,
            y_topo_frente=int(debug_tubete["y_topo_frente"]) if isinstance(debug_tubete, dict) and "y_topo_frente" in debug_tubete else None,
        )
        overlay_base = img_bgr.copy()
        cv2.circle(overlay_base, (int(ponto_base[0]), int(ponto_base[1])), 6, (0, 255, 255), -1)
        cv2.putText(
            overlay_base,
            f"base_coleto=({int(ponto_base[0])},{int(ponto_base[1])})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.imwrite(str(Path(debug_dir) / "overlay_ponto_base_coleto.png"), overlay_base)

    return {
        "img_bgr": img_bgr,
        "mask_fundo": mask_fundo,
        "mask_objetos": mask_objetos,
        "mask_tubete": mask_tubete,
        "y_topo_tubete": y_topo_tubete,
        "bbox_tubete": bbox_tubete,
        "x_centro_tubete": x_centro_tubete,
        "y_limite_planta": int(y_limite_planta),
        "confianca_tubete": confianca_tubete,
        "metodo_tubete": metodo_tubete,
        "debug_tubete": debug_tubete,
        "mask_boca_removida": mask_boca_removida,
        "ponto_base": ponto_base,
        "mask_planta_acima": mask_planta_acima,
        "skeleton": skeleton,
        "mask_caule": mask_caule,
    }
