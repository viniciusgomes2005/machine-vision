from pathlib import Path
import os
import sys

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path(__file__).parent
PIN_PATH = BASE / "Dataset_Projeto1" / "_Pinheiro_Escolhidos1" / "Pinheiro1.jpg"
EUC_PATH = BASE / "Dataset_Projeto1" / "_Eucalipto_Escolhidos1" / "Eucalipto1.jpg"
OUT_DIR = BASE / "_dev_debug"
OUT_DIR.mkdir(exist_ok=True)


def ler_imagem_rgb(caminho):
    bgr = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(caminho)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def maior_componente(mask):
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx


def filtra_area_min(mask, area_min):
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    for idx in range(1, n):
        if stats[idx, cv2.CC_STAT_AREA] >= area_min:
            out |= labels == idx
    return out


def componentes_ligadas_a_semente(mask, ponto_base, altura_faixa, meia_largura, area_min, y_extra):
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    x_base, y_base = ponto_base
    h, w = mask.shape
    y0 = max(0, y_base - altura_faixa)
    y1 = min(h, y_base + y_extra)
    x0 = max(0, x_base - meia_largura)
    x1 = min(w, x_base + meia_largura)
    semente = np.zeros_like(mask, dtype=bool)
    semente[y0:y1, x0:x1] = True
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    ids_validos = np.unique(labels[mask & semente])
    for idx in ids_validos:
        if idx == 0:
            continue
        if stats[idx, cv2.CC_STAT_AREA] >= area_min:
            out |= labels == idx
    return out


def contornos_preenchidos(mask, area_min):
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    contornos, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, dtype=np.uint8)
    for contorno in contornos:
        if cv2.contourArea(contorno) >= area_min:
            cv2.drawContours(out, [contorno], -1, 255, -1)
    return out > 0


def overlay_mask(img_rgb, mask, color, alpha=0.88):
    out = img_rgb.copy()
    if np.any(mask):
        color_arr = np.array(color, dtype=np.uint8)
        out[mask] = np.clip((1 - alpha) * out[mask] + alpha * color_arr, 0, 255).astype(np.uint8)
    return out


def segmenta_base(img_rgb):
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    mask_fundo = (H >= 90) & (H <= 135) & (S >= 60) & (V >= 50) & (B.astype(int) > R.astype(int) + 8)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mask_preto = (V < 80) & (~mask_fundo)
    mask_preto = cv2.morphologyEx(mask_preto.astype(np.uint8) * 255, cv2.MORPH_OPEN, k3) > 0
    mask_preto[: int(0.40 * h)] = False
    mask_tubete = maior_componente(mask_preto)

    mask_branco = (V >= 170) & (S < 60) & (~mask_fundo)
    mask_branco = cv2.morphologyEx(mask_branco.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    mask_branco[: int(0.45 * h)] = False
    mask_cilindro = maior_componente(mask_branco)

    if np.any(mask_tubete):
        ys_t, xs_t = np.where(mask_tubete)
        y_topo_tubete = int(ys_t.min())
        x_centro_tubete = int(np.median(xs_t))
    else:
        y_topo_tubete = int(0.62 * h)
        x_centro_tubete = w // 2

    cor_planta = (
        ((H >= 20) & (H <= 95))
        | ((H <= 15) & (S >= 90))
        | ((G.astype(int) - B.astype(int) > 5) & (G > 60))
    )
    mask_corpo = cor_planta & (~mask_fundo) & (~mask_tubete) & (~mask_cilindro)
    mask_corpo[y_topo_tubete + 10 :, :] = False
    mask_corpo = cv2.morphologyEx(mask_corpo.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    mask_corpo = cv2.morphologyEx(mask_corpo.astype(np.uint8) * 255, cv2.MORPH_OPEN, k3) > 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_corpo.astype(np.uint8), connectivity=8)
    mask_final = np.zeros_like(mask_corpo)
    melhor_idx, melhor_score = -1, -1.0
    for idx in range(1, n):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < 200:
            continue
        y0 = stats[idx, cv2.CC_STAT_TOP]
        hh = stats[idx, cv2.CC_STAT_HEIGHT]
        dist_y = max(0, y_topo_tubete - (y0 + hh))
        score = area * (1.0 / (1.0 + dist_y / 50.0))
        if score > melhor_score:
            melhor_score = score
            melhor_idx = idx
    if melhor_idx >= 0:
        principal = labels == melhor_idx
        dil = cv2.dilate(principal.astype(np.uint8), np.ones((25, 25), np.uint8), iterations=5) > 0
        for idx in range(1, n):
            if stats[idx, cv2.CC_STAT_AREA] < 80:
                continue
            comp = labels == idx
            if np.any(comp & dil):
                mask_final |= comp

    ys, xs = np.where(mask_final)
    y_base = int(ys.max())
    x_base = int(np.median(xs[ys >= max(ys.min(), y_base - 25)]))

    return {
        "hsv": hsv,
        "mask_fundo": mask_fundo,
        "mask_tubete": mask_tubete,
        "mask_cilindro": mask_cilindro,
        "mask_corpo": mask_final,
        "y_topo_tubete": y_topo_tubete,
        "x_centro_tubete": x_centro_tubete,
        "ponto_base": (x_base, y_base),
    }


def mascara_fina(img_rgb, seg, params):
    hsv = seg["hsv"]
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    mask = (
        (H >= params["fine_h_min"])
        & (H <= params["fine_h_max"])
        & (S >= params["fine_s_min"])
        & (V >= params["fine_v_min"])
        & ((G.astype(int) - B.astype(int)) >= params["fine_gb_min"])
        & ((G.astype(int) - R.astype(int)) >= params["fine_gr_min"])
    )
    mask &= (~seg["mask_fundo"]) & (~seg["mask_tubete"]) & (~seg["mask_cilindro"])
    mask[seg["y_topo_tubete"] + params["below_tube_cut"] :, :] = False

    if params["fine_close_k"] != (1, 1):
        mask = cv2.morphologyEx(
            mask.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, params["fine_close_k"]),
        ) > 0
    if params["fine_dilate_k"] != (1, 1):
        mask = cv2.dilate(
            mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, params["fine_dilate_k"]),
            iterations=1,
        ) > 0
    return filtra_area_min(mask, params["fine_area_min"])


def mascara_caule_pinheiro(mask_corpo, ponto_base, params):
    vazio = np.zeros_like(mask_corpo, dtype=bool)
    if not np.any(mask_corpo):
        return vazio, vazio
    vertical = cv2.morphologyEx(
        mask_corpo.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, params["stem_open_k"]),
    ) > 0
    vertical = cv2.morphologyEx(
        vertical.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, params["stem_close_k"]),
    ) > 0
    stem = componentes_ligadas_a_semente(
        vertical,
        ponto_base,
        altura_faixa=params["stem_height"],
        meia_largura=params["stem_x_half"],
        area_min=params["stem_area_min"],
        y_extra=params["stem_y_extra"],
    )
    faixa = np.zeros_like(mask_corpo, dtype=bool)
    x_base, y_base = ponto_base
    h, w = mask_corpo.shape
    y0 = max(0, y_base - params["stem_height"])
    y1 = min(h, y_base + params["stem_y_extra"])
    x0 = max(0, x_base - params["basic_half"])
    x1 = min(w, x_base + params["basic_half"])
    faixa[y0:y1, x0:x1] = mask_corpo[y0:y1, x0:x1]
    stem |= faixa
    stem = cv2.dilate(
        stem.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, params["stem_dilate_k"]),
        iterations=1,
    ) > 0
    if "stem_top_ignore_rel" in params and np.any(mask_corpo):
        ys, _ = np.where(mask_corpo)
        y_top_real = int(ys.min())
        y_base_real = int(ys.max())
        altura = max(1, y_base_real - y_top_real)
        y_cut = int(y_top_real + params["stem_top_ignore_rel"] * altura)
        stem[:y_cut, :] = False
    return stem & mask_corpo, vertical


def mascara_caule_eucalipto(mask_corpo, ponto_base, params):
    vazio = np.zeros_like(mask_corpo, dtype=bool)
    if not np.any(mask_corpo):
        return vazio
    dist = cv2.distanceTransform(mask_corpo.astype(np.uint8), cv2.DIST_L2, 3)
    estreito = (dist <= params["stem_dist_max"]) & mask_corpo
    x_base, y_base = ponto_base
    h, w = mask_corpo.shape
    y0 = max(0, y_base - params["stem_height"])
    y1 = min(h, y_base + params["stem_y_extra"])
    x0 = max(0, x_base - params["stem_x_half"])
    x1 = min(w, x_base + params["stem_x_half"])
    semente = np.zeros_like(mask_corpo, dtype=bool)
    semente[y0:y1, x0:x1] = True
    n, labels, stats, _ = cv2.connectedComponentsWithStats(estreito.astype(np.uint8), connectivity=8)
    stem = np.zeros_like(mask_corpo, dtype=bool)
    for idx in np.unique(labels[estreito & semente]):
        if idx == 0:
            continue
        if stats[idx, cv2.CC_STAT_AREA] >= params["stem_area_min"]:
            stem |= labels == idx
    stem = cv2.dilate(
        stem.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, params["stem_dilate_k"]),
        iterations=1,
    ) > 0
    return stem & mask_corpo


def recorta_vaso(mask, seg, params):
    h, w = mask.shape
    y0 = max(0, seg["y_topo_tubete"] - params["vase_y_above"])
    y1 = min(h, seg["y_topo_tubete"] + params["vase_y_below"])
    x0 = max(0, seg["x_centro_tubete"] - params["vase_x_half"])
    x1 = min(w, seg["x_centro_tubete"] + params["vase_x_half"])
    out = mask.copy()
    rec = np.zeros_like(mask, dtype=bool)
    rec[y0:y1, x0:x1] = True
    out[rec] = False
    return out, rec


def distancia_ao_stem(stem_mask):
    """Distancia euclidiana, em pixels, ate o caule."""
    if not np.any(stem_mask):
        return np.full(stem_mask.shape, 9999.0, dtype=np.float32)
    inv = (~stem_mask).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3)


def eixo_x_por_linha(mask_ref, fallback_x):
    h, _ = mask_ref.shape
    eixo = np.full(h, float(fallback_x), dtype=np.float32)
    known = np.zeros(h, dtype=bool)
    for y in range(h):
        xs = np.where(mask_ref[y])[0]
        if len(xs):
            eixo[y] = float(np.median(xs))
            known[y] = True
    if not np.any(known):
        return eixo
    last = None
    for y in range(h):
        if known[y]:
            last = eixo[y]
        elif last is not None:
            eixo[y] = last
    last = None
    for y in range(h - 1, -1, -1):
        if known[y]:
            last = eixo[y]
        elif last is not None:
            eixo[y] = last
    return np.convolve(eixo, np.ones(21, dtype=np.float32) / 21.0, mode="same")


def mascara_lateral_por_linha(mask, eixo_x, min_dx):
    h, w = mask.shape
    xx = np.arange(w, dtype=np.float32)[None, :]
    lateral = np.abs(xx - eixo_x[:, None]) >= float(min_dx)
    return mask & lateral


def componentes_que_tocam_seed(mask, seed, area_min=1):
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    for idx in np.unique(labels[mask & seed]):
        if idx == 0:
            continue
        if stats[idx, cv2.CC_STAT_AREA] >= area_min:
            out |= labels == idx
    return out


def promove_topo_pinheiro(mask_candidata, seg, params):
    """No topo da muda, o eixo central vira brotação/folha, não caule."""
    if not np.any(mask_candidata):
        return mask_candidata, np.zeros_like(mask_candidata, dtype=bool)
    h, w = mask_candidata.shape
    x_centro = seg["x_centro_tubete"]
    ys, _ = np.where(seg["mask_corpo"])
    y_top_real = int(ys.min()) if len(ys) else 0
    y_base = seg["ponto_base"][1]
    altura = max(1, y_base - y_top_real)
    if "top_leaf_rel" in params:
        y_top = min(h, int(y_top_real + params["top_leaf_rel"] * altura))
    else:
        y_top = max(0, y_base - params["top_leaf_height"])
    x0 = max(0, x_centro - params["top_leaf_half"])
    x1 = min(w, x_centro + params["top_leaf_half"])
    regiao = np.zeros_like(mask_candidata, dtype=bool)
    regiao[y_top_real:y_top, x0:x1] = True
    return mask_candidata | (seg["mask_corpo"] & regiao), regiao


def folhas_laterais_eucalipto(seg, stem, params):
    """Promove áreas laterais do corpo, longe do eixo do caule."""
    mask = seg["mask_corpo"]
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    ys, xs = np.where(stem if np.any(stem) else mask)
    if len(xs) == 0:
        x_ref = seg["x_centro_tubete"]
    else:
        x_ref = int(np.median(xs))
    xx = np.arange(mask.shape[1])[None, :]
    lateral = np.abs(xx - x_ref) >= params["lateral_min_dx"]
    lateral = np.repeat(lateral, mask.shape[0], axis=0)
    topo_livre = np.zeros_like(mask, dtype=bool)
    topo_livre[: seg["y_topo_tubete"] + params["below_tube_cut"], :] = True
    return mask & lateral & topo_livre


def faixa_y(seg, params, shape):
    """Faixas em y para controlar base/topo da muda."""
    h, _ = shape
    ys, _ = np.where(seg["mask_corpo"])
    y_top_real = int(ys.min()) if len(ys) else 0
    top_leaf_height = params.get("top_leaf_height", params.get("stem_height", 80))
    y_top_stem = min(h, y_top_real + top_leaf_height)
    vase_y_below = params.get("vase_y_below", params.get("below_tube_cut", 8))
    y_vase_cut = min(h, seg["y_topo_tubete"] + vase_y_below)
    return y_top_stem, y_vase_cut


def resolve_pinheiro(img_rgb, seg, params):
    corpo = seg["mask_corpo"]
    fino = mascara_fina(img_rgb, seg, params)
    stem, vertical = mascara_caule_pinheiro(corpo, seg["ponto_base"], params)
    dist_stem = distancia_ao_stem(stem)
    union_original = corpo | fino

    y_top_stem, y_vase_cut = faixa_y(seg, params, corpo.shape)
    yy = np.arange(corpo.shape[0])[:, None]
    zona_baixa = yy >= y_vase_cut
    zona_topo = yy <= y_top_stem
    zona_media = (~zona_baixa) & (~zona_topo)

    folhas_cruas = np.zeros_like(corpo, dtype=bool)
    folhas_cruas |= union_original & zona_topo
    folhas_cruas |= fino & zona_media & (dist_stem >= params["mid_min_dist"])
    folhas_cruas |= corpo & zona_media & (dist_stem >= params["mid_body_min_dist"]) & (~vertical)
    folhas_cruas |= fino & zona_baixa & (dist_stem >= params["low_min_dist"])
    folhas_cruas, regiao_topo = promove_topo_pinheiro(folhas_cruas, seg, params)

    folhas_suaves = cv2.morphologyEx(
        folhas_cruas.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, params["leaf_close_k"]),
    ) > 0
    if params["leaf_dilate_k"] != (1, 1):
        folhas_suaves = cv2.dilate(
            folhas_suaves.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, params["leaf_dilate_k"]),
            iterations=1,
        ) > 0
    folhas_suaves = filtra_area_min(folhas_suaves, params["leaf_area_pre"])

    # Metrica final: volta para os pixels da mascara original, sem engordar o resultado.
    # No meio e embaixo, o corpo so entra se estiver suficientemente longe do caule.
    final_candidata = np.zeros_like(corpo, dtype=bool)
    final_candidata |= union_original & zona_topo
    final_candidata |= fino & zona_media & (dist_stem >= params["final_mid_min_dist"])
    final_candidata |= fino & zona_baixa & (dist_stem >= params["final_low_min_dist"])

    folhas = final_candidata & folhas_suaves
    folhas |= regiao_topo & union_original
    folhas[y_vase_cut:, :] = False
    folhas, recorte = recorta_vaso(folhas, seg, params)
    folhas = filtra_area_min(folhas, params["leaf_area_post"])
    return {
        "mask_corpo": corpo,
        "mask_fino": fino,
        "mask_caule": stem,
        "mask_vertical": vertical,
        "folhas_cruas": folhas_suaves,
        "mask_folhas": folhas,
        "recorte_vaso": recorte,
        "area": int(np.count_nonzero(folhas)),
    }


def resolve_eucalipto(img_rgb, seg, params):
    corpo = seg["mask_corpo"]
    fino = mascara_fina(img_rgb, seg, params)
    stem = mascara_caule_eucalipto(corpo, seg["ponto_base"], params)
    dist_stem = distancia_ao_stem(stem)
    lateral = folhas_laterais_eucalipto(seg, stem, params)
    union = corpo | fino | lateral
    yy = np.arange(corpo.shape[0])[:, None]
    y_top_stem, _ = faixa_y(seg, params, corpo.shape)
    zona_topo = yy <= y_top_stem
    eixo_x = eixo_x_por_linha(stem if np.any(stem) else corpo, seg["x_centro_tubete"])
    strategy = params.get("strategy", "baseline")

    if strategy == "baseline":
        folhas_cruas = np.zeros_like(corpo, dtype=bool)
        folhas_cruas |= union & zona_topo
        folhas_cruas |= fino & (dist_stem >= params["fine_min_dist"])
        folhas_cruas |= lateral & (dist_stem >= params["lateral_min_dist"])
        folhas_cruas |= corpo & (dist_stem >= params["body_min_dist"])
    elif strategy == "line_lateral":
        folhas_cruas = mascara_lateral_por_linha(union, eixo_x, params["line_min_dx"])
        folhas_cruas |= union & zona_topo
        folhas_cruas |= fino & (dist_stem >= params["fine_min_dist"])
    elif strategy == "side_components":
        seed = mascara_lateral_por_linha(fino | lateral, eixo_x, params["line_min_dx"])
        candidatos = union & (dist_stem >= params["comp_min_dist"])
        folhas_cruas = componentes_que_tocam_seed(candidatos, seed, area_min=params["component_seed_area"])
        folhas_cruas |= union & zona_topo
    elif strategy == "row_side_recover":
        seed = mascara_lateral_por_linha(union, eixo_x, params["line_min_dx"])
        folhas_cruas = componentes_que_tocam_seed(union, seed, area_min=params["component_seed_area"])
        folhas_cruas &= (dist_stem >= params["recover_min_dist"]) | zona_topo
        folhas_cruas |= union & zona_topo
    elif strategy == "hybrid_recover":
        seed = mascara_lateral_por_linha(fino | lateral, eixo_x, params["line_min_dx"])
        comp = componentes_que_tocam_seed(union, seed, area_min=params["component_seed_area"])
        folhas_cruas = np.zeros_like(corpo, dtype=bool)
        folhas_cruas |= comp & (dist_stem >= params["recover_min_dist"])
        folhas_cruas |= fino & (dist_stem >= params["fine_min_dist"])
        folhas_cruas |= union & zona_topo
    else:
        raise ValueError(strategy)

    folhas_suaves = cv2.morphologyEx(
        folhas_cruas.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, params["leaf_close_k"]),
    ) > 0
    if params["leaf_dilate_k"] != (1, 1):
        folhas_suaves = cv2.dilate(
            folhas_suaves.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, params["leaf_dilate_k"]),
            iterations=1,
        ) > 0
    folhas_suaves = filtra_area_min(folhas_suaves, params["leaf_area_pre"])

    exclusao = cv2.dilate(
        stem.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, params["stem_exclude_k"]),
        iterations=1,
    ) > 0

    folhas = union & folhas_suaves & (~exclusao)
    folhas = filtra_area_min(folhas, params["leaf_area_post"])
    return {
        "mask_corpo": corpo,
        "mask_fino": fino,
        "mask_caule": stem,
        "mask_vertical": stem.copy(),
        "folhas_cruas": folhas_suaves,
        "mask_folhas": folhas,
        "recorte_vaso": np.zeros_like(corpo, dtype=bool),
        "area": int(np.count_nonzero(folhas)),
    }


def variantes_pinheiro():
    base = {
        "fine_h_min": 18,
        "fine_h_max": 105,
        "fine_s_min": 10,
        "fine_v_min": 18,
        "fine_gb_min": 0,
        "fine_gr_min": -8,
        "fine_close_k": (1, 1),
        "fine_dilate_k": (3, 3),
        "fine_area_min": 1,
        "below_tube_cut": 8,
        "stem_open_k": (3, 71),
        "stem_close_k": (5, 21),
        "stem_dilate_k": (11, 21),
        "stem_exclude_k": (9, 25),
        "stem_height": 280,
        "stem_x_half": 55,
        "stem_area_min": 10,
        "stem_y_extra": 16,
        "basic_half": 16,
        "leaf_close_k": (3, 3),
        "leaf_dilate_k": (1, 1),
        "leaf_area_pre": 1,
        "leaf_area_post": 1,
        "vase_y_above": 8,
        "vase_y_below": 46,
        "vase_x_half": 60,
        "top_leaf_height": 85,
        "top_leaf_rel": 0.38,
        "top_leaf_half": 18,
        "mid_min_dist": 6.5,
        "mid_body_min_dist": 10.5,
        "low_min_dist": 14.0,
        "final_mid_min_dist": 16.0,
        "final_low_min_dist": 20.0,
        "stem_top_ignore_rel": 0.42,
    }
    return [
        {
            "name": "P1 topo rel base",
            **base,
            "fine_dilate_k": (5, 5),
            "leaf_dilate_k": (1, 1),
            "stem_exclude_k": (11, 29),
        },
        {
            "name": "P2 topo 3x",
            **base,
            "fine_dilate_k": (5, 5),
            "leaf_dilate_k": (1, 1),
            "stem_exclude_k": (11, 29),
            "top_leaf_half": 28,
            "top_leaf_rel": 0.52,
            "mid_min_dist": 6.0,
            "final_mid_min_dist": 15.0,
            "final_low_min_dist": 19.0,
            "stem_top_ignore_rel": 0.52,
        },
        {
            "name": "P3 topo 4x cantos",
            **base,
            "fine_h_max": 112,
            "fine_s_min": 7,
            "fine_v_min": 14,
            "fine_dilate_k": (5, 5),
            "leaf_dilate_k": (1, 1),
            "stem_exclude_k": (11, 29),
            "top_leaf_half": 24,
            "top_leaf_rel": 0.60,
            "mid_min_dist": 5.5,
            "mid_body_min_dist": 9.5,
            "final_mid_min_dist": 14.0,
            "final_low_min_dist": 18.0,
            "stem_top_ignore_rel": 0.60,
        },
        {
            "name": "P4 topo alto centro duro",
            **base,
            "fine_gb_min": 4,
            "fine_gr_min": 0,
            "fine_dilate_k": (5, 5),
            "leaf_dilate_k": (1, 1),
            "stem_dilate_k": (13, 25),
            "stem_exclude_k": (13, 31),
            "basic_half": 18,
            "top_leaf_half": 22,
            "top_leaf_rel": 0.56,
            "mid_min_dist": 7.0,
            "mid_body_min_dist": 12.0,
            "low_min_dist": 16.0,
            "final_mid_min_dist": 18.0,
            "final_low_min_dist": 22.0,
            "stem_top_ignore_rel": 0.56,
        },
        {
            "name": "P5 topo max vaso forte",
            **base,
            "fine_dilate_k": (5, 5),
            "leaf_dilate_k": (1, 1),
            "stem_exclude_k": (11, 29),
            "vase_y_below": 54,
            "vase_x_half": 80,
            "top_leaf_half": 20,
            "top_leaf_rel": 0.65,
            "low_min_dist": 18.0,
            "final_mid_min_dist": 17.0,
            "final_low_min_dist": 24.0,
            "stem_top_ignore_rel": 0.65,
        },
    ]


def variantes_eucalipto():
    base = {
        "fine_h_min": 18,
        "fine_h_max": 105,
        "fine_s_min": 12,
        "fine_v_min": 22,
        "fine_gb_min": 0,
        "fine_gr_min": -8,
        "fine_close_k": (1, 1),
        "fine_dilate_k": (3, 3),
        "fine_area_min": 1,
        "below_tube_cut": 8,
        "stem_dist_max": 1.2,
        "stem_height": 110,
        "stem_x_half": 90,
        "stem_y_extra": 20,
        "stem_area_min": 80,
        "stem_dilate_k": (5, 7),
        "stem_exclude_k": (5, 3),
        "leaf_close_k": (5, 5),
        "leaf_dilate_k": (1, 1),
        "leaf_area_pre": 40,
        "leaf_area_post": 20,
        "lateral_min_dx": 20,
        "fine_min_dist": 3.0,
        "lateral_min_dist": 6.5,
        "body_min_dist": 9.0,
        "line_min_dx": 16,
        "comp_min_dist": 4.0,
        "recover_min_dist": 3.0,
        "component_seed_area": 12,
    }
    return [
        {
            "name": "E1 baseline",
            **base,
            "strategy": "baseline",
            "fine_dilate_k": (5, 5),
            "leaf_dilate_k": (5, 5),
            "stem_exclude_k": (3, 3),
            "fine_gb_min": 4,
            "fine_gr_min": 0,
            "leaf_area_post": 30,
        },
        {
            "name": "E2 linha lateral",
            **base,
            "strategy": "line_lateral",
            "fine_dilate_k": (5, 5),
            "leaf_dilate_k": (3, 3),
            "stem_exclude_k": (3, 3),
            "fine_gb_min": 2,
            "fine_gr_min": -2,
            "leaf_area_post": 25,
            "lateral_min_dx": 16,
            "fine_min_dist": 2.0,
            "line_min_dx": 12,
        },
        {
            "name": "E3 comp laterais",
            **base,
            "strategy": "side_components",
            "fine_dilate_k": (5, 5),
            "leaf_dilate_k": (1, 1),
            "stem_exclude_k": (3, 3),
            "fine_gb_min": 4,
            "fine_gr_min": 0,
            "leaf_area_post": 28,
            "comp_min_dist": 3.0,
            "line_min_dx": 12,
            "component_seed_area": 8,
        },
        {
            "name": "E4 reanexa folha",
            **base,
            "strategy": "row_side_recover",
            "fine_dilate_k": (5, 5),
            "leaf_dilate_k": (3, 3),
            "stem_exclude_k": (3, 3),
            "fine_gb_min": 4,
            "fine_gr_min": 0,
            "leaf_area_post": 30,
            "line_min_dx": 10,
            "recover_min_dist": 2.0,
            "component_seed_area": 8,
        },
        {
            "name": "E5 hibrido lateral",
            **base,
            "strategy": "hybrid_recover",
            "fine_dilate_k": (5, 5),
            "leaf_dilate_k": (3, 3),
            "stem_exclude_k": (3, 3),
            "fine_gb_min": 3,
            "fine_gr_min": -1,
            "leaf_area_post": 27,
            "lateral_min_dx": 16,
            "fine_min_dist": 2.5,
            "line_min_dx": 12,
            "recover_min_dist": 2.0,
            "component_seed_area": 8,
        },
    ]


def salva_galeria(img_rgb, resultados, variantes, titulo, out_name):
    fig, axs = plt.subplots(len(variantes), 5, figsize=(22, 4.1 * len(variantes)))
    if len(variantes) == 1:
        axs = np.array([axs])
    for i, (res, params) in enumerate(zip(resultados, variantes)):
        v0 = overlay_mask(img_rgb, res["mask_corpo"], (255, 230, 0), alpha=0.65)
        v1 = overlay_mask(img_rgb, res["mask_fino"], (0, 255, 255), alpha=0.8)
        v2 = overlay_mask(img_rgb, res["mask_caule"], (255, 160, 0), alpha=0.85)
        v2 = overlay_mask(v2, res["recorte_vaso"], (255, 255, 255), alpha=0.85)
        v3 = overlay_mask(img_rgb, res["folhas_cruas"], (255, 120, 120), alpha=0.85)
        v4 = overlay_mask(img_rgb, res["mask_folhas"], (255, 0, 0), alpha=0.92)

        axs[i, 0].imshow(v0)
        axs[i, 0].set_title(f"{params['name']}\n1) corpo")
        axs[i, 1].imshow(v1)
        axs[i, 1].set_title("2) detalhe fino")
        axs[i, 2].imshow(v2)
        axs[i, 2].set_title("3) caule/recorte")
        axs[i, 3].imshow(v3)
        axs[i, 3].set_title("4) folhas cruas")
        axs[i, 4].imshow(v4)
        axs[i, 4].set_title(f"5) final\narea={res['area']} px")
        for j in range(5):
            axs[i, j].axis("off")
    fig.suptitle(titulo, fontsize=18)
    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def abre_no_pc(caminho):
    if os.environ.get("NO_OPEN", "") == "1":
        return
    if sys.platform.startswith("win"):
        os.startfile(str(caminho))


def main():
    img_pin = ler_imagem_rgb(PIN_PATH)
    seg_pin = segmenta_base(img_pin)
    vars_pin = variantes_pinheiro()
    res_pin = []
    for params in vars_pin:
        res = resolve_pinheiro(img_pin, seg_pin, params)
        res_pin.append(res)
        print(f"{params['name']} | area={res['area']}")
    out_pin = salva_galeria(img_pin, res_pin, vars_pin, "Exploracao 2.3 - Pinheiro1 (nova estrategia)", "exploracao_pinheiro_23_nova_estrategia.png")
    print(out_pin)

    img_euc = ler_imagem_rgb(EUC_PATH)
    seg_euc = segmenta_base(img_euc)
    vars_euc = variantes_eucalipto()
    res_euc = []
    for params in vars_euc:
        res = resolve_eucalipto(img_euc, seg_euc, params)
        res_euc.append(res)
        print(f"{params['name']} | area={res['area']}")
    out_euc = salva_galeria(img_euc, res_euc, vars_euc, "Exploracao 2.3 - Eucalipto1 (nova estrategia)", "exploracao_eucalipto_23_nova_estrategia.png")
    print(out_euc)

    abre_no_pc(out_pin)
    abre_no_pc(out_euc)


if __name__ == "__main__":
    main()
