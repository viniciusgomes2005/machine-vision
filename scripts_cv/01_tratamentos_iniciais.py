"""
Etapa 1 - tratamentos iniciais.

Aqui transformamos a imagem original em mascaras confiaveis para as outras
medidas. A ideia geral foi:

1. separar o fundo azul dos objetos;
2. detectar o tubete/vaso pela regiao inferior da imagem;
3. cortar a planta acima da boca do vaso, sem deixar o vaso entrar na mascara;
4. limpar residuos da borda do vaso que parecem caule/folha;
5. estimar o ponto de base do caule e uma mascara de caule para as proximas etapas.

O ponto mais delicado é a detecção do topo do vaso. Essencial em TODOS os exercícios. Em algumas imagens a planta
encosta visualmente no tubete, entao uma mascara simples pode confundir folha,
caule e vaso como um objeto só. Por isso o codigo usa geometria, cor escura do
tubete e largura de segmentos para decidir onde realmente comeca o vaso. A regra é geral, entretanto: não há exceção por número de eucalipto.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

HSV_BLUE_LOW = np.array([75, 40, 40], dtype=np.uint8)  # limite inferior do fundo azul em HSV
HSV_BLUE_HIGH = np.array([140, 255, 255], dtype=np.uint8)  # limite superior do fundo azul em HSV
K_OPEN = (3, 3)  # abertura leve para tirar ruido pequeno
K_CLOSE = (7, 7)  # fechamento leve para tapar falhas pequenas
AREA_MIN_OBJ = 40  # menor componente aceito como objeto depois de remover o fundo
AREA_MIN_TUBETE = 1500  # area minima esperada para o corpo do tubete
MARGEM_ACIMA_TUBETE = 2  # folga historica acima do tubete
MARGEM_SEGURA_VASO = 5  # pixels cortados acima do topo do vaso para nao medir a borda
BASE_BAND_UP = 22  # faixa acima do vaso usada para podar tampa/residuo
BASE_BAND_DOWN = 2  # pequena tolerancia abaixo do topo detectado do vaso
BASE_HALF_WIDTH_KEEP = 16  # meia largura preservada perto do caule na poda da tampa
MAX_LARGURA_BASE_COLETO = 30  # segmento mais largo que isso perto da base tende a ser vaso/folha
BOCA_PIX_ACIMA = 26  # quanto procurar acima do topo do vaso para remover boca/tampa
BOCA_PIX_ABAIXO = 16  # quanto procurar abaixo do topo do vaso para remover boca/tampa
BOCA_MEIA_LARGURA = 180  # janela horizontal central da boca do vaso
HSV_VASO_S_MAX = 120  # saturacao maxima para considerar pixel escuro/cinzento de vaso
HSV_VASO_V_MAX = 145  # brilho maximo para considerar pixel escuro do vaso/tubete
MIN_SUPORTE_VERTICAL = 12  # minimo de pixels acima para aceitar uma base como caule
ALTURA_SUPORTE_VERTICAL = 35  # altura da janela de suporte vertical do caule
MEIA_LARGURA_SUPORTE = 4  # meia largura da janela de suporte vertical
ROI_INICIO_TUBETE_FRAC = 0.45  # o tubete sempre deve estar na metade inferior da imagem
LARGURA_MIN_TUBETE_FRAC = 0.08  # largura minima relativa para uma linha parecer corpo de vaso
LINHAS_CONSECUTIVAS_TOPO = 8  # persistencia vertical minima para aceitar topo do vaso
OFFSET_Y_BORDA_TRASEIRA = 2  # ajuste fino da borda traseira quando ela aparece clara
PERSISTENCIA_BORDA_TRASEIRA = 4  # numero de linhas coerentes para confiar na borda traseira
OFFSET_Y_MEDIA_CONFIANCA = 6  # ajuste usado quando a borda aparece com confianca media
OFFSET_Y_BAIXA_CONFIANCA = 2  # ajuste conservador quando a borda traseira nao e confiavel
BASE_SEED_BAND = 120  # faixa de busca de componentes conectados perto da base
RESIDUO_BASE_BANDA_ALTURA = 76  # altura da faixa onde residuos horizontais do vaso aparecem
RESIDUO_BASE_COMP_MAX_H = 14  # altura maxima para residuo fino de base
RESIDUO_BASE_RATIO_MIN = 4.0  # razao largura/altura para classificar barra horizontal
RESIDUO_BASE_MAIN_LARGURA_MIN = 24  # largura minima da regiao principal preservada na base
RESIDUO_BASE_MAIN_KEEP_HALF = 8  # corredor central preservado ao limpar residuo
RESIDUO_BASE_PODA_FAIXA_FINAL = 44  # faixa final onde barras de rodape sao removidas
COMPRIMENTO_BASE_BAND = 8  # faixa usada para recuperar extremos no comprimento
COMPRIMENTO_BASE_AREA_MIN = 15  # menor componente aceito nessa recuperacao
COMPRIMENTO_BASE_WIDTH_MIN = 18  # largura minima de base recuperavel para comprimento
COMPRIMENTO_BASE_HEIGHT_MAX = 12  # altura maxima de residuo recuperavel para comprimento
CAULE_BANDA_PX = 10  # altura de cada banda na estimativa do eixo do caule
CAULE_BEAM_WIDTH = 40  # largura maxima do feixe de candidatos por banda
CAULE_CLUSTER_GAP_PX = 16  # separacao maxima para juntar pontos do eixo
CAULE_LARGURA_MAX_FRAC = 0.28  # largura relativa maxima para ainda parecer caule
CAULE_MIN_PONTOS_EIXO = 4  # minimo de pontos para aceitar o eixo estimado
CAULE_SUAVIZACAO_FRAC = 0.012  # suavizacao relativa do eixo do caule
CAULE_DESVIO_LATERAL_MAX_FRAC = 0.78  # desvio lateral relativo maximo permitido
CAULE_DESVIO_LATERAL_MAX_ABS = 920  # limite absoluto de desvio lateral


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


def _estimar_topo_vaso_escuro_abaixo(
    img_bgr: np.ndarray,
    mask_objetos: np.ndarray,
    y_inicio: int,
    x_centro_ref: int,
    bbox_corpo: Tuple[int, int, int, int],
) -> Optional[int]:
    # Correcao geral para quando folha/caule grudam no tubete: se a borda
    # traseira subiu demais, procuro abaixo dela uma faixa escura e larga,
    # que e muito mais parecida com o corpo real do vaso do que com folha.
    h, _ = mask_objetos.shape
    _, _, bw, _ = bbox_corpo
    bw = int(max(40, bw))

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask_escura = ((s <= HSV_VASO_S_MAX) & (v <= HSV_VASO_V_MAX)).astype(np.uint8) * 255
    mask_escura = cv2.bitwise_and(mask_escura, mask_objetos)
    mask_escura = cv2.morphologyEx(mask_escura, cv2.MORPH_CLOSE, _kernel((5, 3)))

    largura_min = int(max(86, 0.16 * bw))
    x_tol = int(max(42, 0.22 * bw))
    y0 = max(0, min(h - 1, int(y_inicio)))
    y1 = min(h - 1, y0 + 320)

    for y in range(y0, y1 + 1):
        seg = _segmento_principal_linha(mask_escura, y, x_ref=int(x_centro_ref))
        if seg is None:
            continue
        x0, x1 = seg
        largura = int(x1 - x0 + 1)
        xc = int((x0 + x1) / 2)
        if largura >= largura_min and abs(xc - int(x_centro_ref)) <= x_tol:
            return int(y)

    return None


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
        y_topo_frente_detectado = int(y_topo_func)
        y_refinado, dbg_borda = _refinar_topo_borda_traseira(
            mask_objetos=mask_objetos,
            y_topo_frente=y_topo_frente_detectado,
            x_centro_ref=int(x_center_func),
            bbox_corpo=bbox_corpo,
        )
        y_topo_func = int(y_refinado)

        subida_borda = int(y_topo_frente_detectado) - int(y_topo_func)
        if subida_borda > 100:
            # Subida grande demais e sinal de baixa confianca: em vez de aceitar
            # esse topo alto, reconfirmo pelo corpo escuro/largo do tubete.
            y_escuro = _estimar_topo_vaso_escuro_abaixo(
                img_bgr=img_bgr,
                mask_objetos=mask_objetos,
                y_inicio=y_topo_frente_detectado,
                x_centro_ref=int(x_center_func),
                bbox_corpo=bbox_corpo,
            )
            if y_escuro is not None:
                y_topo_func = int(y_escuro)
                dbg_borda["y_topo_corrigido_vaso_escuro"] = int(y_escuro)
            else:
                y_topo_func = int(y_topo_frente_detectado)
                dbg_borda["y_topo_corrigido_vaso_escuro"] = int(y_topo_func)

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


def _extrair_muda_sem_vaso(mask_planta: np.ndarray, y_limite_planta: int, x_centro_tubete: int) -> np.ndarray:
    h, w = mask_planta.shape
    # Fechamento leve para manter continuidade de caule sem engrossar bordas do vaso.
    bridge = cv2.morphologyEx(mask_planta, cv2.MORPH_CLOSE, _kernel((3, 5)))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bridge, connectivity=8)
    if n_labels <= 1:
        return mask_planta.copy()

    y_seed0 = max(0, int(y_limite_planta) - BASE_SEED_BAND)
    y_seed1 = max(0, int(y_limite_planta) - 1)
    x_band = int(max(55, 0.11 * w))
    x0 = max(0, int(x_centro_tubete) - x_band)
    x1 = min(w - 1, int(x_centro_tubete) + x_band)

    melhor_label = 0
    melhor_score = -1e12
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < AREA_MIN_OBJ:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        yb = y + bh - 1
        xc = x + bw // 2
        comp = labels == label

        seed_overlap = False
        if y_seed1 >= y_seed0:
            roi = comp[y_seed0 : y_seed1 + 1, x0 : x1 + 1]
            seed_overlap = bool(np.any(roi))

        verticalidade = float(bh) / max(1.0, float(bw))
        score = float(area) + 1.8 * float(yb) + 140.0 * verticalidade - 1.2 * abs(float(xc - x_centro_tubete))
        if seed_overlap:
            score += 2600.0
        if score > melhor_score:
            melhor_score = score
            melhor_label = label

    if melhor_label == 0:
        return mask_planta.copy()

    principal_bridge = np.zeros_like(mask_planta)
    principal_bridge[labels == melhor_label] = 255

    # Recupera folhas finas conectadas ao corpo principal por pequenas lacunas.
    suporte = cv2.dilate(principal_bridge, _kernel((17, 17)), iterations=1)
    xpb, ypb, wpb, hpb = _bbox_from_mask(principal_bridge)
    px0 = max(0, int(xpb) - 130)
    px1 = min(w - 1, int(xpb + wpb - 1) + 130)
    py0 = max(0, int(ypb) - 240)
    py1 = min(h - 1, int(y_limite_planta) - 10)
    n0, labels0, stats0, _ = cv2.connectedComponentsWithStats(mask_planta, connectivity=8)
    out = np.zeros_like(mask_planta)
    for label in range(1, n0):
        area = int(stats0[label, cv2.CC_STAT_AREA])
        if area < 18:
            continue
        x = int(stats0[label, cv2.CC_STAT_LEFT])
        y = int(stats0[label, cv2.CC_STAT_TOP])
        bw = int(stats0[label, cv2.CC_STAT_WIDTH])
        bh = int(stats0[label, cv2.CC_STAT_HEIGHT])
        yb = y + bh - 1
        comp = labels0 == label
        toca_suporte = np.any(comp & (suporte > 0))
        dentro_bbox_expandido = (
            (x + bw - 1) >= px0
            and x <= px1
            and yb >= py0
            and y <= py1
        )
        if toca_suporte or (dentro_bbox_expandido and area >= 60):
            out[comp] = 255

    if np.count_nonzero(out) == 0:
        out = principal_bridge

    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, _kernel((3, 3)))
    return out


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


def _segmento_mais_proximo_x(row: np.ndarray, x_ref: int) -> Optional[Tuple[int, int]]:
    segs = _segmentos_horizontais(row)
    if not segs:
        return None
    melhor = None
    best_dist = 10**9
    for x0, x1 in segs:
        if x0 <= x_ref <= x1:
            return x0, x1
        dist = min(abs(x_ref - x0), abs(x_ref - x1))
        if dist < best_dist:
            best_dist = dist
            melhor = (x0, x1)
    return melhor


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


def _labels_componentes_visual(mask: np.ndarray) -> np.ndarray:
    n_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    if n_labels <= 1:
        return vis
    for label in range(1, n_labels):
        hue = int((37 * label) % 180)
        comp = (labels == label).astype(np.uint8) * 255
        hsv = np.zeros_like(vis)
        hsv[:, :, 0] = hue
        hsv[:, :, 1] = 220
        hsv[:, :, 2] = comp
        cor = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        vis[comp > 0] = cor[comp > 0]
    return vis


def _limpar_residuo_base_geometrico(
    mask: np.ndarray,
    y_limite_planta: int,
    x_centro_tubete: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    h, w = mask.shape
    y_base = int(max(0, min(h - 1, y_limite_planta - 1)))
    y_inf = max(0, y_base - RESIDUO_BASE_BANDA_ALTURA)

    mask_morf = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel((3, 3)))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_morf, connectivity=8)

    if n_labels <= 1:
        debug = {
            "mask_sem_vaso_inicial": mask.copy(),
            "mask_sem_vaso_pos_morfologia": mask_morf,
            "mask_sem_vaso_componentes": _labels_componentes_visual(mask_morf),
            "mask_residuo_base_removido": np.zeros_like(mask_morf),
            "mask_sem_vaso_final_geometrico": mask_morf.copy(),
        }
        return mask_morf, debug

    # Componente principal: maior suporte vertical + area + proximidade ao centro do vaso.
    label_main = 0
    best_score = -1e12
    for label in range(1, n_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        yb = y + bh - 1
        xc = x + bw // 2
        verticalidade = float(bh) / max(float(bw), 1.0)
        score = float(area) + 2.0 * float(yb) + 180.0 * verticalidade - 0.25 * abs(float(xc - x_centro_tubete))
        if score > best_score:
            best_score = score
            label_main = label

    out = mask_morf.copy()
    removido = np.zeros_like(mask_morf)

    # Ajuste adaptativo para mudas pequenas: tolera componentes menores e poda base mais estreita.
    y_top_main = int(stats[label_main, cv2.CC_STAT_TOP])
    altura_main = int(stats[label_main, cv2.CC_STAT_HEIGHT])
    caso_pequeno = altura_main <= 360
    ratio_min = 3.8 if caso_pequeno else RESIDUO_BASE_RATIO_MIN
    h_max = 12 if caso_pequeno else RESIDUO_BASE_COMP_MAX_H
    keep_half = 6 if caso_pequeno else RESIDUO_BASE_MAIN_KEEP_HALF

    # Remove componentes horizontais residuais na base.
    n_residuos = 0
    for label in range(1, n_labels):
        if label == label_main:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        yb = y + bh - 1
        ratio = float(bw) / max(float(bh), 1.0)
        if yb < y_inf:
            continue
        if bw < 30:
            continue
        if bh > h_max:
            continue
        if ratio < ratio_min:
            continue
        comp = labels == label
        out[comp] = 0
        removido[comp] = 255
        n_residuos += 1

    # Poda no componente principal apenas quando ha evidencia forte de residuo.
    if not (caso_pequeno or n_residuos > 0):
        # Padrão especial: barra horizontal residual persistente na base + cauda fina no fundo.
        x_seed = int(x_centro_tubete)
        ys_b, xs_b = np.where(out[max(0, y_base - 24) : y_base + 1, :] > 0)
        if xs_b.size > 0:
            x_seed = int(np.median(xs_b))

        w_hist = []
        y0 = max(0, y_base - 14)
        y1 = max(0, y_base - 2)
        for y in range(y0, y1 + 1):
            seg = _segmento_mais_proximo_x(out[y, :], x_seed)
            if seg is None:
                continue
            x0, x1 = seg
            w_hist.append(int(x1 - x0 + 1))

        tail_bottom = _segmento_mais_proximo_x(out[y_base, :], x_seed)
        tem_cauda_fina = False
        if tail_bottom is not None:
            bx0, bx1 = tail_bottom
            tem_cauda_fina = int(bx1 - bx0 + 1) <= 4

        hits_barra = sum(1 for ww in w_hist if 28 <= ww <= 46)
        if tem_cauda_fina and hits_barra >= 5:
            keep0 = max(0, int(x_seed) - 6)
            keep1 = min(w - 1, int(x_seed) + 6)
            for y in range(y0, y1 + 1):
                seg = _segmento_mais_proximo_x(out[y, :], x_seed)
                if seg is None:
                    continue
                x0, x1 = seg
                if x0 < keep0:
                    out[y, x0:keep0] = 0
                    removido[y, x0:keep0] = 255
                if x1 > keep1:
                    out[y, keep1 + 1 : x1 + 1] = 0
                    removido[y, keep1 + 1 : x1 + 1] = 255
        else:
            debug = {
                "mask_sem_vaso_inicial": mask.copy(),
                "mask_sem_vaso_pos_morfologia": mask_morf,
                "mask_sem_vaso_componentes": _labels_componentes_visual(mask_morf),
                "mask_residuo_base_removido": removido,
                "mask_sem_vaso_final_geometrico": out,
            }
            return out, debug

    # Poda horizontal residual conectada ao componente principal apenas na faixa inferior.
    main_mask = (labels == label_main).astype(np.uint8) * 255
    x_stem = int(x_centro_tubete)
    ys_low, xs_low = np.where((main_mask > 0) & (np.arange(h)[:, None] >= y_inf))
    if xs_low.size > 0:
        x_stem = int(np.median(xs_low))

    # Refina centro do caule por suporte vertical real na base.
    x_cands = []
    for y in range(y_inf, y_base + 1):
        seg = _segmento_mais_proximo_x(out[y, :], x_stem)
        if seg is None:
            continue
        x0, x1 = seg
        xc = int((x0 + x1) / 2)
        if tem_suporte_vertical(out, xc, y, altura=36, meia_largura=2, min_pixels=12):
            x_cands.append(xc)
    if x_cands:
        x_stem = int(np.median(np.asarray(x_cands, dtype=np.int32)))

    y_poda0 = max(0, y_base - RESIDUO_BASE_PODA_FAIXA_FINAL)
    for y in range(y_poda0, y_base + 1):
        seg = _segmento_mais_proximo_x(out[y, :], x_stem)
        if seg is None:
            continue
        x0, x1 = seg
        largura = int(x1 - x0 + 1)
        if largura < max(RESIDUO_BASE_MAIN_LARGURA_MIN, 2 * keep_half + 6):
            continue
        keep0 = max(0, x_stem - keep_half)
        keep1 = min(w - 1, x_stem + keep_half)
        if x0 < keep0:
            out[y, x0:keep0] = 0
            removido[y, x0:keep0] = 255
        if x1 > keep1:
            out[y, keep1 + 1 : x1 + 1] = 0
            removido[y, keep1 + 1 : x1 + 1] = 255

    # Passo final especifico para barra horizontal residual da borda do vaso.
    x_seed2 = int(x_stem)
    base_band = out[max(0, y_base - 24) : y_base + 1, :]
    ys2, xs2 = np.where(base_band > 0)
    if xs2.size > 0:
        x_seed2 = int(np.median(xs2))
    y_last = int(y_base)
    if ys2.size > 0:
        y_last = int(max(0, y_base - 24 + int(np.max(ys2))))
    y0f = max(0, y_last - 14)
    y1f = max(0, y_last - 2)
    w_hist2 = []
    for y in range(y0f, y1f + 1):
        seg = _segmento_mais_proximo_x(out[y, :], x_seed2)
        if seg is None:
            continue
        x0, x1 = seg
        w_hist2.append(int(x1 - x0 + 1))
    tail2 = _segmento_mais_proximo_x(out[y_last, :], x_seed2)
    tail_ok2 = False
    if tail2 is not None:
        tx0, tx1 = tail2
        tail_ok2 = int(tx1 - tx0 + 1) <= 4
    if tail_ok2 and sum(1 for ww in w_hist2 if 28 <= ww <= 46) >= 5:
        keep0 = max(0, int(x_seed2) - 6)
        keep1 = min(w - 1, int(x_seed2) + 6)
        for y in range(y0f, y1f + 1):
            seg = _segmento_mais_proximo_x(out[y, :], x_seed2)
            if seg is None:
                continue
            x0, x1 = seg
            if x0 < keep0:
                out[y, x0:keep0] = 0
                removido[y, x0:keep0] = 255
            if x1 > keep1:
                out[y, keep1 + 1 : x1 + 1] = 0
                removido[y, keep1 + 1 : x1 + 1] = 255

    debug = {
        "mask_sem_vaso_inicial": mask.copy(),
        "mask_sem_vaso_pos_morfologia": mask_morf,
        "mask_sem_vaso_componentes": _labels_componentes_visual(mask_morf),
        "mask_residuo_base_removido": removido,
        "mask_sem_vaso_final_geometrico": out,
    }
    return out, debug


def _remover_barra_horizontal_rodape(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    out = mask.copy()
    removido = np.zeros_like(mask)
    ys, _ = np.where(out > 0)
    if ys.size == 0:
        return out, removido

    y_last = int(np.max(ys))
    row_last = out[y_last, :]
    segs_last = _segmentos_horizontais(row_last)
    if not segs_last:
        return out, removido
    w_last = max(int(x1 - x0 + 1) for x0, x1 in segs_last)
    if w_last > 4:
        return out, removido

    cands = []
    for y in range(max(0, y_last - 16), max(0, y_last - 1)):
        segs = _segmentos_horizontais(out[y, :])
        if not segs:
            continue
        seg = max(segs, key=lambda s: (s[1] - s[0] + 1))
        x0, x1 = seg
        ww = int(x1 - x0 + 1)
        if 28 <= ww <= 46:
            cands.append((y, x0, x1, ww, int((x0 + x1) / 2)))

    if len(cands) < 5:
        return out, removido

    x_center = int(np.median(np.asarray([c[4] for c in cands], dtype=np.int32)))
    keep0 = max(0, x_center - 6)
    keep1 = min(out.shape[1] - 1, x_center + 6)

    for y, x0, x1, _ww, _xc in cands:
        if x0 < keep0:
            out[y, x0:keep0] = 0
            removido[y, x0:keep0] = 255
        if x1 > keep1:
            out[y, keep1 + 1 : x1 + 1] = 0
            removido[y, keep1 + 1 : x1 + 1] = 255

    return out, removido


def _recuperar_extremos_para_comprimento(
    mask_planta_final: np.ndarray,
    mask_objetos: np.ndarray,
    mask_tubete: np.ndarray,
    y_topo_tubete: int,
) -> np.ndarray:
    sem_tubete = cv2.bitwise_and(mask_objetos, cv2.bitwise_not(mask_tubete))
    candidato = sem_tubete.copy()
    candidato[int(y_topo_tubete) :, :] = 0
    faltante = cv2.bitwise_and(candidato, cv2.bitwise_not(mask_planta_final))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(faltante, connectivity=8)
    out = mask_planta_final.copy()
    y_min_base = int(y_topo_tubete) - COMPRIMENTO_BASE_BAND
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < COMPRIMENTO_BASE_AREA_MIN:
            continue
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        if y < y_min_base:
            continue
        if bw < COMPRIMENTO_BASE_WIDTH_MIN:
            continue
        if bh > COMPRIMENTO_BASE_HEIGHT_MAX:
            continue
        out[labels == label] = 255
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


def _mapa_custo_cor_caule(img_bgr: Optional[np.ndarray], mask_planta_acima: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        return np.full(mask_planta_acima.shape, 20.0, dtype=np.float32)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int16)
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)

    verde_folha = (h >= 30) & (h <= 95) & (s >= 35) & (v >= 35)
    vermelho_caule = ((h <= 18) | (h >= 165)) & (s >= 35) & (v >= 35)
    marrom_escuro = (h <= 35) & (s >= 25) & (v <= 175)
    pouco_verde_escuro = (s <= 95) & (v <= 135)
    pista_caule = vermelho_caule | marrom_escuro | pouco_verde_escuro

    custo = np.full(mask_planta_acima.shape, 24.0, dtype=np.float32)
    custo[verde_folha] += 65.0
    custo[pista_caule] -= 32.0
    custo[mask_planta_acima == 0] = 1e4
    custo = cv2.GaussianBlur(custo, (7, 7), 0)
    return custo.astype(np.float32)


def _agrupar_xs_caule(xs: np.ndarray, gap: int) -> List[np.ndarray]:
    if xs.size == 0:
        return []
    xs_ord = np.sort(xs.astype(np.int32))
    grupos: List[List[int]] = [[int(xs_ord[0])]]
    for x in xs_ord[1:]:
        if int(x) - grupos[-1][-1] > gap:
            grupos.append([])
        grupos[-1].append(int(x))
    return [np.array(g, dtype=np.int32) for g in grupos if g]


def _mask_linha_eixo(shape: Tuple[int, int], pontos_yx: List[Tuple[int, int]], espessura: int = 1) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if not pontos_yx:
        return mask
    if len(pontos_yx) == 1:
        y, x = pontos_yx[0]
        mask[int(y), int(x)] = 255
        return mask
    for (y0, x0), (y1, x1) in zip(pontos_yx, pontos_yx[1:]):
        cv2.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, int(espessura), lineType=cv2.LINE_AA)
    return (mask > 0).astype(np.uint8) * 255


def _suavizar_eixo_caule(pontos_yx: List[Tuple[float, float]], altura: int) -> List[Tuple[int, int]]:
    if len(pontos_yx) <= 2:
        return [(int(round(y)), int(round(x))) for y, x in pontos_yx]

    ys = np.array([p[0] for p in pontos_yx], dtype=np.float64)
    xs = np.array([p[1] for p in pontos_yx], dtype=np.float64)
    grau = 2 if len(pontos_yx) < 7 else 3
    try:
        poly = np.poly1d(np.polyfit(ys, xs, grau))
        xs_fit = poly(ys)
    except np.linalg.LinAlgError:
        xs_fit = xs

    max_desvio = max(12.0, 0.04 * float(max(1, altura)))
    xs_suave = np.clip(xs_fit, xs - max_desvio, xs + max_desvio)
    caminho = [(int(round(float(y))), int(round(float(x)))) for y, x in zip(ys, xs_suave)]

    epsilon = max(4.0, min(16.0, CAULE_SUAVIZACAO_FRAC * float(max(1, altura))))
    pts = np.array([[x, y] for y, x in caminho], dtype=np.float32).reshape((-1, 1, 2))
    approx = cv2.approxPolyDP(pts, epsilon, False).reshape((-1, 2))
    if len(approx) >= 2:
        caminho = [(int(round(float(y))), int(round(float(x)))) for x, y in approx]
    return caminho


def _estimar_eixo_caule_por_pista(
    mask_planta_acima: np.ndarray,
    ponto_base: Tuple[int, int],
    skel: np.ndarray,
    img_bgr: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    ys_all, xs_all = np.where(mask_planta_acima > 0)
    if ys_all.size == 0:
        return None

    h, w = mask_planta_acima.shape
    x_base, y_base = int(ponto_base[0]), int(ponto_base[1])
    y_topo = int(ys_all.min())
    altura = int(max(1, y_base - y_topo))
    largura_bbox = int(xs_all.max() - xs_all.min() + 1)
    largura_max = max(22, int(round(CAULE_LARGURA_MAX_FRAC * float(largura_bbox))))
    desvio_lateral_max = int(
        max(140, min(CAULE_DESVIO_LATERAL_MAX_ABS, round(CAULE_DESVIO_LATERAL_MAX_FRAC * float(largura_bbox))))
    )
    gap = max(CAULE_CLUSTER_GAP_PX, int(round(0.02 * float(largura_bbox))))
    passo = max(6, int(CAULE_BANDA_PX))
    custo_cor = _mapa_custo_cor_caule(img_bgr, mask_planta_acima)

    bandas: List[List[Dict[str, float]]] = []
    for y_centro in range(y_base, y_topo - 1, -passo):
        y0 = max(0, int(y_centro - passo // 2))
        y1 = min(h, int(y_centro + passo // 2 + 1))
        candidatos: List[Dict[str, float]] = []

        ys_skel, xs_skel = np.where(skel[y0:y1, :] > 0)
        for grupo in _agrupar_xs_caule(xs_skel, gap):
            largura = int(grupo.max() - grupo.min() + 1)
            if largura > largura_max:
                continue
            x_med = int(round(float(np.median(grupo))))
            if abs(int(x_med) - int(x_base)) > desvio_lateral_max:
                continue
            idx_grupo = np.isin(xs_skel, grupo)
            y_med = int(round(float(y0 + np.median(ys_skel[idx_grupo])))) if np.any(idx_grupo) else int(y_centro)
            candidatos.append(
                {
                    "x": float(x_med),
                    "y": float(y_med),
                    "largura": float(largura),
                    "custo_cor": float(custo_cor[min(h - 1, max(0, y_med)), min(w - 1, max(0, x_med))]),
                }
            )

        if not candidatos:
            strip = mask_planta_acima[y0:y1, :]
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((strip > 0).astype(np.uint8), connectivity=8)
            for label in range(1, n_labels):
                largura = int(stats[label, cv2.CC_STAT_WIDTH])
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < 2 or largura > largura_max:
                    continue
                ys_lbl, xs_lbl = np.where(labels == label)
                if xs_lbl.size == 0:
                    continue
                x_med = int(round(float(np.median(xs_lbl))))
                if abs(int(x_med) - int(x_base)) > desvio_lateral_max:
                    continue
                y_med = int(round(float(y0 + np.median(ys_lbl))))
                candidatos.append(
                    {
                        "x": float(x_med),
                        "y": float(y_med),
                        "largura": float(largura),
                        "custo_cor": float(custo_cor[min(h - 1, max(0, y_med)), min(w - 1, max(0, x_med))]) + 18.0,
                    }
                )

        unicos: List[Dict[str, float]] = []
        for cand in sorted(candidatos, key=lambda c: c["custo_cor"] + 0.2 * c["largura"]):
            if any(abs(cand["x"] - outro["x"]) <= 5 for outro in unicos):
                continue
            unicos.append(cand)
        if unicos:
            bandas.append(unicos[:10])

    if len(bandas) < CAULE_MIN_PONTOS_EIXO:
        return None

    beam: List[Dict[str, object]] = [
        {
            "custo": 0.0,
            "pontos": [(float(y_base), float(x_base))],
            "ultimo": {"x": float(x_base), "y": float(y_base), "largura": 3.0, "custo_cor": -20.0},
            "slope": 0.0,
        }
    ]
    melhores: List[Dict[str, object]] = []
    for banda in bandas:
        novos: List[Dict[str, object]] = []
        for estado in beam:
            ultimo = estado["ultimo"]
            for cand in banda:
                dy = max(1.0, float(ultimo["y"]) - float(cand["y"]))
                if dy <= 0:
                    continue
                dx = float(cand["x"]) - float(ultimo["x"])
                slope = dx / dy
                curvatura = abs(slope - float(estado["slope"]))
                custo = (
                    float(estado["custo"])
                    + 0.20 * abs(dx)
                    + 70.0 * curvatura
                    + 0.20 * float(cand["largura"])
                    + float(cand["custo_cor"])
                )
                pontos = list(estado["pontos"]) + [(float(cand["y"]), float(cand["x"]))]
                novos.append({"custo": custo, "pontos": pontos, "ultimo": cand, "slope": slope})
        if not novos:
            continue
        novos.sort(key=lambda e: float(e["custo"]))
        beam = novos[:CAULE_BEAM_WIDTH]
        melhores.extend(beam[: min(8, len(beam))])

    if not melhores:
        return None

    elegiveis = [
        e for e in melhores if float(e["pontos"][0][0] - e["pontos"][-1][0]) >= 0.45 * float(altura)
    ] or melhores

    def score_estado(e: Dict[str, object]) -> float:
        pontos = e["pontos"]
        subida = float(pontos[0][0] - pontos[-1][0])
        return 3.0 * subida - 0.22 * float(e["custo"])

    escolhido = max(elegiveis, key=score_estado)
    caminho = _suavizar_eixo_caule(escolhido["pontos"], altura)
    if len(caminho) < 2:
        return None

    path_mask = _mask_linha_eixo(mask_planta_acima.shape, caminho, espessura=2)
    mask_caule = cv2.dilate(path_mask, _kernel((7, 7)), iterations=1)
    mask_caule = cv2.bitwise_and(mask_caule, cv2.dilate(mask_planta_acima, _kernel((5, 5)), iterations=1))
    return mask_caule


def estimar_eixo_caule(
    mask_planta_acima: np.ndarray,
    ponto_base: Tuple[int, int],
    img_bgr: Optional[np.ndarray] = None,
    usar_pista_cor: bool = True,
) -> np.ndarray:
    skel = skeletonize(mask_planta_acima)
    h, w = skel.shape

    ponto_inicio = _ajustar_para_skeleton(skel, ponto_base)
    if ponto_inicio is None:
        return np.zeros_like(mask_planta_acima)

    if usar_pista_cor:
        mask_pista = _estimar_eixo_caule_por_pista(mask_planta_acima, ponto_base, skel, img_bgr)
        if mask_pista is not None and cv2.countNonZero(mask_pista) > 0:
            return mask_pista

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
    mask_planta_acima, debug_residuo = _limpar_residuo_base_geometrico(
        mask_planta_acima,
        y_limite_planta=int(y_limite_planta),
        x_centro_tubete=int(x_centro_tubete),
    )
    mask_planta_acima, mask_barra_rodape_removida = _remover_barra_horizontal_rodape(mask_planta_acima)
    debug_residuo["mask_barra_horizontal_rodape_removida"] = mask_barra_rodape_removida
    mask_planta_acima[y_limite_planta:, :] = 0
    mask_comprimento_total = _recuperar_extremos_para_comprimento(
        mask_planta_final=mask_planta_acima,
        mask_objetos=mask_objetos,
        mask_tubete=mask_tubete,
        y_topo_tubete=int(y_topo_tubete),
    )
    ponto_base = achar_base_coleto(
        mask_planta_acima,
        y_topo_tubete,
        x_centro_tubete,
        mask_tubete=mask_tubete,
        mask_boca_removida=mask_boca_removida,
    )

    skeleton = skeletonize(mask_planta_acima)
    skeleton[y_limite_planta:, :] = 0
    mask_caule = estimar_eixo_caule(mask_planta_acima, ponto_base, usar_pista_cor=False)
    mask_caule_refinado_comprimento = estimar_eixo_caule(
        mask_planta_acima,
        ponto_base,
        img_bgr=img_bgr,
        usar_pista_cor=True,
    )
    mask_caule[y_limite_planta:, :] = 0
    mask_caule_refinado_comprimento[y_limite_planta:, :] = 0

    if debug_dir is not None:
        debug_masks = {
            "mask_fundo": mask_fundo,
            "mask_objetos": mask_objetos,
            "mask_tubete": mask_tubete,
            "mask_planta_acima": mask_planta_acima,
            "mask_planta_antes_remover_boca_vaso": mask_planta_antes_remover_boca,
            "mask_planta_sem_boca_vaso": mask_planta_acima,
            "mask_comprimento_total": mask_comprimento_total,
            "skeleton": skeleton,
            "mask_caule": mask_caule,
            "mask_caule_refinado_comprimento": mask_caule_refinado_comprimento,
        }
        if isinstance(debug_tubete, dict):
            if "mask_vaso_candidata" in debug_tubete:
                debug_masks["mask_vaso_candidata"] = debug_tubete["mask_vaso_candidata"]
            if "mask_corpo_vaso_raw" in debug_tubete:
                debug_masks["mask_corpo_vaso_raw"] = debug_tubete["mask_corpo_vaso_raw"]
            if "mask_corpo_vaso" in debug_tubete:
                debug_masks["mask_corpo_vaso"] = debug_tubete["mask_corpo_vaso"]
        debug_masks.update(debug_residuo)
        debug_masks["mask_boca_removida"] = mask_boca_removida
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
        "mask_comprimento_total": mask_comprimento_total,
        "skeleton": skeleton,
        "mask_caule": mask_caule,
        "mask_caule_refinado_comprimento": mask_caule_refinado_comprimento,
    }
