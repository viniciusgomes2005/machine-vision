from __future__ import annotations

import argparse
import csv
import heapq
import importlib.util
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

PODA_RAMO_CURTO_PX = 12
FAIXA_BASE_SEED_PX = 12
GANHO_VERTICAL_MIN_FRAC = 0.15
COMPRIMENTO_MIN_FRAC = 0.15
DESVIO_LATERAL_LIMITE_FRAC = 0.35
DESVIO_LATERAL_LIMITE_ABS = 60.0
DESVIO_LATERAL_PENALIDADE_FRAC = 0.25
PROGRESSO_VERTICAL_MIN = 0.25
PROGRESSO_VERTICAL_PENALIDADE_BASE = 0.35
PROGRESSO_VERTICAL_ALVO = 0.52
PESO_PROGRESSO_VERTICAL_SCORE = 95.0
PESO_COMPRIMENTO_EXCESSIVO_SCORE = 0.55
SKELETON_REPARO_VERTICAL_FRAC = 0.20
SKELETON_REPARO_HORIZONTAL_FRAC = 0.04
RAIO_FOLHA_EXCESSO_PX = 6.0
PENALIDADE_CAMINHO_FOLHA = 0.85
PENALIDADE_ENDPOINT_FOLHA = 18.0
PESO_SUBIDA_SCORE = 1.20
PESO_COMPRIMENTO_SCORE = 0.45
PESO_DESVIO_LATERAL_SCORE = 0.35
PESO_DESVIO_LATERAL_EXCESSIVO = 1.5
ALTURA_MAX_MUDA_PEQUENA_PX = 360
MIN_PIXELS_ANTES_RAMIFICACAO = 12
ENDPOINT_TOPO_BANDA_FRAC = 0.10
ENDPOINT_TOPO_PROG_MIN = 0.45
SUAVIZACAO_CAMINHO_FRAC = 0.012
SUAVIZACAO_CAMINHO_MIN_PX = 5.0
SUAVIZACAO_CAMINHO_MAX_PX = 18.0
CORREDOR_CAULE_FRAC = 0.035
CORREDOR_CAULE_MIN_PX = 14
CORREDOR_CAULE_MAX_PX = 45
EIXO_BANDA_PX = 10
EIXO_BEAM_WIDTH = 48
EIXO_CLUSTER_GAP_PX = 18
EIXO_LARGURA_REL_MAX = 0.30
EIXO_DIST_FOLHA_PX = 7.0
EIXO_MIN_PONTOS = 4
CORTE_TERMINAL_MIN_FRAC = 0.55
CORTE_TERMINAL_LARGURA_FRAC = 0.09
CORTE_TERMINAL_ESPESSURA_PX = 9.5
CORTE_TERMINAL_RUN = 2
CORTE_TERMINAL_FRAC_SEGMENTO = 0.45
AJUSTE_EIXO_LARGO_FRAC_ALTURA = 0.095
AJUSTE_EIXO_LARGO_RAZAO_MIN = 0.55
MUDA_PEQUENA_COMPR_MIN_PX = 72.0
MUDA_PEQUENA_EXTENSAO_MAX_PX = 46.0


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Falha ao carregar modulo: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE_DIR = Path(__file__).resolve().parent
M01 = _load_module(BASE_DIR / "01_tratamentos_iniciais.py", "m01_tratamentos")


def _kernel(size: Tuple[int, int]) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)


def _normalizar_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255


def _bbox_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    bin_mask = _normalizar_mask(mask)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(bin_mask)

    img = (bin_mask > 0).astype(np.uint8)
    mudou = True
    while mudou:
        mudou = False
        for etapa in (0, 1):
            p2 = img[:-2, 1:-1]
            p3 = img[:-2, 2:]
            p4 = img[1:-1, 2:]
            p5 = img[2:, 2:]
            p6 = img[2:, 1:-1]
            p7 = img[2:, :-2]
            p8 = img[1:-1, :-2]
            p9 = img[:-2, :-2]
            centro = img[1:-1, 1:-1]

            vizinhos = [p2, p3, p4, p5, p6, p7, p8, p9]
            b = sum(vizinhos)
            a = np.zeros_like(centro)
            ciclo = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            for atual, prox in zip(ciclo, ciclo[1:]):
                a += ((atual == 0) & (prox == 1)).astype(np.uint8)

            if etapa == 0:
                cond_extra = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond_extra = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)

            remover = (centro == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond_extra
            if np.any(remover):
                centro[remover] = 0
                mudou = True

    return img.astype(np.uint8) * 255


def _odd(v: int) -> int:
    return v if v % 2 == 1 else v + 1


def _skeletonize_em_bbox(mask: np.ndarray, pad: int = 24) -> np.ndarray:
    mask_u8 = _normalizar_mask(mask)
    if cv2.countNonZero(mask_u8) == 0:
        return np.zeros_like(mask_u8)

    h, w = mask_u8.shape
    x_min, y_min, x_max, y_max = _bbox_mask(mask_u8)
    x0 = max(0, x_min - pad)
    y0 = max(0, y_min - pad)
    x1 = min(w - 1, x_max + pad)
    y1 = min(h - 1, y_max + pad)

    crop = mask_u8[y0 : y1 + 1, x0 : x1 + 1]
    skel_crop = _skeletonize(crop)
    skel = np.zeros_like(mask_u8)
    skel[y0 : y1 + 1, x0 : x1 + 1] = skel_crop
    return skel


def _preparar_skeleton_busca(
    planta: np.ndarray,
    skeleton: np.ndarray | None,
    largura_bbox: int,
    altura_bbox: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    skel_original = _normalizar_mask(skeleton) if skeleton is not None else _skeletonize_em_bbox(planta)
    skel_original = cv2.bitwise_and(skel_original, planta)

    kx = _odd(max(17, min(51, int(round(largura_bbox * SKELETON_REPARO_HORIZONTAL_FRAC)))))
    ky = _odd(max(101, min(301, int(round(altura_bbox * SKELETON_REPARO_VERTICAL_FRAC)))))
    planta_reparada = cv2.morphologyEx(planta, cv2.MORPH_CLOSE, _kernel((kx, ky)))
    planta_reparada = cv2.morphologyEx(planta_reparada, cv2.MORPH_OPEN, _kernel((3, 3)))
    skel_reparado = _skeletonize_em_bbox(planta_reparada)

    n_original, _labels, _stats, _cent = cv2.connectedComponentsWithStats(skel_original, connectivity=8)
    usar_reparo = n_original > 2
    if usar_reparo:
        return skel_reparado, skel_original, True
    return skel_original, skel_original, False


def _vizinhos8(y: int, x: int, h: int, w: int) -> Iterable[Tuple[int, int, float]]:
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yn = y + dy
            xn = x + dx
            if 0 <= yn < h and 0 <= xn < w:
                custo = math.sqrt(2.0) if dy != 0 and dx != 0 else 1.0
                yield yn, xn, custo


def _grau_pixel(skel: np.ndarray, y: int, x: int) -> int:
    h, w = skel.shape
    grau = 0
    for yn, xn, _ in _vizinhos8(y, x, h, w):
        if skel[yn, xn] > 0:
            grau += 1
    return grau


def _podar_ramos_curtos(skeleton: np.ndarray, min_len: int = PODA_RAMO_CURTO_PX) -> np.ndarray:
    skel = _normalizar_mask(skeleton).copy()
    h, w = skel.shape

    mudou = True
    while mudou:
        mudou = False
        ys, xs = np.where(skel > 0)
        endpoints = [(int(y), int(x)) for y, x in zip(ys, xs) if _grau_pixel(skel, int(y), int(x)) == 1]

        for y0, x0 in endpoints:
            if skel[y0, x0] == 0:
                continue

            caminho = [(y0, x0)]
            anterior: Tuple[int, int] | None = None
            atual = (y0, x0)

            while True:
                y, x = atual
                viz = []
                for yn, xn, _ in _vizinhos8(y, x, h, w):
                    if skel[yn, xn] > 0 and (anterior is None or (yn, xn) != anterior):
                        viz.append((yn, xn))

                grau = _grau_pixel(skel, y, x)
                if atual != (y0, x0) and grau != 2:
                    break
                if len(viz) != 1:
                    break

                anterior = atual
                atual = viz[0]
                caminho.append(atual)
                if len(caminho) > min_len:
                    break

            if len(caminho) <= min_len:
                for y, x in caminho[:-1]:
                    skel[y, x] = 0
                mudou = True

    return skel


def _estimar_base(
    planta: np.ndarray,
    bbox: Tuple[int, int, int, int],
    mask_caule_seed: np.ndarray | None,
) -> Tuple[int, int, str]:
    x_min, _y_min, x_max, y_max = bbox

    if mask_caule_seed is not None:
        seed = cv2.bitwise_and(_normalizar_mask(mask_caule_seed), planta)
        ys, xs = np.where(seed > 0)
        if xs.size > 0:
            y_base = int(ys.max())
            faixa = ys >= y_base - FAIXA_BASE_SEED_PX
            if np.any(faixa):
                x_base = int(round(float(np.median(xs[faixa]))))
            else:
                x_base = int(round(float(np.median(xs))))
            return x_base, y_base, "seed_caule"

    return int(round((x_min + x_max) / 2.0)), int(y_max), "centro_bbox"


def _ponto_skeleton_mais_proximo(skeleton: np.ndarray, x_base: int, y_base: int) -> Tuple[int, int]:
    ys, xs = np.where(skeleton > 0)
    if xs.size == 0:
        raise ValueError("Skeleton vazio: nao ha caminho para medir.")
    dist2 = (xs.astype(np.float64) - float(x_base)) ** 2 + (ys.astype(np.float64) - float(y_base)) ** 2
    idx = int(np.argmin(dist2))
    return int(ys[idx]), int(xs[idx])


def _dijkstra_skeleton(
    skeleton: np.ndarray,
    inicio: Tuple[int, int],
    mapa_espessura: np.ndarray | None = None,
    mapa_dist_guia: np.ndarray | None = None,
):
    h, w = skeleton.shape
    dist = np.full((h, w), np.inf, dtype=np.float64)
    custo_folha = np.full((h, w), np.inf, dtype=np.float64)
    parent_y = np.full((h, w), -1, dtype=np.int32)
    parent_x = np.full((h, w), -1, dtype=np.int32)
    visitado = np.zeros((h, w), dtype=bool)

    y0, x0 = inicio
    dist[y0, x0] = 0.0
    custo_folha[y0, x0] = 0.0
    heap: List[Tuple[float, int, int]] = [(0.0, y0, x0)]

    while heap:
        d, y, x = heapq.heappop(heap)
        if visitado[y, x]:
            continue
        visitado[y, x] = True

        for yn, xn, custo in _vizinhos8(y, x, h, w):
            if skeleton[yn, xn] == 0:
                continue
            nd = d + custo
            excesso_folha = 0.0
            if mapa_espessura is not None:
                excesso_folha = max(0.0, float(mapa_espessura[yn, xn]) - RAIO_FOLHA_EXCESSO_PX)
            custo_guia = 0.0
            if mapa_dist_guia is not None:
                custo_guia = 0.06 * float(mapa_dist_guia[yn, xn])
            nd = nd + custo_guia
            if nd < dist[yn, xn]:
                dist[yn, xn] = nd
                custo_folha[yn, xn] = custo_folha[y, x] + excesso_folha * custo
                parent_y[yn, xn] = y
                parent_x[yn, xn] = x
                heapq.heappush(heap, (nd, yn, xn))

    return dist, custo_folha, parent_y, parent_x, visitado


def _comprimento_util_para_score(subida: float, comprimento: float) -> Tuple[float, float, float]:
    if comprimento <= 0.0:
        return 0.0, 0.0, 0.0
    progresso_vertical = subida / max(1.0, comprimento)
    comprimento_util = min(comprimento, subida / max(PROGRESSO_VERTICAL_ALVO, 1e-6))
    comprimento_excessivo = max(0.0, comprimento - comprimento_util)
    return float(comprimento_util), float(comprimento_excessivo), float(progresso_vertical)


def _escolher_ponto_final(
    skeleton_busca: np.ndarray,
    visitado: np.ndarray,
    dist: np.ndarray,
    custo_folha: np.ndarray,
    mapa_espessura: np.ndarray,
    x_base: int,
    y_base: int,
    largura_bbox: int,
    altura_bbox: int,
) -> Tuple[int, int, float, bool, str]:
    ys, xs = np.where(visitado)
    melhor: Tuple[int, int] | None = None
    melhor_score = -np.inf
    fallback = False
    motivo = "criterios_minimos"

    min_subida = GANHO_VERTICAL_MIN_FRAC * float(altura_bbox)
    min_compr = COMPRIMENTO_MIN_FRAC * float(altura_bbox)
    limite_lateral = max(DESVIO_LATERAL_LIMITE_ABS, DESVIO_LATERAL_LIMITE_FRAC * float(largura_bbox))

    endpoints_topo: List[Tuple[float, int, int, float]] = []
    y_topo_visitado = int(ys.min()) if ys.size > 0 else y_base
    banda_topo = max(18, int(round(ENDPOINT_TOPO_BANDA_FRAC * float(altura_bbox))))
    for y, x in zip(ys, xs):
        comprimento = float(dist[y, x])
        if not np.isfinite(comprimento) or comprimento <= 0.0:
            continue
        if int(y) > y_topo_visitado + banda_topo:
            continue

        subida = float(y_base - int(y))
        comprimento_util, comprimento_excessivo, progresso_vertical = _comprimento_util_para_score(subida, comprimento)
        if subida < min_subida or comprimento < min_compr or progresso_vertical < ENDPOINT_TOPO_PROG_MIN:
            continue
        if _grau_pixel(skeleton_busca, int(y), int(x)) > 1:
            continue

        desvio_lateral = abs(float(int(x) - x_base))
        penalidade_folha = float(custo_folha[y, x]) * PENALIDADE_CAMINHO_FOLHA
        score_topo = (
            2.0 * subida
            + 0.35 * comprimento_util
            + 80.0 * progresso_vertical
            - 0.12 * desvio_lateral
            - PESO_COMPRIMENTO_EXCESSIVO_SCORE * comprimento_excessivo
            - penalidade_folha
        )
        endpoints_topo.append((float(score_topo), int(y), int(x), float(comprimento)))

    if endpoints_topo:
        endpoints_topo.sort(reverse=True)
        score_topo, y_topo, x_topo, _comprimento_topo = endpoints_topo[0]
        return y_topo, x_topo, float(score_topo), False, "endpoint_topo_eixo_principal"

    for y, x in zip(ys, xs):
        comprimento = float(dist[y, x])
        if not np.isfinite(comprimento) or comprimento <= 0.0:
            continue

        subida = float(y_base - int(y))
        desvio_lateral = abs(float(int(x) - x_base))
        comprimento_util, comprimento_excessivo, progresso_vertical = _comprimento_util_para_score(subida, comprimento)

        if subida < min_subida or comprimento < min_compr:
            continue
        if desvio_lateral > limite_lateral * 1.35 or progresso_vertical < 0.05:
            continue

        penalidade_lateral_excessiva = max(
            0.0,
            desvio_lateral - DESVIO_LATERAL_PENALIDADE_FRAC * float(largura_bbox),
        ) * PESO_DESVIO_LATERAL_EXCESSIVO
        penalidade_horizontalidade = max(
            0.0,
            PROGRESSO_VERTICAL_PENALIDADE_BASE - progresso_vertical,
        ) * float(altura_bbox)
        penalidade_limite_lateral = max(0.0, desvio_lateral - limite_lateral) * 2.0
        penalidade_folha = (
            float(custo_folha[y, x]) * PENALIDADE_CAMINHO_FOLHA
            + max(0.0, float(mapa_espessura[y, x]) - RAIO_FOLHA_EXCESSO_PX) * PENALIDADE_ENDPOINT_FOLHA
        )

        score = (
            PESO_SUBIDA_SCORE * subida
            + PESO_COMPRIMENTO_SCORE * comprimento_util
            + PESO_PROGRESSO_VERTICAL_SCORE * progresso_vertical
            - PESO_DESVIO_LATERAL_SCORE * desvio_lateral
            - PESO_COMPRIMENTO_EXCESSIVO_SCORE * comprimento_excessivo
            - penalidade_lateral_excessiva
            - penalidade_horizontalidade
            - penalidade_limite_lateral
            - penalidade_folha
        )

        if progresso_vertical < PROGRESSO_VERTICAL_MIN:
            score -= (PROGRESSO_VERTICAL_MIN - progresso_vertical) * float(altura_bbox) * 1.5

        if score > melhor_score:
            melhor_score = float(score)
            melhor = (int(y), int(x))

    if melhor is not None:
        return melhor[0], melhor[1], melhor_score, fallback, motivo

    fallback = True
    motivo = "sem_candidato_score_valido"
    for y, x in zip(ys, xs):
        comprimento = float(dist[y, x])
        if not np.isfinite(comprimento) or comprimento <= 0.0:
            continue

        subida = float(y_base - int(y))
        if subida <= 0.0:
            continue
        desvio_lateral = abs(float(int(x) - x_base))
        comprimento_util, comprimento_excessivo, progresso_vertical = _comprimento_util_para_score(subida, comprimento)
        score = 1.0 * subida + 0.20 * comprimento_util - 1.20 * desvio_lateral
        score -= PESO_COMPRIMENTO_EXCESSIVO_SCORE * comprimento_excessivo
        score -= max(0.0, PROGRESSO_VERTICAL_MIN - progresso_vertical) * float(altura_bbox)
        score -= float(custo_folha[y, x]) * PENALIDADE_CAMINHO_FOLHA

        if score > melhor_score:
            melhor_score = float(score)
            melhor = (int(y), int(x))

    if melhor is None:
        y0, x0 = int(ys[0]), int(xs[0])
        return y0, x0, 0.0, True, "sem_pixel_ascendente"

    return melhor[0], melhor[1], melhor_score, fallback, motivo


def _reconstruir_caminho(
    inicio: Tuple[int, int],
    fim: Tuple[int, int],
    parent_y: np.ndarray,
    parent_x: np.ndarray,
) -> List[Tuple[int, int]]:
    caminho = [fim]
    atual = fim
    limite = parent_y.size + 1

    while atual != inicio and len(caminho) < limite:
        y, x = atual
        py = int(parent_y[y, x])
        px = int(parent_x[y, x])
        if py < 0 or px < 0:
            break
        atual = (py, px)
        caminho.append(atual)

    caminho.reverse()
    return caminho


def _comprimento_real_caminho(caminho: List[Tuple[int, int]]) -> float:
    total = 0.0
    for (y0, x0), (y1, x1) in zip(caminho, caminho[1:]):
        dy = abs(int(y1) - int(y0))
        dx = abs(int(x1) - int(x0))
        if dx == 1 and dy == 1:
            total += math.sqrt(2.0)
        elif dx + dy == 1:
            total += 1.0
        else:
            total += math.hypot(dx, dy)
    return float(total)


def _truncar_caminho_por_comprimento(
    caminho: List[Tuple[int, int]],
    comprimento_alvo: float,
) -> List[Tuple[int, int]]:
    if len(caminho) <= 1 or comprimento_alvo <= 0.0:
        return caminho[:1]

    novo = [caminho[0]]
    acumulado = 0.0
    for (y0, x0), (y1, x1) in zip(caminho, caminho[1:]):
        seg = math.hypot(float(x1) - float(x0), float(y1) - float(y0))
        if seg <= 0.0:
            continue
        if acumulado + seg >= comprimento_alvo:
            frac = max(0.0, min(1.0, (float(comprimento_alvo) - acumulado) / seg))
            y = int(round(float(y0) + frac * float(y1 - y0)))
            x = int(round(float(x0) + frac * float(x1 - x0)))
            if (y, x) != novo[-1]:
                novo.append((y, x))
            return novo
        novo.append((int(y1), int(x1)))
        acumulado += seg

    return novo


def _largura_segmento_linha(mask: np.ndarray, y: int, x: int) -> int:
    h, w = mask.shape
    if not (0 <= int(y) < h and 0 <= int(x) < w):
        return 0
    linha = mask[int(y), :] > 0
    if not linha[int(x)]:
        xs = np.where(linha)[0]
        if xs.size == 0:
            return 0
        idx = int(np.argmin(np.abs(xs.astype(np.int32) - int(x))))
        x = int(xs[idx])
    x0 = int(x)
    while x0 > 0 and linha[x0 - 1]:
        x0 -= 1
    x1 = int(x)
    while x1 < w - 1 and linha[x1 + 1]:
        x1 += 1
    return int(x1 - x0 + 1)


def _cortar_terminal_folha(
    caminho: List[Tuple[int, int]],
    planta: np.ndarray,
    mapa_espessura: np.ndarray,
    largura_bbox: int,
) -> Tuple[List[Tuple[int, int]], Dict[str, float | int | bool]]:
    if len(caminho) < 4:
        return caminho, {"corte_terminal_usado": False, "indice_corte_terminal": len(caminho) - 1}

    comprimento_total = _comprimento_real_caminho(caminho)
    min_antes_corte = max(80.0, CORTE_TERMINAL_MIN_FRAC * comprimento_total)
    largura_limite = max(42.0, CORTE_TERMINAL_LARGURA_FRAC * float(largura_bbox))
    acumulado = 0.0
    ruins = 0
    primeiro_ruim_idx = -1
    primeiro_ruim_acumulado = 0.0

    for idx in range(1, len(caminho)):
        y0, x0 = caminho[idx - 1]
        y, x = caminho[idx]
        seg = math.hypot(float(x) - float(x0), float(y) - float(y0))
        acumulado += seg
        if acumulado < min_antes_corte:
            continue

        largura = float(_largura_segmento_linha(planta, int(y), int(x)))
        espessura = float(mapa_espessura[int(y), int(x)]) if 0 <= int(y) < mapa_espessura.shape[0] and 0 <= int(x) < mapa_espessura.shape[1] else 0.0
        entrou_folha = largura >= largura_limite or espessura >= CORTE_TERMINAL_ESPESSURA_PX
        if entrou_folha:
            if ruins == 0:
                primeiro_ruim_idx = idx
                primeiro_ruim_acumulado = acumulado
            ruins += 1
        else:
            ruins = 0
            primeiro_ruim_idx = -1
            primeiro_ruim_acumulado = 0.0
        if ruins >= CORTE_TERMINAL_RUN:
            idx_corte_base = max(1, primeiro_ruim_idx)
            y_prev, x_prev = caminho[idx_corte_base - 1]
            y_ruim, x_ruim = caminho[idx_corte_base]
            y_interp = int(round(float(y_prev) + CORTE_TERMINAL_FRAC_SEGMENTO * float(y_ruim - y_prev)))
            x_interp = int(round(float(x_prev) + CORTE_TERMINAL_FRAC_SEGMENTO * float(x_ruim - x_prev)))
            caminho_cortado = caminho[:idx_corte_base]
            if (y_interp, x_interp) != caminho_cortado[-1]:
                caminho_cortado.append((y_interp, x_interp))
            return caminho_cortado, {
                "corte_terminal_usado": True,
                "indice_corte_terminal": int(idx_corte_base),
                "largura_terminal_px": float(largura),
                "espessura_terminal_px": float(espessura),
                "comprimento_antes_corte_px": float(acumulado),
                "comprimento_primeiro_terminal_px": float(primeiro_ruim_acumulado),
            }

    return caminho, {
        "corte_terminal_usado": False,
        "indice_corte_terminal": len(caminho) - 1,
        "largura_terminal_px": 0.0,
        "espessura_terminal_px": 0.0,
    }


def _suavizar_caminho(
    caminho: List[Tuple[int, int]],
    altura_bbox: int,
) -> List[Tuple[int, int]]:
    if len(caminho) <= 2:
        return caminho[:]

    epsilon = max(
        SUAVIZACAO_CAMINHO_MIN_PX,
        min(SUAVIZACAO_CAMINHO_MAX_PX, float(altura_bbox) * SUAVIZACAO_CAMINHO_FRAC),
    )
    pontos = np.array([[int(x), int(y)] for y, x in caminho], dtype=np.float32).reshape((-1, 1, 2))
    aproximado = cv2.approxPolyDP(pontos, epsilon, False).reshape((-1, 2))
    if len(aproximado) < 2:
        return caminho[:]
    return [(int(round(float(y))), int(round(float(x)))) for x, y in aproximado]


def _mask_linha_caminho(
    shape: Tuple[int, int],
    caminho: List[Tuple[int, int]],
    espessura: int = 1,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if not caminho:
        return mask
    if len(caminho) == 1:
        y, x = caminho[0]
        mask[int(y), int(x)] = 255
        return mask

    for (y0, x0), (y1, x1) in zip(caminho, caminho[1:]):
        cv2.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, int(espessura), lineType=cv2.LINE_AA)
    return _normalizar_mask(mask)


def _criar_corredor_caule(
    shape: Tuple[int, int],
    caminho_guia: List[Tuple[int, int]],
    largura_bbox: int,
) -> Tuple[np.ndarray, np.ndarray]:
    mask_guia = _mask_linha_caminho(shape, caminho_guia, espessura=3)
    raio = int(max(CORREDOR_CAULE_MIN_PX, min(CORREDOR_CAULE_MAX_PX, round(largura_bbox * CORREDOR_CAULE_FRAC))))
    kernel = _kernel((2 * raio + 1, 2 * raio + 1))
    corredor = cv2.dilate(mask_guia, kernel, iterations=1)
    return mask_guia, corredor


def _agrupar_xs(xs: np.ndarray, gap: int) -> List[np.ndarray]:
    if xs.size == 0:
        return []
    xs_ord = np.sort(xs.astype(np.int32))
    grupos: List[List[int]] = [[int(xs_ord[0])]]
    for x in xs_ord[1:]:
        if int(x) - grupos[-1][-1] > gap:
            grupos.append([])
        grupos[-1].append(int(x))
    return [np.array(g, dtype=np.int32) for g in grupos if g]


def _candidatos_eixo_por_bandas(
    planta: np.ndarray,
    skeleton_busca: np.ndarray,
    mapa_espessura: np.ndarray,
    x_base: int,
    y_base: int,
    largura_bbox: int,
    altura_bbox: int,
    mask_caule_seed: np.ndarray | None,
) -> List[List[Dict[str, float]]]:
    h, w = planta.shape
    y_topo = int(np.min(np.where(planta > 0)[0]))
    passo = max(6, int(EIXO_BANDA_PX))
    gap_cluster = max(EIXO_CLUSTER_GAP_PX, int(round(0.025 * float(largura_bbox))))
    largura_max = max(24, int(round(EIXO_LARGURA_REL_MAX * float(largura_bbox))))
    seed = cv2.bitwise_and(_normalizar_mask(mask_caule_seed), planta) if mask_caule_seed is not None else None

    bandas: List[List[Dict[str, float]]] = []
    for y_centro in range(int(y_base), y_topo - 1, -passo):
        y0 = max(0, int(y_centro - passo // 2))
        y1 = min(h, int(y_centro + passo // 2 + 1))
        candidatos: List[Dict[str, float]] = []

        ys_skel, xs_skel = np.where(skeleton_busca[y0:y1, :] > 0)
        for grupo in _agrupar_xs(xs_skel, gap_cluster):
            x_med = int(round(float(np.median(grupo))))
            y_med = int(round(float(y0 + np.median(ys_skel[np.isin(xs_skel, grupo)])))) if ys_skel.size else int(y_centro)
            largura = int(grupo.max() - grupo.min() + 1)
            if largura > largura_max:
                continue
            candidatos.append(
                {
                    "x": float(x_med),
                    "y": float(y_med),
                    "largura": float(largura),
                    "fonte": 0.0,
                    "espessura": float(mapa_espessura[min(h - 1, max(0, y_med)), min(w - 1, max(0, x_med))]),
                }
            )

        if seed is not None:
            ys_seed, xs_seed = np.where(seed[y0:y1, :] > 0)
            if xs_seed.size > 0:
                candidatos.append(
                    {
                        "x": float(np.median(xs_seed)),
                        "y": float(y0 + np.median(ys_seed)),
                        "largura": 3.0,
                        "fonte": -80.0,
                        "espessura": 2.0,
                    }
                )

        if not candidatos:
            strip = _normalizar_mask(planta[y0:y1, :])
            n_labels, labels, stats, _cent = cv2.connectedComponentsWithStats(strip, connectivity=8)
            for label in range(1, n_labels):
                area = int(stats[label, cv2.CC_STAT_AREA])
                largura = int(stats[label, cv2.CC_STAT_WIDTH])
                if area < 2 or largura > largura_max:
                    continue
                ys_lbl, xs_lbl = np.where(labels == label)
                if xs_lbl.size == 0:
                    continue
                x_med = int(round(float(np.median(xs_lbl))))
                y_med = int(round(float(y0 + np.median(ys_lbl))))
                candidatos.append(
                    {
                        "x": float(x_med),
                        "y": float(y_med),
                        "largura": float(largura),
                        "fonte": 20.0,
                        "espessura": float(mapa_espessura[min(h - 1, max(0, y_med)), min(w - 1, max(0, x_med))]),
                    }
                )

        # Remove duplicatas proximas preservando o candidato de menor custo local.
        unicos: List[Dict[str, float]] = []
        for cand in sorted(candidatos, key=lambda c: (c["fonte"], c["largura"])):
            if any(abs(cand["x"] - outro["x"]) <= 6 for outro in unicos):
                continue
            unicos.append(cand)
        if unicos:
            bandas.append(unicos[:12])

    return bandas


def _comprimento_pontos_float(pontos: List[Tuple[float, float]]) -> float:
    total = 0.0
    for (y0, x0), (y1, x1) in zip(pontos, pontos[1:]):
        total += math.hypot(float(x1) - float(x0), float(y1) - float(y0))
    return float(total)


def _ajustar_eixo_suave(pontos: List[Tuple[float, float]], altura_bbox: int) -> List[Tuple[int, int]]:
    if len(pontos) <= 2:
        return [(int(round(y)), int(round(x))) for y, x in pontos]

    ys = np.array([p[0] for p in pontos], dtype=np.float64)
    xs = np.array([p[1] for p in pontos], dtype=np.float64)
    grau = 2 if len(pontos) < 7 else 3
    try:
        coef = np.polyfit(ys, xs, grau)
        poly = np.poly1d(coef)
        x_fit = poly(ys)
    except np.linalg.LinAlgError:
        return _suavizar_caminho([(int(round(y)), int(round(x))) for y, x in pontos], altura_bbox)

    max_desvio = max(18.0, 0.045 * float(altura_bbox))
    x_suave = np.clip(x_fit, xs - max_desvio, xs + max_desvio)
    caminho = [(int(round(float(y))), int(round(float(x)))) for y, x in zip(ys, x_suave)]
    return _suavizar_caminho(caminho, altura_bbox)


def _medir_eixo_suave_por_bandas(
    planta: np.ndarray,
    skeleton_busca: np.ndarray,
    mapa_espessura: np.ndarray,
    x_base: int,
    y_base: int,
    largura_bbox: int,
    altura_bbox: int,
    mask_caule_seed: np.ndarray | None,
) -> Tuple[float, List[Tuple[int, int]], Dict[str, object]]:
    bandas = _candidatos_eixo_por_bandas(
        planta,
        skeleton_busca,
        mapa_espessura,
        x_base,
        y_base,
        largura_bbox,
        altura_bbox,
        mask_caule_seed,
    )
    if len(bandas) < EIXO_MIN_PONTOS:
        return 0.0, [], {"eixo_suave_usado": False, "motivo_eixo_suave": "poucos_candidatos"}

    beam: List[Dict[str, object]] = [
        {
            "custo": 0.0,
            "pontos": [(float(y_base), float(x_base))],
            "ultimo": {"x": float(x_base), "y": float(y_base), "largura": 3.0, "fonte": -100.0, "espessura": 2.0},
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
                espessura = max(0.0, float(cand["espessura"]) - EIXO_DIST_FOLHA_PX)
                custo = (
                    float(estado["custo"])
                    + 0.18 * abs(dx)
                    + 65.0 * curvatura
                    + 0.10 * float(cand["largura"])
                    + 18.0 * espessura
                    + float(cand["fonte"])
                )
                pontos = list(estado["pontos"]) + [(float(cand["y"]), float(cand["x"]))]
                novos.append({"custo": custo, "pontos": pontos, "ultimo": cand, "slope": slope})

        if not novos:
            continue
        novos.sort(key=lambda e: float(e["custo"]))
        beam = novos[:EIXO_BEAM_WIDTH]
        melhores.extend(beam[: min(8, len(beam))])

    if not melhores:
        return 0.0, [], {"eixo_suave_usado": False, "motivo_eixo_suave": "sem_caminho_dp"}

    def score_estado(estado: Dict[str, object]) -> float:
        pontos = estado["pontos"]
        subida = float(pontos[0][0] - pontos[-1][0])
        compr = _comprimento_pontos_float(pontos)
        compr_util, compr_excessivo, progresso_vertical = _comprimento_util_para_score(subida, compr)
        return (
            2.2 * subida
            + 0.15 * compr_util
            + 55.0 * progresso_vertical
            - 0.35 * compr_excessivo
            - 0.55 * float(estado["custo"])
        )

    elegiveis = [
        estado
        for estado in melhores
        if float(estado["pontos"][0][0] - estado["pontos"][-1][0]) >= 0.55 * float(altura_bbox)
    ]
    if not elegiveis:
        elegiveis = melhores

    def score_estado_final(estado: Dict[str, object]) -> float:
        pontos = estado["pontos"]
        subida = float(pontos[0][0] - pontos[-1][0])
        compr = _comprimento_pontos_float(pontos)
        compr_util, compr_excessivo, progresso_vertical = _comprimento_util_para_score(subida, compr)
        return (
            3.5 * subida
            + 0.25 * compr_util
            + 70.0 * progresso_vertical
            - 0.45 * compr_excessivo
            - 0.16 * float(estado["custo"])
        )

    escolhido = max(elegiveis, key=score_estado_final)
    pontos_escolhidos = escolhido["pontos"]
    caminho = _ajustar_eixo_suave(pontos_escolhidos, altura_bbox)
    comprimento = _comprimento_real_caminho(caminho)
    mask_candidatos = np.zeros_like(planta)
    for banda in bandas:
        for cand in banda:
            cv2.circle(mask_candidatos, (int(round(cand["x"])), int(round(cand["y"]))), 2, 255, -1)
    mask_eixo_dp = _mask_linha_caminho(
        planta.shape,
        [(int(round(y)), int(round(x))) for y, x in pontos_escolhidos],
        espessura=1,
    )
    return float(comprimento), caminho, {
        "eixo_suave_usado": True,
        "motivo_eixo_suave": "ok",
        "mask_candidatos_eixo": mask_candidatos,
        "mask_eixo_dp": mask_eixo_dp,
        "custo_eixo_suave": float(escolhido["custo"]),
        "numero_bandas_eixo": int(len(bandas)),
    }


def _dijkstra_com_corredor_suave(
    skeleton: np.ndarray,
    inicio: Tuple[int, int],
    fim_bruto: Tuple[int, int],
    caminho_bruto: List[Tuple[int, int]],
    mapa_espessura: np.ndarray,
    x_base: int,
    y_base: int,
    largura_bbox: int,
    altura_bbox: int,
) -> Tuple[List[Tuple[int, int]], float, Dict[str, np.ndarray | bool | str | float]]:
    caminho_guia = _suavizar_caminho(caminho_bruto, altura_bbox)
    mask_guia, corredor = _criar_corredor_caule(skeleton.shape, caminho_guia, largura_bbox)
    skel_corredor = cv2.bitwise_and(_normalizar_mask(skeleton), corredor)

    if cv2.countNonZero(skel_corredor) == 0:
        return caminho_guia, _comprimento_real_caminho(caminho_guia), {
            "mask_guia_suavizada": mask_guia,
            "mask_corredor_caule": corredor,
            "mask_skeleton_corredor": skel_corredor,
            "corredor_usado": False,
            "motivo_corredor": "skeleton_corredor_vazio",
        }

    try:
        inicio_corredor = _ponto_skeleton_mais_proximo(skel_corredor, int(inicio[1]), int(inicio[0]))
    except ValueError:
        inicio_corredor = inicio

    mapa_dist_guia = cv2.distanceTransform(255 - mask_guia, cv2.DIST_L2, 5)
    dist2, custo_folha2, parent_y2, parent_x2, visitado2 = _dijkstra_skeleton(
        skel_corredor,
        inicio_corredor,
        mapa_espessura=mapa_espessura,
        mapa_dist_guia=mapa_dist_guia,
    )
    if not np.any(visitado2):
        return caminho_guia, _comprimento_real_caminho(caminho_guia), {
            "mask_guia_suavizada": mask_guia,
            "mask_corredor_caule": corredor,
            "mask_skeleton_corredor": skel_corredor,
            "corredor_usado": False,
            "motivo_corredor": "sem_visita_corredor",
        }

    y_fim2, x_fim2, _score2, fallback2, motivo2 = _escolher_ponto_final(
        skel_corredor,
        visitado2,
        dist2,
        custo_folha2,
        mapa_espessura,
        x_base=x_base,
        y_base=y_base,
        largura_bbox=largura_bbox,
        altura_bbox=altura_bbox,
    )
    caminho2 = _reconstruir_caminho(inicio_corredor, (int(y_fim2), int(x_fim2)), parent_y2, parent_x2)
    if len(caminho2) < 2:
        caminho2 = _reconstruir_caminho(inicio_corredor, fim_bruto, parent_y2, parent_x2)
    if len(caminho2) < 2:
        caminho2 = caminho_bruto

    caminho_final = _suavizar_caminho(caminho2, altura_bbox)
    return caminho_final, _comprimento_real_caminho(caminho_final), {
        "mask_guia_suavizada": mask_guia,
        "mask_corredor_caule": corredor,
        "mask_skeleton_corredor": skel_corredor,
        "corredor_usado": True,
        "motivo_corredor": "ok_fallback_endpoint" if fallback2 else str(motivo2),
    }


def _caminho_ate_primeira_ramificacao(
    skeleton: np.ndarray,
    inicio: Tuple[int, int],
    comprimento_min: float = 0.0,
    extensao_max: float = 0.0,
) -> List[Tuple[int, int]]:
    skel = _normalizar_mask(skeleton)
    h, w = skel.shape
    caminho = [inicio]
    anterior: Tuple[int, int] | None = None
    atual = inicio

    while True:
        y, x = atual
        vizinhos = [
            (yn, xn)
            for yn, xn, _custo in _vizinhos8(y, x, h, w)
            if skel[yn, xn] > 0 and (anterior is None or (yn, xn) != anterior)
        ]

        grau = _grau_pixel(skel, y, x)
        comprimento_atual = _comprimento_real_caminho(caminho)
        if anterior is not None and len(caminho) >= MIN_PIXELS_ANTES_RAMIFICACAO and grau >= 3:
            if comprimento_atual >= float(comprimento_min):
                break
            if float(extensao_max) <= 0.0 or comprimento_atual >= float(comprimento_min) + float(extensao_max):
                break
        if not vizinhos:
            break

        # Em mudas muito pequenas, o caule confiavel costuma terminar na
        # primeira bifurcacao; antes dela seguimos o vizinho que mais sobe.
        proximo = min(vizinhos, key=lambda p: (p[0], abs(p[1] - x)))
        anterior = atual
        atual = proximo
        caminho.append(atual)

        if len(caminho) > skel.size:
            break
        if extensao_max > 0.0 and _comprimento_real_caminho(caminho) >= max(comprimento_min, extensao_max):
            if _grau_pixel(skel, atual[0], atual[1]) >= 3:
                break

    return caminho


def medir_comprimento_caule_skeleton(
    mask_planta_acima: np.ndarray,
    mask_caule_seed: np.ndarray | None = None,
    skeleton: np.ndarray | None = None,
) -> tuple[float, np.ndarray, dict]:
    planta = _normalizar_mask(mask_planta_acima)
    if cv2.countNonZero(planta) == 0:
        vazio = np.zeros_like(planta)
        debug = {
            "x_base": None,
            "y_base": None,
            "altura_bbox": 0,
            "largura_bbox": 0,
            "comprimento_caule_px": 0.0,
            "score_melhor_caminho": 0.0,
            "mask_skeleton": vazio,
            "mask_caminho_caule": vazio,
            "mask_caminho_caule_dilatado": vazio,
            "ponto_inicio": None,
            "ponto_final": None,
            "numero_pixels_caminho": 0,
            "metodo": "caminho_principal_skeleton_score",
            "fallback": True,
            "motivo_fallback": "mascara_planta_vazia",
        }
        return 0.0, vazio, debug

    x_min, y_min, x_max, y_max = _bbox_mask(planta)
    largura_bbox = int(x_max - x_min + 1)
    altura_bbox = int(y_max - y_min + 1)
    x_base, y_base, metodo_base = _estimar_base(planta, (x_min, y_min, x_max, y_max), mask_caule_seed)

    skel, skel_original, skeleton_reparado_usado = _preparar_skeleton_busca(
        planta,
        skeleton=skeleton,
        largura_bbox=largura_bbox,
        altura_bbox=altura_bbox,
    )
    skel_podado = _podar_ramos_curtos(skel, PODA_RAMO_CURTO_PX)
    if cv2.countNonZero(skel_podado) == 0:
        skel_podado = skel

    try:
        inicio = _ponto_skeleton_mais_proximo(skel_podado, x_base, y_base)
    except ValueError:
        vazio = np.zeros_like(planta)
        debug = {
            "x_base": x_base,
            "y_base": y_base,
            "altura_bbox": altura_bbox,
            "largura_bbox": largura_bbox,
            "comprimento_caule_px": 0.0,
            "score_melhor_caminho": 0.0,
            "mask_skeleton": skel_podado,
            "mask_caminho_caule": vazio,
            "mask_caminho_caule_dilatado": vazio,
            "ponto_inicio": None,
            "ponto_final": None,
            "numero_pixels_caminho": 0,
            "metodo": "caminho_principal_skeleton_score",
            "fallback": True,
            "motivo_fallback": "skeleton_vazio",
            "metodo_base": metodo_base,
            "skeleton_reparado_usado": bool(skeleton_reparado_usado),
        }
        return 0.0, vazio, debug

    mapa_espessura = cv2.distanceTransform(planta, cv2.DIST_L2, 5)
    comprimento_eixo, caminho_eixo, debug_eixo = _medir_eixo_suave_por_bandas(
        planta,
        skel_podado,
        mapa_espessura,
        x_base=x_base,
        y_base=y_base,
        largura_bbox=largura_bbox,
        altura_bbox=altura_bbox,
        mask_caule_seed=mask_caule_seed,
    )
    if (
        altura_bbox > ALTURA_MAX_MUDA_PEQUENA_PX
        and debug_eixo.get("eixo_suave_usado")
        and len(caminho_eixo) >= 2
        and comprimento_eixo > 0.0
    ):
        caminho_eixo, debug_corte = _cortar_terminal_folha(
            caminho_eixo,
            planta,
            mapa_espessura,
            largura_bbox,
        )
        ajuste_eixo_largo_usado = False
        razao_largura_altura = float(largura_bbox) / max(1.0, float(altura_bbox))
        if not debug_corte.get("corte_terminal_usado") and razao_largura_altura >= AJUSTE_EIXO_LARGO_RAZAO_MIN:
            comprimento_alvo = max(1.0, _comprimento_real_caminho(caminho_eixo) - AJUSTE_EIXO_LARGO_FRAC_ALTURA * float(altura_bbox))
            caminho_eixo = _truncar_caminho_por_comprimento(caminho_eixo, comprimento_alvo)
            ajuste_eixo_largo_usado = True
        comprimento_eixo = _comprimento_real_caminho(caminho_eixo)
        mask_caminho = _mask_linha_caminho(planta.shape, caminho_eixo, espessura=1)
        mask_caminho_dilatado = cv2.dilate(mask_caminho, _kernel((5, 5)), iterations=1)
        fim = caminho_eixo[-1]
        debug = {
            "x_base": int(x_base),
            "y_base": int(y_base),
            "altura_bbox": int(altura_bbox),
            "largura_bbox": int(largura_bbox),
            "comprimento_caule_px": float(comprimento_eixo),
            "score_melhor_caminho": float(debug_eixo.get("custo_eixo_suave", 0.0)),
            "mask_skeleton": skel_podado,
            "mask_skeleton_original": skel_original,
            "mask_skeleton_busca_sem_poda": skel,
            "mapa_espessura": mapa_espessura,
            "mask_caminho_caule_bruto": debug_eixo.get("mask_eixo_dp", mask_caminho),
            "mask_caminho_caule": mask_caminho,
            "mask_caminho_caule_dilatado": mask_caminho_dilatado,
            "ponto_inicio": (int(inicio[1]), int(inicio[0])),
            "ponto_final": (int(fim[1]), int(fim[0])),
            "numero_pixels_caminho": int(len(caminho_eixo)),
            "metodo": "eixo_suave_por_bandas",
            "fallback": False,
            "motivo_fallback": "",
            "metodo_base": metodo_base,
            "skeleton_reparado_usado": bool(skeleton_reparado_usado),
            "constantes": {
                "eixo_banda_px": EIXO_BANDA_PX,
                "eixo_beam_width": EIXO_BEAM_WIDTH,
                "eixo_largura_rel_max": EIXO_LARGURA_REL_MAX,
                "eixo_dist_folha_px": EIXO_DIST_FOLHA_PX,
                "progresso_vertical_alvo": PROGRESSO_VERTICAL_ALVO,
                "peso_comprimento_excessivo_score": PESO_COMPRIMENTO_EXCESSIVO_SCORE,
                "corte_terminal_largura_frac": CORTE_TERMINAL_LARGURA_FRAC,
                "corte_terminal_frac_segmento": CORTE_TERMINAL_FRAC_SEGMENTO,
                "ajuste_eixo_largo_frac_altura": AJUSTE_EIXO_LARGO_FRAC_ALTURA,
                "ajuste_eixo_largo_razao_min": AJUSTE_EIXO_LARGO_RAZAO_MIN,
            },
        }
        debug.update(debug_eixo)
        debug.update(debug_corte)
        debug["ajuste_eixo_largo_usado"] = bool(ajuste_eixo_largo_usado)
        debug["razao_largura_altura"] = float(razao_largura_altura)
        return float(comprimento_eixo), mask_caminho, debug

    if altura_bbox <= ALTURA_MAX_MUDA_PEQUENA_PX:
        caminho_bruto = _caminho_ate_primeira_ramificacao(
            skel_podado,
            inicio,
            comprimento_min=MUDA_PEQUENA_COMPR_MIN_PX,
            extensao_max=MUDA_PEQUENA_EXTENSAO_MAX_PX,
        )
        caminho = caminho_bruto
        comprimento = _comprimento_real_caminho(caminho)

        mask_caminho_bruto = _mask_linha_caminho(planta.shape, caminho_bruto, espessura=1)
        mask_caminho = _mask_linha_caminho(planta.shape, caminho, espessura=1)
        mask_caminho_dilatado = cv2.dilate(mask_caminho, _kernel((5, 5)), iterations=1)

        fim = caminho[-1] if caminho else inicio
        debug = {
            "x_base": int(x_base),
            "y_base": int(y_base),
            "altura_bbox": int(altura_bbox),
            "largura_bbox": int(largura_bbox),
            "comprimento_caule_px": float(comprimento),
            "score_melhor_caminho": float(comprimento),
            "mask_skeleton": skel_podado,
            "mask_skeleton_original": skel_original,
            "mask_skeleton_busca_sem_poda": skel,
            "mask_caminho_caule_bruto": mask_caminho_bruto,
            "mask_caminho_caule": mask_caminho,
            "mask_caminho_caule_dilatado": mask_caminho_dilatado,
            "ponto_inicio": (int(inicio[1]), int(inicio[0])),
            "ponto_final": (int(fim[1]), int(fim[0])),
            "numero_pixels_caminho": int(len(caminho)),
            "metodo": "caule_pequeno_ate_primeira_ramificacao",
            "fallback": False,
            "motivo_fallback": "",
            "metodo_base": metodo_base,
            "skeleton_reparado_usado": bool(skeleton_reparado_usado),
            "constantes": {
                "altura_max_muda_pequena_px": ALTURA_MAX_MUDA_PEQUENA_PX,
                "min_pixels_antes_ramificacao": MIN_PIXELS_ANTES_RAMIFICACAO,
            },
        }
        return float(comprimento), mask_caminho, debug

    dist, custo_folha, parent_y, parent_x, visitado = _dijkstra_skeleton(
        skel_podado,
        inicio,
        mapa_espessura=mapa_espessura,
    )
    y_fim, x_fim, score, fallback, motivo_fallback = _escolher_ponto_final(
        skel_podado,
        visitado,
        dist,
        custo_folha,
        mapa_espessura,
        x_base=x_base,
        y_base=y_base,
        largura_bbox=largura_bbox,
        altura_bbox=altura_bbox,
    )

    fim = (int(y_fim), int(x_fim))
    caminho_bruto = _reconstruir_caminho(inicio, fim, parent_y, parent_x)
    comprimento_bruto = _comprimento_real_caminho(caminho_bruto)
    caminho, comprimento, debug_corredor = _dijkstra_com_corredor_suave(
        skel_podado,
        inicio,
        fim,
        caminho_bruto,
        mapa_espessura,
        x_base=x_base,
        y_base=y_base,
        largura_bbox=largura_bbox,
        altura_bbox=altura_bbox,
    )

    mask_caminho_bruto = _mask_linha_caminho(planta.shape, caminho_bruto, espessura=1)
    mask_caminho = _mask_linha_caminho(planta.shape, caminho, espessura=1)
    mask_caminho_dilatado = cv2.dilate(mask_caminho, _kernel((5, 5)), iterations=1)

    fim_final = caminho[-1] if caminho else fim
    debug = {
        "x_base": int(x_base),
        "y_base": int(y_base),
        "altura_bbox": int(altura_bbox),
        "largura_bbox": int(largura_bbox),
        "comprimento_caule_px": float(comprimento),
        "comprimento_caule_bruto_px": float(comprimento_bruto),
        "score_melhor_caminho": float(score),
        "mask_skeleton": skel_podado,
        "mask_skeleton_original": skel_original,
        "mask_skeleton_busca_sem_poda": skel,
        "mapa_espessura": mapa_espessura,
        "mask_caminho_caule_bruto": mask_caminho_bruto,
        "mask_caminho_caule": mask_caminho,
        "mask_caminho_caule_dilatado": mask_caminho_dilatado,
        "ponto_inicio": (int(inicio[1]), int(inicio[0])),
        "ponto_final": (int(fim_final[1]), int(fim_final[0])),
        "numero_pixels_caminho": int(len(caminho)),
        "metodo": "caminho_principal_corredor_suave",
        "fallback": bool(fallback),
        "motivo_fallback": motivo_fallback if fallback else "",
        "metodo_base": metodo_base,
        "skeleton_reparado_usado": bool(skeleton_reparado_usado),
        "constantes": {
            "poda_ramo_curto_px": PODA_RAMO_CURTO_PX,
            "ganho_vertical_min_frac": GANHO_VERTICAL_MIN_FRAC,
            "comprimento_min_frac": COMPRIMENTO_MIN_FRAC,
            "desvio_lateral_limite_frac": DESVIO_LATERAL_LIMITE_FRAC,
            "desvio_lateral_limite_abs": DESVIO_LATERAL_LIMITE_ABS,
            "progresso_vertical_min": PROGRESSO_VERTICAL_MIN,
            "skeleton_reparo_vertical_frac": SKELETON_REPARO_VERTICAL_FRAC,
            "skeleton_reparo_horizontal_frac": SKELETON_REPARO_HORIZONTAL_FRAC,
            "raio_folha_excesso_px": RAIO_FOLHA_EXCESSO_PX,
            "penalidade_caminho_folha": PENALIDADE_CAMINHO_FOLHA,
            "penalidade_endpoint_folha": PENALIDADE_ENDPOINT_FOLHA,
            "peso_subida_score": PESO_SUBIDA_SCORE,
            "peso_comprimento_score": PESO_COMPRIMENTO_SCORE,
            "peso_desvio_lateral_score": PESO_DESVIO_LATERAL_SCORE,
            "peso_desvio_lateral_excessivo": PESO_DESVIO_LATERAL_EXCESSIVO,
            "progresso_vertical_alvo": PROGRESSO_VERTICAL_ALVO,
            "peso_progresso_vertical_score": PESO_PROGRESSO_VERTICAL_SCORE,
            "peso_comprimento_excessivo_score": PESO_COMPRIMENTO_EXCESSIVO_SCORE,
            "suavizacao_caminho_frac": SUAVIZACAO_CAMINHO_FRAC,
            "corredor_caule_frac": CORREDOR_CAULE_FRAC,
        },
    }
    debug.update(debug_corredor)
    return float(comprimento), mask_caminho, debug


def _listar_imagens(img_dir: Path, limite: int | None = None) -> List[Path]:
    def natural_key(path: Path):
        partes: List[object] = []
        atual = ""
        modo_num = False
        for ch in path.stem.lower():
            eh_num = ch.isdigit()
            if atual and eh_num != modo_num:
                partes.append(int(atual) if modo_num else atual)
                atual = ch
            else:
                atual += ch
            modo_num = eh_num
        if atual:
            partes.append(int(atual) if modo_num else atual)
        return partes

    imagens = sorted(img_dir.glob("*.jpg"), key=natural_key)
    if limite is not None:
        return imagens[:limite]
    return imagens


def _overlay_resultado(img_bgr: np.ndarray, debug: dict) -> np.ndarray:
    out = img_bgr.copy()
    skel = debug["mask_skeleton"]
    caminho = debug["mask_caminho_caule_dilatado"]
    caminho_bruto = debug.get("mask_caminho_caule_bruto")
    corredor = debug.get("mask_corredor_caule")

    out[skel > 0] = (180, 180, 180)
    if corredor is not None:
        verde = np.zeros_like(out)
        verde[:, :] = (0, 120, 0)
        blend_corredor = cv2.addWeighted(out, 1.0, verde, 0.18, 0)
        out[corredor > 0] = blend_corredor[corredor > 0]
    if caminho_bruto is not None:
        laranja = np.zeros_like(out)
        laranja[:, :] = (0, 160, 255)
        blend_bruto = cv2.addWeighted(out, 1.0, laranja, 0.55, 0)
        bruto_dilatado = cv2.dilate(_normalizar_mask(caminho_bruto), _kernel((3, 3)), iterations=1)
        out[bruto_dilatado > 0] = blend_bruto[bruto_dilatado > 0]
    vermelho = np.zeros_like(out)
    vermelho[:, :] = (0, 0, 255)
    blend = cv2.addWeighted(out, 1.0, vermelho, 0.65, 0)
    out[caminho > 0] = blend[caminho > 0]

    if debug.get("ponto_inicio") is not None:
        cv2.circle(out, tuple(debug["ponto_inicio"]), 8, (0, 255, 255), -1)
    if debug.get("ponto_final") is not None:
        cv2.circle(out, tuple(debug["ponto_final"]), 8, (255, 255, 0), -1)

    cv2.putText(
        out,
        f"Compr caule={debug['comprimento_caule_px']:.1f}px score={debug['score_melhor_caminho']:.1f}",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        3,
    )
    cv2.putText(
        out,
        f"inicio={debug['ponto_inicio']} fim={debug['ponto_final']} fallback={debug['fallback']}",
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
    )
    return out


def _salvar_debug(out_dir: Path, nome: str, img_bgr: np.ndarray, debug: dict) -> str:
    img_dir = out_dir / nome
    img_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_dir / "mask_skeleton.png"), debug["mask_skeleton"])
    if "mask_skeleton_original" in debug:
        cv2.imwrite(str(img_dir / "mask_skeleton_original.png"), debug["mask_skeleton_original"])
    if "mask_skeleton_busca_sem_poda" in debug:
        cv2.imwrite(str(img_dir / "mask_skeleton_busca_sem_poda.png"), debug["mask_skeleton_busca_sem_poda"])
    if "mapa_espessura" in debug:
        mapa_norm = cv2.normalize(debug["mapa_espessura"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(str(img_dir / "mapa_espessura.png"), mapa_norm)
    cv2.imwrite(str(img_dir / "mask_caminho_caule.png"), debug["mask_caminho_caule"])
    cv2.imwrite(str(img_dir / "mask_caminho_caule_dilatado.png"), debug["mask_caminho_caule_dilatado"])
    if "mask_caminho_caule_bruto" in debug:
        cv2.imwrite(str(img_dir / "mask_caminho_caule_bruto.png"), debug["mask_caminho_caule_bruto"])
    if "mask_guia_suavizada" in debug:
        cv2.imwrite(str(img_dir / "mask_guia_suavizada.png"), debug["mask_guia_suavizada"])
    if "mask_corredor_caule" in debug:
        cv2.imwrite(str(img_dir / "mask_corredor_caule.png"), debug["mask_corredor_caule"])
    if "mask_skeleton_corredor" in debug:
        cv2.imwrite(str(img_dir / "mask_skeleton_corredor.png"), debug["mask_skeleton_corredor"])
    if "mask_candidatos_eixo" in debug:
        cv2.imwrite(str(img_dir / "mask_candidatos_eixo.png"), debug["mask_candidatos_eixo"])
    if "mask_eixo_dp" in debug:
        cv2.imwrite(str(img_dir / "mask_eixo_dp.png"), debug["mask_eixo_dp"])
    overlay = _overlay_resultado(img_bgr, debug)
    overlay_path = img_dir / "overlay_caule_skeleton_score.png"
    cv2.imwrite(str(overlay_path), overlay)
    return str(overlay_path)


def salvar_debug_comprimento(out_dir: Path, nome: str, img_bgr: np.ndarray, debug: dict) -> str:
    return _salvar_debug(out_dir, nome, img_bgr, debug)


def _processar_imagem(path_img: Path, out_dir: Path) -> Dict[str, object]:
    dados = M01.processar_tratamentos_iniciais(str(path_img))
    comprimento, _mask, debug = medir_comprimento_caule_skeleton(
        dados["mask_planta_acima"],
        mask_caule_seed=dados.get("mask_caule"),
        skeleton=dados.get("skeleton"),
    )
    overlay_path = _salvar_debug(out_dir, path_img.stem, dados["img_bgr"], debug)
    area_planta = int(np.count_nonzero(_normalizar_mask(dados["mask_planta_acima"])))

    return {
        "imagem": path_img.stem,
        "comprimento_caule_px": round(float(comprimento), 3),
        "area_planta": area_planta,
        "ponto_inicio": debug["ponto_inicio"],
        "ponto_final": debug["ponto_final"],
        "score": round(float(debug["score_melhor_caminho"]), 3),
        "fallback": bool(debug["fallback"]),
        "motivo_fallback": debug["motivo_fallback"],
        "skeleton_reparado_usado": bool(debug.get("skeleton_reparado_usado", False)),
        "overlay": overlay_path,
        "avaliacao_visual": "requer_inspecao_manual",
    }


def _salvar_csv(path_csv: Path, linhas: List[Dict[str, object]]) -> None:
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    campos = [
        "imagem",
        "comprimento_caule_px",
        "area_planta",
        "ponto_inicio",
        "ponto_final",
        "score",
        "fallback",
        "motivo_fallback",
        "skeleton_reparado_usado",
        "avaliacao_visual",
        "overlay",
    ]
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for row in linhas:
            writer.writerow(row)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mede o comprimento plausivel do caule por caminho ascendente no skeleton."
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default="Dataset_Projeto1/_Eucalipto_Escolhidos1",
        help="Pasta com imagens .jpg.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="debug_saida/compr_avancado",
        help="Pasta para mascaras e overlays de debug.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="resultados/resultado_compr_avancado.csv",
        help="CSV de resumo.",
    )
    parser.add_argument("--limite", type=int, default=None, help="Limite opcional de imagens.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    linhas = []

    for path_img in _listar_imagens(img_dir, args.limite):
        row = _processar_imagem(path_img, out_dir)
        linhas.append(row)
        print(
            f"{row['imagem']}: comprimento={row['comprimento_caule_px']} "
            f"area={row['area_planta']} inicio={row['ponto_inicio']} "
            f"fim={row['ponto_final']} score={row['score']} fallback={row['fallback']}"
        )

    _salvar_csv(Path(args.csv), linhas)
    print(f"CSV salvo em: {args.csv}")
    print(f"Debug salvo em: {args.out_dir}")


if __name__ == "__main__":
    main()
