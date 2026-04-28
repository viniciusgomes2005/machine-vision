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
SKELETON_REPARO_VERTICAL_FRAC = 0.20
SKELETON_REPARO_HORIZONTAL_FRAC = 0.04
RAIO_FOLHA_EXCESSO_PX = 6.0
PENALIDADE_CAMINHO_FOLHA = 0.85
PENALIDADE_ENDPOINT_FOLHA = 18.0
PESO_SUBIDA_SCORE = 1.20
PESO_COMPRIMENTO_SCORE = 0.45
PESO_DESVIO_LATERAL_SCORE = 0.35
PESO_DESVIO_LATERAL_EXCESSIVO = 1.5


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
            if nd < dist[yn, xn]:
                dist[yn, xn] = nd
                custo_folha[yn, xn] = custo_folha[y, x] + excesso_folha * custo
                parent_y[yn, xn] = y
                parent_x[yn, xn] = x
                heapq.heappush(heap, (nd, yn, xn))

    return dist, custo_folha, parent_y, parent_x, visitado


def _escolher_ponto_final(
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

    for y, x in zip(ys, xs):
        comprimento = float(dist[y, x])
        if not np.isfinite(comprimento) or comprimento <= 0.0:
            continue

        subida = float(y_base - int(y))
        desvio_lateral = abs(float(int(x) - x_base))
        progresso_vertical = subida / max(1.0, comprimento)

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
            + PESO_COMPRIMENTO_SCORE * comprimento
            - PESO_DESVIO_LATERAL_SCORE * desvio_lateral
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
        progresso_vertical = subida / max(1.0, comprimento)
        score = 1.0 * subida + 0.20 * comprimento - 1.20 * desvio_lateral
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
    dist, custo_folha, parent_y, parent_x, visitado = _dijkstra_skeleton(
        skel_podado,
        inicio,
        mapa_espessura=mapa_espessura,
    )
    y_fim, x_fim, score, fallback, motivo_fallback = _escolher_ponto_final(
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
    caminho = _reconstruir_caminho(inicio, fim, parent_y, parent_x)
    comprimento = _comprimento_real_caminho(caminho)

    mask_caminho = np.zeros_like(planta)
    for y, x in caminho:
        mask_caminho[int(y), int(x)] = 255
    mask_caminho_dilatado = cv2.dilate(mask_caminho, _kernel((5, 5)), iterations=1)

    debug = {
        "x_base": int(x_base),
        "y_base": int(y_base),
        "altura_bbox": int(altura_bbox),
        "largura_bbox": int(largura_bbox),
        "comprimento_caule_px": float(comprimento),
        "score_melhor_caminho": float(score),
        "mask_skeleton": skel_podado,
        "mask_skeleton_original": skel_original,
        "mask_skeleton_busca_sem_poda": skel,
        "mapa_espessura": mapa_espessura,
        "mask_caminho_caule": mask_caminho,
        "mask_caminho_caule_dilatado": mask_caminho_dilatado,
        "ponto_inicio": (int(inicio[1]), int(inicio[0])),
        "ponto_final": (int(fim[1]), int(fim[0])),
        "numero_pixels_caminho": int(len(caminho)),
        "metodo": "caminho_principal_skeleton_score",
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
        },
    }
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

    out[skel > 0] = (180, 180, 180)
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
    overlay = _overlay_resultado(img_bgr, debug)
    overlay_path = img_dir / "overlay_caule_skeleton_score.png"
    cv2.imwrite(str(overlay_path), overlay)
    return str(overlay_path)


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
