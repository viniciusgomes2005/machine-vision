"""
Pipeline de desenvolvimento: teste rapido dos passos.

Objetivo: iterar sobre segmentacao, area, contagem, diametro e comprimento
ate cada parte ficar razoavel no dataset, antes de virar celula de notebook.
"""
from pathlib import Path
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt


BASE = Path(__file__).parent
PASTA_EUCALIPTO = BASE / "Dataset_Projeto1" / "_Eucalipto_Escolhidos1"
PASTA_PINHEIRO = BASE / "Dataset_Projeto1" / "_Pinheiro_Escolhidos1"
PASTA_DEBUG = BASE / "_dev_debug"
PASTA_DEBUG.mkdir(exist_ok=True)


# -----------------------------------------------------------------
# Leitura e especie
# -----------------------------------------------------------------
def LerImagemRGB(caminho):
    img_bgr = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError("Falhei ao abrir " + str(caminho))
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def Especie(caminho):
    return "pinheiro" if "pinheiro" in caminho.name.lower() else "eucalipto"


# -----------------------------------------------------------------
# SEGMENTACAO
# -----------------------------------------------------------------
def _maior_componente(mask):
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    idx_best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx_best


def _filtra_area_min(mask, area_min):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    saida = np.zeros_like(mask, dtype=bool)
    for idx in range(1, n):
        if stats[idx, cv2.CC_STAT_AREA] >= area_min:
            saida |= labels == idx
    return saida


def SegmentaCena(img_rgb):
    """Segmenta planta, tubete preto e cilindro branco sobre fundo azul."""
    h, w = img_rgb.shape[:2]
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    # Fundo azul
    mask_fundo = (H >= 90) & (H <= 135) & (S >= 60) & (V >= 50) & (B.astype(int) > R.astype(int) + 8)
    mask_frente = ~mask_fundo
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_frente = cv2.morphologyEx(mask_frente.astype(np.uint8) * 255, cv2.MORPH_OPEN, k3)
    mask_frente = cv2.morphologyEx(mask_frente, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0

    # Tubete preto
    mask_preto_raw = (V < 80) & mask_frente
    mask_preto_raw = cv2.morphologyEx(mask_preto_raw.astype(np.uint8) * 255, cv2.MORPH_OPEN, k3) > 0
    mask_preto_raw[: int(0.40 * h)] = False
    mask_tubete = _maior_componente(mask_preto_raw)

    # Cilindro branco
    mask_branco_raw = (V >= 170) & (S < 60) & mask_frente
    mask_branco_raw = cv2.morphologyEx(mask_branco_raw.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    mask_branco_raw[: int(0.45 * h)] = False
    mask_cilindro = _maior_componente(mask_branco_raw)

    # Pontos de referencia
    if np.any(mask_tubete):
        ys, xs = np.where(mask_tubete)
        y_topo_tubete = int(ys.min())
        x_centro_tubete = int(np.median(xs))
    else:
        y_topo_tubete = int(0.62 * h)
        x_centro_tubete = w // 2

    # Planta
    # - verde/amarelo por matiz
    # - ou fortemente vermelho-avermelhado no topo (brotos novos)
    # - ou simplesmente nao-azul com canal verde/vermelho dominante
    matiz_verde = (H >= 20) & (H <= 95)
    matiz_avermelhado = (H >= 0) & (H <= 20) & (S >= 80)  # brotos novos do eucalipto
    canal_vegetacao = (G.astype(int) - B.astype(int) > 0) & (G > 50)

    mask_planta_cor = (matiz_verde | matiz_avermelhado | canal_vegetacao)
    mask_planta = mask_planta_cor & mask_frente & (~mask_tubete) & (~mask_cilindro) & (~mask_fundo)

    # So aceito planta ESTRITAMENTE acima do tubete (com uma margem pequena para capturar coleto)
    corte_y = y_topo_tubete + 8
    mask_planta[corte_y:, :] = False

    # Limpeza
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_planta = cv2.morphologyEx(mask_planta.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k3) > 0
    mask_planta = cv2.morphologyEx(mask_planta.astype(np.uint8) * 255, cv2.MORPH_OPEN, k3) > 0

    # Remove componentes pequenos
    area_min = max(120, int(0.000015 * h * w))
    mask_planta = _filtra_area_min(mask_planta, area_min)

    # Agora pego a componente principal: a maior que toque a faixa do coleto
    if np.any(mask_planta):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_planta.astype(np.uint8), connectivity=8)
        # Ordena por area decrescente
        areas = [(stats[idx, cv2.CC_STAT_AREA], idx) for idx in range(1, n)]
        areas.sort(reverse=True)

        faixa_coleto_y0 = max(0, y_topo_tubete - 80)
        faixa_coleto_y1 = min(h, y_topo_tubete + 10)
        faixa_coleto_x0 = max(0, x_centro_tubete - 150)
        faixa_coleto_x1 = min(w, x_centro_tubete + 150)

        idx_principal = None
        for area, idx in areas:
            comp = labels == idx
            if np.any(comp[faixa_coleto_y0:faixa_coleto_y1, faixa_coleto_x0:faixa_coleto_x1]):
                idx_principal = idx
                break

        if idx_principal is None and areas:
            idx_principal = areas[0][1]

        if idx_principal is not None:
            # Ficamos com a componente principal + qualquer outra proxima verticalmente (folhas destacadas)
            principal = labels == idx_principal
            dilatada = cv2.dilate(principal.astype(np.uint8), np.ones((25, 25), np.uint8), iterations=3) > 0
            mask_planta_final = np.zeros_like(mask_planta)
            for area, idx in areas:
                comp = labels == idx
                # se a componente toca a zona expandida da principal, faz parte
                if np.any(comp & dilatada):
                    mask_planta_final |= comp
            mask_planta = mask_planta_final

    # Diametro do cilindro branco
    diametro_cilindro_px = None
    if np.any(mask_cilindro):
        ys_c, xs_c = np.where(mask_cilindro)
        diametro_cilindro_px = float(xs_c.max() - xs_c.min() + 1)

    return {
        "mask_fundo": mask_fundo,
        "mask_frente": mask_frente,
        "mask_tubete": mask_tubete,
        "mask_cilindro": mask_cilindro,
        "mask_planta": mask_planta,
        "y_topo_tubete": y_topo_tubete,
        "x_centro_tubete": x_centro_tubete,
        "diametro_cilindro_px": diametro_cilindro_px,
    }


# -----------------------------------------------------------------
# DEBUG
# -----------------------------------------------------------------
def Debug(caminho):
    print(f"\n== {caminho.name} ==")
    img = LerImagemRGB(caminho)
    esp = Especie(caminho)
    seg = SegmentaCena(img)

    fig, axs = plt.subplots(1, 4, figsize=(20, 6))
    axs[0].imshow(img); axs[0].set_title("original")
    over = img.copy()
    over[seg["mask_planta"]] = (over[seg["mask_planta"]] * 0.25 + np.array([255, 0, 0]) * 0.75).astype(np.uint8)
    axs[1].imshow(over); axs[1].set_title("planta em vermelho")
    axs[2].imshow(seg["mask_planta"], cmap="gray"); axs[2].set_title("mask planta")
    cenario = img.copy()
    cenario[seg["mask_tubete"]] = (cenario[seg["mask_tubete"]] * 0.4 + np.array([0, 0, 0]) * 0.6).astype(np.uint8)
    cenario[seg["mask_cilindro"]] = (cenario[seg["mask_cilindro"]] * 0.4 + np.array([200, 200, 200]) * 0.6).astype(np.uint8)
    cv2.line(cenario, (0, seg["y_topo_tubete"]), (cenario.shape[1]-1, seg["y_topo_tubete"]), (255, 200, 0), 4)
    axs[3].imshow(cenario); axs[3].set_title(f"tubete+cilindro ({esp})")
    for a in axs: a.axis("off")
    fig.suptitle(caminho.name)
    plt.tight_layout()
    out = PASTA_DEBUG / f"dev_{caminho.stem}.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("salvou:", out.name,
          "| y_tubete=", seg["y_topo_tubete"],
          "| diam_cilindro_px=", seg["diametro_cilindro_px"],
          "| planta_pixels=", int(np.sum(seg["mask_planta"])))


if __name__ == "__main__":
    caminhos = sorted(list(PASTA_EUCALIPTO.glob("*.jpg"))) + sorted(list(PASTA_PINHEIRO.glob("*.jpg")))
    for c in caminhos:
        Debug(c)
