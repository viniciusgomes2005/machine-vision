"""
Testes integrados: segmentacao + medidas.
"""
from pathlib import Path
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
PASTA_EUC = BASE / "Dataset_Projeto1" / "_Eucalipto_Escolhidos1"
PASTA_PIN = BASE / "Dataset_Projeto1" / "_Pinheiro_Escolhidos1"
PASTA_DBG = BASE / "_dev_debug"
PASTA_DBG.mkdir(exist_ok=True)


def LerImagemRGB(caminho):
    bgr = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def Especie(caminho):
    return "pinheiro" if "pinheiro" in caminho.name.lower() else "eucalipto"


def _maior_componente(mask):
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx


def SegmentaCena(img_rgb):
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    mask_fundo = (H >= 90) & (H <= 135) & (S >= 60) & (V >= 50) & (B.astype(int) > R.astype(int) + 8)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mask_preto = (V < 80) & (~mask_fundo)
    mask_preto = cv2.morphologyEx(mask_preto.astype(np.uint8) * 255, cv2.MORPH_OPEN, k3) > 0
    mask_preto[: int(0.40 * h)] = False
    mask_tubete = _maior_componente(mask_preto)

    mask_branco = (V >= 170) & (S < 60) & (~mask_fundo)
    mask_branco = cv2.morphologyEx(mask_branco.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    mask_branco[: int(0.45 * h)] = False
    mask_cilindro = _maior_componente(mask_branco)

    if np.any(mask_tubete):
        ys, xs = np.where(mask_tubete)
        y_topo = int(ys.min())
        x_centro = int(np.median(xs))
    else:
        y_topo = int(0.62 * h)
        x_centro = w // 2

    cor_planta = (
        ((H >= 20) & (H <= 95))
        | ((H <= 15) & (S >= 90))
        | ((G.astype(int) - B.astype(int) > 5) & (G > 60))
    )
    mask_planta = cor_planta & (~mask_fundo) & (~mask_tubete) & (~mask_cilindro)
    mask_planta[y_topo + 10:, :] = False

    mask_planta = cv2.morphologyEx(mask_planta.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    mask_planta = cv2.morphologyEx(mask_planta.astype(np.uint8) * 255, cv2.MORPH_OPEN, k3) > 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_planta.astype(np.uint8), connectivity=8)
    melhor_idx, melhor_score = -1, -1.0
    for idx in range(1, n):
        area = stats[idx, cv2.CC_STAT_AREA]
        y0 = stats[idx, cv2.CC_STAT_TOP]
        hh = stats[idx, cv2.CC_STAT_HEIGHT]
        if area < 200:
            continue
        y_max = y0 + hh
        dist_y = max(0, y_topo - y_max)
        score = area * (1.0 / (1.0 + dist_y / 50.0))
        if score > melhor_score:
            melhor_score = score
            melhor_idx = idx

    mask_final = np.zeros_like(mask_planta)
    if melhor_idx >= 0:
        principal = labels == melhor_idx
        dil = cv2.dilate(principal.astype(np.uint8), np.ones((25, 25), np.uint8), iterations=5) > 0
        for idx in range(1, n):
            if stats[idx, cv2.CC_STAT_AREA] < 80:
                continue
            comp = labels == idx
            if np.any(comp & dil):
                mask_final |= comp

    diam_cil_px = None
    if np.any(mask_cilindro):
        _, xs_c = np.where(mask_cilindro)
        diam_cil_px = float(xs_c.max() - xs_c.min() + 1)

    return {
        "mask_fundo": mask_fundo,
        "mask_tubete": mask_tubete,
        "mask_cilindro": mask_cilindro,
        "mask_planta": mask_final,
        "y_topo_tubete": y_topo,
        "x_centro_tubete": x_centro,
        "diam_cilindro_px": diam_cil_px,
    }


# -----------------------------------------------------------------
# Parte 1 - altura basica
# -----------------------------------------------------------------
def AlturaBasica(mask_planta, y_topo_tubete, x_centro_tubete):
    if not np.any(mask_planta):
        return None
    ys, xs = np.where(mask_planta)
    y_top = int(ys.min())
    xs_topo = xs[ys == y_top]
    x_top = int(np.median(xs_topo))
    return {
        "ponto_topo": (x_top, y_top),
        "ponto_base": (x_centro_tubete, y_topo_tubete),
        "altura_px": float(y_topo_tubete - y_top),
        "comprimento_reto_px": float(math.hypot(x_top - x_centro_tubete, y_topo_tubete - y_top)),
    }


# -----------------------------------------------------------------
# Parte 2 - diametro de coleto
# -----------------------------------------------------------------
def DiametroColeto(mask_planta, y_topo_tubete, x_centro_tubete, janela_acima=70, meia_largura_x=180):
    """
    Mede o diametro do caule logo acima da linha do tubete.
    Ideia: em uma faixa de 70 pixels acima do topo do tubete, procuro a linha
    em que o segmento continuo de planta passando PERTO do centro do tubete
    tem a menor largura. Essa largura e o diametro do coleto.
    """
    h, w = mask_planta.shape
    y0 = max(0, y_topo_tubete - janela_acima)
    y1 = max(0, y_topo_tubete - 2)
    if y1 <= y0:
        return None

    larguras = []
    detalhes = []
    for y in range(y0, y1 + 1):
        linha = mask_planta[y]
        if not linha[x_centro_tubete]:
            # tolera desvio: procurar segmento mais proximo do centro
            xs_linha = np.where(linha)[0]
            if len(xs_linha) == 0:
                continue
            xc = int(xs_linha[np.argmin(np.abs(xs_linha - x_centro_tubete))])
        else:
            xc = x_centro_tubete
        if abs(xc - x_centro_tubete) > meia_largura_x:
            continue
        if not linha[xc]:
            continue
        a = xc
        while a > 0 and linha[a - 1]:
            a -= 1
        b = xc
        while b < w - 1 and linha[b + 1]:
            b += 1
        largura = b - a + 1
        # pular linhas que sao claramente folhas (muito largas)
        if largura > 80:
            continue
        larguras.append(largura)
        detalhes.append((a, b, y))

    if not larguras:
        return None

    larguras_np = np.array(larguras)
    # Uso a mediana para robustez, mas pego o segmento cuja largura seja igual a essa mediana
    mediana = float(np.median(larguras_np))
    melhor_i = int(np.argmin(np.abs(larguras_np - mediana)))
    a, b, y = detalhes[melhor_i]
    return {
        "diametro_px": float(mediana),
        "segmento": ((int(a), int(y)), (int(b), int(y))),
    }


# -----------------------------------------------------------------
# Parte 3 - area foliar
# -----------------------------------------------------------------
def AreaFoliar(mask_planta, especie):
    """
    Para eucalipto: folhas = regioes da mascara com largura razoavel
        (removemos o caule que e fino pela distancia-transformada).
    Para pinheiro: a massa verde e a "area foliar aparente"
        (aciculas contam como folhagem).
    """
    if not np.any(mask_planta):
        return np.zeros_like(mask_planta, dtype=bool), 0

    if especie == "pinheiro":
        return mask_planta.copy(), int(np.sum(mask_planta))

    # Eucalipto: opening com kernel para retirar o caule fino
    dist = cv2.distanceTransform(mask_planta.astype(np.uint8), cv2.DIST_L2, 3)
    # O caule costuma ter dist <= 6. Folhas sao maiores.
    mask_folhas = (dist > 6) & mask_planta
    # Fecha para recuperar forma
    mask_folhas = cv2.dilate(mask_folhas.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2) > 0
    mask_folhas = mask_folhas & mask_planta  # confinado a planta
    # Filtra componentes pequenos (pedacos de caule com deformacao)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_folhas.astype(np.uint8), connectivity=8)
    lim = 300
    saida = np.zeros_like(mask_folhas, dtype=bool)
    for idx in range(1, n):
        if stats[idx, cv2.CC_STAT_AREA] >= lim:
            saida |= labels == idx
    return saida, int(np.sum(saida))


# -----------------------------------------------------------------
# Parte 4 - numero de folhas
# -----------------------------------------------------------------
def EsqueletoBinario(mask):
    img = (mask.astype(np.uint8) * 255).copy()
    skel = np.zeros_like(img)
    elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        ero = cv2.erode(img, elem)
        ab = cv2.dilate(ero, elem)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, ab))
        img = ero
        if cv2.countNonZero(img) == 0:
            break
    return skel > 0


def EndpointsEsqueleto(skel):
    # Pixel do esqueleto com apenas 1 vizinho de 8-connectividade = endpoint
    k = np.ones((3, 3), np.uint8)
    soma = cv2.filter2D(skel.astype(np.uint8), -1, k)
    endpoints = (skel.astype(np.uint8) == 1) & (soma == 2)  # centro + 1 vizinho = 2
    return endpoints


def ContaFolhasEucalipto(mask_folhas):
    """Cada folha de eucalipto e uma regiao oval. Abrimos para separar e contamos."""
    if not np.any(mask_folhas):
        return 0, []
    mu8 = mask_folhas.astype(np.uint8) * 255
    # Abertura com kernel elipse para separar folhas que se tocam
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    sep = cv2.morphologyEx(mu8, cv2.MORPH_OPEN, kernel)
    n, labels, stats, centros = cv2.connectedComponentsWithStats(sep, connectivity=8)
    pontos = []
    for idx in range(1, n):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < 500:
            continue
        cx, cy = centros[idx]
        pontos.append((int(cx), int(cy)))
    return len(pontos), pontos


def ContaFolhasPinheiro(mask_planta):
    """Contamos aciculas por endpoints do esqueleto."""
    if not np.any(mask_planta):
        return 0, []
    # Esqueletiza a planta
    # Primeiro, alinho um pouco com closing para nao quebrar aciculas
    mu = cv2.morphologyEx(mask_planta.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    skel = EsqueletoBinario(mu > 0)
    # Endpoint detection
    k = np.ones((3, 3), np.uint8)
    soma = cv2.filter2D(skel.astype(np.uint8), -1, k)
    endpoints = (skel.astype(np.uint8) == 1) & (soma == 2)
    ys, xs = np.where(endpoints)
    # Filtro: descarto endpoints proximos da base do tubete (sao raiz / falsos)
    # E tambem endpoints muito agrupados
    pontos = list(zip(xs.tolist(), ys.tolist()))
    # Agrupo por proximidade (clusterizacao simples: descartar ponto a < D do ja aceito)
    D = 35
    filtrados = []
    for p in sorted(pontos, key=lambda q: q[1]):
        ok = True
        for q in filtrados:
            if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < D * D:
                ok = False
                break
        if ok:
            filtrados.append(p)
    return len(filtrados), filtrados


# -----------------------------------------------------------------
# Parte 5 - comprimento seguindo caule
# -----------------------------------------------------------------
def ComprimentoAvancado(mask_planta, y_topo_tubete, x_centro_tubete):
    """
    Seguimos a linha central da planta (distance-transform maxima em cada linha)
    do topo ate a base, passando pelo 'eixo' da planta.
    """
    if not np.any(mask_planta):
        return None
    h, w = mask_planta.shape
    dist = cv2.distanceTransform(mask_planta.astype(np.uint8), cv2.DIST_L2, 3)
    ys, _ = np.where(mask_planta)
    y_top = int(ys.min())
    # Para cada linha de y_top ate y_topo_tubete, escolhe o x dentro da planta
    # cuja distancia-transformada e maior. Comecamos do ponto mais alto.
    pontos = []
    ultimo_x = None
    for y in range(y_top, y_topo_tubete + 1):
        linha = mask_planta[y]
        if not np.any(linha):
            continue
        xs_linha = np.where(linha)[0]
        d_linha = dist[y, xs_linha]
        if ultimo_x is None:
            # Primeiro ponto: x com maior distancia
            x_esc = int(xs_linha[int(np.argmax(d_linha))])
        else:
            # Priorizar continuidade (pequeno delta em x) combinado com d_linha alta
            pesos = d_linha - 0.25 * np.abs(xs_linha - ultimo_x)
            x_esc = int(xs_linha[int(np.argmax(pesos))])
        pontos.append((x_esc, y))
        ultimo_x = x_esc

    if len(pontos) < 2:
        return None

    # Suavizacao: mediana + media
    xs_pontos = np.array([p[0] for p in pontos], dtype=np.float32)
    ys_pontos = np.array([p[1] for p in pontos], dtype=np.float32)

    def _mediana_janela(arr, k=21):
        k = max(3, k | 1)
        pad = k // 2
        ap = np.pad(arr, (pad, pad), mode="edge")
        out = np.empty_like(arr)
        for i in range(len(arr)):
            out[i] = float(np.median(ap[i : i + k]))
        return out

    xs_pontos = _mediana_janela(xs_pontos, k=41)
    xs_pontos = np.convolve(xs_pontos, np.ones(21)/21, mode="same")
    # Refixa base no x_centro_tubete para garantir que a linha chega no coleto
    if len(xs_pontos) > 3:
        xs_pontos[-1] = x_centro_tubete

    pts = [(int(round(x)), int(y)) for x, y in zip(xs_pontos, ys_pontos)]
    total = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return {"pontos": pts, "comprimento_px": float(total)}


# -----------------------------------------------------------------
# VISUALIZACAO
# -----------------------------------------------------------------
def TesteUnico(caminho):
    img = LerImagemRGB(caminho)
    esp = Especie(caminho)
    seg = SegmentaCena(img)

    alt = AlturaBasica(seg["mask_planta"], seg["y_topo_tubete"], seg["x_centro_tubete"])
    diam = DiametroColeto(seg["mask_planta"], seg["y_topo_tubete"], seg["x_centro_tubete"])
    mask_folhas, area_folhas = AreaFoliar(seg["mask_planta"], esp)
    if esp == "eucalipto":
        n_folhas, pts_folhas = ContaFolhasEucalipto(mask_folhas)
    else:
        n_folhas, pts_folhas = ContaFolhasPinheiro(seg["mask_planta"])
    avan = ComprimentoAvancado(seg["mask_planta"], seg["y_topo_tubete"], seg["x_centro_tubete"])

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.ravel()

    # (0) original
    axs[0].imshow(img); axs[0].set_title("original")

    # (1) altura basica
    vis_h = img.copy()
    if alt:
        cv2.line(vis_h, alt["ponto_topo"], alt["ponto_base"], (255, 0, 0), 6)
        cv2.circle(vis_h, alt["ponto_topo"], 12, (255, 255, 0), -1)
        cv2.circle(vis_h, alt["ponto_base"], 12, (0, 255, 255), -1)
        axs[1].set_title(f"altura basica = {alt['altura_px']:.0f}px")
    else:
        axs[1].set_title("altura basica (n/a)")
    axs[1].imshow(vis_h)

    # (2) diametro
    vis_d = img.copy()
    if diam:
        p0, p1 = diam["segmento"]
        cv2.line(vis_d, p0, p1, (255, 0, 255), 10)
        axs[2].set_title(f"diametro coleto = {diam['diametro_px']:.0f}px")
    else:
        axs[2].set_title("diametro coleto (n/a)")
    axs[2].imshow(vis_d)

    # (3) area foliar em vermelho
    vis_a = img.copy()
    vis_a[mask_folhas] = (vis_a[mask_folhas] * 0.25 + np.array([255, 0, 0]) * 0.75).astype(np.uint8)
    axs[3].imshow(vis_a); axs[3].set_title(f"area foliar = {area_folhas} px^2")

    # (4) folhas
    vis_f = img.copy()
    vis_f[mask_folhas] = (vis_f[mask_folhas] * 0.25 + np.array([255, 0, 0]) * 0.75).astype(np.uint8)
    for (cx, cy) in pts_folhas:
        cv2.circle(vis_f, (cx, cy), 18, (255, 255, 0), -1)
        cv2.circle(vis_f, (cx, cy), 18, (0, 0, 0), 2)
    axs[4].imshow(vis_f); axs[4].set_title(f"numero folhas = {n_folhas}")

    # (5) comprimento avancado
    vis_c = img.copy()
    if avan:
        for i in range(1, len(avan["pontos"])):
            cv2.line(vis_c, avan["pontos"][i - 1], avan["pontos"][i], (255, 0, 0), 5)
        axs[5].set_title(f"compr. avancado = {avan['comprimento_px']:.0f}px")
    else:
        axs[5].set_title("compr. avancado (n/a)")
    axs[5].imshow(vis_c)

    for a in axs:
        a.axis("off")
    fig.suptitle(caminho.name)
    plt.tight_layout()
    out = PASTA_DBG / f"measures_{caminho.stem}.png"
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close()
    alt_s = f"{alt['altura_px']:.0f}" if alt else "n/a"
    diam_s = f"{diam['diametro_px']:.1f}" if diam else "n/a"
    compr_s = f"{avan['comprimento_px']:.0f}" if avan else "n/a"
    print(f"{caminho.name} | alt={alt_s} | diam={diam_s} | area={area_folhas} | n_folhas={n_folhas} | compr_avan={compr_s}")


if __name__ == "__main__":
    caminhos = sorted(PASTA_EUC.glob("*.jpg")) + sorted(PASTA_PIN.glob("*.jpg"))
    for c in caminhos:
        TesteUnico(c)
