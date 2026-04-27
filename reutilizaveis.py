"""
reutilizaveis.py

Biblioteca mínima de funções reutilizáveis para o projeto de Visão de Máquinas
(inspeção de mudas).

Este arquivo NÃO resolve o projeto inteiro. Ele contém apenas blocos pequenos,
genéricos e seguros que serão chamados pelos scripts específicos:

    01_tratamentos_iniciais.py
    02_comprimento_vertical.py
    03_diametro_coleto.py
    04_area_foliar.py
    05_numero_folhas.py
    06_pipeline_csv.py

Decisão central do projeto
--------------------------
Todas as medições finais devem ocorrer no referencial da imagem original.

Isso significa:

1. Não medir em imagem redimensionada.
2. Não medir em máscara redimensionada.
3. Não medir em imagem deformada por escala, rotação, perspectiva ou warp.
4. Resize só deve existir fora deste arquivo, se for necessário apenas para
   visualização local.
5. Crop sem resize não deforma a área, mas muda o shape. Portanto, se uma etapa
   trabalhar em crop, a máscara do crop deve ser recolocada em um canvas do
   tamanho original antes da medição final.

Por que essa rigidez?
---------------------
A área foliar é contagem de pixels. Se a máscara tiver outro tamanho, a área
fica automaticamente em outra escala. Altura, diâmetro e comprimento também
ficam em outro sistema de coordenadas. Para evitar correções posteriores,
o projeto adotará uma regra simples: medir tudo no tamanho original.

As funções abaixo foram escritas como utilidades genéricas. Elas não carregam
parâmetros calibrados, não fazem debug visual e não implementam uma solução
fechada para nenhuma imagem específica.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


Array = np.ndarray
PointYX = Tuple[int, int]
Segment = Tuple[int, int]


# =============================================================================
# Leitura e conversões básicas
# =============================================================================

def ler_imagem_bgr(caminho: Union[str, Path]) -> Array:
    """
    Lê uma imagem em BGR, formato padrão do OpenCV.

    Esta função não redimensiona, não corta e não altera a imagem.
    Isso é importante porque todas as medidas finais devem permanecer
    no referencial original.
    """
    img = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada ou inválida: {caminho}")
    return img


def para_uint8(mask: Array) -> Array:
    """
    Converte máscara booleana/numérica para uint8 com valores 0 ou 255.

    Útil porque muitas funções do OpenCV esperam uint8.
    """
    return (mask > 0).astype(np.uint8) * 255


def para_bool(mask: Array) -> Array:
    """
    Converte máscara para bool.

    Útil para operações lógicas e contagem segura de pixels.
    """
    return mask > 0


# =============================================================================
# Regra de segurança: medição apenas no referencial original
# =============================================================================

def validar_shape_original(mask: Array, img_original: Array, nome: str = "máscara") -> None:
    """
    Garante que a máscara usada em uma medição tem o mesmo shape espacial
    da imagem original.

    Se esta função falhar, não se deve medir área, altura, diâmetro ou
    comprimento nessa máscara. A máscara deve ser refeita no tamanho original
    ou recolocada em um canvas original, caso tenha vindo de crop sem resize.
    """
    shape_mask = mask.shape[:2]
    shape_img = img_original.shape[:2]

    if shape_mask != shape_img:
        raise ValueError(
            f"{nome} tem shape {shape_mask}, mas a imagem original tem shape {shape_img}. "
            "Medição bloqueada para evitar erro de escala. "
            "Use máscaras no tamanho original ou recoloque o crop em um canvas original."
        )


def recolocar_crop_no_canvas(
    mask_crop: Array,
    shape_original: Union[Tuple[int, int], Tuple[int, int, int]],
    x0: int,
    y0: int,
) -> Array:
    """
    Recoloca uma máscara obtida por crop em um canvas do tamanho original.

    Crop sem resize não altera a quantidade de pixels dentro da região cortada,
    mas a máscara fica com shape menor. Para manter a regra do projeto, a máscara
    deve voltar ao canvas original antes de qualquer medição final.

    Parâmetros:
        mask_crop:
            Máscara calculada no recorte, sem resize.
        shape_original:
            Shape da imagem original. Pode ser (H, W) ou (H, W, C).
        x0, y0:
            Coordenadas do canto superior esquerdo do crop no sistema original.

    Retorna:
        canvas:
            Máscara com shape (H_original, W_original), contendo o crop na posição
            correta e zeros fora dele.
    """
    h_orig, w_orig = shape_original[:2]
    h_crop, w_crop = mask_crop.shape[:2]

    if x0 < 0 or y0 < 0:
        raise ValueError("x0 e y0 devem ser não negativos.")

    if y0 + h_crop > h_orig or x0 + w_crop > w_orig:
        raise ValueError(
            "O crop extrapola o tamanho original. Verifique x0, y0 e shape_original."
        )

    canvas = np.zeros((h_orig, w_orig), dtype=mask_crop.dtype)
    canvas[y0:y0 + h_crop, x0:x0 + w_crop] = mask_crop
    return canvas


# =============================================================================
# Morfologia e componentes conexos
# =============================================================================

def kernel_elipse(tamanho: Tuple[int, int]) -> Array:
    """
    Cria kernel elíptico para morfologia.

    O tamanho deve ser pequeno e intencional. Kernels grandes podem apagar caule,
    juntar folhas indevidamente ou distorcer a máscara foliar.
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tamanho)


def morfologia(mask: Array, operacao: str, tamanho_kernel: Tuple[int, int], iteracoes: int = 1) -> Array:
    """
    Aplica operação morfológica simples.

    Operações aceitas:
        "open"   -> remove ruídos pequenos.
        "close"  -> fecha falhas/buracos pequenos.
        "erode"  -> afina/separa regiões.
        "dilate" -> engrossa/conecta regiões.

    Observação importante:
        Morfologia altera a máscara. Isso é aceitável para segmentação,
        mas deve ser usado com cuidado, especialmente antes da área foliar.
        Sempre que possível, prefira operações compostas com cv2.morphologyEx
        como abertura e fechamento, em vez de encadear erode e dilate várias
        vezes sem controle. Isso tende a deixar o tratamento mais claro,
        previsível e fácil de ajustar.
        Se o kernel for agressivo, a área medida deixa de representar bem
        a folha real.
    """
    mask_u8 = para_uint8(mask)
    k = kernel_elipse(tamanho_kernel)

    if operacao == "open":
        return cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=iteracoes)
    if operacao == "close":
        return cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=iteracoes)
    if operacao == "erode":
        return cv2.erode(mask_u8, k, iterations=iteracoes)
    if operacao == "dilate":
        return cv2.dilate(mask_u8, k, iterations=iteracoes)

    raise ValueError("Operação inválida. Use: 'open', 'close', 'erode' ou 'dilate'.")


def filtrar_componentes_por_area(mask: Array, area_min: int) -> Array:
    """
    Remove componentes conectados menores que area_min.

    Reutilizável para:
        - tirar ruído da máscara de planta;
        - tirar pontos soltos da máscara foliar;
        - limpar resíduos após subtrair caule.

    Não usar area_min alto demais sem checar o resultado, pois folhas pequenas
    podem ser removidas.
    """
    mask_u8 = para_uint8(mask)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    saida = np.zeros_like(mask_u8)
    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= area_min:
            saida[labels == label] = 255

    return saida


def maior_componente(mask: Array) -> Array:
    """
    Mantém apenas o maior componente conectado da máscara.

    Útil quando já se sabe que a região de interesse principal é dominante.
    Cuidado: para folhas separadas, esta função NÃO deve ser usada, pois
    descartaria folhas legítimas menores.
    """
    mask_u8 = para_uint8(mask)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    if n_labels <= 1:
        return np.zeros_like(mask_u8)

    label_maior = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == label_maior, 255, 0).astype(np.uint8)


def componentes_que_tocam_seed(mask: Array, seed: Array, area_min: int = 1) -> Array:
    """
    Mantém componentes de 'mask' que encostam em uma máscara-semente.

    Ideia geral:
        Se uma região conhecida é confiável, podemos preservar somente os
        componentes conectados que a tocam.

    Exemplo de uso possível:
        - manter componentes da planta que encostam perto do coleto;
        - manter folhas laterais que tocam uma região de interesse;
        - recuperar partes da planta sem manter ruído solto.

    A função é genérica: a decisão de qual seed usar pertence aos scripts
    específicos do projeto.
    """
    mask_u8 = para_uint8(mask)
    seed_bool = para_bool(seed)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    saida = np.zeros_like(mask_u8)

    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < area_min:
            continue

        comp = labels == label
        if np.any(comp & seed_bool):
            saida[comp] = 255

    return saida


# =============================================================================
# Geometria de máscaras
# =============================================================================

def bbox_mascara(mask: Array) -> Optional[Tuple[int, int, int, int]]:
    """
    Retorna bounding box da máscara no formato (x, y, w, h).

    Se a máscara estiver vazia, retorna None.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())

    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def info_altura(mask: Array) -> Tuple[int, int, int]:
    """
    Retorna (y_topo, y_base, altura) de uma máscara.

    y_topo:
        menor y da máscara.
    y_base:
        maior y da máscara.
    altura:
        y_base - y_topo.

    Observação:
        Isto NÃO substitui a altura vertical oficial se a base correta for o
        coleto/topo do vaso. A função só resume a extensão vertical da máscara.
    """
    ys, _ = np.where(mask > 0)
    if len(ys) == 0:
        return 0, 0, 0

    y_topo = int(ys.min())
    y_base = int(ys.max())
    return y_topo, y_base, int(y_base - y_topo)


def segmentos_horizontais_na_linha(mask: Array, y: int) -> List[Segment]:
    """
    Retorna segmentos horizontais contínuos de pixels positivos em uma linha.

    Cada segmento é (x_inicio, x_fim), inclusive.

    Uso principal esperado:
        Diâmetro do coleto. Mede-se a largura horizontal do segmento do caule
        em uma linha próxima ao topo do vaso.

    Cuidado:
        Se a linha atravessar folha lateral, pode haver mais de um segmento.
        O script de diâmetro deve escolher o segmento mais plausível, geralmente
        o mais próximo do eixo/base/centro de referência.
    """
    if y < 0 or y >= mask.shape[0]:
        return []

    xs = np.where(mask[y] > 0)[0]
    if len(xs) == 0:
        return []

    segmentos: List[Segment] = []
    inicio = int(xs[0])
    anterior = int(xs[0])

    for x in xs[1:]:
        x = int(x)
        if x == anterior + 1:
            anterior = x
        else:
            segmentos.append((inicio, anterior))
            inicio = x
            anterior = x

    segmentos.append((inicio, anterior))
    return segmentos


def escolher_segmento_mais_proximo(segmentos: Sequence[Segment], x_ref: int) -> Optional[Segment]:
    """
    Escolhe o segmento horizontal cujo centro está mais próximo de x_ref.

    Útil para o diâmetro do coleto quando a linha tem ruídos ou folhas laterais.
    """
    if not segmentos:
        return None

    def distancia(seg: Segment) -> float:
        x0, x1 = seg
        centro = 0.5 * (x0 + x1)
        return abs(centro - x_ref)

    return min(segmentos, key=distancia)


# =============================================================================
# Esqueleto e caminhos
# =============================================================================

def skeletonize_morfologico(mask: Array) -> Array:
    """
    Skeletonização morfológica simples, usando apenas OpenCV.

    Uso esperado:
        Comprimento avançado/eixo principal.

    Importante:
        Esqueleto NÃO deve ser usado para medir área foliar. Ele destrói a área,
        deixando apenas uma linha central.
    """
    img = para_uint8(mask)
    skel = np.zeros_like(img)
    elemento = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        erodida = cv2.erode(img, elemento)
        aberta = cv2.morphologyEx(erodida, cv2.MORPH_OPEN, elemento)
        borda = cv2.subtract(erodida, aberta)
        skel = cv2.bitwise_or(skel, borda)
        img = erodida

        if cv2.countNonZero(img) == 0:
            break

    return skel


def comprimento_caminho(pontos_yx: Sequence[PointYX]) -> float:
    """
    Soma o comprimento euclidiano de uma sequência de pontos (y, x).

    Serve para medir caminho no esqueleto/eixo principal quando o script
    específico já tiver decidido qual é o caminho correto.
    """
    if len(pontos_yx) < 2:
        return 0.0

    total = 0.0
    for (y0, x0), (y1, x1) in zip(pontos_yx[:-1], pontos_yx[1:]):
        total += float(np.hypot(y1 - y0, x1 - x0))

    return total


# =============================================================================
# Área foliar e preenchimento de contornos
# =============================================================================

def preencher_contornos(mask: Array, area_min: int = 1) -> Array:
    """
    Preenche contornos externos de uma máscara.

    Pode ser útil quando a máscara foliar tem pequenos buracos internos causados
    por brilho, textura ou falhas de threshold.

    Cuidado:
        Preencher contornos também pode juntar regiões indevidas se a máscara
        estiver muito suja. Deve ser usado apenas quando fizer sentido visual
        e quantitativo.
    """
    mask_u8 = para_uint8(mask)
    contornos, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    saida = np.zeros_like(mask_u8)
    for c in contornos:
        if cv2.contourArea(c) >= area_min:
            cv2.drawContours(saida, [c], contourIdx=-1, color=255, thickness=-1)

    return saida


def medir_area_foliar(mask_folhas: Array, img_original: Array) -> int:
    """
    Mede a área foliar de forma segura.

    Esta é a função oficial para área foliar no projeto.

    Regras:
        1. A máscara deve estar no mesmo shape da imagem original.
        2. A máscara deve representar folhas preenchidas, não esqueleto.
        3. A máscara não deve conter vaso/tubete.
        4. A máscara não deve ser resultado de resize.
        5. Se houve crop, o crop deve ter sido recolocado no canvas original.

    Retorna:
        Quantidade de pixels positivos da máscara.
    """
    validar_shape_original(mask_folhas, img_original, nome="mask_folhas")
    return int(np.count_nonzero(mask_folhas > 0))


# =============================================================================
# Métricas de erro
# =============================================================================

def erro_percentual(valor: float, referencia: float) -> float:
    """
    Erro percentual absoluto.

    Fórmula:
        abs(valor - referencia) / referencia * 100

    Se a referência for zero, a métrica não é definida.
    """
    if referencia == 0:
        raise ValueError("Referência zero: erro percentual indefinido.")
    return abs(float(valor) - float(referencia)) / abs(float(referencia)) * 100.0


def erro_percentual_com_referencias_aceitas(valor: float, referencias: Union[float, Sequence[float]]) -> float:
    """
    Calcula o menor erro percentual quando há mais de uma referência aceita.

    Útil para número de folhas quando o gabarito aceita, por exemplo, 11 ou 12.
    """
    if isinstance(referencias, (int, float, np.integer, np.floating)):
        return erro_percentual(float(valor), float(referencias))

    refs = list(referencias)
    if not refs:
        raise ValueError("Lista de referências vazia.")

    return min(erro_percentual(valor, ref) for ref in refs)


def mape(valores: Sequence[float], referencias: Sequence[Union[float, Sequence[float]]]) -> float:
    """
    Mean Absolute Percentage Error.

    Aceita referências únicas ou múltiplas referências aceitas por item.
    """
    if len(valores) != len(referencias):
        raise ValueError("valores e referencias devem ter o mesmo tamanho.")

    if len(valores) == 0:
        raise ValueError("Não é possível calcular MAPE com listas vazias.")

    erros = [
        erro_percentual_com_referencias_aceitas(v, r)
        for v, r in zip(valores, referencias)
    ]
    return float(np.mean(erros))


# =============================================================================
# Referências oficiais dos cinco eucaliptos do enunciado
# =============================================================================

REFERENCIAS_EUCALIPTO = {
    1: {
        "Altura Vert.": 772,
        "Compr Total": 697,
        "Diâmetro": 12,
        "Área": 62484,
        "Nro Folhas": (11, 12),
    },
    2: {
        "Altura Vert.": 1179,
        "Compr Total": 961,
        "Diâmetro": 19,
        "Área": 217423,
        "Nro Folhas": (11, 12, 13),
    },
    3: {
        "Altura Vert.": 1107,
        "Compr Total": 1340,
        "Diâmetro": 21,
        "Área": 179931,
        "Nro Folhas": (11, 12),
    },
    4: {
        "Altura Vert.": 794,
        "Compr Total": 630,
        "Diâmetro": 14,
        "Área": 43952,
        "Nro Folhas": (13, 14),
    },
    5: {
        "Altura Vert.": 269,
        "Compr Total": 75,
        "Diâmetro": 16,
        "Área": 28524,
        "Nro Folhas": 4,
    },
}


LIMITES_MAPE_CONCEITO_B = {
    "Altura Vert.": 2.0,
    "Compr Total": 5.0,
    "Diâmetro": 20.0,
    "Área": 5.0,
    "Nro Folhas": 10.0,
}


def mape_por_metrica(resultados: Sequence[dict], metrica: str) -> float:
    """
    Calcula MAPE de uma métrica usando os cinco eucaliptos do enunciado.

    Cada item de resultados deve conter:
        "Img": número da imagem, de 1 a 5;
        metrica: valor calculado.

    Esta função não decide aprovação sozinha. Ela só calcula a métrica.
    """
    valores = []
    refs = []

    for item in resultados:
        img_id = int(item["Img"])
        if img_id not in REFERENCIAS_EUCALIPTO:
            continue

        valores.append(item[metrica])
        refs.append(REFERENCIAS_EUCALIPTO[img_id][metrica])

    return mape(valores, refs)


def criterio_b_por_metrica(resultados: Sequence[dict], metrica: str) -> Tuple[float, bool]:
    """
    Calcula o MAPE de uma métrica e informa se ela bate o limite do conceito B.

    Retorna:
        (valor_mape, passou)
    """
    if metrica not in LIMITES_MAPE_CONCEITO_B:
        raise ValueError(f"Métrica desconhecida para conceito B: {metrica}")

    valor_mape = mape_por_metrica(resultados, metrica)
    return valor_mape, valor_mape < LIMITES_MAPE_CONCEITO_B[metrica]


# =============================================================================
# Apoios opcionais para PINHEIRO
# =============================================================================
#
# Estas funções NÃO são uma solução garantida para pinheiro.
#
# Elas foram adicionadas porque, no nosso material antigo, a ideia que parecia
# promissora para pinheiro era diferente da do eucalipto:
#
#   - o pinheiro tende a ter copa/folhagem mais fina e espalhada;
#   - a região central pode ser dominada por caule/eixo;
#   - parte útil da folhagem pode aparecer como estruturas laterais finas;
#   - o topo da muda pode precisar ser preservado mesmo quando está perto do eixo;
#   - a borda superior do vaso/tubete pode contaminar a máscara perto da base.
#
# Portanto, estes blocos só ajudam a construir máscaras candidatas para pinheiro.
# Eles devem ser testados caso a entrega inclua pinheiros. Para a meta principal
# de bater os Eucaliptos do enunciado, eles não são necessários.


def regiao_topo_relativa(mask_corpo: Array, fracao_altura: float, x_centro: int, meia_largura: int) -> Array:
    """
    Cria uma região vertical no topo da planta, limitada horizontalmente.

    Possível utilidade para pinheiro:
        Preservar a parte superior da copa, que pode ser fina e próxima ao eixo.
        Em pinheiro, remover tudo perto do eixo pode apagar estruturas reais do topo.

    Parâmetros:
        mask_corpo:
            Máscara da planta/corpo já separada do vaso.
        fracao_altura:
            Fração da altura total da máscara a preservar desde o topo.
            Ex.: 0.20 preserva os primeiros 20% da altura.
        x_centro:
            Centro aproximado do vaso/eixo.
        meia_largura:
            Metade da largura horizontal da janela preservada.

    Observação:
        Esta função apenas cria uma máscara auxiliar. Ela não deve decidir sozinha
        o que é folha, caule ou ruído.
    """
    if not (0 < fracao_altura <= 1):
        raise ValueError("fracao_altura deve estar no intervalo (0, 1].")

    out = np.zeros_like(mask_corpo, dtype=np.uint8)
    y_topo, _, altura = info_altura(mask_corpo)
    if altura <= 0:
        return out

    h, w = mask_corpo.shape[:2]
    y1 = min(h, int(round(y_topo + fracao_altura * altura)))
    x0 = max(0, int(x_centro - meia_largura))
    x1 = min(w, int(x_centro + meia_largura + 1))

    out[y_topo:y1, x0:x1] = 255
    return out


def eixo_x_por_linha(mask_eixo: Array, x_padrao: int) -> Array:
    """
    Estima um x de eixo para cada linha da imagem.

    Para cada linha com pixels na máscara, usa a mediana dos x positivos.
    Para linhas vazias, repete o último x conhecido; se ainda não houver,
    usa x_padrao.

    Possível utilidade para pinheiro:
        Construir filtros laterais em relação ao eixo da muda, sem assumir que
        o caule é perfeitamente vertical.
    """
    h = mask_eixo.shape[0]
    eixo = np.empty(h, dtype=np.float32)

    ultimo = float(x_padrao)
    for y in range(h):
        xs = np.where(mask_eixo[y] > 0)[0]
        if len(xs) > 0:
            ultimo = float(np.median(xs))
        eixo[y] = ultimo

    return eixo


def mascara_lateral_ao_eixo(mask: Array, eixo_x: Array, distancia_minima: float) -> Array:
    """
    Mantém pixels suficientemente laterais em relação ao eixo estimado.

    Possível utilidade para pinheiro:
        Separar estruturas laterais da copa/folhas do eixo/caule central.

    Cuidado:
        Se distancia_minima for alta demais, folhas próximas ao caule somem.
        Se for baixa demais, caule entra na máscara foliar.
    """
    h, w = mask.shape[:2]
    if len(eixo_x) != h:
        raise ValueError("eixo_x deve ter um valor para cada linha da máscara.")

    xx = np.arange(w, dtype=np.float32)[None, :]
    dist = np.abs(xx - eixo_x[:, None])
    return (para_bool(mask) & (dist >= distancia_minima)).astype(np.uint8) * 255


def mascara_lateral_variavel(
    mask: Array,
    eixo_x: Array,
    y_topo: int,
    y_base: int,
    distancia_topo: float,
    distancia_base: float,
) -> Array:
    """
    Versão com distância lateral variável ao longo da altura.

    A distância mínima muda linearmente entre:
        distancia_topo, perto do topo;
        distancia_base, perto da base.

    Possível utilidade para pinheiro:
        O formato da copa pode exigir tolerância diferente no topo e na base.
        Isso permite uma regra mais suave que um corte lateral fixo.

    Ainda assim, é apenas um filtro auxiliar. Não garante máscara foliar correta.
    """
    h, w = mask.shape[:2]
    if len(eixo_x) != h:
        raise ValueError("eixo_x deve ter um valor para cada linha da máscara.")

    yy_altura = max(1, int(y_base - y_topo))
    xx = np.arange(w, dtype=np.float32)[None, :]
    saida = np.zeros((h, w), dtype=bool)

    for y in range(h):
        if y <= y_topo:
            dist_min = distancia_topo
        elif y >= y_base:
            dist_min = distancia_base
        else:
            t = (y - y_topo) / yy_altura
            dist_min = (1.0 - t) * distancia_topo + t * distancia_base

        saida[y] = np.abs(xx[0] - eixo_x[y]) >= dist_min

    return (para_bool(mask) & saida).astype(np.uint8) * 255


def cortar_regiao_vaso_por_janela(
    mask: Array,
    y_topo_vaso: int,
    x_centro_vaso: int,
    pixels_acima: int,
    pixels_abaixo: int,
    meia_largura: int,
) -> Array:
    """
    Remove uma janela retangular em torno do topo do vaso.

    Possível utilidade para pinheiro:
        A borda superior do tubete pode entrar na máscara como se fosse folha
        ou base da planta. Um corte local pode remover essa contaminação.

    Cuidado:
        Se a janela for grande demais, pode apagar a base real da muda.
        Use apenas como etapa conservadora e documentada.
    """
    out = para_uint8(mask).copy()
    h, w = out.shape[:2]

    y0 = max(0, int(y_topo_vaso - pixels_acima))
    y1 = min(h, int(y_topo_vaso + pixels_abaixo + 1))
    x0 = max(0, int(x_centro_vaso - meia_largura))
    x1 = min(w, int(x_centro_vaso + meia_largura + 1))

    out[y0:y1, x0:x1] = 0
    return out


def estimar_inicio_ramificacao_por_largura(
    mask_corpo: Array,
    y_topo: int,
    y_base: int,
    fator_largura: float,
) -> int:
    """
    Estima a linha onde a planta começa a ficar mais larga.

    Ideia:
        Mede a largura horizontal da máscara em cada linha. A largura inicial
        tende a representar caule/eixo. Quando a largura cresce muito, pode
        indicar início de copa/ramificação.

    Possível utilidade para pinheiro:
        Ajuda a separar uma região superior da copa, que talvez deva ser
        preservada de modo diferente do caule central.

    Cuidado:
        Esta heurística pode falhar se a segmentação estiver quebrada, se houver
        folhas baixas largas ou se o caule estiver inclinado. Use apenas como
        estimativa auxiliar.
    """
    if fator_largura <= 1:
        raise ValueError("fator_largura deve ser maior que 1.")

    larguras: List[int] = []
    linhas: List[int] = []

    h = mask_corpo.shape[0]
    y0 = max(0, int(y_topo))
    y1 = min(h - 1, int(y_base))

    for y in range(y0, y1 + 1):
        xs = np.where(mask_corpo[y] > 0)[0]
        if len(xs) == 0:
            continue
        larguras.append(int(xs.max() - xs.min() + 1))
        linhas.append(y)

    if not larguras:
        return int(y_base)

    n_base = max(3, len(larguras) // 8)
    largura_referencia = float(np.median(larguras[:n_base]))

    for y, largura in zip(linhas, larguras):
        if largura >= largura_referencia * fator_largura:
            return int(y)

    return int(linhas[min(len(linhas) - 1, len(linhas) // 3)])


def podar_caule_por_suporte_vertical(
    mask_caule: Array,
    altura_suporte: int,
    raio_horizontal: int,
    contagem_minima: int,
) -> Array:
    """
    Remove pixels de caule que não têm suporte vertical suficiente acima.

    Possível utilidade para pinheiro:
        Em máscaras finas, podem aparecer falsos pedaços verticais ou ruídos
        parecidos com caule. Exigir suporte vertical ajuda a manter apenas uma
        estrutura mais contínua.

    Cuidado:
        Se a planta estiver muito curvada ou se o caule estiver quebrado na
        segmentação, esta função pode remover caule verdadeiro.
    """
    caule = para_bool(mask_caule)
    if not np.any(caule):
        return np.zeros_like(mask_caule, dtype=np.uint8)

    u8 = caule.astype(np.uint8)
    k = np.ones((int(altura_suporte), 2 * int(raio_horizontal) + 1), dtype=np.uint8)

    suporte = cv2.filter2D(
        u8,
        ddepth=-1,
        kernel=k,
        anchor=(int(raio_horizontal), int(altura_suporte) - 1),
        borderType=cv2.BORDER_CONSTANT,
    )

    saida = caule & (suporte >= int(contagem_minima))
    return saida.astype(np.uint8) * 255
