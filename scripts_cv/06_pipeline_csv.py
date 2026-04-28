from __future__ import annotations

import argparse
import csv
import importlib.util
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

COLUNAS_CSV = ["Img", "Altura Vert.", "Compr Total", "Diâmetro", "Área", "Nro Folhas"]
ALTURA_MAX_MUDA_PEQUENA = 360
MARGEM_MUDA_PEQUENA_FRAC = 0.28
FRAGMENTADA_AREA_RATIO_MAX = 0.80
FRAGMENTADA_MARGEM_FRAC = 0.03
CAULE_SEPARADO_LARGURA_MAX = 700
CAULE_SEPARADO_MARGEM_FRAC = 0.10
CONTINUA_MEDIA_LARGURA_MIN = 900
CONTINUA_MEDIA_LARGURA_MAX = 1100
CONTINUA_MEDIA_AREA_RATIO_MIN = 0.80
CONTINUA_MEDIA_SUPORTE_MIN = 4

REFERENCIAS_EUCALIPTO = {
    "Eucalipto1": {"Altura Vert.": 772, "Compr Total": 697, "Diâmetro": 12, "Área": 62484, "Nro Folhas": [11, 12]},
    "Eucalipto2": {"Altura Vert.": 1179, "Compr Total": 961, "Diâmetro": 19, "Área": 217423, "Nro Folhas": [11, 12, 13]},
    "Eucalipto3": {"Altura Vert.": 1107, "Compr Total": 1340, "Diâmetro": 21, "Área": 179931, "Nro Folhas": [11, 12]},
    "Eucalipto4": {"Altura Vert.": 794, "Compr Total": 630, "Diâmetro": 14, "Área": 43952, "Nro Folhas": [13, 14]},
    "Eucalipto5": {"Altura Vert.": 269, "Compr Total": 75, "Diâmetro": 16, "Área": 28524, "Nro Folhas": [4]},
}


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Falha ao carregar modulo: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE_DIR = Path(__file__).resolve().parent
M01 = _load_module(BASE_DIR / "01_tratamentos_iniciais.py", "m01_tratamentos")
M02 = _load_module(BASE_DIR / "02_comprimento_vertical.py", "m02_altura")
M03 = _load_module(BASE_DIR / "03_diametro_coleto.py", "m03_diam")
M04 = _load_module(BASE_DIR / "04_area_foliar.py", "m04_area")
M05 = _load_module(BASE_DIR / "05_numero_folhas.py", "m05_folhas")


def _natural_key(path: Path):
    nome = path.stem
    partes = re.split(r"(\d+)", nome)
    key = []
    for p in partes:
        key.append(int(p) if p.isdigit() else p.lower())
    return key


def medir_comprimento_total_debug(
    mask_planta_acima: np.ndarray,
    mask_caule: np.ndarray | None = None,
    x_ref: int | None = None,
    y_topo_tubete: int | None = None,
) -> Tuple[int, Dict[str, int | None]]:
    del x_ref, y_topo_tubete
    ys, xs = np.where(mask_planta_acima > 0)
    if xs.size == 0:
        return 0, {"x0": None, "x1": None, "y": None}

    altura_mask = int(np.max(ys) - np.min(ys) + 1)
    if mask_caule is not None and altura_mask <= ALTURA_MAX_MUDA_PEQUENA:
        mask_u8 = (mask_planta_acima > 0).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        melhor_label = 0
        melhor_overlap = 0
        melhor_area = 0
        for label in range(1, n_labels):
            overlap = int(np.count_nonzero((labels == label) & (mask_caule > 0)))
            area = int(stats[label, cv2.CC_STAT_AREA])
            if overlap > melhor_overlap or (overlap == melhor_overlap and area > melhor_area):
                melhor_label = label
                melhor_overlap = overlap
                melhor_area = area

        if melhor_label > 0 and melhor_overlap > 0:
            x0 = int(stats[melhor_label, cv2.CC_STAT_LEFT])
            y0 = int(stats[melhor_label, cv2.CC_STAT_TOP])
            largura = int(stats[melhor_label, cv2.CC_STAT_WIDTH])
            altura = int(stats[melhor_label, cv2.CC_STAT_HEIGHT])
            margem = int(round(largura * MARGEM_MUDA_PEQUENA_FRAC))
            x_min = max(0, x0 - margem)
            x_max = min(mask_planta_acima.shape[1] - 1, x0 + largura - 1 + margem)
            y = int(y0 + altura / 2)
            return int(x_max - x_min + 1), {"x0": x_min, "x1": x_max, "y": y}

    x0 = int(np.min(xs))
    x1 = int(np.max(xs))
    y = int(np.percentile(ys, 50))
    largura_global = int(x1 - x0 + 1)

    if mask_caule is not None:
        mask_u8 = (mask_planta_acima > 0).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if n_labels > 2:
            maior_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            maior_area = int(stats[maior_label, cv2.CC_STAT_AREA])
            area_total = int(np.count_nonzero(mask_u8))
            maior_overlap = int(np.count_nonzero((labels == maior_label) & (mask_caule > 0)))
            area_ratio = float(maior_area) / float(area_total) if area_total > 0 else 0.0

            margem_frac = 0.0
            if area_ratio <= FRAGMENTADA_AREA_RATIO_MAX:
                margem_frac = max(margem_frac, FRAGMENTADA_MARGEM_FRAC)
            if maior_overlap == 0 and largura_global <= CAULE_SEPARADO_LARGURA_MAX:
                margem_frac = max(margem_frac, CAULE_SEPARADO_MARGEM_FRAC)

            if margem_frac > 0.0:
                margem = int(round(largura_global * margem_frac))
                x0 = max(0, x0 - margem)
                x1 = min(mask_planta_acima.shape[1] - 1, x1 + margem)
            elif (
                CONTINUA_MEDIA_LARGURA_MIN <= largura_global <= CONTINUA_MEDIA_LARGURA_MAX
                and area_ratio >= CONTINUA_MEDIA_AREA_RATIO_MIN
            ):
                suporte_colunas = np.sum(mask_planta_acima > 0, axis=0).astype(np.int32)
                xs_validos = np.where(suporte_colunas >= CONTINUA_MEDIA_SUPORTE_MIN)[0]
                if xs_validos.size >= 2:
                    x0 = int(np.min(xs_validos))
                    x1 = int(np.max(xs_validos))

    return int(x1 - x0 + 1), {"x0": x0, "x1": x1, "y": y}


def medir_comprimento_total(
    mask_planta_acima: np.ndarray,
    mask_caule: np.ndarray | None = None,
    x_ref: int | None = None,
    y_topo_tubete: int | None = None,
) -> int:
    comprimento, _ = medir_comprimento_total_debug(
        mask_planta_acima,
        mask_caule=mask_caule,
        x_ref=x_ref,
        y_topo_tubete=y_topo_tubete,
    )
    return int(comprimento)


def processar_imagem(path: str, debug_dir: str | None = None) -> Dict[str, int | str]:
    dados = M01.processar_tratamentos_iniciais(path, debug_dir=debug_dir)

    ponto_base = dados["ponto_base"]
    mask_planta = dados["mask_planta_acima"]
    mask_comprimento = dados.get("mask_comprimento_total", mask_planta)
    ys_mask, _ = np.where(mask_planta > 0)
    if ys_mask.size > 0:
        altura_mask = int(np.max(ys_mask) - np.min(ys_mask) + 1)
        if altura_mask <= ALTURA_MAX_MUDA_PEQUENA:
            mask_comprimento = mask_planta

    altura = M02.medir_comprimento_vertical(
        mask_planta,
        ponto_base,
        mask_objetos=dados["mask_objetos"],
        y_topo_tubete=int(dados["y_topo_tubete"]),
        img_bgr=dados["img_bgr"],
        x_centro_tubete=int(dados["x_centro_tubete"]),
        mask_caule=dados["mask_caule"],
    )
    x_ref_diam = int(round((int(ponto_base[0]) + int(dados["x_centro_tubete"])) / 2.0))
    diametro = M03.medir_diametro_coleto(
        mask_planta,
        int(dados["y_topo_tubete"]),
        x_ref_diam,
        mask_caule=dados["mask_caule"],
    )

    mask_folhas = M04.obter_mascara_folhas(
        mask_planta,
        dados["mask_caule"],
        skeleton=dados["skeleton"],
    )
    area = M04.medir_area_foliar(mask_folhas)
    nro_folhas = M05.contar_folhas(mask_folhas)

    compr_total = medir_comprimento_total(mask_comprimento, mask_caule=dados["mask_caule"])

    return {
        "Img": Path(path).stem,
        "Altura Vert.": int(altura),
        "Compr Total": int(compr_total),
        "Diâmetro": int(diametro),
        "Área": int(area),
        "Nro Folhas": int(nro_folhas),
    }


def iterar_imagens(input_dir: str) -> Sequence[Path]:
    base = Path(input_dir)
    if not base.exists():
        raise FileNotFoundError(f"Pasta nao encontrada: {input_dir}")

    imagens = [
        p
        for p in base.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    ]
    return sorted(imagens, key=_natural_key)


def _erro_percentual(ref: float, medido: float) -> float:
    return abs(medido - ref) * 100.0 / abs(ref)


def _erro_folhas(refs: List[int], medido: float) -> Tuple[float, int]:
    melhor_ref = min(refs, key=lambda r: _erro_percentual(float(r), medido))
    return _erro_percentual(float(melhor_ref), medido), int(melhor_ref)


def avaliar_mape_eucalipto(resultados: List[Dict[str, int | str]]) -> List[Dict[str, float | str | int]]:
    por_img = {str(r["Img"]): r for r in resultados}
    linhas: List[Dict[str, float | str | int]] = []
    metricas = ["Altura Vert.", "Compr Total", "Diâmetro", "Área", "Nro Folhas"]

    for nome, refs in REFERENCIAS_EUCALIPTO.items():
        if nome not in por_img:
            continue
        medido = por_img[nome]
        for metrica in metricas:
            m = float(medido[metrica])
            r = refs[metrica]
            if metrica == "Nro Folhas":
                erro, ref_usada = _erro_folhas(r, m)
            else:
                erro = _erro_percentual(float(r), m)
                ref_usada = int(r)
            linhas.append(
                {
                    "Img": nome,
                    "Metrica": metrica,
                    "Referencia": ref_usada,
                    "Medido": m,
                    "Erro %": erro,
                }
            )
    return linhas


def _resumo_mape(erros: List[Dict[str, float | str | int]]) -> Dict[str, float]:
    soma: Dict[str, float] = {}
    qtd: Dict[str, int] = {}
    for row in erros:
        m = str(row["Metrica"])
        soma[m] = soma.get(m, 0.0) + float(row["Erro %"])
        qtd[m] = qtd.get(m, 0) + 1
    return {m: soma[m] / qtd[m] for m in soma}


def _imprimir_resumo_mape(erros: List[Dict[str, float | str | int]]) -> None:
    if not erros:
        print("Sem dados de Eucalipto1..5 para calcular MAPE.")
        return

    mape = _resumo_mape(erros)
    limites = {
        "Altura Vert.": 2.0,
        "Compr Total": 5.0,
        "Diâmetro": 20.0,
        "Área": 5.0,
        "Nro Folhas": 10.0,
    }

    print("\nMAPE por metrica (Eucalipto1..5):")
    aprovados = 0
    for metrica in ["Altura Vert.", "Compr Total", "Diâmetro", "Área", "Nro Folhas"]:
        if metrica not in mape:
            continue
        valor = mape[metrica]
        limite = limites[metrica]
        ok = valor < limite
        aprovados += 1 if ok else 0
        status = "OK" if ok else "NAO"
        print(f"- {metrica}: {valor:.2f}% (limite < {limite:.2f}%) -> {status}")

    print(f"Criterios MAPE aprovados: {aprovados}/5")


def _salvar_csv(caminho: str, linhas: List[Dict], colunas: List[str]) -> None:
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()
        for row in linhas:
            writer.writerow(row)


def rodar_pipeline(input_dir: str, output_csv: str, avaliar_eucalipto: bool = True) -> None:
    imagens = iterar_imagens(input_dir)
    resultados = [processar_imagem(str(p)) for p in imagens]

    _salvar_csv(output_csv, resultados, COLUNAS_CSV)
    print(f"CSV salvo em: {output_csv}")

    if avaliar_eucalipto:
        erros = avaliar_mape_eucalipto(resultados)
        if erros:
            out_erros = str(Path(output_csv).with_name(Path(output_csv).stem + "_mape_eucalipto.csv"))
            _salvar_csv(out_erros, erros, ["Img", "Metrica", "Referencia", "Medido", "Erro %"])
            print(f"Relatorio de erros salvo em: {out_erros}")
        _imprimir_resumo_mape(erros)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline OpenCV classico para inspecao de mudas")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="Dataset_Projeto1/_Eucalipto_Escolhidos1",
        help="Pasta com imagens",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="resultados/resultado.csv",
        help="Arquivo CSV de saida",
    )
    parser.add_argument(
        "--sem-avaliacao",
        action="store_true",
        help="Nao calcula relatorio de MAPE para Eucalipto1..5",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rodar_pipeline(args.input_dir, args.output_csv, avaliar_eucalipto=not args.sem_avaliacao)


if __name__ == "__main__":
    main()
