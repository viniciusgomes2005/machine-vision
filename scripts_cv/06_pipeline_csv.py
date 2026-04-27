from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

COLUNAS_CSV = ["Img", "Altura Vert.", "Compr Total", "Diâmetro", "Área", "Nro Folhas"]

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


def _neighbors8(y: int, x: int, h: int, w: int):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx


def medir_comprimento_total(skeleton: np.ndarray, ponto_base: Tuple[int, int], fallback: int) -> int:
    ys, xs = np.where(skeleton > 0)
    if ys.size == 0:
        return int(fallback)

    h, w = skeleton.shape
    x_base, y_base = ponto_base
    d2 = (xs - x_base) ** 2 + (ys - y_base) ** 2
    idx0 = int(np.argmin(d2))
    x0, y0 = int(xs[idx0]), int(ys[idx0])

    from collections import deque

    dist = np.full((h, w), -1.0, dtype=np.float32)
    fila = deque()
    fila.append((y0, x0))
    dist[y0, x0] = 0.0

    while fila:
        y, x = fila.popleft()
        for ny, nx in _neighbors8(y, x, h, w):
            if skeleton[ny, nx] == 0:
                continue
            step = math.sqrt(2.0) if (ny != y and nx != x) else 1.0
            nd = dist[y, x] + step
            if dist[ny, nx] < 0 or nd < dist[ny, nx]:
                dist[ny, nx] = nd
                fila.append((ny, nx))

    ys_r, xs_r = np.where(dist >= 0)
    if ys_r.size == 0:
        return int(fallback)

    idx_top = int(np.argmin(ys_r))
    y_top, x_top = int(ys_r[idx_top]), int(xs_r[idx_top])
    comprimento = dist[y_top, x_top]
    if comprimento <= 0:
        return int(fallback)
    return int(round(float(comprimento)))


def processar_imagem(path: str, debug_dir: str | None = None) -> Dict[str, int | str]:
    dados = M01.processar_tratamentos_iniciais(path, debug_dir=debug_dir)

    ponto_base = dados["ponto_base"]
    mask_planta = dados["mask_planta_acima"]

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

    mask_folhas = M04.obter_mascara_folhas(mask_planta, dados["mask_caule"])
    area = M04.medir_area_foliar(mask_folhas)
    nro_folhas = M05.contar_folhas(mask_folhas)

    ys_planta, _ = np.where(mask_planta > 0)
    if ys_planta.size > 0:
        altura_base_mask = int(max(0, int(ponto_base[1]) - int(np.min(ys_planta))))
    else:
        altura_base_mask = int(altura)

    compr_total_raw = medir_comprimento_total(
        dados["skeleton"], ponto_base, fallback=altura_base_mask
    )
    compr_total = (
        int(altura_base_mask)
        if compr_total_raw < int(0.60 * max(altura_base_mask, 1))
        else int(compr_total_raw)
    )

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
