"""
Analise final dos eucaliptos.

Por padrao roda nas imagens de treino em Dataset_Projeto1/_Eucalipto_Escolhidos1.
Tambem aceita outra pasta via --dataset-dir, por exemplo Conjunto_VALIDACAO, e
pode comparar o CSV gerado com uma planilha de referencia via --referencias-xlsx.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR_PADRAO = BASE_DIR / "Dataset_Projeto1" / "_Eucalipto_Escolhidos1"
SCRIPT_PIPELINE = BASE_DIR / "scripts_cv" / "06_pipeline_csv.py"
CSV_SAIDA_PADRAO = BASE_DIR / "resultados_eucaliptos.csv"
DEBUG_DIR_PADRAO = BASE_DIR / "diagnosticos_eucaliptos"

COLUNAS_RESULTADO = [
    "Img",
    "Altura Vert.",
    "Compr Total",
    "Diametro",
    "Area",
    "Nro Folhas",
]

METRICAS = ["Altura Vert.", "Compr Total", "Diametro", "Area", "Nro Folhas"]

LIMITES_RUBRICA_B = {
    "Altura Vert.": 2.0,
    "Compr Total": 5.0,
    "Diametro": 20.0,
    "Area": 5.0,
    "Nro Folhas": 10.0,
}

CRITERIOS_RUBRICA_C = {
    "Altura Vert.": (2.0, 0.65),
    "Diametro": (20.0, 0.65),
    "Area": (5.0, 0.50),
    "Nro Folhas": (10.0, 0.50),
}

REFERENCIAS_PROFESSOR = {
    1: {"Altura Vert.": 772, "Compr Total": 697, "Diametro": 12, "Area": 62484, "Nro Folhas": [11, 12]},
    2: {"Altura Vert.": 1179, "Compr Total": 961, "Diametro": 19, "Area": 217423, "Nro Folhas": [11, 12, 13]},
    3: {"Altura Vert.": 1107, "Compr Total": 1340, "Diametro": 21, "Area": 179931, "Nro Folhas": [11, 12]},
    4: {"Altura Vert.": 794, "Compr Total": 630, "Diametro": 14, "Area": 43952, "Nro Folhas": [13, 14]},
    5: {"Altura Vert.": 269, "Compr Total": 75, "Diametro": 16, "Area": 28524, "Nro Folhas": [4]},
}


def carregar_pipeline() -> Any:
    spec = importlib.util.spec_from_file_location("pipeline_csv", SCRIPT_PIPELINE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nao foi possivel carregar {SCRIPT_PIPELINE}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def id_eucalipto(path: Path | str) -> int:
    stem = Path(path).stem
    match = re.fullmatch(r"Eucalipto0*(\d+)", stem, flags=re.IGNORECASE)
    if not match:
        return 10**9
    return int(match.group(1))


def erro_percentual(medido: float, referencia: float) -> float:
    return abs(float(medido) - float(referencia)) * 100.0 / abs(float(referencia))


def parse_referencia_folhas(valor: Any) -> list[int]:
    if isinstance(valor, list):
        return [int(v) for v in valor]
    texto = str(valor).strip()
    numeros = [int(x) for x in re.findall(r"\d+", texto)]
    if " a " in f" {texto.lower()} " and len(numeros) >= 2:
        return list(range(numeros[0], numeros[1] + 1))
    return numeros


def escolher_melhor_referencia(medido: float, referencia: float | list[int]) -> float:
    if isinstance(referencia, list):
        return float(min(referencia, key=lambda ref: erro_percentual(medido, ref)))
    return float(referencia)


def carregar_referencias_xlsx(path: Path) -> dict[int, dict[str, float | list[int]]]:
    bruto = pd.read_excel(path, header=None)
    header_idx = None
    for idx, row in bruto.iterrows():
        valores = [str(v).strip() for v in row.tolist()]
        if "Img" in valores and "Altura Vert." in valores:
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"Nao encontrei cabecalho na planilha: {path}")

    dados = pd.read_excel(path, header=header_idx)
    dados = dados.dropna(axis=1, how="all")
    dados = dados.dropna(subset=["Img"])

    referencias: dict[int, dict[str, float | list[int]]] = {}
    for _, row in dados.iterrows():
        img_id = int(row["Img"])
        referencias[img_id] = {
            "Altura Vert.": float(row["Altura Vert."]),
            "Compr Total": float(row["Compr Total"]),
            "Diametro": float(row["Diâmetro"]),
            "Area": float(row["Área"]),
            "Nro Folhas": parse_referencia_folhas(row["Nro Folhas"]),
        }
    return referencias


def comparar_resultados(
    resultados: list[dict[str, int | str]],
    referencias: dict[int, dict[str, float | list[int]]],
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, bool], dict[str, dict[str, Any]]]:
    por_id = {id_eucalipto(str(row["Img"])): row for row in resultados}
    linhas: list[dict[str, Any]] = []
    erros_por_metrica: dict[str, list[float]] = {metrica: [] for metrica in METRICAS}

    for img_id in sorted(referencias):
        if img_id not in por_id:
            continue
        medido = por_id[img_id]
        for metrica in METRICAS:
            valor_medido = float(medido[metrica])
            ref_usada = escolher_melhor_referencia(valor_medido, referencias[img_id][metrica])
            erro = erro_percentual(valor_medido, ref_usada)
            erros_por_metrica[metrica].append(erro)
            linhas.append(
                {
                    "Img": f"Eucalipto{img_id:02d}",
                    "Metrica": metrica,
                    "Referencia": ref_usada,
                    "Medido": valor_medido,
                    "Erro_relativo_percentual": erro,
                }
            )

    mape = {
        metrica: sum(erros) / len(erros)
        for metrica, erros in erros_por_metrica.items()
        if erros
    }
    status = {metrica: mape[metrica] < LIMITES_RUBRICA_B[metrica] for metrica in mape}
    status_c: dict[str, dict[str, Any]] = {}
    for metrica, (limite, proporcao_minima) in CRITERIOS_RUBRICA_C.items():
        erros = erros_por_metrica[metrica]
        aprovadas = sum(1 for erro in erros if erro < limite)
        proporcao = aprovadas / len(erros) if erros else 0.0
        status_c[metrica] = {
            "aprovadas": aprovadas,
            "total": len(erros),
            "proporcao": proporcao,
            "limite_erro": limite,
            "proporcao_minima": proporcao_minima,
            "aprovado": proporcao >= proporcao_minima,
        }
    return linhas, mape, status, status_c


def normalizar_medidas(medidas: dict[str, Any]) -> dict[str, int | str]:
    def pegar(*nomes: str) -> Any:
        for nome in nomes:
            if nome in medidas:
                return medidas[nome]
        raise KeyError(f"Nenhuma das chaves encontrada: {nomes}. Chaves disponiveis: {list(medidas)}")

    return {
        "Img": str(medidas["Img"]),
        "Altura Vert.": int(medidas["Altura Vert."]),
        "Compr Total": int(medidas["Compr Total"]),
        "Diametro": int(pegar("Diâmetro", "DiÃ¢metro")),
        "Area": int(pegar("Área", "Ãrea")),
        "Nro Folhas": int(medidas["Nro Folhas"]),
    }


def salvar_csv(path: Path, linhas: list[dict[str, Any]], colunas: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=colunas)
        escritor.writeheader()
        escritor.writerows(linhas)


def imprimir_avaliacao(mape: dict[str, float], status_b: dict[str, bool], status_c: dict[str, dict[str, Any]]) -> None:
    print("\nRubrica C por proporcao de imagens", flush=True)
    c_ok = True
    for metrica in ["Altura Vert.", "Diametro", "Area", "Nro Folhas"]:
        dados = status_c[metrica]
        ok = bool(dados["aprovado"])
        c_ok = c_ok and ok
        texto = "OK" if ok else "NAO"
        print(
            f"{metrica}: {dados['aprovadas']}/{dados['total']} "
            f"({dados['proporcao'] * 100.0:.1f}%) com erro < {dados['limite_erro']:.2f}% "
            f"(min {dados['proporcao_minima'] * 100.0:.0f}%) -> {texto}",
            flush=True,
        )

    print("\nMAPE por metrica", flush=True)
    aprovados_b = 0
    for metrica in METRICAS:
        if metrica not in mape:
            continue
        ok = status_b[metrica]
        aprovados_b += int(ok)
        texto = "OK" if ok else "NAO"
        print(
            f"{metrica}: {mape[metrica]:.2f}% "
            f"(limite B < {LIMITES_RUBRICA_B[metrica]:.2f}%) -> {texto}",
            flush=True,
        )
    print(f"Criterios MAPE da rubrica B atingidos: {aprovados_b}/{len(mape)}", flush=True)
    print(f"Rubrica C atingida: {'SIM' if c_ok else 'NAO'}", flush=True)
    print(f"Rubrica B final: {'SIM' if c_ok and aprovados_b >= 3 else 'NAO'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analisa eucaliptos e calcula MAPE opcional.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR_PADRAO)
    parser.add_argument("--csv-saida", type=Path, default=CSV_SAIDA_PADRAO)
    parser.add_argument("--debug-dir", type=Path, default=DEBUG_DIR_PADRAO)
    parser.add_argument("--sem-debug", action="store_true")
    parser.add_argument("--referencias-xlsx", type=Path, default=None)
    parser.add_argument("--comparacao-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    csv_saida = args.csv_saida.resolve()
    debug_dir = None if args.sem_debug else args.debug_dir.resolve()

    print("Iniciando analise dos eucaliptos...", flush=True)
    print(f"Dataset: {dataset_dir}", flush=True)
    print(f"CSV de saida: {csv_saida}", flush=True)
    if debug_dir is not None:
        print(f"Diagnosticos: {debug_dir}", flush=True)

    pipeline = carregar_pipeline()
    imagens = sorted(dataset_dir.glob("*.jpg"), key=id_eucalipto)
    if not imagens:
        raise FileNotFoundError(f"Nenhuma imagem .jpg encontrada em {dataset_dir}")

    resultados: list[dict[str, int | str]] = []
    for imagem in imagens:
        print(f"Processando {imagem.name}...", flush=True)
        debug_imagem = debug_dir / imagem.stem if debug_dir is not None else None
        medidas = pipeline.processar_imagem(
            str(imagem),
            debug_dir=str(debug_imagem) if debug_imagem is not None else None,
        )
        linha = normalizar_medidas(medidas)
        resultados.append(linha)
        print(
            "OK "
            f"{linha['Img']}: "
            f"Altura={linha['Altura Vert.']} "
            f"Compr={linha['Compr Total']} "
            f"Diam={linha['Diametro']} "
            f"Area={linha['Area']} "
            f"Folhas={linha['Nro Folhas']}",
            flush=True,
        )

    salvar_csv(csv_saida, resultados, COLUNAS_RESULTADO)
    print(f"\nCSV salvo em: {csv_saida}", flush=True)

    referencias = carregar_referencias_xlsx(args.referencias_xlsx.resolve()) if args.referencias_xlsx else REFERENCIAS_PROFESSOR
    linhas_erro, mape, status_b, status_c = comparar_resultados(resultados, referencias)
    if linhas_erro:
        comparacao_csv = (
            args.comparacao_csv.resolve()
            if args.comparacao_csv is not None
            else csv_saida.with_name(csv_saida.stem + "_comparacao.csv")
        )
        salvar_csv(
            comparacao_csv,
            linhas_erro,
            ["Img", "Metrica", "Referencia", "Medido", "Erro_relativo_percentual"],
        )
        print(f"Comparacao salva em: {comparacao_csv}", flush=True)
        imprimir_avaliacao(mape, status_b, status_c)
    else:
        print("Sem imagens em comum com as referencias; MAPE nao calculado.", flush=True)


if __name__ == "__main__":
    main()
