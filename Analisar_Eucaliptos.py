"""
Analise final dos eucaliptos.

Este e o script simples para rodar pelo VSCode. Ele chama a pipeline pronta para
os eucaliptos 1 a 10, salva um unico CSV em `resultados/resultados_eucaliptos.csv`
e imprime no terminal o MAPE das cinco imagens que tem referencia do professor.

O arquivo gerado fica no formato pedido para entrega: uma linha por eucalipto e
as colunas de altura vertical, comprimento avancado, diametro, area foliar e
numero de folhas.
"""

from __future__ import annotations

import csv
import importlib.util
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset_Projeto1" / "_Eucalipto_Escolhidos1"
SCRIPT_PIPELINE = BASE_DIR / "scripts_cv" / "06_pipeline_csv.py"
CSV_SAIDA = BASE_DIR / "resultados_eucaliptos.csv"

COLUNAS = [
    "Altura_vertical",
    "Comprimento_avançado",
    "Diâmetro_Coleto",
    "Area_Foliar",
    "N_Folhas",
]

REFERENCIAS_PROFESSOR = {
    1: {
        "Altura_vertical": 772,
        "Comprimento_avançado": 697,
        "Diâmetro_Coleto": 12,
        "Area_Foliar": 62484,
        "N_Folhas": [11, 12],
    },
    2: {
        "Altura_vertical": 1179,
        "Comprimento_avançado": 961,
        "Diâmetro_Coleto": 19,
        "Area_Foliar": 217423,
        "N_Folhas": [11, 12, 13],
    },
    3: {
        "Altura_vertical": 1107,
        "Comprimento_avançado": 1340,
        "Diâmetro_Coleto": 21,
        "Area_Foliar": 179931,
        "N_Folhas": [11, 12],
    },
    4: {
        "Altura_vertical": 794,
        "Comprimento_avançado": 630,
        "Diâmetro_Coleto": 14,
        "Area_Foliar": 43952,
        "N_Folhas": [13, 14],
    },
    5: {
        "Altura_vertical": 269,
        "Comprimento_avançado": 75,
        "Diâmetro_Coleto": 16,
        "Area_Foliar": 28524,
        "N_Folhas": [4],
    },
}


def carregar_pipeline() -> Any:
    spec = importlib.util.spec_from_file_location("pipeline_csv", SCRIPT_PIPELINE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nao foi possivel carregar {SCRIPT_PIPELINE}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def id_eucalipto(path: Path) -> int:
    match = re.fullmatch(r"Eucalipto(\d+)", path.stem, flags=re.IGNORECASE)
    if not match:
        return 10**9
    return int(match.group(1))


def erro_percentual(medido: float, referencia: float) -> float:
    return abs(float(medido) - float(referencia)) * 100.0 / abs(float(referencia))


def erro_percentual_melhor_ref(medido: float, referencias: float | list[int]) -> float:
    if isinstance(referencias, list):
        return min(erro_percentual(medido, ref) for ref in referencias)
    return erro_percentual(medido, referencias)


def imprimir_mape(resultados: list[dict[str, int]]) -> None:
    print("\nMAPE - Eucaliptos 1 a 5", flush=True)
    for coluna in COLUNAS:
        erros = []
        for linha in resultados:
            img_id = int(linha["Eucalipto"])
            if img_id not in REFERENCIAS_PROFESSOR:
                continue
            erros.append(
                erro_percentual_melhor_ref(
                    float(linha[coluna]),
                    REFERENCIAS_PROFESSOR[img_id][coluna],
                )
            )
        if erros:
            mape = sum(erros) / len(erros)
            print(f"{coluna}: {mape:.2f}%", flush=True)


def main() -> None:
    print("Iniciando analise dos eucaliptos...", flush=True)
    print(f"Dataset: {DATASET_DIR}", flush=True)
    print(f"CSV de saida: {CSV_SAIDA}", flush=True)

    pipeline = carregar_pipeline()
    imagens = sorted(DATASET_DIR.glob("*.jpg"), key=id_eucalipto)
    if not imagens:
        raise FileNotFoundError(f"Nenhuma imagem .jpg encontrada em {DATASET_DIR}")

    resultados = []
    for imagem in imagens:
        print(f"Processando {imagem.name}...", flush=True)
        medidas = pipeline.processar_imagem(str(imagem))
        linha = {
            "Eucalipto": id_eucalipto(imagem),
            "Altura_vertical": int(medidas["Altura Vert."]),
            "Comprimento_avançado": int(medidas["Compr Total"]),
            "Diâmetro_Coleto": int(medidas["Diâmetro"]),
            "Area_Foliar": int(medidas["Área"]),
            "N_Folhas": int(medidas["Nro Folhas"]),
        }
        resultados.append(linha)
        print(
            "OK "
            f"Eucalipto{linha['Eucalipto']}: "
            f"Altura={linha['Altura_vertical']} "
            f"Compr={linha['Comprimento_avançado']} "
            f"Diam={linha['Diâmetro_Coleto']} "
            f"Area={linha['Area_Foliar']} "
            f"Folhas={linha['N_Folhas']}",
            flush=True,
        )

    CSV_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    with CSV_SAIDA.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=["Eucalipto", *COLUNAS])
        escritor.writeheader()
        escritor.writerows(resultados)

    print(f"\nCSV salvo em: {CSV_SAIDA}", flush=True)
    imprimir_mape(resultados)


if __name__ == "__main__":
    main()
