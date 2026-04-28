"""
Analise final dos pinheiros.

Este script e igual ao dos eucaliptos na ideia: chama a pipeline pronta para
cada imagem de pinheiro e salva um unico CSV na raiz do projeto. Como nao temos
referencias oficiais do professor para pinheiros neste arquivo, ele apenas gera
as medidas finais, sem calcular MAPE.
"""

from __future__ import annotations

import csv
import importlib.util
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset_Projeto1" / "_Pinheiro_Escolhidos1"
SCRIPT_PIPELINE = BASE_DIR / "scripts_cv" / "06_pipeline_csv.py"
CSV_SAIDA = BASE_DIR / "resultados_pinheiros.csv"

COLUNAS = [
    "Altura_vertical",
    "Comprimento_avançado",
    "Diâmetro_Coleto",
    "Area_Foliar",
    "N_Folhas",
]


def carregar_pipeline() -> Any:
    spec = importlib.util.spec_from_file_location("pipeline_csv", SCRIPT_PIPELINE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nao foi possivel carregar {SCRIPT_PIPELINE}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def id_pinheiro(path: Path) -> int:
    match = re.fullmatch(r"Pinheiro(\d+)", path.stem, flags=re.IGNORECASE)
    if not match:
        return 10**9
    return int(match.group(1))


def main() -> None:
    print("Iniciando analise dos pinheiros...", flush=True)
    print(f"Dataset: {DATASET_DIR}", flush=True)
    print(f"CSV de saida: {CSV_SAIDA}", flush=True)

    pipeline = carregar_pipeline()
    imagens = sorted(DATASET_DIR.glob("*.jpg"), key=id_pinheiro)
    if not imagens:
        raise FileNotFoundError(f"Nenhuma imagem .jpg encontrada em {DATASET_DIR}")

    resultados = []
    for imagem in imagens:
        print(f"Processando {imagem.name}...", flush=True)
        medidas = pipeline.processar_imagem(str(imagem))
        linha = {
            "Pinheiro": id_pinheiro(imagem),
            "Altura_vertical": int(medidas["Altura Vert."]),
            "Comprimento_avançado": int(medidas["Compr Total"]),
            "Diâmetro_Coleto": int(medidas["Diâmetro"]),
            "Area_Foliar": int(medidas["Área"]),
            "N_Folhas": int(medidas["Nro Folhas"]),
        }
        resultados.append(linha)
        print(
            "OK "
            f"Pinheiro{linha['Pinheiro']}: "
            f"Altura={linha['Altura_vertical']} "
            f"Compr={linha['Comprimento_avançado']} "
            f"Diam={linha['Diâmetro_Coleto']} "
            f"Area={linha['Area_Foliar']} "
            f"Folhas={linha['N_Folhas']}",
            flush=True,
        )

    with CSV_SAIDA.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=["Pinheiro", *COLUNAS])
        escritor.writeheader()
        escritor.writerows(resultados)

    print(f"\nCSV salvo em: {CSV_SAIDA}", flush=True)


if __name__ == "__main__":
    main()
