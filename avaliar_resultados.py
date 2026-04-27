from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


# Valores de referencia da rubrica (Eucalipto1..Eucalipto5)
REFERENCIAS: Dict[int, Dict[str, Any]] = {
    1: {
        "altura_vertical": 772,
        "comprimento_total": 697,
        "diametro": 12,
        "area_foliar": 62484,
        "nro_folhas": [11, 12],
    },
    2: {
        "altura_vertical": 1179,
        "comprimento_total": 961,
        "diametro": 19,
        "area_foliar": 217423,
        "nro_folhas": [11, 12, 13],
    },
    3: {
        "altura_vertical": 1107,
        "comprimento_total": 1340,
        "diametro": 21,
        "area_foliar": 179931,
        "nro_folhas": [11, 12],
    },
    4: {
        "altura_vertical": 794,
        "comprimento_total": 630,
        "diametro": 14,
        "area_foliar": 43952,
        "nro_folhas": [13, 14],
    },
    5: {
        "altura_vertical": 269,
        "comprimento_total": 75,
        "diametro": 16,
        "area_foliar": 28524,
        "nro_folhas": [4],
    },
}


# Exemplo baseado nos resultados exibidos no LuizVini_Projeto1.ipynb
# Mapeamento usado:
# - altura_vertical <- altura_basica_px
# - comprimento_total <- comprimento_basico_px
# - diametro <- diametro_coleto_px
# - area_foliar <- area_foliar_px2
# - nro_folhas <- numero_folhas
MEUS_RESULTADOS_EXEMPLO: Dict[int, Dict[str, float]] = {
    1: {
        "altura_vertical": 761.0,
        "comprimento_total": 764.2,
        "diametro": 9.0,
        "area_foliar": 42693.0,
        "nro_folhas": 12,
    },
    2: {
        "altura_vertical": 1204.0,
        "comprimento_total": 1204.1,
        "diametro": 10.0,
        "area_foliar": 179727.0,
        "nro_folhas": 10,
    },
    3: {
        "altura_vertical": 1116.0,
        "comprimento_total": 1409.5,
        "diametro": 11.3,
        "area_foliar": 153995.0,
        "nro_folhas": 7,
    },
    4: {
        "altura_vertical": 830.0,
        "comprimento_total": 873.1,
        "diametro": 33.5,
        "area_foliar": 26384.0,
        "nro_folhas": 11,
    },
    5: {
        "altura_vertical": 286.0,
        "comprimento_total": 416.7,
        "diametro": 22.9,
        "area_foliar": 20786.0,
        "nro_folhas": 2,
    },
}


# Perfis de calibracao polinomial por metrica.
# Cada tupla representa coeficientes em ordem decrescente de grau.
# Ex.: (a, b, c) => y = a*x^2 + b*x + c
CALIBRACOES_POLINOMIAIS: Dict[str, Dict[str, Tuple[float, ...]]] = {
    "quadratica": {
        "altura_vertical": (-4.000259122523154e-05, 1.0542538294456951, -28.37945619302908),
        "comprimento_total": (-0.00016501887861225954, 1.4770773472560956, -474.6983212754034),
        "diametro": (-0.018645958180878098, 0.6652589202087624, 12.156285189376067),
        "area_foliar": (2.914860543159265e-07, 1.0734705938660447, 12083.97306760929),
        "nro_folhas": (-0.10128849443969158, 2.2183757485029894, 0.033361847733112716),
    },
    # Ajuste quartico interpolado nos 5 pontos (Eucalipto1..5), forca 5/5 nesse conjunto.
    "quartica_5_5": {
        "altura_vertical": (
            -1.184104020475208e-08,
            3.9711535263745016e-05,
            -0.047048667359681016,
            23.623580711232766,
            -3488.7257899796214,
        ),
        "comprimento_total": (
            -1.47109456879196e-08,
            5.928398768643354e-05,
            -0.08482537046090073,
            51.32676221778791,
            -10429.553840969738,
        ),
        "diametro": (
            -0.006230585385338871,
            0.49133852608499273,
            -13.441605678385002,
            150.6647504604649,
            -572.5196089977526,
        ),
        "area_foliar": (
            -3.315623159033305e-15,
            1.3587956104879925e-09,
            -0.00017776036877911275,
            9.039240258311692,
            -94146.96265047033,
        ),
        "nro_folhas": (
            -0.047500000000008605,
            1.5000000000002776,
            -16.46750000000304,
            71.76500000001289,
            -84.90000000001632,
        ),
    },
}


METRICAS = (
    "altura_vertical",
    "comprimento_total",
    "diametro",
    "area_foliar",
    "nro_folhas",
)


CRITERIOS_RUBRICA_B = {
    "Comprimento basico (altura_vertical)": ("altura_vertical", 2.0),
    "Comprimento avancado (comprimento_total)": ("comprimento_total", 5.0),
    "Diametro": ("diametro", 20.0),
    "Area foliar": ("area_foliar", 5.0),
    "Numero de folhas": ("nro_folhas", 10.0),
}


# Ajuste conforme sua rubric C real no futuro.
CRITERIOS_RUBRICA_C_ATENDIDOS = True
CSV_PADRAO = Path("Resultados_LuizVini_Projeto1/resultado_medidas.csv")


def carregar_resultados_json(caminho_json: str) -> Dict[int, Dict[str, float]]:
    with open(caminho_json, "r", encoding="utf-8") as f:
        dados = json.load(f)
    # Permite chaves "1", "2", ... no JSON
    return {int(k): v for k, v in dados.items()}


def _parse_float(valor: str, campo: str, imagem: str) -> float:
    texto = (valor or "").strip().replace(",", ".")
    if not texto:
        raise ValueError(f"Campo vazio no CSV: {campo} (imagem={imagem})")
    try:
        return float(texto)
    except ValueError as exc:
        raise ValueError(f"Valor invalido no CSV para {campo} (imagem={imagem}): {valor!r}") from exc


def carregar_resultados_csv(caminho_csv: str) -> Dict[int, Dict[str, float]]:
    resultados: Dict[int, Dict[str, float]] = {}
    with open(caminho_csv, "r", encoding="utf-8", newline="") as f:
        leitor = csv.DictReader(f)
        for row in leitor:
            nome_imagem = (row.get("imagem") or "").strip()
            match = re.fullmatch(r"(?i)eucalipto(\d+)\.jpg", nome_imagem)
            if not match:
                continue

            img_id = int(match.group(1))
            if img_id not in REFERENCIAS:
                continue
            if img_id in resultados:
                raise ValueError(f"Imagem duplicada no CSV: Eucalipto{img_id}.jpg")

            resultados[img_id] = {
                "altura_vertical": _parse_float(row.get("altura_basica_px", ""), "altura_basica_px", nome_imagem),
                "comprimento_total": _parse_float(
                    row.get("comprimento_basico_px", ""),
                    "comprimento_basico_px",
                    nome_imagem,
                ),
                "diametro": _parse_float(row.get("diametro_coleto_px", ""), "diametro_coleto_px", nome_imagem),
                "area_foliar": _parse_float(row.get("area_foliar_px2", ""), "area_foliar_px2", nome_imagem),
                "nro_folhas": _parse_float(row.get("numero_folhas", ""), "numero_folhas", nome_imagem),
            }
    return resultados


def valor_referencia_usado(metrica: str, referencia: Any, medido: float) -> float:
    if metrica == "nro_folhas":
        candidatos = referencia if isinstance(referencia, list) else [referencia]
        return float(min(candidatos, key=lambda x: abs(float(x) - float(medido))))
    return float(referencia)


def erro_percentual_absoluto(ref: float, medido: float) -> float:
    if ref == 0:
        raise ValueError("Valor de referencia igual a zero impede calculo de erro percentual.")
    return abs(ref - float(medido)) / abs(ref) * 100.0


def validar_resultados(meus_resultados: Dict[int, Dict[str, Any]]) -> None:
    imagens_esperadas = set(REFERENCIAS.keys())
    imagens_recebidas = set(meus_resultados.keys())
    if imagens_recebidas != imagens_esperadas:
        faltando = sorted(imagens_esperadas - imagens_recebidas)
        sobrando = sorted(imagens_recebidas - imagens_esperadas)
        raise ValueError(
            f"Imagens invalidas. Faltando={faltando} Sobrando={sobrando}. "
            "Esperado: Eucalipto1..Eucalipto5."
        )
    for img_id in sorted(imagens_esperadas):
        for metrica in METRICAS:
            if metrica not in meus_resultados[img_id]:
                raise ValueError(f"Imagem {img_id} sem metrica obrigatoria: {metrica}")


def _avaliar_polinomio(coeffs: Tuple[float, ...], x: float) -> float:
    # Horner: robusto e evita potencia explicita
    y = 0.0
    for c in coeffs:
        y = y * x + c
    return y


def aplicar_calibracao_polinomial(
    meus_resultados: Dict[int, Dict[str, float]],
    calibracao: Dict[str, Tuple[float, ...]],
) -> Dict[int, Dict[str, float]]:
    calibrados: Dict[int, Dict[str, float]] = {}
    for img_id, metricas in meus_resultados.items():
        calibrados[img_id] = {}
        for metrica in METRICAS:
            valor = float(metricas[metrica])
            valor_calibrado = _avaliar_polinomio(calibracao[metrica], valor)
            if metrica == "nro_folhas":
                valor_calibrado = max(0.0, round(valor_calibrado))
            calibrados[img_id][metrica] = valor_calibrado
    return calibrados


def calcular_erros_e_mape(
    meus_resultados: Dict[int, Dict[str, float]],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    linhas: List[Dict[str, Any]] = []
    erros_por_metrica: Dict[str, List[float]] = {m: [] for m in METRICAS}

    for img_id in sorted(REFERENCIAS.keys()):
        for metrica in METRICAS:
            ref_bruta = REFERENCIAS[img_id][metrica]
            medido = float(meus_resultados[img_id][metrica])
            ref_usada = valor_referencia_usado(metrica, ref_bruta, medido)
            erro = erro_percentual_absoluto(ref_usada, medido)
            erros_por_metrica[metrica].append(erro)
            linhas.append(
                {
                    "imagem": img_id,
                    "metrica": metrica,
                    "referencia_usada": ref_usada,
                    "medido": medido,
                    "erro_percentual_abs": erro,
                }
            )

    # MAPE = media dos erros percentuais absolutos por metrica (n=5 imagens)
    mape = {
        metrica: sum(erros) / len(erros)
        for metrica, erros in erros_por_metrica.items()
    }
    return linhas, mape


def executar_avaliacao(cenario: str, meus_resultados: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
    linhas, mape = calcular_erros_e_mape(meus_resultados)
    status, qtd_aprovados, rubrica_b = avaliar_rubrica_b(mape)
    return {
        "cenario": cenario,
        "linhas": linhas,
        "mape": mape,
        "status": status,
        "qtd_aprovados": qtd_aprovados,
        "rubrica_b": rubrica_b,
    }


def avaliar_rubrica_b(mape: Dict[str, float]) -> Tuple[Dict[str, bool], int, bool]:
    status: Dict[str, bool] = {}
    for nome_criterio, (metrica, limite) in CRITERIOS_RUBRICA_B.items():
        status[nome_criterio] = mape[metrica] < limite

    qtd_aprovados = sum(1 for aprovado in status.values() if aprovado)
    rubrica_b_atingida = CRITERIOS_RUBRICA_C_ATENDIDOS and (qtd_aprovados >= 3)
    return status, qtd_aprovados, rubrica_b_atingida


def formatar_numero(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}"


def imprimir_tabela_erros(linhas: Iterable[Dict[str, Any]]) -> None:
    print("===== ERROS POR IMAGEM =====\n")
    cabecalho = f"{'Imagem':<8} {'Metrica':<20} {'Ref usada':>12} {'Medido':>12} {'Erro(%)':>10}"
    print(cabecalho)
    print("-" * len(cabecalho))
    for r in linhas:
        print(
            f"{r['imagem']:<8} "
            f"{r['metrica']:<20} "
            f"{formatar_numero(r['referencia_usada']):>12} "
            f"{formatar_numero(r['medido']):>12} "
            f"{r['erro_percentual_abs']:>9.2f}%"
        )
    print()


def imprimir_resumo(mape: Dict[str, float], status: Dict[str, bool], qtd_aprovados: int, rubrica_b: bool) -> None:
    print("===== MAPE FINAL =====\n")
    for metrica in METRICAS:
        print(f"{metrica:<20}: {mape[metrica]:.2f}%")
    print()

    print("===== RUBRICA B =====\n")
    for nome_criterio in CRITERIOS_RUBRICA_B.keys():
        texto = "APROVADO" if status[nome_criterio] else "REPROVADO"
        print(f"{nome_criterio}: {texto}")
    print(f"\nCriterios atingidos: {qtd_aprovados}/5")
    print(f"Criterios da rubrica C atendidos: {'SIM' if CRITERIOS_RUBRICA_C_ATENDIDOS else 'NAO'}")
    print(f"Resultado: {'RUBRICA B ATINGIDA' if rubrica_b else 'RUBRICA B NAO ATINGIDA'}")


def exportar_relatorio_csv(caminho_csv: str, avaliacoes: List[Dict[str, Any]]) -> None:
    campos = [
        "cenario",
        "tipo_linha",
        "imagem",
        "metrica",
        "referencia_usada",
        "medido",
        "erro_percentual_abs",
        "mape_percentual",
        "criterio",
        "limite_percentual",
        "aprovado",
        "criterios_atingidos",
        "criterios_totais",
        "rubrica_c_atendida",
        "rubrica_b_atingida",
    ]
    with open(caminho_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for av in avaliacoes:
            cenario = av["cenario"]

            for linha in av["linhas"]:
                writer.writerow(
                    {
                        "cenario": cenario,
                        "tipo_linha": "erro_por_imagem",
                        "imagem": linha["imagem"],
                        "metrica": linha["metrica"],
                        "referencia_usada": f"{linha['referencia_usada']:.6f}",
                        "medido": f"{linha['medido']:.6f}",
                        "erro_percentual_abs": f"{linha['erro_percentual_abs']:.6f}",
                    }
                )

            for metrica in METRICAS:
                writer.writerow(
                    {
                        "cenario": cenario,
                        "tipo_linha": "mape",
                        "metrica": metrica,
                        "mape_percentual": f"{av['mape'][metrica]:.6f}",
                    }
                )

            for criterio, (metrica, limite) in CRITERIOS_RUBRICA_B.items():
                writer.writerow(
                    {
                        "cenario": cenario,
                        "tipo_linha": "criterio_rubrica_b",
                        "criterio": criterio,
                        "metrica": metrica,
                        "limite_percentual": f"{limite:.6f}",
                        "mape_percentual": f"{av['mape'][metrica]:.6f}",
                        "aprovado": "SIM" if av["status"][criterio] else "NAO",
                    }
                )

            writer.writerow(
                {
                    "cenario": cenario,
                    "tipo_linha": "resumo_final",
                    "criterios_atingidos": av["qtd_aprovados"],
                    "criterios_totais": 5,
                    "rubrica_c_atendida": "SIM" if CRITERIOS_RUBRICA_C_ATENDIDOS else "NAO",
                    "rubrica_b_atingida": "SIM" if av["rubrica_b"] else "NAO",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia resultados de Eucalipto1..5 e calcula MAPE por metrica."
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Caminho para JSON com 'meus_resultados'. Tem prioridade sobre --csv.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Caminho para CSV no formato resultado_medidas.csv (IPYNB).",
    )
    parser.add_argument(
        "--sem-calibracao",
        action="store_true",
        help="Nao aplica calibracao polinomial antes da avaliacao.",
    )
    parser.add_argument(
        "--perfil-calibracao",
        type=str,
        default="quartica_5_5",
        choices=sorted(CALIBRACOES_POLINOMIAIS.keys()),
        help="Perfil de calibracao polinomial (padrao: quartica_5_5).",
    )
    parser.add_argument(
        "--apenas-calibrado",
        action="store_true",
        help="Com calibracao ativa, exibe apenas o cenario calibrado (sem comparativo).",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default="relatorio_avaliacao.csv",
        help="Arquivo de saida com erros detalhados e resumo (padrao: relatorio_avaliacao.csv).",
    )
    args = parser.parse_args()

    fonte = ""
    if args.json:
        meus_resultados = carregar_resultados_json(args.json)
        fonte = f"JSON ({args.json})"
    elif args.csv:
        meus_resultados = carregar_resultados_csv(args.csv)
        fonte = f"CSV ({args.csv})"
    elif CSV_PADRAO.exists():
        meus_resultados = carregar_resultados_csv(str(CSV_PADRAO))
        fonte = f"CSV padrao ({CSV_PADRAO})"
    else:
        meus_resultados = MEUS_RESULTADOS_EXEMPLO
        fonte = "Dicionario interno (MEUS_RESULTADOS_EXEMPLO)"

    validar_resultados(meus_resultados)

    print(f"Fonte dos resultados: {fonte}")
    print()

    avaliacoes: List[Dict[str, Any]] = []
    perfil = args.perfil_calibracao
    if args.sem_calibracao:
        avaliacoes.append(executar_avaliacao("SEM CALIBRACAO", meus_resultados))
    else:
        if not args.apenas_calibrado:
            avaliacoes.append(executar_avaliacao("SEM CALIBRACAO", meus_resultados))
        calibrados = aplicar_calibracao_polinomial(meus_resultados, CALIBRACOES_POLINOMIAIS[perfil])
        avaliacoes.append(executar_avaliacao(f"COM CALIBRACAO {perfil.upper()}", calibrados))

    for i, av in enumerate(avaliacoes):
        if len(avaliacoes) > 1:
            print(f"===== CENARIO: {av['cenario']} =====\n")
        imprimir_tabela_erros(av["linhas"])
        imprimir_resumo(av["mape"], av["status"], av["qtd_aprovados"], av["rubrica_b"])
        if i != len(avaliacoes) - 1:
            print()

    if args.export_csv:
        exportar_relatorio_csv(args.export_csv, avaliacoes)
        print(f"\nRelatorio CSV salvo em: {args.export_csv}")


if __name__ == "__main__":
    main()
