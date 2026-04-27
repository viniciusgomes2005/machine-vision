from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np


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
M06 = _load_module(BASE_DIR / "06_pipeline_csv.py", "m06_pipeline")
MREF = _load_module(BASE_DIR.parent / "reutilizaveis.py", "m_reutilizaveis")


def _salvar(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _overlay_mask(img: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.35) -> np.ndarray:
    out = img.copy()
    color_img = np.zeros_like(img)
    color_img[:, :] = color
    m = (mask > 0).astype(np.uint8) * 255
    blend = cv2.addWeighted(out, 1.0, color_img, alpha, 0)
    out[m > 0] = blend[m > 0]
    return out


def _extrair_id_eucalipto(path_img: str) -> int | None:
    nome = Path(path_img).stem.lower()
    if not nome.startswith("eucalipto"):
        return None
    sufixo = nome.replace("eucalipto", "", 1)
    if not sufixo.isdigit():
        return None
    return int(sufixo)


def _imprimir_erros_professor(path_img: str, medidas: dict) -> None:
    img_id = _extrair_id_eucalipto(path_img)
    if img_id is None or img_id not in MREF.REFERENCIAS_EUCALIPTO:
        print("Sem referencia do professor para esta imagem (erro %/MAPE nao calculados).")
        return

    ref = MREF.REFERENCIAS_EUCALIPTO[img_id]
    metricas = ["Altura Vert.", "Compr Total", "Diâmetro", "Área", "Nro Folhas"]

    erros = []
    print("Erro percentual por metrica (referencia professor):")
    for metrica in metricas:
        valor = float(medidas[metrica])
        referencia = ref[metrica]
        err = MREF.erro_percentual_com_referencias_aceitas(valor, referencia)
        erros.append(float(err))
        print(f"- {metrica}: {err:.2f}%")

    mape_img = float(np.mean(np.array(erros, dtype=np.float32)))
    print(f"MAPE (esta imagem): {mape_img:.2f}%")


def processar_debug(path_img: str, out_dir: str) -> None:
    out = Path(out_dir)
    dados = M01.processar_tratamentos_iniciais(path_img, debug_dir=str(out))

    img = dados["img_bgr"]
    mask_folhas = M04.obter_mascara_folhas(dados["mask_planta_acima"], dados["mask_caule"])

    altura = M02.medir_comprimento_vertical(
        dados["mask_planta_acima"],
        dados["ponto_base"],
        mask_objetos=dados["mask_objetos"],
        y_topo_tubete=int(dados["y_topo_tubete"]),
        img_bgr=dados["img_bgr"],
        x_centro_tubete=int(dados["x_centro_tubete"]),
        mask_caule=dados["mask_caule"],
    )
    x_base_alt, y_base_alt, y_top_alt = M02.obter_pontos_altura_vertical(
        dados["mask_planta_acima"],
        dados["ponto_base"],
        mask_objetos=dados["mask_objetos"],
        y_topo_tubete=int(dados["y_topo_tubete"]),
        img_bgr=dados["img_bgr"],
        x_centro_tubete=int(dados["x_centro_tubete"]),
        mask_caule=dados["mask_caule"],
    )
    x_ref_diam = int(round((int(dados["ponto_base"][0]) + int(dados["x_centro_tubete"])) / 2.0))
    diametro, info_diam = M03.medir_diametro_coleto_debug(
        dados["mask_planta_acima"],
        int(dados["y_topo_tubete"]),
        x_ref_diam,
        mask_caule=dados["mask_caule"],
    )
    n_folhas = M05.contar_folhas(mask_folhas)
    ys_planta, _ = np.where(dados["mask_planta_acima"] > 0)
    if ys_planta.size > 0:
        altura_base_mask = int(max(0, int(dados["ponto_base"][1]) - int(np.min(ys_planta))))
    else:
        altura_base_mask = int(altura)

    compr_total_raw = M06.medir_comprimento_total(
        dados["skeleton"],
        dados["ponto_base"],
        fallback=altura_base_mask,
    )
    compr_total = int(altura_base_mask) if compr_total_raw < int(0.60 * max(altura_base_mask, 1)) else int(compr_total_raw)

    _salvar(out / "original.png", img)
    _salvar(out / "mask_fundo.png", dados["mask_fundo"])
    _salvar(out / "mask_objetos.png", dados["mask_objetos"])
    _salvar(out / "mask_tubete.png", dados["mask_tubete"])
    _salvar(out / "mask_planta_acima.png", dados["mask_planta_acima"])
    _salvar(out / "skeleton.png", dados["skeleton"])
    _salvar(out / "mask_caule.png", dados["mask_caule"])
    _salvar(out / "mask_folhas.png", mask_folhas)

    overlay_altura = img.copy()
    cv2.line(
        overlay_altura,
        (int(x_base_alt), int(y_base_alt)),
        (int(x_base_alt), int(y_top_alt)),
        (0, 0, 255),
        2,
    )
    cv2.circle(overlay_altura, (int(x_base_alt), int(y_base_alt)), 5, (0, 255, 255), -1)
    cv2.putText(overlay_altura, f"Altura Vert: {altura}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(overlay_altura, f"Compr Total: {compr_total}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 240, 50), 2)
    _salvar(out / "overlay_altura_vertical.png", overlay_altura)

    overlay_diam = img.copy()
    if info_diam["y"] is not None:
        y = int(info_diam["y"])
        x0 = int(info_diam["x0"])
        x1 = int(info_diam["x1"])
        cv2.line(overlay_diam, (x0, y), (x1, y), (0, 255, 255), 2)
    cv2.putText(overlay_diam, f"Diametro: {diametro}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    _salvar(out / "overlay_diametro.png", overlay_diam)

    overlay_folhas = _overlay_mask(img, mask_folhas, color=(0, 255, 0), alpha=0.45)
    cv2.putText(overlay_folhas, f"Nro Folhas: {n_folhas}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 0), 2)
    _salvar(out / "overlay_folhas.png", overlay_folhas)

    # Debug de candidatos de base do caule (quando disponivel).
    y_limite = int(dados.get("y_limite_planta", int(dados["y_topo_tubete"]) - 5))
    overlay_cands = img.copy()
    cv2.line(overlay_cands, (0, int(dados["y_topo_tubete"])), (overlay_cands.shape[1] - 1, int(dados["y_topo_tubete"])), (0, 0, 255), 2)
    cv2.line(overlay_cands, (0, y_limite), (overlay_cands.shape[1] - 1, y_limite), (0, 255, 255), 2)
    if hasattr(M02, "_coletar_candidatos_base"):
        try:
            cands = M02._coletar_candidatos_base(dados["mask_planta_acima"], int(dados["y_topo_tubete"]))
            for xc, yc, _w in cands:
                cv2.circle(overlay_cands, (int(xc), int(yc)), 2, (255, 255, 0), -1)
        except Exception:
            pass
    _salvar(out / "overlay_candidatos_base.png", overlay_cands)

    print(f"Debug salvo em: {out}")
    print(f"Altura Vert.: {altura}")
    print(f"Compr Total: {compr_total}")
    print(f"Diametro: {diametro}")
    print(f"Area: {M04.medir_area_foliar(mask_folhas)}")
    print(f"Nro Folhas: {n_folhas}")

    medidas = {
        "Altura Vert.": altura,
        "Compr Total": compr_total,
        "Diâmetro": diametro,
        "Área": M04.medir_area_foliar(mask_folhas),
        "Nro Folhas": n_folhas,
    }
    _imprimir_erros_professor(path_img, medidas)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug de uma unica imagem")
    parser.add_argument("--img", type=str, required=True, help="Caminho da imagem")
    parser.add_argument("--out-dir", type=str, default="debug_saida/uma_imagem", help="Pasta de saida")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    processar_debug(args.img, args.out_dir)


if __name__ == "__main__":
    main()
