from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import List, Tuple

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


def _extrair_id_eucalipto(path_img: Path) -> int:
    nome = path_img.stem.lower()
    if nome.startswith("eucalipto"):
        sufixo = nome.replace("eucalipto", "", 1)
        if sufixo.isdigit():
            return int(sufixo)
    return 10**9


def _listar_imagens(img_dir: Path, limite: int = 10) -> List[Path]:
    imgs = sorted(img_dir.glob("*.jpg"), key=_extrair_id_eucalipto)
    return imgs[:limite]


def _desenhar_altura(path_img: Path) -> np.ndarray:
    dados = M01.processar_tratamentos_iniciais(str(path_img))
    altura = M02.medir_comprimento_vertical(
        dados["mask_planta_acima"],
        dados["ponto_base"],
        mask_objetos=dados["mask_objetos"],
        y_topo_tubete=int(dados["y_topo_tubete"]),
        img_bgr=dados["img_bgr"],
        x_centro_tubete=int(dados["x_centro_tubete"]),
        mask_caule=dados["mask_caule"],
    )
    x_base, y_base, y_top = M02.obter_pontos_altura_vertical(
        dados["mask_planta_acima"],
        dados["ponto_base"],
        mask_objetos=dados["mask_objetos"],
        y_topo_tubete=int(dados["y_topo_tubete"]),
        img_bgr=dados["img_bgr"],
        x_centro_tubete=int(dados["x_centro_tubete"]),
        mask_caule=dados["mask_caule"],
    )

    out = dados["img_bgr"].copy()
    cv2.line(out, (int(x_base), int(y_base)), (int(x_base), int(y_top)), (0, 0, 255), 6)
    cv2.circle(out, (int(x_base), int(y_base)), 10, (0, 255, 255), -1)
    cv2.circle(out, (int(x_base), int(y_top)), 10, (255, 255, 0), -1)
    cv2.putText(
        out,
        f"{path_img.stem} | Altura={int(altura)}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (255, 255, 255),
        3,
    )
    return out


def _montar_grid(
    imagens: List[np.ndarray],
    cols: int = 5,
    tile_size: Tuple[int, int] = (640, 480),
    margem: int = 20,
) -> np.ndarray:
    if not imagens:
        raise ValueError("Nenhuma imagem para montar o grid.")

    tile_w, tile_h = tile_size
    rows = int(np.ceil(len(imagens) / float(cols)))

    canvas_h = margem + rows * (tile_h + margem)
    canvas_w = margem + cols * (tile_w + margem)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(imagens):
                break
            img = cv2.resize(imagens[idx], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            y0 = margem + r * (tile_h + margem)
            x0 = margem + c * (tile_w + margem)
            canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = img
            idx += 1
    return canvas


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera um grid com linha de altura para 10 imagens.")
    parser.add_argument(
        "--img-dir",
        type=str,
        default="Dataset_Projeto1/_Eucalipto_Escolhidos1",
        help="Pasta com as imagens.",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="debug_saida/altura_10_juntas.png",
        help="Arquivo PNG de saida.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    img_dir = Path(args.img_dir)
    out_file = Path(args.out_file)

    paths = _listar_imagens(img_dir, limite=10)
    if len(paths) == 0:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em: {img_dir}")

    overlays = [_desenhar_altura(p) for p in paths]
    grid = _montar_grid(overlays, cols=5, tile_size=(640, 480), margem=20)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), grid)
    print(f"Imagem gerada: {out_file}")


if __name__ == "__main__":
    main()
