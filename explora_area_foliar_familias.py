from pathlib import Path
import os
import sys

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import explora_pinheiro_23 as base


OUT_DIR = Path(__file__).parent / "_dev_debug"
OUT_DIR.mkdir(exist_ok=True)


def height_info(seg):
    ys, _ = np.where(seg["mask_corpo"])
    y_top = int(ys.min()) if len(ys) else 0
    y_base = int(ys.max()) if len(ys) else 0
    return y_top, y_base, max(1, y_base - y_top)


def top_region(seg, rel_height, half_width):
    mask = np.zeros_like(seg["mask_corpo"], dtype=bool)
    y_top, _, height = height_info(seg)
    y1 = min(mask.shape[0], int(y_top + rel_height * height))
    xc = seg["x_centro_tubete"]
    x0 = max(0, xc - half_width)
    x1 = min(mask.shape[1], xc + half_width)
    mask[y_top:y1, x0:x1] = True
    return mask


def row_lateral_band(mask, axis, y_top, y_base, top_dx, low_dx):
    h, w = mask.shape
    xx = np.arange(w, dtype=np.float32)[None, :]
    out = np.zeros_like(mask, dtype=bool)
    height = max(1, y_base - y_top)
    for y in range(h):
        if y < y_top:
            dx = top_dx
        elif y > y_base:
            dx = low_dx
        else:
            t = (y - y_top) / height
            dx = (1.0 - t) * top_dx + t * low_dx
        out[y] = np.abs(xx[0] - axis[y]) >= dx
    return mask & out


def row_lateral_gate(shape, axis, y_top, y_base, top_dx, low_dx):
    h, w = shape
    xx = np.arange(w, dtype=np.float32)[None, :]
    out = np.zeros(shape, dtype=bool)
    height = max(1, y_base - y_top)
    for y in range(h):
        if y < y_top:
            dx = top_dx
        elif y > y_base:
            dx = low_dx
        else:
            t = (y - y_top) / height
            dx = (1.0 - t) * top_dx + t * low_dx
        out[y] = np.abs(xx[0] - axis[y]) >= dx
    return out


def hard_vase_cut(mask, seg, y_below, x_half):
    out = mask.copy()
    h, w = out.shape
    y0 = max(0, seg["y_topo_tubete"] - 8)
    y1 = min(h, seg["y_topo_tubete"] + y_below)
    x0 = max(0, seg["x_centro_tubete"] - x_half)
    x1 = min(w, seg["x_centro_tubete"] + x_half)
    cut = np.zeros_like(out, dtype=bool)
    cut[y0:y1, x0:x1] = True
    out[cut] = False
    return out, cut


def prune_stem_by_support(stem, support_h, support_radius, min_count):
    if not np.any(stem):
        return stem
    u8 = stem.astype(np.uint8)
    k = np.ones((support_h, 2 * support_radius + 1), np.uint8)
    # Count support above each pixel; shift kernel upward by eroding top-heavy continuity.
    support = cv2.filter2D(u8, -1, k, anchor=(support_radius, support_h - 1), borderType=cv2.BORDER_CONSTANT)
    keep = (~stem) | (support >= min_count)
    return stem & keep


def estimate_branch_y(seg, stem, width_factor):
    body = seg["mask_corpo"]
    y_top, y_base, _ = height_info(seg)
    xc = seg["x_centro_tubete"]
    widths = []
    ys = []
    for y in range(y_top, y_base + 1):
        row = body[y]
        if not row[xc] and not np.any(row):
            continue
        xs = np.where(row)[0]
        if len(xs) == 0:
            continue
        widths.append(xs.max() - xs.min() + 1)
        ys.append(y)
    if not widths:
        return y_base
    widths = np.array(widths, dtype=np.float32)
    ys = np.array(ys, dtype=np.int32)
    trunk_w = np.median(widths[: max(3, len(widths) // 8)])
    idxs = np.where(widths >= trunk_w * width_factor)[0]
    return int(ys[idxs[0]]) if len(idxs) else int(ys[min(len(ys) - 1, len(ys) // 3)])


def make_fine_mask(img_rgb, seg, hmin, hmax, smin, vmin, gbmin, grmin, dilate_k):
    params = {
        "fine_h_min": hmin,
        "fine_h_max": hmax,
        "fine_s_min": smin,
        "fine_v_min": vmin,
        "fine_gb_min": gbmin,
        "fine_gr_min": grmin,
        "below_tube_cut": 8,
        "fine_close_k": (1, 1),
        "fine_dilate_k": dilate_k,
        "fine_area_min": 1,
    }
    return base.mascara_fina(img_rgb, seg, params)


def resolve_pinheiro_variant(img_rgb, seg, cfg):
    body = seg["mask_corpo"]
    fine = make_fine_mask(img_rgb, seg, cfg["hmin"], cfg["hmax"], cfg["smin"], cfg["vmin"], cfg["gbmin"], cfg["grmin"], cfg["fine_dilate_k"])

    stem_params = {
        "stem_open_k": cfg["stem_open_k"],
        "stem_close_k": cfg["stem_close_k"],
        "stem_dilate_k": cfg["stem_dilate_k"],
        "stem_height": 280,
        "stem_x_half": 55,
        "stem_area_min": 10,
        "stem_y_extra": 16,
        "basic_half": cfg["basic_half"],
        "stem_top_ignore_rel": cfg["stem_top_ignore_rel"],
    }
    stem, vertical = base.mascara_caule_pinheiro(body, seg["ponto_base"], stem_params)
    if cfg.get("support_prune", False):
        stem = prune_stem_by_support(
            stem,
            cfg.get("support_h", 45),
            cfg.get("support_radius", 5),
            cfg.get("support_min", 10),
        )

    dist = base.distancia_ao_stem(stem)
    y_top, y_base, height = height_info(seg)
    branch_y = estimate_branch_y(seg, stem, cfg["branch_width_factor"])
    top = top_region(seg, cfg["top_rel"], cfg["top_half"])
    top_by_branch = np.zeros_like(body, dtype=bool)
    top_by_branch[y_top:branch_y, :] = True
    top = top | (top_by_branch & np.abs(np.arange(body.shape[1])[None, :] - seg["x_centro_tubete"]) <= cfg["top_half"])

    axis = base.eixo_x_por_linha(stem if np.any(stem) else body, seg["x_centro_tubete"])
    side_seed = base.mascara_lateral_por_linha(fine | body, axis, cfg["side_dx"])
    seeded = base.componentes_que_tocam_seed((fine | body), side_seed, area_min=cfg["seed_area"])

    mode = cfg["mode"]
    if mode == "top_plus_fine":
        cand = top | (fine & (dist >= cfg["dist_mid"])) | (fine & (np.arange(body.shape[0])[:, None] < branch_y))
    elif mode == "seeded_components":
        cand = top | (seeded & (dist >= cfg["dist_mid"]))
    elif mode == "strict_center_ban":
        lateral = base.mascara_lateral_por_linha(fine | body, axis, cfg["side_dx"])
        cand = top | (lateral & (dist >= cfg["dist_mid"]))
    elif mode == "hybrid":
        lateral = base.mascara_lateral_por_linha(fine | body, axis, cfg["side_dx"])
        cand = top | ((seeded | lateral) & (dist >= cfg["dist_mid"]))
    elif mode == "top_heavy":
        top2 = top_region(seg, cfg.get("top_rel2", cfg["top_rel"]), cfg.get("top_half2", cfg["top_half"]))
        cand = top2 | (fine & (dist >= cfg["dist_low"]))
    elif mode == "fine_only_top":
        cand = top | (fine & (dist >= cfg["dist_mid"]))
    elif mode == "fine_seed_top":
        fine_seed = base.componentes_que_tocam_seed(fine, side_seed, area_min=cfg["seed_area"])
        cand = top | (fine_seed & (dist >= cfg["dist_mid"]))
    elif mode == "row_axis_ban":
        band = row_lateral_band(fine | body, axis, y_top, y_base, cfg["top_dx"], cfg["low_dx"])
        cand = top | (band & (dist >= cfg["dist_mid"]))
    elif mode == "row_axis_ban_fine":
        band = row_lateral_band(fine, axis, y_top, y_base, cfg["top_dx"], cfg["low_dx"])
        cand = top | (band & (dist >= cfg["dist_mid"]))
    elif mode == "center_band_fine":
        xx = np.arange(body.shape[1], dtype=np.float32)[None, :]
        band = np.abs(xx - axis[:, None]) >= cfg["center_dx"]
        cand = top | (fine & band & (dist >= cfg["dist_mid"]))
    elif mode == "raw_center_cut":
        xx = np.arange(body.shape[1], dtype=np.float32)[None, :]
        band = np.abs(xx - axis[:, None]) >= cfg["center_dx"]
        cand = top | (fine & band)
    elif mode == "components_center_cut":
        gate = row_lateral_gate(body.shape, axis, y_top, y_base, cfg["top_dx"], cfg["low_dx"])
        cut = (fine | body) & gate
        seed = base.mascara_lateral_por_linha(cut, axis, cfg["side_dx"]) | top
        cand = base.componentes_que_tocam_seed(cut, seed, area_min=cfg["seed_area"])
    elif mode == "fine_components_center_cut":
        gate = row_lateral_gate(body.shape, axis, y_top, y_base, cfg["top_dx"], cfg["low_dx"])
        cut = fine & gate
        seed = base.mascara_lateral_por_linha(cut, axis, cfg["side_dx"]) | top
        cand = base.componentes_que_tocam_seed(cut, seed, area_min=cfg["seed_area"])
    else:
        raise ValueError(mode)

    if mode in {"raw_center_cut", "components_center_cut", "fine_components_center_cut"} and cfg.get("skip_smooth", False):
        smooth = cand.copy()
    else:
        smooth = cv2.morphologyEx(cand.astype(np.uint8) * 255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, cfg["close_k"])) > 0
    if not cfg.get("skip_dilate", False) and mode != "raw_center_cut" and cfg.get("leaf_dilate_k", (1, 1)) != (1, 1):
        smooth = cv2.dilate(smooth.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, cfg["leaf_dilate_k"]), iterations=1) > 0

    if mode in {"fine_only_top", "fine_seed_top", "row_axis_ban_fine", "center_band_fine", "raw_center_cut", "fine_components_center_cut"}:
        final = fine & smooth
        final |= top & (fine | body)
    else:
        final = (fine | body) & smooth

    if mode in {"top_plus_fine", "top_heavy", "fine_only_top", "fine_seed_top", "row_axis_ban", "row_axis_ban_fine", "center_band_fine", "raw_center_cut", "components_center_cut", "fine_components_center_cut"}:
        final &= (dist >= cfg["final_dist"]) | top
    else:
        final &= (dist >= cfg["final_dist"]) | top | top_by_branch

    final, vase = hard_vase_cut(final, seg, cfg["vase_below"], cfg["vase_xhalf"])
    if "final_top_dx" in cfg and "final_low_dx" in cfg:
        final_gate = row_lateral_gate(body.shape, axis, y_top, y_base, cfg["final_top_dx"], cfg["final_low_dx"])
        final = (final & final_gate) | (top & (fine | body))
    final = base.filtra_area_min(final, 1)

    return {
        "mask_corpo": body,
        "mask_fino": fine,
        "mask_caule": stem,
        "mask_vertical": vertical,
        "folhas_cruas": smooth,
        "mask_folhas": final,
        "recorte_vaso": vase,
        "area": int(np.count_nonzero(final)),
    }


def right_leaf_seed(seg, stem, min_dx, side="right"):
    body = seg["mask_corpo"]
    axis = base.eixo_x_por_linha(stem if np.any(stem) else body, seg["x_centro_tubete"])
    xx = np.arange(body.shape[1])[None, :]
    if side == "right":
        return body & ((xx - axis[:, None]) >= min_dx)
    return body & ((axis[:, None] - xx) >= min_dx)


def stem_has_support_above(stem, look_h, radius):
    u8 = stem.astype(np.uint8)
    k = np.ones((look_h, 2 * radius + 1), np.uint8)
    sup = cv2.filter2D(u8, -1, k, anchor=(radius, 0), borderType=cv2.BORDER_CONSTANT)
    return sup > 0


def stem_has_support_below(stem, look_h, radius):
    u8 = stem.astype(np.uint8)
    k = np.ones((look_h, 2 * radius + 1), np.uint8)
    sup = cv2.filter2D(u8, -1, k, anchor=(radius, look_h - 1), borderType=cv2.BORDER_CONSTANT)
    return sup > 0


def remove_pinheiro_vase_rim(mask, seg):
    if not np.any(mask):
        return mask, np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    tub = seg["mask_tubete"]
    rim = cv2.dilate(
        tub.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ) > 0
    ys, xs = np.where(tub)
    if len(xs):
        x0 = max(0, int(xs.min()) - 8)
        x1 = min(w, int(xs.max()) + 9)
    else:
        x0 = max(0, seg["x_centro_tubete"] - 36)
        x1 = min(w, seg["x_centro_tubete"] + 37)
    y0 = max(0, seg["y_topo_tubete"] - 4)
    y1 = min(h, seg["y_topo_tubete"] + 12)
    rim_band = np.zeros_like(mask, dtype=bool)
    rim_band[y0:y1, x0:x1] = True
    rim &= rim_band
    out = mask.copy()
    out[rim] = False
    return out, rim


def resolve_eucalipto_variant(img_rgb, seg, cfg):
    body = seg["mask_corpo"]
    fine = make_fine_mask(img_rgb, seg, cfg["hmin"], cfg["hmax"], cfg["smin"], cfg["vmin"], cfg["gbmin"], cfg["grmin"], cfg["fine_dilate_k"])
    stem = base.mascara_caule_eucalipto(body, seg["ponto_base"], {
        "stem_dist_max": cfg["stem_dist_max"],
        "stem_height": 110,
        "stem_x_half": 90,
        "stem_y_extra": 20,
        "stem_area_min": 80,
        "stem_dilate_k": cfg["stem_dilate_k"],
    })

    dist = base.distancia_ao_stem(stem)
    support_above = stem_has_support_above(stem, cfg["look_h"], cfg["look_radius"])
    support_below = stem_has_support_below(stem, max(20, cfg["look_h"] // 2), cfg["look_radius"])
    stem_core = stem & support_above & support_below
    axis = base.eixo_x_por_linha(stem if np.any(stem) else body, seg["x_centro_tubete"])
    y_top, y_base, height = height_info(seg)
    branch_y = estimate_branch_y(seg, stem, cfg["branch_width_factor"])
    top_mask = np.zeros_like(body, dtype=bool)
    top_mask[y_top:branch_y, :] = True
    right_seed = right_leaf_seed(seg, stem, cfg.get("right_dx", 12), "right")
    left_seed = right_leaf_seed(seg, stem, cfg.get("left_dx", 12), "left")
    right_comps = base.componentes_que_tocam_seed(body | fine, right_seed, area_min=cfg.get("seed_area", 5))
    left_comps = base.componentes_que_tocam_seed(body | fine, left_seed, area_min=cfg.get("seed_area", 5))
    side_comps = right_comps | left_comps

    mode = cfg["mode"]
    if mode == "baseline":
        cand = (body | fine)
        cand &= (dist >= cfg["dist_min"]) | top_mask
    elif mode == "line_right_recover":
        lateral = base.mascara_lateral_por_linha(body | fine, axis, cfg["line_dx"])
        cand = ((lateral | right_seed) & ((dist >= cfg["dist_min"]) | (~support_above))) | top_mask
    elif mode == "component_recover_right":
        comps = right_comps
        cand = (comps & ((dist >= cfg["dist_min"]) | (~support_above))) | top_mask
    elif mode == "support_pruned_stem":
        cand = ((body | fine) & (dist >= cfg["dist_min"])) | ((body | fine) & (~support_above) & (dist >= cfg["recover_dist"])) | top_mask
    elif mode == "hybrid_right":
        comps = right_comps
        lateral = base.mascara_lateral_por_linha(body | fine, axis, cfg["line_dx"])
        cand = top_mask | (fine & (dist >= cfg["dist_min"])) | (comps & (~support_above)) | (lateral & (dist >= cfg["recover_dist"]))
    elif mode == "support_gap_right":
        gap = (~support_above) & (dist >= cfg["recover_dist"])
        cand = top_mask | (fine & (dist >= cfg["dist_min"])) | (right_comps & gap)
    elif mode == "sided_components_both":
        gap = (~support_above) & (dist >= cfg["recover_dist"])
        lateral = base.mascara_lateral_por_linha(body | fine, axis, cfg["line_dx"])
        cand = top_mask | (side_comps & gap) | (lateral & (dist >= cfg["dist_min"]))
    elif mode == "supported_stem_only":
        cand = top_mask | (body | fine)
    else:
        raise ValueError(mode)

    smooth = cv2.morphologyEx(cand.astype(np.uint8) * 255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, cfg["close_k"])) > 0
    if cfg["leaf_dilate_k"] != (1, 1):
        smooth = cv2.dilate(smooth.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, cfg["leaf_dilate_k"]), iterations=1) > 0

    final = (body | fine) & smooth
    stem_for_excl = stem.copy()
    if mode == "supported_stem_only":
        stem_for_excl = stem_core
    elif mode in {"sided_components_both", "support_gap_right", "hybrid_right"}:
        stem_for_excl = stem_core
    excl_kernel = cfg["stem_exclude_k"]
    if mode in {"sided_components_both", "support_gap_right", "hybrid_right"}:
        excl_kernel = cfg.get("stem_core_exclude_k", cfg["stem_exclude_k"])
    excl = cv2.dilate(stem_for_excl.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, excl_kernel), iterations=1) > 0
    if mode in {
        "line_right_recover",
        "component_recover_right",
        "support_pruned_stem",
        "hybrid_right",
        "support_gap_right",
        "sided_components_both",
    }:
        final &= ((~excl) | (~support_above) | (~support_below))
    else:
        final &= (~excl)
    final |= top_mask & (body | fine)
    final = base.filtra_area_min(final, cfg["final_area_min"])

    return {
        "mask_corpo": body,
        "mask_fino": fine,
        "mask_caule": stem,
        "mask_vertical": stem.copy(),
        "folhas_cruas": smooth,
        "mask_folhas": final,
        "recorte_vaso": np.zeros_like(body, dtype=bool),
        "area": int(np.count_nonzero(final)),
    }


def pinheiro_variants():
    return [
        {"name": "P4 lateral + topo22", "fine_h_min": 18, "fine_h_max": 112, "fine_s_min": 7, "fine_v_min": 15, "fine_gb_min": 0, "fine_gr_min": -8, "fine_close_k": (1, 1), "fine_dilate_k": (5, 5), "fine_area_min": 1, "below_tube_cut": 8, "stem_open_k": (3, 41), "stem_close_k": (5, 17), "stem_dilate_k": (7, 13), "stem_height": 280, "stem_x_half": 55, "stem_area_min": 30, "stem_y_extra": 16, "basic_half": 16, "vase_y_above": 8, "vase_y_below": 46, "vase_x_half": 60, "top_leaf_height": 85, "top_leaf_rel": 0.22, "top_leaf_half": 16, "mid_min_dist": 6.5, "mid_body_min_dist": 10.5, "low_min_dist": 14.0, "final_mid_min_dist": 16.0, "final_low_min_dist": 20.0, "stem_top_ignore_rel": 0.22, "leaf_close_k": (5, 5), "leaf_dilate_k": (1, 1), "leaf_area_pre": 1, "leaf_area_post": 2, "stem_exclude_k": (11, 29)},
    ]


def eucalipto_variants():
    return [
        {"name": "E1 base lados", "mode": "sided_components_both", "hmin": 18, "hmax": 108, "smin": 10, "vmin": 20, "gbmin": 2, "grmin": -2, "fine_dilate_k": (5, 5), "stem_dist_max": 1.15, "stem_dilate_k": (5, 7), "stem_exclude_k": (3, 3), "stem_core_exclude_k": (5, 5), "dist_min": 2.0, "recover_dist": 1.0, "line_dx": 10, "right_dx": 8, "left_dx": 8, "seed_area": 3, "branch_width_factor": 1.45, "look_h": 90, "look_radius": 8, "close_k": (5, 5), "leaf_dilate_k": (3, 3), "final_area_min": 8},
    ]


def natural_key(path_obj):
    name = path_obj.stem
    digits = "".join(ch for ch in name if ch.isdigit())
    return (path_obj.parent.name, int(digits) if digits else 0, name)


def selected_species_images():
    base_dir = Path(__file__).parent / "Dataset_Projeto1"
    eucs = sorted((base_dir / "_Eucalipto_Escolhidos1").glob("*.jpg"), key=natural_key)
    pins = sorted((base_dir / "_Pinheiro_Escolhidos1").glob("*.jpg"), key=natural_key)
    return eucs, pins


def overlay_mask(img_rgb, mask, color, alpha=0.88):
    out = img_rgb.copy()
    if np.any(mask):
        color_arr = np.array(color, dtype=np.uint8)
        out[mask] = np.clip((1 - alpha) * out[mask] + alpha * color_arr, 0, 255).astype(np.uint8)
    return out


def save_gallery_all(species_name, image_paths, results, title, out_name):
    n = len(image_paths)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(12, 5.2 * rows))
    axs = np.atleast_2d(axs)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, img_path, res in zip(axs.ravel(), image_paths, results):
        overlay = overlay_mask(base.ler_imagem_rgb(img_path), res["mask_folhas"], (255, 0, 0), alpha=0.92)
        ax.imshow(overlay)
        ax.set_title(f"{img_path.stem}\narea={res['area']} px")
        ax.axis("off")
    fig.suptitle(title, fontsize=18)
    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def open_on_pc(path):
    if os.environ.get("NO_OPEN", "") == "1":
        return
    if sys.platform.startswith("win"):
        os.startfile(str(path))


def main():
    pin_cfg = pinheiro_variants()[0]
    euc_cfg = eucalipto_variants()[0]
    euc_paths, pin_paths = selected_species_images()

    pin_results = []
    for img_path in pin_paths:
        img = base.ler_imagem_rgb(img_path)
        seg = base.segmenta_base(img)
        res = base.resolve_pinheiro(img, seg, pin_cfg)
        res["mask_folhas"], rim = remove_pinheiro_vase_rim(res["mask_folhas"], seg)
        res["area"] = int(np.count_nonzero(res["mask_folhas"]))
        res["recorte_vaso"] |= rim
        pin_results.append(res)
        print(f"{img_path.stem} | area={res['area']}")
    out_pin = save_gallery_all(
        "Pinheiro",
        pin_paths,
        pin_results,
        "Area Foliar Final - Pinheiros (P4 lateral + topo22)",
        "resultado_final_pinheiros.png",
    )
    print(out_pin)

    euc_results = []
    for img_path in euc_paths:
        img = base.ler_imagem_rgb(img_path)
        seg = base.segmenta_base(img)
        res = resolve_eucalipto_variant(img, seg, euc_cfg)
        euc_results.append(res)
        print(f"{img_path.stem} | area={res['area']}")
    out_euc = save_gallery_all(
        "Eucalipto",
        euc_paths,
        euc_results,
        "Area Foliar Final - Eucaliptos (E1 base lados)",
        "resultado_final_eucaliptos.png",
    )
    print(out_euc)

    open_on_pc(out_pin)
    open_on_pc(out_euc)


if __name__ == "__main__":
    main()
