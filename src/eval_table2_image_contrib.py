import argparse, os
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import shape

def bin_erode(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    s = np.zeros_like(m, dtype=np.uint16)
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            s += np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    return (s == 9)

def bin_dilate(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    s = np.zeros_like(m, dtype=np.uint16)
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            s += np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    return (s > 0)

def boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    return (m & (~bin_erode(m)))

def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol_pix: int = 1) -> float:
    bp = boundary(pred)
    bg = boundary(gt)
    # tolerance via dilation
    bg_tol = bg.copy()
    bp_tol = bp.copy()
    for _ in range(tol_pix):
        bg_tol = bin_dilate(bg_tol)
        bp_tol = bin_dilate(bp_tol)
    tp_p = np.logical_and(bp, bg_tol).sum()
    tp_g = np.logical_and(bg, bp_tol).sum()
    p = tp_p / max(bp.sum(), 1)
    r = tp_g / max(bg.sum(), 1)
    return 2*p*r / max(p+r, 1e-12)

def load_corridor_mask(corridor_path, layer, ref_ds):
    gdf = gpd.read_file(corridor_path, layer=layer)
    geoms = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    return rasterize(
        geoms, out_shape=(ref_ds.height, ref_ds.width),
        transform=ref_ds.transform, fill=0, dtype=np.uint8
    ).astype(bool)

def load_tower_mask(tower_path, buf_m, ref_ds):
    gdf = gpd.read_file(tower_path)
    gdf = gdf.to_crs(ref_ds.crs)
    geoms = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        geoms.append((geom.buffer(buf_m), 1))
    if not geoms:
        return np.zeros((ref_ds.height, ref_ds.width), np.uint8).astype(bool)
    return rasterize(
        geoms, out_shape=(ref_ds.height, ref_ds.width),
        transform=ref_ds.transform, fill=0, dtype=np.uint8
    ).astype(bool)

def metrics_binary(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, ~gt).sum()
    TN = np.logical_and(~pred, ~gt).sum()
    FN = np.logical_and(~pred, gt).sum()
    prec = TP / max(TP+FP, 1)
    rec  = TP / max(TP+FN, 1)
    f1   = 2*TP / max(2*TP+FP+FN, 1)
    miou = TP / max(TP+FP+FN, 1)
    oa   = (TP+TN) / max(TP+FP+TN+FN, 1)
    fpr  = FP / max(FP+TN, 1)
    return dict(TP=int(TP), FP=int(FP), TN=int(TN), FN=int(FN),
                Precision=prec, Recall=rec, F1=f1, mIoU=miou, OA=oa, FPR=fpr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_mask", required=True)
    ap.add_argument("--corridor", required=True)
    ap.add_argument("--corridor_layer", default="corridor")
    ap.add_argument("--towers", required=True)
    ap.add_argument("--tower_buf", type=float, default=30.0)
    ap.add_argument("--pred", nargs="+", required=True, help="Name=path pairs")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--tol_pix", type=int, default=1)
    args = ap.parse_args()

    with rasterio.open(args.gt_mask) as gt_ds:
        gt = (gt_ds.read(1) == 1)
        corridor_mask = load_corridor_mask(args.corridor, args.corridor_layer, gt_ds)
        tower_mask = load_tower_mask(args.towers, args.tower_buf, gt_ds)
        roi = corridor_mask  # 统计域：走廊内

        gt_roi = np.logical_and(gt, roi)
        gt_bg_roi = np.logical_and(~gt, roi)

        rows = []
        for item in args.pred:
            name, path = item.split("=", 1)
            with rasterio.open(path) as ds:
                assert ds.transform == gt_ds.transform and ds.crs == gt_ds.crs and ds.width == gt_ds.width and ds.height == gt_ds.height, \
                    f"{name}: pred 与 gt 未对齐"
                pred = (ds.read(1) == 1)

            pred_roi = np.logical_and(pred, roi)
            m = metrics_binary(pred_roi, gt_roi)

            # Tower-FPR：在“塔位缓冲区 ∩ 走廊 ∩ 非风险(gt=0)”区域统计误检
            tower_roi = np.logical_and.reduce([tower_mask, roi, ~gt])
            FP_tower = np.logical_and(pred, tower_roi).sum()
            TN_tower = np.logical_and(~pred, tower_roi).sum()
            tower_fpr = FP_tower / max(FP_tower + TN_tower, 1)

            # Boundary-F1
            b_f1 = boundary_f1(pred_roi, gt_roi, tol_pix=args.tol_pix)

            # Fragmentation（简化：连通域数量；若你想更严谨可换 scipy.ndimage.label）
            # 这里用“边界像元数/面积”做一个几何复杂度 proxy，避免依赖额外库
            edge = boundary(pred_roi).sum()
            area = pred_roi.sum()
            complexity = float(edge) / max(float(area), 1.0)

            rows.append({
                "Method": name,
                **{k: m[k] for k in ["Precision","Recall","F1","mIoU","OA","FPR"]},
                "Tower_FPR": tower_fpr,
                f"Boundary_F1@{args.tol_pix}px": b_f1,
                "BoundaryComplexity(edge/area)": complexity
            })

    df = pd.DataFrame(rows).sort_values("mIoU", ascending=False)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("[OK] wrote:", args.out_csv)
    print("\n[Table2 preview]\n", df.to_string(index=False))

if __name__ == "__main__":
    main()