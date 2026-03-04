#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

def nearest_match(a_pts, b_pts, tol):
    # 返回：a 中被 b 命中的比例、b 中被 a 命中的比例
    if len(a_pts)==0 and len(b_pts)==0: return 1.0, 1.0
    if len(a_pts)==0 or len(b_pts)==0:  return 0.0, 0.0
    def hits(src, dst):
        count=0
        for p in src:
            if any(p.distance(q)<=tol for q in dst):
                count+=1
        return count/len(src)
    return hits(a_pts, b_pts), hits(b_pts, a_pts)

def polygon_iou(poly_pred, poly_gt):
    if poly_pred is None or poly_gt is None: return 0.0
    inter = poly_pred.intersection(poly_gt).area
    union = poly_pred.union(poly_gt).area
    return float(inter/union) if union>0 else 0.0

def main():
    ap = argparse.ArgumentParser("Evaluate vector outputs vs OSM")
    ap.add_argument("--pred_towers", required=True)
    ap.add_argument("--pred_corridor", required=True)
    ap.add_argument("--gt_towers", required=True)
    ap.add_argument("--gt_lines", required=True)
    ap.add_argument("--tol", type=float, default=20.0, help="tower match tolerance (m)")
    ap.add_argument("--buf", type=float, default=25.0, help="buffer half-width for GT lines (m)")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    gdf_pt_pred = gpd.read_file(args.pred_towers)
    gdf_poly_pred = gpd.read_file(args.pred_corridor)
    gdf_pt_gt = gpd.read_file(args.gt_towers)
    gdf_ln_gt = gpd.read_file(args.gt_lines)

    # 统一到同一 CRS（使用 pred 的 CRS）
    crs = gdf_pt_pred.crs or gdf_pt_gt.crs or gdf_ln_gt.crs
    if crs:
        gdf_pt_gt = gdf_pt_gt.to_crs(crs)
        gdf_ln_gt = gdf_ln_gt.to_crs(crs)
        if gdf_poly_pred.crs and gdf_poly_pred.crs!=crs: gdf_poly_pred = gdf_poly_pred.to_crs(crs)

    # 塔匹配
    P_pred = [geom for geom in gdf_pt_pred.geometry if geom is not None]
    P_gt   = [geom for geom in gdf_pt_gt.geometry if geom is not None]
    rec_tower, prec_tower = nearest_match(P_gt, P_pred, args.tol)  # GT 被命中比例、预测的精度
    f1_tower = 0.0 if (rec_tower+prec_tower)==0 else 2*rec_tower*prec_tower/(rec_tower+prec_tower)

    # 走廊 IoU
    corr_pred = unary_union([g for g in gdf_poly_pred.geometry if g is not None]) if len(gdf_poly_pred)>0 else None
    gt_buf = unary_union([ln.buffer(args.buf, join_style=1, cap_style=2) for ln in gdf_ln_gt.geometry if ln is not None])
    iou_corr = polygon_iou(corr_pred, gt_buf)

    out = {
        "tower_recall": rec_tower,
        "tower_precision": prec_tower,
        "tower_f1": f1_tower,
        "corridor_iou_vs_buffered_lines": iou_corr,
        "params": {"tol_m": args.tol, "buffer_m": args.buf}
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"OK: vector metrics -> {args.out_json}")

if __name__ == "__main__":
    main()
