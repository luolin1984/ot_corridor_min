#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
import laspy
import rasterio

def sample_rasters(xs, ys, rasters):
    feats, names = [], []
    for name, src in rasters:
        if src is None: 
            continue
        vals = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype=np.float32)
        nod = src.nodata
        if nod is not None:
            mask = np.isclose(vals, nod) | np.isnan(vals)
            vals[mask] = 0.0
        feats.append(vals); names.append(name)
    return (np.stack(feats, axis=1) if feats else np.zeros((len(xs),0),np.float32)), names

def main():
    ap = argparse.ArgumentParser("Sample rasters to LAS points -> NPZ")
    ap.add_argument("--las", required=True)
    ap.add_argument("--dtm")
    ap.add_argument("--slope")
    ap.add_argument("--rgb")               # 可选：单波段或灰度；多波段可先转灰度
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    las = laspy.read(args.las)
    # laspy 的 ScaledArrayView -> 真实的 NumPy 数组
    xs = np.asarray(las.x, dtype=np.float64)
    ys = np.asarray(las.y, dtype=np.float64)
    zs = np.asarray(las.z, dtype=np.float32)

    # 分类（若存在就取；否则给 0）
    y_cls = (np.asarray(las.classification, dtype=np.uint8)
             if "classification" in las.point_format.standard_dimension_names
             else np.zeros(zs.shape[0], dtype=np.uint8))

    rasters = []
    ds_dtm   = rasterio.open(args.dtm)   if args.dtm   else None
    ds_slope = rasterio.open(args.slope) if args.slope else None
    ds_rgb   = rasterio.open(args.rgb)   if args.rgb   else None

    if ds_dtm:   rasters.append(("DTM", ds_dtm))
    if ds_slope: rasters.append(("SLOPE", ds_slope))
    if ds_rgb:
        # 若是多波段，改为其第1波段（最简）
        if ds_rgb.count > 1:
            ds_rgb = rasterio.open(args.rgb)  # 简化处理：仍取 sample 返回第一波段
        rasters.append(("RGB", ds_rgb))

    # 采样
    R, names = sample_rasters(xs, ys, rasters)
    # HAG（如存在）
    hag = None
    if "HeightAboveGround" in las.point_format.extra_dimension_names:
        hag = np.asarray(las["HeightAboveGround"]).astype(np.float32)
        R = np.hstack([R, hag[:,None]]); names.append("HAG")

    # 坐标也可作为弱特征（常能提升塔/导线区分）
    base_feats = zs.reshape(-1, 1)  # 仅 Z（最简），此时 zs 已是 float32
    X = np.hstack([base_feats, R]) if R.size else base_feats
    feat_names = ["Z"] + names

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X.astype(np.float32), y=y_cls.astype(np.int32),
                        feat_names=np.array(feat_names, dtype=object),
                        xyz=np.stack([xs,ys,zs],axis=1).astype(np.float64),
                        has_hag=np.array([1 if hag is not None else 0], dtype=np.int32))
    print(f"OK: wrote {args.out} with features={feat_names}, N={len(X)}")

if __name__ == "__main__":
    main()
