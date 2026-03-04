#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
import laspy, rasterio, joblib

def sample_raster(src, xs, ys):
    vals = np.array([v[0] for v in src.sample(zip(xs,ys))], dtype=np.float32)
    nod = src.nodata
    if nod is not None:
        m = np.isclose(vals, nod) | np.isnan(vals)
        vals[m] = 0.0
    return vals

def main():
    ap = argparse.ArgumentParser("Infer point labels and write PredLabel ExtraBytes")
    ap.add_argument("--las", required=True)
    ap.add_argument("--dtm")
    ap.add_argument("--slope")
    ap.add_argument("--rgb")
    ap.add_argument("--model", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    las = laspy.read(args.las)
    # 读取 LAS 后
    xs = np.asarray(las.x, dtype=np.float64)
    ys = np.asarray(las.y, dtype=np.float64)
    zs = np.asarray(las.z, dtype=np.float32)

    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feat_names = meta["feat_names"]

    # 构造与训练一致的特征顺序
    feat_map = {}
    if "Z" in feat_names:
        feat_map["Z"] = zs
    if args.dtm and ("DTM" in feat_names):
        with rasterio.open(args.dtm) as ds: feat_map["DTM"] = sample_raster(ds, xs, ys)
    if args.slope and ("SLOPE" in feat_names):
        with rasterio.open(args.slope) as ds: feat_map["SLOPE"] = sample_raster(ds, xs, ys)
    if args.rgb and ("RGB" in feat_names):
        with rasterio.open(args.rgb) as ds: feat_map["RGB"] = sample_raster(ds, xs, ys)
    if "HAG" in feat_names:
        if "HeightAboveGround" in las.point_format.extra_dimension_names:
            feat_map["HAG"] = np.asarray(las["HeightAboveGround"]).astype(np.float32)
        else:
            feat_map["HAG"] = np.zeros_like(zs, dtype=np.float32)

    X = np.stack([feat_map[n] for n in feat_names], axis=1).astype(np.float32)

    clf = joblib.load(args.model)
    pred = clf.predict(X).astype(np.uint16)

    # 写出：附加 ExtraBytes 'PredLabel'
    if "PredLabel" not in las.point_format.extra_dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name="PredLabel", type=np.uint16, description="predicted class"))
    las["PredLabel"] = pred

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    las.write(outp)
    print(f"OK: wrote {outp} with PredLabel ExtraBytes.")

if __name__ == "__main__":
    main()
