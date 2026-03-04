#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按给定参考栅格生成“对齐的零栅格”：
- CHM/SLOPE：1 波段，参照 --ref10m（应是 10 m 对齐）
- RGB：3 波段，参照 --ref_rgb

示例：
  python src/make_zero_rasters.py \
    --ref10m outputs/aoi_chm_10m.tif \
    --ref_rgb data/imagery/aoi_rgb.tif \
    --out_chm outputs/zeros_chm_10m.tif \
    --out_slope outputs/zeros_slope_10m.tif \
    --out_rgb outputs/zeros_rgb.tif
"""

import os
import argparse
import numpy as np
import rasterio

DTYPE_MAP = {
    "uint8":   np.uint8,
    "int16":   np.int16,
    "uint16":  np.uint16,
    "int32":   np.int32,
    "uint32":  np.uint32,
    "float32": np.float32,
    "float64": np.float64,
}

def write_zero_raster(ref_path, out_path, bands, dtype_str, nodata=None, compress="LZW"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if dtype_str not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    dtype = DTYPE_MAP[dtype_str]

    with rasterio.open(ref_path) as src:
        profile = src.profile.copy()
        profile.update(
            count=bands,
            dtype=dtype_str,
            compress=compress
        )
        if nodata is not None:
            profile.update(nodata=nodata)

        zeros = np.zeros((bands, src.height, src.width), dtype=dtype)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(zeros)

        # 回显关键信息
        tr = profile.get("transform")
        px = abs(tr.a) if tr is not None else None
        py = abs(tr.e) if tr is not None else None
        print(f"[OK] Wrote zeros -> {out_path}")
        print(f"     size: {profile['width']} x {profile['height']}, bands: {bands}, dtype: {dtype_str}")
        print(f"     crs: {profile.get('crs')}")
        print(f"     res: {px} x {py} (unit of CRS)")
        print(f"     transform: {tr}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref10m", required=True, help="10m 对齐参考（如 outputs/aoi_chm_10m.tif）")
    ap.add_argument("--ref_rgb", required=True, help="RGB 参考（如 data/imagery/aoi_rgb.tif）")
    ap.add_argument("--out_chm", default="outputs/zeros_chm_10m.tif")
    ap.add_argument("--out_slope", default="outputs/zeros_slope_10m.tif")
    ap.add_argument("--out_rgb", default="outputs/zeros_rgb.tif")
    ap.add_argument("--dtype10m", default="float32", choices=list(DTYPE_MAP.keys()),
                    help="CHM/SLOPE 零栅格 dtype（默认 float32）")
    ap.add_argument("--dtype_rgb", default="uint16", choices=list(DTYPE_MAP.keys()),
                    help="RGB 零栅格 dtype（默认 uint16）")
    ap.add_argument("--nodata10m", type=float, default=None, help="CHM/SLOPE 的 NoData（默认沿用参考或不设）")
    ap.add_argument("--nodata_rgb", type=float, default=None, help="RGB 的 NoData（默认沿用参考或不设）")
    args = ap.parse_args()

    # 1) CHM：1 波段零栅格（与 10m 参考对齐）
    write_zero_raster(
        ref_path=args.ref10m,
        out_path=args.out_chm,
        bands=1,
        dtype_str=args.dtype10m,
        nodata=args.nodata10m
    )

    # 2) SLOPE：1 波段零栅格（与 10m 参考对齐）
    write_zero_raster(
        ref_path=args.ref10m,
        out_path=args.out_slope,
        bands=1,
        dtype_str=args.dtype10m,
        nodata=args.nodata10m
    )

    # 3) RGB：3 波段零栅格（与 RGB 参考对齐）
    #    即使 ref_rgb 不是 3 波段，这里也强制写 3 波段零图供网络使用
    write_zero_raster(
        ref_path=args.ref_rgb,
        out_path=args.out_rgb,
        bands=3,
        dtype_str=args.dtype_rgb,
        nodata=args.nodata_rgb
    )

if __name__ == "__main__":
    main()
