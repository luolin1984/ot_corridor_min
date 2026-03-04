#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将标签栅格对齐到 RGB 的网格（CRS、transform、宽高完全一致）。
- 使用最近邻重采样（保留 0/1 分类）
- 输出 Byte/uint8，nodata=0

用法示例：
python align_label_to_rgb.py \
  --rgb data/imagery/aoi_rgb.tif \
  --label_in outputs/aoi_risk_hag_gt6m.tif \
  --label_out outputs/aoi_risk_hag_gt6m_10m.tif
"""
import argparse
import json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--label_in", required=True)
    ap.add_argument("--label_out", required=True)
    ap.add_argument("--dst_nodata", type=int, default=0)
    args = ap.parse_args()

    with rasterio.open(args.rgb) as ref, rasterio.open(args.label_in) as lab:
        out = np.zeros((ref.height, ref.width), dtype=np.uint8)
        reproject(
            source=rasterio.band(lab, 1),
            destination=out,
            src_transform=lab.transform,
            src_crs=lab.crs,
            src_nodata=0,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            dst_nodata=args.dst_nodata,
            resampling=Resampling.nearest,  # 分类最近邻
        )
        prof = ref.profile.copy()
        prof.update({
            "count": 1,
            "dtype": "uint8",
            "nodata": args.dst_nodata,
            "compress": "DEFLATE",
            "predictor": 2,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        })
        prof.pop("photometric", None)

        with rasterio.open(args.label_out, "w", **prof) as dst:
            dst.write(out, 1)

    # 对齐自检
    with rasterio.open(args.rgb) as a, rasterio.open(args.label_out) as b:
        ok = (a.crs == b.crs and a.transform == b.transform and
              a.width == b.width and a.height == b.height)
        print(json.dumps({
            "aligned": bool(ok),
            "rgb": {"crs": str(a.crs), "shape": (a.height, a.width), "transform": tuple(a.transform)},
            "label_out": {"crs": str(b.crs), "shape": (b.height, b.width), "transform": tuple(b.transform)}
        }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
