#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 CHM / SLOPE 与 RGB 对齐到完全一致的网格（CRS、transform、宽高）。
默认：
- CHM 使用 Resampling.max（保最高冠层）
- SLOPE 使用 Resampling.bilinear（连续变量）

用法示例：
python align_rasters.py \
  --rgb data/imagery/aoi_rgb.tif \
  --chm outputs/aoi_chm.tif \
  --slope outputs/aoi_slope.tif \
  --out_chm outputs/aoi_chm_10m.tif \
  --out_slope outputs/aoi_slope_10m.tif
"""

import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def _parse_resampling(name: str) -> Resampling:
    name = name.lower()
    m = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "lanczos": Resampling.lanczos,
        "average": Resampling.average,
        "mode": Resampling.mode,
        "gauss": Resampling.gauss,
        "max": Resampling.max,
        "min": Resampling.min,
        "med": Resampling.med,
        "q1": Resampling.q1,
        "q3": Resampling.q3,
        "sum": Resampling.sum,
        "rms": Resampling.rms,
    }
    if name not in m:
        raise argparse.ArgumentTypeError(f"Unsupported resampling: {name}")
    return m[name]


@dataclass
class RefGrid:
    crs: any
    transform: any
    width: int
    height: int
    profile: dict


def _ref_from_path(path: str) -> RefGrid:
    with rasterio.open(path) as ds:
        return RefGrid(
            crs=ds.crs,
            transform=ds.transform,
            width=ds.width,
            height=ds.height,
            profile=ds.profile,
        )


def align_one(src_path: str,
              dst_path: str,
              ref: RefGrid,
              resampling: Resampling,
              dst_nodata: float = 0.0,
              dtype: str = "float32") -> str:
    """将单波段栅格重采样/重投影到参考网格并写出。"""
    with rasterio.open(src_path) as src:
        out = np.zeros((1, ref.height, ref.width), dtype=dtype)
        reproject(
            source=rasterio.band(src, 1),
            destination=out[0],
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )

    prof = ref.profile.copy()
    prof.update({
        "count": 1,
        "dtype": out.dtype.name,
        "nodata": dst_nodata,
        "compress": "DEFLATE",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    })
    # 参考自 RGB 的 profile，去掉可能的色彩元数据
    prof.pop("photometric", None)

    with rasterio.open(dst_path, "w", **prof) as dst:
        dst.write(out)

    return dst_path


def check_alignment(paths: List[str]) -> Tuple[bool, List[dict]]:
    """检查一组数据是否在 CRS/transform/宽高上完全一致。"""
    metas = []
    for p in paths:
        with rasterio.open(p) as ds:
            metas.append({
                "path": p,
                "crs": str(ds.crs),
                "transform": tuple(ds.transform),
                "width": ds.width,
                "height": ds.height,
                "dtype": ds.dtypes[0],
                "nodata": ds.nodata,
            })
    first = metas[0]
    ok = True
    for m in metas[1:]:
        ok &= (m["crs"] == first["crs"] and
               m["transform"] == first["transform"] and
               m["width"] == first["width"] and
               m["height"] == first["height"])
    return ok, metas


def main():
    ap = argparse.ArgumentParser(description="Align CHM/SLOPE rasters to RGB grid")
    ap.add_argument("--rgb", required=True, help="参考 RGB（目标网格）")
    ap.add_argument("--chm", required=True, help="源 CHM")
    ap.add_argument("--slope", required=True, help="源 SLOPE")
    ap.add_argument("--out_chm", required=True, help="输出 CHM（对齐后）")
    ap.add_argument("--out_slope", required=True, help="输出 SLOPE（对齐后）")
    ap.add_argument("--chm_resampling", default="max", type=_parse_resampling,
                    help="CHM 重采样方式（默认 max）")
    ap.add_argument("--slope_resampling", default="bilinear", type=_parse_resampling,
                    help="SLOPE 重采样方式（默认 bilinear）")
    ap.add_argument("--dst_nodata", default=0.0, type=float, help="输出 nodata（默认 0）")
    args = ap.parse_args()

    ref = _ref_from_path(args.rgb)

    chm_out = align_one(args.chm, args.out_chm, ref,
                        resampling=args.chm_resampling,
                        dst_nodata=args.dst_nodata, dtype="float32")
    slope_out = align_one(args.slope, args.out_slope, ref,
                          resampling=args.slope_resampling,
                          dst_nodata=args.dst_nodata, dtype="float32")

    ok, metas = check_alignment([args.rgb, chm_out, slope_out])
    print(json.dumps({"aligned": ok, "metas": metas}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
