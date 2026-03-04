#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add HeightAboveGround (HAG) to LAS/LAZ by sampling a DTM.
- 如果输入不是 LAS 1.4，将 header 升级为 1.4（point format 保持一致）。
- 先拷贝 points -> 再 add_extra_dim（避免 point format 冲突）。
- 写出后自检：打印版本/point_format/point_length、是否存在 HeightAboveGround，并给出统计。
"""

import argparse
from pathlib import Path
import numpy as np
import laspy
import rasterio

def _sample_raster(src, xs, ys):
    return np.fromiter((v[0] for v in src.sample(zip(xs, ys))), dtype=np.float32, count=len(xs))

def _print_extent_note(xmin, xmax, ymin, ymax, rb):
    print(f"[INFO] LAS extent  : X[{xmin:.3f},{xmax:.3f}]  Y[{ymin:.3f},{ymax:.3f}]")
    print(f"[INFO] DTM bounds  : X[{rb.left:.3f},{rb.right:.3f}] Y[{rb.bottom:.3f},{rb.top:.3f}]")
    overlaps = not (xmax < rb.left or xmin > rb.right or ymax < rb.bottom or ymin > rb.top)
    if not overlaps:
        print("[WARN] LAS 与 DTM 外包框几乎不重叠：请确认坐标系/单位一致（同一 UTM 或同一经纬度）")

def _header_summary(tag, hdr):
    print(f"[{tag}] version={hdr.version}  point_format.id={hdr.point_format.id}  point_length={hdr.point_format.size}  scale={hdr.scales}  offset={hdr.offsets}")

def add_hag_safe(inp, dtm, out, nodata_fill=0.0, chunk=2_000_000, stat_sample=200_000, quiet=False):
    in_path, dtm_path, out_path = Path(inp), Path(dtm), Path(out)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")
    if not dtm_path.exists():
        raise FileNotFoundError(f"DTM not found: {dtm_path}")

    las_in = laspy.read(in_path)
    _header_summary("IN", las_in.header)

    # ========== 1) 构造输出 header ==========
    # 若不是 1.4，升级为 1.4（point format 保持 id 不变）
    if str(las_in.header.version) != "1.4":
        new_hdr = laspy.LasHeader(version="1.4", point_format=las_in.header.point_format)
        new_hdr.scales = las_in.header.scales
        new_hdr.offsets = las_in.header.offsets
        try:
            # 尽量复制 CRS（若 laspy/las 里有）
            new_hdr.parse_crs(las_in.header.parse_crs())
        except Exception:
            pass
    else:
        new_hdr = las_in.header.copy()

    # 以新 header 构造输出数据，并先拷贝 points（保持格式一致）
    las_out = laspy.LasData(new_hdr)
    las_out.points = las_in.points.copy()
    _header_summary("OUT(before-EB)", las_out.header)

    # ========== 2) 添加 ExtraBytes & 分块写入 ==========
    if "HeightAboveGround" not in las_out.point_format.extra_dimension_names:
        las_out.add_extra_dim(laspy.ExtraBytesParams(
            name="HeightAboveGround", type=np.float32, description="Z - DTM"
        ))
    # 加完 ExtraBytes 后 point_length 会增加
    _header_summary("OUT(after-EB)", las_out.header)

    N = las_out.header.point_count
    with rasterio.open(dtm_path) as src:
        xmin, xmax = float(las_out.x.min()), float(las_out.x.max())
        ymin, ymax = float(las_out.y.min()), float(las_out.y.max())
        if not quiet:
            _print_extent_note(xmin, xmax, ymin, ymax, src.bounds)

        nod = src.nodata
        chunk = max(1, int(chunk))
        for start in range(0, N, chunk):
            end = min(N, start + chunk)
            xs = np.asarray(las_out.x[start:end], dtype=np.float64)
            ys = np.asarray(las_out.y[start:end], dtype=np.float64)
            zs = np.asarray(las_out.z[start:end], dtype=np.float32)

            dtm_z = _sample_raster(src, xs, ys)
            mask = np.isnan(dtm_z)
            if nod is not None:
                mask |= np.isclose(dtm_z, nod)
            if mask.any():
                dtm_z[mask] = float(nodata_fill)

            hag = (zs - dtm_z).astype(np.float32)
            las_out["HeightAboveGround"][start:end] = hag

            if not quiet and ((start // chunk) % 25 == 0 or end == N):
                print(f"[INFO] processed {end}/{N} points ...")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    las_out.write(out_path)

    # ========== 3) 自检 ==========
    las2 = laspy.read(out_path)
    _header_summary("WROTE", las2.header)
    dims = list(las2.point_format.extra_dimension_names)
    print(f"[CHECK] Extra dims: {dims}")
    ok = "HeightAboveGround" in dims
    print(f"[CHECK] HeightAboveGround exists -> {ok}")
    if not ok:
        print("[FAIL] 写出后未发现 HAG 维度（检查 laspy 版本 / 写入路径 / 是否被覆盖）。")
        return False

    h = np.asarray(las2["HeightAboveGround"])
    n = h.size
    k = min(n, int(stat_sample))
    samp = h if k == n else h[np.random.default_rng(42).choice(n, size=k, replace=False)]
    mn, mx, mu, std = float(np.nanmin(samp)), float(np.nanmax(samp)), float(np.nanmean(samp)), float(np.nanstd(samp))
    frac_neg = float((samp < 0).sum()) / samp.size
    frac_zer = float((samp == 0).sum()) / samp.size
    print(f"[STATS] HAG sample({samp.size}/{n})  min={mn:.3f} max={mx:.3f} mean={mu:.3f} std={std:.3f}")
    print(f"[STATS] HAG negatives={100*frac_neg:.2f}%  zeros={100*frac_zer:.2f}%")
    print(f"[OK] wrote {out_path}")
    return True

def main():
    ap = argparse.ArgumentParser(description="Add HeightAboveGround from DTM to LAS/LAZ (upgrade to LAS 1.4 if needed, self-check).")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--dtm", dest="dtm", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--nodata_fill", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--stat_sample", type=int, default=200_000)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ok = add_hag_safe(args.inp, args.dtm, args.out,
                      nodata_fill=args.nodata_fill,
                      chunk=args.chunk,
                      stat_sample=args.stat_sample,
                      quiet=args.quiet)
    if not ok:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
