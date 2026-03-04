#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析脚本：读取 HAG 点云 / CHM/DTM/Slope / OSM 线路塔，生成多张图与摘要报告。
输出目录：--outdir（默认 outputs/analysis）
"""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import laspy
import rasterio
from rasterio.plot import reshape_as_raster
import geopandas as gpd
from shapely.geometry import box
import matplotlib
matplotlib.use("Agg")  # 无界面后端
import matplotlib.pyplot as plt

def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_las_stats(las_path: Path, sample=300_000):
    las = laspy.read(las_path)
    n = las.header.point_count
    cls = np.asarray(las.classification) if "classification" in las.point_format.standard_dimension_names else None

    # HAG（可能不存在）
    has_hag = "HeightAboveGround" in las.point_format.extra_dimension_names
    hag = np.asarray(las["HeightAboveGround"]) if has_hag else None

    # 采样
    take = min(sample, n)
    idx = np.random.default_rng(42).choice(n, size=take, replace=False) if take < n else np.arange(n)
    xs = np.asarray(las.x[idx], dtype=np.float64)
    ys = np.asarray(las.y[idx], dtype=np.float64)
    zs = np.asarray(las.z[idx], dtype=np.float32)
    cls_s = np.asarray(cls[idx], dtype=np.int16) if cls is not None else None
    hag_s = np.asarray(hag[idx], dtype=np.float32) if hag is not None else None

    # 统计
    summary = {
        "n_points": int(n),
        "extent": {
            "minx": float(las.x.min()), "maxx": float(las.x.max()),
            "miny": float(las.y.min()), "maxy": float(las.y.max()),
            "minz": float(las.z.min()), "maxz": float(las.z.max())
        },
        "has_classification": cls is not None,
        "has_hag": has_hag
    }
    if cls is not None:
        u, c = np.unique(cls, return_counts=True)
        summary["class_counts"] = {int(k): int(v) for k, v in zip(u, c)}
    if has_hag:
        hag_all = hag  # 全量统计，用于 percentiles
        q = np.nanpercentile(hag_all, [0, 5, 25, 50, 75, 95, 99, 100])
        summary["hag_stats"] = {
            "min": float(np.nanmin(hag_all)),
            "max": float(np.nanmax(hag_all)),
            "mean": float(np.nanmean(hag_all)),
            "std": float(np.nanstd(hag_all)),
            "percentiles": [float(x) for x in q.tolist()]
        }
    return xs, ys, zs, cls_s, hag_s, summary

def plot_hist(data, out_png: Path, title, xlabel, bins=80):
    plt.figure(figsize=(7,4))
    plt.hist(data[~np.isnan(data)], bins=bins)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_scatter_z_hag(z, hag, out_png: Path, title="Z vs HAG (sample)"):
    mask = (~np.isnan(hag))
    z2 = z[mask]; h2 = hag[mask]
    N = min(120000, h2.size)
    if h2.size > N:
        sel = np.random.default_rng(0).choice(h2.size, size=N, replace=False)
        h2 = h2[sel]; z2 = z2[sel]
    plt.figure(figsize=(6,6))
    plt.scatter(h2, z2, s=1, alpha=0.3)
    plt.xlabel("HeightAboveGround (m)")
    plt.ylabel("Z (m)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_class_bar(counts_dict, out_png: Path, title="Classification counts"):
    ser = pd.Series(counts_dict).sort_index()
    plt.figure(figsize=(7,4))
    ser.plot(kind="bar")
    plt.title(title); plt.xlabel("Class ID"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def quicklook_raster(raster_path: Path, out_png: Path, title, vmin=None, vmax=None, cmap="viridis"):
    with rasterio.open(raster_path) as src:
        arr = src.read(1, masked=True)
        # 限制到 2-98 分位，避免极端值
        if vmin is None or vmax is None:
            q2, q98 = np.nanpercentile(arr.compressed(), [2,98])
            vmin = q2 if vmin is None else vmin
            vmax = q98 if vmax is None else vmax
        plt.figure(figsize=(7,6))
        plt.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(label=title)
        plt.title(f"{title}\n{raster_path}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150); plt.close()

def overlay_lines_on_raster(raster_path: Path, lines_path: Path, towers_path: Path, out_png: Path, title="CHM with OSM power"):
    with rasterio.open(raster_path) as src:
        arr = src.read(1, masked=True)
        q2, q98 = np.nanpercentile(arr.compressed(), [2,98])
        fig, ax = plt.subplots(1,1, figsize=(7,6))
        ax.imshow(arr, vmin=q2, vmax=q98, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
        # 读取矢量
        if lines_path.exists():
            gL = gpd.read_file(lines_path)
            gL = gL.to_crs(src.crs)
            gL.plot(ax=ax, linewidth=0.6, edgecolor="white", alpha=0.9)
        if towers_path and towers_path.exists():
            gT = gpd.read_file(towers_path)
            gT = gT.to_crs(src.crs)
            gT.plot(ax=ax, markersize=6, color="red", alpha=0.9)
        plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--las", default="outputs/synth_osm_corridor_hag.las")
    ap.add_argument("--chm", default="outputs/aoi_chm.tif")
    ap.add_argument("--dtm", default="outputs/aoi_dtm.tif")
    ap.add_argument("--slope", default="outputs/aoi_slope.tif")
    ap.add_argument("--hillshade", default="outputs/aoi_hillshade.tif")
    ap.add_argument("--lines", default="data/osm/aoi_export_power_lines.geojson")
    ap.add_argument("--towers", default="data/osm/aoi_export_power_towers.geojson")
    ap.add_argument("--outdir", default="outputs/analysis")
    args = ap.parse_args()

    outdir = Path(args.outdir); ensure_outdir(outdir)

    # 1) 点云统计
    xs, ys, zs, cls_s, hag_s, summary = read_las_stats(Path(args.las))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 2) 图：HAG 直方图 / Z vs HAG 散点 / 分类统计
    if summary["has_hag"] and hag_s is not None:
        plot_hist(hag_s, outdir/"hist_hag.png", "HAG Histogram (sample)", "HeightAboveGround (m)")
        plot_scatter_z_hag(zs, hag_s, outdir/"scatter_z_hag.png")
    plot_hist(zs, outdir/"hist_z.png", "Z Histogram (sample)", "Z (m)")
    if summary["has_classification"] and "class_counts" in summary:
        plot_class_bar(summary["class_counts"], outdir/"bar_class_counts.png")

    # 3) 栅格快视图
    if Path(args.dtm).exists():
        quicklook_raster(Path(args.dtm), outdir/"quick_dtm.png", "DTM (m)", cmap="terrain")
    if Path(args.slope).exists():
        quicklook_raster(Path(args.slope), outdir/"quick_slope.png", "Slope (deg)", cmap="magma")
    if Path(args.hillshade).exists():
        quicklook_raster(Path(args.hillshade), outdir/"quick_hillshade.png", "Hillshade", cmap="gray")
    if Path(args.chm).exists():
        quicklook_raster(Path(args.chm), outdir/"quick_chm.png", "CHM (max HAG, m)")
        overlay_lines_on_raster(Path(args.chm), Path(args.lines), Path(args.towers), outdir/"chm_with_osm.png")

    # 4) 简易 HTML 报告
    html = []
    html.append("<html><head><meta charset='utf-8'><title>Corridor Analysis</title></head><body>")
    html.append("<h2>Corridor Analysis Report</h2>")
    html.append("<pre>"+json.dumps(summary, indent=2)+"</pre>")
    def img(p): return f"<div><img style='max-width:960px' src='{p}'/></div>"
    for name in ["hist_hag.png", "scatter_z_hag.png", "hist_z.png",
                 "quick_dtm.png", "quick_slope.png", "quick_hillshade.png",
                 "quick_chm.png", "chm_with_osm.png", "bar_class_counts.png"]:
        f = outdir/name
        if f.exists():
            html.append(f"<h3>{name}</h3>"+img(name))
    html.append("</body></html>")
    (outdir/"report.html").write_text("\n".join(html), encoding="utf-8")

    print(f"[OK] Wrote figures & report to: {outdir}")

if __name__ == "__main__":
    main()
