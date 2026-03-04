#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, sys, math, random, subprocess, shutil
from pathlib import Path

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union
import shapely
import laspy
from pyproj import CRS, Transformer

# ---------- utils ----------
def auto_utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6) + 1
    hemi = 326 if lat >= 0 else 327
    return hemi * 100 + zone  # 326xx or 327xx

def read_gdf(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    gdf = gdf.set_crs(gdf.crs or "EPSG:4326", allow_override=True)
    return gdf

def explode_lines_to_linestrings(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # shapely >= 2: explode via geodataframe.explode(index_parts=False)
    e = gdf.explode(index_parts=False)
    e = e[e.geometry.type.isin(["LineString", "LinearRing", "MultiLineString"])]
    # flatten multilines
    rows = []
    for _, r in e.iterrows():
        geom = r.geometry
        if isinstance(geom, LineString):
            rows.append(r)
        elif isinstance(geom, MultiLineString):
            for ls in geom.geoms:
                rr = r.copy()
                rr.geometry = ls
                rows.append(rr)
    return gpd.GeoDataFrame(rows, columns=e.columns, crs=e.crs)

def densify_linestring(ls: LineString, step: float) -> np.ndarray:
    # sample points every "step" meters along line
    if ls.length == 0:
        return np.empty((0, 2))
    n = max(2, int(math.ceil(ls.length / step)) + 1)
    dists = np.linspace(0, ls.length, n)
    coords = [ls.interpolate(d).coords[0] for d in dists]
    return np.asarray(coords)  # (N,2)

def random_points_in_polygon(poly: Polygon, n: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 2))
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    tries = 0
    while len(pts) < n and tries < n * 20:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        if poly.contains(Point(x, y)):
            pts.append((x, y))
        tries += 1
    return np.asarray(pts) if pts else np.empty((0, 2))

def grid_points_in_polygon(poly: Polygon, step: float) -> np.ndarray:
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx, maxx + step, step)
    ys = np.arange(miny, maxy + step, step)
    pts = []
    for x in xs:
        for y in ys:
            p = Point(x, y)
            if poly.contains(p):
                pts.append((x, y))
    return np.asarray(pts) if pts else np.empty((0, 2))

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

# ---------- LAS writer ----------
def write_las_xyzc(xyz: np.ndarray, cls: np.ndarray, epsg: int, out_path: Path):
    # xyz in meters (UTM), cls int (LAS classification)
    header = laspy.LasHeader(point_format=3, version="1.2")
    # scaling for centimeter precision
    header.x_scale = header.y_scale = 0.01
    header.z_scale = 0.01
    header.offsets = np.array([0.0, 0.0, 0.0])

    try:
        # laspy>=2：写入 CRS（若不可用则跳过）
        crs = CRS.from_epsg(int(epsg))
        if hasattr(header, "add_crs"):
            header.add_crs(crs)
    except Exception:
        pass

    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    # 其余维度给默认
    las.classification = cls.astype(np.uint8)
    ensure_dir(out_path)
    las.write(out_path)

# ---------- main synth ----------
def main():
    ap = argparse.ArgumentParser(description="Synthesize power corridor point cloud (minimal).")
    ap.add_argument("--aoi", required=True, help="AOI polygon (GeoJSON)")
    ap.add_argument("--lines", required=True, help="Power lines (GeoJSON)")
    ap.add_argument("--towers", default="", help="Power towers (GeoJSON, optional)")
    ap.add_argument("--out", required=True, help="Output LAS path")
    ap.add_argument("--width", type=float, default=50.0, help="Corridor half-width (m)")
    ap.add_argument("--ground-step", type=float, default=4.0, help="Ground grid step (m)")
    ap.add_argument("--veg-density", type=float, default=0.15, help="Vegetation density (pts/m^2)")
    ap.add_argument("--span", type=float, default=350.0, help="Nominal span length (m)")
    ap.add_argument("--sag", type=float, default=15.0, help="Sag amplitude (m)")
    ap.add_argument("--height", type=float, default=28.0, help="Typical wire/tower height above ground (m)")
    ap.add_argument("--phase-offset", type=float, default=1.5, help="Phase lateral offset (m)")
    args = ap.parse_args()

    aoi = read_gdf(Path(args.aoi))
    lines = read_gdf(Path(args.lines))
    towers = None
    if args.towers and Path(args.towers).exists():
        towers = read_gdf(Path(args.towers))

    # 统一到 EPSG:4326 以取中心点
    aoi = aoi.to_crs("EPSG:4326")
    cen = aoi.geometry.unary_union.centroid
    utm_epsg = auto_utm_epsg(cen.x, cen.y)

    # 投影到 UTM（米制）
    aoi_utm = aoi.to_crs(utm_epsg)
    lines = lines.to_crs(utm_epsg)
    lines = explode_lines_to_linestrings(lines)
    if towers is not None:
        towers = towers.to_crs(utm_epsg)

    # AOI 面
    aoi_poly = unary_union(aoi_utm.geometry)

    # 用走廊宽度对线路缓冲，再与 AOI 相交得到“工作走廊”
    buf = lines.buffer(args.width, cap_style=2)  # 平头
    corridor = unary_union(buf).intersection(aoi_poly)
    if corridor.is_empty:
        print("Corridor empty: AOI × buffered lines have no intersection.", file=sys.stderr)
        # 仍然输出空 LAS（0 点），避免流程中断
        write_las_xyzc(np.zeros((0,3)), np.zeros((0,), dtype=np.uint8), utm_epsg, Path(args.out))
        return

    # -------- 生成地面点（Class=2）--------
    ground_xy = grid_points_in_polygon(corridor, args.ground_step)
    ground_z = np.zeros((ground_xy.shape[0],), dtype=float)
    ground_cls = np.full((ground_xy.shape[0],), 2, dtype=np.uint8)

    # -------- 生成植被点（Class=5）--------
    # 基于走廊面积估计数量
    area = corridor.area  # m^2
    n_veg = int(area * args.veg_density)
    veg_xy = random_points_in_polygon(corridor, n_veg)
    veg_z = np.random.uniform(4.0, 15.0, size=(veg_xy.shape[0],))  # 简化：随机高度
    veg_cls = np.full((veg_xy.shape[0],), 5, dtype=np.uint8)

    # -------- 生成导线点（Class=14）--------
    wire_pts = []
    for _, r in lines.iterrows():
        geom = r.geometry
        if not isinstance(geom, LineString) or geom.length == 0:
            continue
        # 在 UTM 下按 10 m 间距采样
        xy = densify_linestring(geom, step=10.0)
        if xy.size == 0:
            continue
        # 弧垂近似：z = H - sag * 4 t (1-t)，t∈[0,1] 沿线累计距离归一化
        # 先计算每段累积距离
        seglen = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
        acc = np.cumsum(seglen)
        L = acc[-1] if acc[-1] > 0 else 1.0
        t = acc / L
        z = args.height - args.sag * (4 * t * (1 - t))
        # 多相横向偏移（简化：对每个点做切向法线偏移）
        if xy.shape[0] >= 2 and args.phase_offset != 0.0:
            tang = np.gradient(xy, axis=0)
            norms = np.stack([tang[:,1], -tang[:,0]], axis=1)
            nlen = np.linalg.norm(norms, axis=1, keepdims=True)
            nlen[nlen==0] = 1.0
            n = norms / nlen
            # 三相：-off, 0, +off
            for off in (-args.phase_offset, 0.0, args.phase_offset):
                wire_pts.append(np.c_[xy + n * off, z])
        else:
            wire_pts.append(np.c_[xy, z])

    if len(wire_pts) > 0:
        wire_xyz = np.vstack(wire_pts)
        # 仅保留处于 corridor ∩ AOI 的点
        mask = np.array([corridor.contains(Point(x,y)) for x,y,_ in wire_xyz], dtype=bool)
        wire_xyz = wire_xyz[mask]
        wire_cls = np.full((wire_xyz.shape[0],), 14, dtype=np.uint8)
    else:
        wire_xyz = np.zeros((0,3)); wire_cls = np.zeros((0,), dtype=np.uint8)

    # -------- 生成塔点（Class=15）--------
    if towers is not None and len(towers) > 0:
        txy = np.array([[g.x, g.y] for g in towers.geometry if isinstance(g, Point)])
        if txy.size > 0:
            txy = txy.reshape(-1, 2)
            # 塔高近似
            tz = np.full((txy.shape[0],), max(10.0, args.height), dtype=float)
            towers_xyz = np.c_[txy, tz]
            # 仅保留走廊内
            mask = np.array([corridor.contains(Point(x,y)) for x,y,_ in towers_xyz], dtype=bool)
            towers_xyz = towers_xyz[mask]
            towers_cls = np.full((towers_xyz.shape[0],), 15, dtype=np.uint8)
        else:
            towers_xyz = np.zeros((0,3)); towers_cls = np.zeros((0,), dtype=np.uint8)
    else:
        towers_xyz = np.zeros((0,3)); towers_cls = np.zeros((0,), dtype=np.uint8)

    # -------- 拼接所有类别 --------
    parts = []
    if ground_xy.size: parts.append((np.c_[ground_xy, ground_z], ground_cls))
    if veg_xy.size:    parts.append((np.c_[veg_xy,   veg_z],    veg_cls))
    if wire_xyz.size:  parts.append((wire_xyz,                   wire_cls))
    if towers_xyz.size:parts.append((towers_xyz,                 towers_cls))

    if not parts:
        print("No points generated (empty AOI/corridor?)", file=sys.stderr)
        write_las_xyzc(np.zeros((0,3)), np.zeros((0,), dtype=np.uint8), utm_epsg, Path(args.out))
        return

    xyz = np.vstack([p for p,_ in parts])
    cls = np.concatenate([c for _,c in parts])

    # 写 LAS（UTM）
    out_path = Path(args.out)
    write_las_xyzc(xyz, cls, utm_epsg, out_path)

    # 若用户把后缀写成 .laz，尝试用 pdal 压缩
    if out_path.suffix.lower() == ".laz":
        tmp_las = out_path.with_suffix(".las")
        write_las_xyzc(xyz, cls, utm_epsg, tmp_las)
        if shutil.which("pdal"):
            try:
                subprocess.run(["pdal", "translate", str(tmp_las), str(out_path)], check=True)
                tmp_las.unlink(missing_ok=True)
            except Exception as e:
                print(f"PDAL compress failed: {e}. Kept LAS at {tmp_las}", file=sys.stderr)

if __name__ == "__main__":
    main()
