# src/vectorize_fast.py
import argparse, os, json
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape, LineString, mapping
from shapely.ops import linemerge
from skimage.morphology import skeletonize
from scipy.signal import savgol_filter

def raster_to_polygons(risk_tif, min_area=25):
    """把值=1的像元转为面，并按面积过滤"""
    with rasterio.open(risk_tif) as src:
        arr = src.read(1)
        mask = arr == 1
        transform = src.transform
        geoms = []
        for geom, val in features.shapes(arr, mask=mask, transform=transform):
            if val != 1:
                continue
            poly = shape(geom)
            if poly.area >= min_area:  # 像素面积，已在 10m 分辨率下就是 m²
                geoms.append(poly)
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=src.crs)
    return gdf

def polygon_raster_skeleton(risk_tif):
    """在栅格上做骨架化，返回一个二值骨架栅格"""
    with rasterio.open(risk_tif) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs
    mask = arr == 1
    # skimage 要 bool
    skel = skeletonize(mask.astype(bool)).astype(np.uint8)
    return skel, transform, crs

def skeleton_to_lines(skel, transform, step=0.0):
    """把骨架像元转成多条 LineString"""
    # 把骨架像元的位置转成点，再按行走向连成线，这里做一个最简单的“每条骨架一条线”的近似
    # 更精细可以用 graph 追踪
    ys, xs = np.where(skel == 1)
    if len(xs) == 0:
        return []
    # 直接按行排序，适用于你这种走廊很窄、方向比较一致的情况
    order = np.argsort(ys)
    coords = []
    for i in order:
        x, y = xs[i], ys[i]
        X, Y = rasterio.transform.xy(transform, y, x)
        coords.append((X, Y))
    line = LineString(coords)
    if step and step > 0:
        # Douglas–Peucker 简化，公差单位是投影坐标的米
        line = line.simplify(step, preserve_topology=False)
    return [line]

def smooth_line(line: LineString, window=9, poly=2):
    """用 Savitzky–Golay 简单平滑一下线"""
    xs, ys = line.xy
    xs = np.array(xs); ys = np.array(ys)
    # window 必须小于等于点数且为奇数
    win = min(window, len(xs) if len(xs)%2==1 else len(xs)-1)
    if win < 5:
        return line
    xs_s = savgol_filter(xs, win, poly)
    ys_s = savgol_filter(ys, win, poly)
    return LineString(np.stack([xs_s, ys_s], axis=1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk", required=True, help="阈值后的风险栅格，如 outputs/aoi_risk_hag_gt6m.tif")
    ap.add_argument("--corridor", help="可选的走廊缓冲，用来裁剪面")
    ap.add_argument("--out-gpkg", required=True)
    ap.add_argument("--min-area", type=float, default=25.0)
    ap.add_argument("--skel-step", type=float, default=0.0,
                    help="骨架线简化公差(米)，0表示不简化")
    args = ap.parse_args()

    # 1) 栅格->面
    gdf_poly = raster_to_polygons(args.risk, min_area=args.min_area)
    if args.corridor and os.path.exists(args.corridor):
        cor = gpd.read_file(args.corridor)
        gdf_poly = gpd.overlay(gdf_poly, cor, how="intersection")

    # 2) 栅格骨架
    skel, transform, crs = polygon_raster_skeleton(args.risk)
    lines = skeleton_to_lines(skel, transform, step=args.skel_step)
    smooth_lines = [smooth_line(l) for l in lines]

    # 3) 写 GPKG
    os.makedirs(os.path.dirname(args.out_gpkg), exist_ok=True)
    gdf_poly.to_file(args.out_gpkg, layer="risk_poly", driver="GPKG")
    gdf_line = gpd.GeoDataFrame(geometry=smooth_lines, crs=crs)
    gdf_line["rmse"] = 0.0   # 这里先占位，后面真做拟合再写
    gdf_line.to_file(args.out_gpkg, layer="risk_centerline", driver="GPKG")
    print(f"[OK] written to {args.out_gpkg}")

if __name__ == "__main__":
    main()
