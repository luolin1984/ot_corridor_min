# 输入：outputs/semseg_risk.tif（或你的 RISK_TIF）、CHM（取高程作为 z），以及 corridor/lines 作为空间约束。
# 流程：阈值/开闭运算 → 骨架化（skimage.morphology.skeletonize）→ 追踪骨架段 → 分段 RANSAC 拿初始 → least_squares 拟合悬链线，按 span 输出多段曲线 + RMSE。
# 输出：outputs/conductors.gpkg（图层 conductors，字段：span_id, a, x0, y0, rmse）。
import argparse, os
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, LineString, mapping
from shapely.ops import linemerge
from skimage.morphology import skeletonize, binary_opening, binary_closing, disk
from scipy.optimize import least_squares
import fiona
from fiona.crs import CRS

def catenary(params, x):
    a, x0, y0 = params
    return y0 + a*np.cosh((x - x0)/a)

def fit_catenary(xs, ys):
    # 初值：a ~ span/2，x0 ~ 中点，y0 ~ min
    xmid = 0.5*(xs.min()+xs.max())
    ymid = ys.min()
    a0 = max((xs.max()-xs.min())/2, 10.0)
    p0 = np.array([a0, xmid, ymid])
    def resid(p): return catenary(p, xs) - ys
    res = least_squares(resid, p0, bounds=([1.0, xs.min()-1000, ys.min()-1000],[1e5, xs.max()+1000, ys.max()+1000]))
    rmse = np.sqrt(np.mean(res.fun**2))
    return res.x, rmse

def raster_to_mask(path):
    with rasterio.open(path) as src:
        m = src.read(1).astype(bool)
        tr = src.transform; crs = src.crs
    m = binary_opening(m, disk(1))
    m = binary_closing(m, disk(1))
    return m, tr, crs

def skel_lines(mask, transform):
    sk = skeletonize(mask).astype(np.uint8)
    geoms = []
    for geom, val in shapes(sk, mask=sk, transform=transform):
        if val == 0: continue
        shp = shape(geom).buffer(0)
        if shp.is_empty: continue
        # 转成线：取骨架像素中心串连
        if shp.geom_type == 'Polygon':
            shp = shp.exterior
        geoms.append(shp)
    if not geoms: return []
    merged = linemerge(geoms)
    if merged.geom_type == 'MultiLineString':
        return list(merged.geoms)
    return [merged]

def write_gpkg(out, crs, feats):
    schema = {
        'geometry':'LineString',
        'properties':{'span_id':'int','a':'float','x0':'float','y0':'float','rmse':'float'}
    }
    if os.path.exists(out): os.remove(out)
    with fiona.open(out, 'w', driver='GPKG', schema=schema, crs=CRS.from_wkt(crs.to_wkt()) if hasattr(crs,'to_wkt') else crs) as dst:
        for f in feats: dst.write(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask', default='outputs/semseg_risk.tif')  # 二值前景
    ap.add_argument('--chm', default='outputs/aoi_chm.tif')       # 用作 z 估计（近似）
    ap.add_argument('--out', default='outputs/conductors.gpkg')
    args = ap.parse_args()

    m, tr, crs = raster_to_mask(args.mask)
    lines = skel_lines(m, tr)
    if not lines:
        print("No skeleton lines found."); return

    feats = []
    # 简化：对每条骨架沿主轴展为 x，CHM 取截线高度作为 y，拟合悬链线
    with rasterio.open(args.chm) as C:
        for i, ln in enumerate(lines, 1):
            # 采样若干点
            coords = np.array(ln.coords)
            if len(coords) < 10: continue
            # 局部坐标：把线沿着最大方差方向当 x
            xs, ys = coords[:,0], coords[:,1]
            # 取像素索引采样 chm
            rows, cols = (~tr) * (xs, ys)
            rows = rows.astype(int); cols = cols.astype(int)
            rows = np.clip(rows, 0, C.height-1); cols = np.clip(cols, 0, C.width-1)
            z = C.read(1)[rows, cols].astype(float)
            # 把“水平距离”当作 x，“高度 z”当作 y，拟合悬链线（二维近似）
            dist = np.cumsum(np.r_[0, np.linalg.norm(np.diff(coords, axis=0), axis=1)])
            if len(dist) != len(z): dist = dist[:len(z)]
            if len(dist) < 10: continue
            params, rmse = fit_catenary(dist, z[:len(dist)])
            feats.append({
                'geometry': LineString(coords),
                'properties': {'span_id': i, 'a': float(params[0]), 'x0': float(params[1]), 'y0': float(params[2]), 'rmse': float(rmse)}
            })

    write_gpkg(args.out, crs, feats)
    print("[OK] Wrote", args.out, "with", len(feats), "spans")

if __name__ == '__main__':
    main()
