#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import laspy, geopandas as gpd
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN

def main():
    ap = argparse.ArgumentParser("Vectorize predicted corridor from PredLabel")
    ap.add_argument("--las_pred", required=True, help="LAS with PredLabel (or Classification)")
    ap.add_argument("--out_towers", required=True)
    ap.add_argument("--out_corridor", required=True)
    ap.add_argument("--epsg", type=int, default=None, help="EPSG for output GeoJSON")
    ap.add_argument("--tower_eps", type=float, default=12.0, help="DBSCAN eps (m)")
    ap.add_argument("--tower_min", type=int, default=8, help="DBSCAN min_samples")
    ap.add_argument("--corridor_buf", type=float, default=25.0, help="buffer half-width (m)")
    args = ap.parse_args()

    las = laspy.read(args.las_pred)
    xs = np.asarray(las.x, dtype=np.float64)
    ys = np.asarray(las.y, dtype=np.float64)

    # 取预测标签
    if "PredLabel" in las.point_format.extra_dimension_names:
        labels = np.asarray(las["PredLabel"]).astype(np.int32)
    else:
        labels = np.asarray(las.classification).astype(np.int32)

    # 塔（15）
    towers_xy = np.stack([xs[labels==15], ys[labels==15]], axis=1)
    tower_pts = []
    if len(towers_xy) > 0:
        db = DBSCAN(eps=args.tower_eps, min_samples=args.tower_min).fit(towers_xy)
        for cid in sorted(set(db.labels_) - {-1}):
            P = towers_xy[db.labels_==cid]
            cx, cy = P.mean(axis=0)
            tower_pts.append(Point(cx, cy))
    gdf_t = gpd.GeoDataFrame(geometry=tower_pts, crs=f"EPSG:{args.epsg}" if args.epsg else None)

    # 导线（14）→ 走廊面：点集凸包 + 缓冲
    wire_xy = np.stack([xs[labels==14], ys[labels==14]], axis=1)
    corridor = None
    if len(wire_xy) >= 3:
        mp = MultiPoint([Point(xy) for xy in wire_xy])
        hull = mp.convex_hull
        corridor = hull.buffer(args.corridor_buf, join_style=1, cap_style=2)
    gdf_c = gpd.GeoDataFrame(geometry=[corridor] if corridor else [], crs=gdf_t.crs)

    # 写出
    Path(args.out_towers).parent.mkdir(parents=True, exist_ok=True)
    gdf_t.to_file(args.out_towers, driver="GeoJSON")
    gdf_c.to_file(args.out_corridor, driver="GeoJSON")
    print(f"OK: towers -> {args.out_towers}  | corridor -> {args.out_corridor}")

if __name__ == "__main__":
    main()
