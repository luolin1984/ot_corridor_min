import argparse, os
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_pairs(pairs):
    d={}
    for s in pairs:
        k,v = s.split("=",1)
        d[k]=v
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--corridor", required=True)
    ap.add_argument("--corridor_layer", default="corridor")
    ap.add_argument("--towers", required=True)
    ap.add_argument("--tower_buf", type=float, default=30.0)
    ap.add_argument("--vectors", nargs="+", required=True, help="Name=gpkg pairs")
    ap.add_argument("--bbox", nargs=4, type=float, required=True, help="xmin ymin xmax ymax in EPSG:32649")
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    vecs = parse_pairs(args.vectors)
    # 假设 ds 是之前读取的矢量数据，确保 crs 已经设置
    xmin, ymin, xmax, ymax = 748800, 2606900, 750000, 2608100

    with rasterio.open(args.rgb) as ds:
        win = from_bounds(xmin,ymin,xmax,ymax, transform=ds.transform)
        img = ds.read([1,2,3], window=win).astype("float32")  # (3,h,w)
        img = img.transpose(1,2,0)
        # 简单拉伸
        p2, p98 = (img.reshape(-1,3).min(axis=0), img.reshape(-1,3).max(axis=0))
        img = (img - p2) / (p98 - p2 + 1e-6)
        img = img.clip(0,1)

        fig, ax = plt.subplots(1,1, figsize=(7,7), dpi=300)
        ax.imshow(img)
        ax.set_axis_off()

        # corridor outline
        cor = gpd.read_file(args.corridor, layer=args.corridor_layer).to_crs(ds.crs)
        # 创建一个包含边界框的 GeoDataFrame
        bbox = box(xmin, ymin, xmax, ymax)
        geo_series = gpd.GeoSeries([bbox], crs=ds.crs)

        # 将这个边界框应用到你的 `corridor` 数据
        cor_clip = cor.clip(geo_series)
        #cor_clip = cor.clip(gpd.GeoSeries([gpd.GeoSeries.from_bbox((xmin,ymin,xmax,ymax)).unary_union], crs=ds.crs))
        cor_clip.boundary.plot(ax=ax, linewidth=1.2, alpha=0.8)

        # towers buffer
        tw = gpd.read_file(args.towers).to_crs(ds.crs)
        tw["geometry"] = tw.geometry.buffer(args.tower_buf)
        #tw_clip = tw.clip(gpd.GeoSeries([gpd.GeoSeries.from_bbox((xmin,ymin,xmax,ymax)).unary_union], crs=ds.crs))
        # 将这个边界框应用到你的 'tw' 数据（假设 'tw' 是一个 GeoDataFrame）
        tw_clip = tw.clip(geo_series)
        if len(tw_clip) > 0:
            tw_clip.boundary.plot(ax=ax, linewidth=1.0, alpha=0.8)

        # method vectors overlay
        for name, gpkg in vecs.items():
            gdf = gpd.read_file(gpkg).to_crs(ds.crs)
            gdf_clip = gdf.cx[xmin:xmax, ymin:ymax]
            if len(gdf_clip) == 0:
                continue
            gdf_clip.boundary.plot(ax=ax, linewidth=1.2, alpha=0.9, label=name)

        ax.legend(loc="lower left", frameon=True)
        os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.out_png, bbox_inches="tight")
        print("[OK] wrote:", args.out_png)

if __name__ == "__main__":
    main()