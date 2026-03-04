import argparse
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out_gpkg", required=True)
    ap.add_argument("--layer", default="pred")
    ap.add_argument("--min_area", type=float, default=25.0)  # m^2
    args = ap.parse_args()

    with rasterio.open(args.mask) as ds:
        m = ds.read(1)
        tfm = ds.transform
        crs = ds.crs
        pix_area = abs(tfm.a * tfm.e)

        geoms = []
        vals = []
        for geom, val in shapes(m, mask=(m == 1), transform=tfm):
            if int(val) != 1:
                continue
            poly = shape(geom)
            if poly.is_empty:
                continue
            if poly.area < args.min_area:
                continue
            geoms.append(poly)
            vals.append(1)

    gdf = gpd.GeoDataFrame({"DN": vals}, geometry=geoms, crs=crs)
    gdf.to_file(args.out_gpkg, layer=args.layer, driver="GPKG")
    print(f"[OK] {args.out_gpkg} (layer={args.layer}, n={len(gdf)})")

if __name__ == "__main__":
    main()