#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import numpy as np
import laspy
import rasterio

def main():
    ap = argparse.ArgumentParser(description="Add HeightAboveGround from DTM to LAS/LAZ using raster sampling.")
    ap.add_argument("--in", dest="inp", required=True, help="Input LAS/LAZ in UTM (same CRS as DTM)")
    ap.add_argument("--dtm", dest="dtm", required=True, help="DTM GeoTIFF (meters)")
    ap.add_argument("--out", dest="out", required=True, help="Output LAS with ExtraBytes: HeightAboveGround")
    ap.add_argument("--nodata_fill", type=float, default=0.0, help="Value to use when DTM has NODATA")
    ap.add_argument("--chunk", type=int, default=2_000_000, help="Chunk size for streaming")
    args = ap.parse_args()

    in_path = Path(args.inp); out_path = Path(args.out); dtm_path = Path(args.dtm)

    # Open input LAS/LAZ (read all header / VLR / CRS)
    las = laspy.read(in_path)

    # Prepare ExtraBytes dimension
    if "HeightAboveGround" not in las.point_format.extra_dimension_names:
        las.add_extra_dim(
            laspy.ExtraBytesParams(name="HeightAboveGround", type=np.float32, description="Z - DTM")
        )

    # Open DTM (assumed same CRS as LAS coordinates, i.e., UTM meters)
    with rasterio.open(dtm_path) as src:
        nodata = src.nodata
        # Chunked processing to reduce memory
        N = las.header.point_count
        chunk = max(1, int(args.chunk))
        for start in range(0, N, chunk):
            end = min(N, start + chunk)
            xs = las.x[start:end]  # float coordinates in meters
            ys = las.y[start:end]
            zs = las.z[start:end]

            # rasterio expects (row, col) via index(), or use sample() with (x, y) in map coords
            # sample returns iterator of arrays per band; our DTM is single band
            vals = np.array([val[0] for val in src.sample(zip(xs, ys))], dtype=np.float32)

            # handle NODATA
            if nodata is not None:
                mask = np.isclose(vals, nodata) | np.isnan(vals)
                vals[mask] = args.nodata_fill

            hag = (zs - vals).astype(np.float32)
            las["HeightAboveGround"][start:end] = hag

    # Write out (preserve CRS/VLR; laspy keeps header metadata)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(out_path)
    print(f"OK: wrote {out_path} with ExtraBytes 'HeightAboveGround'.")

if __name__ == "__main__":
    main()
