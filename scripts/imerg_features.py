#!/usr/bin/env python
"""Extract precipitation features from one month of raw IMERG HDF5 granules.

One row per feature, one CSV per month. Detection is a plain threshold +
connected-components pass on the precipitation field; the artificial swath is
applied *afterwards* and only ever annotates a feature, never selects it. So a
single run supports any later swath-width analysis:

  * `cross_swath_extent_km` is the feature's extent perpendicular to the swath
    axis. A feature fits inside a swath of width W iff this is <= W. It does not
    depend on where the strip seams happen to fall, so you can re-evaluate any
    width from the CSV without reprocessing.
  * `n_swaths` / `crosses_swath_boundary` describe the *particular* tiling laid
    down here (seams at a fixed phase). A feature narrower than the swath still
    crosses a seam if it straddles one, so these are seam-phase dependent.

In-swath accounting (for the fair DYAMOND-vs-IMERG comparison)
-------------------------------------------------------------
IMERG is a full global grid here, sampled with the SAME artificial swath as the
DYAMOND worker, so the same "how much of the observed feature sits against the
swath edge" fraction applies and is directly comparable between datasets. GPM
only ever sees the part of a feature inside one swath, so each feature reports
the portion in its DOMINANT strip (what one overpass would observe) and how much
of that portion touches a seam:

  * px_in_swath (runner column)      -- pixels in the dominant strip (denominator);
                                        auto-emitted next to swath_id.
  * swath_edge_px_in_dominant (stat) -- of those, pixels touching a seam (numerator).
  * swath_edge_frac_in_dominant      -- numerator / denominator.

Cut on swath_edge_frac_in_dominant (e.g. < 0.05) to keep features that sit
mostly inside the swath; apply the identical cut to the DYAMOND CSVs. The
whole-feature swath_edge_px (both sides of every seam) is kept for comparison
and is always >= swath_edge_px_in_dominant. Requires a gridfeatures build with
dominant_swath / swath_edge_pixels_in_dominant; an older build raises
AttributeError in build_statistics (fails fast, before any file is written).

Units: /Grid/precipitation in IMERG V07 is ALREADY a rate in mm/hr (verified:
units attribute == "mm/hr"). The old 6-hourly netCDF stored "mm 6h-1" and this
worker divided by 6; that conversion is gone. Because the data are rates,
--threshold 1.0 means the same thing in both accumulation modes, and the hourly
mode is a plain mean of two rates -- so these CSVs stay comparable to the
DYAMOND output under the shared threshold and the shared
swath_edge_frac_in_dominant < 0.05 cut.

Accumulation modes (--mode):
    halfhour -- every granule is its own timestep, 48/day
    hourly   -- mean of the :00 and :30 granules, 24/day

The month must be complete: a missing granule raises before any work is done,
because a short month would desynchronise time_index between the monthly CSVs
and break the contiguity assertion in verify_outputs.py.

Usage:
    imerg_features.py <YYYYMM> <outfile.csv> [options]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import ndimage

import gridfeatures as gf
from gridfeatures import stats
from gridfeatures.swath import swath_index, EARTH_RADIUS_KM as R
from gridfeatures.runner import _area_from_2d

import imerg_hdf5_io as io


# --- custom statistics ------------------------------------------------------
def px_above(threshold):
    """Number of feature pixels at or above a rate threshold."""
    def _f(f):
        return int((f.values >= threshold).sum())
    _f.__name__ = f"px_ge_{threshold:g}"
    return _f


def n_cores(threshold, connectivity):
    """Number of distinct contiguous cores at or above a rate threshold."""
    structure = ndimage.generate_binary_structure(2, connectivity)

    def _f(f):
        sub = f.local_field()
        core = f.local_mask() & (sub >= threshold) & ~np.isnan(sub)
        if not core.any():
            return 0
        _, n = ndimage.label(core, structure=structure)
        return int(n)
    _f.__name__ = f"n_cores_{threshold:g}"
    return _f


def extent_km(coord2d):
    """Peak-to-peak extent of the feature along a precomputed 2D coordinate."""
    def _f(f):
        return float(np.ptp(coord2d[f.rows, f.cols]))
    return _f


def fits_in_swath(coord2d, width_km):
    """True if the feature's cross-swath extent fits inside `width_km`.

    Seam-phase independent: asks whether *some* swath of this width could
    contain the feature, not whether this particular tiling does.
    """
    def _f(f):
        return bool(np.ptp(coord2d[f.rows, f.cols]) <= width_km)
    return _f


def build_statistics(perp2d, along2d, width_km, connectivity):
    return {
        # size
        "size_px":                stats.size,
        "area_km2":               stats.area_km2,
        # location
        "centroid_lat":           stats.centroid_lat,
        "centroid_lon":           stats.centroid_lon,
        # intensity
        "max_precip_mm_hr":       stats.max,
        "mean_precip_mm_hr":      stats.mean,
        "total_precip_mm_hr_km2": stats.total,     # sum(rate * cell_area)
        # cores
        "px_ge_5mmhr":            px_above(5.0),
        "px_ge_10mmhr":           px_above(10.0),
        "n_cores_5mmhr":          n_cores(5.0, connectivity),
        "n_cores_10mmhr":         n_cores(10.0, connectivity),
        "largest_core_5mmhr_px":  stats.core_size(5.0, connectivity=connectivity),
        "largest_core_10mmhr_px": stats.core_size(10.0, connectivity=connectivity),
        # swath geometry -- seam-phase independent
        "cross_swath_extent_km":  extent_km(perp2d),
        "along_swath_extent_km":  extent_km(along2d),
        "fits_in_swath":          fits_in_swath(perp2d, width_km),
        # edges
        "swath_edge_px":          stats.swath_edge_pixels,   # whole feature, both sides of every seam
        # in-swath accounting (dominant strip = what one overpass sees). px_in_swath
        # is auto-emitted by the runner; swath_edge_frac_in_dominant matches the
        # DYAMOND-side column, so a single cut (e.g. < 0.05) applies to both datasets.
        "swath_edge_px_in_dominant":   stats.swath_edge_pixels_in_dominant,    # numerator
        "swath_edge_frac_in_dominant": stats.swath_edge_fraction_in_dominant,  # numerator/px_in_swath
        "domain_edge_px":         stats.boundary_pixels,     # adjacent to band edge / NaN
        "touches_domain_edge":    stats.touches_boundary,
    }


def swath_coords(lats2d, lons2d, angle_deg, origin):
    """Along- and cross-swath coordinates (km) in the same projection as swath_index."""
    lat0, lon0 = origin
    x = R * (np.radians(lons2d) - np.radians(lon0)) * np.cos(np.radians(lat0))  # east
    y = R * (np.radians(lats2d) - np.radians(lat0))                             # north
    th = np.radians(angle_deg)
    along = x * np.cos(th) + y * np.sin(th)
    perp = -x * np.sin(th) + y * np.cos(th)
    return along, perp


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("yyyymm", help="month to process, e.g. 202202")
    p.add_argument("outfile")
    p.add_argument("--archive", default="/data05/IMERG_GPM",
                   help="READ-ONLY granule archive root")
    p.add_argument("--mode", default="halfhour", choices=io.MODES,
                   help="halfhour: 48 timesteps/day; hourly: mean of :00 and :30")
    p.add_argument("--lon-convention", default="0-360", choices=io.LON_CONVENTIONS,
                   help="0-360 matches the old 6-hourly netCDF grid and keeps the "
                        "swath seams where output_inswathcounted/ has them")
    p.add_argument("--variable", default="precipitation")
    p.add_argument("--lat-min", type=float, default=-20.0)
    p.add_argument("--lat-max", type=float, default=20.0)
    p.add_argument("--threshold", type=float, default=1.0, help="mm/hr")
    p.add_argument("--min-size", type=int, default=10, help="pixels")
    p.add_argument("--connectivity", type=int, default=2, choices=(1, 2))
    p.add_argument("--swath-width-km", type=float, default=250.0)
    p.add_argument("--swath-angle-deg", type=float, default=65.0)
    p.add_argument("--origin-lat", type=float, default=0.0)
    p.add_argument("--origin-lon", type=float, default=180.0)
    p.add_argument("--max-timesteps", type=int, default=None, help="smoke test")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.outfile) and not args.overwrite:
        print(f"SKIP (exists): {args.outfile}", flush=True)
        return 0

    t_start = time.time()
    origin = (args.origin_lat, args.origin_lon)

    year, month = int(args.yyyymm[:4]), int(args.yyyymm[4:6])

    # Index and group BEFORE touching the grid or building statistics: a gap in
    # the month must fail here, with nothing written.
    index = io.scan_archive(args.archive, year, month)
    timesteps = io.build_timesteps(index, year, month, args.mode)
    ntime = len(timesteps)

    g_lats, lons, roll = io.read_grid(timesteps[0][1][0], args.lon_convention)

    # contiguous latitude band (lat is ascending)
    idx = np.where((g_lats >= args.lat_min) & (g_lats <= args.lat_max))[0]
    if idx.size == 0:
        raise SystemExit(f"no latitudes in [{args.lat_min}, {args.lat_max}]")
    lat_slice = slice(int(idx[0]), int(idx[-1]) + 1)
    lats = g_lats[lat_slice]

    lats2d, lons2d = np.meshgrid(lats, lons, indexing="ij")
    area = _area_from_2d(lats2d, lons2d)
    swath = swath_index(lats2d, lons2d, args.swath_width_km, args.swath_angle_deg,
                        origin=origin)
    along2d, perp2d = swath_coords(lats2d, lons2d, args.swath_angle_deg, origin)

    config = gf.Config(
        files=args.yyyymm, variable=args.variable,
        threshold=args.threshold, min_size=args.min_size,
        comparison=">", connectivity=args.connectivity,
        use_swath=True,
        swath_width_km=args.swath_width_km, swath_angle_deg=args.swath_angle_deg,
        statistics=build_statistics(perp2d, along2d, args.swath_width_km,
                                    args.connectivity),
    )

    print(f"{args.yyyymm} [{args.mode}]: {ntime} timesteps, lon {args.lon_convention}, "
          f"band {lats[0]:.2f}..{lats[-1]:.2f} ({lats.size} rows x {lons.size} cols), "
          f"{swath.max()-swath.min()+1} strips of {args.swath_width_km:g} km "
          f"@ {args.swath_angle_deg:g} deg", flush=True)

    n_steps = ntime if args.max_timesteps is None else min(ntime, args.max_timesteps)
    rows = []
    for i in range(n_steps):
        tval, paths = timesteps[i]
        # Already mm/hr; fills are NaN; band sliced in memory (a chunk spans all
        # 1800 lats, so an on-disk band slice would save no I/O).
        field = io.load_field(paths, lat_slice, roll)

        feats = gf.extract_features(
            field, lats2d, lons2d, config, source=os.path.basename(paths[0]),
            time=tval, time_index=i, area=area, swath=swath,
        )
        rows.extend(gf.feature_row(f, config) for f in feats)

        if (i + 1) % 20 == 0 or i + 1 == n_steps:
            print(f"  [{i+1:4d}/{n_steps}] {len(rows):6d} features  "
                  f"{time.time()-t_start:6.1f}s", flush=True)

    df = pd.DataFrame(rows)
    # Mode is not recoverable from the other columns, and the two runs share a
    # filename pattern, so stamp it on every row.
    df["accum_mode"] = args.mode
    os.makedirs(os.path.dirname(os.path.abspath(args.outfile)), exist_ok=True)
    tmp = args.outfile + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, args.outfile)   # atomic: a partial file never looks complete

    dt = time.time() - t_start
    print(f"OK {args.outfile}: {len(df)} features from {n_steps} timesteps in {dt:.1f}s",
          flush=True)
    if len(df):
        print(f"   fits_in_swath ({args.swath_width_km:g} km): "
              f"{int(df.fits_in_swath.sum())}/{len(df)} "
              f"({100*df.fits_in_swath.mean():.1f}%)", flush=True)
        print(f"   crosses a seam:                {int(df.crosses_swath_boundary.sum())}/{len(df)} "
              f"({100*df.crosses_swath_boundary.mean():.1f}%)", flush=True)
        print(f"   mostly-in-swath (<5% edge):    "
              f"{int((df.swath_edge_frac_in_dominant < 0.05).sum())}/{len(df)} "
              f"({100*(df.swath_edge_frac_in_dominant < 0.05).mean():.1f}%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())