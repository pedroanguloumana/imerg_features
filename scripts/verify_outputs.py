"""Post-hoc invariant checks on the monthly feature CSVs.

Usage:
    verify_outputs.py [halfhour|hourly|inswathcounted]   (default: halfhour)

`inswathcounted` checks the original 6-hourly netCDF run. The raw-HDF5 runs
live in per-mode folders and carry an accum_mode column; the expected timestep
count per month depends on the mode, so it is checked explicitly -- a month
that came up short would otherwise pass the contiguity test while silently
holding fewer timesteps than it should.
"""

import calendar
import glob
import os
import sys

import pandas as pd

PROJECT = "/home1/pedro/Projects/imerg_precipitation_features"

TARGET = sys.argv[1] if len(sys.argv) > 1 else "halfhour"
if TARGET == "inswathcounted":
    OUT, STEPS_PER_DAY = f"{PROJECT}/output_inswathcounted", 4
else:
    OUT = f"{PROJECT}/output_{TARGET}_inswathcounted"
    STEPS_PER_DAY = {"halfhour": 48, "hourly": 24}[TARGET]

# These must match the values the run was submitted with (submit_imerg_jobs.sh
# defaults). They are not recorded in the CSV, so they are stated here: with
# the stale literals 250/4 the width and min-size checks fail against a run made
# at 245 km / min-size 5.
SWATH_WIDTH_KM = float(os.environ.get("SWATH_WIDTH_KM", 245))
MIN_SIZE = int(os.environ.get("MIN_SIZE", 5))

fs = sorted(glob.glob(f"{OUT}/imerg_features_*.csv"))
print(f"{TARGET}: {len(fs)} monthly CSVs in {os.path.basename(OUT)}\n")
if not fs:
    raise SystemExit(f"no CSVs found in {OUT}")

hdr = ("month", "rows", "tsteps", "min_px", "maxext_km", "fits%", "cross%")
print("{:8s} {:>9s} {:>7s} {:>7s} {:>10s} {:>7s} {:>7s}".format(*hdr))

frames = []
for f in fs:
    d = pd.read_csv(f)
    frames.append(d)
    m = f.split("_")[-1][:6]
    ts = sorted(d.time_index.unique())
    print("{:8s} {:9,d} {:7d} {:7d} {:10.0f} {:6.1f}% {:6.1f}%".format(
        m, len(d), len(ts), int(d.size_px.min()), d.cross_swath_extent_km.max(),
        100 * d.fits_in_swath.mean(), 100 * d.crosses_swath_boundary.mean()))
    assert ts == list(range(len(ts))), f"{m}: gap in time_index"
    # Contiguity alone cannot catch a month that is short at the END, which is
    # exactly what a missing granule would produce.
    expect = calendar.monthrange(int(m[:4]), int(m[4:6]))[1] * STEPS_PER_DAY
    assert len(ts) == expect, f"{m}: {len(ts)} timesteps, expected {expect}"

df = pd.concat(frames, ignore_index=True)
print(f"\nTOTAL {len(df):,} features across {len(fs)} months")

print("\n--- invariants across all months ---")
checks = [
    ("n_swaths==1 => fits_in_swath",   bool((~(df.n_swaths == 1) | df.fits_in_swath).all())),
    (f"fits_in_swath == extent<={SWATH_WIDTH_KM:g}",
        bool((df.fits_in_swath == (df.cross_swath_extent_km <= SWATH_WIDTH_KM)).all())),
    ("crosses => swath_edge_px>0",     bool((~df.crosses_swath_boundary | (df.swath_edge_px > 0)).all())),
    ("largest_core5 <= px_ge_5",       bool((df.largest_core_5mmhr_px <= df.px_ge_5mmhr).all())),
    ("largest_core10 <= px_ge_10",     bool((df.largest_core_10mmhr_px <= df.px_ge_10mmhr).all())),
    ("px_ge_10 <= px_ge_5 <= size_px", bool(((df.px_ge_10mmhr <= df.px_ge_5mmhr) & (df.px_ge_5mmhr <= df.size_px)).all())),
    ("n_cores_5==0 iff px_ge_5==0",    bool(((df.n_cores_5mmhr == 0) == (df.px_ge_5mmhr == 0)).all())),
    ("n_cores_10==0 iff px_ge_10==0",  bool(((df.n_cores_10mmhr == 0) == (df.px_ge_10mmhr == 0)).all())),
    ("max>=1.0 mm/hr, max>=mean",      bool(((df.max_precip_mm_hr >= 1.0) & (df.max_precip_mm_hr >= df.mean_precip_mm_hr)).all())),
    ("centroids within +-20 lat",      bool(df.centroid_lat.between(-20, 20).all())),
    (f"size_px.min() >= {MIN_SIZE}",   int(df.size_px.min()) >= MIN_SIZE),
    ("feature_id unique",              bool(df.feature_id.is_unique)),
    ("no NaNs anywhere",               not bool(df.isna().any().any())),
    # --- in-swath accounting (dominant strip = what one overpass observes) ----
    ("px_in_swath >= 1",               bool((df.px_in_swath >= 1).all())),
    ("px_in_swath <= size_px",         bool((df.px_in_swath <= df.size_px).all())),
    ("edge_in_dom <= px_in_swath",     bool((df.swath_edge_px_in_dominant <= df.px_in_swath).all())),
    ("edge_in_dom <= swath_edge_px",   bool((df.swath_edge_px_in_dominant <= df.swath_edge_px).all())),
    ("frac == edge_in_dom/px_in_swath",
        bool((abs(df.swath_edge_frac_in_dominant - df.swath_edge_px_in_dominant / df.px_in_swath) < 1e-9).all())),
    ("frac in [0,1]",                  bool(df.swath_edge_frac_in_dominant.between(0.0, 1.0).all())),
]

if TARGET != "inswathcounted":
    checks += [
        # A single mode per output folder: a stray CSV from the other mode would
        # otherwise be silently concatenated into the analysis.
        ("accum_mode column present",  "accum_mode" in df.columns),
        (f"accum_mode == '{TARGET}'",  bool((df.accum_mode == TARGET).all())),
        # Rates, not accumulations: the mm/hr values must not carry a stale /6.
        # Detection is `>` THRESHOLD, so every retained max is above it.
        ("max_precip > 1.0 mm/hr",     bool((df.max_precip_mm_hr > 1.0).all())),
    ]
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")

print("\n--- swath-width study, from this ONE run ---")
widths = [150, 200, 245, 250, 300, 400, 600, 1000]
counts = [int((df.cross_swath_extent_km <= W).sum()) for W in widths]
for W, n in zip(widths, counts):
    print(f"  width {W:4d} km : {n:7,d}/{len(df):,} fit  ({100*n/len(df):5.1f}%)")
print(f"  monotone in W: {counts == sorted(counts)}")

print("\n--- in-swath edge-fraction study (pick your 'mostly inside' cut) ---")
# swath_edge_frac_in_dominant = fraction of the observed (dominant-strip) pixels
# that sit against a swath seam. Apply the SAME cut to the DYAMOND CSVs.
for c in [0.01, 0.02, 0.05, 0.10, 0.20]:
    n = int((df.swath_edge_frac_in_dominant < c).sum())
    print(f"  frac < {c:>4.2f} : {n:7,d}/{len(df):,} kept  ({100*n/len(df):5.1f}%)")
n0 = int((df.swath_edge_frac_in_dominant == 0).sum())
print(f"  frac == 0  : {n0:7,d}/{len(df):,} fully interior to a strip ({100*n0/len(df):5.1f}%)")

edge = int(df.touches_domain_edge.sum())
print(f"\nband-edge-truncated features: {edge:,} ({100*edge/len(df):.2f}%) "
      "-- drop these from size distributions")
print(f"size_px: min {df.size_px.min()}  median {int(df.size_px.median())}  max {df.size_px.max():,}")

assert all(ok for _, ok in checks), "INVARIANT FAILED"
print("\nALL CHECKS PASSED")