#!/usr/bin/env python
"""Load raw half-hourly IMERG V07 HDF5 granules as mm/hr fields.

Replaces the old monthly-netCDF load path. Everything downstream of the field
(detection, statistics, swath annotation) is unchanged; only the read and the
unit handling live here.

Why this is a separate module
-----------------------------
Filename parsing, granule indexing and hourly pairing are pure functions of
strings and dicts, so they unit-test without touching /data05.

Empirically verified against
/data05/IMERG_GPM/2022/3B-HHR.MS.MRG.3IMERG.20220101-S000000-E002959.0000.V07B.HDF5:

  * /Grid/precipitation units = "mm/hr" -- ALREADY A RATE. The netCDF path
    stored "mm 6h-1" and divided by 6; that conversion is gone. A rate means
    --threshold 1.0 keeps its meaning in both accumulation modes, and hourly is
    a plain arithmetic mean of two rates.
  * _FillValue = -9999.9 (not NaN). Must become NaN on load: n_cores and
    stats.boundary_pixels test ~np.isnan, so a raw -9999.9 would pass as valid
    weak precipitation and corrupt the domain-edge accounting.
  * shape (time=1, lon=3600, lat=1800) -- lon-major, like the netCDF.
  * lat ascending -89.95..89.95, lon ascending, both 0.1 deg.
  * Chunks {1, 145, 1800}, deflate level 6. A chunk spans ALL 1800 latitudes,
    so slicing the 20S-20N band on disk still decompresses every chunk it
    touches -- no I/O saved. Read the full (lon, lat) block and slice in
    memory, exactly as the netCDF path did.

Longitude convention
--------------------
IMERG HDF5 is -180..180. The old monthly netCDF was 0..360 (verified:
0.05..359.95). Same grid, rolled by 1800 columns. The artificial swath tiling
is anchored at origin lon 180.0, so on a -180..180 grid the seams land in
different places and the CSVs stop being comparable to output_inswathcounted/.
Default is therefore "0-360"; "pm180" (the native -180..180 grid) is available
but only for a run not meant to be compared with the existing output.

Note that connected-component labelling is not periodic in longitude, so a
feature never spans the grid's own seam -- it is split there instead. Under
"0-360" that split sits at Greenwich, which is where the existing
output_inswathcounted/ run put it too, so the two remain like-for-like.

Archive quirks handled here
---------------------------
  * /data05/IMERG_GPM/<YYYY>/ is flat, but contains non-granule files
    (imerg_YYYY.sh) and re-download duplicates (*.HDF5.1). Only the canonical
    filename pattern is accepted.
  * Some granules are MISFILED: 2020/ holds three 20210301 granules. The index
    is keyed on the date inside the filename, never on the directory name, and
    neighbouring year directories are scanned for the target month.
"""

import calendar
import datetime as dt
import os
import re

import numpy as np

GRANULES_PER_DAY = 48
GRANULE_MINUTES = 30

# Canonical name, e.g.
# 3B-HHR.MS.MRG.3IMERG.20220101-S000000-E002959.0000.V07B.HDF5
# Trailing $ rejects the *.HDF5.1 re-download duplicates in 2014/.
GRANULE_RE = re.compile(
    r"^3B-HHR\.MS\.MRG\.3IMERG\."
    r"(?P<date>\d{8})-S(?P<start>\d{6})-E(?P<end>\d{6})\."
    r"(?P<offset>\d{4})\.(?P<version>V\d{2}[A-Z]?)\.HDF5$"
)

PRECIP_PATH = "/Grid/precipitation"
LAT_PATH = "/Grid/lat"
LON_PATH = "/Grid/lon"

# _FillValue is -9999.9; CodeMissingValue matches. Precipitation is physically
# non-negative, so anything this far below zero is missing data, however the
# sentinel was rounded in storage.
MISSING_BELOW = -9000.0

MODES = ("halfhour", "hourly")
# "pm180" rather than "-180-180": a leading dash makes argparse treat the value
# as another flag, so the literal grid name is unusable on the command line.
LON_CONVENTIONS = ("0-360", "pm180")


class MissingGranuleError(Exception):
    """A month is not fully covered by granules on disk.

    Raised before any work is done. A short month would desynchronise
    time_index across the monthly CSVs and break the contiguity assertion in
    verify_outputs.py, so a gap is fatal rather than silently tolerated.
    """


def parse_granule_filename(name):
    """Parse a granule filename into (start_datetime, offset_minutes).

    Returns None for anything that is not a canonical granule name, so callers
    can filter a directory listing without special-casing the stray shell
    scripts and *.HDF5.1 duplicates in the archive.
    """
    m = GRANULE_RE.match(os.path.basename(name))
    if m is None:
        return None
    date = dt.datetime.strptime(m.group("date"), "%Y%m%d")
    hh, mm, ss = (int(m.group("start")[i:i + 2]) for i in (0, 2, 4))
    start = date + dt.timedelta(hours=hh, minutes=mm, seconds=ss)
    offset = int(m.group("offset"))
    # The offset field is minutes-since-midnight and must agree with S-time;
    # a mismatch means a malformed name, not a granule we can place in time.
    if offset != hh * 60 + mm:
        return None
    return start, offset


def expected_starts(year, month):
    """Every granule start time a complete month must contain (48 x days)."""
    ndays = calendar.monthrange(year, month)[1]
    base = dt.datetime(year, month, 1)
    return [base + dt.timedelta(minutes=GRANULE_MINUTES * k)
            for k in range(ndays * GRANULES_PER_DAY)]


def index_granules(names):
    """Map start_datetime -> name for every canonical granule in `names`.

    Pure function over an iterable of paths/filenames; the unit tests drive it
    with plain strings. Later duplicates of the same start time are ignored.
    """
    out = {}
    for name in names:
        parsed = parse_granule_filename(name)
        if parsed is None:
            continue
        start, _ = parsed
        out.setdefault(start, name)
    return out


def scan_archive(root, year, month):
    """Index the granules for one month from a flat <root>/<YYYY>/ archive.

    Neighbouring year directories are scanned too: granules are keyed on the
    date in the filename, and the archive is known to misfile some (2020/ holds
    three 20210301 granules).
    """
    names = []
    for y in (year - 1, year, year + 1):
        d = os.path.join(root, str(y))
        if not os.path.isdir(d):
            continue
        names.extend(os.path.join(d, fn) for fn in os.listdir(d))
    index = index_granules(names)
    lo = dt.datetime(year, month, 1)
    hi = lo + dt.timedelta(days=calendar.monthrange(year, month)[1])
    return {k: v for k, v in index.items() if lo <= k < hi}


def build_timesteps(index, year, month, mode):
    """Group an index of granules into timesteps. Raises on any gap.

    halfhour -> one timestep per granule, 48/day.
    hourly   -> one timestep per hour, 24/day, from the :00 and :30 granules.

    Returns a list of (time_label, [paths]) in ascending time order. The month
    must be complete: a missing granule (or a missing member of an hourly pair)
    raises MissingGranuleError listing what is absent.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    wanted = expected_starts(year, month)
    missing = [t for t in wanted if t not in index]
    if missing:
        raise MissingGranuleError(
            f"{year}-{month:02d}: {len(missing)} of {len(wanted)} granules "
            f"missing on disk; first {min(5, len(missing))}: "
            + ", ".join(t.strftime("%Y%m%d-S%H%M%S") for t in missing[:5])
        )

    if mode == "halfhour":
        return [(t.isoformat(), [index[t]]) for t in wanted]

    # hourly: pair each :00 with the :30 that follows it.
    steps = []
    for t in wanted:
        if t.minute != 0:
            continue
        mate = t + dt.timedelta(minutes=GRANULE_MINUTES)
        if mate not in index:
            # Unreachable while the completeness check above passes, but the
            # pairing is the thing that must not silently degrade to a
            # single-granule "hour", so it is checked on its own terms.
            raise MissingGranuleError(
                f"{year}-{month:02d}: hourly pair incomplete at "
                f"{t.isoformat()} (missing the :30 granule)"
            )
        steps.append((t.isoformat(), [index[t], index[mate]]))
    return steps


def lon_roll(lons_hdf5, convention):
    """Return (lons, roll) that put the grid in `convention`.

    `roll` is the shift to apply to the longitude axis of the data with
    np.roll; the HDF5 grid is -180..180 ascending, so mapping to 0..360 is a
    pure roll by -n/2 plus adding 360 to the negative half.
    """
    if convention not in LON_CONVENTIONS:
        raise ValueError(f"lon convention must be one of {LON_CONVENTIONS}")
    lons = np.asarray(lons_hdf5, dtype=float)
    if convention == "pm180":
        return lons, 0
    n = lons.size
    roll = -(n // 2)
    out = np.roll(lons, roll)
    out = np.where(out < 0, out + 360.0, out)
    if not np.all(np.diff(out) > 0):
        raise ValueError("longitude did not become ascending under 0-360; "
                         "grid is not the expected -180..180 layout")
    return out, roll


def read_grid(path, lon_convention="0-360"):
    """Read (lats, lons, roll) from a granule, in the requested convention."""
    import h5py
    with h5py.File(path, "r") as f:
        lats = f[LAT_PATH][:].astype(float)
        lons_raw = f[LON_PATH][:].astype(float)
    if not np.all(np.diff(lats) > 0):
        raise ValueError(f"{path}: latitude is not ascending")
    lons, roll = lon_roll(lons_raw, lon_convention)
    return lats, lons, roll


def _read_one(path, lat_slice, roll):
    """One granule as a (lat, lon) float64 array in mm/hr, fills -> NaN."""
    import h5py
    with h5py.File(path, "r") as f:
        dset = f[PRECIP_PATH]
        # Stored (time, lon, lat) with chunks spanning all 1800 lats, so a
        # band slice on disk saves no I/O. Read the block contiguously.
        raw = dset[0]                       # (lon, lat)
    arr = np.asarray(raw, dtype=float).T    # -> (lat, lon)
    if roll:
        arr = np.roll(arr, roll, axis=1)
    arr = arr[lat_slice, :]
    # -9999.9 must not survive: downstream stats test ~np.isnan, and a raw
    # sentinel would read as valid weak precipitation.
    arr[arr <= MISSING_BELOW] = np.nan
    return arr


def load_field(paths, lat_slice, roll):
    """Load one timestep as a (lat, lon) mm/hr field.

    One path -> that granule. Two paths -> their arithmetic mean, which is the
    hourly mean *rate* (the data are already rates, so no re-scaling). NaN
    propagates: if either half-hour is missing data at a pixel, the hour is
    missing there too, rather than silently reporting the one valid half.
    """
    if not paths:
        raise ValueError("no granule paths given")
    acc = _read_one(paths[0], lat_slice, roll)
    for p in paths[1:]:
        acc += _read_one(p, lat_slice, roll)
    if len(paths) > 1:
        acc /= len(paths)
    return acc
