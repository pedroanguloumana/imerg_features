"""Unit tests for the IMERG HDF5 load path.

Filename parsing, granule indexing, hourly pairing and the gap failures are
pure functions of strings and dicts, so none of this touches /data05.
"""

import datetime as dt

import numpy as np
import pytest

import imerg_hdf5_io as io


def gname(date, hhmmss, offset, version="V07B"):
    return (f"3B-HHR.MS.MRG.3IMERG.{date}-S{hhmmss}-E002959."
            f"{offset:04d}.{version}.HDF5")


# --- filename parsing -------------------------------------------------------
def test_parse_first_granule_of_day():
    start, offset = io.parse_granule_filename(
        "3B-HHR.MS.MRG.3IMERG.20220101-S000000-E002959.0000.V07B.HDF5")
    assert start == dt.datetime(2022, 1, 1, 0, 0)
    assert offset == 0


def test_parse_last_granule_of_day():
    start, offset = io.parse_granule_filename(
        "3B-HHR.MS.MRG.3IMERG.20220101-S233000-E235959.1410.V07B.HDF5")
    assert start == dt.datetime(2022, 1, 1, 23, 30)
    assert offset == 1410


def test_parse_accepts_full_path():
    start, _ = io.parse_granule_filename(
        "/data05/IMERG_GPM/2022/"
        "3B-HHR.MS.MRG.3IMERG.20220215-S120000-E122959.0720.V07B.HDF5")
    assert start == dt.datetime(2022, 2, 15, 12, 0)


@pytest.mark.parametrize("name", [
    # re-download duplicates really present in 2014/
    "3B-HHR.MS.MRG.3IMERG.20140523-S000000-E002959.0000.V07B.HDF5.1",
    # stray shell scripts really present in 2015/, 2016/, 2020/
    "imerg_2015.sh",
    "",
    "3B-HHR.MS.MRG.3IMERG.20220101-S000000-E002959.0000.V07B.nc",
    "random.HDF5",
])
def test_parse_rejects_non_granules(name):
    assert io.parse_granule_filename(name) is None


def test_parse_rejects_offset_disagreeing_with_start_time():
    # 12:00 is minute 720; a name claiming 0000 is malformed and cannot be
    # placed in time, so it must not be indexed.
    assert io.parse_granule_filename(gname("20220101", "120000", 0)) is None


# --- indexing ---------------------------------------------------------------
def test_index_skips_junk_and_keys_on_filename_date():
    names = [
        gname("20220101", "000000", 0),
        gname("20220101", "003000", 30),
        "imerg_2022.sh",
        gname("20220101", "010000", 60) + ".1",
    ]
    idx = io.index_granules(names)
    assert set(idx) == {dt.datetime(2022, 1, 1, 0, 0),
                        dt.datetime(2022, 1, 1, 0, 30)}


def test_index_is_directory_agnostic():
    """Mirrors the real misfiling: 2020/ holds three 20210301 granules."""
    idx = io.index_granules(["/data05/IMERG_GPM/2020/" + gname("20210301", "000000", 0)])
    assert dt.datetime(2021, 3, 1, 0, 0) in idx


def test_expected_starts_counts():
    assert len(io.expected_starts(2022, 2)) == 28 * 48
    assert len(io.expected_starts(2020, 2)) == 29 * 48      # leap
    assert io.expected_starts(2022, 2)[-1] == dt.datetime(2022, 2, 28, 23, 30)


# --- grouping ---------------------------------------------------------------
def full_month(year, month):
    return {t: gname(t.strftime("%Y%m%d"), t.strftime("%H%M%S"),
                     t.hour * 60 + t.minute)
            for t in io.expected_starts(year, month)}


def test_halfhour_one_timestep_per_granule():
    steps = io.build_timesteps(full_month(2022, 2), 2022, 2, "halfhour")
    assert len(steps) == 28 * 48 == 1344
    assert all(len(paths) == 1 for _, paths in steps)
    assert steps[0][0] == "2022-02-01T00:00:00"
    assert steps[1][0] == "2022-02-01T00:30:00"


def test_hourly_pairs_00_and_30():
    steps = io.build_timesteps(full_month(2022, 2), 2022, 2, "hourly")
    assert len(steps) == 28 * 24 == 672
    assert all(len(paths) == 2 for _, paths in steps)
    # first hour pairs S000000 with S003000
    _, paths = steps[0]
    assert "S000000" in paths[0] and "S003000" in paths[1]
    # labels are hour starts, ascending, contiguous
    labels = [dt.datetime.fromisoformat(t) for t, _ in steps]
    assert labels[0] == dt.datetime(2022, 2, 1, 0, 0)
    assert all((b - a) == dt.timedelta(hours=1) for a, b in zip(labels, labels[1:]))


def test_leap_february_is_longer():
    assert len(io.build_timesteps(full_month(2020, 2), 2020, 2, "halfhour")) == 29 * 48


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        io.build_timesteps(full_month(2022, 2), 2022, 2, "daily")


# --- gap failures -----------------------------------------------------------
def test_missing_granule_raises_in_halfhour():
    idx = full_month(2022, 2)
    del idx[dt.datetime(2022, 2, 14, 6, 30)]
    with pytest.raises(io.MissingGranuleError) as e:
        io.build_timesteps(idx, 2022, 2, "halfhour")
    assert "20220214-S063000" in str(e.value)


def test_missing_half_hour_breaks_the_hourly_pair():
    idx = full_month(2022, 2)
    del idx[dt.datetime(2022, 2, 3, 9, 30)]          # the :30 of an hour
    with pytest.raises(io.MissingGranuleError):
        io.build_timesteps(idx, 2022, 2, "hourly")


def test_missing_on_the_hour_also_fails_hourly():
    idx = full_month(2022, 2)
    del idx[dt.datetime(2022, 2, 3, 9, 0)]
    with pytest.raises(io.MissingGranuleError):
        io.build_timesteps(idx, 2022, 2, "hourly")


def test_empty_month_raises_rather_than_returning_nothing():
    with pytest.raises(io.MissingGranuleError):
        io.build_timesteps({}, 2022, 3, "halfhour")   # 2022-03 really is absent


def test_error_reports_how_many_are_missing():
    idx = full_month(2022, 2)
    for t in list(idx)[:10]:
        del idx[t]
    with pytest.raises(io.MissingGranuleError) as e:
        io.build_timesteps(idx, 2022, 2, "halfhour")
    assert "10 of 1344" in str(e.value)


def test_wrong_month_granules_do_not_fill_a_gap():
    """A January granule must not satisfy a February slot."""
    idx = full_month(2022, 2)
    del idx[dt.datetime(2022, 2, 1, 0, 0)]
    idx[dt.datetime(2022, 1, 1, 0, 0)] = gname("20220101", "000000", 0)
    with pytest.raises(io.MissingGranuleError):
        io.build_timesteps(idx, 2022, 2, "halfhour")


# --- longitude convention ---------------------------------------------------
def imerg_lons():
    return -179.95 + 0.1 * np.arange(3600)


def test_0_360_matches_the_old_netcdf_grid():
    lons, roll = io.lon_roll(imerg_lons(), "0-360")
    assert roll == -1800
    assert np.all(np.diff(lons) > 0)
    assert lons[0] == pytest.approx(0.05)
    assert lons[-1] == pytest.approx(359.95)


def test_pm180_is_a_no_op():
    lons, roll = io.lon_roll(imerg_lons(), "pm180")
    assert roll == 0
    assert lons[0] == pytest.approx(-179.95)


def test_convention_names_are_argparse_safe():
    """A leading dash makes argparse read the value as another flag."""
    assert not any(c.startswith("-") for c in io.LON_CONVENTIONS)


def test_roll_moves_data_consistently_with_the_axis():
    lons = imerg_lons()
    data = lons.copy()                      # value == its own longitude
    out_lons, roll = io.lon_roll(lons, "0-360")
    out_data = np.roll(data, roll)
    # after rolling, column j holds the value that belongs at out_lons[j]
    assert np.allclose(np.where(out_data < 0, out_data + 360.0, out_data), out_lons)


def test_unknown_convention_rejected():
    with pytest.raises(ValueError):
        io.lon_roll(imerg_lons(), "0..360")


# --- field assembly ---------------------------------------------------------
def test_hourly_is_the_mean_of_two_rates(monkeypatch):
    """Rates, so the hour is a plain mean -- not a sum, and no /6 anywhere."""
    fields = {"a": np.full((4, 4), 2.0), "b": np.full((4, 4), 4.0)}
    monkeypatch.setattr(io, "_read_one", lambda p, s, r: fields[p].copy())
    assert np.allclose(io.load_field(["a", "b"], slice(None), 0), 3.0)
    assert np.allclose(io.load_field(["a"], slice(None), 0), 2.0)


def test_nan_propagates_through_the_hourly_mean(monkeypatch):
    a = np.array([[1.0, np.nan]])
    b = np.array([[3.0, 5.0]])
    monkeypatch.setattr(io, "_read_one", lambda p, s, r: {"a": a, "b": b}[p].copy())
    out = io.load_field(["a", "b"], slice(None), 0)
    assert out[0, 0] == 2.0
    assert np.isnan(out[0, 1])       # not silently 5.0


def test_load_field_requires_paths():
    with pytest.raises(ValueError):
        io.load_field([], slice(None), 0)
