# -*- coding: utf-8 -*-
"""
Aftershock/ΔCFF statistical correlation testing.

Replicates coulomb.m's three related MATLAB callbacks in one coherent
Python module, ground-truthed against that source directly:

  menu3_ratio_callback (coulomb.m ~27720)
      Observed aftershocks only: interpolate ΔCFF at each real
      aftershock location, sweep a threshold 0..xMax, plot fraction/
      count with CFF>=+thr ("GE", promoted) and CFF<=-thr ("LE",
      inhibited) vs threshold. No randomization -- this is
      threshold_curve() applied to the real catalog.

  menu_aftershocks_in_threshold_region (coulomb.m ~26746)
      Same interpolation + GE/LE classification, but at a single
      threshold rather than a sweep. Subsumed here: call
      threshold_curve() with a length-1 thr_vec, or read one element
      out of the full sweep -- no separate function needed.

  menu_random_eq_null_test_callback (coulomb.m ~27934)
      The actual NULL TEST: Monte Carlo over `Mrun` runs of `Npts`
      randomly-placed points within the 3D domain, same GE/LE
      threshold sweep on each run, mean + 5-95th percentile bands
      across runs. Two depth-sampling modes -- "uniform" (z uniform
      across the domain's depth range) and "match_eq_depth" (z
      bootstrap-resampled from the observed aftershock depth
      distribution, since ΔCFF varies strongly with depth and real
      aftershocks cluster in specific depth bands -- sampling depth
      uniformly would bias the null distribution). Falls back to
      "uniform" if no depth data is available, same as coulomb.m.

`observed_vs_null()` ties the observed curve and the null bands
together into one result, which is the actual publication-standard
comparison ("do aftershocks preferentially occur in positive-ΔCFF
lobes vs random placement") -- coulomb.m computes these as two
separately-invoked, separately-plotted callbacks; nothing stops a user
from running one without the other, but there's no reason not to offer
the combined call as the primary entry point here.

Deliberate unit divergence from coulomb.m: thresholds here are in MPa,
not bar (1 bar = 0.1 MPa). This project's ΔCFF has been in MPa
throughout (core.okada_engine.compute_coulomb_grid's own docstring) --
reintroducing bar units for this one feature would be inconsistent
with everything else in the plugin. UI layer is responsible for
labeling threshold inputs as MPa.

Percentile method: numpy's default (linear interpolation), not
MATLAB's `prctile(...,'Method','approximate')`. No MATLAB reference
output exists to diff this against digit-for-digit (unlike the Okada/
DC3D core, which was cross-validated against Coulomb 3.4.2's actual
.dat output) -- the two methods agree to within a fraction of a percent
for this sample-size range, which is what matters for a 5-95% band
that's illustrating Monte Carlo spread, not a precise inference.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Sequence, Callable

import numpy as np

from .cff_volume import CFFVolume, interpolate_cff_at_points


class MCTestCancelled(Exception):
    """Raised by monte_carlo_null()/observed_vs_null() when the caller's
    cancel_check() callback returns True mid-run -- mirrors coulomb.m's
    own uiprogressdlg 'Cancelable' + d.CancelRequested pattern, just
    surfaced as an exception instead of a GUI progress-dialog field."""
    pass


# ─── Threshold vector ────────────────────────────────────────────────────

def build_threshold_vector(x_max: float, n_points: int = 80) -> np.ndarray:
    """0..x_max inclusive, n_points samples -- matches coulomb.m's
    `nThr = 80; thrVec = linspace(0, xMax, nThr)` default in the null
    test. (menu3_ratio_callback instead uses a fixed step `dThr`; use
    `np.arange(0, x_max + step/2, step)` directly if a step-based sweep
    is wanted instead of a fixed point count -- both are just a thr_vec
    to every function below, no special-casing needed.)"""
    if not np.isfinite(x_max) or x_max <= 0:
        raise ValueError(f"x_max must be a finite positive number, got {x_max!r}")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points!r}")
    return np.linspace(0.0, x_max, n_points)


# ─── Core GE/LE classification (shared by observed + every MC run) ──────

@dataclass
class ThresholdCurve:
    thr_vec: np.ndarray
    frac_ge: np.ndarray    # fraction with CFF >= +thr, per threshold
    frac_le: np.ndarray    # fraction with CFF <= -thr, per threshold
    cnt_ge: np.ndarray
    cnt_le: np.ndarray
    n_valid: int            # number of finite CFF values this curve was built from


def threshold_curve(cff_values: np.ndarray, thr_vec: np.ndarray) -> ThresholdCurve:
    """
    cff_values: 1D array of already-interpolated ΔCFF values (MPa) --
    NaN/inf entries are dropped here (matches coulomb.m's `ok =
    isfinite(cff); cff = cff(ok)` before any threshold classification).
    Returns per-threshold GE ("promoted", CFF>=+thr) and LE
    ("inhibited", CFF<=-thr) fractions and counts, vectorized over
    thr_vec (no explicit per-threshold Python loop, unlike coulomb.m's
    `for k = 1:nThr` -- broadcasting does the same comparison faster).
    """
    thr_vec = np.asarray(thr_vec, dtype=float)
    vals = np.asarray(cff_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = vals.size

    if n == 0:
        nan_arr = np.full(thr_vec.shape, np.nan)
        return ThresholdCurve(thr_vec=thr_vec, frac_ge=nan_arr, frac_le=nan_arr,
                              cnt_ge=np.zeros(thr_vec.shape), cnt_le=np.zeros(thr_vec.shape),
                              n_valid=0)

    # vals[:, None] vs thr_vec[None, :] -> (n, n_thr) boolean, summed over axis 0
    ge_mask = vals[:, None] >= (+thr_vec)[None, :]
    le_mask = vals[:, None] <= (-thr_vec)[None, :]
    cnt_ge = ge_mask.sum(axis=0).astype(float)
    cnt_le = le_mask.sum(axis=0).astype(float)

    return ThresholdCurve(thr_vec=thr_vec, frac_ge=cnt_ge / n, frac_le=cnt_le / n,
                          cnt_ge=cnt_ge, cnt_le=cnt_le, n_valid=n)


# ─── Observed aftershocks ────────────────────────────────────────────────

def interpolate_cff_at_eq(volume: CFFVolume, eq_array: Sequence[dict]) -> np.ndarray:
    """
    eq_array: list of dicts with "lon"/"lat"/"depth" keys (e.g.
    core.eq_catalog_import.events_to_eq_array() output). Rows with a
    None coordinate are dropped before interpolation (matches
    coulomb.m's `ok = isfinite(eqX) & isfinite(eqY) & isfinite(eqDepth)`
    pre-filter); points outside the volume's lon/lat/depth box come
    back as NaN from the interpolator itself and are left in the
    returned array -- callers that need "valid only" should filter with
    np.isfinite(...) themselves, or just pass straight into
    threshold_curve()/observed_vs_null(), which already do that.
    """
    pts = [(e["lon"], e["lat"], e["depth"]) for e in eq_array
           if e.get("lon") is not None and e.get("lat") is not None and e.get("depth") is not None]
    if not pts:
        return np.array([])
    return interpolate_cff_at_points(volume, pts)


def observed_threshold_curve(volume: CFFVolume, eq_array: Sequence[dict],
                              thr_vec: np.ndarray) -> ThresholdCurve:
    """Replicates menu3_ratio_callback (full sweep) and
    menu_aftershocks_in_threshold_region (single threshold is just a
    length-1 thr_vec) in one call: interpolate ΔCFF at every real
    aftershock, classify against thr_vec, no randomization."""
    cff_at_eq = interpolate_cff_at_eq(volume, eq_array)
    return threshold_curve(cff_at_eq, thr_vec)


# ─── Monte Carlo null test ──────────────────────────────────────────────

@dataclass
class MonteCarloNullResult:
    thr_vec: np.ndarray
    n_points: int
    n_runs: int
    depth_mode_requested: str
    depth_mode_used: str        # may differ from requested if "match_eq_depth" fell back
    n_valid_per_run: np.ndarray   # (n_runs,) -- points landing inside the volume, per run
    frac_ge_mean: np.ndarray
    frac_ge_p05: np.ndarray
    frac_ge_p95: np.ndarray
    frac_le_mean: np.ndarray
    frac_le_p05: np.ndarray
    frac_le_p95: np.ndarray
    cnt_ge_mean: np.ndarray
    cnt_ge_p05: np.ndarray
    cnt_ge_p95: np.ndarray
    cnt_le_mean: np.ndarray
    cnt_le_p05: np.ndarray
    cnt_le_p95: np.ndarray


def monte_carlo_null(volume: CFFVolume, thr_vec: np.ndarray,
                      n_points: int = 2000, n_runs: int = 100,
                      depth_mode: str = "uniform",
                      eq_depths: Optional[Sequence[float]] = None,
                      rng: Optional[np.random.Generator] = None,
                      progress_callback: Optional[Callable[[int], None]] = None,
                      cancel_check: Optional[Callable[[], bool]] = None) -> MonteCarloNullResult:
    """
    Monte Carlo null distribution: `n_runs` runs of `n_points` randomly
    -placed points within `volume`'s lon/lat/depth box, ΔCFF interpolated
    at each, GE/LE threshold-classified exactly like the observed curve.
    Matches coulomb.m's `menu_random_eq_null_test_callback` algorithm.

    depth_mode:
      "uniform"         -- z ~ Uniform(min(volume.depths_km), max(...))
      "match_eq_depth"  -- z bootstrap-resampled (with replacement) from
                            `eq_depths`, restricted to the volume's depth
                            range first (coulomb.m: `eqDepth(isfinite(..)
                            & eqDepth>=zmin & eqDepth<=zmax)`). Falls back
                            to "uniform" with a note in
                            `.depth_mode_used` if `eq_depths` is None/
                            empty or nothing survives the range filter --
                            same silent fallback coulomb.m performs
                            (`depthMode = 1`), not an error.

    x/y are always sampled uniformly across volume.lons/volume.lats'
    min/max (matches coulomb.m: x,y always uniform regardless of
    depth_mode -- only z sampling differs between the two modes).
    """
    if depth_mode not in ("uniform", "match_eq_depth"):
        raise ValueError(f"depth_mode must be 'uniform' or 'match_eq_depth', got {depth_mode!r}")
    if n_points < 10:
        raise ValueError("n_points must be >= 10")
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")

    rng = rng or np.random.default_rng()
    thr_vec = np.asarray(thr_vec, dtype=float)
    n_thr = thr_vec.size

    lon_min, lon_max = float(volume.lons.min()), float(volume.lons.max())
    lat_min, lat_max = float(volume.lats.min()), float(volume.lats.max())
    z_min, z_max = float(volume.depths_km.min()), float(volume.depths_km.max())

    depth_mode_used = depth_mode
    eq_depth_pool = None
    if depth_mode == "match_eq_depth":
        if eq_depths is not None:
            arr = np.asarray([d for d in eq_depths if d is not None], dtype=float)
            arr = arr[np.isfinite(arr) & (arr >= z_min) & (arr <= z_max)]
            if arr.size > 0:
                eq_depth_pool = arr
        if eq_depth_pool is None:
            depth_mode_used = "uniform"  # silent fallback, same as coulomb.m

    frac_ge = np.full((n_runs, n_thr), np.nan)
    frac_le = np.full((n_runs, n_thr), np.nan)
    cnt_ge = np.full((n_runs, n_thr), np.nan)
    cnt_le = np.full((n_runs, n_thr), np.nan)
    n_valid = np.full(n_runs, np.nan)

    for r in range(n_runs):
        if cancel_check is not None and cancel_check():
            raise MCTestCancelled(f"cancelled at run {r}/{n_runs}")

        rx = rng.uniform(lon_min, lon_max, n_points)
        ry = rng.uniform(lat_min, lat_max, n_points)
        if depth_mode_used == "uniform":
            rz = rng.uniform(z_min, z_max, n_points)
        else:
            rz = rng.choice(eq_depth_pool, size=n_points, replace=True)

        cff = interpolate_cff_at_points(volume, list(zip(rx, ry, rz)))
        curve = threshold_curve(cff, thr_vec)

        frac_ge[r], frac_le[r] = curve.frac_ge, curve.frac_le
        cnt_ge[r], cnt_le[r] = curve.cnt_ge, curve.cnt_le
        n_valid[r] = curve.n_valid

        if progress_callback:
            progress_callback(int(100 * (r + 1) / n_runs))

    def _mean(a):
        return np.nanmean(a, axis=0)

    def _p(a, q):
        return np.nanpercentile(a, q, axis=0)

    return MonteCarloNullResult(
        thr_vec=thr_vec, n_points=n_points, n_runs=n_runs,
        depth_mode_requested=depth_mode, depth_mode_used=depth_mode_used,
        n_valid_per_run=n_valid,
        frac_ge_mean=_mean(frac_ge), frac_ge_p05=_p(frac_ge, 5), frac_ge_p95=_p(frac_ge, 95),
        frac_le_mean=_mean(frac_le), frac_le_p05=_p(frac_le, 5), frac_le_p95=_p(frac_le, 95),
        cnt_ge_mean=_mean(cnt_ge), cnt_ge_p05=_p(cnt_ge, 5), cnt_ge_p95=_p(cnt_ge, 95),
        cnt_le_mean=_mean(cnt_le), cnt_le_p05=_p(cnt_le, 5), cnt_le_p95=_p(cnt_le, 95),
    )


# ─── Combined observed-vs-null (primary entry point) ────────────────────

@dataclass
class AftershockMCTestResult:
    observed: ThresholdCurve
    null: MonteCarloNullResult


def observed_vs_null(volume: CFFVolume, eq_array: Sequence[dict],
                      x_max: float, n_thr: int = 80,
                      n_points: int = 2000, n_runs: int = 100,
                      depth_mode: str = "match_eq_depth",
                      rng: Optional[np.random.Generator] = None,
                      progress_callback: Optional[Callable[[int], None]] = None,
                      cancel_check: Optional[Callable[[], bool]] = None) -> AftershockMCTestResult:
    """
    Primary entry point: observed aftershock GE/LE curve +  Monte Carlo
    null bands over the same threshold vector, ready to overlay on one
    plot (coulomb.m draws these from two separately-invoked callbacks;
    nothing here requires two calls). eq_array is used BOTH as the
    observed catalog and (when depth_mode="match_eq_depth", the default
    here -- see note below) as the source depth distribution for the
    null test's depth resampling.

    Default depth_mode="match_eq_depth" (not "uniform", coulomb.m's own
    default of depthMode=1... actually coulomb.m's inputdlg default is
    '2' i.e. match_eq_depth -- matching that): ΔCFF is strongly depth-
    dependent and real aftershocks cluster in specific depth bands, so
    a null test that samples depth uniformly would be comparing against
    an unrealistic null population. Falls back to uniform automatically
    (see monte_carlo_null docstring) if eq_array has no usable depths.
    """
    thr_vec = build_threshold_vector(x_max, n_thr)
    observed = observed_threshold_curve(volume, eq_array, thr_vec)

    eq_depths = [e.get("depth") for e in eq_array if e.get("depth") is not None]
    null = monte_carlo_null(volume, thr_vec, n_points=n_points, n_runs=n_runs,
                            depth_mode=depth_mode, eq_depths=eq_depths, rng=rng,
                            progress_callback=progress_callback, cancel_check=cancel_check)

    return AftershockMCTestResult(observed=observed, null=null)
