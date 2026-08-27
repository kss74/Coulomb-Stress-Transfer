# -*- coding: utf-8 -*-
"""
3D ΔCFF volume: a stack of core.okada_engine.compute_coulomb_grid_depth()
depth slices, wrapped in a trilinear interpolator, matching the role
Coulomb 3.4.2's own cached "dcff_3D_slices.mat" plays for
menu_random_eq_null_test_callback (see coulomb.m) -- except computed
fresh from this plugin's own Okada engine rather than loaded from a
MATLAB cache file, and cached to a local .npz instead.

Consumers: core.aftershock_mc_test (not yet built) interpolates each
observed aftershock's (lon,lat,depth) against this volume; any future
rate-and-state or period-comparison module would do the same.

Kept deliberately separate from okada_engine.py itself -- this module
only orchestrates repeated calls to the already-validated single-depth
grid function and packages the result; it adds no new physics, so it
doesn't belong inside the validated engine module (project convention:
new functionality goes in new files, existing validated modules are
only touched when necessary).
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .okada_engine import (
    FaultParameters, ElasticParameters, GridParameters,
    compute_coulomb_grid_depth,
)
from .optimal_plane import RegionalStress, compute_optimal_cff_grid_depth


# ─── Depth slice selection ──────────────────────────────────────────────

def auto_depth_slices(eq_events: Sequence[dict], min_slices: int = 5,
                       max_slices: int = 15, pad_km: float = 2.0,
                       default_range_km=(0.0, 5.0, 10.0, 15.0, 20.0)) -> List[float]:
    """
    Pick depth slices spanning the observed catalog's own depth range,
    padded by `pad_km` on each side (so aftershocks near the shallowest/
    deepest observed depth don't sit right at the interpolation volume's
    edge, where RegularGridInterpolator's bounds_error=False -> NaN
    behavior would otherwise clip them). Slice count scales with how
    many distinct depths the catalog actually reports, clamped to
    [min_slices, max_slices] -- each slice is a full-grid Okada/DC3D
    evaluation, so this is a direct cost knob, not just a display choice.

    `eq_events` : list of dicts with a "depth" key (e.g.
    core.eq_catalog_import.events_to_eq_array() output) -- rows with
    depth=None (shouldn't happen; depth is required at import time, but
    defensive here) are ignored. Falls back to `default_range_km` if the
    catalog is empty or has no valid depths, so this function always
    returns something usable.
    """
    depths = [e.get("depth") for e in eq_events if e.get("depth") is not None]
    if not depths:
        return list(default_range_km)

    dmin = max(0.0, min(depths) - pad_km)
    dmax = max(depths) + pad_km
    if dmax <= dmin:
        dmax = dmin + pad_km  # degenerate single-depth catalog

    n_distinct = len(set(round(d, 1) for d in depths))
    n_slices = max(min_slices, min(max_slices, n_distinct))
    return list(np.linspace(dmin, dmax, n_slices))


# ─── Volume container ───────────────────────────────────────────────────

@dataclass
class CFFVolume:
    lons: np.ndarray          # 1D, shape (n_lon,)
    lats: np.ndarray          # 1D, shape (n_lat,)
    depths_km: np.ndarray     # 1D, shape (n_depth,), ascending
    cff_mpa: np.ndarray       # 3D, shape (n_depth, n_lat, n_lon)
    used_dc3d: List[bool] = field(default_factory=list)   # per depth slice
    near_field_mask: Optional[np.ndarray] = None           # 3D bool, same shape as cff_mpa
    mode: str = "fixed"        # "fixed" (specified receiver) or "optimal" (optimally-oriented plane)


@dataclass
class CFFFieldStats:
    """
    Summary statistics of a ΔCFF field (a whole CFFVolume.cff_mpa array,
    or any subset/slice of one -- this function has no opinion on shape,
    it just flattens and reduces). Added 2026-08-22c per the external-
    review synthesis's top-priority item (session 3, "ΔCFF field
    statistics before running" -- see PROJECT_HANDOVER_ADDENDUM_2026-08-
    21c_external_review_synthesis.md): surfacing these before/alongside
    a rate-and-state run answers "is this a real signal or a units/
    parameter mismatch?" at a glance, which is exactly the question
    session 3's own worked example (a 1,841x background-rate spike)
    needed answered.

    All fields are plain floats/ints, not numpy scalars, so this
    round-trips cleanly through str.format / report text without numpy's
    own repr leaking in.
    """
    n_points: int          # total array size (all cells, finite or not)
    n_finite: int           # cells actually used for the stats below
    min: float
    max: float
    mean: float
    median: float
    std: float
    p5: float
    p95: float
    frac_positive: float    # fraction of FINITE cells with cff > 0
    frac_negative: float    # fraction of FINITE cells with cff < 0
    frac_zero: float        # fraction of FINITE cells with cff == 0 exactly (rare; keeps the three fractions summing to 1)
    n_near_field_excluded: int = 0   # cells dropped because near_field_mask>0 (see apply_near_field_mask); 0 if no mask was available/applicable


def apply_near_field_mask(cff_mpa, near_field_mask, exclude_near_field: bool = True):
    """
    Returns a COPY of `cff_mpa` with every cell flagged by
    `near_field_mask` (integer-coded, >0 meaning "near-field": see
    core.okada_engine.near_field_grid_mask's own 0/1/2 docstring) set to
    NaN. `cff_mpa` itself is never mutated.

    Root-cause fix (2026-08-22 smoke-test): near-field cells are known-
    unreliable Okada/DC3D singularities -- their |ΔCFF| can run one to
    two orders of magnitude beyond the reliable field (a handful of
    cells at a few MPa against a field whose std is a few tenths of a
    MPa is typical). Before this function existed, nothing downstream
    excluded them: cff_field_stats() and plot_cff_field_histogram()'s
    histogram both summarized/binned the raw field with these outliers
    included (axis dominated by 1-2 extreme cells, the real distribution
    crushed into an unreadable spike), and forecast_from_cff_volume()/
    calibrate_rate_state() fed the raw field straight into d94's
    exp(-dcff/asig) term, letting a few singular cells' rate swamp
    total_rate()/total_cumulative() (the "steep rise then flat,
    grossly-inflated peak" symptom).

    NaN is the exclusion convention this project already uses for
    out-of-interpolation-range grid points (CFFVolume's own
    RegularGridInterpolator fill_value=nan) -- masking here piggybacks
    on the nan-aware reducers/nansum already in place at every
    downstream consumer, rather than adding a second exclusion
    mechanism those consumers would each need to learn about
    separately.

    exclude_near_field=False, or near_field_mask=None (e.g. mode=
    "optimal" volumes -- see build_cff_volume's own "Known gap"
    docstring note), or a shape mismatch: returns `cff_mpa` as a plain
    float array, unmodified, so callers can always call this
    unconditionally without special-casing the "no mask available"
    case themselves.
    """
    arr = np.asarray(cff_mpa, dtype=float)
    if not exclude_near_field or near_field_mask is None:
        return arr
    mask = np.asarray(near_field_mask)
    if mask.shape != arr.shape:
        return arr
    out = arr.copy()
    out[mask > 0] = np.nan
    return out


def cff_field_stats(cff, exclude_near_field: bool = True) -> CFFFieldStats:
    """
    Reduce a ΔCFF field (a CFFVolume, or a plain ndarray -- e.g. one
    depth slice, or a whole volume.cff_mpa array) to summary statistics.
    Accepts a CFFVolume directly (reads .cff_mpa) or any array-like, so
    callers can pass either the whole 3D volume or an already-sliced 2D
    array without converting first.

    exclude_near_field (default True): when `cff` is a CFFVolume with a
    populated near_field_mask, cells flagged near-field (mask>0) are
    excluded from every statistic below via apply_near_field_mask() --
    see that function's docstring for why. Has no effect when `cff` is
    a plain array (no mask information travels with a bare ndarray) or
    when the volume's mask is None/all-zero (e.g. mode="optimal").

    NaN-aware throughout (RegularGridInterpolator's fill_value=nan
    convention means points outside an interpolated volume are commonly
    NaN -- see CFFVolume's own module docstring) via nan* reducers;
    n_finite/frac_* are computed against the finite subset only, so a
    volume with some out-of-range NaN corners doesn't silently bias the
    fractions toward whichever sign happens to dominate the NaN
    locations. Near-field-excluded cells are folded into this same
    finite/non-finite split (they become NaN before any reduction runs),
    so n_finite already reflects them; n_near_field_excluded reports how
    many of the originally-finite cells that was, for transparency.

    Raises ValueError if there are zero finite points (nothing to
    summarize) rather than returning silently-meaningless nan-filled
    stats -- callers should treat that as "check the volume/receiver
    setup", not a valid (if boring) result.
    """
    if isinstance(cff, CFFVolume):
        arr_raw = np.asarray(cff.cff_mpa, dtype=float)
        arr = apply_near_field_mask(cff.cff_mpa, cff.near_field_mask, exclude_near_field)
    else:
        arr_raw = np.asarray(cff, dtype=float)
        arr = arr_raw

    finite = arr[np.isfinite(arr)]
    n_finite = int(finite.size)
    n_near_field_excluded = int(np.sum(np.isfinite(arr_raw) & ~np.isfinite(arr)))
    if n_finite == 0:
        raise ValueError("cff_field_stats: no finite ΔCFF values to summarize "
                         "(every point is NaN/inf, or excluded as near-field).")

    return CFFFieldStats(
        n_points=int(arr.size), n_finite=n_finite,
        min=float(np.min(finite)), max=float(np.max(finite)),
        mean=float(np.mean(finite)), median=float(np.median(finite)),
        std=float(np.std(finite)),
        p5=float(np.percentile(finite, 5)), p95=float(np.percentile(finite, 95)),
        frac_positive=float(np.mean(finite > 0)),
        frac_negative=float(np.mean(finite < 0)),
        frac_zero=float(np.mean(finite == 0)),
        n_near_field_excluded=n_near_field_excluded,
    )


def build_cff_volume(sources: List[FaultParameters], receiver: Optional[FaultParameters],
                      elastic: ElasticParameters, grid: GridParameters,
                      depths_km: Sequence[float],
                      progress_callback=None, mode: str = "fixed",
                      regional: Optional[RegionalStress] = None,
                      friction: Optional[float] = None) -> CFFVolume:
    """
    Compute a ΔCFF depth-slice stack, one core.okada_engine/
    core.optimal_plane grid call per depth in `depths_km` (ascending
    order not required on input; sorted here so the interpolator's
    depth axis is monotonic as scipy requires).

    mode="fixed" (default, backward-compatible): ΔCFF resolved on the
    single specified `receiver` fault -- calls
    okada_engine.compute_coulomb_grid_depth() per slice, exactly as
    before this parameter existed. `receiver` is required.

    mode="optimal": ΔCFF resolved on the OPTIMALLY-ORIENTED fault plane
    at each grid point (regional stress + friction, elementwise-max of
    the two conjugate planes) -- calls
    optimal_plane.compute_optimal_cff_grid_depth() per slice instead.
    `regional` (RegionalStress) is required; `receiver` is ignored
    entirely in this mode (not even read); `friction` defaults to
    `elastic.friction` if not given, same default
    compute_optimal_cff_grid_depth itself uses. This is generally the
    better choice for aftershock/null-test work specifically: real
    aftershocks don't all occur on one predetermined fault orientation,
    they occur on whichever plane is locally best-oriented for
    failure -- optimal-plane ΔCFF is what most published aftershock-
    forecasting studies (Toda & Stein, King/Stein/Lin) actually test
    against, not a single fixed receiver.

    Known gap in "optimal" mode: `near_field_mask` is NOT populated
    (stays all-False) -- compute_optimal_cff_grid[_depth]() doesn't
    compute one (there's no single receiver-fault trace to measure
    near-field distance against when every point uses its own locally-
    optimal plane). Not a defect introduced here, just an existing
    concept that doesn't carry over cleanly to this mode -- flagged so
    a future consumer doesn't assume near-field exclusion is happening
    when it isn't.

    `grid`'s own .depth_km is ignored in both modes -- each slice
    supplies its own via a shallow copy of `grid`, so the caller's
    original GridParameters object is never mutated.
    """
    if mode not in ("fixed", "optimal"):
        raise ValueError(f"mode must be 'fixed' or 'optimal', got {mode!r}")
    if mode == "fixed" and receiver is None:
        raise ValueError("receiver is required when mode='fixed'")
    if mode == "optimal" and regional is None:
        raise ValueError("regional is required when mode='optimal'")

    depths_sorted = sorted(float(d) for d in depths_km)
    if not depths_sorted:
        raise ValueError("depths_km must contain at least one depth")

    slices = []
    used_dc3d_flags = []
    masks = []
    lons = lats = None

    n = len(depths_sorted)
    for i, d in enumerate(depths_sorted):
        slice_grid = GridParameters(
            lon_min=grid.lon_min, lon_max=grid.lon_max,
            lat_min=grid.lat_min, lat_max=grid.lat_max,
            depth_km=d, n_lon=grid.n_lon, n_lat=grid.n_lat,
        )

        def _slice_progress(pct, _i=i, _n=n):
            if progress_callback:
                progress_callback(int(100 * (_i + pct / 100.0) / _n))

        if mode == "fixed":
            lon2d, lat2d, cff_mpa, used_dc3d, near_field = compute_coulomb_grid_depth(
                sources, receiver, elastic, slice_grid, progress_callback=_slice_progress)
        else:  # "optimal"
            result = compute_optimal_cff_grid_depth(
                sources, regional, elastic, slice_grid, friction=friction,
                progress_callback=_slice_progress)
            lon2d, lat2d, cff_mpa = result[0], result[1], result[2]
            used_dc3d = result[12]
            near_field = np.zeros(cff_mpa.shape, dtype=bool)  # see docstring: not available in this mode

        if lons is None:
            lons = lon2d[0, :].copy()
            lats = lat2d[:, 0].copy()
        slices.append(cff_mpa)
        used_dc3d_flags.append(used_dc3d)
        masks.append(near_field)

    if progress_callback:
        progress_callback(100)

    cff_stack = np.stack(slices, axis=0)          # (n_depth, n_lat, n_lon)
    mask_stack = np.stack(masks, axis=0)           # (n_depth, n_lat, n_lon)

    return CFFVolume(
        lons=lons, lats=lats, depths_km=np.array(depths_sorted),
        cff_mpa=cff_stack, used_dc3d=used_dc3d_flags, near_field_mask=mask_stack,
        mode=mode,
    )


# ─── Interpolation ───────────────────────────────────────────────────────

def build_interpolator(volume: CFFVolume):
    """
    Trilinear interpolator over (depth, lat, lon), matching Coulomb's own
    griddedInterpolant usage in menu_random_eq_null_test_callback.
    Points outside the computed volume return NaN (bounds_error=False,
    fill_value=nan) rather than extrapolating -- ΔCFF falls off in a way
    that's not safe to linearly extrapolate, and a NaN is a much safer
    failure mode for a downstream statistical test than a silently wrong
    extrapolated value.
    """
    from scipy.interpolate import RegularGridInterpolator
    return RegularGridInterpolator(
        (volume.depths_km, volume.lats, volume.lons), volume.cff_mpa,
        method="linear", bounds_error=False, fill_value=np.nan)


def interpolate_cff_at_points(volume: CFFVolume, points) -> np.ndarray:
    """
    points : iterable of (lon, lat, depth_km) tuples, e.g.
    core.eq_catalog_import.events_to_eq_array() rows via
    [(e["lon"], e["lat"], e["depth"]) for e in eq_array].
    Returns an array of interpolated ΔCFF (MPa), one per point, NaN
    where the point falls outside the computed volume.
    """
    interp = build_interpolator(volume)
    pts = np.asarray([(depth, lat, lon) for lon, lat, depth in points], dtype=float)
    if pts.size == 0:
        return np.array([])
    return interp(pts)


# ─── Caching ─────────────────────────────────────────────────────────────

def _fault_cache_repr(f: FaultParameters) -> tuple:
    """A plain-tuple projection of the FaultParameters fields that affect
    the computed stress field, used only to build a cache key -- not a
    full serialization. Any field not listed here doesn't affect
    ΔCFF/displacement output, so leaving it out of the key is correct,
    not an oversight; if okada_engine.FaultParameters ever gains a new
    field that DOES affect the physics, it must be added here too or the
    cache will silently return stale results for it."""
    return (round(f.lon, 6), round(f.lat, 6), round(f.depth, 6),
            round(f.strike, 4), round(f.dip, 4), round(f.rake, 4),
            round(f.length, 4), round(f.width, 4), round(f.slip, 6))


def _regional_cache_repr(reg: RegionalStress) -> tuple:
    """Plain-tuple projection of the RegionalStress fields, all of
    which affect the computed optimal-plane stress field -- unlike
    _fault_cache_repr() there's no subset to omit here, every field on
    this dataclass is physically meaningful."""
    return (round(reg.S1, 6), round(reg.S2, 6), round(reg.S3, 6),
            round(reg.S1_strike, 4), round(reg.S1_plunge, 4),
            round(reg.S2_strike, 4), round(reg.S2_plunge, 4))


def cff_volume_cache_key(sources: List[FaultParameters], receiver: Optional[FaultParameters],
                          elastic: ElasticParameters, grid: GridParameters,
                          depths_km: Sequence[float], mode: str = "fixed",
                          regional: Optional[RegionalStress] = None,
                          friction: Optional[float] = None) -> str:
    """Deterministic cache key covering everything that affects the
    computed volume: source geometry/slip, elastic parameters, grid
    extent/resolution, the exact depth-slice list, and -- new --
    `mode` plus whichever of receiver/(regional+friction) that mode
    actually uses. Switching mode, or changing the regional stress
    tensor / friction while in "optimal" mode, changes the key, so a
    cache entry from one mode is never mistakenly reused for the
    other. Callers own where the cache lives (project-scoped temp dir
    recommended, not this module's concern) -- this just derives the
    filename-safe key."""
    if mode not in ("fixed", "optimal"):
        raise ValueError(f"mode must be 'fixed' or 'optimal', got {mode!r}")
    payload = {
        "mode": mode,
        "sources": [_fault_cache_repr(s) for s in sources],
        "mu": elastic.mu, "nu": elastic.nu,
        "grid": (grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max,
                 grid.n_lon, grid.n_lat),
        "depths_km": sorted(round(float(d), 4) for d in depths_km),
    }
    if mode == "fixed":
        payload["receiver"] = _fault_cache_repr(receiver)
        payload["friction"] = elastic.friction
    else:
        payload["regional"] = _regional_cache_repr(regional)
        payload["friction"] = round(friction, 6) if friction is not None else elastic.friction
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.md5(blob).hexdigest()


def save_cff_volume(volume: CFFVolume, path: str) -> None:
    """Cache a computed volume to a .npz file (recompute avoidance --
    each slice is a full grid x sources Okada/DC3D evaluation, and the
    aftershock MC test may want to re-run with different N/M parameters
    against the same volume repeatedly)."""
    np.savez_compressed(
        path, lons=volume.lons, lats=volume.lats, depths_km=volume.depths_km,
        cff_mpa=volume.cff_mpa, used_dc3d=np.array(volume.used_dc3d, dtype=bool),
        near_field_mask=(volume.near_field_mask if volume.near_field_mask is not None
                          else np.zeros_like(volume.cff_mpa, dtype=bool)),
        mode=volume.mode,
    )


def load_cff_volume(path: str) -> Optional[CFFVolume]:
    """Returns None (never raises) if the file doesn't exist or fails to
    load -- caller should treat that as a cache miss and recompute via
    build_cff_volume(), not as an error."""
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        # "mode" was added after the first cache format shipped -- older
        # cache files on disk won't have it; default to "fixed" (every
        # volume built before this field existed WAS built in fixed mode,
        # since "optimal" didn't exist yet) rather than raising a KeyError
        # on an otherwise-valid, still-useful cached volume.
        mode = str(data["mode"]) if "mode" in data else "fixed"
        return CFFVolume(
            lons=data["lons"], lats=data["lats"], depths_km=data["depths_km"],
            cff_mpa=data["cff_mpa"], used_dc3d=list(data["used_dc3d"]),
            near_field_mask=data["near_field_mask"], mode=mode,
        )
    except Exception:
        return None
