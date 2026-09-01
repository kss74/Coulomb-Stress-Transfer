# -*- coding: utf-8 -*-
"""
Calibration and validation of core.rate_state_seismicity's Dieterich
(1994) forecast against a REAL earthquake catalog
(core.eq_catalog_import.EQCatalogEvent / events_to_eq_array() output).

This is the item flagged in
PROJECT_HANDOVER_ADDENDUM_2026-08-21c_external_review_synthesis.md,
Session 3 point 6 ("explicit calibration-vs-forecast workflow
separation") as the largest of that session's asks and deserving its
own design pass rather than a checkbox addition. Three distinct pieces,
kept as three groups of functions in this one new module (new physics/
statistics, existing rate_state_seismicity.py untouched per project
convention):

1. BACKGROUND RATE r0 from catalog data -- Dieterich (1994)'s r0 is
   meant to be an independently-MEASURED quantity (the seismicity rate
   before the mainshock/stress step), not a free parameter fit to the
   post-sequence decay. `background_rate_from_catalog()` computes it
   directly: event count / duration over a user-chosen pre-mainshock
   window, optionally restricted to the modeled region (a CFFVolume's
   lon/lat/depth box) and to M >= Mc.

2. CALIBRATION of asig/ta -- given r0 fixed (from step 1) and a real
   post-mainshock aftershock sequence, `calibrate_rate_state()` fits
   asig and ta (optionally r0 too, via `fit_r0=True`) by nonlinear
   least squares against the OBSERVED cumulative event count curve,
   reusing the exact same closed-form d94()/dieterich_rate_grid()
   solution already validated in rate_state_seismicity.py -- no new
   physics, this module only adds the inverse problem (fit parameters
   to data) on top of the existing forward problem (parameters ->
   forecast).

3. VALIDATION -- `score_forecast()` compares a forecast (calibrated or
   otherwise) against observed catalog counts using two standard
   forecast-verification statistics: an N-test (CSEP-style: is the
   total observed count consistent with a Poisson distribution around
   the total predicted count?) and RMSE/R^2 on the cumulative-count
   curve, plus a Poisson log-likelihood on the per-bin counts. None of
   these require the fit in (2) -- score_forecast() works equally well
   on a held-out time window, a manually-chosen parameter set, or a
   forecast built from the full catalog, which is why it's a separate
   function rather than folded into calibrate_rate_state().

Time handling: catalog events carry an absolute epoch (POSIX seconds
UTC, core.eq_catalog_import.EQCatalogEvent.epoch_s); the rate-state
model works in a caller-chosen relative time unit anchored at the
source event ("mainshock") origin time, also epoch seconds. This
module is the one place that does that epoch-seconds -> model-time-unit
conversion (TIME_UNIT_SECONDS below) -- rate_state_seismicity.py itself
stays unit-agnostic per its own docstring, and this module's UI caller
is responsible for keeping r0/asig/ta/t0/ts expressed in the SAME
chosen unit as everything computed here. See
RateStateParams.time_unit's own docstring (added alongside this
consistency guard) for the single authoritative place that unit is
now recorded, and assert_consistent_time_unit() below for the guard
that catches it drifting out of sync with a `time_unit` argument
passed into this module's own functions.

Spatial filtering: `volume` (a core.cff_volume.CFFVolume) is used only
for its lon/lat/depth bounding box, via `_filter_to_volume()` -- a
cheap box filter, not the trilinear interpolation
core.cff_volume.interpolate_cff_at_points() does for the aftershock MC
test. A box filter is the right tool here because the quantity being
counted is "how many real events occurred in the modeled region", not
"what ΔCFF value applies at each event's exact location" -- box
membership, not interpolated value, is what's needed to decide whether
an event counts toward the region-wide observed rate/count.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Sequence, Tuple

import numpy as np

from .rate_state_seismicity import RateStateParams, dieterich_rate_grid
from .cff_volume import CFFVolume, apply_near_field_mask


# ─── Time units ──────────────────────────────────────────────────────────

TIME_UNIT_SECONDS: Dict[str, float] = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
    "years": 365.25 * 86400.0,
}


def _sec_per_unit(time_unit: str) -> float:
    if time_unit not in TIME_UNIT_SECONDS:
        raise ValueError(
            f"time_unit must be one of {list(TIME_UNIT_SECONDS)}, got {time_unit!r}")
    return TIME_UNIT_SECONDS[time_unit]


def assert_consistent_time_unit(forecast, time_unit: str) -> None:
    """
    Guards against the exact silent-mismatch failure mode this
    function exists to catch: `forecast` (a
    core.rate_state_seismicity.RateStateForecast) carries its own
    `params.time_unit` -- the unit its r0/asig/ta/t0/ts were actually
    set up in (see that dataclass's own docstring) -- and every
    function in this module also takes an explicit `time_unit` used to
    convert a catalog's absolute timestamps into the SAME relative time
    axis. Nothing forces those two to agree; if the UI (or a script)
    lets them drift apart, calibration/validation would silently run
    against a catalog converted into the wrong unit and produce
    confidently-wrong numbers with no error anywhere.

    Call this once, right before using `time_unit` against `forecast`,
    and let the caller decide what to do with the ValueError (the
    dialog layer turns it into a warning box; a script might just let
    it propagate).

    `forecast.params.time_unit is None` (a forecast built by an older
    caller, or directly via RateStateParams without ever setting
    time_unit -- see that field's own "None means caller didn't record
    one" docstring note) is NOT treated as a mismatch: there is nothing
    to check it against, so this is a deliberate no-op in that case,
    not a silent pass-by-default assumption that the units happen to
    agree.
    """
    forecast_unit = getattr(getattr(forecast, "params", None), "time_unit", None)
    if forecast_unit is None:
        return
    if forecast_unit != time_unit:
        raise ValueError(
            f"Time unit mismatch: the forecast was set up in {forecast_unit!r} "
            f"but {time_unit!r} was requested here. Catalog timestamps would be "
            "converted into the wrong time axis relative to the forecast's own "
            "r0/asig/ta/t0/ts -- re-run the forecast with the matching time "
            "unit, or select the matching unit here, before calibrating or "
            "validating against it.")


# ─── Spatial / magnitude filtering ────────────────────────────────────────

def _filter_to_volume(eq_array: Sequence[dict], volume: Optional[CFFVolume]) -> List[dict]:
    """Bounding-box filter on volume.lons/.lats/.depths_km -- see module
    docstring for why a box filter (not interpolation) is the right
    tool here. `volume=None` is a no-op (returns eq_array unchanged),
    so every function below can accept an optional region restriction
    without a separate branch at every call site."""
    if volume is None:
        return list(eq_array)
    lon_min, lon_max = float(volume.lons.min()), float(volume.lons.max())
    lat_min, lat_max = float(volume.lats.min()), float(volume.lats.max())
    z_min, z_max = float(volume.depths_km.min()), float(volume.depths_km.max())
    out = []
    for e in eq_array:
        lon, lat, depth = e.get("lon"), e.get("lat"), e.get("depth")
        if lon is None or lat is None or depth is None:
            continue
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max and z_min <= depth <= z_max:
            out.append(e)
    return out


def _filter_by_magnitude(eq_array: Sequence[dict], mag_min: Optional[float]) -> List[dict]:
    """mag_min=None is a no-op. Events with magnitude=None are DROPPED
    once a cutoff is requested (an unknown magnitude can't be verified
    to be >= Mc), matching the conservative choice
    core.aftershock_mc_test makes for missing coordinates."""
    if mag_min is None:
        return list(eq_array)
    return [e for e in eq_array if e.get("magnitude") is not None and e["magnitude"] >= mag_min]


# ─── 1. Background rate from catalog ──────────────────────────────────────

@dataclass
class BackgroundRateResult:
    r0: float                # events per time_unit
    n_events: int
    duration: float          # in time_unit
    time_unit: str
    t_start: float            # epoch seconds, as given
    t_end: float               # epoch seconds, as given
    mag_min: Optional[float]
    region_restricted: bool   # whether a volume bbox filter was applied
    n_events_before_filters: int   # catalog size before region/magnitude filtering


def background_rate_from_catalog(eq_array: Sequence[dict],
                                  t_start_epoch_s: float, t_end_epoch_s: float,
                                  time_unit: str = "days",
                                  mag_min: Optional[float] = None,
                                  volume: Optional[CFFVolume] = None) -> BackgroundRateResult:
    """
    Empirical background rate r0 = (event count) / (window duration),
    counted over a PRE-MAINSHOCK (or any reference) window
    [t_start_epoch_s, t_end_epoch_s) -- both absolute POSIX seconds
    UTC, matching core.eq_catalog_import.EQCatalogEvent.epoch_s. This
    is the direct, catalog-measured r0 Dieterich (1994) itself assumes
    is known independently (see module docstring) -- distinct from
    fitting r0 to the post-sequence decay, which calibrate_rate_state()
    below supports via fit_r0=True for cases where a clean pre-sequence
    window isn't available.

    Events with epoch_s=None (unparseable/unmapped time) are excluded
    -- they can't be placed inside or outside the window.
    """
    if t_end_epoch_s <= t_start_epoch_s:
        raise ValueError("t_end_epoch_s must be greater than t_start_epoch_s")
    sec_per_unit = _sec_per_unit(time_unit)

    n_before = len(eq_array)
    events = _filter_to_volume(eq_array, volume)
    events = _filter_by_magnitude(events, mag_min)

    n_events = sum(
        1 for e in events
        if e.get("epoch_s") is not None and t_start_epoch_s <= e["epoch_s"] < t_end_epoch_s
    )
    duration = (t_end_epoch_s - t_start_epoch_s) / sec_per_unit
    r0 = n_events / duration if duration > 0 else float("nan")

    return BackgroundRateResult(
        r0=r0, n_events=n_events, duration=duration, time_unit=time_unit,
        t_start=t_start_epoch_s, t_end=t_end_epoch_s, mag_min=mag_min,
        region_restricted=(volume is not None),
        n_events_before_filters=n_before,
    )


# ─── 1b. Catalog timeline (QA/QC, window-independent) ─────────────────────

@dataclass
class CatalogTimeline:
    """
    Window-INDEPENDENT view of a catalog's time distribution relative to
    the mainshock -- see build_catalog_timeline()'s docstring for why
    this exists as a separate function/dataclass from
    background_rate_from_catalog()/build_observed_time_series() rather
    than folded into either of them.
    """
    rel_times: np.ndarray          # (n,) sorted, region+magnitude filtered, epoch_s-having events only
    magnitudes: np.ndarray         # (n,) matching magnitudes, NaN where unknown
    time_unit: str
    mainshock_epoch_s: float
    mag_min: Optional[float]
    region_restricted: bool
    n_total: int                   # eq_array length, no filters at all
    n_with_time: int                # of those, events with a parseable epoch_s (region/mag NOT yet applied)
    n_after_mag_filter: int         # after magnitude cutoff only, still requiring epoch_s
    n_after_region_filter: int      # after BOTH magnitude and region filters, still requiring epoch_s -- len(rel_times)
    t0: Optional[float]
    t_max: Optional[float]
    t_fit_max: Optional[float]
    n_in_forecast_window: Optional[int]     # rel_times in [t0, t_max], or None if t0/t_max not given
    n_in_calibration_window: Optional[int]  # rel_times in [t0, t_fit_max], or None if t_fit_max/t0 not given
    n_in_validation_window: Optional[int]   # rel_times in (t_fit_max, t_max], or None if t_fit_max/t_max not given


def build_catalog_timeline(eq_array: Sequence[dict], mainshock_epoch_s: float,
                            time_unit: str = "days",
                            mag_min: Optional[float] = None,
                            volume: Optional[CFFVolume] = None,
                            t0: Optional[float] = None, t_max: Optional[float] = None,
                            t_fit_max: Optional[float] = None) -> CatalogTimeline:
    """
    QA/QC view of a catalog's time distribution: unlike
    build_observed_time_series() (item 2 below), which only counts
    events that fall INSIDE a specific forecast's [t0, ts[-1]] window
    and silently drops everything else, this returns EVERY qualifying
    event's relative time regardless of whether it falls inside any
    particular window, plus the exact per-stage funnel a real catalog's
    calibration/validation counts are built from:

        n_total -> n_with_time -> n_after_mag_filter ->
        n_after_region_filter -> n_in_forecast_window ->
        (n_in_calibration_window + n_in_validation_window)

    Purpose-built to make a "1570 events fit against the calibration
    window, but 0 events scored in validation" kind of report
    immediately diagnosable at a glance: is that genuinely because the
    aftershock sequence ended early (a real, interesting result), or
    because every epoch_s silently came back None from a DateTime-typed
    GeoPackage column that got stringified wrong (see
    core.observation_import._stringify_attr's docstring for exactly
    that failure mode -- n_with_time stuck at 0 while n_total is large
    is the smoking gun for that specific bug), or a mismatched region/
    magnitude filter between two calls that were supposed to agree (the
    OTHER failure mode this project already hit once, see
    rate_state_dialog.py's _on_finished() stale-background-rate notes)?
    Reading the funnel top-to-bottom answers that without re-deriving it
    from two separately-reported numbers by hand.

    t0/t_max/t_fit_max are all optional and purely for the window-count
    fields / plot overlay -- rel_times/magnitudes themselves are NOT
    restricted by them (that's the whole point: this function shows the
    catalog's full time distribution; the caller decides what window(s)
    to draw on top of it). t_max is typically the forecast's ts[-1].
    """
    sec_per_unit = _sec_per_unit(time_unit)

    n_total = len(eq_array)
    n_with_time = sum(1 for e in eq_array if e.get("epoch_s") is not None)

    after_mag = _filter_by_magnitude(eq_array, mag_min)
    n_after_mag = sum(1 for e in after_mag if e.get("epoch_s") is not None)

    after_region = _filter_to_volume(after_mag, volume)
    rel_list: List[float] = []
    mag_list: List[float] = []
    for e in after_region:
        if e.get("epoch_s") is None:
            continue
        rel_list.append((e["epoch_s"] - mainshock_epoch_s) / sec_per_unit)
        m = e.get("magnitude")
        mag_list.append(float(m) if m is not None else float("nan"))

    rel_arr = np.asarray(rel_list, dtype=float)
    mag_arr = np.asarray(mag_list, dtype=float)
    order = np.argsort(rel_arr)
    rel_times = rel_arr[order]
    magnitudes = mag_arr[order]
    n_after_region = int(rel_times.size)

    def _count_between(lo, hi):
        mask = np.ones(rel_times.shape, dtype=bool)
        if lo is not None:
            mask &= rel_times >= lo
        if hi is not None:
            mask &= rel_times <= hi
        return int(mask.sum())

    n_in_forecast = _count_between(t0, t_max) if (t0 is not None or t_max is not None) else None
    if t0 is not None and t_fit_max is not None:
        n_in_calib = _count_between(t0, t_fit_max)
    else:
        n_in_calib = None
    if t_fit_max is not None:
        # Strict '>' at the lower edge matches the "clean, non-overlapping
        # windows" convention (t_fit_max belongs to calibration, not
        # validation) rather than double-counting the boundary point in
        # both windows.
        mask = rel_times > t_fit_max
        if t_max is not None:
            mask &= rel_times <= t_max
        n_in_valid = int(mask.sum())
    else:
        n_in_valid = None

    return CatalogTimeline(
        rel_times=rel_times, magnitudes=magnitudes, time_unit=time_unit,
        mainshock_epoch_s=mainshock_epoch_s, mag_min=mag_min,
        region_restricted=(volume is not None),
        n_total=n_total, n_with_time=n_with_time,
        n_after_mag_filter=n_after_mag, n_after_region_filter=n_after_region,
        t0=t0, t_max=t_max, t_fit_max=t_fit_max,
        n_in_forecast_window=n_in_forecast,
        n_in_calibration_window=n_in_calib,
        n_in_validation_window=n_in_valid,
    )


# ─── 2. Observed post-mainshock time series ───────────────────────────────

@dataclass
class ObservedTimeSeries:
    """Observed aftershock counts on the SAME time grid `ts` a
    RateStateForecast was (or will be) evaluated on, so
    observed.cumulative can be plotted/scored directly against
    forecast.total_cumulative() point-for-point. Both `cumulative` and
    `counts_per_bin` are counted since t0 (cumulative[i] = number of
    qualifying events with t0 <= rel_t <= ts[i]), mirroring how
    RateStateForecast.total_cumulative() is itself integrated from t0
    (core.rate_state_seismicity.forecast_from_cff_volume docstring)."""
    ts: np.ndarray
    t0: float
    mainshock_epoch_s: float
    time_unit: str
    cumulative: np.ndarray        # (n_t,) observed cumulative count at each ts[i]
    counts_per_bin: np.ndarray    # (n_t,) counts_per_bin[0] = cumulative[0]; counts_per_bin[i>0] = cumulative[i]-cumulative[i-1]
    rel_times: np.ndarray         # sorted (t - mainshock)/sec_per_unit for every qualifying event, for histograms/plots
    mag_min: Optional[float]
    region_restricted: bool
    n_events_total: int            # events falling within [t0, ts[-1]]
    n_events_before_filters: int


def build_observed_time_series(eq_array: Sequence[dict], mainshock_epoch_s: float,
                                ts, t0: float, time_unit: str = "days",
                                mag_min: Optional[float] = None,
                                volume: Optional[CFFVolume] = None) -> ObservedTimeSeries:
    """
    Bins a real catalog against forecast time grid `ts` (same array
    passed to core.rate_state_seismicity.forecast_from_cff_volume) and
    forecast start `t0`, relative to `mainshock_epoch_s` (the source
    event's absolute origin time -- FaultParameters carries no time
    field in this project, see module docstring, so this is always a
    separate caller-supplied value).

    Events before t0 or after ts[-1] don't contribute to `cumulative`/
    `counts_per_bin` (they fall outside the forecast window being
    scored) but ARE included in `rel_times` if within [ts.min(),
    ts.max()]... actually rel_times mirrors the same [t0, ts[-1]]
    window as cumulative, for consistency between the two.
    """
    ts = np.asarray(ts, dtype=float)
    if ts.size == 0:
        raise ValueError("ts is empty")
    sec_per_unit = _sec_per_unit(time_unit)

    n_before = len(eq_array)
    events = _filter_to_volume(eq_array, volume)
    events = _filter_by_magnitude(events, mag_min)

    rel = []
    for e in events:
        if e.get("epoch_s") is None:
            continue
        rel.append((e["epoch_s"] - mainshock_epoch_s) / sec_per_unit)
    rel = np.asarray(rel, dtype=float)

    window = rel[(rel >= t0) & (rel <= ts[-1])]
    sorted_rel = np.sort(window)

    cumulative = np.searchsorted(sorted_rel, ts, side="right").astype(float)
    counts_per_bin = np.diff(cumulative, prepend=0.0)

    return ObservedTimeSeries(
        ts=ts, t0=t0, mainshock_epoch_s=mainshock_epoch_s, time_unit=time_unit,
        cumulative=cumulative, counts_per_bin=counts_per_bin, rel_times=sorted_rel,
        mag_min=mag_min, region_restricted=(volume is not None),
        n_events_total=int(sorted_rel.size), n_events_before_filters=n_before,
    )


# ─── 3. Calibration (fit asig/ta, optionally r0) ──────────────────────────

@dataclass
class CalibrationResult:
    params: RateStateParams        # fitted (or partly-fitted) parameters
    fit_r0: bool
    success: bool
    message: str
    observed: ObservedTimeSeries
    predicted_cumulative: np.ndarray   # on the SAME observed.ts grid, full range (not just the fit window)
    n_fit_points: int                  # number of ts points actually used in the fit (<= len(ts) if t_fit_max given)
    t_fit_max: Optional[float]
    # ── Added 2026-08-24 (bounded fit + uncertainty, see
    # PROJECT_HANDOVER_ADDENDUM_2026-08-24b_calibration_ta_runaway_fix.md):
    # bounds actually used this fit, so a caller/UI can tell "the fitted
    # value landed at/near a bound" apart from "the fit converged to an
    # interior optimum" without re-deriving the bounds itself.
    asig_bounds: Tuple[float, float]
    ta_bounds: Tuple[float, float]
    r0_bounds: Optional[Tuple[float, float]]   # None when fit_r0=False (r0 was fixed, not bounded)
    # 1-sigma standard errors in REAL (not log) parameter space, from the
    # linearized (delta-method) covariance of the log-space fit -- see
    # _fit_uncertainty()'s docstring. None for a parameter that wasn't
    # fit (r0_stderr when fit_r0=False) or when the covariance couldn't
    # be computed (singular/near-singular J^T J -- itself a sign of a
    # poorly-determined parameter, see ta_at_bound/well_determined below).
    asig_stderr: Optional[float]
    ta_stderr: Optional[float]
    r0_stderr: Optional[float]
    ta_at_bound: bool           # fitted ta within 1% (log-space) of ta_bounds[0] or [1]
    well_determined: bool       # False if ta_at_bound, or ta_stderr is None/undefined, or
                                 # ta's relative stderr exceeds WELL_DETERMINED_RELERR_MAX --
                                 # a caller-facing "don't trust this ta at face value" flag


# Relative-standard-error threshold (in log-parameter space, i.e.
# roughly "fractional uncertainty") above which a fitted ta is flagged
# well_determined=False even if it didn't literally hit a bound -- a
# generous, deliberately non-tight threshold (50% = the fit can't even
# say the parameter is right to within a factor of ~1.5) chosen to flag
# genuinely unusable fits without also flagging every merely-noisy one.
WELL_DETERMINED_RELERR_MAX = 0.5

# How close (in log-space, i.e. a ratio) a fitted value has to land to
# either bound edge before being flagged ta_at_bound=True. 1% in log-
# space is well inside floating-point/optimizer-tolerance noise at a
# bound, so this reliably distinguishes "the optimizer was stopped BY
# the bound" from "the optimizer happened to converge to a similar
# value on its own".
_BOUND_PROXIMITY_LOG = 0.01


def _default_log_bounds(asig0: float, ta0: float, r0_guess: Optional[float],
                         fit_r0: bool, fit_window: float,
                         ta_ceiling_factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bounds (in LOG-parameter space, matching log_x's own ordering) for
    the bounded (method="trf") fit -- see
    PROJECT_HANDOVER_ADDENDUM_2026-08-24b_calibration_ta_runaway_fix.md
    for the diagnosis this responds to: an unbounded LM fit can walk ta
    to an arbitrary, physically-meaningless value (observed: 78231.59)
    once ta exceeds roughly 10-20x the fit window, because the model's
    exp(-t/ta) term goes flat to 1 there and the cost surface stops
    depending on ta at all (verified directly: cost changed <0.3% over
    a 12x range in ta at that point) -- least_squares then reports
    success=True purely because ftol (cost stopped changing) was
    satisfied, not because ta was determined.

    ta bounds: [ta_floor, ta_ceiling_factor * fit_window]. The ceiling
    is deliberately expressed as a MULTIPLE of the fit window (not a
    fixed number) since "how much longer than the data we have could a
    plausible decay time be" is a window-relative question, not an
    absolute one -- a 50x default means ta can be fit anywhere from
    "much faster than the data resolves" up to "50x longer than the
    window", which already covers the physically-uninformative regime
    (the model is essentially indistinguishable from ta=infinity well
    before 50x) while still producing a finite, actionable number
    instead of an arbitrary one. ta_floor is a fixed small constant,
    not window-relative, since a legitimately very fast decay (much
    shorter than one bin) is physically fine and shouldn't be
    window-limited the way an implausibly slow one should.

    asig bounds: wide and NOT window-relative (asig is a stress-scale
    parameter, unrelated to the time axis) -- 1e-8 to 1e6 MPa is many
    orders of magnitude beyond any physically plausible a*sigma value
    and is not expected to bind in practice (see the exploration this
    fix followed: asig recovered well even in every scenario where ta
    ran away). Its purpose here is only to keep the optimizer's search
    space finite for method="trf" (which requires finite bounds), not
    to add a real physical constraint the way the ta ceiling does.

    r0 bounds (fit_r0=True only): centered on r0_guess, +/- 4 orders of
    magnitude -- generous enough not to bind for any sane initial
    guess, while still keeping the search space finite.
    """
    ta_floor = 1e-6
    ta_ceiling = max(ta_ceiling_factor * fit_window, ta_floor * 10.0)
    asig_floor, asig_ceiling = 1e-8, 1e6

    lo = [asig_floor, ta_floor]
    hi = [asig_ceiling, ta_ceiling]
    if fit_r0:
        r0g = max(r0_guess, 1e-9)
        lo = [r0g / 1e4] + lo
        hi = [r0g * 1e4] + hi

    return np.log(np.array(lo, dtype=float)), np.log(np.array(hi, dtype=float))


def _clip_log_x0(log_x0: np.ndarray, log_lo: np.ndarray, log_hi: np.ndarray) -> np.ndarray:
    """trf requires x0 strictly inside (or on) its bounds; an initial
    guess (asig0/ta0, or a caller-supplied r0 used as fit_r0=True's
    starting point) that happens to fall outside the bounds computed
    above (e.g. a ta0 default left at the UI's flat 100.0 for a very
    short fit window, or a very large one) would otherwise raise
    instead of just starting from the nearest edge -- clip with a tiny
    inward margin so it's strictly interior, not sitting exactly on the
    edge where trf's own internal scaling can behave poorly."""
    margin = 1e-6
    return np.clip(log_x0, log_lo + margin, log_hi - margin)


def _fit_uncertainty(result, n_params: int) -> Optional[np.ndarray]:
    """
    1-sigma standard errors of the FITTED LOG-PARAMETERS (not yet
    converted to real space -- see calibrate_rate_state's docstring for
    that conversion), from the standard linearized nonlinear-least-
    squares covariance estimate: cov = s^2 * pinv(J^T J), where
    s^2 = 2*cost/dof is the residual variance estimate (result.cost is
    scipy's own 0.5*sum(residuals**2), hence the factor of 2) and J is
    the fit's own final Jacobian (result.jac) -- the same estimator
    curve_fit() uses internally, applied here to the residuals()
    closure directly since this fit already needed a custom closure
    (per-bin sqrt-transform) that curve_fit()'s own interface doesn't
    fit naturally.

    Deliberately uses pinv (Moore-Penrose pseudo-inverse), not a plain
    inverse: a genuinely flat direction (exactly the ta-runaway failure
    mode this whole fix responds to) makes J^T J singular or
    near-singular, and a plain inv() would raise or return garbage
    right when the uncertainty estimate matters most. pinv() instead
    returns a large-but-finite variance along that flat direction,
    correctly reporting "this parameter is poorly determined" as a
    large stderr rather than crashing.

    Returns None (rather than an array of NaN/inf) if dof<=0 (fit had
    exactly as many data points as free parameters -- no residual
    degrees of freedom to estimate a variance from) so callers can
    write a single `if stderr is None` check instead of also handling
    NaN-laced arrays.
    """
    jac = np.atleast_2d(result.jac)
    n_data = jac.shape[0]
    dof = n_data - n_params
    if dof <= 0:
        return None
    s_sq = 2.0 * result.cost / dof
    jtj = jac.T @ jac
    cov = s_sq * np.linalg.pinv(jtj)
    diag = np.diag(cov)
    # pinv on a near-singular matrix can occasionally return a tiny
    # negative diagonal entry from floating-point round-off rather than
    # exactly zero; clip before sqrt so that shows up as "~0 stderr"
    # rather than raising on sqrt(negative).
    return np.sqrt(np.clip(diag, 0.0, None))


def _predicted_cumulative(log_x, cff: np.ndarray, n_points: int, ts: np.ndarray, t0: float,
                           fixed_r0: Optional[float], fit_r0: bool) -> np.ndarray:
    """Shared by the least-squares objective and the final full-range
    evaluation. `log_x` holds log(r0), log(asig), log(ta) (or just
    log(asig), log(ta) when fit_r0=False) -- see calibrate_rate_state's
    docstring for why the fit is done in log-parameter space. Builds a
    per-grid-point RateStateParams (r0 split across n_points, exactly
    as forecast_from_cff_volume() does) from the current trial
    parameters, evaluates dieterich_rate_grid() directly against the
    volume's own cff array (not through forecast_from_cff_volume(), to
    avoid rebuilding a RateStateForecast object every optimizer
    iteration), and returns the whole-region total cumulative count --
    the nansum-over-grid-points reduction mirrors
    RateStateForecast.total_cumulative() exactly (nan-safe: grid
    points outside the interpolated CFFVolume stay NaN and are
    excluded, same rationale as that method)."""
    if fit_r0:
        r0, asig, ta = np.exp(log_x)
    else:
        asig, ta = np.exp(log_x)
        r0 = fixed_r0
    per_point = RateStateParams(r0=r0 / n_points, asig=asig, ta=ta)
    _, C = dieterich_rate_grid(ts, t0, per_point, cff)
    return np.nansum(C, axis=tuple(range(C.ndim - 1)))


def _predicted_bins(log_x, cff, n_points, ts, t0, fixed_r0, fit_r0):
    """Per-bin (not cumulative) predicted counts: diff of
    _predicted_cumulative(), prepended with 0 exactly as
    ObservedTimeSeries.counts_per_bin is built (bin 0 = count in
    [t0, ts[0]]). See calibrate_rate_state's docstring for why the fit
    objective uses these bins (via a variance-stabilizing transform)
    rather than raw cumulative counts."""
    C = _predicted_cumulative(log_x, cff, n_points, ts, t0, fixed_r0, fit_r0)
    if not np.all(np.isfinite(C)):
        return np.full(ts.shape, np.inf)
    return np.diff(C, prepend=0.0)


def calibrate_rate_state(volume: CFFVolume, eq_array: Sequence[dict],
                          mainshock_epoch_s: float, t0: float, ts,
                          time_unit: str = "days",
                          r0: Optional[float] = None, fit_r0: bool = False,
                          asig0: float = 0.01, ta0: float = 100.0,
                          mag_min: Optional[float] = None,
                          t_fit_max: Optional[float] = None,
                          exclude_near_field: bool = True,
                          ta_ceiling_factor: float = 50.0) -> CalibrationResult:
    """
    Fits asig and ta (and, if fit_r0=True, r0 as well) by nonlinear
    least squares so the model's per-bin predicted event counts match
    the REAL observed per-bin aftershock counts from `eq_array`,
    reusing the exact closed-form solution already validated in
    rate_state_seismicity.py (dieterich_rate_grid) -- see module
    docstring, point 2.

    Fits PER-BIN counts, not raw cumulative counts: with the default
    tdotr=asig/ta (Dieterich's own default when tdotr isn't
    independently specified, see RateStateParams), the Dieterich rate
    R(t) always relaxes back to exactly r0 as t grows (A = r0*tdot/
    tdotr = r0 identically in that case -- see d94()'s eq. 12 form),
    regardless of asig/ta. That means the CUMULATIVE
    count curve is dominated, at any reasonably long forecast horizon,
    by the linear r0*t background contribution -- asig/ta only shape
    the early transient "bump" on top of it, a small fraction of the
    total cumulative count once t >> ta. Least-squares on raw
    cumulative counts would then be numerically insensitive to asig/ta
    (verified empirically during this module's own build: fits landed
    arbitrarily far from the generating parameters despite hundreds of
    synthetic events). Fitting per-bin counts through a
    variance-stabilizing sqrt transform (sqrt(predicted) -
    sqrt(observed), the standard Poisson-count least-squares
    weighting) instead gives the early, transient-dominated bins their
    correct statistical weight, which is where the asig/ta signal
    actually lives.

    Fits in LOG-parameter space (log(asig), log(ta), and log(r0) if
    fit_r0=True), not linear space: asig (~1e-3 to 1e-1 MPa) and ta
    (~1 to 1e3 time units) routinely differ by several orders of
    magnitude, which without rescaling leaves scipy's trust-region
    solver effectively blind to the badly-conditioned direction --
    verified empirically during this module's own build, a linear-
    space fit on a clean noiseless synthetic curve (where the true
    parameters are the unique, verified zero-residual minimum) still
    wandered to a ta over 15x the true value, while the identical
    problem in log-space converges to within a few percent. Log-space
    fitting is also physically appropriate here since both asig and ta
    are strictly-positive scale parameters -- it enforces positivity
    for free, with no need for the bounds= a linear-space fit would
    otherwise require.

    r0 : fixed background rate to use when fit_r0=False (the
        recommended mode -- pass the output of
        background_rate_from_catalog().r0, an independently-measured
        quantity, per Dieterich's own framing). Required if
        fit_r0=False. Ignored (used only as the initial guess) if
        fit_r0=True.
    t_fit_max : if given, only observed points with ts <= t_fit_max are
        used to FIT asig/ta -- the remaining later points are still
        scored in `predicted_cumulative` (evaluated over the FULL ts
        range) so the caller can validate the fit against a genuinely
        held-out later time window via score_forecast(), the actual
        "checking/validation of aftershock prediction with aftershock
        data" workflow this module exists for. None (default) fits
        against the entire observed window.

    exclude_near_field (default True, added 2026-08-22 smoke-test fix):
    same near-field exclusion forecast_from_cff_volume() applies (see
    core.cff_volume.apply_near_field_mask's docstring) -- a handful of
    near-field singular ΔCFF cells feeding raw into dieterich_rate_grid
    here would bias the fit the same way they'd distort a forecast, so
    this mirrors that default rather than silently fitting against an
    unfiltered field.

    ta_ceiling_factor (default 50.0, added 2026-08-24 -- see
    PROJECT_HANDOVER_ADDENDUM_2026-08-24b_calibration_ta_runaway_fix.md):
    the fit is now BOUNDED (method="trf", not the previous unbounded
    method="lm") specifically to stop ta from running away to an
    arbitrary, physically-meaningless value when the data doesn't
    constrain it (observed in practice: a fit landing on ta=78231.59).
    ta is bounded to [1e-6, ta_ceiling_factor * fit_window] where
    fit_window = ts_fit[-1] - t0; asig and (if fit_r0=True) r0 are also
    given wide, non-binding-in-practice bounds since method="trf"
    requires finite bounds on every parameter -- see
    _default_log_bounds()'s own docstring for the full reasoning on
    each bound. CalibrationResult.ta_at_bound / .well_determined report
    whether the fit actually needed that ceiling (a "don't trust this
    number" signal, not a reason to think the fit itself failed --
    hitting the ceiling under the OLD unbounded fit is exactly the
    78231.59 symptom this replaces with a flagged, capped value).

    Raises ValueError if fit_r0=False and r0 is None, or if there are
    fewer qualifying observed events than free parameters (an
    under-determined fit isn't silently attempted).
    """
    if not fit_r0 and r0 is None:
        raise ValueError("r0 must be given when fit_r0=False (use "
                          "background_rate_from_catalog(), or pass fit_r0=True)")

    from scipy.optimize import least_squares

    ts = np.asarray(ts, dtype=float)
    observed = build_observed_time_series(
        eq_array, mainshock_epoch_s, ts, t0, time_unit=time_unit,
        mag_min=mag_min, volume=volume)

    n_free = 3 if fit_r0 else 2
    if observed.n_events_total < n_free:
        raise ValueError(
            f"only {observed.n_events_total} observed event(s) in the forecast "
            f"window [t0={t0:g}, ts[-1]={ts[-1]:g}] {time_unit} -- need at least "
            f"{n_free} to fit {n_free} free parameter(s).")

    if t_fit_max is not None:
        fit_mask = ts <= t_fit_max
        if fit_mask.sum() < n_free:
            raise ValueError(
                f"t_fit_max={t_fit_max:g} leaves only {int(fit_mask.sum())} ts "
                f"point(s) -- need at least {n_free}.")
    else:
        fit_mask = np.ones(ts.shape, dtype=bool)

    ts_fit = ts[fit_mask]
    observed_fit_bins = observed.counts_per_bin[fit_mask]
    n_points = int(volume.cff_mpa.size)
    cff = apply_near_field_mask(volume.cff_mpa, volume.near_field_mask, exclude_near_field)

    fit_window = max(ts_fit[-1] - t0, 1e-9)
    r0_guess = None
    if fit_r0:
        r0_guess = r0 if r0 is not None else max(
            observed.cumulative[fit_mask][-1] / fit_window, 1e-6)
        log_x0 = np.log(np.array([r0_guess, asig0, ta0], dtype=float))
    else:
        log_x0 = np.log(np.array([asig0, ta0], dtype=float))

    log_lo, log_hi = _default_log_bounds(
        asig0, ta0, r0_guess, fit_r0, fit_window, ta_ceiling_factor)
    log_x0 = _clip_log_x0(log_x0, log_lo, log_hi)

    sqrt_obs = np.sqrt(np.clip(observed_fit_bins, 0.0, None))

    def residuals(log_x):
        pred_bins = _predicted_bins(log_x, cff, n_points, ts_fit, t0, r0, fit_r0)
        pred_bins = np.where(np.isfinite(pred_bins), pred_bins, 1e12)
        return np.sqrt(np.clip(pred_bins, 0.0, None)) - sqrt_obs

    # method="trf" (not the previous unbounded "lm"): bounds ta (and,
    # loosely, asig/r0) to a finite, physically-motivated range -- see
    # ta_ceiling_factor's own docstring above and _default_log_bounds()
    # for why. trf is the standard scipy choice for bounded nonlinear
    # least squares (lm does not support bounds= at all).
    result = least_squares(residuals, log_x0, method="trf", bounds=(log_lo, log_hi))

    if fit_r0:
        fitted_r0, fitted_asig, fitted_ta = np.exp(result.x)
    else:
        fitted_r0 = r0
        fitted_asig, fitted_ta = np.exp(result.x)

    fitted_params = RateStateParams(r0=fitted_r0, asig=fitted_asig, ta=fitted_ta,
                                    time_unit=time_unit)

    # Re-evaluate over the FULL ts range (not just ts_fit) using the
    # fitted parameter vector, so held-out later points (t_fit_max
    # case) get a genuine out-of-sample prediction to score against.
    full_pred = _predicted_cumulative(result.x, cff, n_points, ts, t0, r0, fit_r0)

    # ── Uncertainty + bound-proximity diagnostics (2026-08-24) ────────
    log_stderr = _fit_uncertainty(result, n_params=log_x0.size)
    if fit_r0:
        idx_r0, idx_asig, idx_ta = 0, 1, 2
        r0_bounds = (float(np.exp(log_lo[idx_r0])), float(np.exp(log_hi[idx_r0])))
        r0_stderr = (float(fitted_r0 * log_stderr[idx_r0])
                     if log_stderr is not None else None)
    else:
        idx_asig, idx_ta = 0, 1
        r0_bounds = None
        r0_stderr = None
    asig_bounds = (float(np.exp(log_lo[idx_asig])), float(np.exp(log_hi[idx_asig])))
    ta_bounds = (float(np.exp(log_lo[idx_ta])), float(np.exp(log_hi[idx_ta])))
    # Delta-method conversion from log-space stderr to real-space stderr:
    # for x = exp(log_x), d(x)/d(log_x) = x, so stderr(x) ~= x * stderr(log_x)
    # to first order -- exact for the small-stderr case this is meant to
    # characterize; a huge log_stderr (the flat-direction case) already
    # produces a huge, correctly-alarming real-space stderr under this
    # same formula, so no separate large-uncertainty branch is needed.
    asig_stderr = (float(fitted_asig * log_stderr[idx_asig])
                   if log_stderr is not None else None)
    ta_stderr = (float(fitted_ta * log_stderr[idx_ta])
                 if log_stderr is not None else None)

    ta_at_bound = bool(
        (result.x[idx_ta] - log_lo[idx_ta] < _BOUND_PROXIMITY_LOG) or
        (log_hi[idx_ta] - result.x[idx_ta] < _BOUND_PROXIMITY_LOG))
    ta_relerr = (ta_stderr / fitted_ta) if (ta_stderr is not None and fitted_ta > 0) else None
    well_determined = bool(
        (not ta_at_bound) and ta_relerr is not None and ta_relerr <= WELL_DETERMINED_RELERR_MAX)

    return CalibrationResult(
        params=fitted_params, fit_r0=fit_r0, success=bool(result.success),
        message=str(result.message), observed=observed,
        predicted_cumulative=full_pred, n_fit_points=int(fit_mask.sum()),
        t_fit_max=t_fit_max,
        asig_bounds=asig_bounds, ta_bounds=ta_bounds, r0_bounds=r0_bounds,
        asig_stderr=asig_stderr, ta_stderr=ta_stderr, r0_stderr=r0_stderr,
        ta_at_bound=ta_at_bound, well_determined=well_determined,
    )


# ─── 4. Validation / scoring ──────────────────────────────────────────────

@dataclass
class ValidationScore:
    n_obs: float                # observed total count over the scored window
    n_pred: float                # predicted total count over the scored window
    n_ratio: float                # n_obs / n_pred
    n_test_p_value: float        # two-sided Poisson N-test p-value (CSEP-style)
    rmse: float                    # on the cumulative-count curve
    nrmse: float                   # rmse / (observed cumulative range), nan if range is 0
    r2: float                      # coefficient of determination on the cumulative curve
    poisson_loglik: float          # sum over bins of Poisson log-likelihood of counts_per_bin given predicted per-bin counts
    n_points_scored: int
    t_range: Tuple[float, float]   # (ts.min(), ts.max()) actually scored, after masking


def score_forecast(observed: ObservedTimeSeries, predicted_cumulative,
                    t_min: Optional[float] = None, t_max: Optional[float] = None) -> ValidationScore:
    """
    Scores a model forecast's total_cumulative() (or
    CalibrationResult.predicted_cumulative, or any array on the same
    `observed.ts` grid) against the real observed catalog counts
    already binned into `observed` (build_observed_time_series()).
    Deliberately takes ObservedTimeSeries + a bare predicted array
    (not a CalibrationResult or RateStateForecast) so it works equally
    for: scoring a fresh forecast that was never fit to this catalog,
    scoring calibrate_rate_state()'s own fit, or scoring only a
    held-out later window via t_min/t_max (e.g. t_min=t_fit_max to
    check predictive skill beyond the window a calibration was fit on
    -- the actual "checking/validation... with aftershock data"
    workflow requested).

    N-test: standard CSEP-style number test -- is the observed total
    count consistent with Poisson(n_pred)? Two-sided: p = 2*min(
    CDF(n_obs; n_pred), SF(n_obs-1; n_pred)), capped at 1. A p-value
    near 0 (either extreme) signals the model's total event budget is
    inconsistent with what was actually observed; this does NOT test
    the SHAPE of the decay (RMSE/R2/log-likelihood below do).

    Poisson log-likelihood uses PER-BIN counts (observed.counts_per_bin
    vs the predicted array's own bin-to-bin differences), the standard
    point-process scoring quantity (bins with zero predicted count are
    only included if the corresponding observed count is also zero --
    log(0) would otherwise blow up a well-fit bin's score for a single
    numerically-negligible zero-crossing).
    """
    from scipy.stats import poisson as _poisson

    pred = np.asarray(predicted_cumulative, dtype=float)
    if pred.shape != observed.ts.shape:
        raise ValueError(
            f"predicted_cumulative shape {pred.shape} must match "
            f"observed.ts shape {observed.ts.shape}")

    mask = np.ones(observed.ts.shape, dtype=bool)
    if t_min is not None:
        mask &= observed.ts >= t_min
    if t_max is not None:
        mask &= observed.ts <= t_max
    finite_mask = mask & np.isfinite(pred) & np.isfinite(observed.cumulative)
    idx = np.where(finite_mask)[0]
    if idx.size == 0:
        raise ValueError("no valid (finite, in-range) points to score")

    obs_c = observed.cumulative[idx]
    pred_c = pred[idx]

    # Rebaseline both curves to 0 at the first scored point so t_min>0
    # (a held-out later window) scores the INCREMENT over that window,
    # not the cumulative total carried in from before t_min -- matches
    # how counts_per_bin is itself a since-t0 increment.
    obs_c = obs_c - obs_c[0]
    pred_c = pred_c - pred_c[0]

    resid = pred_c - obs_c
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    obs_range = float(obs_c.max() - obs_c.min())
    nrmse = float(rmse / obs_range) if obs_range > 0 else float("nan")
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((obs_c - obs_c.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    n_obs_total = float(obs_c[-1])
    n_pred_total = float(pred_c[-1])
    n_ratio = n_obs_total / n_pred_total if n_pred_total > 0 else float("inf")
    if n_pred_total > 0:
        k = n_obs_total
        p = 2.0 * min(_poisson.cdf(k, n_pred_total), _poisson.sf(max(k - 1, -1), n_pred_total))
        p = float(min(1.0, max(0.0, p)))
    else:
        p = float("nan")

    obs_bins = np.diff(obs_c, prepend=0.0)
    pred_bins = np.diff(pred_c, prepend=0.0)
    ll = 0.0
    for k_obs, lam in zip(obs_bins, pred_bins):
        if lam <= 0:
            if k_obs == 0:
                continue  # 0 observed, 0 predicted: perfectly consistent, contributes nothing
            ll = -np.inf
            break
        ll += k_obs * math.log(lam) - lam - math.lgamma(k_obs + 1.0)

    return ValidationScore(
        n_obs=n_obs_total, n_pred=n_pred_total, n_ratio=n_ratio, n_test_p_value=p,
        rmse=rmse, nrmse=nrmse, r2=r2, poisson_loglik=float(ll),
        n_points_scored=int(idx.size),
        t_range=(float(observed.ts[idx].min()), float(observed.ts[idx].max())),
    )
