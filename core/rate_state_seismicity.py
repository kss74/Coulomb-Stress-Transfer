# -*- coding: utf-8 -*-
"""
Rate-and-state seismicity forecasting following Dieterich (1994).

Dieterich, J. (1994), A constitutive law for rate of earthquake
production and its application to earthquake clustering, J. Geophys.
Res., 99(B2), 2601-2618, doi:10.1029/93JB02581.

`d94()` below is an independent implementation of Dieterich (1994)
eq. 12/13, re-derived directly from the paper's rate-and-state
constitutive framework rather than transcribed from any third-party
script: starting from the state-variable evolution law
dγ/dt = (1 - γ·τ̇)/(Aσ) together with the standard instantaneous
coseismic-stress-step jump condition γ(0+) = γ(0-)·exp(-ΔCFF/Aσ) and
R(t) = r0/(γ(t)·τ̇), solving that ODE and substituting reproduces
d94()'s R(t) and C(t) closed forms exactly (verified symbolically,
2026-08-26 -- see
PROJECT_HANDOVER_ADDENDUM_2026-08-26b_d94_independent_rederivation.md).
This is the same closed-form solution published, in equivalent
notation, across the rate-and-state seismicity literature (e.g. Toda
& Stein 2003; Toda, Stein & Sagiya 2002; Console & Catalli 2006;
King & Cocco 2001) -- it is Dieterich's own published equation, not a
creative expression original to any one script. This implementation's
numerical output has NOT been cross-checked against camcat/d94-mtmod
(Camilla Cattania, MTMOD summer-school tools,
https://github.com/camcat/d94-mtmod) or any other independent
implementation -- only the analytical/symbolic derivation above has
been verified. No source code from that or any other repository is
reproduced here. The "array Dcmb -> ensemble-average the resulting R(t) curves" behavior when `dcff` has more than one element
is this project's own API convention (a natural extension for
averaging a forecast over e.g. several receiver-plane orientations),
chosen for outer-call-site convenience -- not a requirement of eq. 12
itself, which is inherently a single-ΔCFF-value solution.

`dieterich_rate_grid()` and `forecast_from_cff_volume()` are this
project's own extension, built on top of that independently-derived
core: they apply the same closed-form solution INDEPENDENTLY at every
point of an arbitrary-shaped ΔCFF array (e.g. every point of a
core.cff_volume.CFFVolume) rather than averaging over the array. This
mirrors the general approach used by per-grid-point aftershock
forecasting tools in the literature (evaluate eq. 12 once per grid
cell rather than averaging), vectorized here (rather than looped)
since a CFFVolume can have tens of thousands of grid points x depth
slices. forecast_from_cff_volume() splits a single whole-region
background rate r0 equally across grid points, which is a common,
independently-motivated convention for going from a region-total
seismicity rate to a per-cell rate when no finer-grained r0 field is
available.

Units: no units are hard-coded. All of t, t0, ta must share one time
unit (Dieterich's own worked examples and camcat/d94-mtmod's defaults
use days); asig and every ΔCFF value must share one stress unit. This
project's own ΔCFF grids/volumes (core.okada_engine,
core.cff_volume.CFFVolume.cff_mpa) are in MPa throughout, so `asig`
should normally be given in MPa here, not the kPa camcat/d94-mtmod's
coulomb2forecast.m defaults to -- deliberate unit divergence, same
reasoning as core.aftershock_mc_test's MPa-not-bar threshold choice:
staying consistent with the rest of this plugin beats matching an
external script's default unit. UI layer is responsible for labeling
asig inputs as MPa.

RateStateParams.time_unit (added 2026-08-22, see
PROJECT_HANDOVER_ADDENDUM_2026-08-22b_time_unit_consistency.md) is
LABELING METADATA ONLY -- it does not change any arithmetic here.
d94()/dieterich_rate_grid()/forecast_from_cff_volume() remain exactly
as unit-agnostic as before; every t/t0/ta value is used as a bare
number, whatever unit it happens to be in. What time_unit buys is a
single authoritative place to RECORD which unit that number set is
supposed to be in, so a caller with several unit-sensitive inputs
(most notably core.rate_state_calibration, which converts a real
catalog's absolute timestamps into a relative time value) has one
place to read that unit from instead of maintaining an independently-
entered, silently-driftable copy of the same string. See
core.rate_state_calibration.assert_consistent_time_unit().

New physics module: kept entirely separate from cff_volume.py and
okada_engine.py per project convention (new functionality goes in new
files; existing validated modules are touched only when necessary).
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .cff_volume import CFFVolume, apply_near_field_mask


# ─── Rate-state parameters ───────────────────────────────────────────────

@dataclass
class RateStateParams:
    """Named fields for Dieterich (1994) eq. 12's four parameters,
    exposed positionally via as_rs_par() as `(r0, asig, ta[, tdotr])`
    for callers of d94(). See module docstring for required unit
    consistency (asig in the same stress unit as the ΔCFF field it's
    applied to; r0/ta/t/t0 all in one consistent time unit)."""

    r0: float                    # background seismicity rate [1/time]
    asig: float                  # a*sigma: rate-state direct-effect parameter x effective normal stress [stress]
    ta: float                    # aftershock decay time (Dieterich 1994) [time]
    tdotr: Optional[float] = None  # background (pre-mainshock) stressing rate [stress/time]; defaults to asig/ta if None (Dieterich's own default absent an independently-specified background rate)
    time_unit: Optional[str] = None  # e.g. "days" -- LABELING ONLY, see module docstring; None means "caller didn't record one" (legacy/script callers), not "seconds" or any other implied default

    def as_rs_par(self) -> Tuple[float, ...]:
        """Round-trips to the plain positional `(r0, asig, ta[, tdotr])`
        tuple d94() accepts. time_unit is deliberately NOT included --
        this tuple is purely numeric, matching d94()'s own signature."""
        if self.tdotr is None:
            return (self.r0, self.asig, self.ta)
        return (self.r0, self.asig, self.ta, self.tdotr)


# ─── Dieterich (1994) eq. 12/13 closed-form solution ─────────────────────

def d94(t, t0: float, rs_par: Sequence[float], dcff, verbose: bool = False):
    """
    [R, C] = d94(t, t0, rs_par, dcff, verbose)

    Closed-form solution of Dieterich (1994) eq. 12/13 (see module
    docstring for the independent derivation). Outputs seismicity
    rate R(t) and cumulative event count C(t) (integrated between t0
    and t) at times `t` following an instantaneous Coulomb stress step
    `dcff` applied at t=0.

    rs_par = [r0, asig, ta] or [r0, asig, ta, tdotr] -- either a
    RateStateParams.as_rs_par() or a plain 3/4-element sequence, both
    accepted.

    If `dcff` has more than one element, R and C are each the mean,
    over every element of `dcff`, of the R(t)/C(t) curve that element
    alone would produce at the same t/t0/rs_par (this project's own
    ensemble-averaging convention, see module docstring). This is NOT
    independent per-point evaluation -- see dieterich_rate_grid() for
    that.

    t0=0 gives C=inf for some choices of rs_par -- an inherent
    property of the closed form (the log-argument's limit), not
    special-cased here.
    """
    t = np.asarray(t, dtype=float)
    r0 = float(rs_par[0])
    asig = float(rs_par[1])
    ta = float(rs_par[2])
    tdot = asig / ta
    tdotr = float(rs_par[3]) if len(rs_par) > 3 else tdot

    dcff_arr = np.atleast_1d(np.asarray(dcff, dtype=float))
    if dcff_arr.size > 1:
        R = np.zeros_like(t)
        C = np.zeros_like(t)
        N = dcff_arr.size
        for dc in dcff_arr.ravel():
            r, c = d94(t, t0, rs_par, dc, verbose=False)
            R = R + r / N
            C = C + c / N
        return R, C

    dc = float(dcff_arr.reshape(-1)[0])

    # Eq. 12 in Dieterich (1994), written as: r = A / (B*exp(-t/ta) + 1)
    A = r0 * tdot / tdotr
    with np.errstate(over="ignore"):
        B = tdot / tdotr * np.exp(-dc / asig) - 1.0
        R = A / (B * np.exp(-t / ta) + 1.0)

        if np.isinf(B):
            C = np.zeros_like(t)
            if verbose:
                warnings.warn(
                    "Large negative Dcmb: setting C=0 (this is ok for "
                    "t<~Dcmb/tdot)"
                )
        else:
            C = A * ta * (
                np.log(np.exp(t / ta) + B) - np.log(np.exp(t0 / ta) + B)
            )

    return R, C


# ─── Independent (non-averaging) grid evaluation ─────────────────────────

def dieterich_rate_grid(t, t0: float, params: RateStateParams, dcff,
                         verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized, elementwise evaluation of the same closed-form solution
    as d94(), applied INDEPENDENTLY at every point of an arbitrary-
    shaped `dcff` array (e.g. one CFFVolume depth slice with shape
    (n_lat, n_lon), or the full 3D volume with shape
    (n_depth, n_lat, n_lon)) against a shared time vector `t`.

    This deliberately does NOT reproduce d94()'s own "array Dcmb ->
    average the resulting curves together" behavior above -- it
    evaluates eq. 12 independently at every grid point n, each keeping
    its own R(t)/C(t) curve, which is the appropriate behavior when
    `dcff` represents genuinely distinct spatial locations rather than
    an ensemble to be averaged over. See module docstring.

    Returns (R, C), each shaped dcff.shape + (len(t),).
    """
    t = np.asarray(t, dtype=float)
    dcff = np.asarray(dcff, dtype=float)
    r0, asig, ta = params.r0, params.asig, params.ta
    tdot = asig / ta
    tdotr = tdot if params.tdotr is None else params.tdotr

    dcff_b = dcff[..., np.newaxis]   # (..., 1) so it broadcasts against t's (nt,)
    A = r0 * tdot / tdotr
    with np.errstate(over="ignore", invalid="ignore"):
        B = tdot / tdotr * np.exp(-dcff_b / asig) - 1.0    # (..., 1)
        R = A / (B * np.exp(-t / ta) + 1.0)                 # (..., nt)
        C = A * ta * (np.log(np.exp(t / ta) + B) - np.log(np.exp(t0 / ta) + B))

    inf_mask = np.isinf(B)[..., 0]   # (...,)
    if np.any(inf_mask):
        C = np.where(inf_mask[..., np.newaxis], 0.0, C)
        if verbose:
            warnings.warn(
                f"{int(np.sum(inf_mask))} grid point(s) have large "
                "negative ΔCFF relative to asig: setting C=0 there "
                "(this is ok for t << |ΔCFF|/tdot at those points)"
            )
    return R, C


# ─── CFFVolume-driven forecast ────────────────────────────────────────────

@dataclass
class RateStateForecast:
    """Result of forecast_from_cff_volume(). Shapes mirror the source
    CFFVolume: lons (n_lon,), lats (n_lat,), depths_km (n_depth,),
    ts (n_t,); rate and cumulative are (n_depth, n_lat, n_lon, n_t)."""

    lons: np.ndarray
    lats: np.ndarray
    depths_km: np.ndarray
    ts: np.ndarray
    t0: float
    rate: np.ndarray          # seismicity rate, same time unit as params [1/time] per grid cell
    cumulative: np.ndarray    # cumulative expected no. of events per grid cell, integrated from t0
    params: RateStateParams   # the ORIGINAL params passed in (r0 not yet split across points)
    r0_per_point: float       # actual per-grid-point background rate used (params.r0 / n_points)

    def total_rate(self) -> np.ndarray:
        """nansum of per-cell rate over the whole volume at each time --
        the total forecast seismicity rate for the modeled region,
        shape (n_t,). Valid because r0 was split equally across grid
        points (see forecast_from_cff_volume docstring), so summing
        recovers a rate in the same [1/time] units as the input r0.
        nansum (not sum) so grid points outside the interpolated
        volume -- NaN, per core.cff_volume's fill_value=nan convention
        -- don't NaN out the whole-region total; see forecast_from_cff_
        volume's NaN-handling docstring note."""
        return np.nansum(self.rate, axis=(0, 1, 2))

    def total_cumulative(self) -> np.ndarray:
        """nansum of per-cell cumulative count over the whole volume at
        each time -- total expected no. of events since t0, shape
        (n_t,). See total_rate() for the nansum rationale."""
        return np.nansum(self.cumulative, axis=(0, 1, 2))

    def amplification(self) -> np.ndarray:
        """Region-total rate amplification R(t)/R0, shape (n_t,) -- added
        2026-08-22c per the external-review synthesis's session-3 item 1
        ("Rate amplification R/R0 -- not computed or plotted anywhere...
        cheap: forecast.total_rate() / params.r0 is a one-line derived
        series"). Deliberately divides by params.r0 (the ORIGINAL
        whole-region background rate the person entered), not
        r0_per_point -- this matches the same reference level total_rate()
        is already plotted against (plot_rate_state_forecast's
        mode="rate" axhline), so amplification()==1 exactly where that
        axhline sits. A value of e.g. 1841 means the whole-region
        forecast rate is 1841x background at that time -- the same kind
        of number session 3's own worked example needed an at-a-glance
        interpretation aid for."""
        return self.total_rate() / self.params.r0


def forecast_from_cff_volume(volume: CFFVolume, ts, t0: float,
                              params: RateStateParams,
                              verbose: bool = False,
                              exclude_near_field: bool = True) -> RateStateForecast:
    """
    Reads a ΔCFF field (a core.cff_volume.CFFVolume) and calculates a
    Dieterich (1994) seismicity forecast at every grid point.

    Treats `params.r0` as a UNIFORM total background rate for the
    whole modeled volume, split equally among grid points -- NOT as a
    per-point rate. Use RateStateForecast.total_rate()/
    .total_cumulative() to recover the whole-region forecast in the
    original r0 units, or index .rate/.cumulative directly for the
    per-grid-cell curves.

    `ts` : 1D sequence of forecast times, same time unit as
    params.ta/t0.
    `t0` : start time for the cumulative count (note t0=0 can give an
    infinite cumulative count for some rs_par choices -- an inherent
    property of the closed form, not specially handled here).

    exclude_near_field (default True, added 2026-08-22 smoke-test fix):
    near-field grid cells (volume.near_field_mask>0 -- known-unreliable
    Okada/DC3D singularities, see core.cff_volume.apply_near_field_mask's
    docstring) are set to NaN before evaluating dieterich_rate_grid(),
    so a handful of singular ΔCFF cells can no longer dominate
    total_rate()/total_cumulative() with an unphysical rate spike.
    r0 is still split across the FULL grid point count (n_points below
    is volume.cff_mpa.size, not the post-exclusion finite count) --
    same reasoning as the existing out-of-volume-NaN handling one
    paragraph down: r0 represents the whole modeled region's background
    rate regardless of how many of its cells turned out to be
    NaN/excluded.

    NaN handling: CFFVolume grid points outside the interpolated/
    computed region can be NaN (see core.cff_volume's
    RegularGridInterpolator fill_value=nan convention), and near-field
    cells become NaN too when exclude_near_field=True (see above) --
    both propagate through as NaN rate/cumulative at that point, and are
    excluded from total_rate()/total_cumulative() via nan-aware
    summation so one out-of-range or near-field corner doesn't NaN the
    whole-region total.
    """
    ts = np.asarray(ts, dtype=float)
    n_points = volume.cff_mpa.size
    if n_points == 0:
        raise ValueError("volume.cff_mpa is empty")

    cff_for_forecast = apply_near_field_mask(
        volume.cff_mpa, volume.near_field_mask, exclude_near_field)

    per_point_params = RateStateParams(
        r0=params.r0 / n_points, asig=params.asig, ta=params.ta,
        tdotr=params.tdotr,
    )
    rate, cumulative = dieterich_rate_grid(
        ts, t0, per_point_params, cff_for_forecast, verbose=verbose)

    return RateStateForecast(
        lons=volume.lons, lats=volume.lats, depths_km=volume.depths_km,
        ts=ts, t0=t0, rate=rate, cumulative=cumulative,
        params=params, r0_per_point=per_point_params.r0,
    )
