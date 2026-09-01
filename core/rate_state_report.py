# -*- coding: utf-8 -*-
"""
Post-run reporting for core.rate_state_seismicity.forecast_from_cff_volume():
a human-readable summary report and a CSV row builder for the
whole-region total_rate()/total_cumulative() time series.

Follows the same shape as core.aftershock_mc_report / core.slip_inversion_report:
pure stdlib, no Qt import (testable outside a real QGIS session). PDF
export reuses core.aftershock_mc_report.write_report_pdf() directly
rather than duplicating a second PdfPages writer -- that function takes
report text + an optional matplotlib Figure and has no
aftershock-specific content in it, so it's a general-purpose paginated
text(+figure) PDF writer that happens to live in that module for
historical reasons (it was built for the aftershock feature first).
"""

import math


def _finite(x):
    try:
        return math.isfinite(x)
    except TypeError:
        return False


def _fmt_depths(depths_km):
    if depths_km is None or len(depths_km) == 0:
        return "n/a"
    return ", ".join(f"{d:g}" for d in depths_km) + " km"


def _format_calibration_section(background, calibration, validation, time_unit, stress_unit):
    """
    Formats an optional calibration/validation section (see
    core.rate_state_calibration -- background rate from catalog,
    asig/ta fit against real aftershock counts, N-test/RMSE/R2/Poisson
    log-likelihood scoring). All three arguments are optional and
    independent of each other (e.g. `validation` alone, scoring a
    hand-entered parameter set against the catalog with no fit ever
    run) -- only the sections with data actually supplied are emitted.
    """
    lines = []
    if background is None and calibration is None and validation is None:
        return lines

    lines.append("")
    lines.append("Calibration / validation against real earthquake catalog data")
    lines.append("-" * 55)

    if background is not None:
        region_note = " (region-restricted)" if background.region_restricted else ""
        mag_note = f", M >= {background.mag_min:g}" if background.mag_min is not None else ""
        lines.append(
            f"  Background rate r0 from catalog{region_note}{mag_note}: "
            f"{background.n_events} event(s) / {background.duration:g} {time_unit} "
            f"= {background.r0:g} events/{time_unit}")

    if calibration is not None:
        p = calibration.params
        mode = "r0, asig, ta all fit" if calibration.fit_r0 else "asig, ta fit (r0 fixed)"
        window_note = (f", fit window t<={calibration.t_fit_max:g} "
                       f"({calibration.n_fit_points} of {len(calibration.observed.ts)} points)"
                       if calibration.t_fit_max is not None else
                       f" (full {calibration.n_fit_points}-point window)")
        lines.append(f"  Calibration ({mode}){window_note}:")
        lines.append(f"    Fitted r0:   {p.r0:g} events/{time_unit}"
                     + (" (fit)" if calibration.fit_r0 else " (fixed, from background rate)"))
        asig_stderr_note = (f"  (+/- {calibration.asig_stderr:.3g}, "
                            f"bounds [{calibration.asig_bounds[0]:.3g}, "
                            f"{calibration.asig_bounds[1]:.3g}])"
                            if calibration.asig_stderr is not None else "")
        lines.append(f"    Fitted asig: {p.asig:g} {stress_unit}{asig_stderr_note}")
        ta_stderr_note = (f"  (+/- {calibration.ta_stderr:.3g}, "
                          f"bounds [{calibration.ta_bounds[0]:.3g}, "
                          f"{calibration.ta_bounds[1]:.3g}])"
                          if calibration.ta_stderr is not None else "")
        lines.append(f"    Fitted ta:   {p.ta:g} {time_unit}{ta_stderr_note}")
        if calibration.ta_at_bound:
            lines.append(
                "    WARNING: ta landed at (or within 1% of) its fit ceiling -- "
                "this is a LOWER BOUND on the decay time, not a resolved value. "
                "The data's fit window is too short relative to the true decay "
                "to pin ta down; treat it as \"at least this long\", not literal.")
        elif not calibration.well_determined:
            lines.append(
                "    WARNING: ta's relative uncertainty is large -- this fit is "
                "poorly constrained by the available catalog data. Treat the "
                "fitted ta with caution (asig is typically much better "
                "determined than ta from sparse/short aftershock sequences).")
        lines.append(f"    Optimizer:   {'converged' if calibration.success else 'DID NOT CONVERGE'} "
                     f"({calibration.message})")
        obs = calibration.observed
        mag_note = f", M >= {obs.mag_min:g}" if obs.mag_min is not None else ""
        lines.append(f"    Observed catalog window: {obs.n_events_total} event(s){mag_note}, "
                     f"t in [{obs.t0:g}, {obs.ts[-1]:g}] {time_unit}")

    if validation is not None:
        v = validation
        lines.append(f"  Validation (t in [{v.t_range[0]:g}, {v.t_range[1]:g}] {time_unit}, "
                     f"{v.n_points_scored} point(s) scored):")
        lines.append(f"    Observed total count:  {v.n_obs:g}")
        lines.append(f"    Predicted total count: {v.n_pred:g}")
        lines.append(f"    N-ratio (obs/pred):    {v.n_ratio:g}")
        lines.append(f"    N-test p-value:        {v.n_test_p_value:g}  "
                     "(CSEP-style two-sided Poisson number test -- near 0 means the "
                     "model's total event budget is inconsistent with what was "
                     "actually observed; does NOT by itself test the shape of the decay)")
        lines.append(f"    RMSE (cumulative count): {v.rmse:g}"
                     + (f"  (normalized: {v.nrmse:g})" if _finite(v.nrmse) else ""))
        lines.append(f"    R^2 (cumulative count):  {v.r2:g}")
        lines.append(f"    Poisson log-likelihood (per-bin counts): {v.poisson_loglik:g}")

    lines.append("")
    return lines


def _format_cff_stats_section(stats, unit="MPa"):
    """
    Formats the optional ΔCFF field statistics block (core.cff_volume.
    CFFFieldStats) -- added 2026-08-22c per the external-review
    synthesis's session-3 items 2-3. Placed ahead of the run-setup
    section in build_rate_state_report() since it's most useful as the
    FIRST thing read, before the person has to interpret any forecast
    numbers -- "is this a real signal or a units/parameter mismatch?"
    """
    if stats is None:
        return []
    lines = []
    lines.append("ΔCFF field statistics (input field, before forecasting)")
    lines.append("-" * 55)
    excluded_note = ""
    n_other_excluded = stats.n_points - stats.n_finite - stats.n_near_field_excluded
    if stats.n_near_field_excluded and n_other_excluded > 0:
        excluded_note = (f"  ({stats.n_near_field_excluded} near-field excluded, "
                         f"{n_other_excluded} NaN/outside volume)")
    elif stats.n_near_field_excluded:
        excluded_note = f"  ({stats.n_near_field_excluded} near-field excluded)"
    elif n_other_excluded > 0:
        excluded_note = f"  ({n_other_excluded} NaN/outside volume)"
    lines.append(f"  Points:        {stats.n_finite} finite of {stats.n_points} total"
                 + excluded_note)
    lines.append(f"  min / max:     {stats.min:.4g} / {stats.max:.4g} {unit}")
    lines.append(f"  mean / median: {stats.mean:.4g} / {stats.median:.4g} {unit}")
    lines.append(f"  std:           {stats.std:.4g} {unit}")
    lines.append(f"  P5 / P95:      {stats.p5:.4g} / {stats.p95:.4g} {unit}")
    lines.append(f"  % positive / negative / zero: "
                 f"{100*stats.frac_positive:.1f}% / {100*stats.frac_negative:.1f}% / "
                 f"{100*stats.frac_zero:.1f}%")
    lines.append("")
    return lines


def build_rate_state_report(forecast, meta=None, background=None,
                            calibration=None, validation=None,
                            cff_stats=None) -> str:
    """
    Plain-text report for a RateStateForecast: run setup (parameters,
    grid, ΔCFF mode), then the whole-region total rate/cumulative-count
    time series (same numbers the CSV export carries), so the text
    report is self-contained without needing the CSV alongside it.

    meta : optional dict of run context the UI layer knows that this
        module doesn't (mirrors core.aftershock_mc_report's own `meta`
        convention). Recognized keys, all optional:
          "cff_mode_label"  (str)   -- e.g. "specified receiver fault" /
                                        "optimally-oriented plane"
          "exclude_near_field" (bool) -- whether near-field singular cells
                                        were masked out of the field/forecast/
                                        calibration for this run (added
                                        2026-08-24c; see the "Near-field
                                        exclusion" workflow-improvement
                                        addendum). Omitted key (older
                                        callers): line is skipped entirely,
                                        report unchanged from before this
                                        field existed.
          "depths_km"       (list of float) -- volume depth slices
          "grid_shape"      (tuple) -- (n_depth, n_lat, n_lon)
          "time_unit"       (str)   -- e.g. "days" (labeling only; the
                                        module itself is unit-agnostic)
          "stress_unit"     (str)   -- e.g. "MPa" (labeling only)
          "mc"              (float) -- magnitude of completeness, added
                                        2026-08-22c (external-review
                                        synthesis, session 3 item 4).
                                        LABEL ONLY -- this module has no
                                        magnitude dependence anywhere;
                                        when given, every "events"/
                                        "expected events" line below is
                                        worded "events with M>=mc"
                                        instead, so the forecast output
                                        isn't misread as an unconditional
                                        event count. None (default):
                                        wording stays "events", exactly
                                        as before this field existed.

    cff_stats : optional core.cff_volume.CFFFieldStats (typically
        core.cff_volume.cff_field_stats(volume) run against the SAME
        CFFVolume the forecast was built from) -- added 2026-08-22c, see
        _format_cff_stats_section(). None (default): section omitted,
        report unchanged from before this parameter existed.

    background, calibration, validation : optional
        core.rate_state_calibration.BackgroundRateResult /
        CalibrationResult / ValidationScore -- when any are given, an
        extra "Calibration / validation against real earthquake catalog
        data" section is appended (see _format_calibration_section()).
        All independent and optional, matching how the dialog's
        Calibration group lets each step (compute r0, fit asig/ta,
        score against the catalog) be run on its own.
    """
    meta = meta or {}
    params = forecast.params
    total_rate = forecast.total_rate()
    total_cum = forecast.total_cumulative()
    # Prefer the forecast's OWN recorded time_unit (RateStateParams.
    # time_unit, added 2026-08-22 -- see PROJECT_HANDOVER_ADDENDUM_
    # 2026-08-22b_time_unit_consistency.md) over meta's, since it's the
    # single authoritative source now; meta["time_unit"] remains as a
    # fallback for forecasts built before that field existed (params.
    # time_unit is None in that case) or by a script that never set it.
    time_unit = getattr(params, "time_unit", None) or meta.get("time_unit", "time units")
    stress_unit = meta.get("stress_unit", "stress units")
    mc = meta.get("mc")
    event_word = f"events with M>={mc:g}" if mc is not None else "events"

    lines = []
    lines.append("Rate-and-State Seismicity Forecast (Dieterich, 1994)")
    lines.append("=" * 55)
    lines.append("")
    lines.extend(_format_cff_stats_section(cff_stats, unit=stress_unit))
    lines.append("Run setup")
    lines.append("-" * 55)
    lines.append(f"  ΔCFF field:            {meta.get('cff_mode_label', 'n/a')}")
    if "exclude_near_field" in meta:
        excl = meta["exclude_near_field"]
        lines.append(
            f"  Near-field exclusion:  {'ON' if excl else 'OFF'}"
            + ("" if excl else
               " -- near-fault singular cells are INCLUDED in this field/"
               "forecast/calibration; min/max/std and any fitted parameters "
               "may be distorted by them. See the ΔCFF field statistics above."))
    lines.append(f"  Depth slices:          {_fmt_depths(meta.get('depths_km'))}")
    grid_shape = meta.get("grid_shape")
    n_points = None
    if grid_shape:
        n_points = 1
        for d in grid_shape:
            n_points *= d
        lines.append(f"  Grid shape (depth,lat,lon): {grid_shape} "
                     f"({n_points} points)")
    lines.append(f"  Background rate r0:    {params.r0:g} {event_word}/{time_unit} (total, region-wide)")
    if n_points:
        lines.append(f"                         = {forecast.r0_per_point:g} {event_word}/{time_unit} per grid point")
    lines.append(f"  a·sigma (asig):        {params.asig:g} {stress_unit}")
    lines.append(f"  Aftershock decay time ta: {params.ta:g} {time_unit}")
    if params.tdotr is not None:
        lines.append(f"  Background stressing rate tdotr: {params.tdotr:g} {stress_unit}/{time_unit} (explicit)")
    else:
        tdot = params.asig / params.ta
        lines.append(f"  Background stressing rate tdotr: {tdot:g} {stress_unit}/{time_unit} "
                     "(default = asig/ta)")
    lines.append(f"  Forecast start t0:     {forecast.t0:g} {time_unit}")
    lines.append(f"  Forecast times:        {forecast.ts[0]:g} to {forecast.ts[-1]:g} "
                 f"{time_unit} ({len(forecast.ts)} steps)")
    lines.append("")

    lines.append(
        "Model: Dieterich (1994) closed-form seismicity-rate response to an\n"
        "instantaneous Coulomb stress step at each grid point, with a\n"
        "uniform total background rate r0 split equally across all grid\n"
        "points (matching Dieterich's own coulomb2forecast.m reference\n"
        "implementation's convention). Rate spikes immediately after a\n"
        "positive ΔCFF and decays back toward the background rate over the\n"
        "aftershock decay time ta; a negative ΔCFF suppresses the rate\n"
        "below background and it recovers over the same timescale.")
    lines.append("")

    finite_mask = [_finite(r) for r in total_rate]
    if any(finite_mask):
        peak_idx = max((i for i in range(len(total_rate)) if finite_mask[i]),
                       key=lambda i: total_rate[i])
        lines.append(
            f"  Peak total forecast rate: {total_rate[peak_idx]:g} {event_word}/{time_unit} "
            f"at t={forecast.ts[peak_idx]:g} {time_unit}")
        if _finite(params.r0) and params.r0 != 0:
            amp = forecast.amplification()
            finite_amp = [a for a in amp if _finite(a)]
            if finite_amp:
                lines.append(f"  Peak rate amplification R/R0: {max(finite_amp):g}x background")
    if _finite(total_cum[-1]):
        lines.append(
            f"  Total expected {event_word} by t={forecast.ts[-1]:g} {time_unit} "
            f"(since t0={forecast.t0:g}): {total_cum[-1]:g}")
    if mc is not None:
        lines.append(
            f"  Note: this forecast has no magnitude dependence of its own -- "
            f"the M>={mc:g} labeling above is a display convention only, "
            "reflecting the completeness threshold of whatever catalog r0/asig/ta "
            "were calibrated against (if any). Change/clear it in the dialog if "
            "that assumption doesn't apply.")
    n_nan_points = None
    if n_points:
        import numpy as np
        n_nan_points = int(np.sum(np.isnan(forecast.rate[..., 0])))
        if n_nan_points:
            lines.append(
                f"  Note: {n_nan_points} of {n_points} grid point(s) fell outside the "
                "computed ΔCFF volume (NaN) and are excluded from the totals above.")
    lines.append("")

    if background is None and calibration is None and validation is None:
        lines.append(
            "  Note: this reproduces Dieterich (1994)'s analytic point-process\n"
            "  rate solution exactly, but the choice of r0/asig/ta is the\n"
            "  user's own -- these parameters are typically calibrated against\n"
            "  an observed aftershock sequence (e.g. via the aftershock decay\n"
            "  rate and background seismicity rate before the mainshock), not\n"
            "  derived from the ΔCFF field itself. Treat the forecast as\n"
            "  conditional on that calibration.")
        lines.append("")
    else:
        lines.extend(_format_calibration_section(
            background, calibration, validation, time_unit, stress_unit))

    lines.append(f"Whole-region forecast time series ({len(forecast.ts)} steps)")
    lines.append("-" * 55)
    lines.append(f"{'t':>12} {'rate':>14} {'cumulative':>14}")
    for i, t in enumerate(forecast.ts):
        r = total_rate[i]
        c = total_cum[i]
        r_str = f"{r:.6g}" if _finite(r) else "nan"
        c_str = f"{c:.6g}" if _finite(c) else "nan"
        lines.append(f"{t:>12.6g} {r_str:>14} {c_str:>14}")

    return "\n".join(lines) + "\n"


def build_rate_state_csv_rows(forecast):
    """
    One row per forecast time step: the whole-region total_rate()/
    total_cumulative() series. Per-grid-cell curves are NOT included
    here (n_depth × n_lat × n_lon × n_t would typically be far too
    large for a flat CSV, and most users want the whole-region total
    for a first look) -- see forecast.rate/forecast.cumulative directly
    for the full per-cell arrays if a spatial breakdown is needed.
    """
    total_rate = forecast.total_rate()
    total_cum = forecast.total_cumulative()
    rows = []
    for i, t in enumerate(forecast.ts):
        rows.append({
            "t": t,
            "rate": total_rate[i],
            "cumulative": total_cum[i],
        })
    return rows
