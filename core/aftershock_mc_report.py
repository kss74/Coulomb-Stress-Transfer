# -*- coding: utf-8 -*-
"""
Post-run reporting for core.aftershock_mc_test.observed_vs_null(): a
human-readable summary report (with a plain-language interpretation
section, not just a numbers dump), a per-threshold CSV row builder so
users can re-plot the null-test curves themselves, and a matplotlib-
based PDF writer (report text + the on-screen plot, paginated).

Follows the same shape as core.slip_inversion_report: pure stdlib for
the text/CSV pieces (no Qt import, so it's testable outside a real
QGIS session), matplotlib only for the optional PDF path -- matplotlib
is already a hard dependency of this plugin (plot_widget.py,
beachball.py), so pulling it in here for PDF pagination adds no new
dependency.

Report interpretation (`_find_significant_bands` / `build_aftershock_mc_report`)
is a descriptive check in the spirit of coulomb.m's own null-test plots,
NOT a formal hypothesis test / p-value -- it reports where (over which
threshold sub-range) and by how much the observed curve sits outside
the Monte Carlo 5-95th percentile band, which is the same thing a
person would read off the plot by eye. Framed as such explicitly in
the report text so it isn't mistaken for more than it is.

2026-08-19 revision: the original version of this module summarized
significance as "fraction of ALL swept thresholds where observed
exceeds the null band". That statistic is misleading whenever the
threshold sweep (`x_max`, set by the user) extends well past the range
where the \u0394CFF field actually has any values -- which is common,
since \u0394CFF magnitudes are rarely known in advance of a run. Every
threshold beyond the field's actual range has obs=null=0 on both
sides, which is correctly never "significant" (0 is not > 0) but still
dilutes the percentage's denominator, making a strong, real, contiguous
excess look like weak/noisy evidence. Replaced with: (a) the
enrichment ratio E(T) = obs/null_mean at each threshold (added to both
the printed table and the CSV), and (b) the contiguous threshold
range(s) where observed continuously exceeds the null band, together
with the peak enrichment ratio in each range -- not diluted by an
arbitrary, user-chosen upper sweep bound. See
PROJECT_HANDOVER_ADDENDUM_2026-08-19_aftershock_mc_report_enrichment.md.
"""

import math
from datetime import datetime


# ─── Plain-text report ───────────────────────────────────────────────────

def _fmt_depths(depths_km):
    if depths_km is None:
        return "n/a"
    return ", ".join(f"{d:g}" for d in depths_km) + " km"


def _finite(x):
    try:
        return math.isfinite(x)
    except TypeError:
        return False


def _enrichment_ratio(obs_val, null_mean):
    """
    E(T) = obs / null_mean. When null_mean == 0 (no random point in any
    MC run ever reached this threshold) but obs > 0, the ratio is
    formally infinite -- real events reached a threshold pure-chance
    placement never did in N x M draws, which IS the strongest possible
    enrichment signal, so this returns float('inf') rather than NaN or
    0 (both of which would silently hide a genuine result). When both
    are 0 there is no information either way -- returns NaN (excluded
    from peak-finding, printed as "n/a").
    """
    if not (_finite(obs_val) and _finite(null_mean)):
        return float("nan")
    if null_mean > 0:
        return obs_val / null_mean
    return float("inf") if obs_val > 0 else float("nan")


def _find_significant_bands(thr, obs_vals, null_p_bound, direction, obs_ref, null_mean, start_idx):
    """
    Contiguous runs of thresholds where the observed curve sits outside
    the null percentile band, each summarized as
    (thr_start, thr_end, peak_thr, peak_ratio).

    direction : "above" (GE side: obs_vals > null_p_bound, i.e. obs
        exceeds the null's 95th percentile) or "below" (LE side:
        obs_vals < null_p_bound, i.e. obs sits below the null's 5th
        percentile). obs_ref/null_mean feed the enrichment ratio used
        to pick each run's peak; for the "below" side this is a
        depletion ratio (< 1 means suppressed relative to random).
    """
    n = len(thr)
    runs = []
    i = start_idx
    while i < n:
        ok = (_finite(obs_vals[i]) and _finite(null_p_bound[i]) and
              (obs_vals[i] > null_p_bound[i] if direction == "above"
               else obs_vals[i] < null_p_bound[i]))
        if ok:
            j = i
            while j + 1 < n:
                ok_next = (_finite(obs_vals[j + 1]) and _finite(null_p_bound[j + 1]) and
                          (obs_vals[j + 1] > null_p_bound[j + 1] if direction == "above"
                           else obs_vals[j + 1] < null_p_bound[j + 1]))
                if not ok_next:
                    break
                j += 1
            # peak = most extreme enrichment/depletion ratio within [i, j]
            ratios = [(_enrichment_ratio(obs_ref[k], null_mean[k]), k) for k in range(i, j + 1)]
            finite_ratios = [(r, k) for r, k in ratios if _finite(r)]
            if finite_ratios:
                if direction == "above":
                    peak_ratio, peak_k = max(finite_ratios, key=lambda rk: rk[0])
                else:
                    peak_ratio, peak_k = min(finite_ratios, key=lambda rk: rk[0])
            else:
                # every ratio in the run was +inf (null_mean==0 throughout,
                # obs>0 throughout) -- still a real run, just report +inf
                peak_ratio, peak_k = float("inf"), i
            runs.append((thr[i], thr[j], thr[peak_k], peak_ratio))
            i = j + 1
        else:
            i += 1
    return runs


def _data_extent_idx(thr, obs_ge, null_ge_mean, obs_le, null_le_mean):
    """
    Highest threshold index at which there is still ANY nonzero signal
    on either side. Beyond this, a sweep that was configured with a
    generous x_max is just flat zeros -- worth flagging so the user can
    tighten x_max on the next run for a less diluted-looking plot/report.
    """
    last_i = 0
    for i in range(len(thr)):
        if ((_finite(obs_ge[i]) and obs_ge[i] > 0) or
            (_finite(null_ge_mean[i]) and null_ge_mean[i] > 0) or
            (_finite(obs_le[i]) and obs_le[i] > 0) or
            (_finite(null_le_mean[i]) and null_le_mean[i] > 0)):
            last_i = i
    return last_i


def build_aftershock_mc_report(result, meta=None):
    """
    Plain-text report: run setup, headline results, a plain-language
    interpretation paragraph, then the full per-threshold table (same
    data the CSV export carries, included here too so the text report
    is self-contained without needing the CSV alongside it).

    meta : optional dict of run context assembled by the caller (the
        UI layer knows things this module doesn't, e.g. how many
        catalog events were imported in total vs. how many landed
        inside the CFF volume). All keys optional; recognized keys:
          "n_catalog_events"   (int)   -- total events imported
          "cff_mode_label"     (str)   -- e.g. "specified receiver
                                           fault" / "optimally-oriented
                                           plane"
          "depths_km"          (list of float) -- volume depth slices
          "x_max"              (float) -- MPa, threshold sweep max
    """
    meta = meta or {}
    obs, null = result.observed, result.null
    thr = null.thr_vec
    n = len(thr)
    # thr==0 excluded from significance search: GE(0) is trivially ~1
    # for both curves by construction and carries no information.
    start = 1 if n > 1 and thr[0] <= 0 else 0

    ge_bands = _find_significant_bands(
        thr, obs.frac_ge, null.frac_ge_p95, "above", obs.frac_ge, null.frac_ge_mean, start)
    le_bands = _find_significant_bands(
        thr, obs.frac_le, null.frac_le_p05, "below", obs.frac_le, null.frac_le_mean, start)
    last_data_i = _data_extent_idx(thr, obs.frac_ge, null.frac_ge_mean, obs.frac_le, null.frac_le_mean)

    lines = []
    lines.append("Aftershock / \u0394CFF Monte Carlo Null Test -- Summary Report")
    lines.append("=" * 68)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("Run setup")
    lines.append("-" * 68)
    if "n_catalog_events" in meta:
        lines.append(f"  Catalog events imported     : {meta['n_catalog_events']}")
    lines.append(f"  Events used (inside volume)  : {obs.n_valid}")
    if "cff_mode_label" in meta:
        lines.append(f"  \u0394CFF resolved on              : {meta['cff_mode_label']}")
    if "depths_km" in meta:
        lines.append(f"  Depth slices                 : {_fmt_depths(meta['depths_km'])}")
    if "x_max" in meta:
        lines.append(f"  Threshold sweep              : 0 to {meta['x_max']:g} MPa "
                     f"({len(null.thr_vec)} steps)")
    else:
        lines.append(f"  Threshold sweep              : 0 to {null.thr_vec[-1]:g} MPa "
                     f"({len(null.thr_vec)} steps)")
    lines.append(f"  Monte Carlo null              : N={null.n_points} random points "
                 f"x M={null.n_runs} runs")
    depth_note = ""
    if null.depth_mode_used != null.depth_mode_requested:
        depth_note = (f" (requested '{null.depth_mode_requested}' -- fell back, "
                      f"no usable observed depths)")
    lines.append(f"  Null depth sampling           : {null.depth_mode_used}{depth_note}")
    lines.append("")

    lines.append("Interpretation")
    lines.append("-" * 68)
    lines.append(
        "This test asks whether real aftershocks preferentially occur where\n"
        "the mainshock increased Coulomb failure stress (\u0394CFF), compared to\n"
        "where they would fall if scattered at random within the same volume.\n"
        "The Monte Carlo null test repeatedly places random points in the\n"
        "volume and tracks what fraction would exceed each threshold purely\n"
        "by chance; its 5th-95th percentile spread across runs is the grey\n"
        "band on the plot. Enrichment ratio E(T) = observed / null-mean at\n"
        "a threshold -- e.g. E=4 means 4x as many real events cleared that\n"
        "threshold as random placement predicts.")
    lines.append("")

    if ge_bands:
        lines.append("  Promoted (\u0394CFF >= +thr) -- observed above the null 95th percentile:")
        for t0, t1, tpeak, ratio in ge_bands:
            span = f"{t0:.4f}-{t1:.4f} MPa" if t1 > t0 else f"{t0:.4f} MPa"
            ratio_str = "inf (never reached by chance)" if math.isinf(ratio) else f"{ratio:.2f}x"
            lines.append(
                f"    * {span}: peak enrichment {ratio_str} at thr={tpeak:.4f} MPa")
    else:
        lines.append("  Promoted (\u0394CFF >= +thr): no threshold range where observed "
                     "exceeded the null 95th percentile.")
    lines.append("")

    if le_bands:
        lines.append("  Inhibited (\u0394CFF <= -thr) -- observed below the null 5th percentile:")
        for t0, t1, tpeak, ratio in le_bands:
            span = f"{t0:.4f}-{t1:.4f} MPa" if t1 > t0 else f"{t0:.4f} MPa"
            ratio_str = "0 (never occupied)" if ratio == 0 else f"{ratio:.2f}x"
            lines.append(
                f"    * {span}: strongest depletion {ratio_str} at thr={tpeak:.4f} MPa")
    else:
        lines.append("  Inhibited (\u0394CFF <= -thr): no threshold range where observed "
                     "sat below the null 5th percentile.")
    lines.append("")

    if ge_bands or le_bands:
        lines.append(
            "  A contiguous range with enrichment well above 1x (promoted side)\n"
            "  or well below 1x (inhibited side) is the signature of Coulomb\n"
            "  stress triggering; an isolated single-threshold hit is within\n"
            "  the null test's own ~5-10% false-positive rate and should not\n"
            "  be over-interpreted on its own.")
    lines.append(
        "  Note: this is a descriptive check -- the same comparison a\n"
        "  person would read off the plot by eye -- not a formal\n"
        "  hypothesis test or p-value. Thresholds are nested (\u0394CFF>=T1\n"
        "  contains \u0394CFF>=T2 for T1<T2), so treat the range/peak above as\n"
        "  one piece of evidence, not N independent tests.")

    if last_data_i < n - 1:
        thr_data_max = thr[last_data_i]
        thr_swept_max = thr[-1]
        if thr_data_max < 0.5 * thr_swept_max:
            lines.append("")
            lines.append(
                f"  Note: the \u0394CFF field only reaches values up to about\n"
                f"  {thr_data_max:.4f} MPa in this run, well under the swept 'Max\n"
                f"  threshold' of {thr_swept_max:.4f} MPa -- consider lowering it on\n"
                f"  the next run for a less flat-lined plot and a finer-resolution\n"
                f"  look at the range that actually matters.")

    if obs.n_valid < 30:
        lines.append("")
        lines.append(
            f"  Caution: only {obs.n_valid} observed event(s) landed inside the\n"
            f"  CFF volume. The interpretation above is based on a small\n"
            f"  sample and should be treated as indicative, not conclusive.")
    lines.append("")

    lines.append("Per-threshold results (fraction of points; E = enrichment ratio)")
    lines.append("-" * 68)
    lines.append(f"{'thr(MPa)':>9} {'obs_GE':>8} {'null_GE_mean':>13} "
                 f"{'null_GE_5-95%':>17} {'E_GE':>7} {'obs_LE':>8} {'null_LE_mean':>13} "
                 f"{'null_LE_5-95%':>17} {'E_LE':>7}")
    for i, t in enumerate(null.thr_vec):
        ge_band = f"[{null.frac_ge_p05[i]:.4f},{null.frac_ge_p95[i]:.4f}]"
        le_band = f"[{null.frac_le_p05[i]:.4f},{null.frac_le_p95[i]:.4f}]"
        e_ge = _enrichment_ratio(obs.frac_ge[i], null.frac_ge_mean[i])
        e_le = _enrichment_ratio(obs.frac_le[i], null.frac_le_mean[i])
        e_ge_str = "inf" if math.isinf(e_ge) else ("n/a" if not _finite(e_ge) else f"{e_ge:.2f}")
        e_le_str = "inf" if math.isinf(e_le) else ("n/a" if not _finite(e_le) else f"{e_le:.2f}")
        lines.append(
            f"{t:>9.4f} {obs.frac_ge[i]:>8.4f} {null.frac_ge_mean[i]:>13.4f} "
            f"{ge_band:>17} {e_ge_str:>7} {obs.frac_le[i]:>8.4f} {null.frac_le_mean[i]:>13.4f} "
            f"{le_band:>17} {e_le_str:>7}")

    return "\n".join(lines) + "\n"


# ─── CSV rows (for re-plotting) ──────────────────────────────────────────

def build_aftershock_mc_csv_rows(result):
    """
    One row per swept threshold, all fraction- and count-mode series
    included so a single CSV covers both display modes the dialog
    offers, plus the enrichment ratio E(T) = obs/null_mean on each side
    (inf when null_mean==0 and obs>0; blank/NaN when both are 0 -- see
    _enrichment_ratio docstring). Column names are self-explanatory for
    someone re-plotting in Excel/pandas without this codebase.
    """
    obs, null = result.observed, result.null
    rows = []
    for i, t in enumerate(null.thr_vec):
        rows.append({
            "threshold_mpa": t,
            "obs_frac_ge": obs.frac_ge[i],
            "obs_frac_le": obs.frac_le[i],
            "obs_cnt_ge": obs.cnt_ge[i],
            "obs_cnt_le": obs.cnt_le[i],
            "null_frac_ge_mean": null.frac_ge_mean[i],
            "null_frac_ge_p05": null.frac_ge_p05[i],
            "null_frac_ge_p95": null.frac_ge_p95[i],
            "null_frac_le_mean": null.frac_le_mean[i],
            "null_frac_le_p05": null.frac_le_p05[i],
            "null_frac_le_p95": null.frac_le_p95[i],
            "null_cnt_ge_mean": null.cnt_ge_mean[i],
            "null_cnt_ge_p05": null.cnt_ge_p05[i],
            "null_cnt_ge_p95": null.cnt_ge_p95[i],
            "null_cnt_le_mean": null.cnt_le_mean[i],
            "null_cnt_le_p05": null.cnt_le_p05[i],
            "null_cnt_le_p95": null.cnt_le_p95[i],
            "enrichment_ge": _enrichment_ratio(obs.frac_ge[i], null.frac_ge_mean[i]),
            "enrichment_le": _enrichment_ratio(obs.frac_le[i], null.frac_le_mean[i]),
        })
    return rows


# ─── PDF export (report text [+ plot] as a paginated PDF) ───────────────

def _paginate_text(text, lines_per_page):
    lines = text.split("\n")
    return ["\n".join(lines[i:i + lines_per_page])
            for i in range(0, len(lines), lines_per_page)] or [""]


def write_report_pdf(path, report_text, plot_figure=None,
                     page_size_in=(8.5, 11), fontsize=7.0, margin_in=0.6):
    """
    Write `report_text` as a paginated PDF, optionally preceded by a
    full copy of `plot_figure` (the caller's already-drawn
    matplotlib Figure, e.g. PlotWidget.figure) as its own first page.
    Uses matplotlib.backends.backend_pdf.PdfPages rather than pulling
    in a new PDF-text dependency (reportlab/fpdf) -- matplotlib is
    already a hard dependency of this plugin.

    Monospace font so the report's aligned table columns stay aligned.
    """
    import matplotlib
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    page_w, page_h = page_size_in
    usable_h = page_h - 2 * margin_in
    # Rough line height in inches for a monospace font at `fontsize` pt,
    # with a little leading -- good enough for pagination since a few
    # lines of slop just means the last page is a bit under-full.
    line_height_in = (fontsize * 1.25) / 72.0
    lines_per_page = max(10, int(usable_h / line_height_in))

    with PdfPages(path) as pdf:
        if plot_figure is not None:
            pdf.savefig(plot_figure)

        for page_text in _paginate_text(report_text, lines_per_page):
            fig = Figure(figsize=(page_w, page_h))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            ax.text(margin_in / page_w, 1 - margin_in / page_h, page_text,
                    transform=ax.transAxes, family="monospace", fontsize=fontsize,
                    va="top", ha="left", wrap=False)
            pdf.savefig(fig)
