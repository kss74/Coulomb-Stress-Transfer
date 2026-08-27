# -*- coding: utf-8 -*-
"""Embedded matplotlib preview widget for Coulomb Stress Change maps."""

import numpy as np
from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from ..core.beachball import draw_beachball_batch


class PlotWidget(QWidget):
    """
    Embedded matplotlib canvas showing a CFF colour map.

    Uses a FIXED pair of axes (main plot + colorbar) created once in
    __init__ and reused on every redraw. Letting matplotlib auto-create
    a new colorbar axes on every plot_cff()/plot_displacement() call
    (the previous approach) causes the main axes to shrink a little more
    each time under tight_layout, since the figure never fully reclaims
    the space taken by earlier colorbar axes — after a few computations
    the plot appears to "zoom out" and shrink into a corner.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        # Floor so an embedding dialog with lots of stacked controls
        # above/beside the plot can't squeeze it down to an unreadable
        # sliver -- the plot degrades gracefully to "needs a resize"
        # rather than "title and legend overlapping" (2026-08-17 fix;
        # see aftershock_mc_dialog.py layout notes for the actual root
        # cause, this is just a backstop).
        self.canvas.setMinimumHeight(280)
        # Width floor added 2026-08-23 alongside the height floor above,
        # same backstop reasoning: this widget's axes are FIXED-fraction
        # (self.ax/self.cax below), not laid out via tight_layout/
        # constrained_layout -- there is no "automated layout" anywhere
        # in this class to mis-fire. What CAN and did happen (see
        # PROJECT_HANDOVER_ADDENDUM_2026-08-23b_rate_state_plot_layout_
        # and_near_field_toggle.md) is the canvas itself getting
        # squeezed narrow by the dialog's own layout (this widget had no
        # width floor, only a height one), which shrinks the fixed-
        # fraction axes down to a few dozen pixels wide -- at that point
        # matplotlib's default tick locator still tries to place several
        # ticks and their labels collide/overlap (the reported "0" and
        # "500" rendering on top of each other), independent of ANY data
        # or masking issue. 320px chosen as a reasonable floor for a
        # 2-line x-axis label plus a handful of tick labels at the
        # default 8pt font used throughout this widget.
        self.canvas.setMinimumWidth(320)
        layout.addWidget(self.canvas)

        # Fixed axes: [left, bottom, width, height] in figure-fraction coords.
        # Created once; reused (cleared, not recreated) on every redraw.
        self.ax = self.figure.add_axes([0.10, 0.10, 0.72, 0.80])
        self.cax = self.figure.add_axes([0.85, 0.10, 0.04, 0.80])
        self._twin_ax = None   # tracks a self.ax.twinx() created on demand
                                # (currently only plot_eq_catalog_timeline()'s
                                # magnitude overlay) -- see _reset_axes()
        self._stereonet_ax = None  # tracks a mplstereonet 'stereonet'-
                                # projection axes created on demand by
                                # plot_stress_inversion_stereonet() -- a
                                # SEPARATE axes object occupying self.ax's
                                # own [left,bottom,width,height] slot
                                # (self.ax itself is hidden, not reused,
                                # since a polar/stereonet projection can't
                                # be applied in-place to an existing
                                # rectilinear Axes) -- see _reset_axes()

        self._draw_placeholder()

    def _reset_axes(self):
        self.ax.clear()
        self.ax.set_visible(True)
        self.cax.clear()
        self.cax.set_visible(True)
        # A twin axes created by a PREVIOUS plot_eq_catalog_timeline()
        # call (self.ax.twinx()) is a SEPARATE axes object -- self.ax.
        # clear() above does not remove it. Left alone, every redraw
        # would stack another twin axes on top of the last one (same
        # "figure never fully reclaims the space" failure class this
        # class's own docstring already documents for colorbar axes,
        # here for a twin y-axis instead). Remove it explicitly so each
        # redraw starts from exactly zero or one twin axes, never more.
        if self._twin_ax is not None:
            self._twin_ax.remove()
            self._twin_ax = None
        # Same "remove, don't just clear" reasoning for a stereonet axes
        # left over from a PREVIOUS plot_stress_inversion_stereonet()
        # call -- it is a wholly separate Axes object sitting on top of
        # (not sharing) self.ax, so self.ax.clear() above does not touch
        # it either.
        if self._stereonet_ax is not None:
            self._stereonet_ax.remove()
            self._stereonet_ax = None

    def _draw_placeholder(self):
        self._reset_axes()
        self.cax.set_visible(False)
        self.ax.text(0.5, 0.5, "Run a computation to preview\nthe Coulomb stress map",
                     ha="center", va="center", transform=self.ax.transAxes,
                     fontsize=10, color="gray")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.canvas.draw()

    def plot_cff(self, lon2d, lat2d, cff, depth_label=None, near_field_mask=None):
        """
        Plot ΔCFF (MPa) as a diverging colour map.

        `near_field_mask`, if given, is the int-coded array from
        okada_engine.near_field_grid_mask (0=clear, 1=magnitude caution,
        2=sign untrustworthy -- see that function's docstring). Drawn as
        a hatch overlay so the underlying colour data stays visible:
        sparse dots for the magnitude-caution band, dense cross-hatch for
        the sign-untrustworthy zone closest to a fault edge.
        """
        self._reset_axes()

        vmax = np.nanpercentile(np.abs(cff), 98)
        vmax = max(vmax, 1e-6)
        im = self.ax.pcolormesh(lon2d, lat2d, cff, cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax, shading="auto")
        self.figure.colorbar(im, cax=self.cax, label="ΔCFF (MPa)")

        if near_field_mask is not None and np.any(near_field_mask > 0):
            self.ax.contourf(lon2d, lat2d, near_field_mask,
                             levels=[0.5, 1.5, 2.5], colors="none",
                             hatches=["..", "xxx"])
            self.ax.text(0.01, -0.11,
                         "hatched: near-field (Okada/DC3D singularity) — "
                         "· magnitude uncertain, × sign untrustworthy",
                         transform=self.ax.transAxes, fontsize=6.5, color="dimgray")

        title = "Coulomb Stress Change"
        if depth_label:
            title += f"  [{depth_label}]"
        self.ax.set_title(title, fontsize=10, fontweight="bold")
        self.ax.set_xlabel("Longitude", fontsize=8)
        self.ax.set_ylabel("Latitude", fontsize=8)
        self.ax.set_aspect("equal")
        self.canvas.draw()

    def plot_displacement(self, lon2d, lat2d, ux, uy, uz, depth_label=None):
        """Plot vertical displacement with horizontal displacement arrows."""
        self._reset_axes()

        vmax = np.nanpercentile(np.abs(uz), 98)
        vmax = max(vmax, 1e-6)
        im = self.ax.pcolormesh(lon2d, lat2d, uz, cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax, shading="auto")
        self.figure.colorbar(im, cax=self.cax, label="Vertical displacement (m)")

        step = max(1, lon2d.shape[0] // 15)
        self.ax.quiver(lon2d[::step, ::step], lat2d[::step, ::step],
                       ux[::step, ::step], uy[::step, ::step],
                       color="k", scale_units="xy", angles="xy", alpha=0.6)

        title = "Surface Deformation"
        if depth_label:
            title += f"  [{depth_label}]"
        self.ax.set_title(title, fontsize=10, fontweight="bold")
        self.ax.set_xlabel("Longitude", fontsize=8)
        self.ax.set_ylabel("Latitude", fontsize=8)
        self.ax.set_aspect("equal")
        self.canvas.draw()

    def plot_optimal_cff(self, lon2d, lat2d, cff_opt_mpa, strike1, strike2,
                         cff1_mpa, cff2_mpa, depth_label=None):
        """
        Plot Coulomb stress CHANGE on optimally-oriented planes: the
        color map shows cff_opt_mpa (= elementwise max of the two
        conjugate planes' own CFF change -- see
        `optimal_plane.optimal_plane_solution()`'s docstring, fixed
        2026-08-11 so this is the coseismic CHANGE, not the total
        post-earthquake stress). Short tick marks show the STRIKE of
        whichever of the two conjugate planes actually attains that max
        at each subsampled point -- since the fix, the two planes
        generally have DIFFERENT CFF, so which one "wins" varies across
        the map and is worth showing, not just assumed.
        """
        self._reset_axes()

        vmax = np.nanpercentile(np.abs(cff_opt_mpa), 98)
        vmax = max(vmax, 1e-6)
        im = self.ax.pcolormesh(lon2d, lat2d, cff_opt_mpa, cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax, shading="auto")
        self.figure.colorbar(im, cax=self.cax, label="ΔCFF on optimal plane (MPa)")

        # Strike ticks for the winning plane, subsampled (same spacing
        # convention as plot_displacement's quiver step).
        step = max(1, lon2d.shape[0] // 15)
        lon_ext = lon2d.max() - lon2d.min()
        lat_ext = lat2d.max() - lat2d.min()
        tick_len = 0.02 * max(lon_ext, lat_ext, 1e-6)

        plane1_wins = cff1_mpa >= cff2_mpa
        winning_strike = np.where(plane1_wins, strike1, strike2)

        for i in range(0, lon2d.shape[0], step):
            for j in range(0, lon2d.shape[1], step):
                strike_rad = np.radians(winning_strike[i, j])
                dx = tick_len * np.sin(strike_rad)
                dy = tick_len * np.cos(strike_rad)
                self.ax.plot(
                    [lon2d[i, j] - dx, lon2d[i, j] + dx],
                    [lat2d[i, j] - dy, lat2d[i, j] + dy],
                    color="k", linewidth=0.8, alpha=0.7)

        title = "Coulomb Stress Change on Optimally-Oriented Planes"
        if depth_label:
            title += f"  [{depth_label}]"
        self.ax.set_title(title, fontsize=10, fontweight="bold")
        self.ax.set_xlabel("Longitude", fontsize=8)
        self.ax.set_ylabel("Latitude", fontsize=8)
        self.ax.set_aspect("equal")
        self.canvas.draw()

    def plot_cross_section(self, dist_km, depth_km, cff_2d, fault_traces=None,
                           title="Cross-Section"):
        """
        Plot a vertical cross-section of ΔCFF (MPa) below a profile line.

        dist_km  : 1D array, distance along profile (km)
        depth_km : 1D array, depth (km, positive down)
        cff_2d   : 2D array, shape (len(depth_km), len(dist_km))
        fault_traces : optional list of (dist_km, depth_km) polylines to
                       overlay (e.g. fault plane intersections with the profile)
        """
        self._reset_axes()

        vmax = np.nanpercentile(np.abs(cff_2d), 98)
        vmax = max(vmax, 1e-6)
        im = self.ax.pcolormesh(dist_km, depth_km, cff_2d, cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax, shading="auto")
        self.figure.colorbar(im, cax=self.cax, label="ΔCFF (MPa)")

        if fault_traces:
            for trace_x, trace_z in fault_traces:
                self.ax.plot(trace_x, trace_z, color="k", linewidth=1.5)

        self.ax.invert_yaxis()
        self.ax.set_title(title, fontsize=10, fontweight="bold")
        self.ax.set_xlabel("Distance along profile (km)", fontsize=8)
        self.ax.set_ylabel("Depth (km)", fontsize=8)
        self.ax.set_aspect("equal")
        self.canvas.draw()

    def plot_focal_mechanisms(self, results, diameter_deg=None):
        """
        Plot ΔCFF-colored beachballs (core.beachball) for a batch of
        focal-mechanism results (core.focal_mechanism.compute_focal_mechanism_cff()
        output) at their event lon/lat locations. Standalone plot, like
        every other plot_*() method here -- clears and redraws self.ax/
        self.cax rather than overlaying on whatever was there before,
        for consistency (and to avoid the exact "colorbar axes pile up
        and the plot shrinks" failure mode this class's docstring
        already warns about, which an ad-hoc overlay-without-clearing
        design would have reintroduced).
        """
        self._reset_axes()

        if not results:
            self.cax.set_visible(False)
            self.ax.text(0.5, 0.5, "No focal mechanism results to show",
                         ha="center", va="center", transform=self.ax.transAxes,
                         fontsize=10, color="gray")
            self.ax.set_xticks([]); self.ax.set_yticks([])
            self.canvas.draw()
            return

        lons = [r["event"].lon for r in results]
        lats = [r["event"].lat for r in results]
        lon_span = max(lons) - min(lons) if len(lons) > 1 else 1.0
        lat_span = max(lats) - min(lats) if len(lats) > 1 else 1.0
        pad = max(lon_span, lat_span, 0.1) * 0.15
        self.ax.set_xlim(min(lons) - pad, max(lons) + pad)
        self.ax.set_ylim(min(lats) - pad, max(lats) + pad)
        self.ax.set_aspect("equal")

        if diameter_deg is None:
            extent = max(min(lon_span, lat_span), 1e-4)
            diameter_deg = max(extent * 0.08, 1e-4)

        norm, cmap = draw_beachball_batch(self.ax, results, diameter_deg)

        import matplotlib.cm as cm
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        self.figure.colorbar(sm, cax=self.cax, label="ΔCFF (MPa)")

        self.ax.set_title("Stress on Focal Mechanisms", fontsize=10, fontweight="bold")
        self.ax.set_xlabel("Longitude", fontsize=8)
        self.ax.set_ylabel("Latitude", fontsize=8)
        self.canvas.draw()

    def plot_aftershock_mc_test(self, result, mode="fraction"):
        """
        Plot core.aftershock_mc_test.AftershockMCTestResult: observed
        aftershock GE/LE threshold curve overlaid on the Monte Carlo
        null test's mean + 5-95th percentile bands, replicating
        coulomb.m's fig1 (fraction)/fig2 (count) pairing from
        menu_random_eq_null_test_callback -- both are the same plot
        shape here, toggled by `mode` rather than drawn as two separate
        figures, since this widget shows one plot at a time anyway.

        No colorbar needed for this plot type (unlike every other
        plot_*() method here) -- self.cax is hidden rather than filled.
        GE ("promoted": ΔCFF >= +thr) drawn as solid/darker, LE
        ("inhibited": ΔCFF <= -thr) as dashed/lighter, matching
        coulomb.m's own line-style convention for these two curves.
        """
        self._reset_axes()
        self.cax.set_visible(False)

        obs = result.observed
        null = result.null
        thr = null.thr_vec

        if mode == "fraction":
            obs_ge, obs_le = obs.frac_ge, obs.frac_le
            null_ge_mean, null_ge_p05, null_ge_p95 = null.frac_ge_mean, null.frac_ge_p05, null.frac_ge_p95
            null_le_mean, null_le_p05, null_le_p95 = null.frac_le_mean, null.frac_le_p05, null.frac_le_p95
            ylabel = "Fraction of points"
        elif mode == "count":
            obs_ge, obs_le = obs.cnt_ge, obs.cnt_le
            null_ge_mean, null_ge_p05, null_ge_p95 = null.cnt_ge_mean, null.cnt_ge_p05, null.cnt_ge_p95
            null_le_mean, null_le_p05, null_le_p95 = null.cnt_le_mean, null.cnt_le_p05, null.cnt_le_p95
            ylabel = "Count"
        else:
            raise ValueError(f"mode must be 'fraction' or 'count', got {mode!r}")

        # Null bands (Monte Carlo 5-95th percentile) -- drawn first so
        # the observed/mean lines sit visibly on top.
        self.ax.fill_between(thr, null_ge_p05, null_ge_p95, color="0.75",
                             alpha=0.45, label="Null GE 5-95%", linewidth=0)
        self.ax.fill_between(thr, null_le_p05, null_le_p95, color="0.55",
                             alpha=0.30, label="Null LE 5-95%", linewidth=0)
        self.ax.plot(thr, null_ge_mean, color="k", linestyle="-", linewidth=1.2,
                     alpha=0.6, label="Null GE mean")
        self.ax.plot(thr, null_le_mean, color="k", linestyle="--", linewidth=1.2,
                     alpha=0.6, label="Null LE mean")

        # Observed curve -- the actual result being tested against the null.
        self.ax.plot(thr, obs_ge, color="crimson", linestyle="-", linewidth=2.2,
                     label="Observed GE (ΔCFF ≥ +thr)")
        self.ax.plot(thr, obs_le, color="steelblue", linestyle="--", linewidth=2.2,
                     label="Observed LE (ΔCFF ≤ -thr)")

        depth_note = f" (depth: {null.depth_mode_used}"
        if null.depth_mode_used != null.depth_mode_requested:
            depth_note += f", requested {null.depth_mode_requested} unavailable"
        depth_note += ")"
        self.ax.set_title(
            f"Aftershock ΔCFF null test — N obs={obs.n_valid}, "
            f"N={null.n_points}×M={null.n_runs}{depth_note}",
            fontsize=9, fontweight="bold")
        self.ax.set_xlabel("Threshold |ΔCFF| (MPa)", fontsize=8)
        self.ax.set_ylabel(ylabel, fontsize=8)
        # Legend placed OUTSIDE the axes, in the space normally reserved
        # for self.cax (hidden for this plot type -- see docstring
        # above), rather than loc="best" inside the axes. loc="best"
        # was landing the legend box right under the title whenever the
        # figure's aspect ratio got short and wide (a squeezed embedding
        # window forces exactly that aspect -- see dialog layout notes),
        # visually overlapping the two. Anchoring outside the axes makes
        # the legend placement independent of the embedding window's
        # aspect ratio entirely.
        self.ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                       borderaxespad=0.0, frameon=True)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def plot_rate_state_forecast(self, forecast, mode="rate", observed=None):
        """
        Plot core.rate_state_seismicity.RateStateForecast: the whole-
        region total_rate()/total_cumulative() time series (see that
        class's own docstring for why these are the nansum over every
        grid cell, not a per-cell map -- a spatial snapshot map is a
        separate, not-yet-built view, same as plot_cff()'s single-
        depth-slice map is a different view from this one).

        mode="rate": seismicity rate vs t, with a thin horizontal
        reference line at the background rate r0 (per-region total, not
        per-point) so the eye has an immediate "back to background"
        anchor -- mirrors how Dieterich (1994)'s own published rate
        curves are usually plotted against their asymptote.
        mode="cumulative": cumulative expected event count vs t.
        mode="amplification": RateStateForecast.amplification() (=
        total_rate()/params.r0) vs t, with a thin reference line at 1.0
        (background level) -- added 2026-08-22c, see that method's own
        docstring. Same "back to background" visual anchor as mode=
        "rate"'s r0 line, just expressed as a ratio instead of an
        absolute rate, which is the more directly interpretable number
        for "how many times above background is this" questions (the
        external-review synthesis's own motivating example was a
        1,841x spike -- reading that off the raw-rate axis alone
        requires mentally dividing by r0 every time).

        observed : optional core.rate_state_calibration.ObservedTimeSeries
            (calibration/validation workflow -- see rate_state_dialog.py's
            Calibration group). When given and mode="cumulative", overlays
            the REAL catalog's observed cumulative count as a stepped grey
            line/markers directly against the model curve -- the actual
            "checking predicted vs real aftershock data" view. Ignored in
            mode="rate" (observed.counts_per_bin is a noisy per-bin count,
            not a rate -- overlaying it against a smooth rate curve would
            invite over-reading bin-to-bin noise as signal; the cumulative
            view is the honest one for a visual by-eye comparison).

        No colorbar needed (unlike every CFF-map plot_*() method here)
        -- self.cax is hidden rather than filled, same choice
        plot_aftershock_mc_test() makes for the same reason.
        """
        self._reset_axes()
        self.cax.set_visible(False)

        t = forecast.ts
        if mode == "rate":
            y = forecast.total_rate()
            ylabel = f"Total seismicity rate"
            self.ax.plot(t, y, color="crimson", linewidth=2.0, label="Forecast rate")
            self.ax.axhline(forecast.params.r0, color="k", linestyle="--",
                            linewidth=1.0, alpha=0.5,
                            label=f"Background rate r0={forecast.params.r0:g}")
        elif mode == "cumulative":
            y = forecast.total_cumulative()
            ylabel = "Cumulative expected events"
            self.ax.plot(t, y, color="steelblue", linewidth=2.0, label="Cumulative count")
            if observed is not None:
                self.ax.step(observed.ts, observed.cumulative, where="post",
                             color="dimgray", linewidth=1.5, alpha=0.85,
                             label=f"Observed catalog (N={observed.n_events_total})")
        elif mode == "amplification":
            y = forecast.amplification()
            ylabel = "Rate amplification R/R\u2080"
            self.ax.plot(t, y, color="darkorange", linewidth=2.0, label="Amplification R/R\u2080")
            self.ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0, alpha=0.5,
                            label="Background (R/R\u2080=1)")
            finite_y = y[np.isfinite(y)]
            if finite_y.size:
                peak = float(np.max(finite_y))
                self.ax.text(0.99, 0.97, f"Peak: {peak:.3g}\u00d7 background",
                             transform=self.ax.transAxes, ha="right", va="top",
                             fontsize=7.5, color="darkorange",
                             bbox=dict(boxstyle="round", fc="white", ec="darkorange", alpha=0.85))
        else:
            raise ValueError(f"mode must be 'rate', 'cumulative', or 'amplification', got {mode!r}")

        n_points = forecast.rate.shape[0] * forecast.rate.shape[1] * forecast.rate.shape[2]
        self.ax.set_title(
            f"Rate-and-state seismicity forecast (Dieterich, 1994) — "
            f"{n_points} grid point(s)",
            fontsize=9, fontweight="bold")
        self.ax.set_xlabel("t", fontsize=8)
        self.ax.set_ylabel(ylabel, fontsize=8)
        # Explicit tick cap + label rotation (2026-08-23, see the
        # width-floor comment in __init__ for the actual root cause this
        # is a defensive second layer for): even with the new
        # setMinimumWidth(320) floor, capping the tick COUNT and giving
        # each label a little rotation is a cheap, always-safe guard
        # against overlapping edge labels ("0"/"500" reported) on
        # whatever the narrowest embedding this ends up in turns out to
        # be, rather than relying on width alone.
        self.ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        self.ax.tick_params(axis="x", labelsize=7, labelrotation=20)
        self.ax.tick_params(axis="y", labelsize=7)
        self.ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                       borderaxespad=0.0, frameon=True)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def plot_rate_state_map(self, lon2d, lat2d, values, mode="rate",
                             depth_label=None, time_label=None):
        """
        Spatial snapshot of core.rate_state_seismicity.RateStateForecast:
        one depth slice's rate (or cumulative count) map at one forecast
        time, as a sequential colour map -- unlike plot_cff()'s diverging
        RdBu_r (ΔCFF can be positive or negative), Dieterich (1994)'s
        rate R(t) and cumulative count C(t) are both always >= 0 (R's
        denominator B*exp(-t/ta)+1 stays positive for every finite B>-1,
        which every finite dcff/asig combination produces -- see
        core.rate_state_seismicity.d94's own docstring), so a diverging
        map here would waste half the colour range on a sign that never
        occurs. "viridis" chosen for the same reason plot_cff() uses
        RdBu_r: perceptually uniform, colourblind-safe, and distinct
        from every other map this widget draws so a screenshot is
        unambiguous about which plot it came from.

        values : 2D array (n_lat, n_lon) -- ALREADY sliced by the caller
        to one depth/time (this method has no opinion on which slice;
        RateStateForecastDialog._replot() does that slicing against the
        full (n_depth, n_lat, n_lon, n_t) forecast.rate/.cumulative
        arrays).
        """
        self._reset_axes()

        vmax = np.nanpercentile(values, 98)
        vmax = max(vmax, 1e-12)
        vmin = 0.0
        im = self.ax.pcolormesh(lon2d, lat2d, values, cmap="viridis",
                                vmin=vmin, vmax=vmax, shading="auto")
        label = "Seismicity rate" if mode == "rate" else "Cumulative expected events"
        self.figure.colorbar(im, cax=self.cax, label=label)

        title = "Rate-and-state seismicity forecast — spatial snapshot"
        subtitle_bits = [b for b in (depth_label, time_label) if b]
        if subtitle_bits:
            title += "  [" + ", ".join(subtitle_bits) + "]"
        self.ax.set_title(title, fontsize=9, fontweight="bold")
        self.ax.set_xlabel("Longitude", fontsize=8)
        self.ax.set_ylabel("Latitude", fontsize=8)
        self.ax.set_aspect("equal")
        self.canvas.draw()

    def plot_cff_field_histogram(self, cff_values, stats, title_extra=None):
        """
        Histogram of a ΔCFF field's finite values (core.cff_volume.
        cff_field_stats()'s companion plot -- see that function's
        docstring), with vertical reference lines at zero, mean, median,
        P5, and P95, and a stats text box carrying the full CFFFieldStats
        breakdown. Added 2026-08-22c per the external-review synthesis's
        session-3 items 2-3 (field stats + histogram, flagged as the
        single most useful addition to this dialog).

        cff_values : array-like, any shape (flattened here) -- NaNs are
        dropped before histogramming, same finite-only convention
        cff_field_stats() uses.
        stats : a core.cff_volume.CFFFieldStats (already computed by the
        caller -- this method doesn't recompute it, so the numbers in
        the text box are guaranteed to match whatever the caller is
        showing/reporting elsewhere for the same field).

        No colorbar needed (unlike every CFF-map plot_*() method here)
        -- self.cax is hidden rather than filled, same choice
        plot_aftershock_mc_test()/plot_rate_state_forecast() make.
        """
        self._reset_axes()
        self.cax.set_visible(False)

        values = np.asarray(cff_values, dtype=float).ravel()
        finite = values[np.isfinite(values)]

        n_bins = min(80, max(10, int(np.sqrt(max(finite.size, 1)))))
        self.ax.hist(finite, bins=n_bins, color="steelblue", alpha=0.75,
                     edgecolor="white", linewidth=0.3)

        ref_lines = [
            (0.0, "k", "-", "ΔCFF=0"),
            (stats.mean, "crimson", "--", "mean"),
            (stats.median, "darkorange", "--", "median"),
            (stats.p5, "gray", ":", "P5/P95"),
            (stats.p95, "gray", ":", None),
        ]
        for x, color, ls, label in ref_lines:
            self.ax.axvline(x, color=color, linestyle=ls, linewidth=1.2,
                            alpha=0.8, label=label)

        stats_text = (
            f"n = {stats.n_finite}/{stats.n_points}\n"
            f"min = {stats.min:.4g}\n"
            f"max = {stats.max:.4g}\n"
            f"mean = {stats.mean:.4g}\n"
            f"median = {stats.median:.4g}\n"
            f"std = {stats.std:.4g}\n"
            f"P5 = {stats.p5:.4g}\n"
            f"P95 = {stats.p95:.4g}\n"
            f"+: {100*stats.frac_positive:.1f}%   "
            f"-: {100*stats.frac_negative:.1f}%"
        )
        self.ax.text(1.02, 0.98, stats_text, transform=self.ax.transAxes,
                     ha="left", va="top", fontsize=7.5, family="monospace",
                     bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

        title = "ΔCFF field statistics"
        if title_extra:
            title += f"  [{title_extra}]"
        self.ax.set_title(title, fontsize=9, fontweight="bold")
        self.ax.set_xlabel("ΔCFF (MPa)", fontsize=8)
        self.ax.set_ylabel("Count", fontsize=8)
        self.ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 0.55),
                       borderaxespad=0.0, frameon=True)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def plot_eq_catalog_timeline(self, timeline, title_extra=None):
        """
        QA/QC view of core.rate_state_calibration.CatalogTimeline: a
        histogram of real catalog events over time-since-mainshock, with
        vertical reference lines marking t0 (forecast start),
        t_fit_max (calibration/validation split, if set), and t_max
        (forecast end) -- so the person can see AT A GLANCE whether the
        catalog's actual event density lines up with wherever the
        calibration/validation windows are currently drawn, rather than
        inferring it from separately-reported counts. Added 2026-08-23
        per request, alongside the datetime-parsing fix in
        core.observation_import._stringify_attr() this view exists to
        make failures of immediately visible (an all-flat-at-t=0 or
        empty histogram despite timeline.n_total being large is exactly
        that failure mode -- see this dialog's own status text, which
        surfaces the same n_total/n_with_time funnel as numbers next to
        this plot).

        If timeline.magnitudes has any finite values, they're overlaid
        as a faint scatter on a secondary (right) y-axis -- optional per
        the request ("Y maybe count, option magnitude") -- purely for a
        visual check that Mc filtering and magnitude parsing look sane;
        no reference lines/statistics are computed on it.

        No colorbar needed (unlike every CFF-map plot_*() method here)
        -- self.cax is hidden, same choice the other non-map rate-state
        views make.
        """
        self._reset_axes()
        self.cax.set_visible(False)

        rel = np.asarray(timeline.rel_times, dtype=float)
        n = rel.size
        if n == 0:
            self.ax.text(0.5, 0.5,
                         f"No events with a usable time "
                         f"(n_total={timeline.n_total}, n_with_time="
                         f"{timeline.n_with_time}) -- check the mapped time "
                         "column/schema in Load Catalog.",
                         ha="center", va="center", transform=self.ax.transAxes,
                         fontsize=9, color="crimson", wrap=True)
            self.ax.set_xticks([]); self.ax.set_yticks([])
            title = "Catalog timeline (QA/QC)"
            if title_extra:
                title += f"  [{title_extra}]"
            self.ax.set_title(title, fontsize=9, fontweight="bold")
            self.canvas.draw()
            return

        n_bins = min(80, max(10, int(np.sqrt(max(n, 1)))))
        self.ax.hist(rel, bins=n_bins, color="steelblue", alpha=0.75,
                     edgecolor="white", linewidth=0.3, label=f"Events (N={n})")

        ref_lines = []
        if timeline.t0 is not None:
            ref_lines.append((timeline.t0, "darkgreen", "--", f"t0={timeline.t0:g}"))
        if timeline.t_fit_max is not None:
            ref_lines.append((timeline.t_fit_max, "purple", "--",
                              f"calib/valid split={timeline.t_fit_max:g}"))
        if timeline.t_max is not None:
            ref_lines.append((timeline.t_max, "k", "--", f"t_max={timeline.t_max:g}"))
        for x, color, ls, label in ref_lines:
            self.ax.axvline(x, color=color, linestyle=ls, linewidth=1.3,
                            alpha=0.85, label=label)

        mag = np.asarray(timeline.magnitudes, dtype=float)
        finite_mag = np.isfinite(mag)
        if np.any(finite_mag):
            self._twin_ax = self.ax.twinx()
            self._twin_ax.scatter(rel[finite_mag], mag[finite_mag], s=6, color="darkorange",
                       alpha=0.5, label="Magnitude")
            self._twin_ax.set_ylabel("Magnitude", fontsize=8, color="darkorange")
            self._twin_ax.tick_params(axis="y", labelsize=7, labelcolor="darkorange")

        counts_text = (
            f"n_total = {timeline.n_total}\n"
            f"n_with_time = {timeline.n_with_time}\n"
            f"n_after_mag_filter = {timeline.n_after_mag_filter}\n"
            f"n_after_region_filter = {timeline.n_after_region_filter}"
        )
        if timeline.n_in_forecast_window is not None:
            counts_text += f"\nn_in_forecast_window = {timeline.n_in_forecast_window}"
        if timeline.n_in_calibration_window is not None:
            counts_text += f"\nn_in_calibration_window = {timeline.n_in_calibration_window}"
        if timeline.n_in_validation_window is not None:
            counts_text += f"\nn_in_validation_window = {timeline.n_in_validation_window}"
        self.ax.text(1.02, 0.98, counts_text, transform=self.ax.transAxes,
                     ha="left", va="top", fontsize=7.5, family="monospace",
                     bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

        title = "Catalog timeline (QA/QC)"
        if title_extra:
            title += f"  [{title_extra}]"
        self.ax.set_title(title, fontsize=9, fontweight="bold")
        self.ax.set_xlabel(f"t since mainshock ({timeline.time_unit})", fontsize=8)
        self.ax.set_ylabel("Event count", fontsize=8)
        self.ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        self.ax.tick_params(axis="x", labelsize=7, labelrotation=20)
        self.ax.tick_params(axis="y", labelsize=7)
        # Anchored low (y=0.28) and stacked BELOW the counts textbox
        # above (which top-anchors at y=0.98 and, with up to 7 lines of
        # monospace text, can run past the plot's vertical midpoint) --
        # sharing the same anchor height the two boxes visibly
        # overlapped in testing.
        self.ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 0.28),
                       borderaxespad=0.0, frameon=True)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def plot_stress_inversion_stereonet(self, result, boot_axes_end=None,
                                        title_extra=None):
        """
        Lower-hemisphere equal-area stereonet of the S1/S2/S3 principal-
        stress axes returned by `core.stress_inversion.invert_regional_stress()`
        (`result["axes_end"]`, each a (strike_deg, plunge_deg) tuple in
        this plugin's own convention -- see that module's docstring),
        optionally overlaid with a bootstrap confidence cloud
        (`core.stress_inversion.bootstrap_regional_stress()`'s
        `boot_axes_end`, each an (n_resamplings, 2) array of the same
        (strike_deg, plunge_deg) pairs).

        Ported into this class's own fixed-axes/self.cax-hidden pattern
        (2026-08-25 addendum decision) rather than calling ILSI's own
        `inversion_one_set_instability(..., plot=True)` plotting code
        directly, for the same reason every other plot in this project
        goes through PlotWidget: consistent styling/sizing/save-to-file
        behaviour with the rest of the plugin's output, rather than a
        visually inconsistent one-off from a third-party library.

        Needs `mplstereonet` -- a genuinely NEW dependency (see
        `core.stress_inversion.check_mplstereonet()`), separate from
        ILSI itself (ILSI's own inversion math needs only numpy/scipy).
        If it isn't importable, this draws an explanatory message on
        self.ax instead of raising, so a dialog can call this
        unconditionally and let the message (rather than an uncaught
        ImportError) be what the user sees.

        A `stereonet`-projection Axes is a SEPARATE object from
        self.ax/self.cax (matplotlib does not support changing an
        existing rectilinear Axes' projection in place) -- see
        self._stereonet_ax tracking in __init__/_reset_axes().
        """
        self._reset_axes()
        self.cax.set_visible(False)

        try:
            import mplstereonet
        except ImportError as e:
            self.ax.text(
                0.5, 0.5,
                "mplstereonet is not installed -- this plot needs it.\n"
                "pip install mplstereonet\n"
                f"(import error: {e})",
                ha="center", va="center", transform=self.ax.transAxes,
                fontsize=8.5, color="crimson", wrap=True)
            self.ax.set_xticks([]); self.ax.set_yticks([])
            self.canvas.draw()
            return

        # self.ax is hidden (not removed -- _reset_axes() un-hides and
        # reuses it on the next non-stereonet plot) and a fresh
        # 'stereonet'-projection axes takes over its exact figure-
        # fraction footprint, so this plot lines up with every other
        # plot_*() method's axes position/size.
        self.ax.set_visible(False)
        left, bottom, width, height = self.ax.get_position().bounds
        self._stereonet_ax = self.figure.add_axes(
            [left, bottom, width, height], projection="stereonet")
        sax = self._stereonet_ax

        # AXES colour convention: sigma1 (most compressive) red,
        # sigma2 green, sigma3 (least compressive) blue -- matches this
        # project's convention nowhere else yet, chosen simply as a
        # common, readable red/green/blue ordering for a 3-axis triad
        # rather than any external standard.
        colors = {"S1": "crimson", "S2": "seagreen", "S3": "steelblue"}

        if boot_axes_end is not None:
            for name, color in colors.items():
                arr = np.asarray(boot_axes_end.get(name))
                if arr is None or arr.size == 0:
                    continue
                strikes_b, plunges_b = arr[:, 0], arr[:, 1]
                sax.line(plunges_b, strikes_b, marker="o", markersize=2.5,
                         color=color, alpha=0.15, linestyle="none")

        axes_end = result["axes_end"]
        for name, color in colors.items():
            strike, plunge = axes_end[name]
            sax.line(plunge, strike, marker="o", markersize=10,
                     markeredgecolor="k", markeredgewidth=0.8,
                     color=color, linestyle="none",
                     label=f"{name} ({strike:.0f}°/{plunge:.0f}°)")

        sax.grid(True, alpha=0.4)
        sax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   borderaxespad=0.0, frameon=True)

        title = (f"Principal stress axes  (R={result['shape_ratio']:.2f}, "
                 f"friction={result['friction_coefficient']:.2f}, "
                 f"n={result['n_events']})")
        if title_extra:
            title += f"  [{title_extra}]"
        sax.set_title(title, fontsize=9, fontweight="bold", pad=14)
        if boot_axes_end is not None:
            n_boot = next((np.asarray(v).shape[0] for v in boot_axes_end.values()
                          if np.asarray(v).size), 0)
            self.figure.text(
                0.02, 0.02,
                f"faint dots: {n_boot} bootstrap resample(s) per axis",
                fontsize=6.5, color="dimgray")
        self.canvas.draw()

    def save_to_file(self, path, dpi=300):
        """
        Save the currently-displayed plot (CFF map, displacement map, or
        cross-section — whichever was drawn most recently) to an image
        file. Format is inferred from the file extension; matplotlib
        supports .png, .svg, .pdf, .eps, .jpg, .tif among others.
        """
        self.figure.savefig(path, dpi=dpi, bbox_inches="tight")
