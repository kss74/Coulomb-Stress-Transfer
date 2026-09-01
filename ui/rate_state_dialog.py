# -*- coding: utf-8 -*-
"""
Dialog for rate-and-state seismicity forecasting
(core.rate_state_seismicity, Dieterich 1994).

Workflow: configure depth slices for a 3D ΔCFF volume (reusing
core.cff_volume exactly as AftershockMCTestDialog does -- same
build/cache/load machinery, same volume-grid resolution cap), set the
Dieterich (1994) parameters (r0, asig, ta, optional tdotr) and the
forecast time axis (t0, t_max, n_t), run in the background (building
the volume is a real DC3D cost, same as the aftershock null test), plot
the whole-region total_rate()/total_cumulative() forecast, export a
report/CSV.

Constructed with the same (sources, receiver, elastic, grid) tuple
main_dialog.py's ComputeWorker already has, plus optional
(regional, friction) for the optimally-oriented-plane ΔCFF mode -- same
convention AftershockMCTestDialog already established, deliberately
mirrored here rather than reinvented (this dialog is essentially
AftershockMCTestDialog's volume-building half, followed by a different
computation and a different plot/report, not a different UI paradigm).

Layout follows the same left(scrollable settings)/right(plot) split as
AftershockMCTestDialog and main_dialog.py, for the same reason
documented there: a single stacked QVBoxLayout column starves the plot
of vertical space once configure_resizable_dialog()'s screen-height cap
kicks in.
"""

import os
import tempfile
import csv

import numpy as np
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, QLabel,
    QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar, QGroupBox,
    QMessageBox, QFileDialog, QRadioButton, QWidget, QComboBox, QButtonGroup
)
from qgis.PyQt.QtCore import QThread, pyqtSignal

from ..core.okada_engine import GridParameters
from ..core.optimal_plane import RegionalStress
from ..core.cff_volume import (
    build_cff_volume, cff_volume_cache_key, save_cff_volume, load_cff_volume,
    cff_field_stats, apply_near_field_mask,
)
from ..core.rate_state_seismicity import RateStateParams, forecast_from_cff_volume
from ..core.rate_state_report import build_rate_state_report, build_rate_state_csv_rows
from ..core.aftershock_mc_report import write_report_pdf
from ..core import rate_state_calibration as rsc
from ..core.eq_catalog_import import parse_datetime, events_to_eq_array
from datetime import timezone
from .dialog_utils import configure_resizable_dialog, wrap_widget_in_scroll_area
from .plot_widget import PlotWidget
from .eq_catalog_import_dialog import EQCatalogImportDialog


# Same reasoning/value as aftershock_mc_dialog._MAX_VOLUME_GRID_POINTS_PER_AXIS:
# this dialog only samples the volume at every grid cell to sum a
# whole-region total, it never needs a finer volume than the display
# grid already provides, and a high-resolution volume buys nothing but
# DC3D cost. Kept as the SAME cap (not re-derived) so a volume built by
# one dialog is reusable (same cache key) by the other whenever the
# caller-supplied grid+depths match, rather than needlessly recomputing.
_MAX_VOLUME_GRID_POINTS_PER_AXIS = 40

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "coulomb_stress_transfer_cff_volume_cache")

_DEFAULT_DEPTHS_KM = "0, 5, 10, 15, 20"


class RateStateForecastWorker(QThread):
    """
    Background worker: (1) build/load-from-cache the 3D CFF volume,
    (2) run forecast_from_cff_volume() against it. Progress split
    0-70% volume build / 70-100% forecast, same approximate split
    AftershockMCWorker uses and for the same reason (forecast itself is
    pure numpy over an already-built volume, fast; volume build is the
    dominant, input-dependent cost).
    """
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(object)   # RateStateForecast
    error = pyqtSignal(str)

    def __init__(self, sources, receiver, elastic, volume_grid, depths_km,
                 ts, t0, params, use_cache=True, cff_mode="fixed",
                 regional=None, friction=None, exclude_near_field=True):
        super().__init__()
        self.sources = sources
        self.receiver = receiver
        self.elastic = elastic
        self.volume_grid = volume_grid
        self.depths_km = depths_km
        self.ts = ts
        self.t0 = t0
        self.params = params
        self.use_cache = use_cache
        self.cff_mode = cff_mode
        self.regional = regional
        self.friction = friction
        self.exclude_near_field = exclude_near_field
        self._cancelled = False
        self.volume = None   # populated on success -- see run(); read by
                              # RateStateForecastDialog._on_finished() so the
                              # Calibration group can reuse this exact
                              # already-built CFFVolume (core.rate_state_
                              # calibration.calibrate_rate_state() needs the
                              # raw cff array, which RateStateForecast itself
                              # doesn't carry) without rebuilding it.

    def request_cancel(self):
        self._cancelled = True

    def run(self):
        try:
            volume = None
            cache_path = None
            if self.use_cache:
                key = cff_volume_cache_key(self.sources, self.receiver, self.elastic,
                                           self.volume_grid, self.depths_km,
                                           mode=self.cff_mode, regional=self.regional,
                                           friction=self.friction)
                os.makedirs(_CACHE_DIR, exist_ok=True)
                cache_path = os.path.join(_CACHE_DIR, f"{key}.npz")
                volume = load_cff_volume(cache_path)

            if volume is None:
                if self._cancelled:
                    return
                volume = build_cff_volume(
                    self.sources, self.receiver, self.elastic, self.volume_grid,
                    self.depths_km, progress_callback=lambda p: self.progress.emit(int(0.7 * p)),
                    mode=self.cff_mode, regional=self.regional, friction=self.friction)
                if self.use_cache and cache_path:
                    try:
                        save_cff_volume(volume, cache_path)
                    except Exception:
                        pass
            else:
                self.progress.emit(70)

            if self._cancelled:
                return

            self.progress.emit(85)
            forecast = forecast_from_cff_volume(
                volume, self.ts, self.t0, self.params,
                exclude_near_field=self.exclude_near_field)
            self.progress.emit(100)
            self.volume = volume
            self.finished_ok.emit(forecast)

        except Exception as e:
            self.error.emit(str(e))


class RateStateForecastDialog(QDialog):
    """
    sources/receiver/elastic/grid: same objects main_dialog.py's
    ComputeWorker already has on hand -- see AftershockMCTestDialog's
    own docstring for why grid's .depth_km is irrelevant here (only
    lon/lat extent + resolution matter). regional/friction: same
    RegionalStress object and friction value the "Optimal-Plane ΔCFF"
    tab already gathers; None disables the "optimally-oriented fault"
    ΔCFF option, exactly as in AftershockMCTestDialog.
    """

    def __init__(self, sources, receiver, elastic, grid, regional=None, friction=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rate-and-State Seismicity Forecast (Dieterich, 1994)")
        configure_resizable_dialog(self, 1080, 720, min_width=760, min_height=520)
        self.sources = sources
        self.receiver = receiver
        self.elastic = elastic
        self.grid = grid
        self.regional = regional
        self.friction = friction
        self._worker = None
        self._last_forecast = None
        self._last_meta = None
        self._last_volume = None          # CFFVolume from the last completed forecast run
        self.eq_events = []                # List[EQCatalogEvent], loaded via "Load Catalog…"
        self._last_background = None       # rate_state_calibration.BackgroundRateResult
        self._last_calibration = None      # rate_state_calibration.CalibrationResult
        self._last_observed = None         # rate_state_calibration.ObservedTimeSeries
        self._last_validation = None       # rate_state_calibration.ValidationScore
        self._last_cff_stats = None        # core.cff_volume.CFFFieldStats, from the last completed forecast's volume

        outer_layout = QHBoxLayout(self)

        left_inner = QWidget()
        left = QVBoxLayout(left_inner)
        left_scroll = wrap_widget_in_scroll_area(left_inner, self)
        left_scroll.setMaximumWidth(400)
        outer_layout.addWidget(left_scroll, 0)

        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        outer_layout.addWidget(right_widget, 1)

        # ── ΔCFF resolution mode ────────────────────────────────────────
        cff_mode_group = QGroupBox("ΔCFF resolved on")
        cff_mode_layout = QVBoxLayout(cff_mode_group)
        self.radio_cff_fixed = QRadioButton("Specified receiver fault (Receiver Fault tab)")
        self.radio_cff_optimal = QRadioButton(
            "Optimally-oriented fault at each point (regional stress + friction)")
        self.radio_cff_fixed.setChecked(True)
        cff_mode_layout.addWidget(self.radio_cff_fixed)
        cff_mode_layout.addWidget(self.radio_cff_optimal)
        cff_note = QLabel(
            "Optimally-oriented ΔCFF is generally the better choice for a real "
            "aftershock forecast, for the same reason it's recommended in the "
            "Aftershock/ΔCFF Null Test dialog: real aftershocks occur on "
            "whichever plane is locally best-oriented for failure. Requires a "
            "regional stress tensor (Optimal-Plane ΔCFF tab).")
        cff_note.setWordWrap(True)
        cff_note.setStyleSheet("color: gray; font-size: 10px;")
        cff_mode_layout.addWidget(cff_note)
        if self.regional is None:
            self.radio_cff_optimal.setEnabled(False)
            self.radio_cff_optimal.setToolTip(
                "Configure a regional stress tensor in the Optimal-Plane ΔCFF tab first.")
        left.addWidget(cff_mode_group)

        # ── Depth slices ─────────────────────────────────────────────────
        depth_group = QGroupBox("Depth slices (3D ΔCFF volume)")
        depth_form = QFormLayout(depth_group)
        self.depth_edit = QLineEdit(_DEFAULT_DEPTHS_KM)
        depth_form.addRow("Depths (km, comma-separated):", self.depth_edit)
        res_note = QLabel(f"Volume grid resolution capped at "
                          f"{_MAX_VOLUME_GRID_POINTS_PER_AXIS}×{_MAX_VOLUME_GRID_POINTS_PER_AXIS} "
                          f"(same cap as the Aftershock/ΔCFF Null Test dialog -- volumes "
                          f"are reused between the two via the same on-disk cache when the "
                          f"source/receiver/grid/depths/ΔCFF mode all match).")
        res_note.setWordWrap(True)
        res_note.setStyleSheet("color: gray; font-size: 10px;")
        depth_form.addRow(res_note)
        left.addWidget(depth_group)

        # ── Dieterich (1994) parameters ────────────────────────────────
        params_group = QGroupBox("Rate-and-state parameters")
        params_form = QFormLayout(params_group)

        # Single authoritative time unit for r0/ta/t0/t_max/ts below AND
        # for the Calibration group's catalog-time conversion further
        # down -- see PROJECT_HANDOVER_ADDENDUM_2026-08-22b_time_unit_
        # consistency.md. Previously the Calibration group had its own,
        # independently-set combo; that allowed the two to silently
        # drift apart (catalog timestamps converted into a different
        # unit than the forecast's own r0/ta/t0 were set up in, with no
        # error anywhere). One combo now, read by both, plus
        # core.rate_state_calibration.assert_consistent_time_unit() as
        # a second line of defense for the case a forecast that was run
        # under a DIFFERENT unit selection is still the one loaded when
        # this combo gets changed afterwards.
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(list(rsc.TIME_UNIT_SECONDS.keys()))
        self.time_unit_combo.setCurrentText("days")
        params_form.addRow("Time unit (r0/ta/t0/t_max, and catalog conversion):",
                           self.time_unit_combo)

        self.r0_spin = QDoubleSpinBox()
        self.r0_spin.setRange(1e-6, 1e9)
        self.r0_spin.setDecimals(4)
        self.r0_spin.setValue(1.0)
        params_form.addRow("Background rate r0 (events/time, whole region):", self.r0_spin)

        self.asig_spin = QDoubleSpinBox()
        self.asig_spin.setRange(1e-6, 1e6)
        self.asig_spin.setDecimals(6)
        self.asig_spin.setValue(0.01)
        self.asig_spin.setSuffix(" MPa")
        params_form.addRow("a·sigma (asig):", self.asig_spin)

        self.ta_spin = QDoubleSpinBox()
        self.ta_spin.setRange(1e-6, 1e9)
        self.ta_spin.setDecimals(4)
        self.ta_spin.setValue(100.0)
        params_form.addRow("Aftershock decay time ta:", self.ta_spin)

        self.tdotr_checkbox = QCheckBox("Override background stressing rate tdotr")
        self.tdotr_checkbox.setToolTip(
            "Leave unchecked to default tdotr = asig/ta (Dieterich's own "
            "default when tdotr isn't independently specified).")
        params_form.addRow(self.tdotr_checkbox)
        self.tdotr_spin = QDoubleSpinBox()
        self.tdotr_spin.setRange(1e-9, 1e6)
        self.tdotr_spin.setDecimals(9)
        self.tdotr_spin.setValue(1e-4)
        self.tdotr_spin.setSuffix(" MPa/time")
        self.tdotr_spin.setEnabled(False)
        self.tdotr_checkbox.toggled.connect(self.tdotr_spin.setEnabled)
        params_form.addRow("tdotr:", self.tdotr_spin)

        units_note = QLabel(
            "Time and stress units are whatever you use consistently: r0/ta/t0/"
            "forecast times must share one time unit (e.g. days), asig must "
            "share MPa with this plugin's own ΔCFF fields (deliberately NOT "
            "kPa, to stay consistent with the rest of this plugin).")
        units_note.setWordWrap(True)
        units_note.setStyleSheet("color: gray; font-size: 10px;")
        params_form.addRow(units_note)

        left.addWidget(params_group)

        # ── Forecast time axis ──────────────────────────────────────────
        time_group = QGroupBox("Forecast time axis")
        time_form = QFormLayout(time_group)

        self.t0_spin = QDoubleSpinBox()
        self.t0_spin.setRange(1e-9, 1e9)
        self.t0_spin.setDecimals(6)
        self.t0_spin.setValue(0.01)
        time_form.addRow("t0 (cumulative-count start):", self.t0_spin)
        t0_note = QLabel(
            "t0=0 exactly can give an infinite cumulative count for some "
            "parameter choices (an inherent property of the closed-form "
            "solution) -- defaults to a small positive value rather than 0.")
        t0_note.setWordWrap(True)
        t0_note.setStyleSheet("color: gray; font-size: 10px;")
        time_form.addRow(t0_note)

        self.t_max_spin = QDoubleSpinBox()
        self.t_max_spin.setRange(1e-6, 1e9)
        self.t_max_spin.setDecimals(4)
        self.t_max_spin.setValue(500.0)
        time_form.addRow("t_max:", self.t_max_spin)

        self.n_t_spin = QSpinBox()
        self.n_t_spin.setRange(2, 5000)
        self.n_t_spin.setValue(200)
        time_form.addRow("Number of time steps:", self.n_t_spin)

        # Magnitude of completeness -- LABEL ONLY, see core.rate_state_
        # report.build_rate_state_report()'s "mc" meta key docstring.
        # This module has no magnitude dependence of its own; the sole
        # effect of setting this is that the report/plot wording changes
        # from "events" to "events with M>=mc" so the forecast output
        # isn't misread as an unconditional event count. Added 2026-08-22c,
        # external-review synthesis session 3 item 4.
        self.mc_checkbox = QCheckBox("Label forecast events as M >= Mc")
        self.mc_spin = QDoubleSpinBox()
        self.mc_spin.setRange(-2.0, 10.0)
        self.mc_spin.setDecimals(2)
        self.mc_spin.setValue(2.5)
        self.mc_spin.setEnabled(False)
        self.mc_checkbox.toggled.connect(self.mc_spin.setEnabled)
        mc_row = QHBoxLayout()
        mc_row.addWidget(self.mc_checkbox)
        mc_row.addWidget(self.mc_spin)
        time_form.addRow(mc_row)
        mc_note = QLabel(
            "Labeling only -- this forecast has no magnitude dependence. "
            "Set to whatever completeness threshold r0/asig/ta were "
            "calibrated against, if any, so the report doesn't imply an "
            "unconditional event count.")
        mc_note.setWordWrap(True)
        mc_note.setStyleSheet("color: gray; font-size: 10px;")
        time_form.addRow(mc_note)

        left.addWidget(time_group)

        # ── Calibration (real earthquake catalog) ────────────────────────
        # Item flagged in PROJECT_HANDOVER_ADDENDUM_2026-08-21c_external_
        # review_synthesis.md, Session 3 point 6 -- see core.rate_state_
        # calibration's own module docstring for the full physics/stats
        # design. Workflow: Run Forecast once first (builds/caches the CFF
        # volume the fit needs) -> Load Catalog -> Compute r0 from a pre-
        # mainshock window -> Fit asig/ta to the real aftershock sequence
        # -> Run Forecast again with the fitted values now in the spinboxes
        # above -> Validate Against Catalog to score the result.
        calib_group = QGroupBox("Calibration (real earthquake catalog)")
        calib_layout = QVBoxLayout(calib_group)

        calib_intro = QLabel(
            "Run Forecast once first (builds the ΔCFF volume this needs), "
            "then load a catalog below to compute an observed background "
            "rate, fit asig/ta to the real aftershock decay, and validate "
            "the forecast against what actually occurred.")
        calib_intro.setWordWrap(True)
        calib_intro.setStyleSheet("color: gray; font-size: 10px;")
        calib_layout.addWidget(calib_intro)

        catalog_row = QHBoxLayout()
        self.btn_load_catalog = QPushButton("Load Catalog…")
        self.btn_load_catalog.clicked.connect(self._load_catalog)
        catalog_row.addWidget(self.btn_load_catalog)
        self.catalog_status_label = QLabel("No catalog loaded.")
        self.catalog_status_label.setWordWrap(True)
        catalog_row.addWidget(self.catalog_status_label, 1)
        calib_layout.addLayout(catalog_row)

        calib_form = QFormLayout()
        self.mainshock_time_edit = QLineEdit()
        self.mainshock_time_edit.setPlaceholderText("YYYY-MM-DD HH:MM:SS (UTC)")
        calib_form.addRow("Mainshock origin time:", self.mainshock_time_edit)

        self.calib_time_unit_note = QLabel()
        self.calib_time_unit_note.setWordWrap(True)
        self.calib_time_unit_note.setStyleSheet("color: gray; font-size: 10px;")
        calib_form.addRow(self.calib_time_unit_note)
        self._update_calib_time_unit_note()
        self.time_unit_combo.currentIndexChanged.connect(self._update_calib_time_unit_note)

        self.mag_cutoff_checkbox = QCheckBox("Magnitude cutoff (Mc)")
        self.mag_cutoff_spin = QDoubleSpinBox()
        self.mag_cutoff_spin.setRange(-2.0, 10.0)
        self.mag_cutoff_spin.setDecimals(2)
        self.mag_cutoff_spin.setValue(2.5)
        self.mag_cutoff_spin.setEnabled(False)
        self.mag_cutoff_checkbox.toggled.connect(self.mag_cutoff_spin.setEnabled)
        mag_row = QHBoxLayout()
        mag_row.addWidget(self.mag_cutoff_checkbox)
        mag_row.addWidget(self.mag_cutoff_spin)
        calib_form.addRow(mag_row)
        calib_layout.addLayout(calib_form)

        bg_group = QGroupBox("Background rate from catalog")
        bg_form = QFormLayout(bg_group)
        self.bg_start_edit = QLineEdit()
        self.bg_start_edit.setPlaceholderText("YYYY-MM-DD HH:MM:SS (pre-mainshock window start)")
        bg_form.addRow("Window start:", self.bg_start_edit)
        self.bg_end_edit = QLineEdit()
        self.bg_end_edit.setPlaceholderText("YYYY-MM-DD HH:MM:SS (usually the mainshock time)")
        bg_form.addRow("Window end:", self.bg_end_edit)
        self.btn_compute_r0 = QPushButton("Compute r0 from Catalog")
        self.btn_compute_r0.clicked.connect(self._compute_background_rate)
        bg_form.addRow(self.btn_compute_r0)
        calib_layout.addWidget(bg_group)

        fit_row = QHBoxLayout()
        self.fit_r0_checkbox = QCheckBox("Also fit r0 (default: fixed from background rate)")
        fit_row.addWidget(self.fit_r0_checkbox)
        calib_layout.addLayout(fit_row)
        held_out_form = QFormLayout()
        self.t_fit_max_checkbox = QCheckBox("Hold out a later time window for validation")
        held_out_form.addRow(self.t_fit_max_checkbox)
        self.t_fit_max_spin = QDoubleSpinBox()
        self.t_fit_max_spin.setRange(1e-6, 1e9)
        self.t_fit_max_spin.setDecimals(4)
        self.t_fit_max_spin.setValue(100.0)
        self.t_fit_max_spin.setEnabled(False)
        self.t_fit_max_checkbox.toggled.connect(self.t_fit_max_spin.setEnabled)
        held_out_form.addRow("Fit only t <=:", self.t_fit_max_spin)
        calib_layout.addLayout(held_out_form)

        self.btn_calibrate = QPushButton("Fit asig/ta to Aftershocks")
        self.btn_calibrate.clicked.connect(self._run_calibration)
        calib_layout.addWidget(self.btn_calibrate)

        self.btn_validate = QPushButton("Validate Forecast Against Catalog")
        self.btn_validate.clicked.connect(self._run_validation)
        calib_layout.addWidget(self.btn_validate)

        self.calib_status_label = QLabel("")
        self.calib_status_label.setWordWrap(True)
        calib_layout.addWidget(self.calib_status_label)

        left.addWidget(calib_group)

        self.cache_checkbox = QCheckBox("Reuse cached CFF volume if available (recommended)")
        self.cache_checkbox.setChecked(True)
        left.addWidget(self.cache_checkbox)

        # ── Near-field exclusion toggle (2026-08-23) ─────────────────────
        # Previously exclude_near_field was hardcoded True at every call
        # site (forecast_from_cff_volume, calibrate_rate_state,
        # cff_field_stats, the histogram plot) with no way to turn it
        # off short of editing code. Made a real UI toggle because the
        # near-field mask was suspected of causing a display artifact
        # that turned out to persist even with the mask applied -- the
        # only way to confirm/rule that out from the UI is to be able to
        # switch it off and compare, rather than trusting the suspicion
        # either way. Default stays True (unchanged behavior) since the
        # near-field singularity itself (see PROJECT_HANDOVER_ADDENDUM_
        # 2026-08-14_near_field_threshold_derivation.md) is real
        # regardless of whether it explains any particular display bug.
        self.exclude_near_field_checkbox = QCheckBox(
            "Exclude near-field cells (Okada/DC3D singularities within "
            "~10 km of the fault trace)")
        self.exclude_near_field_checkbox.setChecked(True)
        left.addWidget(self.exclude_near_field_checkbox)
        near_field_note = QLabel(
            "Applies to the forecast, calibration/validation, ΔCFF field "
            "stats, and the ΔCFF histogram view -- all four stay in sync "
            "with this one toggle. Turning it off does NOT remove the "
            "near-field singularity itself, only the masking of it; if a "
            "display issue persists with this off exactly as it did with "
            "it on, the near-field mask is not the cause and the issue is "
            "elsewhere (e.g. the plot layout -- see Cross-Section tab's "
            "layout override for the same escape hatch on an automated-"
            "layout failure).")
        near_field_note.setWordWrap(True)
        near_field_note.setStyleSheet("color: gray; font-size: 10px;")
        left.addWidget(near_field_note)

        # ── Run controls ─────────────────────────────────────────────────
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run Forecast")
        self.btn_run.clicked.connect(self._run)
        run_row.addWidget(self.btn_run)
        self.btn_cancel_run = QPushButton("Cancel")
        self.btn_cancel_run.setEnabled(False)
        self.btn_cancel_run.clicked.connect(self._cancel_run)
        run_row.addWidget(self.btn_cancel_run)
        left.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        left.addWidget(self.status_label)
        left.addStretch(1)

        # ── Results (right column) ──────────────────────────────────────
        results_row = QHBoxLayout()
        self.radio_rate = QRadioButton("Rate")
        self.radio_cumulative = QRadioButton("Cumulative")
        self.radio_amplification = QRadioButton("Amplification (R/R\u2080)")
        self.radio_amplification.setToolTip(
            "Whole-region total_rate()/r0, plotted vs t. Only meaningful in "
            "\"Whole-region time series\" view (r0 is a single region-wide "
            "number) -- in \"Spatial snapshot\" view this falls back to the "
            "Rate map instead.")
        self.radio_rate.setChecked(True)
        self.radio_rate.toggled.connect(self._replot)
        self.radio_cumulative.toggled.connect(self._replot)
        self.radio_amplification.toggled.connect(self._replot)
        # Explicit QButtonGroup rather than relying on same-parent auto-
        # exclusive grouping (the default for QRadioButton) -- with two
        # logically-independent radio sets (Quantity, View) both ending
        # up parented under the same right_widget once their layouts are
        # installed, an implicit single auto-exclusive group across ALL
        # of them would make choosing a View option silently clobber the
        # Quantity selection (and vice versa). Discovered while adding
        # this third Quantity option; fixed for the pre-existing pair
        # too rather than left ambiguous. See PROJECT_HANDOVER_ADDENDUM_
        # 2026-08-22c for the full note.
        self._quantity_button_group = QButtonGroup(self)
        for b in (self.radio_rate, self.radio_cumulative, self.radio_amplification):
            self._quantity_button_group.addButton(b)
        results_row.addWidget(QLabel("Quantity:"))
        results_row.addWidget(self.radio_rate)
        results_row.addWidget(self.radio_cumulative)
        results_row.addWidget(self.radio_amplification)
        results_row.addSpacing(16)
        self.radio_view_series = QRadioButton("Whole-region time series")
        self.radio_view_map = QRadioButton("Spatial snapshot")
        self.radio_view_hist = QRadioButton("\u0394CFF field histogram")
        self.radio_view_timeline = QRadioButton("Catalog timeline (QA/QC)")
        self.radio_view_timeline.setToolTip(
            "Real catalog events vs. time-since-mainshock, with the "
            "calibration/validation windows drawn on top -- diagnostic "
            "view, not a forecast/CFF plot. Needs a loaded catalog and a "
            "filled-in mainshock time (both from the Calibration group "
            "below), not a run forecast.")
        self.radio_view_series.setChecked(True)
        self.radio_view_series.toggled.connect(self._on_view_toggled)
        self.radio_view_map.toggled.connect(self._on_view_toggled)
        self.radio_view_hist.toggled.connect(self._on_view_toggled)
        self.radio_view_timeline.toggled.connect(self._on_view_toggled)
        self._view_button_group = QButtonGroup(self)
        for b in (self.radio_view_series, self.radio_view_map, self.radio_view_hist,
                  self.radio_view_timeline):
            self._view_button_group.addButton(b)
        results_row.addWidget(QLabel("View:"))
        results_row.addWidget(self.radio_view_series)
        results_row.addWidget(self.radio_view_map)
        results_row.addWidget(self.radio_view_hist)
        results_row.addWidget(self.radio_view_timeline)
        results_row.addStretch(1)
        self.btn_save_plot = QPushButton("Save Plot As…")
        self.btn_save_plot.clicked.connect(self._save_plot)
        self.btn_save_plot.setEnabled(False)
        results_row.addWidget(self.btn_save_plot)
        right.addLayout(results_row)

        # Snapshot controls -- only meaningful (and only enabled) in
        # "Spatial snapshot" view; kept in their own row rather than
        # folded into results_row so results_row doesn't get crowded
        # in "time series" view, where these two widgets do nothing.
        snapshot_row = QHBoxLayout()
        snapshot_row.addWidget(QLabel("Depth slice:"))
        self.map_depth_combo = QComboBox()
        self.map_depth_combo.setEnabled(False)
        self.map_depth_combo.currentIndexChanged.connect(self._replot)
        snapshot_row.addWidget(self.map_depth_combo)
        snapshot_row.addSpacing(12)
        snapshot_row.addWidget(QLabel("Time step:"))
        self.map_time_spin = QSpinBox()
        self.map_time_spin.setRange(0, 0)
        self.map_time_spin.setEnabled(False)
        self.map_time_spin.valueChanged.connect(self._replot)
        snapshot_row.addWidget(self.map_time_spin)
        self.map_time_label = QLabel("")
        snapshot_row.addWidget(self.map_time_label)
        snapshot_row.addStretch(1)
        right.addLayout(snapshot_row)

        self.plot_widget = PlotWidget()
        right.addWidget(self.plot_widget, 1)

        # Raster export -- only meaningful for the "Spatial snapshot"
        # view (a time-series line plot has no spatial grid to
        # rasterize), so enabled/disabled by _on_view_toggled() exactly
        # like the depth/time selectors above. Reuses the SAME
        # depth/time selection those controls hold, rather than adding
        # a second selector -- "export what you're currently looking
        # at" (plus, optionally, every other time step at that depth).
        raster_row = QHBoxLayout()
        self.raster_all_times_checkbox = QCheckBox("All time steps (multiband GeoTIFF)")
        self.raster_all_times_checkbox.setEnabled(False)
        self.raster_all_times_checkbox.setToolTip(
            "Unchecked: export only the currently-selected time step. "
            "Checked: export every forecast time step at the currently-"
            "selected depth slice as one multi-band GeoTIFF (one band "
            "per time step) -- step through bands in QGIS's own band "
            "selector, or open a specific band's statistics/description "
            "to see which forecast time it is.")
        raster_row.addWidget(self.raster_all_times_checkbox)
        raster_row.addStretch(1)
        self.btn_raster_add_to_project = QPushButton("🗺️ Add Snapshot to QGIS Project")
        self.btn_raster_add_to_project.setEnabled(False)
        self.btn_raster_add_to_project.clicked.connect(self._raster_add_to_project)
        raster_row.addWidget(self.btn_raster_add_to_project)
        self.btn_raster_save_file = QPushButton("🗺️ Save Snapshot As GeoTIFF…")
        self.btn_raster_save_file.setEnabled(False)
        self.btn_raster_save_file.clicked.connect(self._raster_save_to_file)
        raster_row.addWidget(self.btn_raster_save_file)
        right.addLayout(raster_row)

        export_row = QHBoxLayout()
        self.btn_export_report = QPushButton("📄 Export Report (.txt/.pdf)…")
        self.btn_export_report.setEnabled(False)
        self.btn_export_report.clicked.connect(self._export_report)
        export_row.addWidget(self.btn_export_report)
        self.btn_export_csv = QPushButton("💾 Export Time Series (.csv)…")
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(self._export_csv)
        export_row.addWidget(self.btn_export_csv)
        export_row.addStretch(1)
        right.addLayout(export_row)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_close)
        right.addLayout(btn_row)

    # ── Inputs ───────────────────────────────────────────────────────────

    def _resolve_depths_km(self):
        text = self.depth_edit.text().strip()
        try:
            depths = [float(x.strip()) for x in text.split(",") if x.strip()]
        except ValueError:
            raise ValueError("Depths must be a comma-separated list of numbers, e.g. 0, 5, 10, 15")
        if not depths:
            raise ValueError("Depths list is empty.")
        return depths

    def _volume_grid(self):
        n_lon = min(self.grid.n_lon, _MAX_VOLUME_GRID_POINTS_PER_AXIS)
        n_lat = min(self.grid.n_lat, _MAX_VOLUME_GRID_POINTS_PER_AXIS)
        return GridParameters(lon_min=self.grid.lon_min, lon_max=self.grid.lon_max,
                              lat_min=self.grid.lat_min, lat_max=self.grid.lat_max,
                              depth_km=0.0, n_lon=n_lon, n_lat=n_lat)

    def _resolve_params(self):
        tdotr = self.tdotr_spin.value() if self.tdotr_checkbox.isChecked() else None
        return RateStateParams(r0=self.r0_spin.value(), asig=self.asig_spin.value(),
                               ta=self.ta_spin.value(), tdotr=tdotr,
                               time_unit=self.time_unit_combo.currentText())

    def _resolve_ts(self):
        t0 = self.t0_spin.value()
        t_max = self.t_max_spin.value()
        if t_max <= t0:
            raise ValueError("t_max must be greater than t0.")
        return np.linspace(t0, t_max, self.n_t_spin.value())

    def _resolve_mainshock_epoch(self):
        """Thin wrapper around _resolve_epoch_from_text() for the one
        field (mainshock origin time) every calibration handler needs;
        the background-rate window fields reuse _resolve_epoch_from_text()
        directly since they don't share a single fixed widget."""
        return self._resolve_epoch_from_text(
            self.mainshock_time_edit.text(), "mainshock origin time")

    def _resolve_mag_min(self):
        return self.mag_cutoff_spin.value() if self.mag_cutoff_checkbox.isChecked() else None

    # ── Settings capture/restore (2026-08-24) ───────────────────────────
    #
    # This dialog is reconstructed FRESH (new RateStateForecastDialog
    # instance) every time main_dialog.open_rate_state_forecast_action()
    # runs -- unlike self.xs_config, there was previously no persistent
    # object anywhere holding this dialog's last-used configuration, so
    # every re-open silently reset every field to its hardcoded default
    # and there was nothing for ui.project_io's "Save Setup" to capture.
    # get_settings()/apply_settings() are a plain-dict snapshot of every
    # CONFIGURATION widget (not results, not the loaded catalog itself --
    # see module docstring's "not meant to..." framing elsewhere in this
    # project for why imported datasets stay out of the native JSON setup
    # file the same way fault-derived focal_events/eq catalogs already do).
    # main_dialog.py calls get_settings() when this dialog closes (regardless
    # of accept/reject -- it's "current widget state", not a result) and
    # apply_settings() right after constructing the next instance, and
    # threads the same dict through project_io.build_setup_dict/
    # apply_setup_dict so it round-trips through a saved setup file too.
    def get_settings(self):
        """Serializable snapshot of every configuration widget in this
        dialog. See apply_settings() for the inverse."""
        return {
            "cff_mode": "optimal" if self.radio_cff_optimal.isChecked() else "fixed",
            "depths_km": self.depth_edit.text(),
            "time_unit": self.time_unit_combo.currentText(),
            "r0": self.r0_spin.value(),
            "asig": self.asig_spin.value(),
            "ta": self.ta_spin.value(),
            "tdotr_override": self.tdotr_checkbox.isChecked(),
            "tdotr": self.tdotr_spin.value(),
            "t0": self.t0_spin.value(),
            "t_max": self.t_max_spin.value(),
            "n_t": self.n_t_spin.value(),
            "label_mc": self.mc_checkbox.isChecked(),
            "mc": self.mc_spin.value(),
            "mainshock_time": self.mainshock_time_edit.text(),
            "mag_cutoff_enabled": self.mag_cutoff_checkbox.isChecked(),
            "mag_cutoff": self.mag_cutoff_spin.value(),
            "bg_start": self.bg_start_edit.text(),
            "bg_end": self.bg_end_edit.text(),
            "fit_r0": self.fit_r0_checkbox.isChecked(),
            "held_out_enabled": self.t_fit_max_checkbox.isChecked(),
            "t_fit_max": self.t_fit_max_spin.value(),
            "use_cache": self.cache_checkbox.isChecked(),
            "exclude_near_field": self.exclude_near_field_checkbox.isChecked(),
        }

    def apply_settings(self, s):
        """Inverse of get_settings(). Missing/unknown keys are ignored
        (forward/backward-compatible with older saved setups), matching
        project_io's own .get()-based tolerance elsewhere."""
        if not s:
            return
        if "cff_mode" in s:
            if s["cff_mode"] == "optimal" and self.radio_cff_optimal.isEnabled():
                self.radio_cff_optimal.setChecked(True)
            else:
                self.radio_cff_fixed.setChecked(True)
        if "depths_km" in s: self.depth_edit.setText(s["depths_km"])
        if "time_unit" in s:
            idx = self.time_unit_combo.findText(s["time_unit"])
            if idx >= 0:
                self.time_unit_combo.setCurrentIndex(idx)
        if "r0" in s: self.r0_spin.setValue(s["r0"])
        if "asig" in s: self.asig_spin.setValue(s["asig"])
        if "ta" in s: self.ta_spin.setValue(s["ta"])
        if "tdotr_override" in s: self.tdotr_checkbox.setChecked(s["tdotr_override"])
        if "tdotr" in s: self.tdotr_spin.setValue(s["tdotr"])
        if "t0" in s: self.t0_spin.setValue(s["t0"])
        if "t_max" in s: self.t_max_spin.setValue(s["t_max"])
        if "n_t" in s: self.n_t_spin.setValue(s["n_t"])
        if "label_mc" in s: self.mc_checkbox.setChecked(s["label_mc"])
        if "mc" in s: self.mc_spin.setValue(s["mc"])
        if "mainshock_time" in s: self.mainshock_time_edit.setText(s["mainshock_time"])
        if "mag_cutoff_enabled" in s: self.mag_cutoff_checkbox.setChecked(s["mag_cutoff_enabled"])
        if "mag_cutoff" in s: self.mag_cutoff_spin.setValue(s["mag_cutoff"])
        if "bg_start" in s: self.bg_start_edit.setText(s["bg_start"])
        if "bg_end" in s: self.bg_end_edit.setText(s["bg_end"])
        if "fit_r0" in s: self.fit_r0_checkbox.setChecked(s["fit_r0"])
        if "held_out_enabled" in s: self.t_fit_max_checkbox.setChecked(s["held_out_enabled"])
        if "t_fit_max" in s: self.t_fit_max_spin.setValue(s["t_fit_max"])
        if "use_cache" in s: self.cache_checkbox.setChecked(s["use_cache"])
        if "exclude_near_field" in s: self.exclude_near_field_checkbox.setChecked(s["exclude_near_field"])

    # ── Run / cancel ─────────────────────────────────────────────────────

    def _run(self):
        if self.radio_cff_optimal.isChecked() and self.regional is None:
            QMessageBox.warning(self, "No regional stress",
                                "Configure a regional stress tensor in the "
                                "Optimal-Plane ΔCFF tab first.")
            return
        try:
            depths_km = self._resolve_depths_km()
            ts = self._resolve_ts()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid input", str(e))
            return

        params = self._resolve_params()
        t0 = self.t0_spin.value()

        self.btn_run.setEnabled(False)
        self.btn_cancel_run.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        cff_mode_label = "optimally-oriented plane" if self.radio_cff_optimal.isChecked() else "specified receiver fault"
        self.status_label.setText(
            f"Building {len(depths_km)}-slice CFF volume ({cff_mode_label}), then "
            f"computing the forecast over {len(ts)} time steps…")

        self._last_meta = dict(
            cff_mode_label=cff_mode_label, depths_km=depths_km,
            grid_shape=None,  # filled in on completion, once the volume shape is known
            time_unit=self.time_unit_combo.currentText(), stress_unit="MPa",
            mc=(self.mc_spin.value() if self.mc_checkbox.isChecked() else None),
            exclude_near_field=self.exclude_near_field_checkbox.isChecked(),
        )

        self._worker = RateStateForecastWorker(
            self.sources, self.receiver, self.elastic, self._volume_grid(), depths_km,
            ts, t0, params, use_cache=self.cache_checkbox.isChecked(),
            cff_mode=("optimal" if self.radio_cff_optimal.isChecked() else "fixed"),
            regional=self.regional, friction=self.friction,
            exclude_near_field=self.exclude_near_field_checkbox.isChecked())
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._on_thread_finished)
        self._worker.start()

    def _cancel_run(self):
        if self._worker is not None:
            self._worker.request_cancel()
            self.status_label.setText("Cancelling…")
            self.btn_cancel_run.setEnabled(False)

    def _on_thread_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_cancel_run.setEnabled(False)
        self.progress_bar.setVisible(False)

    def _on_worker_error(self, message):
        self.status_label.setText(f"Error: {message}")
        QMessageBox.critical(self, "Forecast failed", message)

    def _on_finished(self, forecast):
        self._last_forecast = forecast
        self._last_volume = self._worker.volume if self._worker is not None else None
        # A fresh forecast run means a new ts/t0/grid AND a new
        # self._last_volume -- any previous calibration/validation was
        # computed against the OLD run and would silently mismatch
        # (build_observed_time_series binds to a specific ts array).
        # Clear it rather than let a stale overlay or "validated" status
        # survive a re-run with different settings.
        #
        # self._last_background MUST be cleared here too (2026-08-23
        # fix -- previously it wasn't). background_rate_from_catalog()
        # is region-restricted by self._last_volume at the time it's
        # computed; if the person computes a background rate, then
        # (re-)runs a forecast -- which swaps in a new, differently-
        # scoped self._last_volume -- a stale background silently
        # persisted here would get paired against calibration counts
        # from the NEW region while itself reflecting the OLD (or no)
        # region restriction. Both background_rate_from_catalog() and
        # calibrate_rate_state()/build_observed_time_series() already
        # take the SAME self._last_volume/mag_min at the point they're
        # each called (see core.rate_state_calibration module
        # docstring) -- the mismatch was never in the filtering logic
        # itself, only in this dialog letting an old background result
        # outlive the volume it was measured against.
        self._last_calibration = None
        self._last_observed = None
        self._last_validation = None
        self._last_background = None
        self.calib_status_label.setText(
            "Forecast (re)built -- previous calibration/validation/background "
            "rate cleared (the modeled region may have changed). Recompute the "
            "background rate, then the Calibration group above can fit against "
            "this run's ΔCFF volume.")
        self._last_meta["grid_shape"] = forecast.rate.shape[:3]
        n_points = int(np.prod(forecast.rate.shape[:3]))
        n_nan = int(np.sum(np.isnan(forecast.rate[..., 0])))
        nan_note = f", {n_nan} outside the volume (excluded from totals)" if n_nan else ""

        # ΔCFF field statistics on the just-built volume's own cff_mpa
        # array -- added 2026-08-22c, see core.cff_volume.cff_field_stats()
        # and _format_cff_stats_section() in core.rate_state_report. Session
        # 3's own worked example (a 1,841x background-rate spike) is
        # exactly the kind of number these stats explain at a glance,
        # before the person has to guess whether it's a real signal or a
        # units/parameter mismatch -- surfaced here in the status line
        # (always visible) rather than only in the exported report, and
        # additionally available as its own plot view ("ΔCFF field
        # histogram" radio, see _replot()).
        stats_note = ""
        if self._last_volume is not None:
            try:
                self._last_cff_stats = cff_field_stats(
                    self._last_volume,
                    exclude_near_field=self.exclude_near_field_checkbox.isChecked())
                s = self._last_cff_stats
                nf_note = (f" ({s.n_near_field_excluded} near-field cell(s) excluded)"
                          if s.n_near_field_excluded else "")
                stats_note = (f" ΔCFF field: mean={s.mean:.3g}, P5/P95="
                              f"{s.p5:.3g}/{s.p95:.3g} MPa, "
                              f"+{100*s.frac_positive:.0f}%/-{100*s.frac_negative:.0f}%."
                              f"{nf_note}")
            except ValueError:
                self._last_cff_stats = None  # every point NaN -- nothing to summarize
        else:
            self._last_cff_stats = None

        peak_amp = np.nanmax(forecast.amplification())
        amp_note = f" Peak amplification: {peak_amp:g}x background." if np.isfinite(peak_amp) else ""
        self.status_label.setText(
            f"Done. {n_points} grid point(s){nan_note}. "
            f"Peak total rate {np.nanmax(forecast.total_rate()):g} at "
            f"t={forecast.ts[int(np.nanargmax(forecast.total_rate()))]:g}."
            f"{amp_note}{stats_note}")
        self.btn_save_plot.setEnabled(True)
        self.btn_export_report.setEnabled(True)
        self.btn_export_csv.setEnabled(True)

        # Populate the spatial-snapshot controls now that depths_km/ts
        # are known -- done unconditionally (not only when the "Spatial
        # snapshot" view is selected) so switching to that view later
        # doesn't need a second populate step, just an enable step (see
        # _on_view_toggled()).
        self.map_depth_combo.blockSignals(True)
        self.map_depth_combo.clear()
        for i, d in enumerate(forecast.depths_km):
            self.map_depth_combo.addItem(f"{d:g} km", userData=i)
        self.map_depth_combo.blockSignals(False)
        self.map_time_spin.blockSignals(True)
        self.map_time_spin.setRange(0, max(0, len(forecast.ts) - 1))
        self.map_time_spin.setValue(0)
        self.map_time_spin.blockSignals(False)
        self._update_map_time_label()
        self._on_view_toggled()  # sets enabled state to match current view

        self._replot()

    def _on_view_toggled(self, *_args):
        """
        Enables/disables the spatial-snapshot controls (depth combo,
        time-step spinbox, raster export row) to match the currently-
        selected view radio. *_args absorbs the bool QRadioButton.
        toggled() passes -- called both as a signal slot (with that
        bool) and directly from _on_finished() (with no args), same
        "accept and ignore whatever args a caller happens to pass"
        shape as _replot() already has via
        radio_rate.toggled.connect(self._replot).
        """
        is_map = self.radio_view_map.isChecked()
        have_forecast = self._last_forecast is not None
        enabled = is_map and have_forecast
        self.map_depth_combo.setEnabled(enabled)
        self.map_time_spin.setEnabled(enabled)
        self.raster_all_times_checkbox.setEnabled(enabled)
        self.btn_raster_add_to_project.setEnabled(enabled)
        self.btn_raster_save_file.setEnabled(enabled)
        self._replot()

    def _update_map_time_label(self):
        if self._last_forecast is None:
            self.map_time_label.setText("")
            return
        idx = self.map_time_spin.value()
        ts = self._last_forecast.ts
        if 0 <= idx < len(ts):
            self.map_time_label.setText(f"(t={ts[idx]:g})")

    def _replot(self, *_args):
        # Catalog timeline (QA/QC) is deliberately checked BEFORE the
        # "no forecast yet" guard below -- unlike every other view, it
        # only needs a loaded catalog + a resolved mainshock time, not a
        # run forecast (t0/t_max are used if a forecast happens to be
        # available, purely to draw the window-overlay lines, but their
        # absence isn't a reason to show nothing).
        if self.radio_view_timeline.isChecked():
            self._replot_catalog_timeline()
            return
        if self._last_forecast is None:
            return
        if self.radio_rate.isChecked():
            mode = "rate"
        elif self.radio_cumulative.isChecked():
            mode = "cumulative"
        else:
            mode = "amplification"

        if self.radio_view_map.isChecked():
            self._update_map_time_label()
            fc = self._last_forecast
            depth_idx = self.map_depth_combo.currentData()
            time_idx = self.map_time_spin.value()
            if depth_idx is None or not (0 <= time_idx < len(fc.ts)):
                return
            map_mode = "cumulative" if mode == "cumulative" else "rate"
            values = fc.rate if map_mode == "rate" else fc.cumulative
            values_2d = values[depth_idx, :, :, time_idx]
            lon2d, lat2d = np.meshgrid(fc.lons, fc.lats)
            depth_label = f"{fc.depths_km[depth_idx]:g} km"
            time_label = f"t={fc.ts[time_idx]:g}"
            self.plot_widget.plot_rate_state_map(
                lon2d, lat2d, values_2d, mode=map_mode,
                depth_label=depth_label, time_label=time_label)
        elif self.radio_view_hist.isChecked():
            if self._last_volume is None or self._last_cff_stats is None:
                return
            depth_note = f"{len(self._last_volume.depths_km)} depth slice(s)"
            # Same near-field exclusion cff_field_stats() already applied
            # when building self._last_cff_stats -- now driven by the
            # exclude_near_field_checkbox toggle (2026-08-23) rather than
            # a hardcoded True -- mask the histogram's own input array
            # the SAME way, or the bars would show a different filtering
            # than the stats text box next to them, a silent mismatch
            # between the two halves of the same plot. See
            # core.cff_volume.apply_near_field_mask's docstring for why
            # near-field cells need excluding at all (2026-08-22 smoke-
            # test fix), and this dialog's near_field_note for why the
            # toggle exists (2026-08-23).
            hist_values = apply_near_field_mask(
                self._last_volume.cff_mpa, self._last_volume.near_field_mask,
                exclude_near_field=self.exclude_near_field_checkbox.isChecked())
            self.plot_widget.plot_cff_field_histogram(
                hist_values, self._last_cff_stats, title_extra=depth_note)
        else:
            observed = self._last_observed if mode == "cumulative" else None
            self.plot_widget.plot_rate_state_forecast(self._last_forecast, mode=mode, observed=observed)

    def _replot_catalog_timeline(self):
        """
        Build and draw the "Catalog timeline (QA/QC)" view -- see
        core.rate_state_calibration.build_catalog_timeline()'s
        docstring for what this is for. Degrades gracefully (status text
        explaining why, no plot) rather than raising a dialog, since
        this fires on every radio toggle/window-field edit, not just on
        an explicit button click.
        """
        if not self.eq_events:
            self.calib_status_label.setText(
                "Catalog timeline: load a catalog first (Load Catalog\u2026 "
                "button below).")
            return
        try:
            mainshock_epoch = self._resolve_mainshock_epoch()
        except ValueError as e:
            self.calib_status_label.setText(f"Catalog timeline: {e}")
            return
        eq_array = events_to_eq_array(self.eq_events)
        t0 = self._last_forecast.t0 if self._last_forecast is not None else None
        t_max = self._last_forecast.ts[-1] if self._last_forecast is not None else None
        t_fit_max = self.t_fit_max_spin.value() if self.t_fit_max_checkbox.isChecked() else None
        timeline = rsc.build_catalog_timeline(
            eq_array, mainshock_epoch, time_unit=self.time_unit_combo.currentText(),
            mag_min=self._resolve_mag_min(), volume=self._last_volume,
            t0=t0, t_max=t_max, t_fit_max=t_fit_max)
        region_note = ("region-restricted to last forecast's volume" if self._last_volume is not None
                       else "no region restriction -- run a forecast first to restrict this")
        self.plot_widget.plot_eq_catalog_timeline(timeline, title_extra=region_note)
        if timeline.n_total > 0 and timeline.n_with_time == 0:
            self.calib_status_label.setText(
                f"Catalog timeline: {timeline.n_total} event(s) loaded but NONE have "
                "a parseable time -- check the Time column mapping in Load Catalog "
                "(a GeoPackage DateTime-typed column vs. a plain text column that "
                "merely looks like a date can behave differently; see this dialog's "
                "docs/addendum for the exact failure mode this was hit by before).")



    # ── Calibration handlers (real earthquake catalog) ─────────────────
    # See core.rate_state_calibration's module docstring for the physics/
    # statistics design (background rate, per-bin log-space asig/ta fit,
    # N-test/RMSE/R2/Poisson-LL validation) this UI layer drives.

    def _update_calib_time_unit_note(self, *_args):
        unit = self.time_unit_combo.currentText()
        self.calib_time_unit_note.setText(
            f"Catalog timestamps below will be converted to \"{unit}\" -- the "
            "same unit r0/ta/t0/t_max use (set in the Rate-and-state "
            "parameters group above). Change it there, not here.")

    def _load_catalog(self):
        dlg = EQCatalogImportDialog(self)
        if dlg.exec_() == dlg.Accepted:
            self.eq_events = dlg.imported_events
            n_with_time = sum(1 for e in self.eq_events if e.epoch_s is not None)
            n_with_mag = sum(1 for e in self.eq_events if e.magnitude is not None)
            self.catalog_status_label.setText(
                f"{len(self.eq_events)} event(s) loaded "
                f"({n_with_time} with a usable time, {n_with_mag} with a magnitude).")

    def _compute_background_rate(self):
        if not self.eq_events:
            QMessageBox.warning(self, "No catalog", "Load an earthquake catalog first.")
            return
        try:
            t_start = self._resolve_epoch_from_text(self.bg_start_edit.text(), "window start")
            t_end = self._resolve_epoch_from_text(self.bg_end_edit.text(), "window end")
        except ValueError as e:
            QMessageBox.warning(self, "Invalid input", str(e))
            return

        eq_array = events_to_eq_array(self.eq_events)
        try:
            result = rsc.background_rate_from_catalog(
                eq_array, t_start, t_end,
                time_unit=self.time_unit_combo.currentText(),
                mag_min=self._resolve_mag_min(), volume=self._last_volume)
        except ValueError as e:
            QMessageBox.warning(self, "Could not compute background rate", str(e))
            return

        self._last_background = result
        self.r0_spin.setValue(result.r0)
        region_note = " (restricted to the last forecast's modeled region)" if result.region_restricted else " (no region restriction -- run a forecast first to restrict to the modeled volume)"
        self.calib_status_label.setText(
            f"Background rate: {result.n_events} event(s) / {result.duration:g} "
            f"{result.time_unit} = {result.r0:g} events/{result.time_unit}{region_note}. "
            "r0 above has been updated.")

    def _run_calibration(self):
        if self._last_volume is None:
            QMessageBox.warning(self, "No forecast yet",
                                "Run a forecast first -- calibration needs the ΔCFF "
                                "volume it builds.")
            return
        if not self.eq_events:
            QMessageBox.warning(self, "No catalog", "Load an earthquake catalog first.")
            return
        fit_r0 = self.fit_r0_checkbox.isChecked()
        if not fit_r0 and self._last_background is None:
            QMessageBox.warning(
                self, "No background rate",
                "Compute a background rate from the catalog first (or check "
                "\"Also fit r0\" to fit it along with asig/ta instead).")
            return

        try:
            mainshock_epoch = self._resolve_mainshock_epoch()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid input", str(e))
            return
        try:
            rsc.assert_consistent_time_unit(self._last_forecast, self.time_unit_combo.currentText())
        except ValueError as e:
            QMessageBox.warning(self, "Time unit mismatch", str(e))
            return

        t_fit_max = self.t_fit_max_spin.value() if self.t_fit_max_checkbox.isChecked() else None
        eq_array = events_to_eq_array(self.eq_events)
        try:
            calib = rsc.calibrate_rate_state(
                self._last_volume, eq_array, mainshock_epoch,
                self._last_forecast.t0, self._last_forecast.ts,
                time_unit=self.time_unit_combo.currentText(),
                r0=(None if fit_r0 else self._last_background.r0), fit_r0=fit_r0,
                asig0=self.asig_spin.value(), ta0=self.ta_spin.value(),
                mag_min=self._resolve_mag_min(), t_fit_max=t_fit_max,
                exclude_near_field=self.exclude_near_field_checkbox.isChecked())
        except ValueError as e:
            QMessageBox.warning(self, "Calibration failed", str(e))
            return

        self._last_calibration = calib
        self._last_observed = calib.observed
        self.r0_spin.setValue(calib.params.r0)
        self.asig_spin.setValue(calib.params.asig)
        self.ta_spin.setValue(calib.params.ta)
        status = ("converged" if calib.success else "DID NOT CONVERGE -- inspect the fit before trusting it")
        # ta warning (added 2026-08-24, see
        # PROJECT_HANDOVER_ADDENDUM_2026-08-24b_calibration_ta_runaway_fix.md):
        # calibrate_rate_state() now bounds ta rather than letting it run
        # away to an arbitrary value (previously observed: 78231.59), but
        # a fit that NEEDS that ceiling (ta_at_bound) or whose ta isn't
        # otherwise well-constrained (well_determined=False) still means
        # the catalog/window doesn't actually pin down ta -- the dialog
        # surfaces that here instead of silently loading a falsely-precise
        # number into the spinbox above.
        if calib.ta_at_bound:
            ta_note = (f" ⚠ ta hit the fit's upper bound "
                       f"({calib.ta_bounds[1]:.3g} {self.time_unit_combo.currentText()}) -- "
                       "the data doesn't constrain how slowly the sequence decays; treat "
                       "this ta as a lower bound on the true value, not a point estimate.")
        elif not calib.well_determined:
            ta_note = (" ⚠ ta is poorly constrained by this fit "
                       f"(1σ ≈ {calib.ta_stderr:.3g}" if calib.ta_stderr is not None else
                       " ⚠ ta is poorly constrained by this fit (uncertainty not computable")
            ta_note += (f" {self.time_unit_combo.currentText()}, vs fitted "
                       f"{calib.params.ta:.3g}) -- more/longer aftershock data would help."
                       if calib.ta_stderr is not None else
                       ") -- more/longer aftershock data would help.")
        else:
            ta_note = ""
        self.calib_status_label.setText(
            f"Calibration {status} ({calib.message}). Fitted against "
            f"{calib.observed.n_events_total} observed event(s), "
            f"{calib.n_fit_points} of {len(calib.observed.ts)} time points. "
            f"r0/asig/ta above have been updated -- click Run Forecast to "
            "recompute the forecast with these values." + ta_note)

    def _run_validation(self):
        if self._last_forecast is None:
            QMessageBox.warning(self, "No forecast yet", "Run a forecast first.")
            return
        if not self.eq_events:
            QMessageBox.warning(self, "No catalog", "Load an earthquake catalog first.")
            return
        try:
            mainshock_epoch = self._resolve_mainshock_epoch()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid input", str(e))
            return
        try:
            rsc.assert_consistent_time_unit(self._last_forecast, self.time_unit_combo.currentText())
        except ValueError as e:
            QMessageBox.warning(self, "Time unit mismatch", str(e))
            return

        eq_array = events_to_eq_array(self.eq_events)
        observed = rsc.build_observed_time_series(
            eq_array, mainshock_epoch, self._last_forecast.ts, self._last_forecast.t0,
            time_unit=self.time_unit_combo.currentText(),
            mag_min=self._resolve_mag_min(), volume=self._last_volume)
        t_min = (self.t_fit_max_spin.value() if
                self.t_fit_max_checkbox.isChecked() and self._last_calibration is not None
                else None)
        try:
            score = rsc.score_forecast(
                observed, self._last_forecast.total_cumulative(), t_min=t_min)
        except ValueError as e:
            QMessageBox.warning(self, "Validation failed", str(e))
            return

        self._last_observed = observed
        self._last_validation = score
        held_out_note = f" (held-out window t>={t_min:g})" if t_min is not None else ""
        self.calib_status_label.setText(
            f"Validation{held_out_note}: N-ratio (obs/pred)={score.n_ratio:g}, "
            f"N-test p={score.n_test_p_value:g}, R2={score.r2:g}, "
            f"RMSE={score.rmse:g}, Poisson LL={score.poisson_loglik:g}. "
            "See the exported report for the full breakdown.")
        self._replot()

    def _resolve_epoch_from_text(self, text, field_label):
        text = text.strip()
        if not text:
            raise ValueError(f"Enter the {field_label} first.")
        dt = parse_datetime(text)
        if dt is None:
            raise ValueError(f"Could not parse {field_label}: {text!r}")
        return dt.replace(tzinfo=timezone.utc).timestamp()

    def _save_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if path:
            self.plot_widget.save_to_file(path)

    # ── Raster export ────────────────────────────────────────────────────

    def _resolve_snapshot_selection(self):
        """
        Shared by both raster export handlers and _replot(): resolves
        the current mode/depth-index/lon2d/lat2d from the dialog's own
        widget state, common to both the "single time step" and "all
        time steps" export paths. Returns None (after showing a
        warning) if the selection isn't currently valid rather than
        raising, since both callers are UI click handlers, not
        internal logic that should propagate an exception.
        """
        if self._last_forecast is None:
            return None
        fc = self._last_forecast
        depth_idx = self.map_depth_combo.currentData()
        if depth_idx is None:
            QMessageBox.warning(self, "No depth slice selected",
                                "Choose a depth slice first.")
            return None
        # "Amplification" has no per-cell raster equivalent here (it's a
        # whole-region-total quantity, r0 isn't a per-cell number) --
        # falls back to "rate", same fallback plot_rate_state_map() gets
        # in _replot()'s spatial-snapshot view.
        mode = "cumulative" if self.radio_cumulative.isChecked() else "rate"
        lon2d, lat2d = np.meshgrid(fc.lons, fc.lats)
        return dict(fc=fc, depth_idx=depth_idx, mode=mode, lon2d=lon2d, lat2d=lat2d)

    def _build_snapshot_arrays(self):
        """
        Returns either (values_2d, layer_name, band_descriptions=None)
        for a single time step, or (values_3d, layer_name,
        band_descriptions) for "all time steps" -- values_3d has shape
        (n_t, n_lat, n_lon), i.e. already transposed from
        forecast.rate/.cumulative's own (n_depth, n_lat, n_lon, n_t)
        layout into the (n_bands, n_lat, n_lon) shape
        core.raster_utils.write_geotiff_multiband() expects. Returns
        None if the current selection isn't valid (see
        _resolve_snapshot_selection()).
        """
        sel = self._resolve_snapshot_selection()
        if sel is None:
            return None
        fc, depth_idx, mode = sel["fc"], sel["depth_idx"], sel["mode"]
        values = fc.rate if mode == "rate" else fc.cumulative
        depth_label = f"{fc.depths_km[depth_idx]:g}km"
        quantity_label = "rate" if mode == "rate" else "cumulative"

        if self.raster_all_times_checkbox.isChecked():
            values_3d = np.moveaxis(values[depth_idx], -1, 0)   # (n_lat,n_lon,n_t) -> (n_t,n_lat,n_lon)
            band_descriptions = [f"t={t:g}" for t in fc.ts]
            layer_name = f"Rate-State {quantity_label} {depth_label} (all t)"
            return sel, values_3d, layer_name, band_descriptions
        else:
            time_idx = self.map_time_spin.value()
            if not (0 <= time_idx < len(fc.ts)):
                QMessageBox.warning(self, "No time step selected",
                                    "Choose a time step first.")
                return None
            values_2d = values[depth_idx, :, :, time_idx]
            layer_name = f"Rate-State {quantity_label} {depth_label} t={fc.ts[time_idx]:g}"
            return sel, values_2d, layer_name, None

    def _raster_add_to_project(self):
        built = self._build_snapshot_arrays()
        if built is None:
            return
        sel, values, layer_name, band_descriptions = built
        from ..utils.raster_utils import add_raster_to_project, add_multiband_raster_to_project
        if band_descriptions is not None:
            layer = add_multiband_raster_to_project(
                sel["lon2d"], sel["lat2d"], values, layer_name=layer_name,
                band_descriptions=band_descriptions, sequential=True)
        else:
            layer = add_raster_to_project(
                sel["lon2d"], sel["lat2d"], values, layer_name=layer_name,
                sequential=True)
        if layer is not None:
            self.status_label.setText(f"Raster layer added to project: {layer_name}")
        else:
            self.status_label.setText("Failed to add raster layer.")

    def _raster_save_to_file(self):
        built = self._build_snapshot_arrays()
        if built is None:
            return
        sel, values, layer_name, band_descriptions = built
        path, _ = QFileDialog.getSaveFileName(self, "Save GeoTIFF", "", "GeoTIFF (*.tif)")
        if not path:
            return
        if not path.lower().endswith((".tif", ".tiff")):
            path += ".tif"
        from ..utils.raster_utils import (
            write_geotiff, write_geotiff_multiband, load_raster_layer)
        if band_descriptions is not None:
            write_geotiff_multiband(path, sel["lon2d"], sel["lat2d"], values,
                                    band_descriptions=band_descriptions)
        else:
            write_geotiff(path, sel["lon2d"], sel["lat2d"], values)
        load_raster_layer(path, layer_name, sequential=True)
        self.status_label.setText(f"Raster saved and added: {os.path.basename(path)}")

    # ── Report / data export ────────────────────────────────────────────

    def _default_export_stem(self):
        """
        Builds a run-identifying filename stem (no extension) from
        self._last_meta, e.g. "rate_state_forecast_specfault_nfON_
        20260824-142317" -- added so exported reports/CSVs from
        different runs (different ΔCFF mode, near-field exclusion
        state) don't collide under one generic name and have to be
        told apart after the fact by re-deriving settings from their
        content. Falls back to the old generic name if self._last_meta
        is somehow unavailable (shouldn't happen given the callers'
        own `self._last_forecast is None` guard, but kept defensive).
        """
        import datetime
        meta = self._last_meta or {}
        mode = meta.get("cff_mode_label", "")
        mode_tag = "optfault" if "optimal" in mode else ("specfault" if mode else "run")
        nf = meta.get("exclude_near_field")
        nf_tag = "" if nf is None else ("_nfON" if nf else "_nfOFF")
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"rate_state_forecast_{mode_tag}{nf_tag}_{stamp}"

    def _export_report(self):
        if self._last_forecast is None:
            return
        default_name = self._default_export_stem() + ".txt"
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Report", default_name,
            "Text (*.txt);;PDF (*.pdf)")
        if not path:
            return
        report_text = build_rate_state_report(
            self._last_forecast, self._last_meta,
            background=self._last_background, calibration=self._last_calibration,
            validation=self._last_validation, cff_stats=self._last_cff_stats)

        is_pdf = path.lower().endswith(".pdf") or "PDF" in selected_filter
        if is_pdf:
            if not path.lower().endswith(".pdf"):
                path += ".pdf"
            try:
                write_report_pdf(path, report_text, plot_figure=self.plot_widget.figure)
            except Exception as e:
                QMessageBox.critical(self, "Export failed", f"Could not write PDF report:\n{e}")
                return
        else:
            if not path.lower().endswith(".txt"):
                path += ".txt"
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(report_text)
            except Exception as e:
                QMessageBox.critical(self, "Export failed", f"Could not write report:\n{e}")
                return
        self.status_label.setText(f"Report exported to {path}")

    def _export_csv(self):
        if self._last_forecast is None:
            return
        default_name = self._default_export_stem() + ".csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Time Series", default_name, "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        rows = build_rate_state_csv_rows(self._last_forecast)
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", f"Could not write CSV:\n{e}")
            return
        self.status_label.setText(f"Time series exported to {path}")
