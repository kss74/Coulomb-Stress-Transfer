# -*- coding: utf-8 -*-
"""
Dialog for the aftershock/ΔCFF Monte Carlo null test (core.aftershock_mc_test).

Workflow: import an EQ catalog (or reuse one already imported this
session), configure depth slices + MC parameters, run in the
background (building the 3D CFF volume is a real DC3D cost -- one
Okada evaluation per depth slice -- so this follows main_dialog.py's
own ComputeWorker(QThread) pattern rather than blocking the UI), view
the observed-vs-null plot, save it.

Constructed with the same (sources, receiver, elastic, grid) tuple
main_dialog.py's ComputeWorker already takes, plus optional
(regional, friction) for the optimally-oriented-plane CFF mode -- same
RegionalStress object and friction value the existing "Optimal-Plane
ΔCFF" tab already gathers, passed straight through by main_dialog.py's
launch action rather than duplicated here. This dialog is meant to be
opened from the main dialog once a source/receiver/grid configuration
exists, not used standalone.

Layout (2026-08-18 fix): left/right split rather than one long
vertically-stacked column. Previously every settings group (catalog,
ΔCFF mode, depth slices, MC parameters, run controls, progress/status)
sat directly above the plot in a single QVBoxLayout; on any display
short enough that configure_resizable_dialog()'s 0.88x-screen-height
cap kicks in, those groups' combined minimumSizeHint left little to no
room for the plot even with stretch=1, and plot_widget.py's fixed-
aspect axes then rendered into a short, wide canvas where the title
and "best"-placed legend visually collided (see also the legend fix in
plot_widget.py itself). Now follows main_dialog.py's own left(controls)
/ right(plot) pattern: all the settings groups are wrapped in a
QScrollArea on the left (dialog_utils.wrap_in_scroll_area) so they can
never blow out the dialog's minimum size, and the plot + its controls
occupy their own column on the right where they compete for space with
nothing but the run/save buttons above them.
"""

import os
import tempfile
import csv

import numpy as np
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QPushButton,
    QLabel, QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar,
    QGroupBox, QMessageBox, QFileDialog, QRadioButton, QButtonGroup, QWidget
)
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal

from ..core.okada_engine import GridParameters
from ..core.optimal_plane import RegionalStress
from ..core.cff_volume import (
    build_cff_volume, auto_depth_slices, cff_volume_cache_key,
    save_cff_volume, load_cff_volume,
)
from ..core.aftershock_mc_test import observed_vs_null, MCTestCancelled
from ..core.aftershock_mc_report import (
    build_aftershock_mc_report, build_aftershock_mc_csv_rows, write_report_pdf,
)
from .eq_catalog_import_dialog import EQCatalogImportDialog
from .dialog_utils import configure_resizable_dialog, wrap_widget_in_scroll_area
from .plot_widget import PlotWidget


# Volume grid resolution is independent of (and normally much coarser
# than) the main display grid's resolution -- this dialog only
# INTERPOLATES ΔCFF at scattered points (observed + random), it never
# rasterizes the volume for display, so a high-resolution volume grid
# buys nothing but DC3D cost. Decision made unilaterally: cap at 40x40
# per slice regardless of what the caller's display grid uses, unless
# the caller's grid is already coarser (then just use that).
_MAX_VOLUME_GRID_POINTS_PER_AXIS = 40

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "coulomb_stress_transfer_cff_volume_cache")


class AftershockMCWorker(QThread):
    """
    Background worker: (1) build/load-from-cache the 3D CFF volume,
    (2) run observed_vs_null() against it. Progress is split 0-70%
    volume build (dominant cost: one DC3D evaluation per depth slice)
    and 70-100% Monte Carlo (pure numpy interpolation, fast once the
    volume exists) -- an approximate split, not measured, since the
    two phases have genuinely different and input-dependent costs;
    good enough for a progress bar that's mostly reassurance that
    something is happening, not a precise ETA.
    """
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(object)   # AftershockMCTestResult
    error = pyqtSignal(str)

    def __init__(self, sources, receiver, elastic, volume_grid, depths_km,
                 eq_array, x_max, n_thr, n_points, n_runs, depth_mode,
                 use_cache=True, cff_mode="fixed", regional=None, friction=None):
        super().__init__()
        self.sources = sources
        self.receiver = receiver
        self.elastic = elastic
        self.volume_grid = volume_grid
        self.depths_km = depths_km
        self.eq_array = eq_array
        self.x_max = x_max
        self.n_thr = n_thr
        self.n_points = n_points
        self.n_runs = n_runs
        self.depth_mode = depth_mode
        self.use_cache = use_cache
        self.cff_mode = cff_mode          # "fixed" or "optimal"
        self.regional = regional          # RegionalStress, required if cff_mode="optimal"
        self.friction = friction          # optional override; defaults to elastic.friction
        self._cancelled = False

    def request_cancel(self):
        self._cancelled = True

    def _cancel_check(self):
        return self._cancelled

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
                        pass  # cache write failure shouldn't fail the whole computation
            else:
                self.progress.emit(70)

            if self._cancelled:
                return

            result = observed_vs_null(
                volume, self.eq_array, x_max=self.x_max, n_thr=self.n_thr,
                n_points=self.n_points, n_runs=self.n_runs, depth_mode=self.depth_mode,
                progress_callback=lambda p: self.progress.emit(70 + int(0.3 * p)),
                cancel_check=self._cancel_check)

            self.finished_ok.emit(result)

        except MCTestCancelled:
            pass  # user-initiated; not an error
        except Exception as e:
            self.error.emit(str(e))


class AftershockMCTestDialog(QDialog):
    """
    sources/receiver/elastic/grid: same objects main_dialog.py's
    ComputeWorker already has on hand for the "cff" mode -- `grid`'s
    lon/lat extent defines the domain random null points are sampled
    within (its own .depth_km is irrelevant here, only extent+resolution
    matter, and resolution is capped -- see _MAX_VOLUME_GRID_POINTS_PER_AXIS).
    regional/friction: same RegionalStress object and friction value the
    "Optimal-Plane ΔCFF" tab already gathers -- pass None (the default)
    if no regional stress has been configured yet; the "optimally-
    oriented fault" radio option is simply disabled in that case rather
    than the dialog raising or silently falling back.
    """

    def __init__(self, sources, receiver, elastic, grid, regional=None, friction=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Aftershock / ΔCFF Monte Carlo Null Test")
        configure_resizable_dialog(self, 1080, 720, min_width=760, min_height=520)
        self.sources = sources
        self.receiver = receiver
        self.elastic = elastic
        self.grid = grid
        self.regional = regional      # RegionalStress or None -- enables the "optimal" CFF option
        self.friction = friction      # override; None -> elastic.friction default
        self.eq_events = []      # List[EQCatalogEvent]
        self.eq_array = []       # events_to_eq_array() output
        self._worker = None
        self._last_result = None

        # Left (settings, scrollable) / right (results, plot gets the
        # lion's share of vertical space) split -- see class/module
        # docstring for why this replaced one long stacked column.
        outer_layout = QHBoxLayout(self)

        left_inner = QWidget()
        left = QVBoxLayout(left_inner)
        left_scroll = wrap_widget_in_scroll_area(left_inner, self)
        left_scroll.setMaximumWidth(400)
        outer_layout.addWidget(left_scroll, 0)

        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        outer_layout.addWidget(right_widget, 1)

        # ── Catalog ──────────────────────────────────────────────────────
        catalog_group = QGroupBox("Earthquake catalog")
        catalog_layout = QHBoxLayout(catalog_group)
        self.catalog_status = QLabel("No catalog imported yet.")
        self.catalog_status.setWordWrap(True)
        catalog_layout.addWidget(self.catalog_status, 1)
        self.btn_import_catalog = QPushButton("Import Catalog…")
        self.btn_import_catalog.clicked.connect(self._import_catalog)
        catalog_layout.addWidget(self.btn_import_catalog)
        left.addWidget(catalog_group)

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
            "Optimally-oriented ΔCFF is generally the better choice for aftershock "
            "forecasting: real aftershocks occur on whichever plane is locally "
            "best-oriented for failure, not on one predetermined receiver -- this "
            "matches what most published aftershock-forecasting studies test "
            "against. Requires a regional stress tensor (Optimal-Plane ΔCFF tab).")
        cff_note.setWordWrap(True)
        cff_note.setStyleSheet("color: gray; font-size: 10px;")
        cff_mode_layout.addWidget(cff_note)
        if self.regional is None:
            self.radio_cff_optimal.setEnabled(False)
            self.radio_cff_optimal.setToolTip(
                "Configure a regional stress tensor in the Optimal-Plane ΔCFF tab first.")
        left.addWidget(cff_mode_group)

        # ── Depth slices ─────────────────────────────────────────────────
        depth_group = QGroupBox("Depth slices (3D CFF volume)")
        depth_form = QFormLayout(depth_group)

        self.radio_depth_auto = QRadioButton("Auto (from catalog depth range)")
        self.radio_depth_custom = QRadioButton("Custom list (km, comma-separated)")
        self.radio_depth_auto.setChecked(True)
        depth_radio_row = QHBoxLayout()
        depth_radio_row.addWidget(self.radio_depth_auto)
        depth_radio_row.addWidget(self.radio_depth_custom)
        depth_form.addRow(depth_radio_row)

        self.depth_custom_edit = QLineEdit("0, 5, 10, 15, 20")
        self.depth_custom_edit.setEnabled(False)
        self.radio_depth_custom.toggled.connect(self.depth_custom_edit.setEnabled)
        depth_form.addRow("Custom depths:", self.depth_custom_edit)

        res_note = QLabel(f"Volume grid resolution capped at "
                          f"{_MAX_VOLUME_GRID_POINTS_PER_AXIS}×{_MAX_VOLUME_GRID_POINTS_PER_AXIS} "
                          f"(interpolation only -- not rasterized for display).")
        res_note.setWordWrap(True)
        res_note.setStyleSheet("color: gray; font-size: 10px;")
        depth_form.addRow(res_note)

        left.addWidget(depth_group)

        # ── MC parameters ────────────────────────────────────────────────
        mc_group = QGroupBox("Monte Carlo null test parameters")
        mc_form = QFormLayout(mc_group)

        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(0.0001, 1000.0)
        self.x_max_spin.setDecimals(4)
        self.x_max_spin.setValue(0.3)
        self.x_max_spin.setSuffix(" MPa")
        mc_form.addRow("Max threshold:", self.x_max_spin)

        self.n_thr_spin = QSpinBox()
        self.n_thr_spin.setRange(2, 500)
        self.n_thr_spin.setValue(80)
        mc_form.addRow("Threshold resolution:", self.n_thr_spin)

        self.n_points_spin = QSpinBox()
        self.n_points_spin.setRange(10, 200000)
        self.n_points_spin.setValue(2000)
        mc_form.addRow("Random points per run (N):", self.n_points_spin)

        self.n_runs_spin = QSpinBox()
        self.n_runs_spin.setRange(1, 10000)
        self.n_runs_spin.setValue(100)
        mc_form.addRow("Monte Carlo runs (M):", self.n_runs_spin)

        self.depth_mode_combo = QComboBox()
        self.depth_mode_combo.addItem("Match observed EQ depth distribution", userData="match_eq_depth")
        self.depth_mode_combo.addItem("Uniform across depth range", userData="uniform")
        mc_form.addRow("Null-point depth sampling:", self.depth_mode_combo)

        self.cache_checkbox = QCheckBox("Reuse cached CFF volume if available (recommended)")
        self.cache_checkbox.setChecked(True)
        mc_form.addRow(self.cache_checkbox)

        left.addWidget(mc_group)

        # ── Run controls ─────────────────────────────────────────────────
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run Null Test")
        self.btn_run.clicked.connect(self._run)
        self.btn_run.setEnabled(False)
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
        self.radio_fraction = QRadioButton("Fraction")
        self.radio_count = QRadioButton("Count")
        self.radio_fraction.setChecked(True)
        self.radio_fraction.toggled.connect(self._replot)
        results_row.addWidget(QLabel("Display:"))
        results_row.addWidget(self.radio_fraction)
        results_row.addWidget(self.radio_count)
        results_row.addStretch(1)
        self.btn_save_plot = QPushButton("Save Plot As…")
        self.btn_save_plot.clicked.connect(self._save_plot)
        self.btn_save_plot.setEnabled(False)
        results_row.addWidget(self.btn_save_plot)
        right.addLayout(results_row)

        self.plot_widget = PlotWidget()
        right.addWidget(self.plot_widget, 1)

        # ── Export (report + raw data) ──────────────────────────────────
        export_row = QHBoxLayout()
        self.btn_export_report = QPushButton("📄 Export Report (.txt/.pdf)…")
        self.btn_export_report.setEnabled(False)
        self.btn_export_report.clicked.connect(self._export_report)
        export_row.addWidget(self.btn_export_report)
        self.btn_export_csv = QPushButton("💾 Export Plot Data (.csv)…")
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

    # ── Catalog import ──────────────────────────────────────────────────

    def _import_catalog(self):
        dlg = EQCatalogImportDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self.eq_events = dlg.imported_events
            self.eq_array = dlg.imported_array
            n = len(self.eq_events)
            n_with_time = sum(1 for e in self.eq_events if e.time is not None)
            self.catalog_status.setText(
                f"{n} event(s) imported ({n_with_time} with usable time). "
                f"Ready to run the null test.")
            self.btn_run.setEnabled(n > 0)

    # ── Depth slices ─────────────────────────────────────────────────────

    def _resolve_depths_km(self):
        if self.radio_depth_auto.isChecked():
            return auto_depth_slices(self.eq_array)
        text = self.depth_custom_edit.text().strip()
        try:
            depths = [float(x.strip()) for x in text.split(",") if x.strip()]
        except ValueError:
            raise ValueError("Custom depths must be a comma-separated list of numbers, e.g. 0, 5, 10, 15")
        if not depths:
            raise ValueError("Custom depths list is empty.")
        return depths

    def _volume_grid(self):
        n_lon = min(self.grid.n_lon, _MAX_VOLUME_GRID_POINTS_PER_AXIS)
        n_lat = min(self.grid.n_lat, _MAX_VOLUME_GRID_POINTS_PER_AXIS)
        return GridParameters(lon_min=self.grid.lon_min, lon_max=self.grid.lon_max,
                              lat_min=self.grid.lat_min, lat_max=self.grid.lat_max,
                              depth_km=0.0, n_lon=n_lon, n_lat=n_lat)

    # ── Settings capture/restore (2026-08-24) ───────────────────────────
    # See RateStateForecastDialog.get_settings()/apply_settings() for the
    # full rationale -- this dialog is reconstructed fresh on every
    # open_aftershock_mc_test_action() call the same way, so its
    # configuration was previously lost on close and unreachable from
    # ui.project_io's "Save Setup". Imported catalog data (self.eq_events/
    # self.eq_array) is deliberately NOT included here, consistent with
    # this project's existing setup-file scope (fault rows are saved,
    # but e.g. main_dialog's own focal_events catalog import is not) --
    # only the configuration widgets are.
    def get_settings(self):
        """Serializable snapshot of every configuration widget in this
        dialog. See apply_settings() for the inverse."""
        return {
            "cff_mode": "optimal" if self.radio_cff_optimal.isChecked() else "fixed",
            "depth_mode": "custom" if self.radio_depth_custom.isChecked() else "auto",
            "depths_km": self.depth_custom_edit.text(),
            "x_max": self.x_max_spin.value(),
            "n_thr": self.n_thr_spin.value(),
            "n_points": self.n_points_spin.value(),
            "n_runs": self.n_runs_spin.value(),
            "depth_sampling": self.depth_mode_combo.currentData(),
            "use_cache": self.cache_checkbox.isChecked(),
            "display_mode": "fraction" if self.radio_fraction.isChecked() else "count",
        }

    def apply_settings(self, s):
        """Inverse of get_settings(). Missing/unknown keys are ignored
        (forward/backward-compatible with older saved setups)."""
        if not s:
            return
        if "cff_mode" in s:
            if s["cff_mode"] == "optimal" and self.radio_cff_optimal.isEnabled():
                self.radio_cff_optimal.setChecked(True)
            else:
                self.radio_cff_fixed.setChecked(True)
        if "depth_mode" in s:
            if s["depth_mode"] == "custom":
                self.radio_depth_custom.setChecked(True)
            else:
                self.radio_depth_auto.setChecked(True)
        if "depths_km" in s: self.depth_custom_edit.setText(s["depths_km"])
        if "x_max" in s: self.x_max_spin.setValue(s["x_max"])
        if "n_thr" in s: self.n_thr_spin.setValue(s["n_thr"])
        if "n_points" in s: self.n_points_spin.setValue(s["n_points"])
        if "n_runs" in s: self.n_runs_spin.setValue(s["n_runs"])
        if "depth_sampling" in s:
            idx = self.depth_mode_combo.findData(s["depth_sampling"])
            if idx >= 0:
                self.depth_mode_combo.setCurrentIndex(idx)
        if "use_cache" in s: self.cache_checkbox.setChecked(s["use_cache"])
        if "display_mode" in s:
            if s["display_mode"] == "count":
                self.radio_count.setChecked(True)
            else:
                self.radio_fraction.setChecked(True)

    # ── Run / cancel ─────────────────────────────────────────────────────

    def _run(self):
        if not self.eq_array:
            QMessageBox.warning(self, "No catalog", "Import an earthquake catalog first.")
            return
        if self.radio_cff_optimal.isChecked() and self.regional is None:
            QMessageBox.warning(self, "No regional stress",
                                "Configure a regional stress tensor in the "
                                "Optimal-Plane ΔCFF tab first.")
            return
        try:
            depths_km = self._resolve_depths_km()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid depth slices", str(e))
            return

        self.btn_run.setEnabled(False)
        self.btn_cancel_run.setEnabled(True)
        self.btn_import_catalog.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        cff_mode_label = "optimally-oriented plane" if self.radio_cff_optimal.isChecked() else "specified receiver fault"
        self.status_label.setText(
            f"Building {len(depths_km)}-slice CFF volume ({cff_mode_label}), then running "
            f"{self.n_runs_spin.value()} Monte Carlo runs…")

        self._worker = AftershockMCWorker(
            self.sources, self.receiver, self.elastic, self._volume_grid(), depths_km,
            self.eq_array, self.x_max_spin.value(), self.n_thr_spin.value(),
            self.n_points_spin.value(), self.n_runs_spin.value(),
            self.depth_mode_combo.currentData(), use_cache=self.cache_checkbox.isChecked(),
            cff_mode=("optimal" if self.radio_cff_optimal.isChecked() else "fixed"),
            regional=self.regional, friction=self.friction)
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
        self.btn_run.setEnabled(bool(self.eq_array))
        self.btn_cancel_run.setEnabled(False)
        self.btn_import_catalog.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _on_worker_error(self, message):
        self.status_label.setText(f"Error: {message}")
        QMessageBox.critical(self, "Null test failed", message)

    def _on_finished(self, result):
        self._last_result = result
        n_ge_valid = result.observed.n_valid
        depth_note = ""
        if result.null.depth_mode_used != result.null.depth_mode_requested:
            depth_note = (f" (requested depth mode "
                          f"'{result.null.depth_mode_requested}' fell back to "
                          f"'{result.null.depth_mode_used}' -- no usable observed depths)")
        self.status_label.setText(
            f"Done. {n_ge_valid} observed events landed inside the CFF volume "
            f"(out of {len(self.eq_array)} imported). "
            f"Null test: N={result.null.n_points} × M={result.null.n_runs}{depth_note}.")
        self.btn_save_plot.setEnabled(True)
        self.btn_export_report.setEnabled(True)
        self.btn_export_csv.setEnabled(True)
        self._replot()

    def _replot(self):
        if self._last_result is None:
            return
        mode = "fraction" if self.radio_fraction.isChecked() else "count"
        self.plot_widget.plot_aftershock_mc_test(self._last_result, mode=mode)

    def _save_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if path:
            self.plot_widget.save_to_file(path)

    # ── Report / data export ────────────────────────────────────────────

    def _report_meta(self):
        """Assembles the UI-side context core.aftershock_mc_report needs
        but doesn't have on its own (it only sees the result object) --
        catalog size, ΔCFF mode label, depth slices, threshold sweep max.
        """
        cff_mode_label = ("optimally-oriented plane" if self.radio_cff_optimal.isChecked()
                          else "specified receiver fault")
        try:
            depths_km = self._resolve_depths_km()
        except ValueError:
            depths_km = None
        return dict(
            n_catalog_events=len(self.eq_array),
            cff_mode_label=cff_mode_label,
            depths_km=depths_km,
            x_max=self.x_max_spin.value(),
        )

    def _export_report(self):
        if self._last_result is None:
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Report", "aftershock_mc_report.txt",
            "Text (*.txt);;PDF (*.pdf)")
        if not path:
            return
        report_text = build_aftershock_mc_report(self._last_result, self._report_meta())

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
        if self._last_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot Data", "aftershock_mc_data.csv", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        rows = build_aftershock_mc_csv_rows(self._last_result)
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", f"Could not write CSV:\n{e}")
            return
        self.status_label.setText(f"Plot data exported to {path}")
