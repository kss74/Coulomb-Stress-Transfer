# -*- coding: utf-8 -*-
"""
Dialog for inverting a regional (tectonic) stress tensor from an already-
imported focal-mechanism catalog (core.stress_inversion, wrapping ILSI --
Beaucé, van der Hilst & Campillo 2022, BSSA), for use as the Optimal
Faults tab's `RegionalStress` input instead of a hand-entered
strike/plunge guess. See PROJECT_HANDOVER_ADDENDUM_2026-08-25_
stress_inversion_wrapper.md for the physics-wrapper session this
dialog was scoped from.

Deliberately does NOT duplicate catalog-picking UI: it takes the
already-imported `focal_events` list (Focal Mechanisms tab's
`self.focal_events`, populated by FocalMechanismImportDialog) as a
constructor argument, exactly the same convention AftershockMCTestDialog
and RateStateForecastDialog already use for (sources, receiver, elastic,
grid) -- this dialog is meant to be opened from the main dialog once a
focal-mechanism catalog already exists, not used standalone.

Layout follows the same left(scrollable settings)/right(plot) split as
AftershockMCTestDialog/RateStateForecastDialog, for the same documented
reason: a single stacked column starves the plot of vertical space once
configure_resizable_dialog()'s screen-height cap kicks in.

Two SEPARATE optional dependencies are involved, and both are surfaced
independently rather than conflated:
  - ILSI itself (core.stress_inversion.check_ilsi()) -- required for the
    inversion to run at all; the Invert button is disabled without it.
  - mplstereonet (core.stress_inversion.check_mplstereonet()) -- needed
    only for the stereonet plot (ui/plot_widget.py's
    plot_stress_inversion_stereonet()); the inversion still runs and its
    numeric results are still shown/usable without it, just with no plot.
"""

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, QLabel,
    QCheckBox, QDoubleSpinBox, QSpinBox, QGroupBox, QMessageBox,
    QFileDialog, QWidget
)
from qgis.PyQt.QtCore import QThread, pyqtSignal

from ..core import stress_inversion as si
from .dialog_utils import configure_resizable_dialog, wrap_widget_in_scroll_area
from .plot_widget import PlotWidget


class StressInversionWorker(QThread):
    """
    Runs invert_regional_stress() and, optionally,
    bootstrap_regional_stress() off the UI thread. The bootstrap in
    particular re-runs the full inversion once per resampling
    (n_resamplings, default 200) and can take real wall-clock time for
    larger catalogs -- same reasoning as every other *Worker(QThread) in
    this project (AftershockMCWorker, rate_state_dialog's worker, etc.):
    a real, input-dependent cost that shouldn't block the UI.
    """
    finished_ok = pyqtSignal(object)   # dict: {"result": ..., "boot": ... or None}
    error = pyqtSignal(str)

    def __init__(self, events, friction_coefficient, variable_shear,
                 n_stress_iter, n_random_selections, n_averaging,
                 signed_instability, weighted, run_bootstrap, n_resamplings):
        super().__init__()
        self.events = events
        self.friction_coefficient = friction_coefficient
        self.variable_shear = variable_shear
        self.n_stress_iter = n_stress_iter
        self.n_random_selections = n_random_selections
        self.n_averaging = n_averaging
        self.signed_instability = signed_instability
        self.weighted = weighted
        self.run_bootstrap = run_bootstrap
        self.n_resamplings = n_resamplings

    def run(self):
        try:
            result = si.invert_regional_stress(
                self.events,
                friction_coefficient=self.friction_coefficient,
                variable_shear=self.variable_shear,
                n_stress_iter=self.n_stress_iter,
                n_random_selections=self.n_random_selections,
                n_averaging=self.n_averaging,
                signed_instability=self.signed_instability,
                weighted=self.weighted,
            )
            boot = None
            if self.run_bootstrap:
                boot = si.bootstrap_regional_stress(
                    self.events, result, n_resamplings=self.n_resamplings,
                    n_stress_iter=self.n_stress_iter,
                    variable_shear=self.variable_shear,
                    signed_instability=self.signed_instability,
                    weighted=self.weighted)
            self.finished_ok.emit({"result": result, "boot": boot})
        except Exception as e:
            self.error.emit(str(e))


class StressInversionDialog(QDialog):
    """
    Configure and run an ILSI regional-stress inversion against
    `focal_events`, preview the result on a stereonet, and hand the
    resulting `optimal_plane.RegionalStress` back to the caller via
    `get_result()` once the user clicks "Use as Regional Stress".
    """

    def __init__(self, focal_events, default_friction=0.6, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Invert Regional Stress from Focal Mechanisms")
        configure_resizable_dialog(self, 980, 700, min_width=700, min_height=480)

        self.focal_events = focal_events or []
        self._worker = None
        self._result = None
        self._boot = None
        self._regional_stress = None
        self._used_friction = None

        ok, _msg = si.check_ilsi()
        self._ilsi_ok = ok

        # Left (settings, scrollable) / right (results, plot) split --
        # same reasoning as AftershockMCTestDialog/RateStateForecastDialog.
        outer_layout = QHBoxLayout(self)

        left_inner = QWidget()
        left = QVBoxLayout(left_inner)
        left_scroll = wrap_widget_in_scroll_area(left_inner, self)
        left_scroll.setMaximumWidth(420)
        outer_layout.addWidget(left_scroll, 0)

        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        outer_layout.addWidget(right_widget, 1)

        left.addWidget(QLabel(
            "<b>Regional stress from focal mechanisms</b><br><i>Inverts "
            f"the {len(self.focal_events)} imported focal mechanism(s) "
            "(Focal Mechanisms tab) for the regional stress tensor's "
            "ORIENTATION and SHAPE RATIO using ILSI's instability-"
            "parameter method (Beaucé, van der Hilst &amp; Campillo "
            "2022) -- resolves each event's nodal-plane ambiguity and "
            "the stress tensor simultaneously.</i>"))

        self.ilsi_status = QLabel(("✅ " if ok else "⚠️ ") + _msg)
        self.ilsi_status.setWordWrap(True)
        self.ilsi_status.setStyleSheet(
            f"color: {'darkgreen' if ok else 'darkorange'};")
        left.addWidget(self.ilsi_status)

        if len(self.focal_events) < 4:
            few_events_note = QLabel(
                f"⚠️ Only {len(self.focal_events)} focal mechanism(s) "
                "imported -- the inversion needs at least 4 (a 5-parameter "
                "deviatoric stress tensor). Import more via the Focal "
                "Mechanisms tab first.")
            few_events_note.setWordWrap(True)
            few_events_note.setStyleSheet("color: darkorange;")
            left.addWidget(few_events_note)

        # ── Inversion parameters ────────────────────────────────────────
        inv_group = QGroupBox("Inversion parameters")
        inv_form = QFormLayout(inv_group)

        self.friction_auto = QCheckBox("Auto (grid-search for max instability)")
        self.friction_auto.toggled.connect(self._on_friction_auto_toggled)
        inv_form.addRow(self.friction_auto)

        self.friction_spin = QDoubleSpinBox()
        self.friction_spin.setRange(0.0, 1.0)
        self.friction_spin.setDecimals(2)
        self.friction_spin.setSingleStep(0.05)
        self.friction_spin.setValue(default_friction)
        inv_form.addRow("Friction coefficient:", self.friction_spin)

        self.variable_shear = QCheckBox(
            "Variable shear (Beaucé et al. 2022, recommended)")
        self.variable_shear.setChecked(True)
        inv_form.addRow(self.variable_shear)

        self.signed_instability = QCheckBox("Signed instability")
        inv_form.addRow(self.signed_instability)

        self.weighted = QCheckBox("Weight events (ILSI default: unweighted)")
        inv_form.addRow(self.weighted)

        self.n_stress_iter = QSpinBox()
        self.n_stress_iter.setRange(1, 1000)
        self.n_stress_iter.setValue(10)
        inv_form.addRow("Stress iterations:", self.n_stress_iter)

        self.n_random_selections = QSpinBox()
        self.n_random_selections.setRange(1, 1000)
        self.n_random_selections.setValue(20)
        inv_form.addRow("Random nodal-plane selections:", self.n_random_selections)

        self.n_averaging = QSpinBox()
        self.n_averaging.setRange(1, 100)
        self.n_averaging.setValue(1)
        inv_form.addRow("Repeat & average:", self.n_averaging)

        left.addWidget(inv_group)

        # ── Bootstrap ────────────────────────────────────────────────────
        boot_group = QGroupBox("Bootstrap uncertainty (optional)")
        boot_form = QFormLayout(boot_group)
        self.run_bootstrap = QCheckBox("Compute bootstrap confidence cloud")
        boot_form.addRow(self.run_bootstrap)
        self.n_resamplings = QSpinBox()
        self.n_resamplings.setRange(10, 5000)
        self.n_resamplings.setValue(200)
        boot_form.addRow("Resamplings:", self.n_resamplings)
        boot_note = QLabel(
            "Re-runs the inversion once per resampling -- can take real "
            "time for larger catalogs/resampling counts; runs in the "
            "background so the UI stays responsive.")
        boot_note.setWordWrap(True)
        boot_note.setStyleSheet("color: gray; font-size: 10px;")
        boot_form.addRow(boot_note)
        left.addWidget(boot_group)

        # ── Stress magnitude -- REQUIRED, not recoverable from the fit ───
        mag_group = QGroupBox("Stress magnitude (required -- not from the inversion)")
        mag_form = QFormLayout(mag_group)
        mag_warn = QLabel(
            "⚠️ Focal-mechanism inversion (Wallace-Bott assumption) is "
            "scale-free -- it CANNOT recover absolute stress magnitude. "
            "Supply a differential-stress value from an independent "
            "source (borehole data, published regional studies, or a "
            "sensitivity sweep).")
        mag_warn.setWordWrap(True)
        mag_warn.setStyleSheet("color: darkorange;")
        mag_form.addRow(mag_warn)

        self.diff_stress = QDoubleSpinBox()
        self.diff_stress.setRange(0.01, 10000.0)
        self.diff_stress.setDecimals(2)
        self.diff_stress.setValue(100.0)
        self.diff_stress.setSuffix(" bar")
        mag_form.addRow("Differential stress (S1-S3):", self.diff_stress)

        self.iso_offset = QDoubleSpinBox()
        self.iso_offset.setRange(-10000.0, 10000.0)
        self.iso_offset.setDecimals(2)
        self.iso_offset.setValue(0.0)
        self.iso_offset.setSuffix(" bar")
        mag_form.addRow("Isotropic offset ((S1+S3)/2):", self.iso_offset)

        left.addWidget(mag_group)

        # ── Run controls ────────────────────────────────────────────────
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Invert")
        self.btn_run.clicked.connect(self._run)
        run_row.addWidget(self.btn_run)
        left.addLayout(run_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        left.addWidget(self.status_label)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(
            "background-color: rgba(0,0,0,15); padding: 8px; border-radius: 4px;")
        left.addWidget(self.result_label)
        left.addStretch(1)

        # ── Right column: plot + buttons ────────────────────────────────
        plot_row = QHBoxLayout()
        plot_row.addStretch(1)
        self.btn_save_plot = QPushButton("Save Plot As…")
        self.btn_save_plot.setEnabled(False)
        self.btn_save_plot.clicked.connect(self._save_plot)
        plot_row.addWidget(self.btn_save_plot)
        right.addLayout(plot_row)

        self.plot_widget = PlotWidget()
        right.addWidget(self.plot_widget, 1)

        mplst_ok, mplst_msg = si.check_mplstereonet()
        if not mplst_ok:
            mplst_note = QLabel("⚠️ " + mplst_msg)
            mplst_note.setWordWrap(True)
            mplst_note.setStyleSheet("color: darkorange; font-size: 10px;")
            right.addWidget(mplst_note)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_use = QPushButton("Use as Regional Stress")
        self.btn_use.setEnabled(False)
        self.btn_use.clicked.connect(self._use_as_regional_stress)
        btn_row.addWidget(self.btn_use)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_close)
        right.addLayout(btn_row)

        self._set_running(False)

    # ── UI state helpers ────────────────────────────────────────────────

    def _on_friction_auto_toggled(self, checked):
        self.friction_spin.setEnabled(not checked)

    def _set_running(self, running):
        can_run = self._ilsi_ok and len(self.focal_events) >= 4
        self.btn_run.setEnabled((not running) and can_run)
        for w in (self.friction_auto, self.variable_shear,
                  self.signed_instability, self.weighted, self.n_stress_iter,
                  self.n_random_selections, self.n_averaging,
                  self.run_bootstrap, self.n_resamplings):
            w.setEnabled(not running)
        if running:
            self.friction_spin.setEnabled(False)
        else:
            self._on_friction_auto_toggled(self.friction_auto.isChecked())

    # ── Run ──────────────────────────────────────────────────────────────

    def _run(self):
        if len(self.focal_events) < 4:
            QMessageBox.warning(self, "Too few events",
                "Need at least 4 focal mechanisms to invert.")
            return
        self._set_running(True)
        self.status_label.setStyleSheet("")
        self.status_label.setText("Running inversion…")
        self.result_label.setText("")
        self.btn_use.setEnabled(False)

        friction = None if self.friction_auto.isChecked() else self.friction_spin.value()
        self._worker = StressInversionWorker(
            events=self.focal_events,
            friction_coefficient=friction,
            variable_shear=self.variable_shear.isChecked(),
            n_stress_iter=self.n_stress_iter.value(),
            n_random_selections=self.n_random_selections.value(),
            n_averaging=self.n_averaging.value(),
            signed_instability=self.signed_instability.isChecked(),
            weighted=self.weighted.isChecked(),
            run_bootstrap=self.run_bootstrap.isChecked(),
            n_resamplings=self.n_resamplings.value(),
        )
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, payload):
        self._set_running(False)
        self._result = payload["result"]
        self._boot = payload["boot"]
        r = self._result

        lines = [
            f"n_events = {r['n_events']}",
            f"shape ratio R = {r['shape_ratio']:.3f}",
            f"friction (used/found) = {r['friction_coefficient']:.3f}",
        ]
        for name in ("S1", "S2", "S3"):
            strike, plunge = r["axes_end"][name]
            lines.append(f"{name}: strike={strike:.1f}°, plunge={plunge:.1f}°")
        self.result_label.setText("<br>".join(lines))
        self.status_label.setText(
            "Inversion complete." + (" Bootstrap complete." if self._boot else ""))
        self.btn_use.setEnabled(True)
        self.btn_save_plot.setEnabled(True)

        boot_axes = self._boot["boot_axes_end"] if self._boot else None
        self.plot_widget.plot_stress_inversion_stereonet(r, boot_axes_end=boot_axes)

    def _on_error(self, msg):
        self._set_running(False)
        self.status_label.setText(f"Error: {msg}")
        self.status_label.setStyleSheet("color: crimson;")

    # ── Save / apply ─────────────────────────────────────────────────────

    def _save_plot(self):
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save Plot", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.plot_widget.save_to_file(path)

    def _use_as_regional_stress(self):
        if self._result is None:
            return
        try:
            self._regional_stress = si.regional_stress_from_inversion(
                self._result, differential_stress_bars=self.diff_stress.value(),
                isotropic_offset_bars=self.iso_offset.value())
        except ValueError as e:
            QMessageBox.warning(self, "Cannot build regional stress", str(e))
            return
        self._used_friction = self._result["friction_coefficient"]
        self.accept()

    def get_result(self):
        """
        Returns (RegionalStress, friction_coefficient) if the user ran an
        inversion and clicked "Use as Regional Stress", else (None, None).
        friction_coefficient is ILSI's own used/found value -- the caller
        may optionally also apply it to the Optimal Faults tab's
        independent friction-override field (rs_friction), which is NOT
        the same value ILSI's instability-parameter search uses
        internally unless the caller chooses to synchronize them.
        """
        return self._regional_stress, self._used_friction
