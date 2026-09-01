# -*- coding: utf-8 -*-
"""
Post-run validation view for SlipInversionDialog: lets the user load an
independently-known REFERENCE fault-patch model (a published finite-
fault/geodetic-inversion solution, a synthetic checkerboard test, or
any other already-formatted fault-patch table -- reuses
FaultTableImportDialog UNCHANGED as the file/column-mapping picker,
exactly as core.slip_inversion_validation's docstring's SCOPE note
requires) and plots it side-by-side against the inversion's OWN solved
slip distribution, via PlotWidget.plot_slip_inversion_validation().

Opened as a separate lightweight top-level dialog from
SlipInversionDialog's new "🧪 Compare against reference slip model…"
button, rather than adding a plot panel directly into that already-
tall/scrolling dialog -- same "small standalone dialog for an optional
view" pattern as e.g. cross_section_config_dialog.py.

SCOPE (mirrors core.slip_inversion_validation's own docstring): only
the single-fault case is supported. A "Group" inversion (several fault
rows solved jointly) would need one correctly-ordered reference file
PER segment, aligned to that segment's own patch order -- not
implemented; SlipInversionDialog only offers this button when
`not self._is_group`.
"""

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QMessageBox
)

from .dialog_utils import configure_resizable_dialog
from .plot_widget import PlotWidget
from .fault_table_import_dialog import FaultTableImportDialog
from ..core.slip_inversion_validation import compare_slip_to_reference


class SlipInversionValidationDialog(QDialog):
    """
    diag         : the diagnostics dict from run_slip_inversion_group()
                   for a single-segment (non-group) run -- i.e. exactly
                   SlipInversionDialog._run_diag from that case.
    n_length,
    n_width      : the fault's own Subdiv.(L)/Subdiv.(W).
    mu           : elastic shear modulus (Pa) -- same value the
                   inversion itself used.
    patch_area_m2: this fault's own per-patch area (m^2) -- uniform
                   across the grid since run_slip_inversion() subdivides
                   via FaultParameters.subdivide() (equal-area patches),
                   so a single float is correct here (see
                   core.slip_inversion_validation.compare_slip_to_reference's
                   patch_areas_m2 scalar-broadcast support).
    fault_name   : for the window title / plot title only.
    """

    def __init__(self, parent, diag, n_length, n_width, mu, patch_area_m2,
                fault_name=""):
        super().__init__(parent)
        self._diag = diag
        self._n_length = n_length
        self._n_width = n_width
        self._mu = mu
        self._patch_area_m2 = patch_area_m2
        self._fault_name = fault_name
        self._result = None  # SlipValidationResult, set once a reference model is loaded

        title = "Validate slip inversion against a reference model"
        if fault_name:
            title += f" — {fault_name}"
        self.setWindowTitle(title)
        configure_resizable_dialog(self, 900, 620, min_width=520, min_height=420)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"This fault's inversion grid is <b>{n_length}×{n_width} = "
            f"{n_length * n_width} patches</b>. Load a reference slip "
            f"model on the EXACT SAME patch grid, in the same flat "
            f"(down-dip row, along-strike column) order — e.g. a "
            f"published finite-fault solution for this earthquake, or "
            f"a synthetic checkerboard test — to compare it against "
            f"this run's own solved slip distribution."))

        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("📂 Load reference fault-patch table…")
        self.btn_load.clicked.connect(self._load_reference)
        btn_row.addWidget(self.btn_load)
        self.btn_save_fig = QPushButton("💾 Save figure…")
        self.btn_save_fig.setEnabled(False)
        self.btn_save_fig.clicked.connect(self._save_figure)
        btn_row.addWidget(self.btn_save_fig)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.status_label = QLabel("No reference model loaded yet.")
        layout.addWidget(self.status_label)

        self.plot_widget = PlotWidget(self)
        layout.addWidget(self.plot_widget)

        close_row = QHBoxLayout()
        close_row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        close_row.addWidget(btn_close)
        layout.addLayout(close_row)

    def _load_reference(self):
        dlg = FaultTableImportDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        imported_rows = dlg.imported_rows
        if not imported_rows:
            QMessageBox.warning(self, "No rows imported",
                                "The reference file produced no usable rows.")
            return

        try:
            result = compare_slip_to_reference(
                self._diag, imported_rows, self._n_length, self._n_width,
                self._mu, patch_areas_m2=self._patch_area_m2)
        except ValueError as e:
            QMessageBox.critical(self, "Cannot compare", str(e))
            return

        self._result = result
        self.plot_widget.plot_slip_inversion_validation(
            result, title_extra=self._fault_name)
        self.btn_save_fig.setEnabled(True)

        corr_text = "n/a" if result.magnitude_corr != result.magnitude_corr else f"{result.magnitude_corr:.3f}"
        self.status_label.setText(
            f"Loaded {len(imported_rows)} reference patch(es). "
            f"Slip-magnitude correlation = {corr_text}, "
            f"RMS = {result.magnitude_rms_m:.3f} m, "
            f"reference Mw = {result.true_mw:.2f}, "
            f"achieved Mw = {result.achieved_mw:.2f}."
            + (f" ({'; '.join(result.errors)})" if result.errors else ""))

    def _save_figure(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure", f"{self._safe_name()}_slip_validation.png",
            "PNG image (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path:
            return
        try:
            self.plot_widget.save_to_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return
        QMessageBox.information(self, "Saved", f"Figure saved:\n{path}")

    def _safe_name(self):
        return "".join(c if c.isalnum() else "_" for c in (self._fault_name or "fault"))
