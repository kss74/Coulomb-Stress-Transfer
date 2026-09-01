# -*- coding: utf-8 -*-
"""
Point Calculator dialog (2026-09-01 addition).

Lets the user pick a set of observation points (CSV/TSV file or a QGIS
point layer already in the project), map columns onto lon/lat/elevation
(required) plus an optional observed East/North/Up displacement-or-slip
vector, then computes predicted stress (ΔCFF/shear/normal) and
predicted displacement at each point from the plugin's CURRENT source
faults -- and, where an observation was supplied, the residual against
it (the "validation against field-measured slip" use case).

Deliberately mirrors ui/eq_catalog_import_dialog.py's structure
(source selection / column mapping form / preview table) for the import
half, then adds a results table + "Add to QGIS as layer" / "Export
CSV" actions for the compute half -- so the workflow is familiar to
anyone who has already used the earthquake-catalog or focal-mechanism
importers in this plugin.
"""

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QPushButton,
    QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QRadioButton, QButtonGroup, QGroupBox, QMessageBox, QTabWidget, QWidget
)
from qgis.PyQt.QtCore import Qt

from ..core.point_calculation import (
    POSITION_FIELDS, OPTIONAL_FIELDS, FIELD_DISPLAY_NAMES,
    suggest_column_mapping, detect_point_mapping,
    build_points_from_mapped_rows, compute_point_results,
    results_to_csv_text, RESULT_COLUMNS,
)
from ..core.focal_mechanism_import import read_delimited_table
from ..core.observation_import import read_qgis_layer_table
from .dialog_utils import configure_resizable_dialog

NONE_LABEL = "(not mapped)"

# Subset of RESULT_COLUMNS shown in the on-screen results table (all of
# them are still written to CSV/the QGIS layer via RESULT_COLUMNS) --
# the raw Pa stress-tensor components are omitted here to keep the table
# readable; bar-unit CFF/shear/normal (this plugin's usual display unit)
# and the displacement/residual columns are what someone reviewing
# results at a glance actually wants.
DISPLAY_COLUMNS = [
    "label", "lon", "lat", "elev_m", "used_dc3d",
    "cff_bar", "shear_bar", "normal_bar",
    "pred_e_m", "pred_n_m", "pred_u_m", "pred_horiz_mag_m",
    "obs_e_m", "obs_n_m", "obs_u_m",
    "resid_e_m", "resid_n_m", "resid_u_m", "resid_3d_mag_m",
]
DISPLAY_COLUMN_LABELS = {
    "label": "Label", "lon": "Lon", "lat": "Lat", "elev_m": "Elev (m)",
    "used_dc3d": "DC3D?",
    "cff_bar": "ΔCFF (bar)", "shear_bar": "Shear (bar)", "normal_bar": "Normal (bar)",
    "pred_e_m": "Pred E (m)", "pred_n_m": "Pred N (m)", "pred_u_m": "Pred U (m)",
    "pred_horiz_mag_m": "Pred horiz (m)",
    "obs_e_m": "Obs E (m)", "obs_n_m": "Obs N (m)", "obs_u_m": "Obs U (m)",
    "resid_e_m": "Resid E (m)", "resid_n_m": "Resid N (m)", "resid_u_m": "Resid U (m)",
    "resid_3d_mag_m": "Resid 3D (m)",
}


def _fmt(val):
    if val is None:
        return ""
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float):
        return f"{val:.5g}"
    return str(val)


class PointCalculatorDialog(QDialog):
    """
    Constructor callbacks (mirroring FaultTableWidget.set_elastic_provider()'s
    existing callback pattern in this plugin, rather than duplicating
    source/receiver/elastic state):

      get_sources()  -> list[FaultParameters], the current STRESS-SOURCE
                        faults (main_dialog._get_sources())
      get_receiver() -> FaultParameters, the shared receiver strike/dip/
                        rake (main_dialog._get_receiver())
      get_elastic()  -> ElasticParameters (main_dialog._get_elastic())

    Results are kept on self.results (list of dicts, RESULT_COLUMNS
    keys) after a successful Compute, in case a caller wants them
    without going through the Add-to-QGIS/Export buttons.
    """

    def __init__(self, parent=None, get_sources=None, get_receiver=None, get_elastic=None):
        super().__init__(parent)
        self.setWindowTitle("Point Calculator — Stress & Displacement at Points")
        configure_resizable_dialog(self, 900, 700, min_width=520, min_height=420)

        self._get_sources = get_sources
        self._get_receiver = get_receiver
        self._get_elastic = get_elastic

        self._columns = []
        self._rows = []
        self._field_combos = {}
        self.results = []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<b>Point Calculator</b><br><i>Compute predicted Coulomb stress "
            "(ΔCFF/shear/normal, resolved onto the Receiver Fault tab's "
            "orientation) and predicted displacement at arbitrary points, "
            "from the current source faults. Optionally map an observed "
            "East/North/Up displacement or slip vector to validate the "
            "model against field measurements (GNSS offsets, leveling "
            "benchmarks, measured rupture slip, etc.) — residuals are "
            "computed automatically wherever an observation is mapped.</i>"))

        # ── Source selection ────────────────────────────────────────────
        source_group = QGroupBox("Observation points — source")
        source_layout = QVBoxLayout(source_group)
        self.radio_file = QRadioButton("File (CSV/TSV)")
        self.radio_layer = QRadioButton("QGIS layer already in this project")
        self.radio_file.setChecked(True)
        radio_row = QHBoxLayout()
        radio_row.addWidget(self.radio_file)
        radio_row.addWidget(self.radio_layer)
        source_layout.addLayout(radio_row)

        file_row = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse_file)
        file_row.addWidget(self.file_label, 1)
        file_row.addWidget(self.btn_browse)
        source_layout.addLayout(file_row)

        self.layer_combo = QComboBox()
        self._populate_layers()
        source_layout.addWidget(self.layer_combo)

        self.btn_load = QPushButton("Load columns")
        self.btn_load.clicked.connect(self._load_source)
        source_layout.addWidget(self.btn_load)

        layout.addWidget(source_group)

        # ── Column mapping ──────────────────────────────────────────────
        self.mapping_group = QGroupBox("Column mapping")
        self.mapping_form = QFormLayout(self.mapping_group)
        layout.addWidget(self.mapping_group)

        # ── Preview ──────────────────────────────────────────────────────
        layout.addWidget(QLabel("<b>Preview (first 10 rows)</b>"))
        self.preview_table = QTableWidget(0, 0)
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.preview_table.setMaximumHeight(150)
        layout.addWidget(self.preview_table)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        compute_row = QHBoxLayout()
        self.btn_compute = QPushButton("▶  Compute")
        self.btn_compute.clicked.connect(self._do_compute)
        self.btn_compute.setEnabled(False)
        compute_row.addWidget(self.btn_compute)
        compute_row.addStretch()
        layout.addLayout(compute_row)

        # ── Results ──────────────────────────────────────────────────────
        layout.addWidget(QLabel("<b>Results</b>"))
        self.results_table = QTableWidget(0, 0)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.results_table, 1)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        btn_row = QHBoxLayout()
        self.btn_add_layer = QPushButton("Add to QGIS as layer")
        self.btn_add_layer.clicked.connect(self._add_to_qgis)
        self.btn_add_layer.setEnabled(False)
        btn_row.addWidget(self.btn_add_layer)
        self.btn_export_csv = QPushButton("Export → CSV…")
        self.btn_export_csv.clicked.connect(self._export_csv)
        self.btn_export_csv.setEnabled(False)
        btn_row.addWidget(self.btn_export_csv)
        btn_row.addStretch()
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

    # ── Source loading ──────────────────────────────────────────────────

    def _populate_layers(self):
        from qgis.core import QgsProject, QgsMapLayer, QgsWkbTypes
        self.layer_combo.clear()
        self._point_layers = []
        for layer in QgsProject.instance().mapLayers().values():
            if (layer.type() == QgsMapLayer.VectorLayer
                    and layer.geometryType() == QgsWkbTypes.PointGeometry):
                self._point_layers.append(layer)
                self.layer_combo.addItem(layer.name())

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select observation points", "",
            "Delimited text (*.csv *.tsv *.txt);;All files (*.*)")
        if path:
            self.file_label.setText(path)
            self._file_path = path

    def _load_source(self):
        try:
            if self.radio_file.isChecked():
                path = getattr(self, "_file_path", None)
                if not path:
                    QMessageBox.warning(self, "No file", "Choose a file first.")
                    return
                self._columns, self._rows = read_delimited_table(path, is_path=True)
            else:
                if not self._point_layers:
                    QMessageBox.warning(self, "No layers",
                                        "No point layers found in this project.")
                    return
                layer = self._point_layers[self.layer_combo.currentIndex()]
                self._columns, self._rows = read_qgis_layer_table(layer)
                # Surface the synthetic geometry-derived lon/lat columns
                # (see eq_catalog_import_dialog._load_source()'s identical
                # comment) so a layer with no explicit lon/lat attribute
                # columns can still be mapped from its geometry.
                if any("__geom_x__" in row for row in self._rows):
                    for geom_col in ("__geom_x__", "__geom_y__"):
                        if geom_col not in self._columns:
                            self._columns.append(geom_col)
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not read source:\n{e}")
            return

        if not self._columns:
            self.status_label.setText("No columns found in source.")
            return

        self._rebuild_mapping_ui()
        self._update_preview()
        self.btn_compute.setEnabled(True)

    # ── Mapping UI ───────────────────────────────────────────────────────

    def _rebuild_mapping_ui(self):
        while self.mapping_form.rowCount():
            self.mapping_form.removeRow(0)
        self._field_combos = {}

        if not self._columns:
            return

        fields = list(POSITION_FIELDS) + list(OPTIONAL_FIELDS)
        mapping = detect_point_mapping(self._columns)

        for f in fields:
            combo = QComboBox()
            combo.addItem(NONE_LABEL)
            combo.addItems(self._columns)
            guess = mapping.get(f)
            if guess:
                combo.setCurrentText(guess)
            self.mapping_form.addRow(FIELD_DISPLAY_NAMES.get(f, f) + ":", combo)
            self._field_combos[f] = combo

    def _current_column_map(self):
        out = {}
        for f, combo in self._field_combos.items():
            val = combo.currentText()
            out[f] = None if val == NONE_LABEL else val
        return out

    def _update_preview(self):
        preview_rows = self._rows[:10]
        self.preview_table.setColumnCount(len(self._columns))
        self.preview_table.setHorizontalHeaderLabels(self._columns)
        self.preview_table.setRowCount(len(preview_rows))
        for r, row in enumerate(preview_rows):
            for c, col in enumerate(self._columns):
                item = QTableWidgetItem(str(row.get(col, "")))
                item.setTextAlignment(Qt.AlignCenter)
                self.preview_table.setItem(r, c, item)

    # ── Compute ──────────────────────────────────────────────────────────

    def _do_compute(self):
        column_map = self._current_column_map()
        missing = [f for f in POSITION_FIELDS if not column_map.get(f)]
        if missing:
            names = ", ".join(FIELD_DISPLAY_NAMES.get(f, f) for f in missing)
            QMessageBox.warning(self, "Missing mapping",
                                f"These required fields aren't mapped to a column:\n{names}")
            return

        import_result = build_points_from_mapped_rows(self._rows, column_map)
        if not import_result.points:
            detail = "\n".join(import_result.errors[:10])
            QMessageBox.warning(self, "Nothing to compute",
                                f"No valid points could be built from this mapping.\n\n{detail}")
            return

        if self._get_sources is None or self._get_receiver is None or self._get_elastic is None:
            QMessageBox.critical(self, "Not configured",
                                 "This dialog was opened without a source/receiver/elastic "
                                 "provider — cannot compute.")
            return

        sources = self._get_sources()
        if not sources:
            QMessageBox.warning(self, "No source faults",
                                "No stress-source faults are defined (Source Faults tab) — "
                                "add at least one fault with nonzero slip before computing.")
            return
        receiver = self._get_receiver()
        elastic = self._get_elastic()

        try:
            self.results = compute_point_results(sources, import_result.points, receiver, elastic)
        except Exception as e:
            QMessageBox.critical(self, "Compute error", f"Could not compute point results:\n{e}")
            return

        msg = f"Computed {len(self.results)} point(s)."
        if import_result.n_skipped:
            msg += f" Skipped {import_result.n_skipped} row(s) with missing lon/lat/elevation."
        n_clamped = sum(1 for r in self.results if r["elevation_clamped"])
        if n_clamped:
            msg += (f" {n_clamped} point(s) had elevation above the surface and were "
                   f"evaluated AT the surface (see the note on this in the addendum/README).")
        n_dc3d_fallback = sum(1 for r in self.results
                              if r["depth_km"] > 0.0 and not r["used_dc3d"])
        if n_dc3d_fallback:
            msg += (f" {n_dc3d_fallback} below-surface point(s) fell back to the z=0 surface "
                   f"formula (no working external Python/DC3D configured) — see Dependencies.")
        self.status_label.setText(msg)
        if import_result.errors:
            detail = "\n".join(import_result.errors[:15])
            more = f"\n… and {len(import_result.errors) - 15} more" if len(import_result.errors) > 15 else ""
            QMessageBox.information(self, "Compute complete with some skipped rows",
                                    f"{msg}\n\nSkipped rows:\n{detail}{more}")

        self._update_results_table()
        self._update_summary()
        self.btn_add_layer.setEnabled(True)
        self.btn_export_csv.setEnabled(True)

    def _update_results_table(self):
        self.results_table.setColumnCount(len(DISPLAY_COLUMNS))
        self.results_table.setHorizontalHeaderLabels(
            [DISPLAY_COLUMN_LABELS.get(c, c) for c in DISPLAY_COLUMNS])
        self.results_table.setRowCount(len(self.results))
        for r, row in enumerate(self.results):
            for c, col in enumerate(DISPLAY_COLUMNS):
                item = QTableWidgetItem(_fmt(row.get(col)))
                item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(r, c, item)

    def _update_summary(self):
        """RMS misfit summary across points that carry at least one
        observed component — a quick at-a-glance validation metric,
        supplementing (not replacing) the per-point residual columns."""
        mags = [r["resid_3d_mag_m"] for r in self.results if r["resid_3d_mag_m"] is not None]
        if not mags:
            self.summary_label.setText(
                "No points had an observed displacement/slip vector mapped — "
                "showing predicted stress/displacement only, no validation residuals.")
            return
        import math
        rms = math.sqrt(sum(m * m for m in mags) / len(mags))
        mean_abs = sum(abs(m) for m in mags) / len(mags)
        self.summary_label.setText(
            f"Validation: {len(mags)}/{len(self.results)} point(s) had an observed vector. "
            f"RMS residual = {rms:.4g} m, mean |residual| = {mean_abs:.4g} m "
            f"(using whichever of E/N/U components each point supplied).")

    # ── Output ───────────────────────────────────────────────────────────

    def _add_to_qgis(self):
        if not self.results:
            return
        try:
            from .vector_utils import create_point_calc_layer
            create_point_calc_layer(self.results)
            QMessageBox.information(self, "Layer added",
                                    "Point results added to the project as a new layer.")
        except Exception as e:
            QMessageBox.critical(self, "Layer error", f"Could not create layer:\n{e}")

    def _export_csv(self):
        if not self.results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export point results", "point_calculator_results.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                f.write(results_to_csv_text(self.results))
            QMessageBox.information(self, "Exported", f"Results exported to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", f"Could not export CSV:\n{e}")
