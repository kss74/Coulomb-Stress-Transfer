# -*- coding: utf-8 -*-
"""
Dialog for inverting scattered surface-displacement observations for
per-sub-patch (rt-lateral, reverse) slip on a subdivided source fault
row of FaultTableWidget.

Two OBSERVATION TYPES can be imported and combined in one joint solve:
  - "GNSS / field measurements": component-wise (E, N, U), any subset
    per point (leveling = U only, etc.)
  - "InSAR (LOS)": one LOS-projected scalar per point + its own unit
    look vector

Each type can be loaded, independently and repeatedly, from either a
delimited text file (CSV/TSV) or a QGIS point layer already in the
project -- see core.observation_import for the column-mapping machinery
(auto-suggested, always reviewable/overridable before committing a
batch). Imported points accumulate in two internal lists (GNSS, LOS)
that both feed the same inversion; either may be empty.

The physics itself -- Green's-matrix assembly, Laplacian smoothing,
bounded least squares, optional moment constraint -- lives entirely in
core.okada_engine.run_slip_inversion() / dc3d_worker.py's
"slip_inversion" mode. This dialog is presentation + import plumbing
only.

Patch indexing / ordering matches core.okada_engine.FaultParameters.
subdivide() exactly, same as distributed_slip_dialog.py (i=down-dip,
j=along-strike, flat i*n_length+j).
"""

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QPushButton,
    QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QRadioButton, QGroupBox, QMessageBox, QDoubleSpinBox,
    QCheckBox, QTabWidget, QWidget
)
from qgis.PyQt.QtCore import Qt

import math

from .fault_table_widget import _annex_labels
from .dialog_utils import configure_resizable_dialog, wrap_in_scroll_area
from ..core.observation_import import (
    SCHEMAS, SCHEMA_REQUIRED_FIELDS, POSITION_FIELDS,
    suggest_column_mapping, detect_schema, read_delimited_table,
    build_observations_from_mapped_rows,
)

FIELD_DISPLAY_NAMES = {
    "lon": "Longitude", "lat": "Latitude",
    "e": "East disp. (m)", "n": "North disp. (m)", "u": "Up disp. (m)",
    "sigma_e": "1-sigma East (optional)", "sigma_n": "1-sigma North (optional)",
    "sigma_u": "1-sigma Up (optional)",
    "los": "LOS displacement (m)",
    "look_e": "Look vector E", "look_n": "Look vector N", "look_u": "Look vector U",
    "sigma": "1-sigma LOS (optional)",
}
NONE_LABEL = "(not mapped)"
SCHEMA_LABELS_FOR_EXPORT = {"gnss": "GNSS", "insar_los": "InSAR"}


class _ObservationImportPanel(QWidget):
    """One reusable file-or-layer / column-mapping / preview panel,
    parameterized to a single schema (used twice: once for the "GNSS"
    tab, once for the "InSAR" tab -- each tab always imports as its own
    fixed schema, unlike FocalMechanismImportDialog which supports
    several schemas in one panel)."""

    def __init__(self, schema, on_batch_ready, parent=None):
        super().__init__(parent)
        self._schema = schema
        self._on_batch_ready = on_batch_ready  # callback(list_of_dicts)
        self._columns = []
        self._rows = []
        self._field_combos = {}
        self._point_layers = []
        self._file_path = None

        layout = QVBoxLayout(self)

        source_group = QGroupBox("Source")
        source_layout = QVBoxLayout(source_group)
        self.radio_file = QRadioButton("File (CSV/TSV)")
        self.radio_layer = QRadioButton("QGIS point layer already in this project")
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

        self.mapping_group = QGroupBox("Column mapping")
        self.mapping_form = QFormLayout(self.mapping_group)
        layout.addWidget(self.mapping_group)

        layout.addWidget(QLabel("<b>Preview (first 10 rows)</b>"))
        self.preview_table = QTableWidget(0, 0)
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.preview_table.setMaximumHeight(150)
        layout.addWidget(self.preview_table)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        add_row = QHBoxLayout()
        self.btn_add_batch = QPushButton("+ Add these points to the inversion")
        self.btn_add_batch.setEnabled(False)
        self.btn_add_batch.clicked.connect(self._do_add_batch)
        add_row.addWidget(self.btn_add_batch)
        add_row.addStretch()
        layout.addLayout(add_row)

        self._rebuild_mapping_ui()

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
            self, "Select observation table", "",
            "Delimited text (*.csv *.tsv *.txt);;All files (*.*)")
        if path:
            self.file_label.setText(path)
            self._file_path = path

    def _load_source(self):
        try:
            if self.radio_file.isChecked():
                if not self._file_path:
                    QMessageBox.warning(self, "No file", "Choose a file first.")
                    return
                self._columns, self._rows = read_delimited_table(self._file_path, is_path=True)
            else:
                if not self._point_layers:
                    QMessageBox.warning(self, "No layers",
                                        "No point layers found in this project.")
                    return
                layer = self._point_layers[self.layer_combo.currentIndex()]
                from ..core.observation_import import read_qgis_layer_table
                self._columns, self._rows = read_qgis_layer_table(layer)
                # Prefer geometry-derived lon/lat over any same-named
                # attribute columns, and put them first so they win
                # ties in suggest_column_mapping()'s alias matching.
                if "__geom_x__" in self._columns:
                    self._columns = ["__geom_x__", "__geom_y__"] + \
                        [c for c in self._columns if c not in ("__geom_x__", "__geom_y__")]
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not read source:\n{e}")
            return

        if not self._columns:
            self.status_label.setText("No columns found in source.")
            return

        _, mapping = detect_schema(self._columns)
        if not mapping:
            fields = list(POSITION_FIELDS) + list(SCHEMAS[self._schema])
            mapping = suggest_column_mapping(self._columns, fields)
        # __geom_x__/__geom_y__ (from a QGIS layer) are always lon/lat.
        if "__geom_x__" in self._columns:
            mapping = dict(mapping, lon="__geom_x__", lat="__geom_y__")
        self._rebuild_mapping_ui(mapping)
        self._update_preview()
        self.btn_add_batch.setEnabled(True)

    def _rebuild_mapping_ui(self, mapping=None):
        while self.mapping_form.rowCount():
            self.mapping_form.removeRow(0)
        self._field_combos = {}
        fields = list(POSITION_FIELDS) + list(SCHEMAS[self._schema])
        mapping = mapping or {}
        for f in fields:
            combo = QComboBox()
            combo.addItem(NONE_LABEL)
            combo.addItems(self._columns)
            guess = mapping.get(f)
            if guess:
                combo.setCurrentText(guess)
            required = f in POSITION_FIELDS or f in SCHEMA_REQUIRED_FIELDS[self._schema]
            label = FIELD_DISPLAY_NAMES.get(f, f) + (":" if not required else " (required):")
            self.mapping_form.addRow(label, combo)
            self._field_combos[f] = combo

    def _current_column_map(self):
        return {f: (None if combo.currentText() == NONE_LABEL else combo.currentText())
                for f, combo in self._field_combos.items()}

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

    def _do_add_batch(self):
        column_map = self._current_column_map()
        required = list(POSITION_FIELDS) + list(SCHEMA_REQUIRED_FIELDS[self._schema])
        missing = [f for f in required if not column_map.get(f)]
        if missing:
            names = ", ".join(FIELD_DISPLAY_NAMES.get(f, f) for f in missing)
            QMessageBox.warning(self, "Missing mapping",
                                f"These required fields aren't mapped to a column:\n{names}")
            return

        result = build_observations_from_mapped_rows(self._rows, column_map, self._schema)
        msg = f"Added {len(result.rows)} point(s)."
        if result.n_skipped:
            msg += f" Skipped {result.n_skipped} row(s) (see below)."
        self.status_label.setText(msg)

        if not result.rows:
            detail = "\n".join(result.errors[:10])
            QMessageBox.warning(self, "Nothing added",
                                f"No valid points could be built from this mapping.\n\n{detail}")
            return
        if result.errors:
            detail = "\n".join(result.errors[:15])
            more = f"\n… and {len(result.errors) - 15} more" if len(result.errors) > 15 else ""
            QMessageBox.information(self, "Added with some skipped rows",
                                    f"{msg}\n\nSkipped rows:\n{detail}{more}")

        self._on_batch_ready(result.rows)


class SlipInversionDialog(QDialog):
    """
    Invert scattered surface-displacement observations (GNSS and/or
    InSAR) for per-sub-patch slip on one OR SEVERAL subdivided source
    fault rows at once (a "Group" of rows with different strikes
    tracing one bent/kinked fault -- see FaultTableWidget's "Group"
    column), then write the result into the SAME slip_overrides storage
    the manual "Edit distributed slip…" dialog already uses (see
    FaultTableWidget._set_distributed_slip()) -- so it round-trips
    through JSON save/load and the stress pipeline with no other change
    needed. A single-fault run is just a one-element `fault_specs`.
    """

    def __init__(self, parent, fault_specs, elastic):
        """
        fault_specs : list of dicts, one per fault row involved, each:
          {"row": <FaultTableWidget row index -- returned as-is in
                  get_overrides(), not interpreted here>,
           "name": str,                      # for labels/titles
           "fault": core.okada_engine.FaultParameters,  # that row's own
                  (un-subdivided) geometry -- its own slip is ignored,
                  only geometry feeds the Green's matrix
           "n_length": int, "n_width": int}  # that row's Subdiv.(L)/(W)
        elastic     : core.okada_engine.ElasticParameters (mu, nu, ...)
                     -- same elastic model as the rest of the plugin,
                     shared across every fault segment in the group.
        """
        super().__init__(parent)
        self._fault_specs = fault_specs
        self._is_group = len(fault_specs) > 1
        total_patches = sum(s["n_length"] * s["n_width"] for s in fault_specs)
        if self._is_group:
            names = ", ".join(s["name"] for s in fault_specs)
            self.setWindowTitle(f"Invert for slip — Group of {len(fault_specs)} faults ({names})")
        else:
            self.setWindowTitle(f"Invert for slip — {fault_specs[0]['name']}")
        configure_resizable_dialog(self, 760, 680, min_width=460, min_height=380)

        self._elastic = elastic
        self._gnss_points = []
        self._los_points = []
        self._overrides = None  # {row: {(i,j):(rt,rev)}}, set on accept

        # Total rupture area (m^2), summed over every fault segment's
        # OWN (un-subdivided) geometry -- this is what bounds the
        # achievable seismic moment for a given max_slip bound,
        # mirroring the mu*sum(area_p*sqrt(2)*max_slip) feasibility
        # check in dc3d_worker._run_slip_inversion_core(). Computed
        # once here since geometry doesn't change over the dialog's
        # lifetime.
        self._total_area_m2 = sum(
            s["fault"].length * 1000.0 * s["fault"].width * 1000.0
            for s in fault_specs)

        inner = wrap_in_scroll_area(self, lambda w: None)
        layout = QVBoxLayout(inner)

        if self._is_group:
            seg_lines = "".join(
                f"<li><b>{s['name']}</b>: {s['n_length']}×{s['n_width']} = "
                f"{s['n_length'] * s['n_width']} sub-patches</li>"
                for s in fault_specs)
            layout.addWidget(QLabel(
                f"<b>Group of {len(fault_specs)} fault segments</b> "
                f"({total_patches} total sub-patches), inverted JOINTLY "
                f"(one combined linear system, each segment's own patches "
                f"smoothed only against its own along-strike/down-dip "
                f"neighbors — not across the strike bend between "
                f"segments):<ul>{seg_lines}</ul>"
                f"Import surface-displacement observations below (GNSS "
                f"and/or InSAR — either or both, repeated imports "
                f"accumulate), then run the inversion to solve "
                f"independent right-lateral/reverse slip for each "
                f"sub-patch of every segment at once. Requires the "
                f"external Python (DC3D) to also have <b>scipy</b> "
                f"installed alongside okada_wrapper."))
        else:
            s = fault_specs[0]
            layout.addWidget(QLabel(
                f"<b>{s['name']}</b> ({s['n_length']}×{s['n_width']} = "
                f"{total_patches} sub-patches). "
                f"Import surface-displacement observations below (GNSS and/or InSAR — either "
                f"or both, repeated imports accumulate), then run the inversion to solve "
                f"independent right-lateral/reverse slip for each sub-patch. Requires the "
                f"external Python (DC3D) to also have <b>scipy</b> installed alongside "
                f"okada_wrapper."))

        self.tabs = QTabWidget()
        self.gnss_panel = _ObservationImportPanel("gnss", self._on_gnss_batch)
        self.los_panel = _ObservationImportPanel("insar_los", self._on_los_batch)
        self.tabs.addTab(self.gnss_panel, "GNSS / field measurements")
        self.tabs.addTab(self.los_panel, "InSAR (LOS)")
        layout.addWidget(self.tabs)

        self.points_label = QLabel("No observations added yet.")
        layout.addWidget(self.points_label)

        # ── Inversion parameters ──────────────────────────────────────
        params_group = QGroupBox("Inversion parameters")
        params_form = QFormLayout(params_group)

        self.spin_smoothing = QDoubleSpinBox()
        self.spin_smoothing.setRange(0.0, 1000.0)
        self.spin_smoothing.setDecimals(4)
        self.spin_smoothing.setSingleStep(0.01)
        self.spin_smoothing.setValue(0.05)
        params_form.addRow("Laplacian smoothing factor:", self.spin_smoothing)

        self.spin_max_slip = QDoubleSpinBox()
        self.spin_max_slip.setRange(0.001, 1000.0)
        self.spin_max_slip.setDecimals(3)
        self.spin_max_slip.setValue(10.0)
        params_form.addRow("Max |slip| bound (m):", self.spin_max_slip)

        self.check_target_mw = QCheckBox(
            "Constrain total moment to target Mw" +
            (" (whole group)" if self._is_group else ""))
        self.check_target_mw.setChecked(False)
        self.spin_target_mw = QDoubleSpinBox()
        self.spin_target_mw.setRange(0.0, 10.0)
        self.spin_target_mw.setDecimals(2)
        self.spin_target_mw.setValue(6.5)
        self.spin_target_mw.setEnabled(False)
        self.check_target_mw.toggled.connect(self.spin_target_mw.setEnabled)
        mw_row = QHBoxLayout()
        mw_row.addWidget(self.check_target_mw)
        mw_row.addWidget(self.spin_target_mw)
        params_form.addRow(mw_row)

        # ── Fixed-rake constraint ────────────────────────────────────
        # Reduces each patch from 2 independent unknowns (rt-lateral,
        # reverse) to 1 (a signed slip magnitude along this rake) --
        # see dc3d_worker.py's "slip_inversion" schema docstring for
        # the full rationale. Applied UNIFORMLY to every patch of every
        # fault in the group; a per-patch-varying rake is not exposed
        # here (the worker supports it, but there's no natural single
        # UI value for it -- see okada_engine.run_slip_inversion_group()
        # docstring). Pre-filled from the first fault's own .rake field
        # if it's non-zero (e.g. set from an imported focal mechanism --
        # see PROJECT_HANDOVER_ADDENDUM_2026-08-14b), purely a
        # convenience default the user can override.
        self.check_fixed_rake = QCheckBox("Constrain to fixed rake (°)")
        self.check_fixed_rake.setChecked(False)
        self.spin_fixed_rake = QDoubleSpinBox()
        self.spin_fixed_rake.setRange(-180.0, 180.0)
        self.spin_fixed_rake.setDecimals(1)
        default_rake = fault_specs[0]["fault"].rake if fault_specs else 0.0
        self.spin_fixed_rake.setValue(float(default_rake))
        self.spin_fixed_rake.setEnabled(False)
        self.check_fixed_rake.toggled.connect(self.spin_fixed_rake.setEnabled)
        self.check_fixed_rake.toggled.connect(self._update_feasibility_hint)
        rake_row = QHBoxLayout()
        rake_row.addWidget(self.check_fixed_rake)
        rake_row.addWidget(self.spin_fixed_rake)
        params_form.addRow(rake_row)
        if self._is_group:
            params_form.addRow(QLabel(
                "<i>Fixed rake applies the SAME value to every patch of "
                "every fault segment in this group.</i>"))

        # ── Feasibility hint + auto-fill ────────────────────────────
        # max_slip and target_mw trade off against the FIXED total
        # rupture area: M0 = mu * area * |slip|, so for two independent
        # rt-lateral/reverse unknowns per patch the achievable moment
        # tops out at mu * area * sqrt(2) * max_slip (see feasibility
        # check in dc3d_worker.py). Surfacing that relationship here
        # avoids the "Target Mw is not feasible" error appearing only
        # after Run is pressed.
        self.feasibility_label = QLabel("")
        self.feasibility_label.setWordWrap(True)
        self.feasibility_label.setStyleSheet("color: #555;")
        params_form.addRow(self.feasibility_label)

        self.btn_use_min_slip = QPushButton("Set max |slip| to minimum needed for target Mw")
        self.btn_use_min_slip.setEnabled(False)
        self.btn_use_min_slip.clicked.connect(self._use_min_slip)
        params_form.addRow(self.btn_use_min_slip)

        self.check_target_mw.toggled.connect(self._update_feasibility_hint)
        self.spin_target_mw.valueChanged.connect(self._update_feasibility_hint)
        self.spin_max_slip.valueChanged.connect(self._update_feasibility_hint)
        self._update_feasibility_hint()

        layout.addWidget(params_group)

        run_row = QHBoxLayout()
        self.btn_run = QPushButton("▶ Run inversion")
        self.btn_run.clicked.connect(self._run_inversion)
        run_row.addWidget(self.btn_run)
        run_row.addStretch()
        layout.addLayout(run_row)

        # ── Results ──────────────────────────────────────────────────
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        # "Fault" column only matters for a group, but keep it always
        # present (blank/single-name for the 1-fault case) so the table
        # layout doesn't change shape between the two cases.
        self.result_table = QTableWidget(0, 4)
        self.result_table.setHorizontalHeaderLabels(
            ["Fault", "Patch", "Rt-lateral slip (m)", "Reverse slip (m)"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setMaximumHeight(240)
        layout.addWidget(self.result_table)

        ok_row = QHBoxLayout()
        self.btn_export_report = QPushButton("📄 Export QA report…")
        self.btn_export_report.setEnabled(False)
        self.btn_export_report.clicked.connect(self._export_report)
        self.btn_export_csv = QPushButton("💾 Export results as CSV…")
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(self._export_results_csv)
        self.btn_export_layer = QPushButton("📍 Add results as QGIS layer(s)")
        self.btn_export_layer.setEnabled(False)
        self.btn_export_layer.clicked.connect(self._export_results_layer)
        ok_row.addWidget(self.btn_export_report)
        ok_row.addWidget(self.btn_export_csv)
        ok_row.addWidget(self.btn_export_layer)
        ok_row.addStretch()
        self.btn_apply = QPushButton("Apply to fault(s)")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._on_accept)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        ok_row.addWidget(self.btn_apply)
        ok_row.addWidget(self.btn_cancel)
        layout.addLayout(ok_row)

    # ── Accumulating imported points ────────────────────────────────

    def _on_gnss_batch(self, rows):
        self._gnss_points.extend(rows)
        self._refresh_points_label()

    def _on_los_batch(self, rows):
        self._los_points.extend(rows)
        self._refresh_points_label()

    def _refresh_points_label(self):
        self.points_label.setText(
            f"<b>{len(self._gnss_points)}</b> GNSS/field point(s), "
            f"<b>{len(self._los_points)}</b> InSAR/LOS point(s) ready for inversion.")

    # ── Slip-bound / moment feasibility calculator ──────────────────
    # M0 = mu * area * |slip| per patch. With independent rt-lateral +
    # reverse unknowns per patch (the default), the worst case (all
    # patches maxed on BOTH components) is mu * total_area * sqrt(2) *
    # max_slip. With a fixed rake, there's only one bounded unknown per
    # patch (the signed slip magnitude), so the worst case is just
    # mu * total_area * max_slip -- no sqrt(2). Both exactly mirror the
    # two branches dc3d_worker.py checks before raising "Target Mw ...
    # not feasible", so the dialog can warn (and offer a fix) up front.

    def _channel_factor(self):
        return 1.0 if self.check_fixed_rake.isChecked() else math.sqrt(2.0)

    def _min_max_slip_for_mw(self, target_mw):
        """Smallest max_slip (m) that makes target_mw reachable."""
        mu = self._elastic.mu
        if mu <= 0 or self._total_area_m2 <= 0:
            return float("inf")
        m0_target = 10.0 ** (1.5 * float(target_mw) + 9.1)
        return m0_target / (mu * self._total_area_m2 * self._channel_factor())

    def _max_mw_for_slip(self, max_slip):
        """Largest Mw reachable at the given max_slip (m) bound."""
        mu = self._elastic.mu
        m0_max = mu * self._total_area_m2 * self._channel_factor() * float(max_slip)
        if m0_max <= 0:
            return float("-inf")
        return (2.0 / 3.0) * (math.log10(m0_max) - 9.1)

    def _update_feasibility_hint(self):
        max_slip = float(self.spin_max_slip.value())
        max_mw = self._max_mw_for_slip(max_slip)
        if self.check_target_mw.isChecked():
            target_mw = float(self.spin_target_mw.value())
            min_slip = self._min_max_slip_for_mw(target_mw)
            if min_slip > max_slip:
                self.feasibility_label.setText(
                    f"⚠ Not feasible: Mw {target_mw:.2f} over this rupture's "
                    f"area needs max |slip| ≥ {min_slip:.3f} m, but the bound "
                    f"above is {max_slip:.3f} m (max reachable: Mw {max_mw:.2f}).")
                self.btn_use_min_slip.setEnabled(True)
            else:
                self.feasibility_label.setText(
                    f"✓ Feasible: Mw {target_mw:.2f} needs max |slip| ≥ "
                    f"{min_slip:.3f} m (bound above: {max_slip:.3f} m, "
                    f"max reachable: Mw {max_mw:.2f}).")
                self.btn_use_min_slip.setEnabled(False)
        else:
            self.feasibility_label.setText(
                f"Max |slip| = {max_slip:.3f} m allows up to Mw {max_mw:.2f} "
                f"over this rupture's total area (unconstrained inversion is "
                f"never blocked by this -- it's just the bound the solver "
                f"clips to).")
            self.btn_use_min_slip.setEnabled(False)

    def _use_min_slip(self):
        target_mw = float(self.spin_target_mw.value())
        min_slip = self._min_max_slip_for_mw(target_mw)
        # Small margin so floating-point rounding doesn't leave it
        # exactly on the infeasible boundary.
        self.spin_max_slip.setValue(min(min_slip * 1.02, self.spin_max_slip.maximum()))

    # ── Run ──────────────────────────────────────────────────────────

    def _run_inversion(self):
        if not self._gnss_points and not self._los_points:
            QMessageBox.warning(self, "No observations",
                                "Import at least one GNSS or InSAR point first.")
            return

        target_mw = float(self.spin_target_mw.value()) if self.check_target_mw.isChecked() else None
        fixed_rake_deg = float(self.spin_fixed_rake.value()) if self.check_fixed_rake.isChecked() else None
        self.btn_run.setEnabled(False)
        self.result_label.setText("Running inversion (external Python subprocess)…")
        from qgis.PyQt.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            from ..core.okada_engine import run_slip_inversion_group
            specs = [{"key": s["row"], "fault": s["fault"],
                     "n_length": s["n_length"], "n_width": s["n_width"]}
                    for s in self._fault_specs]
            overrides_by_row, diag = run_slip_inversion_group(
                specs, self._gnss_points, self._los_points, self._elastic,
                smoothing_factor=float(self.spin_smoothing.value()),
                max_slip=float(self.spin_max_slip.value()),
                target_mw=target_mw, fixed_rake_deg=fixed_rake_deg)
        except Exception as e:
            self.result_label.setText("")
            QMessageBox.critical(self, "Inversion failed", str(e))
            self.btn_run.setEnabled(True)
            return
        finally:
            self.btn_run.setEnabled(True)

        self._overrides = overrides_by_row
        self.btn_apply.setEnabled(True)
        # Snapshot the exact points/diagnostics THIS run used, so later
        # exports stay consistent even if more points are imported into
        # the tabs afterward (before a re-run).
        self._run_gnss_points = list(self._gnss_points)
        self._run_los_points = list(self._los_points)
        self._run_diag = diag
        self._run_params = dict(
            smoothing_factor=float(self.spin_smoothing.value()),
            max_slip=float(self.spin_max_slip.value()),
            target_mw=target_mw, fixed_rake_deg=fixed_rake_deg)
        self.btn_export_report.setEnabled(True)
        self.btn_export_csv.setEnabled(True)
        self.btn_export_layer.setEnabled(True)

        status = "converged" if diag.get("solver_success") else "did NOT fully converge"
        mw_line = (f" | achieved Mw {diag['achieved_mw']:.3f}"
                   f" (target {target_mw:.3f})" if target_mw is not None
                   else f" | achieved Mw {diag['achieved_mw']:.3f}")
        rake_line = (f" | fixed rake {fixed_rake_deg:.1f}°" if fixed_rake_deg is not None else "")
        self.result_label.setText(
            f"Solver {status}: {diag.get('solver_message', '')}<br>"
            f"n_data={diag.get('n_data')}  RMS misfit={diag.get('rms_misfit'):.4g}"
            f"{mw_line}{rake_line}")

        self.result_table.setRowCount(0)
        row_out = 0
        for spec in self._fault_specs:
            n_length, n_width = spec["n_length"], spec["n_width"]
            overrides = overrides_by_row[spec["row"]]
            n_patches = n_length * n_width
            labels = _annex_labels(n_patches)
            self.result_table.setRowCount(self.result_table.rowCount() + n_patches)
            flat = 0
            for i in range(n_width):
                for j in range(n_length):
                    rt, rev = overrides[(i, j)]
                    self.result_table.setItem(row_out, 0, QTableWidgetItem(spec["name"]))
                    self.result_table.setItem(row_out, 1, QTableWidgetItem(labels[flat]))
                    self.result_table.setItem(row_out, 2, QTableWidgetItem(f"{rt:.6g}"))
                    self.result_table.setItem(row_out, 3, QTableWidgetItem(f"{rev:.6g}"))
                    flat += 1
                    row_out += 1

    def _on_accept(self):
        if self._overrides is None:
            return
        self.accept()

    def get_overrides(self):
        """{row: {(i, j): (rt_lateral_slip, reverse_slip)}} -- one entry
        per fault_specs["row"], each dict shaped exactly like
        DistributedSlipDialog.get_overrides()'s return, ready for
        FaultTableWidget._set_distributed_slip() per row."""
        return self._overrides or {}

    # ── Export: QA report (text) ────────────────────────────────────

    def _export_report(self):
        from ..core.slip_inversion_report import build_slip_inversion_report
        path, _ = QFileDialog.getSaveFileName(
            self, "Save QA report", f"{self._safe_name()}_slip_inversion_report.txt",
            "Text (*.txt)")
        if not path:
            return
        fault_segments = [
            {"name": s["name"], "n_length": s["n_length"], "n_width": s["n_width"],
             "overrides": self._overrides[s["row"]]}
            for s in self._fault_specs
        ]
        text = build_slip_inversion_report(
            fault_segments=fault_segments, elastic=self._elastic,
            gnss_points=self._run_gnss_points, los_points=self._run_los_points,
            diag=self._run_diag, **self._run_params)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        QMessageBox.information(self, "Exported", f"QA report saved:\n{path}")

    # ── Export: augmented results table (CSV or QGIS layer) ─────────

    def _safe_name(self):
        base = (f"Group_{len(self._fault_specs)}faults" if self._is_group
               else self._fault_specs[0]["name"])
        return "".join(c if c.isalnum() else "_" for c in base)

    def _build_result_tables(self):
        from ..core.slip_inversion_report import build_augmented_gnss_rows, build_augmented_los_rows
        tables = {}
        if self._run_gnss_points:
            tables["gnss"] = build_augmented_gnss_rows(self._run_gnss_points, self._run_diag)
        if self._run_los_points:
            tables["insar_los"] = build_augmented_los_rows(self._run_los_points, self._run_diag)
        return tables

    def _export_results_csv(self):
        import csv
        tables = self._build_result_tables()
        if not tables:
            QMessageBox.information(self, "Nothing to export", "No observations in the last run.")
            return
        base = self._safe_name()
        written = []
        for kind, rows in tables.items():
            default_name = f"{base}_{kind}_results.csv"
            path, _ = QFileDialog.getSaveFileName(
                self, f"Save {SCHEMA_LABELS_FOR_EXPORT.get(kind, kind)} results CSV",
                default_name, "CSV (*.csv)")
            if not path:
                continue
            fieldnames = list(rows[0].keys())
            try:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            except OSError as e:
                QMessageBox.critical(self, "Export failed", str(e))
                continue
            written.append(path)
        if written:
            QMessageBox.information(self, "Exported", "Saved:\n" + "\n".join(written))

    def _export_results_layer(self):
        tables = self._build_result_tables()
        if not tables:
            QMessageBox.information(self, "Nothing to export", "No observations in the last run.")
            return
        from ..utils.vector_utils import create_points_table_layer
        base = self._safe_name()
        added = []
        for kind, rows in tables.items():
            label = SCHEMA_LABELS_FOR_EXPORT.get(kind, kind)
            layer = create_points_table_layer(rows, f"{base} — {label} results")
            if layer is not None:
                added.append(layer.name())
        if added:
            QMessageBox.information(self, "Added", "Added to project:\n" + "\n".join(added))
        else:
            QMessageBox.information(self, "Nothing to add", "No result rows to add.")
