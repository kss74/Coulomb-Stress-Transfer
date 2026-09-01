# -*- coding: utf-8 -*-
"""
Import dialog for fault-patch tables (distributed-slip models from
external sources). Pick a file, pick a schema (auto-detected,
overridable), map columns (auto-suggested, overridable per field), set
units + depth convention, preview, import -- adds one row per patch
directly into the calling ui.fault_table_widget.FaultTableWidget via
its own add_row().

Mirrors ui.focal_import_dialog.FocalMechanismImportDialog's structure;
see that file for the shared column-mapping UX rationale. The main
addition here is the units/depth-convention panel: fault-patch tables
(unlike observation/focal-mechanism catalogs) can plausibly arrive in
km or m, and with depth as either the patch's top edge or its
volumetric centroid -- ambiguity a column header alone can't resolve,
so this module asks explicitly (core.fault_table_import's module
docstring explains why this is a deliberate non-guess).

Layout note (2026-08-29b): the source/schema/mapping/units/preview
stack is wrapped in a QScrollArea (same pattern as
main_dialog.py's Cross-Section tab, see PROJECT_HANDOVER_ADDENDUM
_2026-08-22b_xs_tab_scroll_area.md) so a schema with many mapped
fields plus the units panel doesn't get squeezed on shorter screens.
The Import/Cancel button row stays outside the scroll area, pinned to
the bottom, so it's always reachable regardless of scroll position.
The dialog window also now carries minimize/maximize hints, since it's
opened as a standalone top-level dialog (unlike the tabbed panels this
pattern was first used for) and a long/wide source table benefits from
being maximized.
"""

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QPushButton,
    QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QMessageBox, QLineEdit, QScrollArea, QWidget
)
from qgis.PyQt.QtCore import Qt

from ..core.fault_table_import import (
    POSITION_FIELDS, SCHEMAS, SCHEMA_LABELS,
    suggest_column_mapping, detect_schema, read_fault_table,
    build_fault_rows_from_mapped_rows,
)

FIELD_DISPLAY_NAMES = {
    "lon": "Longitude", "lat": "Latitude", "depth": "Depth",
    "length": "Length", "width": "Width", "strike": "Strike (°)", "dip": "Dip (°)",
    "rake": "Rake (°, Aki-Richards)", "slip": "Slip magnitude",
    "rt_lateral_slip": "Right-lateral slip", "reverse_slip": "Reverse slip",
}

NONE_LABEL = "(not mapped)"


class FaultTableImportDialog(QDialog):
    """Returns imported patch rows via .imported_rows after
    exec_() == Accepted. Each row is a dict matching
    ui.fault_table_widget.FaultTableWidget.add_row()'s DATA_COLUMNS
    order once unpacked by the caller (see _do_import())."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Fault-Patch Table")
        # Standalone top-level dialog: add minimize/maximize alongside the
        # default close button, and let it actually resize (a fixed-size
        # dialog can't be usefully maximized). setMinimumWidth keeps the
        # mapping/units panel from ever getting too cramped; resize() just
        # sets a sane initial size, not a ceiling.
        self.setWindowFlags(self.windowFlags()
                             | Qt.WindowMinimizeButtonHint
                             | Qt.WindowMaximizeButtonHint)
        self.setMinimumWidth(680)
        self.resize(720, 640)
        self.imported_rows = []
        self._columns = []
        self._rows = []
        self._field_combos = {}

        outer_layout = QVBoxLayout(self)

        # Everything except the Import/Cancel row lives inside a scroll
        # area, so a schema with many mapped fields (plus the units panel
        # and preview table) stays reachable at a reduced dialog height
        # instead of squeezing every row into an unusable strip.
        content = QWidget()
        layout = QVBoxLayout(content)

        layout.addWidget(QLabel(
            "<b>Import Fault-Patch Table</b><br><i>From a delimited text file "
            "(CSV/TSV, or whitespace-separated with a '#'-prefixed header line "
            "-- e.g. GSI/geodetic-inversion outputs). Column names rarely match "
            "this plugin's field names exactly -- the mapping below is "
            "auto-suggested; review and correct it before importing. Each "
            "imported row becomes one independent fault patch, exactly as if "
            "typed into the Source Faults table by hand.</i>"))

        # ── Source ──────────────────────────────────────────────────────
        source_group = QGroupBox("Source file")
        source_layout = QHBoxLayout(source_group)
        self.file_label = QLabel("No file selected")
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse_file)
        self.btn_load = QPushButton("Load columns")
        self.btn_load.clicked.connect(self._load_source)
        source_layout.addWidget(self.file_label, 1)
        source_layout.addWidget(self.btn_browse)
        source_layout.addWidget(self.btn_load)
        layout.addWidget(source_group)

        # ── Schema ──────────────────────────────────────────────────────
        schema_form = QFormLayout()
        self.schema_combo = QComboBox()
        for key in SCHEMAS:
            self.schema_combo.addItem(SCHEMA_LABELS[key], userData=key)
        self.schema_combo.currentIndexChanged.connect(self._rebuild_mapping_ui)
        schema_form.addRow("Slip representation:", self.schema_combo)
        layout.addLayout(schema_form)

        # ── Column mapping ─────────────────────────────────────────────
        self.mapping_group = QGroupBox("Column mapping")
        self.mapping_form = QFormLayout(self.mapping_group)
        layout.addWidget(self.mapping_group)

        # ── Units & depth convention ───────────────────────────────────
        units_group = QGroupBox("Units && depth convention")
        units_form = QFormLayout(units_group)
        self.combo_length_unit = QComboBox()
        self.combo_length_unit.addItems(["km", "m"])
        units_form.addRow("Length unit:", self.combo_length_unit)
        self.combo_width_unit = QComboBox()
        self.combo_width_unit.addItems(["(same as length)", "km", "m"])
        units_form.addRow("Width unit:", self.combo_width_unit)
        self.combo_depth_unit = QComboBox()
        self.combo_depth_unit.addItems(["km", "m"])
        units_form.addRow("Depth unit:", self.combo_depth_unit)
        self.combo_slip_unit = QComboBox()
        self.combo_slip_unit.addItems(["m", "cm", "mm"])
        units_form.addRow("Slip unit:", self.combo_slip_unit)
        self.combo_depth_convention = QComboBox()
        self.combo_depth_convention.addItems([
            "Centroid (patch's own volumetric center)",
            "Top edge (Coulomb's native convention)",
        ])
        units_form.addRow("Depth convention:", self.combo_depth_convention)
        units_form.addRow(QLabel(
            "<i>Not auto-detected from the column header -- verify against the "
            "source's own documentation. One check: if depth steps between "
            "along-dip rows equal width*sin(dip), the depth column is almost "
            "certainly centroid, not top-edge.</i>"))
        self.edit_group_label = QLineEdit()
        self.edit_group_label.setPlaceholderText("(optional -- e.g. dataset name)")
        units_form.addRow("Group label for all imported rows:", self.edit_group_label)
        layout.addWidget(units_group)

        # ── Preview ─────────────────────────────────────────────────────
        layout.addWidget(QLabel("<b>Preview (first 10 rows)</b>"))
        self.preview_table = QTableWidget(0, 0)
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.preview_table.setMaximumHeight(180)
        layout.addWidget(self.preview_table)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer_layout.addWidget(scroll, 1)

        # Import/Cancel row stays outside the scroll area, always visible.
        btn_row = QHBoxLayout()
        self.btn_import = QPushButton("Import")
        self.btn_import.clicked.connect(self._do_import)
        self.btn_import.setEnabled(False)
        btn_row.addWidget(self.btn_import)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)
        outer_layout.addLayout(btn_row)

    # ── Source loading ────────────────────────────────────────────────

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select fault-patch table", "",
            "Delimited text (*.csv *.tsv *.txt *.dat);;All files (*.*)")
        if path:
            self.file_label.setText(path)
            self._file_path = path

    def _load_source(self):
        path = getattr(self, "_file_path", None)
        if not path:
            QMessageBox.warning(self, "No file", "Choose a file first.")
            return
        try:
            self._columns, self._rows = read_fault_table(path)
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not read source:\n{e}")
            return

        if not self._columns:
            self.status_label.setText("No columns found in source.")
            return

        detected_schema, _ = detect_schema(self._columns)
        if detected_schema:
            idx = self.schema_combo.findData(detected_schema)
            if idx >= 0:
                self.schema_combo.setCurrentIndex(idx)
        self._rebuild_mapping_ui()
        self._update_preview()
        self.btn_import.setEnabled(True)
        self.status_label.setText(f"Loaded {len(self._rows)} row(s), {len(self._columns)} column(s).")

    # ── Mapping UI ────────────────────────────────────────────────────

    def _current_schema(self):
        return self.schema_combo.currentData()

    def _rebuild_mapping_ui(self):
        while self.mapping_form.rowCount():
            self.mapping_form.removeRow(0)
        self._field_combos = {}

        if not self._columns:
            return

        schema = self._current_schema()
        fields = list(POSITION_FIELDS) + list(SCHEMAS[schema])
        _, mapping = detect_schema(self._columns)  # best-effort guesses regardless of chosen schema
        if not mapping:
            mapping = suggest_column_mapping(self._columns, fields)

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

    # ── Import ────────────────────────────────────────────────────────

    def _do_import(self):
        schema = self._current_schema()
        column_map = self._current_column_map()
        required = list(POSITION_FIELDS) + list(SCHEMAS[schema])
        missing = [f for f in required if not column_map.get(f)]
        if missing:
            names = ", ".join(FIELD_DISPLAY_NAMES.get(f, f) for f in missing)
            QMessageBox.warning(self, "Missing mapping",
                                f"These required fields aren't mapped to a column:\n{names}")
            return

        length_unit = self.combo_length_unit.currentText()
        width_unit_sel = self.combo_width_unit.currentText()
        width_unit = None if width_unit_sel == "(same as length)" else width_unit_sel
        depth_unit = self.combo_depth_unit.currentText()
        slip_unit = self.combo_slip_unit.currentText()
        depth_convention = ("top" if self.combo_depth_convention.currentIndex() == 1
                            else "centroid")
        group = self.edit_group_label.text().strip() or None

        try:
            result = build_fault_rows_from_mapped_rows(
                self._rows, column_map, schema,
                length_unit=length_unit, width_unit=width_unit,
                depth_unit=depth_unit, slip_unit=slip_unit,
                depth_convention=depth_convention, group=group)
        except ValueError as e:
            QMessageBox.critical(self, "Import error", str(e))
            return

        self.imported_rows = result.rows
        msg = f"Imported {len(result.rows)} patch(es)."
        if result.n_skipped:
            msg += f" Skipped {result.n_skipped} row(s) with errors."
        self.status_label.setText(msg)

        if not result.rows:
            detail = "\n".join(result.errors[:10])
            QMessageBox.warning(self, "Nothing imported",
                                f"No valid patches could be built from this mapping.\n\n{detail}")
            return

        if result.errors:
            detail = "\n".join(result.errors[:15])
            more = f"\n… and {len(result.errors) - 15} more" if len(result.errors) > 15 else ""
            QMessageBox.information(self, "Import complete with some skipped rows",
                                    f"{msg}\n\nSkipped rows:\n{detail}{more}")

        self.accept()
