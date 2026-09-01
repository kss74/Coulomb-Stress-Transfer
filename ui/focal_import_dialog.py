# -*- coding: utf-8 -*-
"""
Import dialog for focal-mechanism catalogs: pick a source (file or
QGIS point layer), pick a schema (auto-detected, overridable), map
columns (auto-suggested, overridable per field), preview, import.
"""

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QPushButton,
    QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QRadioButton, QButtonGroup, QGroupBox, QMessageBox
)
from qgis.PyQt.QtCore import Qt

from ..core.focal_mechanism_import import (
    POSITION_FIELDS, POSITION_OPTIONAL_FIELDS, SCHEMAS, SCHEMA_LABELS,
    suggest_column_mapping, detect_schema, read_delimited_table,
    build_events_from_mapped_rows,
)

FIELD_DISPLAY_NAMES = {
    "lon": "Longitude", "lat": "Latitude", "depth": "Depth (km)",
    "magnitude": "Magnitude (optional)", "label": "Event ID/label (optional)",
    "strike1": "Strike 1 (°)", "dip1": "Dip 1 (°)", "rake1": "Rake 1 (°)",
    "strike2": "Strike 2 (°)", "dip2": "Dip 2 (°)", "rake2": "Rake 2 (°)",
    "mnn": "Mnn", "mee": "Mee", "mdd": "Mdd", "mne": "Mne", "mnd": "Mnd", "med": "Med",
    "mrr": "Mrr", "mtt": "Mtt", "mpp": "Mpp", "mrt": "Mrt", "mrp": "Mrp", "mtp": "Mtp",
}

NONE_LABEL = "(not mapped)"


class FocalMechanismImportDialog(QDialog):
    """Returns imported events via .imported_events after exec_() == Accepted."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Focal Mechanism Catalog")
        self.setMinimumWidth(640)
        self.imported_events = []
        self._columns = []
        self._rows = []
        self._field_combos = {}

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<b>Import Focal Mechanisms</b><br><i>From a delimited text file "
            "(CSV/TSV) or a point layer already loaded in this QGIS project. "
            "Column names rarely match this plugin's field names exactly — "
            "the mapping below is auto-suggested; review and correct it "
            "before importing.</i>"))

        # ── Source selection ────────────────────────────────────────────
        source_group = QGroupBox("Source")
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

        # ── Schema selection ────────────────────────────────────────────
        schema_form = QFormLayout()
        self.schema_combo = QComboBox()
        for key in SCHEMAS:
            self.schema_combo.addItem(SCHEMA_LABELS[key], userData=key)
        self.schema_combo.currentIndexChanged.connect(self._rebuild_mapping_ui)
        schema_form.addRow("Data schema:", self.schema_combo)
        layout.addLayout(schema_form)

        # ── Column mapping ──────────────────────────────────────────────
        self.mapping_group = QGroupBox("Column mapping")
        self.mapping_form = QFormLayout(self.mapping_group)
        layout.addWidget(self.mapping_group)

        # ── Preview ──────────────────────────────────────────────────────
        layout.addWidget(QLabel("<b>Preview (first 10 rows)</b>"))
        self.preview_table = QTableWidget(0, 0)
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.preview_table.setMaximumHeight(180)
        layout.addWidget(self.preview_table)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        self.btn_import = QPushButton("Import")
        self.btn_import.clicked.connect(self._do_import)
        self.btn_import.setEnabled(False)
        btn_row.addWidget(self.btn_import)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)
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
            self, "Select focal mechanism catalog", "",
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
                from ..core.focal_mechanism_import import read_qgis_layer_table
                self._columns, self._rows = read_qgis_layer_table(layer)
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

    # ── Mapping UI ───────────────────────────────────────────────────────

    def _current_schema(self):
        return self.schema_combo.currentData()

    def _rebuild_mapping_ui(self):
        # clear existing rows
        while self.mapping_form.rowCount():
            self.mapping_form.removeRow(0)
        self._field_combos = {}

        if not self._columns:
            return

        schema = self._current_schema()
        fields = list(POSITION_FIELDS) + list(POSITION_OPTIONAL_FIELDS) + list(SCHEMAS[schema])
        _, mapping = detect_schema(self._columns)  # best-effort guesses regardless of chosen schema
        if not mapping:
            mapping = suggest_column_mapping(self._columns, fields)

        for field in fields:
            combo = QComboBox()
            combo.addItem(NONE_LABEL)
            combo.addItems(self._columns)
            guess = mapping.get(field)
            if guess:
                combo.setCurrentText(guess)
            self.mapping_form.addRow(FIELD_DISPLAY_NAMES.get(field, field) + ":", combo)
            self._field_combos[field] = combo

    def _current_column_map(self):
        out = {}
        for field, combo in self._field_combos.items():
            val = combo.currentText()
            out[field] = None if val == NONE_LABEL else val
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

    # ── Import ──────────────────────────────────────────────────────────

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

        result = build_events_from_mapped_rows(self._rows, column_map, schema)
        self.imported_events = result.events
        msg = f"Imported {len(result.events)} event(s)."
        if result.n_skipped:
            msg += f" Skipped {result.n_skipped} row(s) with errors."
        self.status_label.setText(msg)

        if not result.events:
            detail = "\n".join(result.errors[:10])
            QMessageBox.warning(self, "Nothing imported",
                                f"No valid events could be built from this mapping.\n\n{detail}")
            return

        if result.errors:
            detail = "\n".join(result.errors[:15])
            more = f"\n… and {len(result.errors) - 15} more" if len(result.errors) > 15 else ""
            QMessageBox.information(self, "Import complete with some skipped rows",
                                    f"{msg}\n\nSkipped rows:\n{detail}{more}")

        self.accept()
