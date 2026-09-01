# -*- coding: utf-8 -*-
"""
Import dialog for earthquake (aftershock) catalogs: pick a source (file
or QGIS point layer), pick a time schema (auto-detected, overridable),
map columns (auto-suggested, overridable per field), preview, import.

Deliberately mirrors ui/focal_import_dialog.py's structure field-for-
field (source selection / schema combo / mapping form / preview table /
import+cancel buttons) -- same workflow, same widget layout, so anyone
who has used the focal-mechanism importer already knows this one.
"""

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QPushButton,
    QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QRadioButton, QButtonGroup, QGroupBox, QMessageBox
)
from qgis.PyQt.QtCore import Qt

from ..core.eq_catalog_import import (
    POSITION_FIELDS, OPTIONAL_SCALAR_FIELDS, SCHEMAS, SCHEMA_LABELS,
    SPLIT_TIME_OPTIONAL_FIELDS, suggest_column_mapping, detect_schema,
    build_events_from_mapped_rows,
)
from ..core.focal_mechanism_import import read_delimited_table
from ..core.observation_import import read_qgis_layer_table
from .dialog_utils import configure_resizable_dialog

FIELD_DISPLAY_NAMES = {
    "lon": "Longitude", "lat": "Latitude", "depth": "Depth (km, positive down)",
    "magnitude": "Magnitude (optional)", "label": "Event ID/label (optional)",
    "time": "Date+time", "year": "Year", "month": "Month", "day": "Day",
    "hour": "Hour (optional, default 0)", "minute": "Minute (optional, default 0)",
    "second": "Second (optional, default 0)",
}

NONE_LABEL = "(not mapped)"


class EQCatalogImportDialog(QDialog):
    """Returns imported events via .imported_events (List[EQCatalogEvent])
    and the packed lon/lat/depth(/epoch_s/magnitude) dict list via
    .imported_array (core.eq_catalog_import.events_to_eq_array() output,
    already time-sorted) after exec_() == QDialog.Accepted."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Earthquake Catalog")
        configure_resizable_dialog(self, 660, 560, min_width=420, min_height=360)
        self.imported_events = []
        self.imported_array = []
        self._columns = []
        self._rows = []
        self._field_combos = {}

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<b>Import Earthquake Catalog</b><br><i>From a delimited text file "
            "(CSV/TSV) or a point layer already loaded in this QGIS project. "
            "Different agencies use very different column names and time "
            "formats (combined date+time, separate year/month/day, or no "
            "time at all) — the mapping below is auto-suggested; review and "
            "correct it before importing.</i>"))

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

        # ── Time schema selection ───────────────────────────────────────
        schema_form = QFormLayout()
        self.schema_combo = QComboBox()
        for key in SCHEMAS:
            self.schema_combo.addItem(SCHEMA_LABELS[key], userData=key)
        self.schema_combo.currentIndexChanged.connect(self._rebuild_mapping_ui)
        schema_form.addRow("Time format:", self.schema_combo)
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
            self, "Select earthquake catalog", "",
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
                # read_qgis_layer_table() (core.observation_import) injects
                # "__geom_x__"/"__geom_y__" (WGS84 lon/lat, CRS-transformed
                # if the layer's CRS differs) into each ROW's dict, but its
                # returned `fields` list is attribute names only -- for a
                # layer with no explicit lon/lat attribute columns (common;
                # position lives in the geometry itself), those synthetic
                # columns would otherwise be invisible to the mapping combos
                # below. Surface them explicitly so lon/lat can still be
                # mapped from geometry.
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
        while self.mapping_form.rowCount():
            self.mapping_form.removeRow(0)
        self._field_combos = {}

        if not self._columns:
            return

        schema = self._current_schema()
        fields = list(POSITION_FIELDS) + list(OPTIONAL_SCALAR_FIELDS) + list(SCHEMAS[schema])
        if schema == "split_time":
            fields = fields + list(SPLIT_TIME_OPTIONAL_FIELDS)
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
        from ..core.eq_catalog_import import events_to_eq_array
        self.imported_array = events_to_eq_array(result.events)

        msg = f"Imported {len(result.events)} event(s)."
        if result.n_skipped:
            msg += f" Skipped {result.n_skipped} row(s) with errors."
        if result.n_missing_time and schema != "no_time":
            msg += f" {result.n_missing_time} row(s) imported with unusable/missing time."
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
