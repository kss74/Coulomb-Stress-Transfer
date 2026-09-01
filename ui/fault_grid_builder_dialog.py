# -*- coding: utf-8 -*-
"""
Dialog for core.fault_grid_builder.build_variable_dip_fault_grid() --
construct a grid of independent fixed-size fault sub-patches (e.g.
1 km x 1 km) directly into the Source Faults table, with each down-dip
ROW allowed its own dip and width. This is the plugin's third fault-
construction path alongside "+ Add fault" (one row, typed by hand) and
"📋 Import fault-patch table…" (many rows, read from an external file):
this one instead GENERATES many rows from a compact on-screen spec,
for building a synthetic segmented/listric fault (dip changing with
depth) without needing an external file at all -- the motivating
example being a GSI/Kobayashi-et-al.-2018-style fault-patch grid,
where the plane's dip steepens from ~30° near the surface to ~50° at
depth across a stack of 1 km x 1 km patches.

Row stacking is always CONTINUOUS (row i+1's top edge = row i's bottom
edge) -- see core.fault_grid_builder's module docstring for the
depth/position-stepping derivation and its verification against that
GSI dataset's own row-to-row depth stepping.

Like "Import fault-patch table…", this tool only builds GEOMETRY:
every generated patch starts at the uniform rake/slip given here
(default 0/0) and is added to the table as an independent row with
Lon/Lat mode "Centroid" -- slip is filled in afterward by hand, via a
slip inversion, or by editing rows directly, exactly as for an
imported table.
"""

import math

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QMessageBox, QSpinBox, QWidget, QScrollArea
)
from qgis.PyQt.QtCore import Qt

from ..core.fault_grid_builder import build_variable_dip_fault_grid, FaultGridRowSpec
from .dialog_utils import configure_resizable_dialog

COL_ROW, COL_WIDTH, COL_DIP = range(3)


class FaultGridBuilderDialog(QDialog):
    """Returns generated patch rows via .built_rows after
    exec_() == Accepted. Each row is a dict with the same shape as
    core.fault_table_import's output, ready for the caller (see
    ui.fault_table_widget.FaultTableWidget._open_fault_grid_builder_dialog())
    to hand to add_row()."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Construct Fault Grid (Variable Dip)")
        configure_resizable_dialog(self, 620, 640, min_width=420, min_height=380)
        self.built_rows = []

        outer_layout = QVBoxLayout(self)

        content = QWidget()
        layout = QVBoxLayout(content)

        layout.addWidget(QLabel(
            "<b>Construct Fault Grid</b><br><i>Generates a grid of "
            "independent, fixed-size sub-patches (e.g. 1 km × 1 km) "
            "directly into the Source Faults table. Give each down-dip "
            "ROW below its own width and dip to build a segmented/"
            "listric fault whose dip changes with depth -- rows stack "
            "continuously (each row's bottom edge is the next row's top "
            "edge). Slip is left at the uniform value below; fill it in "
            "afterward by editing rows, or invert for slip.</i>"))

        # ── Fault-wide parameters ──────────────────────────────────────
        top_group = QGroupBox("Fault-wide parameters")
        top_form = QFormLayout(top_group)

        self.edit_name = QLineEdit("Fault")
        top_form.addRow("Name / Group:", self.edit_name)

        self.edit_lon = QLineEdit("0.0")
        top_form.addRow("Start Lon (° — row 1, col 1 top-edge corner):", self.edit_lon)
        self.edit_lat = QLineEdit("0.0")
        top_form.addRow("Start Lat (°):", self.edit_lat)
        self.edit_top_depth = QLineEdit("0.0")
        top_form.addRow("Top depth of row 1 (km):", self.edit_top_depth)
        self.edit_strike = QLineEdit("0.0")
        top_form.addRow("Strike (° clockwise from North):", self.edit_strike)

        self.edit_patch_length = QLineEdit("1.0")
        top_form.addRow("Patch length (km, fixed, along strike):", self.edit_patch_length)
        self.spin_n_cols = QSpinBox()
        self.spin_n_cols.setRange(1, 500)
        self.spin_n_cols.setValue(10)
        top_form.addRow("Number of columns (along strike):", self.spin_n_cols)

        self.edit_rake = QLineEdit("0.0")
        top_form.addRow("Initial rake (°, Aki-Richards):", self.edit_rake)
        self.edit_slip = QLineEdit("0.0")
        top_form.addRow("Initial slip (m):", self.edit_slip)

        layout.addWidget(top_group)

        # ── Down-dip rows (width + dip per row) ─────────────────────────
        rows_group = QGroupBox("Down-dip rows (top-to-bottom)")
        rows_layout = QVBoxLayout(rows_group)
        rows_layout.addWidget(QLabel(
            "<i>Each row applies its own Width/Dip to every patch in "
            "that row. Add rows to extend the fault down-dip; a row's "
            "dip may differ from its neighbours to build a listric "
            "fault (e.g. the widely-used pattern of ~30° near the "
            "surface steepening to ~50° at depth).</i>"))

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Row", "Width (km)", "Dip (°)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        rows_layout.addWidget(self.table)

        row_btn_row = QHBoxLayout()
        self.btn_add_row = QPushButton("+ Add row")
        self.btn_remove_row = QPushButton("- Remove selected row(s)")
        self.btn_add_row.clicked.connect(self._add_table_row)
        self.btn_remove_row.clicked.connect(self._remove_selected_rows)
        row_btn_row.addWidget(self.btn_add_row)
        row_btn_row.addWidget(self.btn_remove_row)
        row_btn_row.addStretch()
        rows_layout.addLayout(row_btn_row)

        layout.addWidget(rows_group)

        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer_layout.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        self.btn_preview = QPushButton("Preview")
        self.btn_build = QPushButton("Build")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_preview.clicked.connect(self._update_preview)
        self.btn_build.clicked.connect(self._do_build)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_preview)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_build)
        btn_row.addWidget(self.btn_cancel)
        outer_layout.addLayout(btn_row)

        # Start with two example rows (matching the common "shallow
        # gentler dip, steeper at depth" listric pattern) rather than a
        # single uninformative blank row.
        self._add_table_row(width_km=1.0, dip_deg=30.0)
        self._add_table_row(width_km=1.0, dip_deg=50.0)
        self._update_preview()

    # ── Row-table helpers ────────────────────────────────────────────

    def _add_table_row(self, checked=False, width_km=1.0, dip_deg=45.0):
        row = self.table.rowCount()
        self.table.insertRow(row)
        row_item = QTableWidgetItem(str(row + 1))
        row_item.setFlags(row_item.flags() & ~Qt.ItemIsEditable)
        row_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, COL_ROW, row_item)
        w_item = QTableWidgetItem(f"{width_km:.6g}")
        w_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, COL_WIDTH, w_item)
        d_item = QTableWidgetItem(f"{dip_deg:.6g}")
        d_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, COL_DIP, d_item)

    def _remove_selected_rows(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self._renumber_rows()

    def _renumber_rows(self):
        for row in range(self.table.rowCount()):
            item = self.table.item(row, COL_ROW)
            if item is not None:
                item.setText(str(row + 1))

    # ── Build ────────────────────────────────────────────────────────

    def _read_row_specs(self):
        """Returns (row_specs, error_message). error_message is None on
        success."""
        n_rows = self.table.rowCount()
        if n_rows == 0:
            return None, "Add at least one down-dip row first."
        specs = []
        for row in range(n_rows):
            try:
                width_km = float(self.table.item(row, COL_WIDTH).text())
                dip_deg = float(self.table.item(row, COL_DIP).text())
            except (ValueError, AttributeError):
                return None, f"Row {row + 1} has an invalid number."
            if width_km <= 0:
                return None, f"Row {row + 1}: width must be > 0."
            if not (0.0 < dip_deg <= 90.0):
                return None, f"Row {row + 1}: dip must be between 0 (exclusive) and 90."
            specs.append(FaultGridRowSpec(width_km=width_km, dip_deg=dip_deg))
        return specs, None

    def _read_common_params(self):
        """Returns (params_dict, error_message)."""
        try:
            lon = float(self.edit_lon.text())
            lat = float(self.edit_lat.text())
            top_depth = float(self.edit_top_depth.text())
            strike = float(self.edit_strike.text())
            patch_length = float(self.edit_patch_length.text())
            rake = float(self.edit_rake.text())
            slip = float(self.edit_slip.text())
        except ValueError:
            return None, "One of the fault-wide parameters is not a valid number."
        if patch_length <= 0:
            return None, "Patch length must be > 0."
        n_cols = self.spin_n_cols.value()
        name = self.edit_name.text().strip() or "Fault"
        return dict(lon=lon, lat=lat, top_depth=top_depth, strike=strike,
                   patch_length=patch_length, rake=rake, slip=slip,
                   n_cols=n_cols, name=name), None

    def _update_preview(self):
        specs, err = self._read_row_specs()
        if err:
            self.preview_label.setText(f"<i>{err}</i>")
            return
        params, err = self._read_common_params()
        if err:
            self.preview_label.setText(f"<i>{err}</i>")
            return
        n_rows = len(specs)
        n_cols = params["n_cols"]
        bottom_depth = params["top_depth"] + sum(
            s.width_km * math.sin(math.radians(s.dip_deg)) for s in specs)
        self.preview_label.setText(
            f"Will create <b>{n_rows} × {n_cols} = {n_rows * n_cols}</b> "
            f"patch(es), spanning top depth {params['top_depth']:.3g} km "
            f"to bottom depth {bottom_depth:.3g} km.")

    def _do_build(self):
        specs, err = self._read_row_specs()
        if err:
            QMessageBox.warning(self, "Construct Fault Grid", err)
            return
        params, err = self._read_common_params()
        if err:
            QMessageBox.warning(self, "Construct Fault Grid", err)
            return

        try:
            result = build_variable_dip_fault_grid(
                start_lon=params["lon"], start_lat=params["lat"],
                top_depth_km=params["top_depth"], strike_deg=params["strike"],
                n_cols=params["n_cols"], patch_length_km=params["patch_length"],
                row_specs=specs, rake_deg=params["rake"], slip_m=params["slip"],
                name_prefix=params["name"], group=params["name"])
        except ValueError as e:
            QMessageBox.critical(self, "Construct Fault Grid", str(e))
            return

        self.built_rows = result.rows
        self.accept()
