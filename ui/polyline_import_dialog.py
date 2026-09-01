# -*- coding: utf-8 -*-
"""
Dialog for importing fault segments from a QGIS line layer, letting the
user pick the layer, whether to use only selected features, and default
parameters for the fields not determined by geometry (depth, dip, rake,
slip, width) before the rows are added to the fault table for editing.
"""

from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QComboBox,
                                  QDoubleSpinBox, QPushButton, QLabel, QCheckBox,
                                  QHBoxLayout)
from qgis.core import QgsProject, QgsWkbTypes, QgsMapLayer


class PolylineImportDialog(QDialog):
    """Pick a line layer and default fault parameters for a polyline import."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Fault from QGIS Polyline")
        self.setMinimumWidth(480)
        self._rows = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<b>Import fault geometry from a line layer</b><br>"
            "Digitize a fault trace as a line in QGIS, then select it here. "
            "Each segment between vertices becomes one fault row: length "
            "and strike are computed automatically from the segment's "
            "endpoints. Depth, dip, rake, and slip use the defaults below "
            "and can be edited afterward in the fault table."))

        form = QFormLayout()

        self.layer_combo = QComboBox()
        self._populate_layers()
        form.addRow("Line layer:", self.layer_combo)

        self.only_selected_check = QCheckBox(
            "Use only selected features (unchecked = use all features)")
        form.addRow(self.only_selected_check)

        self.line_represents_combo = QComboBox()
        self.line_represents_combo.addItems([
            "Fault Top Projection (top edge, at the depth below)",
            "Surface Trace (extrapolated to z = 0, e.g. a blind/buried fault)",
        ])
        form.addRow("Digitized line represents:", self.line_represents_combo)
        line_represents_note = QLabel(
            "<i>If your line was digitized as where the fault would break "
            "the surface (Coulomb's own \"surface trace\" concept for a "
            "buried fault), choose Surface Trace — the actual top edge "
            "will be recovered by shifting down-dip by Top depth/tan(Dip), "
            "using the defaults below. If your line already traces the "
            "fault's actual top edge (e.g. Top depth = 0, or you mapped "
            "the buried trace directly), choose Fault Top Projection "
            "(no correction applied).</i>")
        line_represents_note.setWordWrap(True)
        form.addRow(line_represents_note)

        self.top_depth_spin = QDoubleSpinBox()
        self.top_depth_spin.setRange(0, 700)
        self.top_depth_spin.setValue(0.0)
        self.top_depth_spin.setSuffix(" km")
        form.addRow("Default TOP depth:", self.top_depth_spin)

        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.1, 500)
        self.width_spin.setValue(10.0)
        self.width_spin.setSuffix(" km")
        form.addRow("Default width:", self.width_spin)

        self.dip_spin = QDoubleSpinBox()
        self.dip_spin.setRange(0, 90)
        self.dip_spin.setValue(90.0)
        self.dip_spin.setSuffix(" °")
        form.addRow("Default dip:", self.dip_spin)

        self.rt_lat_spin = QDoubleSpinBox()
        self.rt_lat_spin.setRange(-100, 100)
        self.rt_lat_spin.setDecimals(3)
        self.rt_lat_spin.setValue(0.0)
        self.rt_lat_spin.setSuffix(" m")
        form.addRow("Default right-lateral slip:", self.rt_lat_spin)

        self.reverse_spin = QDoubleSpinBox()
        self.reverse_spin.setRange(-100, 100)
        self.reverse_spin.setDecimals(3)
        self.reverse_spin.setValue(0.0)
        self.reverse_spin.setSuffix(" m")
        form.addRow("Default reverse slip:", self.reverse_spin)

        layout.addLayout(form)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        btn_row = QHBoxLayout()
        self.btn_import = QPushButton("Import")
        self.btn_import.clicked.connect(self._do_import)
        btn_row.addWidget(self.btn_import)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

    def _populate_layers(self):
        self.layer_combo.clear()
        self._line_layers = []
        for layer in QgsProject.instance().mapLayers().values():
            if (layer.type() == QgsMapLayer.VectorLayer
                    and layer.geometryType() == QgsWkbTypes.LineGeometry):
                self._line_layers.append(layer)
                self.layer_combo.addItem(layer.name())

    def _do_import(self):
        if not self._line_layers:
            self.info_label.setText(
                "No line layers found in the project. Digitize a fault "
                "trace as a line layer first, then reopen this dialog.")
            return

        layer = self._line_layers[self.layer_combo.currentIndex()]
        from ..utils.polyline_import import faults_from_line_layer

        line_represents = ("surface_trace"
                          if self.line_represents_combo.currentIndex() == 1
                          else "top_edge")

        rows = faults_from_line_layer(
            layer,
            default_top_depth=self.top_depth_spin.value(),
            default_dip=self.dip_spin.value(),
            default_rt_lateral_slip=self.rt_lat_spin.value(),
            default_reverse_slip=self.reverse_spin.value(),
            default_width=self.width_spin.value(),
            only_selected=self.only_selected_check.isChecked(),
            line_represents=line_represents,
        )

        if not rows:
            self.info_label.setText(
                "No usable segments found (layer may be empty, or only "
                "selected features were requested but nothing is selected).")
            return

        self._rows = rows
        self.accept()

    def get_rows(self):
        """
        Returns the imported rows (list of dicts, see
        utils.polyline_import.faults_from_line_layer for the schema), or
        None if the dialog was cancelled / nothing was imported.
        """
        return self._rows
