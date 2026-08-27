# -*- coding: utf-8 -*-
"""Read-only results table showing ΔCFF resolved on individual receiver faults."""

from qgis.PyQt.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor

RESULT_COLUMNS = ["Lon", "Lat", "Depth (km)", "Strike (°)", "Dip (°)", "Rake (°)",
                  "ΔCFF (bar)", "Shear (bar)", "Normal (bar)", "Method"]


class ReceiverResultsWidget(QTableWidget):
    """Displays one row per receiver fault with its resolved ΔCFF."""

    def __init__(self, parent=None):
        super().__init__(0, len(RESULT_COLUMNS), parent)
        self.setHorizontalHeaderLabels(RESULT_COLUMNS)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setEditTriggers(QTableWidget.NoEditTriggers)

    def set_results(self, results):
        """
        results: list of dicts as returned by
        core.okada_engine.compute_cff_on_receiver_faults(), i.e.
        {"fault": FaultParameters, "cff_mpa": float, "shear_mpa": float,
         "normal_mpa": float, "used_dc3d": bool}
        """
        self.setRowCount(0)
        for res in results:
            f = res["fault"]
            row = self.rowCount()
            self.insertRow(row)

            cff_bar = res["cff_mpa"] * 10
            shear_bar = res["shear_mpa"] * 10
            normal_bar = res["normal_mpa"] * 10
            method = "DC3D" if res["used_dc3d"] else "Surface (z=0)"

            vals = [
                f"{f.lon:.5f}", f"{f.lat:.5f}", f"{f.depth:.3f}",
                f"{f.strike:.1f}", f"{f.dip:.1f}", f"{f.rake:.1f}",
                f"{cff_bar:+.5f}",
                f"{shear_bar:+.5f}" if shear_bar == shear_bar else "n/a",  # NaN check
                f"{normal_bar:+.5f}" if normal_bar == normal_bar else "n/a",
                method,
            ]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 6:  # ΔCFF column: color by sign
                    if cff_bar > 0:
                        item.setBackground(QColor(255, 210, 210))
                    elif cff_bar < 0:
                        item.setBackground(QColor(210, 220, 255))
                self.setItem(row, col, item)
