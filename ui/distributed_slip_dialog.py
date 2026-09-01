# -*- coding: utf-8 -*-
"""
Dialog for editing per-sub-patch ("distributed") slip on a subdivided
source fault row of FaultTableWidget.

Normally, subdividing a source fault (Subdiv.(L)/Subdiv.(W) > 1) is a
uniform-slip, display-time-only split: every sub-patch inherits the
parent row's single right-lateral/reverse slip, so the total computed
stress field is identical to the un-subdivided fault (Okada's
rectangular-fault solution already integrates slip exactly over the
whole patch). This dialog lets the user instead give one or more
sub-patches their OWN slip, turning the subdivision into a genuine
variable-slip source: each sub-patch becomes an independent Okada
dislocation with its own moment, so the computed stress field actually
changes.

Patch indexing / ordering matches core.okada_engine.FaultParameters.
subdivide() exactly: i = down-dip (width) index (0..n_width-1), j =
along-strike (length) index (0..n_length-1), flat order i*n_length+j
(width outer, length inner). Patch labels use the SAME "A, B, C, ..."
annex-letter convention as everywhere else sub-areas are named in this
plugin (fault subdivision naming shown elsewhere as "Fault 1-A",
"Fault 1-B", ...; merged-fault-group naming) -- see
fault_table_widget._annex_labels().
"""

from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                  QTableWidget, QTableWidgetItem, QPushButton,
                                  QHeaderView, QMessageBox)
from qgis.PyQt.QtCore import Qt

from .fault_table_widget import _annex_labels
from .dialog_utils import configure_resizable_dialog

COL_PATCH, COL_ALONG_STRIKE_IDX, COL_DOWN_DIP_IDX, COL_RTLAT, COL_REVERSE = range(5)


class DistributedSlipDialog(QDialog):
    """
    Edit right-lateral / reverse slip individually for each sub-patch of
    a subdivided source fault row, instead of every sub-patch inheriting
    the parent row's single uniform slip.
    """

    def __init__(self, parent, fault_name, n_length, n_width,
                default_rt_lateral, default_reverse, existing_overrides=None):
        """
        fault_name          : the row's own name, for the dialog title/label.
        n_length, n_width   : this row's Subdiv.(L) / Subdiv.(W) values.
        default_rt_lateral,
        default_reverse     : the row's own uniform slip -- the value any
                              sub-patch reverts to if not overridden.
        existing_overrides  : optional dict {(i, j): (rt_lateral_slip,
                              reverse_slip)} of already-stored overrides
                              to pre-populate (e.g. re-opening this
                              dialog on a row that already has some).
        """
        super().__init__(parent)
        self.setWindowTitle(f"Distributed slip — {fault_name}")
        configure_resizable_dialog(self, 560, 420, min_width=340, min_height=260)

        self._n_length = n_length
        self._n_width = n_width
        self._default_rt = default_rt_lateral
        self._default_rev = default_reverse
        self._overrides = None  # result, set on accept

        n_patches = n_length * n_width
        labels = _annex_labels(n_patches)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"<b>{fault_name}</b> is divided into {n_length}×{n_width} = "
            f"{n_patches} sub-patch(es). Give any patch its own right-"
            f"lateral / reverse slip below (Coulomb sign convention, "
            f"metres) to make it an independent source; patches left at "
            f"the uniform default (Rt-lateral={default_rt_lateral:.4g}, "
            f"Reverse={default_reverse:.4g}) keep behaving exactly as a "
            f"plain subdivision does today. Patches are labeled with the "
            f"same 'A, B, C, ...' convention used for 'Fault 1-A', "
            f"'Fault 1-B', etc. elsewhere in this table."))

        self.table = QTableWidget(n_patches, 5)
        self.table.setHorizontalHeaderLabels(
            ["Patch", "Along-strike idx (j)", "Down-dip idx (i)",
             "Rt-lateral slip (m)", "Reverse slip (m)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        existing_overrides = existing_overrides or {}
        flat = 0
        for i in range(n_width):
            for j in range(n_length):
                label_item = QTableWidgetItem(labels[flat])
                label_item.setFlags(label_item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(flat, COL_PATCH, label_item)

                j_item = QTableWidgetItem(str(j))
                j_item.setFlags(j_item.flags() & ~Qt.ItemIsEditable)
                j_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(flat, COL_ALONG_STRIKE_IDX, j_item)

                i_item = QTableWidgetItem(str(i))
                i_item.setFlags(i_item.flags() & ~Qt.ItemIsEditable)
                i_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(flat, COL_DOWN_DIP_IDX, i_item)

                rt, rev = existing_overrides.get((i, j), (default_rt_lateral, default_reverse))
                rt_item = QTableWidgetItem(f"{rt:.6g}")
                rt_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(flat, COL_RTLAT, rt_item)
                rev_item = QTableWidgetItem(f"{rev:.6g}")
                rev_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(flat, COL_REVERSE, rev_item)

                flat += 1

        btn_row = QHBoxLayout()
        self.btn_reset = QPushButton("Reset all to uniform (parent row's slip)")
        self.btn_reset.clicked.connect(self._reset_uniform)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        ok_row = QHBoxLayout()
        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_ok.clicked.connect(self._on_accept)
        self.btn_cancel.clicked.connect(self.reject)
        ok_row.addStretch()
        ok_row.addWidget(self.btn_ok)
        ok_row.addWidget(self.btn_cancel)
        layout.addLayout(ok_row)

    def _reset_uniform(self):
        for row in range(self.table.rowCount()):
            rt_item = QTableWidgetItem(f"{self._default_rt:.6g}")
            rt_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, COL_RTLAT, rt_item)
            rev_item = QTableWidgetItem(f"{self._default_rev:.6g}")
            rev_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, COL_REVERSE, rev_item)

    def _on_accept(self):
        overrides = {}
        for row in range(self.table.rowCount()):
            try:
                j = int(self.table.item(row, COL_ALONG_STRIKE_IDX).text())
                i = int(self.table.item(row, COL_DOWN_DIP_IDX).text())
                rt = float(self.table.item(row, COL_RTLAT).text())
                rev = float(self.table.item(row, COL_REVERSE).text())
            except (ValueError, AttributeError):
                QMessageBox.warning(
                    self, "Distributed slip",
                    f"Row {row + 1} has an invalid number; fix it or Cancel.")
                return
            # Only store an explicit override where it actually differs
            # from the uniform default -- keeps stored data compact and
            # makes "Reset to uniform" meaningfully equivalent to "no
            # distributed slip at all" rather than a pile of no-op entries.
            if abs(rt - self._default_rt) > 1e-12 or abs(rev - self._default_rev) > 1e-12:
                overrides[(i, j)] = (rt, rev)
        self._overrides = overrides
        self.accept()

    def get_overrides(self):
        """dict {(i, j): (rt_lateral_slip, reverse_slip)} of patches that
        differ from the parent row's uniform slip. Empty dict = fully
        uniform (equivalent to no distributed slip / a plain subdivision)."""
        return self._overrides or {}
