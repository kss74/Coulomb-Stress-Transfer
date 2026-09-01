# -*- coding: utf-8 -*-
"""
Popup window for the cross-section plot.

Addresses "cross section layout is affected by the window height and
does not get displayed properly" (2026-08-18): the cross-section used
to draw into PlotWidget's small embedded canvas, sized by whatever
space main_dialog's stacked tabs/controls left over. A standalone
QMainWindow gets its own resizable, minimize/maximize/close-capable
top-level window instead, sized independently of the main dialog, and
is simply reused (figure swapped, not recreated) across recomputes so
the user's chosen window size/position persists between runs.

Parenting (2026-08-19 follow-up, "window didn't reappear after
closing and reopening the plugin"): this window must NOT be parented
to CoulombMainDialog. Even with the independent Qt.Window flag set, a
widget that has a Qt *parent* still gets hidden when that parent is
hidden/closed -- Qt propagates visibility down the ownership tree for
window-flagged children exactly as it does for embedded ones, it just
doesn't also resize/reposition them. So `CrossSectionWindow(main_dlg)`
closes along with the plugin's main dialog, and nothing later calls
show() on it again except a fresh "compute cross-section" -- from the
user's side that reads as "the window didn't come back." ui.main_dialog
now parents this to `iface.mainWindow()` instead (QGIS's own top-level
window, which outlives the plugin dialog for the whole QGIS session),
so this window's visibility is independent of whether the plugin
dialog itself is open. This is a reasoned fix based on Qt's documented
parent/child visibility propagation, not something run against real
QGIS -- still needs the standing in-QGIS smoke test to confirm.
"""

from qgis.PyQt.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QFileDialog
from qgis.PyQt.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class CrossSectionWindow(QMainWindow):
    """
    Standalone, resizable popup showing the composed cross-section
    Figure (core.cross_section_plot.build_cross_section_figure output).
    Normal window flags -- minimize/maximize/close all present, same as
    any other top-level window, addressing point 9 of the 2026-08-18
    cross-section overhaul request.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Coulomb Cross-Section")
        self.setWindowFlags(Qt.Window)
        # Defensive/explicit: this window's C++ object must survive
        # being closed/hidden so its Python-side reference
        # (main_dialog._xs_window) stays valid across recomputes. False
        # is Qt's own default for WA_DeleteOnClose, but setting it
        # explicitly documents that this is a load-bearing assumption
        # here, not an accident of the default -- see the parenting
        # note below on why closing the *parent* was the actual bug.
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.resize(1000, 750)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)

        self._canvas = None
        self._toolbar = None
        self._layout = layout

    def set_figure(self, fig):
        """
        Replace the currently-displayed figure. Tears down the old
        canvas/toolbar (matplotlib Figures aren't meant to be
        re-parented onto a live canvas) and builds fresh ones around
        the new Figure -- cheap relative to the cross-section
        computation itself, and avoids the "stale canvas holding a
        reference to a closed figure" failure mode.
        """
        if self._canvas is not None:
            self._layout.removeWidget(self._toolbar)
            self._layout.removeWidget(self._canvas)
            self._toolbar.deleteLater()
            self._canvas.deleteLater()

        self._canvas = FigureCanvas(fig)
        self._toolbar = NavigationToolbar(self._canvas, self)
        self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas)
        self._canvas.draw()

    def save_image(self, parent=None):
        if self._canvas is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            parent or self, "Save Cross-Section Image", "",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;EPS (*.eps)")
        if path:
            self._canvas.figure.savefig(path, dpi=300, bbox_inches="tight")

    def show_and_raise(self):
        self.show()
        self.raise_()
        self.activateWindow()
