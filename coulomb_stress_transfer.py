# -*- coding: utf-8 -*-
"""Coulomb Stress Transfer QGIS Plugin — Main Plugin Class."""

import os

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import (QAction, QDialog, QVBoxLayout, QHBoxLayout,
                                  QLabel, QPushButton, QTextEdit, QLineEdit,
                                  QFileDialog)

from .ui.main_dialog import CoulombMainDialog
from .core.okada_engine import (
    _has_okada_wrapper, check_external_python,
    _get_external_python_path, _set_external_python_path,
)


class DependencyDialog(QDialog):
    """
    Lets the user point the plugin at an EXTERNAL Python interpreter
    (e.g. a conda/venv environment) that already has okada-wrapper
    installed and working.

    QGIS's own bundled Python typically lacks Python.h / python3XX.lib,
    so compiled extensions like okada-wrapper cannot be built or imported
    inside QGIS's Python process on most systems. Rather than trying to
    install it there, the plugin calls out to your existing external
    Python via subprocess whenever exact depth-dependent CFF is needed.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Coulomb Stress Transfer — Dependency Notice")
        self.setMinimumWidth(640)
        layout = QVBoxLayout(self)

        self._refresh_status(layout)

        explanation = QLabel(
            "<hr><b>Why an external Python?</b><br>"
            "okada-wrapper wraps a compiled Fortran routine (Okada 1992 DC3D) "
            "and needs Python development headers to build. QGIS's bundled "
            "Python usually does not include these, so okada-wrapper cannot "
            "be built or imported inside QGIS itself — even with a working "
            "Fortran compiler installed.<br><br>"
            "If you already have a separate Python installation (e.g. conda, "
            "venv, or a python.org install) where "
            "<code>pip install okada-wrapper</code> succeeded, point this "
            "plugin at that Python below. The plugin will call it via "
            "subprocess whenever exact depth-dependent CFF is requested.<br><br>"
            "<b>Without this:</b> CFF is computed at the surface (z=0), which "
            "matches the standard Coulomb stress map and is validated against "
            "Coulomb 3.4.2 (r=0.983, 99.6% sign agreement). The quadrant sign "
            "pattern is correct at all depths — only exact magnitudes at "
            "depth require the external Python.<br><br>"
            "<b>Don't have okada-wrapper anywhere yet?</b> In any terminal "
            "with a normal (non-QGIS) Python that has a Fortran compiler "
            "available, run:<br>"
            "<code>pip install okada-wrapper</code><br>"
            "then come back here and browse to that Python's "
            "<code>python.exe</code> / <code>python3</code>."
        )
        explanation.setWordWrap(True)
        explanation.setOpenExternalLinks(True)
        layout.addWidget(explanation)

        layout.addWidget(QLabel("<hr><b>External Python interpreter:</b>"))
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit(_get_external_python_path() or "")
        self.path_edit.setPlaceholderText(
            r"e.g. C:\Users\you\miniconda3\envs\py39_env\python.exe")
        path_row.addWidget(self.path_edit)
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse)
        path_row.addWidget(self.btn_browse)
        layout.addLayout(path_row)

        btn_row = QHBoxLayout()
        self.btn_test = QPushButton("Test this Python")
        self.btn_test.clicked.connect(self._test)
        btn_row.addWidget(self.btn_test)
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self._save)
        btn_row.addWidget(self.btn_save)
        layout.addLayout(btn_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(90)
        self.log.setPlaceholderText("Test results will appear here…")
        layout.addWidget(self.log)

        btn_continue = QPushButton("Continue")
        btn_continue.setDefault(True)
        btn_continue.clicked.connect(self.accept)
        layout.addWidget(btn_continue)

    def _refresh_status(self, layout):
        if getattr(self, "_status_label", None) is not None:
            self._status_label.deleteLater()
        installed = _has_okada_wrapper()
        if installed:
            text = "✅  External Python with okada-wrapper is configured and working."
            color = "green"
        else:
            text = "⚠️  No working external Python with okada-wrapper is configured yet."
            color = "darkorange"
        self._status_label = QLabel(f'<span style="color:{color}">{text}</span>')
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

    def _browse(self):
        filt = "python.exe (python.exe)" if os.name == "nt" else "python3 (python3*)"
        path, _ = QFileDialog.getOpenFileName(
            self, "Locate external Python interpreter", "", f"{filt};;All files (*)")
        if path:
            self.path_edit.setText(path)

    def _test(self):
        path = self.path_edit.text().strip()
        ok, msg = check_external_python(path)
        prefix = "✅ " if ok else "❌ "
        self.log.setPlainText(prefix + msg)

    def _save(self):
        path = self.path_edit.text().strip()
        if not path:
            self.log.setPlainText("❌ Please enter or browse to a Python path first.")
            return
        _set_external_python_path(path)
        ok, msg = check_external_python(path)
        if ok:
            self.log.setPlainText(f"✅ Saved and verified: {msg}")
        else:
            self.log.setPlainText(
                f"⚠️ Saved, but verification failed: {msg}\n"
                f"You can still use the plugin — it will fall back to the "
                f"surface (z=0) formula until this is fixed.")


class CoulombStressTransferPlugin:
    """Main QGIS plugin class."""

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = "&Coulomb Stress Transfer"
        self.toolbar = self.iface.addToolBar("CoulombStressTransfer")
        self.toolbar.setObjectName("CoulombStressTransfer")
        self.dialog = None
        self._dep_shown = False

    def add_action(self, icon_path, text, callback, enabled_flag=True,
                   add_to_menu=True, add_to_toolbar=True,
                   status_tip=None, whats_this=None, parent=None):
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)
        if status_tip: action.setStatusTip(status_tip)
        if whats_this: action.setWhatsThis(whats_this)
        if add_to_toolbar: self.toolbar.addAction(action)
        if add_to_menu: self.iface.addPluginToMenu(self.menu, action)
        self.actions.append(action)
        return action

    def initGui(self):
        icon_path = os.path.join(self.plugin_dir, "resources", "icon.png")
        self.add_action(icon_path, text="Coulomb Stress Transfer", callback=self.run,
                        parent=self.iface.mainWindow(),
                        status_tip="Calculate Coulomb Failure Function changes",
                        whats_this="Compute and visualize Coulomb Stress Transfer")
        self.add_action(icon_path, text="Check / Configure Depth-Dependent CFF…",
                        callback=self.show_dependency_dialog,
                        parent=self.iface.mainWindow(), add_to_toolbar=False,
                        status_tip="Configure external Python with okada-wrapper for exact depth CFF")

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu(self.menu, action)
            self.iface.removeToolBarIcon(action)
        del self.toolbar

    def show_dependency_dialog(self):
        DependencyDialog(self.iface.mainWindow()).exec_()

    def run(self):
        if not self._dep_shown and not _has_okada_wrapper():
            self._dep_shown = True
            DependencyDialog(self.iface.mainWindow()).exec_()
        if self.dialog is None:
            self.dialog = CoulombMainDialog(self.iface)
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
