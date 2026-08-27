# -*- coding: utf-8 -*-
"""
Dialog for estimating fault length, width, and average slip from moment
magnitude using published empirical scaling relations, with the estimated
slip correctly decomposed into right-lateral + reverse components based
on rake, rather than being dumped entirely into one column.
"""

import math

from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QComboBox,
                                  QDoubleSpinBox, QPushButton, QLabel, QHBoxLayout)

from ..core.scaling_relations import (
    SCALING_RELATIONS, FAULT_STYLES, SCALING_RELATION_NOTES,
    MOMENT_BALANCED_RELATION_NAMES, compute_scaling_result,
)

# Relations that ignore style/rake entirely (subduction interface): one
# regression covers all interface events, so the style combo has no
# effect on length/width. Kept enabled (rake still drives the
# right-lateral/reverse slip split for whatever the user sets) but its
# selection doesn't change L/W the way it does for crustal relations.
_STYLE_IGNORED_RELATION_NAMES = {
    "Thingbaijam, Mai & Goda (2017) — subduction interface",
    "Strasser, Arango & Bommer (2010) — subduction interface",
}

# Representative rake (degrees, Aki-Richards) for each canonical style,
# used to decompose the scaling relation's scalar slip into right-lateral
# and reverse components. "all" has no single representative rake, since
# it's an omnidirectional/mixed regression category.
_STYLE_RAKE_DEG = {
    "strike-slip": 0.0,    # pure lateral (left-lateral in Aki-Richards; see note below)
    "reverse": 90.0,       # pure reverse/thrust
    "normal": -90.0,       # pure normal
}


class ScalingRelationsDialog(QDialog):
    """
    Lets the user pick a magnitude and fault style, choose a published
    scaling relation, and preview the resulting length/width/slip
    (decomposed into right-lateral + reverse components by rake) before
    applying it to a fault-table row.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Estimate Fault Dimensions from Magnitude")
        self.setMinimumWidth(520)
        self._result = (None, None, None, None)  # length, width, rt_lat, reverse

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<b>Empirical earthquake scaling relations</b><br>"
            "Estimate rupture length, width, and slip from moment magnitude "
            "(Mw), by fault style. The estimated slip is decomposed into "
            "right-lateral and reverse components using the rake below, so "
            "both slip columns in the fault table get a value — not just "
            "reverse."))

        form = QFormLayout()

        self.relation_combo = QComboBox()
        self.relation_combo.addItems(list(SCALING_RELATIONS.keys()))
        form.addRow("Scaling relation:", self.relation_combo)

        # Short, relation-specific explanation -- updated whenever the
        # relation combo changes. Exists because two entries both cite
        # "Wells & Coppersmith (1994)" but give different numbers for the
        # same Mw/style, which is confusing without an explanation of why.
        self.relation_note_label = QLabel("")
        self.relation_note_label.setWordWrap(True)
        self.relation_note_label.setStyleSheet("color: palette(dark); font-style: italic;")
        layout.addWidget(self.relation_note_label)

        # Shear modulus -- NOT currently used by either relation (see
        # scaling_relations.py). It USED to be read by the Coulomb-
        # compatible relation's slip estimate, but that estimate was
        # corrected 2026-08-10: Coulomb's actual GUI formula for this
        # dialog uses a hardcoded internal constant and does not read
        # the model's elastic parameters at all, so this field no longer
        # affects any result. Left wired up (hidden, never shown -- see
        # _on_relation_changed) rather than deleted, in case a future
        # relation genuinely needs a user-supplied shear modulus.
        self.mu_spin = QDoubleSpinBox()
        self.mu_spin.setRange(1.0, 200.0)
        self.mu_spin.setDecimals(1)
        self.mu_spin.setSingleStep(1.0)
        self.mu_spin.setValue(32.0)
        self.mu_spin.setSuffix(" GPa")
        self.mu_row_label = QLabel("Shear modulus (mu, for moment-balanced slip):")
        form.addRow(self.mu_row_label, self.mu_spin)

        self.style_combo = QComboBox()
        self.style_combo.addItems(FAULT_STYLES)
        form.addRow("Fault style (for L/W/slip magnitude):", self.style_combo)

        self.mw_spin = QDoubleSpinBox()
        self.mw_spin.setRange(3.0, 9.5)
        self.mw_spin.setDecimals(2)
        self.mw_spin.setSingleStep(0.1)
        self.mw_spin.setValue(7.0)
        form.addRow("Moment magnitude Mw:", self.mw_spin)

        self.rake_spin = QDoubleSpinBox()
        self.rake_spin.setRange(-180.0, 180.0)
        self.rake_spin.setDecimals(1)
        # Initialize from whatever style is ALREADY selected in style_combo
        # (its default is FAULT_STYLES[0] = "strike-slip"), not a hardcoded
        # 90 degrees — otherwise opening the dialog with the default style
        # pre-selected (no change event fires) leaves the rake stuck at a
        # reverse-fault value, silently misattributing all slip to
        # "reverse" for what should be a pure-lateral fault.
        self.rake_spin.setValue(_STYLE_RAKE_DEG.get(self.style_combo.currentText(), 90.0))
        self.rake_spin.setSuffix(" °")
        form.addRow("Rake (for slip decomposition):", self.rake_spin)

        layout.addLayout(form)

        layout.addWidget(QLabel(
            "<i>Rake sets how the estimated slip magnitude is split between "
            "right-lateral and reverse. It's set automatically when you pick "
            "a strike-slip/reverse/normal style below, or you can override "
            "it directly for an oblique-slip fault. 'all' has no single "
            "representative rake — set it manually for that style.</i>"))

        self.btn_calc = QPushButton("Calculate")
        self.btn_calc.clicked.connect(self._calculate)
        layout.addWidget(self.btn_calc)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(
            "background-color: rgba(0,0,0,15); padding: 8px; border-radius: 4px;")
        layout.addWidget(self.result_label)

        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: darkorange;")
        layout.addWidget(self.warning_label)

        btn_row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply to fault table")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_apply)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

        self.relation_combo.currentIndexChanged.connect(self._on_relation_changed)
        self.style_combo.currentIndexChanged.connect(self._on_style_changed)
        self.mw_spin.valueChanged.connect(self._reset)
        self.rake_spin.valueChanged.connect(self._reset)
        self.mu_spin.valueChanged.connect(self._reset)

        self._on_relation_changed()  # populate note + mu-row visibility for initial selection

    def _on_relation_changed(self):
        name = self.relation_combo.currentText()
        self.relation_note_label.setText(SCALING_RELATION_NOTES.get(name, ""))
        # The two W&C94 relations don't use mu_pa at all (see
        # scaling_relations.py) so the field stays hidden for those, same
        # as before 2026-08-17. The Thingbaijam/Strasser relations added
        # that date solve slip by moment balance against mu_pa, so it
        # needs to be visible and user-adjustable for those.
        needs_mu = name in MOMENT_BALANCED_RELATION_NAMES
        self.mu_row_label.setVisible(needs_mu)
        self.mu_spin.setVisible(needs_mu)
        # Subduction-interface relations ignore style/rake for L/W (one
        # regression for all interface events) -- disable rather than
        # hide the style combo so it's clear why changing it has no
        # effect on the resulting length/width.
        style_matters = name not in _STYLE_IGNORED_RELATION_NAMES
        self.style_combo.setEnabled(style_matters)
        self._reset()

    def _on_style_changed(self):
        style = self.style_combo.currentText()
        if style in _STYLE_RAKE_DEG:
            self.rake_spin.blockSignals(True)
            self.rake_spin.setValue(_STYLE_RAKE_DEG[style])
            self.rake_spin.blockSignals(False)
        self._reset()

    def _reset(self):
        self.btn_apply.setEnabled(False)
        self.result_label.setText("")
        self.warning_label.setText("")
        self._result = (None, None, None, None)

    def _calculate(self):
        relation_name = self.relation_combo.currentText()
        style = self.style_combo.currentText()
        mw = self.mw_spin.value()
        rake_deg = self.rake_spin.value()
        mu_pa = self.mu_spin.value() * 1e9

        try:
            scaling = compute_scaling_result(
                relation_name=relation_name, style=style, mw=mw,
                rake_deg=rake_deg, mu_pa=mu_pa)
        except Exception as e:
            self.result_label.setText(f"Error: {e}")
            self.btn_apply.setEnabled(False)
            return

        result = scaling["raw"]
        length, width = scaling["length_km"], scaling["width_km"]
        rt_lat, reverse = scaling["rt_lateral_slip_m"], scaling["reverse_slip_m"]

        lines = [f"<b>{relation_name}</b>, style = {style}, Mw = {mw:.2f}, rake = {rake_deg:.1f}°"]
        for key, val in result.items():
            if isinstance(val, float):
                lines.append(f"&nbsp;&nbsp;{key}: {val:.4g}")
        if rt_lat is not None:
            lines.append(f"&nbsp;&nbsp;→ right-lateral slip: {rt_lat:+.4g} m")
            lines.append(f"&nbsp;&nbsp;→ reverse slip: {reverse:+.4g} m")
        self.result_label.setText("<br>".join(lines))

        # scaling["warnings"] is plain text (shared with non-UI callers);
        # add the UI's own icon glyphs here rather than in the shared helper.
        icon_by_hint = {
            "not been independently verified": "⚠️",
            "no single representative rake": "⚠️",
            "verified exact matches": "ℹ️",
        }
        warnings = []
        for w in scaling["warnings"]:
            icon = next((v for k, v in icon_by_hint.items() if k in w), "")
            warnings.append(f"{icon} {w}".strip())
        self.warning_label.setText(" ".join(warnings))

        if length is not None and width is not None and rt_lat is not None:
            self._result = (float(length), float(width), float(rt_lat), float(reverse))
            self.btn_apply.setEnabled(True)
        else:
            self.btn_apply.setEnabled(False)

    def get_result(self):
        """
        Returns (length_km, width_km, rt_lateral_slip_m, reverse_slip_m),
        or (None, None, None, None) if the dialog was cancelled or no valid
        calculation was made.
        """
        return self._result
