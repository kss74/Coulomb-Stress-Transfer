# -*- coding: utf-8 -*-
"""
Full symbology configuration dialog for the cross-section tool
(PROJECT_HANDOVER_ADDENDUM_2026-08-18b_cross_section_overhaul.md's
Phase 3, continued 2026-08-19 after Phase 2's focal-mechanism side-view
derivation and topo/annotation picker UI).

Tabbed: EQ / Faults / Contours / Topo / Annotations / Focal Mechanisms /
Legend -- one tab per core.cross_section_config dataclass, editing the
SAME CrossSectionConfig object the caller passes in (in place, on OK;
Cancel discards all edits by simply not writing anything back).

Deliberately does NOT duplicate the "add/remove a topo panel or
annotation source" workflow -- that's ui.main_dialog's own +/- list
UI (Phase 2), which owns *identifying* a data source (file path, QGIS
layer id, label field, sample count, etc.). This dialog's Topo/
Annotations tabs only edit the COSMETIC fields of whatever panels/
sources already exist in config.topo_panels / config.annotations --
selecting one from a list and editing its color/fill/marker/etc. If
the list is empty, the tab says so and points back at the main tab
rather than trying to also be an add-workflow.

Not tied to PyQt beyond the standard qgis.PyQt.QtWidgets/QtCore
imports every other dialog in this project uses; not testable via a
real Qt event loop in the sandbox (standing project constraint) --
see verify_cross_section_config_dialog.py for the stub-based
functional test this session ran instead.
"""

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QFormLayout,
    QDoubleSpinBox, QSpinBox, QPushButton, QLabel, QComboBox, QCheckBox,
    QLineEdit, QListWidget, QGroupBox,
)
from qgis.PyQt.QtCore import Qt

from .dialog_utils import configure_resizable_dialog, wrap_widget_in_scroll_area

LEGEND_LOC_CHOICES = [
    "outside_right", "best", "upper left", "upper right", "lower left",
    "lower right", "center left", "center right", "upper center",
    "lower center", "center",
]


def _dspin(minv, maxv, val, decimals=2, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(minv, maxv)
    s.setValue(val)
    s.setDecimals(decimals)
    if suffix:
        s.setSuffix(f" {suffix}")
    return s


def _ispin(minv, maxv, val):
    s = QSpinBox()
    s.setRange(minv, maxv)
    s.setValue(val)
    return s


def _opt_dspin(minv, maxv, val, decimals=3, suffix=""):
    """A QDoubleSpinBox for an Optional[float] field. Pair with a
    QCheckBox (see _add_optional_float_row) to represent None."""
    s = QDoubleSpinBox()
    s.setRange(minv, maxv)
    s.setDecimals(decimals)
    s.setValue(val if val is not None else 0.0)
    if suffix:
        s.setSuffix(f" {suffix}")
    return s


def _add_optional_float_row(form, label, minv, maxv, val, decimals=3, suffix=""):
    """
    Adds one form row editing an Optional[float] config field as a
    checkbox ("set explicitly") + spinbox pair, since QDoubleSpinBox
    has no native "empty/auto" state and overloading its own minimum
    to mean None would make that minimum unreachable as a real value.
    Returns (checkbox, spinbox); read back as
    `spinbox.value() if checkbox.isChecked() else None`.
    """
    row = QWidget()
    hl = QHBoxLayout(row)
    hl.setContentsMargins(0, 0, 0, 0)
    chk = QCheckBox("Set:")
    chk.setChecked(val is not None)
    spin = _opt_dspin(minv, maxv, val, decimals, suffix)
    spin.setEnabled(val is not None)
    chk.toggled.connect(spin.setEnabled)
    hl.addWidget(chk)
    hl.addWidget(spin)
    form.addRow(label, row)
    return chk, spin


class CrossSectionConfigDialog(QDialog):
    """
    config : core.cross_section_config.CrossSectionConfig -- edited IN
             PLACE on OK (including its .topo_panels / .annotations
             lists, which the caller should have already pointed at
             its own persistent lists, e.g. main_dialog.py's
             self.xs_topo_panels / self.xs_annotations, so edits here
             are visible there too without any extra copy-back step).
    """

    def __init__(self, parent, config):
        super().__init__(parent)
        self.setWindowTitle("Cross-Section Display Configuration")
        configure_resizable_dialog(self, 640, 660, min_width=440, min_height=380)
        self.config = config

        outer = QVBoxLayout(self)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)

        self._build_eq_tab()
        self._build_fault_tab()
        self._build_contours_tab()
        self._build_topo_tab()
        self._build_annotations_tab()
        self._build_extra_lines_tab()
        self._build_focal_tab()
        self._build_legend_tab()

        btn_row = QHBoxLayout()
        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_ok.clicked.connect(self._on_accept)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_ok)
        btn_row.addWidget(self.btn_cancel)
        outer.addLayout(btn_row)

    # ── EQ ───────────────────────────────────────────────────────────
    def _build_eq_tab(self):
        cfg = self.config.eq
        tab = QWidget()
        form = QFormLayout(tab)

        self.eq_enabled = QCheckBox("Show earthquake catalog")
        self.eq_enabled.setChecked(cfg.enabled)
        form.addRow(self.eq_enabled)

        self.eq_search_width = _dspin(0.5, 500, cfg.search_width_km, 1, "km")
        form.addRow("Search width:", self.eq_search_width)

        self.eq_color_by = QComboBox()
        self.eq_color_by.addItems(["depth", "none"])
        self.eq_color_by.setCurrentText(cfg.color_by)
        form.addRow("Color by:", self.eq_color_by)

        self.eq_single_color = QLineEdit(cfg.single_color)
        self.eq_single_color.setToolTip(
            "matplotlib color spec. A single number as a string, e.g. "
            "\"0.4\", is grayscale shorthand from 0 (black) to 1 "
            "(white) -- NOT a fraction of anything else. Also accepts "
            "named colors (\"steelblue\"), hex (\"#3366aa\"), or an "
            "(r,g,b) tuple as text.")
        form.addRow("Single color (if color-by=none):", self.eq_single_color)

        self.eq_cmap = QLineEdit(cfg.cmap)
        form.addRow("Colormap (if color-by=depth):", self.eq_cmap)

        self.eq_size_by = QComboBox()
        self.eq_size_by.addItems(["magnitude", "fixed"])
        self.eq_size_by.setCurrentText(cfg.size_by)
        form.addRow("Size by:", self.eq_size_by)

        self.eq_fixed_size = _dspin(0.1, 1000, cfg.fixed_size_pt2, 1, "pt²")
        form.addRow("Fixed marker area:", self.eq_fixed_size)
        self.eq_mag_size_min = _dspin(0.1, 1000, cfg.mag_size_min_pt2, 1, "pt²")
        form.addRow("Min marker area (smallest Mw):", self.eq_mag_size_min)
        self.eq_mag_size_max = _dspin(0.1, 1000, cfg.mag_size_max_pt2, 1, "pt²")
        form.addRow("Max marker area (largest Mw):", self.eq_mag_size_max)

        self.eq_marker = QLineEdit(cfg.marker)
        form.addRow("Marker style:", self.eq_marker)
        self.eq_alpha = _dspin(0.0, 1.0, cfg.alpha, 2)
        self.eq_alpha.setToolTip(
            "Marker opacity/transparency: 0 = fully transparent "
            "(invisible), 1 = fully opaque. Useful under 1.0 when many "
            "earthquakes overlap, so denser clusters read visibly "
            "darker than isolated events.")
        form.addRow("Alpha:", self.eq_alpha)
        self.eq_edgecolor = QLineEdit(cfg.edgecolor)
        form.addRow("Edge color:", self.eq_edgecolor)
        self.eq_edge_lw = _dspin(0.0, 10.0, cfg.edge_linewidth, 2)
        form.addRow("Edge line width:", self.eq_edge_lw)
        self.eq_zorder = _ispin(0, 100, cfg.zorder)
        self.eq_zorder.setToolTip(
            "Drawing/stacking order -- higher numbers draw ON TOP of "
            "lower ones (matplotlib's zorder). The ΔCFF mesh is "
            "drawn around z-order 3, contours around 4, this overlay "
            "defaults higher so earthquakes sit on top of both; raise "
            "or lower it to change what covers what when overlays "
            "visually collide.")
        form.addRow("Z-order:", self.eq_zorder)

        self.tabs.addTab(wrap_widget_in_scroll_area(tab), "EQ")

    # ── Faults ───────────────────────────────────────────────────────
    def _build_fault_tab(self):
        cfg = self.config.fault
        tab = QWidget()
        form = QFormLayout(tab)

        self.fault_enabled = QCheckBox("Show source fault traces")
        self.fault_enabled.setChecked(cfg.enabled)
        form.addRow(self.fault_enabled)

        self.fault_search_width = _dspin(0.5, 500, cfg.search_width_km, 1, "km")
        form.addRow("Search width:", self.fault_search_width)
        self.fault_color = QLineEdit(cfg.color)
        form.addRow("Color:", self.fault_color)
        self.fault_linewidth = _dspin(0.1, 20.0, cfg.linewidth, 2)
        form.addRow("Line width:", self.fault_linewidth)
        self.fault_linestyle = QLineEdit(cfg.linestyle)
        form.addRow("Line style (matplotlib):", self.fault_linestyle)
        self.fault_label_sources = QCheckBox("Label source faults")
        self.fault_label_sources.setChecked(cfg.label_sources)
        form.addRow(self.fault_label_sources)
        self.fault_label_fontsize = _dspin(4.0, 24.0, cfg.label_fontsize, 1, "pt")
        form.addRow("Label font size:", self.fault_label_fontsize)
        self.fault_zorder = _ispin(0, 100, cfg.zorder)
        form.addRow("Z-order:", self.fault_zorder)

        self.tabs.addTab(wrap_widget_in_scroll_area(tab), "Faults")

    # ── ΔCFF display (color mesh + contour lines) ──────────────────────
    def _build_contours_tab(self):
        """
        Both halves of the ΔCFF display -- the color mesh
        (config.mesh) and the contour-line overlay (config.contours) --
        live in one tab since they're two independent views of the same
        grid. Both have their own `enabled` checkbox and can be toggled
        off independently (mesh off + contours on = contour-lines-only;
        mesh off + contours off + EQ/faults/focal-mechanisms on =
        those overlays alone, with no ΔCFF grid drawn at all).
        """
        mesh_cfg = self.config.mesh
        cfg = self.config.contours
        tab = QWidget()
        outer = QVBoxLayout(tab)

        mesh_box = QGroupBox("ΔCFF color mesh")
        form = QFormLayout(mesh_box)

        self.mesh_enabled = QCheckBox("Show ΔCFF color mesh")
        self.mesh_enabled.setChecked(mesh_cfg.enabled)
        form.addRow(self.mesh_enabled)

        self.mesh_cmap = QLineEdit(mesh_cfg.cmap)
        form.addRow("Colormap:", self.mesh_cmap)

        self.mesh_vmin_chk, self.mesh_vmin = _add_optional_float_row(
            form, "Min (MPa):", -1e6, 1e6, mesh_cfg.vmin_mpa, 4, "MPa")
        self.mesh_vmax_chk, self.mesh_vmax = _add_optional_float_row(
            form, "Max (MPa):", -1e6, 1e6, mesh_cfg.vmax_mpa, 4, "MPa")
        form.addRow(QLabel(
            "Leave both unset for auto-scaling (symmetric about the "
            "percentile below). Set both to fix the color scale -- "
            "useful if small real ΔCFF values are washing out near "
            "white on a scale dominated by a few large outliers."))

        self.mesh_percentile = _dspin(50.0, 100.0, mesh_cfg.color_scale_percentile, 1, "pct")
        form.addRow("Auto-scale percentile:", self.mesh_percentile)

        self.mesh_interpolate = QCheckBox("Smooth (spline-interpolate) the display")
        self.mesh_interpolate.setChecked(mesh_cfg.interpolate)
        form.addRow(self.mesh_interpolate)
        form.addRow(QLabel(
            "Display-only smoothing of the grid you already computed -- "
            "does not change the Distance/Depth increment on the Cross-"
            "Section tab (the actual DC3D sampling resolution)."))
        self.mesh_interp_factor = _ispin(1, 20, mesh_cfg.interpolation_factor)
        form.addRow("Interpolation factor:", self.mesh_interp_factor)

        outer.addWidget(mesh_box)

        contour_box = QGroupBox("ΔCFF contour lines")
        form = QFormLayout(contour_box)

        self.contours_enabled = QCheckBox("Show ΔCFF contours")
        self.contours_enabled.setChecked(cfg.enabled)
        form.addRow(self.contours_enabled)

        self.contours_baseline_chk, self.contours_baseline = _add_optional_float_row(
            form, "Baseline (MPa):", -1e6, 1e6, cfg.baseline_mpa, 4, "MPa")
        self.contours_spacing_chk, self.contours_spacing = _add_optional_float_row(
            form, "Spacing (MPa):", 1e-6, 1e6, cfg.spacing_mpa, 4, "MPa")
        form.addRow(QLabel(
            "Baseline + spacing gives contours at baseline + k*spacing. "
            "Leave spacing unset to divide the color scale into "
            "'Number of levels' below instead."))

        self.contours_n_levels = _ispin(2, 50, cfg.n_levels)
        form.addRow("Number of levels (auto, if spacing unset):", self.contours_n_levels)
        self.contours_color = QLineEdit(cfg.color)
        form.addRow("Color:", self.contours_color)
        self.contours_linewidth = _dspin(0.1, 10.0, cfg.linewidth, 2)
        form.addRow("Line width:", self.contours_linewidth)
        self.contours_alpha = _dspin(0.0, 1.0, cfg.alpha, 2)
        form.addRow("Alpha:", self.contours_alpha)
        self.contours_inline_labels = QCheckBox("Inline labels")
        self.contours_inline_labels.setChecked(cfg.inline_labels)
        form.addRow(self.contours_inline_labels)
        self.contours_label_fontsize = _dspin(4.0, 24.0, cfg.label_fontsize, 1, "pt")
        form.addRow("Label font size:", self.contours_label_fontsize)
        self.contours_fmt = QLineEdit(cfg.fmt)
        form.addRow("Label format (%-style):", self.contours_fmt)
        self.contours_zorder = _ispin(0, 100, cfg.zorder)
        form.addRow("Z-order:", self.contours_zorder)

        outer.addWidget(contour_box)
        outer.addStretch()

        self.tabs.addTab(wrap_widget_in_scroll_area(tab), "ΔCFF Display")
        # 2026-08-21 fix: this tab widget was being added a SECOND time
        # under the old "Contours" label, left over from the 2026-08-20
        # tab rename. QTabWidget/QStackedWidget only lets one widget
        # instance live at one index -- adding the same `tab` object a
        # second time silently MOVES it to the new tab, leaving "ΔCFF
        # Display" showing an empty page and all the real controls
        # (mesh min/max, contour spacing, etc.) sitting under a
        # duplicate "Contours" tab instead. This is why "the ΔCFF
        # Display tab is empty" -- the content was there, just filed
        # under a stale duplicate tab label. Removed the duplicate call.

    # ── Topo panels (cosmetic fields only -- see module docstring) ─────
    def _build_topo_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        panels = self.config.topo_panels

        if not panels:
            layout.addWidget(QLabel(
                "No topo panels yet. Add one from the Cross-Section tab's "
                "'+ Raster file…' / '+ QGIS raster layer…' buttons, then "
                "reopen this dialog to style it."))
            self.tabs.addTab(tab, "Topo")
            return

        layout.addWidget(QLabel(
            "Select a panel, edit its appearance, then click "
            "'Apply to selected panel'."))
        self.topo_list = QListWidget()
        for p in panels:
            self.topo_list.addItem(p.label)
        layout.addWidget(self.topo_list)

        form = QFormLayout()
        self.topo_color = QLineEdit()
        form.addRow("Color:", self.topo_color)
        self.topo_fill = QCheckBox("Fill under the profile line")
        form.addRow(self.topo_fill)
        self.topo_fill_alpha = _dspin(0.0, 1.0, 0.25, 2)
        form.addRow("Fill alpha:", self.topo_fill_alpha)
        self.topo_linewidth = _dspin(0.1, 20.0, 1.2, 2)
        form.addRow("Line width:", self.topo_linewidth)
        self.topo_height_ratio = _dspin(0.05, 5.0, 0.35, 2)
        form.addRow("Height ratio (vs. main panel):", self.topo_height_ratio)
        self.topo_vexag_auto = QCheckBox("Auto vertical exaggeration")
        form.addRow(self.topo_vexag_auto)
        self.topo_vexag = _dspin(0.01, 100.0, 1.0, 2)
        form.addRow("Vertical exaggeration (if not auto):", self.topo_vexag)
        layout.addLayout(form)

        self.btn_topo_apply = QPushButton("Apply to selected panel")
        self.btn_topo_apply.clicked.connect(self._apply_topo_panel_fields)
        layout.addWidget(self.btn_topo_apply)

        self.topo_list.currentRowChanged.connect(self._load_topo_panel_fields)
        self.topo_list.setCurrentRow(0)
        self._load_topo_panel_fields(0)

        self.tabs.addTab(tab, "Topo")

    def _load_topo_panel_fields(self, row):
        panels = self.config.topo_panels
        if row is None or row < 0 or row >= len(panels):
            return
        p = panels[row]
        self.topo_color.setText(p.color)
        self.topo_fill.setChecked(p.fill)
        self.topo_fill_alpha.setValue(p.fill_alpha)
        self.topo_linewidth.setValue(p.linewidth)
        self.topo_height_ratio.setValue(p.height_ratio)
        auto = p.vertical_exaggeration is None
        self.topo_vexag_auto.setChecked(auto)
        self.topo_vexag.setValue(p.vertical_exaggeration if not auto else 1.0)

    def _apply_topo_panel_fields(self):
        panels = self.config.topo_panels
        row = self.topo_list.currentRow()
        if row < 0 or row >= len(panels):
            return
        p = panels[row]
        p.color = self.topo_color.text()
        p.fill = self.topo_fill.isChecked()
        p.fill_alpha = self.topo_fill_alpha.value()
        p.linewidth = self.topo_linewidth.value()
        p.height_ratio = self.topo_height_ratio.value()
        p.vertical_exaggeration = (
            None if self.topo_vexag_auto.isChecked() else self.topo_vexag.value())

    # ── Annotations (cosmetic fields only) ──────────────────────────
    def _build_annotations_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        sources = self.config.annotations

        if not sources:
            layout.addWidget(QLabel(
                "No annotation sources yet. Add one from the Cross-Section "
                "tab's '+ QGIS point layer…' / '+ CSV file…' buttons, then "
                "reopen this dialog to style it."))
            self.tabs.addTab(tab, "Annotations")
            return

        layout.addWidget(QLabel(
            "Select a source, edit its appearance, then click "
            "'Apply to selected source'."))
        self.ann_list = QListWidget()
        for a in sources:
            self.ann_list.addItem(a.label)
        layout.addWidget(self.ann_list)

        form = QFormLayout()
        self.ann_enabled = QCheckBox("Enabled")
        form.addRow(self.ann_enabled)
        self.ann_search_width = _dspin(0.1, 500, 5.0, 1, "km")
        form.addRow("Search width:", self.ann_search_width)
        self.ann_marker = QLineEdit()
        form.addRow("Marker style:", self.ann_marker)
        self.ann_color = QLineEdit()
        form.addRow("Color:", self.ann_color)
        self.ann_size = _dspin(1.0, 100.0, 9.0, 1, "pt")
        form.addRow("Marker size:", self.ann_size)
        self.ann_label_fontsize = _dspin(4.0, 24.0, 7.0, 1, "pt")
        form.addRow("Label font size:", self.ann_label_fontsize)
        self.ann_zorder = _ispin(0, 100, 7)
        form.addRow("Z-order:", self.ann_zorder)
        layout.addLayout(form)

        self.btn_ann_apply = QPushButton("Apply to selected source")
        self.btn_ann_apply.clicked.connect(self._apply_annotation_fields)
        layout.addWidget(self.btn_ann_apply)

        self.ann_list.currentRowChanged.connect(self._load_annotation_fields)
        self.ann_list.setCurrentRow(0)
        self._load_annotation_fields(0)

        self.tabs.addTab(tab, "Annotations")

    def _load_annotation_fields(self, row):
        sources = self.config.annotations
        if row is None or row < 0 or row >= len(sources):
            return
        a = sources[row]
        self.ann_enabled.setChecked(a.enabled)
        self.ann_search_width.setValue(a.search_width_km)
        self.ann_marker.setText(a.marker)
        self.ann_color.setText(a.color)
        self.ann_size.setValue(a.size_pt)
        self.ann_label_fontsize.setValue(a.label_fontsize)
        self.ann_zorder.setValue(a.zorder)

    def _apply_annotation_fields(self):
        sources = self.config.annotations
        row = self.ann_list.currentRow()
        if row < 0 or row >= len(sources):
            return
        a = sources[row]
        a.enabled = self.ann_enabled.isChecked()
        a.search_width_km = self.ann_search_width.value()
        a.marker = self.ann_marker.text()
        a.color = self.ann_color.text()
        a.size_pt = self.ann_size.value()
        a.label_fontsize = self.ann_label_fontsize.value()
        a.zorder = self.ann_zorder.value()

    # ── Extra depth-section elements (2026-08-21, request item 7) ──
    def _build_extra_lines_tab(self):
        """
        Per-entry cosmetic editing for config.extra_lines
        (ExtraSectionLineConfig) -- same list + form + "Apply to
        selected" pattern as _build_topo_tab()/_build_annotations_tab()
        above, so an imported slab interface / other fault's geometry /
        raster-sampled horizon / etc. can be given its own color, line
        style, label, and (for a "vector_line" entry) search width
        after import, instead of being stuck with whatever defaults
        ui.main_dialog's add_xs_extra_line*_action() methods gave it.

        Search width is disabled (with an explanatory note) for a
        raster-sourced entry ("raster_file"/"qgis_layer") -- consistent
        with core.cross_section_config.ExtraSectionLineConfig's own
        docstring, that field only means anything for a "vector_line"
        entry's independently-positioned vertices; a raster element is
        sampled exactly on the profile itself, so there's no off-profile
        distance to filter on. Matches this dialog's own stated
        philosophy elsewhere (see module docstring) of exposing only
        cosmetics here, not add-time data-source fields (band, sample
        count, unit divisor, sign) -- those stay in main_dialog's
        add-raster prompts, same as a topo panel's source/band/divisor
        aren't editable here either.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        lines = self.config.extra_lines

        if not lines:
            layout.addWidget(QLabel(
                "No extra depth-section elements yet. Add one from the "
                "Cross-Section tab's '+ QGIS line layer (with depth "
                "field)…', '+ Raster file (e.g. Slab2)…', or '+ QGIS "
                "raster layer…' buttons (e.g. a subducting slab "
                "interface or another fault's known geometry), then "
                "reopen this dialog to style it."))
            self.tabs.addTab(tab, "Extra Lines")
            return

        layout.addWidget(QLabel(
            "Select an element, edit its appearance, then click "
            "'Apply to selected element'."))
        self.extra_line_list = QListWidget()
        for line_cfg in lines:
            self.extra_line_list.addItem(line_cfg.label or "(unlabeled)")
        layout.addWidget(self.extra_line_list)

        form = QFormLayout()
        self.extra_line_label = QLineEdit()
        form.addRow("Label:", self.extra_line_label)
        self.extra_line_color = QLineEdit()
        form.addRow("Color:", self.extra_line_color)
        self.extra_line_linewidth = _dspin(0.1, 20.0, 1.5, 2)
        form.addRow("Line width:", self.extra_line_linewidth)
        self.extra_line_linestyle = QComboBox()
        self.extra_line_linestyle.addItems(["-", "--", "-.", ":"])
        form.addRow("Line style:", self.extra_line_linestyle)
        self.extra_line_search_width = _dspin(0.1, 500, 15.0, 1, "km")
        form.addRow("Search width (± perpendicular):", self.extra_line_search_width)
        self.extra_line_search_width_note = QLabel(
            "<i>N/A for a raster-sourced element -- it's sampled exactly "
            "on the profile, so nothing to search for.</i>")
        self.extra_line_search_width_note.setVisible(False)
        form.addRow(self.extra_line_search_width_note)
        self.extra_line_show_label = QCheckBox("Show label")
        form.addRow(self.extra_line_show_label)
        self.extra_line_label_fontsize = _dspin(4.0, 24.0, 7.0, 1, "pt")
        form.addRow("Label font size:", self.extra_line_label_fontsize)
        self.extra_line_zorder = _ispin(0, 100, 5)
        form.addRow("Z-order:", self.extra_line_zorder)
        layout.addLayout(form)

        self.btn_extra_line_apply = QPushButton("Apply to selected element")
        self.btn_extra_line_apply.clicked.connect(self._apply_extra_line_fields)
        layout.addWidget(self.btn_extra_line_apply)

        self.extra_line_list.currentRowChanged.connect(self._load_extra_line_fields)
        self.extra_line_list.setCurrentRow(0)
        self._load_extra_line_fields(0)

        self.tabs.addTab(tab, "Extra Lines")

    def _load_extra_line_fields(self, row):
        lines = self.config.extra_lines
        if row is None or row < 0 or row >= len(lines):
            return
        line_cfg = lines[row]
        self.extra_line_label.setText(line_cfg.label)
        self.extra_line_color.setText(line_cfg.color)
        self.extra_line_linewidth.setValue(line_cfg.linewidth)
        idx = self.extra_line_linestyle.findText(line_cfg.linestyle)
        self.extra_line_linestyle.setCurrentIndex(idx if idx >= 0 else 0)
        self.extra_line_search_width.setValue(line_cfg.search_width_km)
        is_raster = line_cfg.source_kind in ("raster_file", "qgis_layer")
        self.extra_line_search_width.setEnabled(not is_raster)
        self.extra_line_search_width_note.setVisible(is_raster)
        self.extra_line_show_label.setChecked(line_cfg.show_label)
        self.extra_line_label_fontsize.setValue(line_cfg.label_fontsize)
        self.extra_line_zorder.setValue(line_cfg.zorder)

    def _apply_extra_line_fields(self):
        lines = self.config.extra_lines
        row = self.extra_line_list.currentRow()
        if row < 0 or row >= len(lines):
            return
        line_cfg = lines[row]
        line_cfg.label = self.extra_line_label.text()
        line_cfg.color = self.extra_line_color.text()
        line_cfg.linewidth = self.extra_line_linewidth.value()
        line_cfg.linestyle = self.extra_line_linestyle.currentText()
        line_cfg.search_width_km = self.extra_line_search_width.value()
        line_cfg.show_label = self.extra_line_show_label.isChecked()
        line_cfg.label_fontsize = self.extra_line_label_fontsize.value()
        line_cfg.zorder = self.extra_line_zorder.value()
        # Keep the list's own display text in sync with a relabel.
        self.extra_line_list.item(row).setText(line_cfg.label or "(unlabeled)")

    # ── Focal mechanisms ────────────────────────────────────────────
    def _build_focal_tab(self):
        cfg = self.config.focal_mechanisms
        tab = QWidget()
        form = QFormLayout(tab)

        self.fm_enabled = QCheckBox("Show focal mechanisms (side view)")
        self.fm_enabled.setChecked(cfg.enabled)
        form.addRow(self.fm_enabled)

        self.fm_search_width = _dspin(0.5, 500, cfg.search_width_km, 1, "km")
        form.addRow("Search width:", self.fm_search_width)

        self.fm_plane = QComboBox()
        self.fm_plane.addItems(["selected", "plane1", "plane2"])
        self.fm_plane.setCurrentText(cfg.plane)
        self.fm_plane.setToolTip(
            "A focal mechanism (moment tensor) is inherently ambiguous "
            "between its two nodal planes -- there's no way to tell "
            "which one is the actual rupture plane from the mechanism "
            "alone. This picks WHICH of the two to draw (matching "
            "Coulomb's own PLANE_MODES choice), not which side of the "
            "cross-section to show -- the profile's own orientation "
            "separately determines how that chosen plane's true 3-D "
            "strike/dip/rake project into this 2-D side view. "
            "'selected' uses each event's own resolved plane if the "
            "import supplied one, falling back to plane 1 otherwise.")
        form.addRow("Plane to draw:", self.fm_plane)

        # -- Size --
        self.fm_size_by = QComboBox()
        self.fm_size_by.addItems(["fixed", "magnitude"])
        self.fm_size_by.setCurrentText(cfg.size_by)
        form.addRow("Size by:", self.fm_size_by)
        self.fm_diameter = _dspin(0.01, 500.0, cfg.diameter_km, 2, "km")
        form.addRow("Symbol diameter (if size-by=fixed):", self.fm_diameter)
        self.fm_mag_diameter_min = _dspin(0.01, 500.0, cfg.mag_diameter_min_km, 2, "km")
        form.addRow("Diameter at smallest Mw (if size-by=magnitude):", self.fm_mag_diameter_min)
        self.fm_mag_diameter_max = _dspin(0.01, 500.0, cfg.mag_diameter_max_km, 2, "km")
        form.addRow("Diameter at largest Mw (if size-by=magnitude):", self.fm_mag_diameter_max)

        # -- Color --
        self.fm_color_by = QComboBox()
        self.fm_color_by.addItems(["cff", "single", "type", "depth"])
        self.fm_color_by.setCurrentText(cfg.color_by)
        self.fm_color_by.setToolTip(
            "cff: colored by each event's resolved ΔCFF (diverging scale). "
            "single: one fixed color for every symbol. "
            "type: Frohlich (1992) P/T/B classification -- normal/"
            "reverse/strike-slip/oblique, the same ternary-diagram "
            "convention other focal-mechanism plugins use -- colors are "
            "fixed per type, not user-set, so this coloring stays "
            "consistent everywhere it's used in the plugin. "
            "depth: colored by event depth (same idea as the Earthquake "
            "Catalog tab's depth coloring).")
        form.addRow("Color by:", self.fm_color_by)
        self.fm_single_color = QLineEdit(cfg.single_color)
        self.fm_single_color.setToolTip(
            "matplotlib color spec. A single number as a string, e.g. "
            "\"0.4\", is grayscale shorthand from 0 (black) to 1 "
            "(white) -- NOT a fraction of anything else. Also accepts "
            "named colors (\"steelblue\"), hex (\"#3366aa\"), or an "
            "(r,g,b) tuple as text.")
        form.addRow("Single color (if color-by=single):", self.fm_single_color)
        self.fm_cmap = QLineEdit(cfg.cmap)
        self.fm_cmap.setToolTip(
            "matplotlib colormap name (if color-by=cff), e.g. \"RdBu_r\" "
            "(the default -- diverging, blue=negative/red=positive "
            "ΔCFF, white=~zero by design). See matplotlib's colormap "
            "reference for the full list.")
        form.addRow("Colormap (if color-by=cff):", self.fm_cmap)
        self.fm_vmin_chk, self.fm_vmin = _add_optional_float_row(
            form, "Min (MPa, if color-by=cff):", -1e6, 1e6, cfg.vmin_mpa, 4, "MPa")
        self.fm_vmax_chk, self.fm_vmax = _add_optional_float_row(
            form, "Max (MPa, if color-by=cff):", -1e6, 1e6, cfg.vmax_mpa, 4, "MPa")
        form.addRow(QLabel(
            "Leave both unset for auto-scaling across this overlay's own "
            "events. If low-ΔCFF events all look washed-out near white, "
            "narrowing min/max here (independently of the ΔCFF mesh's "
            "own scale) is usually the fix."))
        self.fm_depth_cmap = QLineEdit(cfg.depth_cmap)
        form.addRow("Colormap (if color-by=depth):", self.fm_depth_cmap)

        # -- Type colors (if color-by=type) --
        tc = cfg.type_colors or {}
        self.fm_type_color_normal = QLineEdit(tc.get("normal", ""))
        self.fm_type_color_normal.setPlaceholderText("default: red (WSM standard)")
        form.addRow("Normal color (if color-by=type):", self.fm_type_color_normal)
        self.fm_type_color_reverse = QLineEdit(tc.get("reverse", ""))
        self.fm_type_color_reverse.setPlaceholderText("default: blue (WSM standard)")
        form.addRow("Reverse color (if color-by=type):", self.fm_type_color_reverse)
        self.fm_type_color_ss = QLineEdit(tc.get("strike-slip", ""))
        self.fm_type_color_ss.setPlaceholderText("default: green (WSM standard)")
        form.addRow("Strike-slip color (if color-by=type):", self.fm_type_color_ss)
        self.fm_type_color_oblique = QLineEdit(tc.get("oblique", ""))
        self.fm_type_color_oblique.setPlaceholderText("default: #ff7f00 (orange)")
        form.addRow("Oblique color (if color-by=type):", self.fm_type_color_oblique)
        form.addRow(QLabel(
            "Type colors default to the World Stress Map convention "
            "(red=normal, green=strike-slip, blue=reverse/thrust) -- "
            "leave any field blank to keep that default; fill in only "
            "the ones you want to override."))
        self.fm_bgcolor = QLineEdit(cfg.bgcolor)
        self.fm_bgcolor.setToolTip(
            "The complementary (dilatational) beachball lobe color. "
            "Avoid pure white if color-by=cff and low-ΔCFF events "
            "matter to you -- the compressional lobe will also read "
            "near-white at low |CFF| against a white background, and "
            "the symbol can nearly disappear. A light gray (e.g. "
            "\"0.85\") keeps the lobe split and outline visible even "
            "at ΔCFF ≈ 0.")
        form.addRow("Background (dilatational) color:", self.fm_bgcolor)
        self.fm_edgecolor = QLineEdit(cfg.edgecolor)
        form.addRow("Edge color:", self.fm_edgecolor)

        self.fm_highlight = QCheckBox("Highlight the selected nodal plane")
        self.fm_highlight.setChecked(cfg.highlight_selected_plane)
        form.addRow(self.fm_highlight)

        # -- Labels --
        self.fm_show_labels = QCheckBox("Show event labels")
        self.fm_show_labels.setChecked(cfg.show_labels)
        form.addRow(self.fm_show_labels)
        self.fm_label_source = QComboBox()
        self.fm_label_source.addItems(["magnitude", "custom"])
        self.fm_label_source.setCurrentText(cfg.label_source)
        form.addRow("Label from:", self.fm_label_source)
        self.fm_label_fmt = QLineEdit(cfg.label_fmt)
        self.fm_label_fmt.setToolTip(
            "%-style format string applied to each event's own Mw, "
            "e.g. \"M%.1f\" -> \"M5.3\" (if label-from=magnitude).")
        form.addRow("Magnitude label format:", self.fm_label_fmt)
        self.fm_label_fontsize = _dspin(4.0, 24.0, cfg.label_fontsize, 1, "pt")
        form.addRow("Label font size:", self.fm_label_fontsize)
        self.fm_label_offset_chk, self.fm_label_offset = _add_optional_float_row(
            form, "Label gap (km):", 0.0, 1e6, cfg.label_offset_km, 3, "km")
        form.addRow(QLabel(
            "Gap between the beachball's edge and its label, in the "
            "panel's own km units. Leave unset for auto (0.9x the "
            "symbol's own diameter)."))
        self.fm_label_leader = QCheckBox("Draw a leader line from symbol to label")
        self.fm_label_leader.setChecked(cfg.label_leader_line)
        self.fm_label_leader.setToolTip(
            "Turn on once labels start crowding neighboring symbols and "
            "no longer sit obviously next to their own beachball.")
        form.addRow(self.fm_label_leader)
        self.fm_zorder = _ispin(0, 100, cfg.zorder)
        form.addRow("Z-order:", self.fm_zorder)

        self.tabs.addTab(wrap_widget_in_scroll_area(tab), "Focal Mechanisms")

    # ── Legend ───────────────────────────────────────────────────────
    def _build_legend_tab(self):
        cfg = self.config.legend
        tab = QWidget()
        form = QFormLayout(tab)

        self.legend_enabled = QCheckBox("Show legend")
        self.legend_enabled.setChecked(cfg.enabled)
        form.addRow(self.legend_enabled)

        self.legend_loc = QComboBox()
        self.legend_loc.addItems(LEGEND_LOC_CHOICES)
        if cfg.loc in LEGEND_LOC_CHOICES:
            self.legend_loc.setCurrentText(cfg.loc)
        else:
            self.legend_loc.addItem(cfg.loc)
            self.legend_loc.setCurrentText(cfg.loc)
        form.addRow("Placement:", self.legend_loc)

        self.legend_fontsize = _dspin(4.0, 24.0, cfg.fontsize, 1, "pt")
        form.addRow("Font size:", self.legend_fontsize)

        self.tabs.addTab(tab, "Legend")

    # ── Write back on OK ─────────────────────────────────────────────
    def _on_accept(self):
        eq = self.config.eq
        eq.enabled = self.eq_enabled.isChecked()
        eq.search_width_km = self.eq_search_width.value()
        eq.color_by = self.eq_color_by.currentText()
        eq.single_color = self.eq_single_color.text()
        eq.cmap = self.eq_cmap.text()
        eq.size_by = self.eq_size_by.currentText()
        eq.fixed_size_pt2 = self.eq_fixed_size.value()
        eq.mag_size_min_pt2 = self.eq_mag_size_min.value()
        eq.mag_size_max_pt2 = self.eq_mag_size_max.value()
        eq.marker = self.eq_marker.text()
        eq.alpha = self.eq_alpha.value()
        eq.edgecolor = self.eq_edgecolor.text()
        eq.edge_linewidth = self.eq_edge_lw.value()
        eq.zorder = self.eq_zorder.value()

        fault = self.config.fault
        fault.enabled = self.fault_enabled.isChecked()
        fault.search_width_km = self.fault_search_width.value()
        fault.color = self.fault_color.text()
        fault.linewidth = self.fault_linewidth.value()
        fault.linestyle = self.fault_linestyle.text()
        fault.label_sources = self.fault_label_sources.isChecked()
        fault.label_fontsize = self.fault_label_fontsize.value()
        fault.zorder = self.fault_zorder.value()

        contours = self.config.contours
        contours.enabled = self.contours_enabled.isChecked()
        contours.baseline_mpa = (self.contours_baseline.value()
                                  if self.contours_baseline_chk.isChecked() else None)
        contours.spacing_mpa = (self.contours_spacing.value()
                                 if self.contours_spacing_chk.isChecked() else None)
        contours.n_levels = self.contours_n_levels.value()
        contours.color = self.contours_color.text()
        contours.linewidth = self.contours_linewidth.value()
        contours.alpha = self.contours_alpha.value()
        contours.inline_labels = self.contours_inline_labels.isChecked()
        contours.label_fontsize = self.contours_label_fontsize.value()
        contours.fmt = self.contours_fmt.text()
        contours.zorder = self.contours_zorder.value()

        mesh = self.config.mesh
        mesh.enabled = self.mesh_enabled.isChecked()
        mesh.cmap = self.mesh_cmap.text()
        mesh.vmin_mpa = self.mesh_vmin.value() if self.mesh_vmin_chk.isChecked() else None
        mesh.vmax_mpa = self.mesh_vmax.value() if self.mesh_vmax_chk.isChecked() else None
        mesh.color_scale_percentile = self.mesh_percentile.value()
        mesh.interpolate = self.mesh_interpolate.isChecked()
        mesh.interpolation_factor = self.mesh_interp_factor.value()

        # Topo/annotation cosmetic edits are already written back live by
        # _apply_topo_panel_fields()/_apply_annotation_fields() (so
        # switching the list selection without clicking Apply first
        # doesn't silently lose the panel/source currently being edited
        # if OK is clicked next) -- nothing further to do here for them.

        fm = self.config.focal_mechanisms
        fm.enabled = self.fm_enabled.isChecked()
        fm.search_width_km = self.fm_search_width.value()
        fm.diameter_km = self.fm_diameter.value()
        fm.plane = self.fm_plane.currentText()
        fm.size_by = self.fm_size_by.currentText()
        fm.mag_diameter_min_km = self.fm_mag_diameter_min.value()
        fm.mag_diameter_max_km = self.fm_mag_diameter_max.value()
        fm.color_by = self.fm_color_by.currentText()
        fm.single_color = self.fm_single_color.text()
        fm.cmap = self.fm_cmap.text()
        fm.vmin_mpa = self.fm_vmin.value() if self.fm_vmin_chk.isChecked() else None
        fm.vmax_mpa = self.fm_vmax.value() if self.fm_vmax_chk.isChecked() else None
        fm.depth_cmap = self.fm_depth_cmap.text()
        type_colors = {}
        if self.fm_type_color_normal.text().strip():
            type_colors["normal"] = self.fm_type_color_normal.text().strip()
        if self.fm_type_color_reverse.text().strip():
            type_colors["reverse"] = self.fm_type_color_reverse.text().strip()
        if self.fm_type_color_ss.text().strip():
            type_colors["strike-slip"] = self.fm_type_color_ss.text().strip()
        if self.fm_type_color_oblique.text().strip():
            type_colors["oblique"] = self.fm_type_color_oblique.text().strip()
        fm.type_colors = type_colors or None
        fm.bgcolor = self.fm_bgcolor.text()
        fm.edgecolor = self.fm_edgecolor.text()
        fm.highlight_selected_plane = self.fm_highlight.isChecked()
        fm.show_labels = self.fm_show_labels.isChecked()
        fm.label_source = self.fm_label_source.currentText()
        fm.label_fmt = self.fm_label_fmt.text()
        fm.label_fontsize = self.fm_label_fontsize.value()
        fm.label_offset_km = (self.fm_label_offset.value()
                              if self.fm_label_offset_chk.isChecked() else None)
        fm.label_leader_line = self.fm_label_leader.isChecked()
        fm.zorder = self.fm_zorder.value()

        legend = self.config.legend
        legend.enabled = self.legend_enabled.isChecked()
        legend.loc = self.legend_loc.currentText()
        legend.fontsize = self.legend_fontsize.value()

        self.accept()
