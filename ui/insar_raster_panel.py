# -*- coding: utf-8 -*-
"""
UI panel for ingesting raw InSAR rasters (LOS + look-vector geometry)
via core.insar_raster_import, downsampling them, and handing the
resulting points to the same callback contract slip_inversion_dialog.
_ObservationImportPanel already uses: on_batch_ready(list_of_dicts) in
observation_import.py's "insar_los" row schema
({"lon","lat","los","look_e","look_n","look_u","sigma"}).

This is the raster counterpart of _ObservationImportPanel (which reads
an already-downsampled CSV/TSV or QGIS point layer) -- see
core.insar_raster_import's module docstring for why raw-raster
ingestion was originally out of scope and what convention choices this
panel is exposing to the user rather than guessing silently:
  - ENU look-vector rasters vs incidence+azimuth/heading rasters
  - azimuth vs heading angle convention, right- vs left-looking radar
  - LOS sign (ground-to-satellite-positive vs the opposite)
  - uniform-stride vs adaptive-quadtree downsampling

Intended usage (mirrors _ObservationImportPanel): embedded as a second
sub-tab alongside the existing table/layer importer, both feeding the
same accumulating point list in the parent dialog.
"""

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QMessageBox, QRadioButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView,
)
from qgis.PyQt.QtCore import Qt

from ..core.insar_raster_import import (
    check_gdal_available, load_los_with_enu_rasters,
    load_los_with_angle_rasters, downsample_uniform, downsample_quadtree,
    load_component_rasters, downsample_components_uniform,
    downsample_components_quadtree,
)


class _FileRow(QWidget):
    """One labeled 'Browse…' file-picker row, reused for every raster
    input this panel needs (LOS, E/N/U, incidence, azimuth/heading,
    sigma, mask)."""

    def __init__(self, label, required=True, parent=None):
        super().__init__(parent)
        self.path = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        suffix = " (required):" if required else " (optional):"
        self.label_widget = QLabel(label + suffix)
        self.label_widget.setMinimumWidth(220)
        self.path_label = QLabel("No file selected")
        self.path_label.setStyleSheet("color: #777;")
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse)
        layout.addWidget(self.label_widget)
        layout.addWidget(self.path_label, 1)
        layout.addWidget(btn)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select raster", "",
            "GeoTIFF / GDAL raster (*.tif *.tiff *.geotiff);;All files (*.*)")
        if path:
            self.path = path
            self.path_label.setText(path)
            self.path_label.setStyleSheet("")


class _InsarRasterImportPanel(QWidget):
    """Raster (GeoTIFF) InSAR import panel. Same on_batch_ready(rows)
    callback contract as _ObservationImportPanel in
    slip_inversion_dialog.py."""

    def __init__(self, on_batch_ready, parent=None):
        super().__init__(parent)
        self._on_batch_ready = on_batch_ready
        self._pending_rows = []

        layout = QVBoxLayout(self)

        ok, msg = check_gdal_available()
        if not ok:
            warn = QLabel(
                f"<b style='color:#a33;'>GDAL not available:</b> {msg}")
            warn.setWordWrap(True)
            layout.addWidget(warn)
        self._gdal_ok = ok

        # ── LOS raster ──────────────────────────────────────────────
        los_group = QGroupBox("LOS displacement raster")
        los_layout = QVBoxLayout(los_group)
        self.row_los = _FileRow("Unwrapped LOS raster")
        los_layout.addWidget(self.row_los)

        sign_row = QHBoxLayout()
        sign_row.addWidget(QLabel("LOS sign convention:"))
        self.combo_los_sign = QComboBox()
        self.combo_los_sign.addItems([
            "As-is (positive = motion toward satellite)",
            "Flip sign (positive in my raster = motion AWAY from satellite)",
        ])
        sign_row.addWidget(self.combo_los_sign, 1)
        los_layout.addLayout(sign_row)
        los_layout.addWidget(QLabel(
            "<i>This module cannot detect which convention your processor "
            "uses -- see core.insar_raster_import's module docstring. Get "
            "this wrong and the inversion's recovered rake/slip sign flips, "
            "not just its magnitude.</i>"))
        layout.addWidget(los_group)

        # ── Look-vector source ──────────────────────────────────────
        lv_group = QGroupBox("Look-vector source")
        lv_layout = QVBoxLayout(lv_group)
        self.radio_enu = QRadioButton(
            "E/N/U look-vector rasters (already resolved -- e.g. LiCSBAS "
            "E.geo.tif/N.geo.tif/U.geo.tif; no conversion, no convention risk)")
        self.radio_angle = QRadioButton(
            "Incidence + azimuth/heading rasters (converted here -- see below)")
        self.radio_enu.setChecked(True)
        lv_layout.addWidget(self.radio_enu)
        lv_layout.addWidget(self.radio_angle)

        self.row_e = _FileRow("East look-vector raster")
        self.row_n = _FileRow("North look-vector raster")
        self.row_u = _FileRow("Up look-vector raster")
        lv_layout.addWidget(self.row_e)
        lv_layout.addWidget(self.row_n)
        lv_layout.addWidget(self.row_u)

        self.row_inc = _FileRow("Incidence angle raster")
        self.row_angle = _FileRow("Azimuth/heading angle raster")
        angle_type_row = QHBoxLayout()
        angle_type_row.addWidget(QLabel("Angle convention:"))
        self.combo_angle_type = QComboBox()
        self.combo_angle_type.addItems([
            "LOS azimuth (from north, anti-clockwise +) -- ISCE/MintPy az_angle",
            "Satellite heading (from north, clockwise +) -- ROI_PAC/GAMMA head_angle",
        ])
        angle_type_row.addWidget(self.combo_angle_type, 1)
        look_dir_row = QHBoxLayout()
        look_dir_row.addWidget(QLabel("Radar look direction:"))
        self.combo_look_dir = QComboBox()
        self.combo_look_dir.addItems([
            "Right-looking (default -- Sentinel-1, ALOS-2, TerraSAR-X, "
            "COSMO-SkyMed, ICEYE, RADARSAT-2)",
            "Left-looking (rare -- only if positively confirmed)",
        ])
        look_dir_row.addWidget(self.combo_look_dir, 1)
        lv_layout.addWidget(self.row_inc)
        lv_layout.addWidget(self.row_angle)
        lv_layout.addLayout(angle_type_row)
        lv_layout.addLayout(look_dir_row)

        self.radio_enu.toggled.connect(self._update_lv_visibility)
        self._update_lv_visibility()
        layout.addWidget(lv_group)

        # ── Optional inputs ──────────────────────────────────────────
        opt_group = QGroupBox("Optional inputs")
        opt_layout = QVBoxLayout(opt_group)
        self.row_sigma = _FileRow("1-sigma uncertainty raster", required=False)
        opt_layout.addWidget(self.row_sigma)
        self.row_mask = _FileRow("Validity mask raster (e.g. coherence threshold)", required=False)
        opt_layout.addWidget(self.row_mask)
        mask_val_row = QHBoxLayout()
        mask_val_row.addWidget(QLabel("Mask valid value (blank = any value > 0):"))
        self.spin_mask_value = QDoubleSpinBox()
        self.spin_mask_value.setRange(-1e9, 1e9)
        self.spin_mask_value.setDecimals(4)
        self.check_mask_value = QCheckBox("Use exact value")
        self.check_mask_value.setChecked(False)
        self.spin_mask_value.setEnabled(False)
        self.check_mask_value.toggled.connect(self.spin_mask_value.setEnabled)
        mask_val_row.addWidget(self.check_mask_value)
        mask_val_row.addWidget(self.spin_mask_value)
        opt_layout.addLayout(mask_val_row)
        layout.addWidget(opt_group)

        # ── Downsampling ────────────────────────────────────────────
        ds_group = QGroupBox("Downsampling")
        ds_layout = QVBoxLayout(ds_group)
        self.radio_uniform = QRadioButton(
            "Uniform stride (simple, spends the same density everywhere)")
        self.radio_quadtree = QRadioButton(
            "Adaptive quadtree (denser near LOS gradients/fault trace, "
            "standard for slip inversion)")
        self.radio_quadtree.setChecked(True)
        ds_layout.addWidget(self.radio_uniform)
        ds_layout.addWidget(self.radio_quadtree)

        stride_row = QHBoxLayout()
        stride_row.addWidget(QLabel("Stride (pixels):"))
        self.spin_stride = QSpinBox()
        self.spin_stride.setRange(1, 1000)
        self.spin_stride.setValue(5)
        stride_row.addWidget(self.spin_stride)
        ds_layout.addLayout(stride_row)

        qt_form = QFormLayout()
        self.spin_max_points = QSpinBox()
        self.spin_max_points.setRange(10, 200000)
        self.spin_max_points.setValue(2000)
        qt_form.addRow("Max points (soft cap):", self.spin_max_points)
        self.spin_min_cell = QSpinBox()
        self.spin_min_cell.setRange(1, 256)
        self.spin_min_cell.setValue(4)
        qt_form.addRow("Min cell size (pixels):", self.spin_min_cell)
        self.check_auto_threshold = QCheckBox(
            "Auto std threshold (75th percentile of local variability)")
        self.check_auto_threshold.setChecked(True)
        self.spin_std_threshold = QDoubleSpinBox()
        self.spin_std_threshold.setRange(0.0, 1e6)
        self.spin_std_threshold.setDecimals(6)
        self.spin_std_threshold.setEnabled(False)
        self.check_auto_threshold.toggled.connect(
            lambda checked: self.spin_std_threshold.setEnabled(not checked))
        qt_form.addRow(self.check_auto_threshold, self.spin_std_threshold)
        # Default OFF (2026-08-30 fix): using each leaf's own local std as
        # a stand-in for measurement uncertainty, when no real sigma
        # raster is given, was CONFIRMED to destabilize
        # run_slip_inversion() -- flat far-field cells get a near-zero
        # std and hence an enormous implied weight (observed up to
        # ~8000x against a genuinely informative near-fault cell in one
        # real validation run), which can push scipy.optimize.lsq_linear
        # to fail to converge. Left OFF by default; only offered at all
        # for anyone who wants density-adaptive weighting and
        # understands that trade-off. Has no effect once a real "1-sigma
        # uncertainty raster" is given above -- that always wins.
        self.check_sigma_local_std = QCheckBox(
            "Use local variability as a stand-in uncertainty when no "
            "sigma raster is given (advanced — can destabilize the "
            "inversion; leave unchecked unless you understand why)")
        self.check_sigma_local_std.setChecked(False)
        qt_form.addRow(self.check_sigma_local_std)
        ds_layout.addLayout(qt_form)

        self.radio_uniform.toggled.connect(self._update_ds_visibility)
        self._update_ds_visibility()
        layout.addWidget(ds_group)

        # ── Action ──────────────────────────────────────────────────
        action_row = QHBoxLayout()
        self.btn_process = QPushButton("Load rasters && downsample")
        self.btn_process.setEnabled(self._gdal_ok)
        self.btn_process.clicked.connect(self._process)
        action_row.addWidget(self.btn_process)
        action_row.addStretch()
        layout.addLayout(action_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addWidget(QLabel("<b>Preview (first 10 points)</b>"))
        self.preview_table = QTableWidget(0, 7)
        self.preview_table.setHorizontalHeaderLabels(
            ["lon", "lat", "los", "look_e", "look_n", "look_u", "sigma"])
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.preview_table.setMaximumHeight(150)
        layout.addWidget(self.preview_table)

        add_row = QHBoxLayout()
        self.btn_add_batch = QPushButton("+ Add these points to the inversion")
        self.btn_add_batch.setEnabled(False)
        self.btn_add_batch.clicked.connect(self._do_add_batch)
        add_row.addWidget(self.btn_add_batch)
        add_row.addStretch()
        layout.addLayout(add_row)

    # ── Visibility toggles ────────────────────────────────────────────

    def _update_lv_visibility(self):
        enu = self.radio_enu.isChecked()
        for w in (self.row_e, self.row_n, self.row_u):
            w.setVisible(enu)
        for w in (self.row_inc, self.row_angle):
            w.setVisible(not enu)
        self.combo_angle_type.setVisible(not enu)
        self.combo_look_dir.setVisible(not enu)

    def _update_ds_visibility(self):
        uniform = self.radio_uniform.isChecked()
        self.spin_stride.setEnabled(uniform)
        self.spin_max_points.setEnabled(not uniform)
        self.spin_min_cell.setEnabled(not uniform)
        self.check_auto_threshold.setEnabled(not uniform)
        self.spin_std_threshold.setEnabled(not uniform and not self.check_auto_threshold.isChecked())

    # ── Processing ──────────────────────────────────────────────────

    def _process(self):
        if not self.row_los.path:
            QMessageBox.warning(self, "Missing input", "Choose a LOS raster first.")
            return

        los_sign = -1.0 if self.combo_los_sign.currentIndex() == 1 else 1.0
        mask_path = self.row_mask.path
        mask_val = self.spin_mask_value.value() if self.check_mask_value.isChecked() else None
        sigma_path = self.row_sigma.path

        try:
            if self.radio_enu.isChecked():
                if not (self.row_e.path and self.row_n.path and self.row_u.path):
                    QMessageBox.warning(self, "Missing input",
                                        "Choose all three E/N/U look-vector rasters.")
                    return
                lon2d, lat2d, los2d, le2d, ln2d, lu2d, sigma2d = load_los_with_enu_rasters(
                    self.row_los.path, self.row_e.path, self.row_n.path, self.row_u.path,
                    sigma_path=sigma_path, mask_path=mask_path, mask_valid_value=mask_val)
                los2d = los2d * los_sign
            else:
                if not (self.row_inc.path and self.row_angle.path):
                    QMessageBox.warning(self, "Missing input",
                                        "Choose both the incidence and azimuth/heading rasters.")
                    return
                angle_type = "head" if self.combo_angle_type.currentIndex() == 1 else "az"
                look_direction = "left" if self.combo_look_dir.currentIndex() == 1 else "right"
                lon2d, lat2d, los2d, le2d, ln2d, lu2d, sigma2d = load_los_with_angle_rasters(
                    self.row_los.path, self.row_inc.path, self.row_angle.path,
                    angle_type=angle_type, look_direction=look_direction,
                    sigma_path=sigma_path, mask_path=mask_path, mask_valid_value=mask_val,
                    los_sign=los_sign)
        except Exception as e:
            QMessageBox.critical(self, "Raster read error", f"{type(e).__name__}: {e}")
            return

        try:
            if self.radio_uniform.isChecked():
                rows = downsample_uniform(lon2d, lat2d, los2d, le2d, ln2d, lu2d,
                                          stride=self.spin_stride.value(), sigma2d=sigma2d)
            else:
                std_threshold = (None if self.check_auto_threshold.isChecked()
                                 else self.spin_std_threshold.value())
                sigma_fallback = ("local_std" if self.check_sigma_local_std.isChecked()
                                  else "none")
                rows = downsample_quadtree(
                    lon2d, lat2d, los2d, le2d, ln2d, lu2d,
                    max_points=self.spin_max_points.value(),
                    min_cell_px=self.spin_min_cell.value(),
                    std_threshold=std_threshold, sigma2d=sigma2d,
                    sigma_fallback=sigma_fallback)
        except Exception as e:
            QMessageBox.critical(self, "Downsampling error", f"{type(e).__name__}: {e}")
            return

        self._pending_rows = rows
        self.status_label.setText(
            f"<b>{len(rows)}</b> point(s) produced. Review the preview below, "
            f"then add them to the inversion.")
        self._update_preview()
        self.btn_add_batch.setEnabled(bool(rows))

    def _update_preview(self):
        preview_rows = self._pending_rows[:10]
        self.preview_table.setRowCount(len(preview_rows))
        cols = ["lon", "lat", "los", "look_e", "look_n", "look_u", "sigma"]
        for r, row in enumerate(preview_rows):
            for c, key in enumerate(cols):
                val = row.get(key)
                text = "" if val is None else (f"{val:.6g}" if isinstance(val, float) else str(val))
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.preview_table.setItem(r, c, item)

    def _do_add_batch(self):
        if not self._pending_rows:
            return
        self._on_batch_ready(self._pending_rows)
        self.status_label.setText(
            f"Added {len(self._pending_rows)} point(s) to the inversion. "
            f"Load another raster set to add more.")
        self._pending_rows = []
        self.btn_add_batch.setEnabled(False)
        self.preview_table.setRowCount(0)


class _ComponentRasterImportPanel(QWidget):
    """
    Raster (GeoTIFF) import panel for already-resolved DISPLACEMENT-
    COMPONENT rasters (East, North, Vertical/Up) -- a separate,
    GNSS-schema counterpart of _InsarRasterImportPanel above. Any ONE
    OR TWO of the three components may be omitted (a vertical-only
    product is common and fully supported) -- see
    core.insar_raster_import's module docstring, "COMPONENT-
    DISPLACEMENT RASTERS" section, for why this differs from the LOS
    path's all-or-nothing requirement.

    Same on_batch_ready(rows) callback contract as the other import
    panels in this module/slip_inversion_dialog.py, but rows are in
    observation_import.py's "gnss" schema ({"lon","lat","e","n","u",
    "sigma_e","sigma_n","sigma_u"}) -- intended to be wired to the SAME
    accumulating point list as the GNSS table/layer tab
    (_on_gnss_batch in SlipInversionDialog), not the LOS one, even
    though the source happens to be a raster.
    """

    def __init__(self, on_batch_ready, parent=None):
        super().__init__(parent)
        self._on_batch_ready = on_batch_ready
        self._pending_rows = []

        layout = QVBoxLayout(self)

        ok, msg = check_gdal_available()
        if not ok:
            warn = QLabel(
                f"<b style='color:#a33;'>GDAL not available:</b> {msg}")
            warn.setWordWrap(True)
            layout.addWidget(warn)
        self._gdal_ok = ok

        layout.addWidget(QLabel(
            "<i>Provide any one, two, or all three displacement-"
            "component rasters (e.g. an already-decomposed multi-track "
            "InSAR product, or any raster giving physical East/North/"
            "Vertical displacement directly -- NOT a LOS + look-vector "
            "product, which belongs on the other raster tab). A "
            "vertical-only import is fully supported.</i>"))

        # ── Component rasters ──────────────────────────────────────
        comp_group = QGroupBox("Displacement-component rasters (at least one required)")
        comp_layout = QVBoxLayout(comp_group)
        self.row_e = _FileRow("East (E/W) displacement raster", required=False)
        self.row_n = _FileRow("North (N/S) displacement raster", required=False)
        self.row_u = _FileRow("Vertical (Up/Down) displacement raster", required=False)
        comp_layout.addWidget(self.row_e)
        comp_layout.addWidget(self.row_n)
        comp_layout.addWidget(self.row_u)
        layout.addWidget(comp_group)

        # ── Optional inputs ────────────────────────────────────────
        opt_group = QGroupBox("Optional inputs")
        opt_layout = QVBoxLayout(opt_group)
        self.row_sigma_e = _FileRow("1-sigma uncertainty raster (East)", required=False)
        self.row_sigma_n = _FileRow("1-sigma uncertainty raster (North)", required=False)
        self.row_sigma_u = _FileRow("1-sigma uncertainty raster (Vertical)", required=False)
        opt_layout.addWidget(self.row_sigma_e)
        opt_layout.addWidget(self.row_sigma_n)
        opt_layout.addWidget(self.row_sigma_u)
        self.row_mask = _FileRow("Validity mask raster (applied to all components given)",
                                 required=False)
        opt_layout.addWidget(self.row_mask)
        mask_val_row = QHBoxLayout()
        mask_val_row.addWidget(QLabel("Mask valid value (blank = any value > 0):"))
        self.spin_mask_value = QDoubleSpinBox()
        self.spin_mask_value.setRange(-1e9, 1e9)
        self.spin_mask_value.setDecimals(4)
        self.check_mask_value = QCheckBox("Use exact value")
        self.check_mask_value.setChecked(False)
        self.spin_mask_value.setEnabled(False)
        self.check_mask_value.toggled.connect(self.spin_mask_value.setEnabled)
        mask_val_row.addWidget(self.check_mask_value)
        mask_val_row.addWidget(self.spin_mask_value)
        opt_layout.addLayout(mask_val_row)
        layout.addWidget(opt_group)

        # ── Downsampling ────────────────────────────────────────────
        ds_group = QGroupBox("Downsampling")
        ds_layout = QVBoxLayout(ds_group)
        self.radio_uniform = QRadioButton(
            "Uniform stride (simple, spends the same density everywhere)")
        self.radio_quadtree = QRadioButton(
            "Adaptive quadtree (denser where the NOISIEST provided "
            "component varies most, standard for slip inversion)")
        self.radio_quadtree.setChecked(True)
        ds_layout.addWidget(self.radio_uniform)
        ds_layout.addWidget(self.radio_quadtree)

        stride_row = QHBoxLayout()
        stride_row.addWidget(QLabel("Stride (pixels):"))
        self.spin_stride = QSpinBox()
        self.spin_stride.setRange(1, 1000)
        self.spin_stride.setValue(5)
        stride_row.addWidget(self.spin_stride)
        ds_layout.addLayout(stride_row)

        qt_form = QFormLayout()
        self.spin_max_points = QSpinBox()
        self.spin_max_points.setRange(10, 200000)
        self.spin_max_points.setValue(2000)
        qt_form.addRow("Max points (soft cap):", self.spin_max_points)
        self.spin_min_cell = QSpinBox()
        self.spin_min_cell.setRange(1, 256)
        self.spin_min_cell.setValue(4)
        qt_form.addRow("Min cell size (pixels):", self.spin_min_cell)
        self.check_auto_threshold = QCheckBox(
            "Auto std threshold per component (75th percentile of local variability)")
        self.check_auto_threshold.setChecked(True)
        self.spin_std_threshold = QDoubleSpinBox()
        self.spin_std_threshold.setRange(0.0, 1e6)
        self.spin_std_threshold.setDecimals(6)
        self.spin_std_threshold.setEnabled(False)
        self.check_auto_threshold.toggled.connect(
            lambda checked: self.spin_std_threshold.setEnabled(not checked))
        qt_form.addRow(self.check_auto_threshold, self.spin_std_threshold)
        # Same 2026-08-30 fix/rationale as _InsarRasterImportPanel's own
        # copy of this checkbox above -- see there for the full
        # explanation. Default OFF.
        self.check_sigma_local_std = QCheckBox(
            "Use local variability as a stand-in uncertainty when no "
            "sigma raster is given (advanced — can destabilize the "
            "inversion; leave unchecked unless you understand why)")
        self.check_sigma_local_std.setChecked(False)
        qt_form.addRow(self.check_sigma_local_std)
        ds_layout.addLayout(qt_form)

        self.radio_uniform.toggled.connect(self._update_ds_visibility)
        self._update_ds_visibility()
        layout.addWidget(ds_group)

        # ── Action ──────────────────────────────────────────────────
        action_row = QHBoxLayout()
        self.btn_process = QPushButton("Load rasters && downsample")
        self.btn_process.setEnabled(self._gdal_ok)
        self.btn_process.clicked.connect(self._process)
        action_row.addWidget(self.btn_process)
        action_row.addStretch()
        layout.addLayout(action_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addWidget(QLabel("<b>Preview (first 10 points)</b>"))
        self.preview_table = QTableWidget(0, 8)
        self.preview_table.setHorizontalHeaderLabels(
            ["lon", "lat", "e", "n", "u", "sigma_e", "sigma_n", "sigma_u"])
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.preview_table.setMaximumHeight(150)
        layout.addWidget(self.preview_table)

        add_row = QHBoxLayout()
        self.btn_add_batch = QPushButton("+ Add these points to the inversion")
        self.btn_add_batch.setEnabled(False)
        self.btn_add_batch.clicked.connect(self._do_add_batch)
        add_row.addWidget(self.btn_add_batch)
        add_row.addStretch()
        layout.addLayout(add_row)

    def _update_ds_visibility(self):
        uniform = self.radio_uniform.isChecked()
        self.spin_stride.setEnabled(uniform)
        self.spin_max_points.setEnabled(not uniform)
        self.spin_min_cell.setEnabled(not uniform)
        self.check_auto_threshold.setEnabled(not uniform)
        self.spin_std_threshold.setEnabled(not uniform and not self.check_auto_threshold.isChecked())

    def _process(self):
        if not (self.row_e.path or self.row_n.path or self.row_u.path):
            QMessageBox.warning(self, "Missing input",
                                "Choose at least one of East/North/Vertical rasters.")
            return

        mask_path = self.row_mask.path
        mask_val = self.spin_mask_value.value() if self.check_mask_value.isChecked() else None

        try:
            lon2d, lat2d, e2d, n2d, u2d, sigma_e2d, sigma_n2d, sigma_u2d = load_component_rasters(
                e_path=self.row_e.path, n_path=self.row_n.path, u_path=self.row_u.path,
                sigma_e_path=self.row_sigma_e.path, sigma_n_path=self.row_sigma_n.path,
                sigma_u_path=self.row_sigma_u.path,
                mask_path=mask_path, mask_valid_value=mask_val)
        except Exception as e:
            QMessageBox.critical(self, "Raster read error", f"{type(e).__name__}: {e}")
            return

        try:
            if self.radio_uniform.isChecked():
                rows = downsample_components_uniform(
                    lon2d, lat2d, e2d, n2d, u2d, stride=self.spin_stride.value(),
                    sigma_e2d=sigma_e2d, sigma_n2d=sigma_n2d, sigma_u2d=sigma_u2d)
            else:
                std_threshold = (None if self.check_auto_threshold.isChecked()
                                 else self.spin_std_threshold.value())
                sigma_fallback = ("local_std" if self.check_sigma_local_std.isChecked()
                                  else "none")
                rows = downsample_components_quadtree(
                    lon2d, lat2d, e2d, n2d, u2d,
                    max_points=self.spin_max_points.value(),
                    min_cell_px=self.spin_min_cell.value(),
                    std_threshold=std_threshold,
                    sigma_e2d=sigma_e2d, sigma_n2d=sigma_n2d, sigma_u2d=sigma_u2d,
                    sigma_fallback=sigma_fallback)
        except Exception as e:
            QMessageBox.critical(self, "Downsampling error", f"{type(e).__name__}: {e}")
            return

        self._pending_rows = rows
        self.status_label.setText(
            f"<b>{len(rows)}</b> point(s) produced. Review the preview below, "
            f"then add them to the inversion.")
        self._update_preview()
        self.btn_add_batch.setEnabled(bool(rows))

    def _update_preview(self):
        preview_rows = self._pending_rows[:10]
        self.preview_table.setRowCount(len(preview_rows))
        cols = ["lon", "lat", "e", "n", "u", "sigma_e", "sigma_n", "sigma_u"]
        for r, row in enumerate(preview_rows):
            for c, key in enumerate(cols):
                val = row.get(key)
                text = "" if val is None else (f"{val:.6g}" if isinstance(val, float) else str(val))
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.preview_table.setItem(r, c, item)

    def _do_add_batch(self):
        if not self._pending_rows:
            return
        self._on_batch_ready(self._pending_rows)
        self.status_label.setText(
            f"Added {len(self._pending_rows)} point(s) to the inversion. "
            f"Load another raster set to add more.")
        self._pending_rows = []
        self.btn_add_batch.setEnabled(False)
        self.preview_table.setRowCount(0)
