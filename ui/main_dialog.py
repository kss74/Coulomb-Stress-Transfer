# -*- coding: utf-8 -*-
"""Main tabbed dialog for the Coulomb Stress Transfer plugin."""

import os
import numpy as np

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QFormLayout,
    QDoubleSpinBox, QPushButton, QLabel, QProgressBar, QFileDialog,
    QComboBox, QGroupBox, QScrollArea, QTextEdit, QCheckBox,
    QListWidget, QInputDialog
)
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal

from ..core.okada_engine import (
    FaultParameters, ElasticParameters, GridParameters,
    compute_coulomb_grid, compute_coulomb_grid_depth,
    compute_cff_on_receiver_faults,
    compute_surface_deformation, compute_surface_deformation_depth,
    compute_cross_section, compute_cross_section_multi, _has_okada_wrapper,
    grid_counts_from_spacing,
    near_field_fault_pairs,
    total_seismic_moment, format_seismic_moment_message,
)
from ..core.optimal_plane import (
    RegionalStress, compute_optimal_cff_grid_depth,
    compute_cross_section_optimal, compute_cross_section_optimal_multi,
)
from ..core.focal_mechanism import (
    compute_focal_mechanism_cff, PLANE_MODES, build_source_fault_row,
)
from ..core.geo_profile import (
    project_points_to_profile, project_points_to_polyline,
    filter_within_search_width, profile_direction, polyline_segment_info,
)
from ..core.cross_section_config import (
    CrossSectionConfig, TopoPanelConfig, AnnotationOverlayConfig,
    ExtraSectionLineConfig,
)
from ..core.cross_section_faults import (
    project_fault_traces_onto_section, project_fault_traces_onto_polyline,
)
from ..core.cross_section_plot import build_cross_section_figure
from ..core.raster_profile import sample_raster_along_line, sample_raster_along_polyline
from ..core.annotation_gather import gather_annotation_points
from .cross_section_config_dialog import CrossSectionConfigDialog
from ..utils import vector_utils
from .fault_table_widget import FaultTableWidget
from .plot_widget import PlotWidget
from .receiver_results_widget import ReceiverResultsWidget
from .cross_section_window import CrossSectionWindow
from .eq_catalog_import_dialog import EQCatalogImportDialog
from .point_calc_dialog import PointCalculatorDialog
from .focal_import_dialog import FocalMechanismImportDialog
from .focal_mechanism_widget import build_focal_mechanism_tab
from .dialog_utils import configure_resizable_dialog
from .aftershock_mc_dialog import AftershockMCTestDialog
from .rate_state_dialog import RateStateForecastDialog
from .stress_inversion_dialog import StressInversionDialog


def spin(minv, maxv, val, decimals=2, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(minv, maxv)
    s.setValue(val)
    s.setDecimals(decimals)
    if suffix:
        s.setSuffix(f" {suffix}")
    return s


_SETTINGS_KEY_GRID_MAX_POINTS = "CoulombStressTransfer/grid_max_points_per_axis"
DEFAULT_GRID_MAX_POINTS_PER_AXIS = 2000


def _get_default_grid_max_points_per_axis():
    """
    Return the persisted default safety cap on grid points per axis, via
    QgsSettings so it's remembered across QGIS sessions/restarts -- mirrors
    the same pattern used for the default Lon/Lat mode
    (ui/fault_table_widget.py, _get_default_lonlat_mode()). Falls back to
    DEFAULT_GRID_MAX_POINTS_PER_AXIS if unset or invalid.

    This is a defensive cap, not a physics-derived limit: it exists only to
    stop a mis-specified spacing (e.g. sub-meter spacing over a
    multi-degree extent) from silently trying to build a grid large enough
    to hang the UI. See grid_counts_from_spacing() in core/okada_engine.py.
    """
    try:
        from qgis.core import QgsSettings
        value = QgsSettings().value(
            _SETTINGS_KEY_GRID_MAX_POINTS,
            DEFAULT_GRID_MAX_POINTS_PER_AXIS, type=int)
    except Exception:
        return DEFAULT_GRID_MAX_POINTS_PER_AXIS
    return value if value and value >= 2 else DEFAULT_GRID_MAX_POINTS_PER_AXIS


def _set_default_grid_max_points_per_axis(value):
    """Persist the grid points-per-axis safety cap via QgsSettings."""
    try:
        from qgis.core import QgsSettings
        QgsSettings().setValue(_SETTINGS_KEY_GRID_MAX_POINTS, int(value))
    except Exception:
        pass


class ComputeWorker(QThread):
    """
    Background worker for CFF / surface-deformation / cross-section
    computations, keeping the UI responsive during (potentially slow,
    subprocess-based) DC3D calls.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, mode, sources, receiver, elastic, grid=None,
                cross_section_params=None, receiver_faults=None,
                regional=None, opt_friction=None,
                focal_events=None, focal_mode=None):
        super().__init__()
        self.mode = mode  # 'cff' | 'displacement' | 'cross_section' | 'receiver_faults' | 'optimal' | 'focal_mech'
        self.sources = sources
        self.receiver = receiver
        self.elastic = elastic
        self.grid = grid
        self.cross_section_params = cross_section_params
        self.receiver_faults = receiver_faults
        self.regional = regional          # RegionalStress, 'optimal' mode only
        self.opt_friction = opt_friction  # friction override, 'optimal' mode only
        self.focal_events = focal_events  # list[FocalMechanismEvent], 'focal_mech' mode only
        self.focal_mode = focal_mode      # one of PLANE_MODES, 'focal_mech' mode only

    def run(self):
        try:
            result = {}
            if self.mode == "cff":
                lon2d, lat2d, cff, used_dc3d, near_field_mask = compute_coulomb_grid_depth(
                    self.sources, self.receiver, self.elastic, self.grid,
                    progress_callback=self.progress.emit,
                )
                result.update(lon2d=lon2d, lat2d=lat2d, cff=cff, used_dc3d=used_dc3d,
                             near_field_mask=near_field_mask)

            elif self.mode == "displacement":
                lon2d, lat2d, ux, uy, uz, used_dc3d = compute_surface_deformation_depth(
                    self.sources, self.elastic, self.grid,
                    progress_callback=self.progress.emit,
                )
                result.update(lon2d=lon2d, lat2d=lat2d, ux=ux, uy=uy, uz=uz,
                             used_dc3d=used_dc3d)

            elif self.mode == "cross_section":
                p = self.cross_section_params
                cff_source = p.get("cff_source", "receiver")
                # 2026-08-21: profiles are now always a vertex list
                # (>=2 points) -- a plain 2-point lon1,lat1->lon2,lat2
                # profile is just the degenerate 1-segment case, see
                # core.geo_profile.polyline_segment_info(). Falls back
                # to building a 2-vertex list from the legacy
                # lon1/lat1/lon2/lat2 keys if "vertices" wasn't supplied
                # (keeps any external caller that still builds params
                # the old way working).
                vertices = p.get("vertices") or [
                    (p["lon1"], p["lat1"]), (p["lon2"], p["lat2"])]
                if cff_source == "optimal":
                    if self.regional is None:
                        raise RuntimeError(
                            "CFF source is set to 'Optimal Plane' but no "
                            "regional stress was supplied -- set the "
                            "Regional Stress tab first (same requirement "
                            "as the Opt Faults map-view tab).")
                    dist_km, depth_km, cff_2d, used_dc3d, seg_info = \
                        compute_cross_section_optimal_multi(
                            self.sources, self.regional, self.elastic, vertices,
                            p["dist_increment_km"], p["max_depth_km"], p["depth_increment_km"],
                            friction=self.opt_friction, progress_callback=self.progress.emit,
                        )
                else:
                    dist_km, depth_km, cff_2d, used_dc3d, seg_info = compute_cross_section_multi(
                        self.sources, self.receiver, self.elastic, vertices,
                        p["dist_increment_km"], p["max_depth_km"], p["depth_increment_km"],
                        progress_callback=self.progress.emit,
                    )
                result.update(dist_km=dist_km, depth_km=depth_km, cff_2d=cff_2d,
                             used_dc3d=used_dc3d, segment_info=seg_info, profile_vertices=vertices)

            elif self.mode == "receiver_faults":
                receiver_results = compute_cff_on_receiver_faults(
                    self.sources, self.receiver_faults, self.elastic,
                    progress_callback=self.progress.emit,
                )
                result.update(receiver_results=receiver_results)

            elif self.mode == "optimal":
                (lon2d, lat2d, cff_opt_mpa, strike1, dip1, rake1,
                 strike2, dip2, rake2, orthogonality_error_deg,
                 cff1_mpa, cff2_mpa, used_dc3d) = compute_optimal_cff_grid_depth(
                    self.sources, self.regional, self.elastic, self.grid,
                    friction=self.opt_friction, progress_callback=self.progress.emit,
                )
                result.update(lon2d=lon2d, lat2d=lat2d, cff_opt_mpa=cff_opt_mpa,
                             strike1=strike1, dip1=dip1, rake1=rake1,
                             strike2=strike2, dip2=dip2, rake2=rake2,
                             orthogonality_error_deg=orthogonality_error_deg,
                             cff1_mpa=cff1_mpa, cff2_mpa=cff2_mpa, used_dc3d=used_dc3d)

            elif self.mode == "focal_mech":
                focal_results = compute_focal_mechanism_cff(
                    self.sources, self.focal_events, self.elastic,
                    mode=self.focal_mode, progress_callback=self.progress.emit,
                )
                result.update(focal_results=focal_results)

            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class CoulombMainDialog(QDialog):
    def __init__(self, iface):
        super().__init__(iface.mainWindow())
        self.iface = iface
        self.setWindowTitle("Coulomb Stress Transfer")
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )
        self.setMinimumSize(1000, 680)
        self._last_result = None
        self._last_mode = None
        # Last-used configuration widgets for the two "own-QThread-worker"
        # sub-dialogs (RateStateForecastDialog / AftershockMCTestDialog),
        # which are reconstructed fresh on every open_*_action() call and
        # so have no persistent instance of their own to hold this state
        # between openings. Captured via each dialog's get_settings() when
        # it closes, applied via apply_settings() the next time one opens,
        # and round-tripped through project_io.build_setup_dict/
        # apply_setup_dict so "Save Setup" captures them too (2026-08-24).
        self._rate_state_dialog_settings = None
        self._aftershock_dialog_settings = None
        self.compute_thread = None
        self.eq_events = []      # List[EQCatalogEvent], cross-section EQ overlay (2026-08-18b)
        self._xs_window = None   # lazily-created CrossSectionWindow, reused across recomputes
        self.xs_topo_panels = []      # List[TopoPanelConfig], cross-section topo panel(s) (2026-08-19)
        self.xs_annotations = []      # List[AnnotationOverlayConfig], cross-section annotation source(s)
        self._focal_mech_results = None  # last compute_focal_mechanisms_action() output,
                                          # kept independent of self._last_result so the
                                          # cross-section focal-mechanism overlay can still
                                          # color by CFF after switching tabs (2026-08-19)
        self.xs_config = CrossSectionConfig()  # persistent display-symbology config, edited
                                          # by CrossSectionConfigDialog (Phase 3, 2026-08-19);
                                          # .topo_panels/.annotations are the SAME list objects
                                          # as self.xs_topo_panels/self.xs_annotations (aliased
                                          # below), not copies, so add/remove there and cosmetic
                                          # edits in the dialog both land on one shared list.
        self.xs_config.topo_panels = self.xs_topo_panels
        self.xs_config.annotations = self.xs_annotations

        main_layout = QVBoxLayout(self)
        content = QHBoxLayout()
        main_layout.addLayout(content)

        # ── Left: tabs ──────────────────────────────────────────────────
        self.tabs = QTabWidget()
        content.addWidget(self.tabs, 2)

        self._build_sources_tab()
        self._build_receiver_tab()
        self._build_grid_tab()
        self._build_elastic_tab()
        self._build_cross_section_tab()
        self._build_receiver_faults_tab()
        self._build_optimal_tab()
        build_focal_mechanism_tab(self)

        # ── Right: preview + actions ────────────────────────────────────
        right = QVBoxLayout()
        content.addLayout(right, 3)

        self.plot_widget = PlotWidget()
        right.addWidget(self.plot_widget, 1)  # stretch=1: the plot always
        # gets priority for available vertical space -- the Notes area
        # below is deliberately confined to a fixed height (see below)
        # so it can never squeeze the plot down regardless of how much
        # status text accumulates in one run (2026-08-15c fix; see
        # module-level note above self.status_label's construction).

        action_row = QHBoxLayout()
        self.btn_compute = QPushButton("▶  Compute Coulomb Stress")
        self.btn_compute.clicked.connect(self.compute_cff)
        action_row.addWidget(self.btn_compute)

        self.btn_compute_disp = QPushButton("▶  Compute Surface Deformation")
        self.btn_compute_disp.clicked.connect(self.compute_displacement)
        action_row.addWidget(self.btn_compute_disp)

        self.btn_compute_xs = QPushButton("▶  Compute Cross-Section")
        self.btn_compute_xs.clicked.connect(self.compute_cross_section_action)
        action_row.addWidget(self.btn_compute_xs)

        # 2026-08-20 follow-up (item 13): re-show the existing
        # cross-section window without recomputing anything -- the
        # window itself now survives closing/reopening the plugin (see
        # cross_section_window.py's parenting fix), but if the user
        # closed just the popup window itself (not the whole plugin),
        # there was previously no way back to it except recomputing.
        self.btn_show_xs = QPushButton("🗗  Show Cross-Section Window")
        self.btn_show_xs.clicked.connect(self.show_cross_section_window_action)
        action_row.addWidget(self.btn_show_xs)
        right.addLayout(action_row)

        action_row2 = QHBoxLayout()
        self.btn_compute_optimal = QPushButton("▶  Compute Optimal-Plane ΔCFF")
        self.btn_compute_optimal.clicked.connect(self.compute_optimal_action)
        action_row2.addWidget(self.btn_compute_optimal)
        right.addLayout(action_row2)

        action_row3 = QHBoxLayout()
        self.btn_aftershock_mc_test = QPushButton("▶  Aftershock / ΔCFF Null Test…")
        self.btn_aftershock_mc_test.clicked.connect(self.open_aftershock_mc_test_action)
        action_row3.addWidget(self.btn_aftershock_mc_test)
        self.btn_rate_state_forecast = QPushButton("▶  Rate-and-State Forecast…")
        self.btn_rate_state_forecast.clicked.connect(self.open_rate_state_forecast_action)
        action_row3.addWidget(self.btn_rate_state_forecast)
        self.btn_point_calculator = QPushButton("▶  Point Calculator…")
        self.btn_point_calculator.clicked.connect(self.open_point_calculator_action)
        action_row3.addWidget(self.btn_point_calculator)
        right.addLayout(action_row3)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right.addWidget(self.progress_bar)

        export_row1 = QHBoxLayout()
        self.btn_add_raster = QPushButton("Add to Project (temp raster)")
        self.btn_add_raster.clicked.connect(self.add_raster_to_project)
        export_row1.addWidget(self.btn_add_raster)

        self.btn_save_raster = QPushButton("Save Raster to File…")
        self.btn_save_raster.clicked.connect(self.save_raster_to_file)
        export_row1.addWidget(self.btn_save_raster)

        self.btn_save_plot = QPushButton("Save Plot As…")
        self.btn_save_plot.clicked.connect(self.save_plot_to_file)
        export_row1.addWidget(self.btn_save_plot)
        right.addLayout(export_row1)

        export_row2 = QHBoxLayout()
        self.btn_export_csv = QPushButton("Export → CSV/XYZ")
        self.btn_export_csv.clicked.connect(self.export_csv)
        export_row2.addWidget(self.btn_export_csv)

        self.btn_export_vectors = QPushButton("Export → Fault/Vector Layers")
        self.btn_export_vectors.clicked.connect(self.export_vectors)
        export_row2.addWidget(self.btn_export_vectors)
        right.addLayout(export_row2)

        setup_row = QHBoxLayout()
        self.btn_save_setup = QPushButton("💾 Save Setup…")
        self.btn_save_setup.clicked.connect(self.save_setup_action)
        setup_row.addWidget(self.btn_save_setup)

        self.btn_load_setup = QPushButton("📂 Load Setup…")
        self.btn_load_setup.clicked.connect(self.load_setup_action)
        setup_row.addWidget(self.btn_load_setup)

        self.btn_export_inp = QPushButton("Export → Coulomb .inp…")
        self.btn_export_inp.clicked.connect(self.export_inp_action)
        setup_row.addWidget(self.btn_export_inp)

        self.btn_import_inp = QPushButton("Import ← Coulomb .inp…")
        self.btn_import_inp.clicked.connect(self.import_inp_action)
        setup_row.addWidget(self.btn_import_inp)
        right.addLayout(setup_row)

        # ── Notes / status area ─────────────────────────────────────────
        # Was previously a plain QLabel added directly into `right`'s
        # QVBoxLayout with no stretch factor -- a run that combined the
        # mode-specific message + total-moment readout + several
        # pairwise near-field warnings (see compute_cff()'s tail, where
        # these are concatenated together) could grow tall enough to
        # squeeze self.plot_widget down to almost nothing, since the
        # layout had no reason to prefer the plot's space over the
        # label's. Fixed by (a) giving the plot an explicit stretch
        # factor above so it always keeps its space, and (b) confining
        # this label to a fixed-height SCROLLABLE strip instead of
        # letting it grow the layout. All existing
        # `self.status_label.setText(...)` call sites elsewhere in this
        # file are UNCHANGED -- this is still a plain QLabel, just
        # wrapped in a QScrollArea, so nothing about how status text is
        # set/concatenated needed to change.
        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        notes_scroll = QScrollArea()
        notes_scroll.setWidgetResizable(True)
        notes_scroll.setWidget(self.status_label)
        notes_scroll.setFixedHeight(110)
        notes_scroll.setFrameShape(QScrollArea.StyledPanel)
        right.addWidget(notes_scroll, 0)  # stretch=0: never grows

        notes_btn_row = QHBoxLayout()
        notes_btn_row.addStretch()
        self.btn_full_notes = QPushButton("🗒 Open notes in window…")
        self.btn_full_notes.clicked.connect(self._open_notes_dialog)
        notes_btn_row.addWidget(self.btn_full_notes)
        right.addLayout(notes_btn_row)

    # ── Notes window ─────────────────────────────────────────────────────

    def _open_notes_dialog(self):
        """
        Pop out the CURRENT full contents of the (fixed-height, scrollable)
        Notes strip into a larger, independently resizable window -- for
        reading a long status message (several near-field warnings, a
        moment/Mw readout, etc. all concatenated together) comfortably
        rather than scrolling a small strip. Not live-updating -- it's a
        snapshot dialog; re-open it after the next computation to see
        the new text (matches how every other "…" dialog in this plugin
        behaves, e.g. the QA report/export dialogs).
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Coulomb Stress Transfer — Notes")
        configure_resizable_dialog(dlg, 620, 420, min_width=320, min_height=200)
        layout = QVBoxLayout(dlg)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(self.status_label.text())
        layout.addWidget(text)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)
        dlg.exec_()

    # ── Tab builders ────────────────────────────────────────────────────

    def _build_sources_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("<b>Source Faults</b> (one row per fault)"))
        self.fault_table = FaultTableWidget()
        self.fault_table.set_elastic_provider(self._get_elastic)
        layout.addWidget(self.fault_table)
        self.tabs.addTab(tab, "Source Faults")

    def _build_receiver_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)
        form.addRow(QLabel("<b>Receiver Fault Geometry</b>"))
        self.r_strike = spin(0, 360, 0.0, 1, "°")
        self.r_dip    = spin(0, 90, 90.0, 1, "°")
        self.r_rake   = spin(-180, 180, 0.0, 1, "°")
        form.addRow("Strike:", self.r_strike)
        form.addRow("Dip:", self.r_dip)
        form.addRow("Rake:", self.r_rake)
        self.tabs.addTab(tab, "Receiver Fault")

    def _build_grid_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)
        form.addRow(QLabel("<b>Output Grid</b>"))
        self.g_lon_min = spin(-180, 180, -1.0, 3, "°")
        self.g_lon_max = spin(-180, 180, 1.0, 3, "°")
        self.g_lat_min = spin(-90, 90, -1.0, 3, "°")
        self.g_lat_max = spin(-90, 90, 1.0, 3, "°")
        self.g_depth = spin(0, 700, 0.0, 1, "km")

        form.addRow("Lon min:", self.g_lon_min)
        form.addRow("Lon max:", self.g_lon_max)
        form.addRow("Lat min:", self.g_lat_min)
        form.addRow("Lat max:", self.g_lat_max)

        # ── Grid resolution: point-count (legacy) OR spacing in km/degrees ──
        form.addRow(QLabel("<hr><b>Grid Resolution</b>"))

        self.g_res_mode = QComboBox()
        self.g_res_mode.addItems([
            "Grid point count", "Spacing (km)", "Spacing (degrees)"])
        form.addRow("Defined by:", self.g_res_mode)

        cap_default = _get_default_grid_max_points_per_axis()
        self.g_n_lon = spin(2, cap_default, min(100, cap_default), 0)
        self.g_n_lat = spin(2, cap_default, min(100, cap_default), 0)
        form.addRow("Grid points (lon):", self.g_n_lon)
        form.addRow("Grid points (lat):", self.g_n_lat)

        self.g_spacing = spin(0.0001, 10000, 1.0, 4)
        form.addRow("Spacing:", self.g_spacing)
        form.addRow(QLabel(
            "<i>\"Spacing (km)\" converts to lon/lat degrees using the "
            "grid's center latitude (1° of longitude covers less ground "
            "away from the equator, same convention used everywhere else "
            "in this plugin) — spacing is exact at the center latitude and "
            "drifts slightly toward the grid's north/south edges. \"Spacing "
            "(degrees)\" is exact on both axes.</i>"))

        self.lbl_resolved_grid = QLabel()
        self.lbl_resolved_grid.setWordWrap(True)
        form.addRow(self.lbl_resolved_grid)

        self.g_max_points_per_axis = spin(2, 1_000_000, cap_default, 0)
        form.addRow("Max points per axis (safety cap):",
                     self.g_max_points_per_axis)
        form.addRow(QLabel(
            "<i>Defensive cap only — not physics-derived. Prevents a "
            "mis-specified spacing (e.g. sub-meter spacing over a "
            "multi-degree extent) from silently building a grid large "
            "enough to hang the UI. Raise it if you deliberately want a "
            "finer regional grid; large values mean more DC3D calls per "
            "source fault and longer compute times. Remembered across QGIS "
            "sessions.</i>"))

        form.addRow("Receiver / observation depth:", self.g_depth)
        form.addRow(QLabel(
            "<i>Used as the CFF receiver depth AND the surface-deformation "
            "observation depth (0 = free surface).</i>"))

        self.lbl_depth_status = QLabel()
        self.lbl_depth_status.setWordWrap(True)
        self._update_depth_status()
        form.addRow(self.lbl_depth_status)

        self.g_res_mode.currentIndexChanged.connect(self._on_res_mode_changed)
        for w in (self.g_lon_min, self.g_lon_max, self.g_lat_min, self.g_lat_max,
                 self.g_spacing, self.g_n_lon, self.g_n_lat):
            w.valueChanged.connect(self._update_resolved_grid_label)
        self.g_max_points_per_axis.valueChanged.connect(
            self._on_max_points_per_axis_changed)
        self._on_res_mode_changed()

        self.tabs.addTab(tab, "Grid Output")

    def _on_res_mode_changed(self):
        """Enable only the controls relevant to the selected resolution mode."""
        is_points = self.g_res_mode.currentText() == "Grid point count"
        self.g_n_lon.setEnabled(is_points)
        self.g_n_lat.setEnabled(is_points)
        self.g_spacing.setEnabled(not is_points)
        self._update_resolved_grid_label()

    def _on_max_points_per_axis_changed(self):
        """Persist the new safety cap, widen/narrow the point-count spin
        boxes to match, and refresh the resolved-grid label."""
        cap = int(self.g_max_points_per_axis.value())
        _set_default_grid_max_points_per_axis(cap)
        for w in (self.g_n_lon, self.g_n_lat):
            w.setMaximum(cap)
        self._update_resolved_grid_label()

    def _update_resolved_grid_label(self):
        """Show the effective grid point counts (and a clamp warning, if
        any) for the current resolution settings, regardless of mode."""
        mode = self.g_res_mode.currentText()
        if mode == "Grid point count":
            n_lon, n_lat = int(self.g_n_lon.value()), int(self.g_n_lat.value())
            self.lbl_resolved_grid.setText(f"→ {n_lon} × {n_lat} grid points")
            self.lbl_resolved_grid.setStyleSheet("color: gray;")
            return

        units = "km" if mode == "Spacing (km)" else "deg"
        cap = int(self.g_max_points_per_axis.value())
        try:
            n_lon, n_lat, clamped = grid_counts_from_spacing(
                self.g_lon_min.value(), self.g_lon_max.value(),
                self.g_lat_min.value(), self.g_lat_max.value(),
                self.g_spacing.value(), units=units,
                max_points_per_axis=cap)
        except ValueError as e:
            self.lbl_resolved_grid.setText(f"⚠️ {e}")
            self.lbl_resolved_grid.setStyleSheet("color: darkorange;")
            return

        if clamped:
            self.lbl_resolved_grid.setText(
                f"→ {n_lon} × {n_lat} grid points (⚠️ capped — requested "
                f"spacing would exceed {cap} points on an axis; raise "
                f"\"Max points per axis\" above if this was intentional)")
            self.lbl_resolved_grid.setStyleSheet("color: darkorange;")
        else:
            self.lbl_resolved_grid.setText(f"→ {n_lon} × {n_lat} grid points")
            self.lbl_resolved_grid.setStyleSheet("color: gray;")

    def _build_elastic_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)
        form.addRow(QLabel("<b>Elastic Parameters</b>"))
        self.e_mu = spin(1e9, 1e11, 3.2e10, 0, "Pa")
        self.e_mu.setSingleStep(1e9)
        self.e_nu = spin(0.0, 0.5, 0.25, 3)
        self.e_friction = spin(0.0, 1.0, 0.4, 2)
        form.addRow("Shear modulus μ:", self.e_mu)
        form.addRow("Poisson's ratio ν:", self.e_nu)
        form.addRow("Friction coefficient μ':", self.e_friction)
        self.tabs.addTab(tab, "Elastic Params")

    def _build_cross_section_tab(self):
        # Wrapped in a QScrollArea (2026-08-22) -- this tab accumulated the
        # most rows of any tab across the 2026-08-18b..08-21 cross-section
        # overhaul sessions (profile, CFF source, waypoints, search width,
        # display toggles, focal mechanisms, extra depth-section elements,
        # topo panels, annotations, export) and had no scroll mechanism, so
        # on smaller screens/dialog heights everything above got squeezed
        # into an unusably compressed strip. Same setWidgetResizable(True)
        # pattern already used for the notes strip earlier in this file
        # (see `notes_scroll` in _build... above). The inner content widget
        # (`content`) is what actually holds the QFormLayout; `scroll` is
        # what gets added to the tab widget below.
        content = QWidget()
        form = QFormLayout(content)
        form.addRow(QLabel(
            "<b>Cross-Section</b><br><i>Vertical ΔCFF profile below a "
            "surface line, matching Coulomb's cross-section tool. The "
            "surface (z=0) row is always exact; rows below the surface "
            "require an external Python with okada-wrapper configured.</i>"))

        self.xs_lon1 = spin(-180, 180, -1.0, 4, "°")
        self.xs_lat1 = spin(-90, 90, -1.0, 4, "°")
        self.xs_lon2 = spin(-180, 180, 1.0, 4, "°")
        self.xs_lat2 = spin(-90, 90, 1.0, 4, "°")
        self.xs_dist_inc = spin(0.1, 100, 5.0, 2, "km")
        self.xs_max_depth = spin(1, 700, 30.0, 1, "km")
        self.xs_depth_inc = spin(0.1, 100, 2.0, 2, "km")

        form.addRow("Start lon:", self.xs_lon1)
        form.addRow("Start lat:", self.xs_lat1)
        form.addRow("Finish lon:", self.xs_lon2)
        form.addRow("Finish lat:", self.xs_lat2)
        form.addRow("Distance increment:", self.xs_dist_inc)
        form.addRow("Max depth:", self.xs_max_depth)
        form.addRow("Depth increment:", self.xs_depth_inc)

        form.addRow(QLabel(
            "<b>CFF source</b> <i>(answers \"is the displayed "
            "ΔCFF resolved onto the receiver fault or the optimal "
            "plane?\")</i>"))
        self.xs_cff_source = QComboBox()
        self.xs_cff_source.addItems(["Receiver Fault", "Optimal Plane"])
        self.xs_cff_source.setToolTip(
            "Receiver Fault: ΔCFF resolved onto the single fixed "
            "orientation set in the Receiver Fault tab (strike/dip/"
            "rake).\n"
            "Optimal Plane: ΔCFF resolved onto the optimally-oriented "
            "plane at each point, using the Regional Stress tab's "
            "tensor (King/Stein/Lin 1994) -- same requirement as the "
            "Opt Faults map-view tab.")
        form.addRow("Resolve ΔCFF onto:", self.xs_cff_source)

        form.addRow(QLabel(
            "<b>Multi-Segment Profile</b> <i>(optional "
            "waypoints inserted between Start and Finish above, letting "
            "a section bend to follow several strikes -- e.g. an "
            "arcuate subduction trench. Leave empty for a single "
            "straight segment.)</i>"))
        self.xs_waypoints = []  # list of (lon, lat), between Start/Finish
        self.xs_waypoint_list = QListWidget()
        self.xs_waypoint_list.setMaximumHeight(70)
        form.addRow(self.xs_waypoint_list)
        xs_waypoint_row = QHBoxLayout()
        self.btn_xs_add_waypoint = QPushButton("Add waypoint…")
        self.btn_xs_add_waypoint.setToolTip(
            "Insert a lon/lat waypoint, appended to the end of the list "
            "(between the current last waypoint and Finish). Use "
            "'Remove selected' to delete a specific one out of order.")
        self.btn_xs_add_waypoint.clicked.connect(self.add_xs_waypoint_action)
        xs_waypoint_row.addWidget(self.btn_xs_add_waypoint)
        self.btn_xs_remove_waypoint = QPushButton("Remove selected")
        self.btn_xs_remove_waypoint.clicked.connect(self.remove_xs_waypoint_action)
        xs_waypoint_row.addWidget(self.btn_xs_remove_waypoint)
        self.btn_xs_import_profile_polyline = QPushButton(
            "Import profile from QGIS polyline…")
        self.btn_xs_import_profile_polyline.clicked.connect(
            self.import_xs_profile_polyline_action)
        xs_waypoint_row.addWidget(self.btn_xs_import_profile_polyline)
        self.btn_xs_clear_waypoints = QPushButton("Clear waypoints (single segment)")
        self.btn_xs_clear_waypoints.clicked.connect(self.clear_xs_waypoints_action)
        xs_waypoint_row.addWidget(self.btn_xs_clear_waypoints)
        form.addRow(xs_waypoint_row)

        form.addRow(QLabel("<b>Search Width & Display</b> "
                           "<i>(see the popup "
                           "Cross-Section window for the result)</i>"))
        self.xs_search_width = spin(0.5, 500, 15.0, 1, "km")
        form.addRow("Search width (± perpendicular):", self.xs_search_width)

        self.xs_exaggeration = spin(0.1, 20.0, 1.0, 2, "x")
        form.addRow("Vertical exaggeration (main panel):", self.xs_exaggeration)

        self.xs_show_mesh = QCheckBox("Show ΔCFF color mesh")
        self.xs_show_mesh.setChecked(True)
        form.addRow(self.xs_show_mesh)

        self.xs_show_eq = QCheckBox("Show earthquake catalog")
        self.xs_show_eq.setChecked(False)
        form.addRow(self.xs_show_eq)

        self.xs_show_faults = QCheckBox("Show source fault traces")
        self.xs_show_faults.setChecked(True)
        form.addRow(self.xs_show_faults)

        self.xs_show_contours = QCheckBox("Show ΔCFF contours")
        self.xs_show_contours.setChecked(False)
        form.addRow(self.xs_show_contours)

        self.xs_show_focal_mechanisms = QCheckBox(
            "Show focal mechanisms (side view)")
        self.xs_show_focal_mechanisms.setChecked(False)
        form.addRow(self.xs_show_focal_mechanisms)
        self.xs_focal_search_width = spin(0.5, 500, 15.0, 1, "km")
        form.addRow("  Focal mechanism search width:", self.xs_focal_search_width)
        self.xs_focal_diameter = spin(0.05, 100, 3.0, 2, "km")
        form.addRow("  Focal mechanism symbol size:", self.xs_focal_diameter)

        xs_config_row = QHBoxLayout()
        self.btn_xs_configure_display = QPushButton("Configure display… (colors, sizes, legend)")
        self.btn_xs_configure_display.clicked.connect(self.open_xs_config_dialog_action)
        xs_config_row.addWidget(self.btn_xs_configure_display)
        form.addRow(xs_config_row)

        xs_eq_row = QHBoxLayout()
        self.btn_xs_import_eq = QPushButton("Import EQ catalog…")
        self.btn_xs_import_eq.clicked.connect(self.import_xs_eq_catalog_action)
        xs_eq_row.addWidget(self.btn_xs_import_eq)
        form.addRow(xs_eq_row)

        form.addRow(QLabel(
            "<b>Extra Depth-Section Elements</b> <i>(import "
            "other lines to draw in the depth section -- e.g. a "
            "subducting slab interface, another fault's known geometry, "
            "a seismic-reflector pick. Either a QGIS line layer with a "
            "depth/elevation field giving each vertex's depth in km, or "
            "a raster sampled along this profile -- e.g. a Slab2 slab-"
            "interface depth grid, which is commonly distributed as a "
            "raster rather than a digitized line.)</i>"))
        self.xs_extra_line_list = QListWidget()
        self.xs_extra_line_list.setMaximumHeight(70)
        form.addRow(self.xs_extra_line_list)
        xs_extra_line_row = QHBoxLayout()
        self.btn_xs_add_extra_line = QPushButton("+ QGIS line layer (with depth field)…")
        self.btn_xs_add_extra_line.clicked.connect(self.add_xs_extra_line_action)
        xs_extra_line_row.addWidget(self.btn_xs_add_extra_line)
        self.btn_xs_add_extra_line_raster_file = QPushButton("+ Raster file (e.g. Slab2)…")
        self.btn_xs_add_extra_line_raster_file.clicked.connect(
            self.add_xs_extra_line_raster_file_action)
        xs_extra_line_row.addWidget(self.btn_xs_add_extra_line_raster_file)
        self.btn_xs_add_extra_line_raster_layer = QPushButton("+ QGIS raster layer…")
        self.btn_xs_add_extra_line_raster_layer.clicked.connect(
            self.add_xs_extra_line_raster_qgis_layer_action)
        xs_extra_line_row.addWidget(self.btn_xs_add_extra_line_raster_layer)
        self.btn_xs_remove_extra_line = QPushButton("Remove selected")
        self.btn_xs_remove_extra_line.clicked.connect(self.remove_xs_extra_line_action)
        xs_extra_line_row.addWidget(self.btn_xs_remove_extra_line)
        form.addRow(xs_extra_line_row)

        form.addRow(QLabel("<b>Topographic Profile Panel(s)</b> "
                           "<i>(stacked above the main ΔCFF "
                           "section, one row per panel added below)</i>"))
        self.xs_topo_list = QListWidget()
        self.xs_topo_list.setMaximumHeight(90)
        form.addRow(self.xs_topo_list)
        xs_topo_row = QHBoxLayout()
        self.btn_xs_add_topo_file = QPushButton("+ Raster file…")
        self.btn_xs_add_topo_file.clicked.connect(self.add_xs_topo_panel_file_action)
        xs_topo_row.addWidget(self.btn_xs_add_topo_file)
        self.btn_xs_add_topo_layer = QPushButton("+ QGIS raster layer…")
        self.btn_xs_add_topo_layer.clicked.connect(self.add_xs_topo_panel_qgis_layer_action)
        xs_topo_row.addWidget(self.btn_xs_add_topo_layer)
        self.btn_xs_remove_topo = QPushButton("Remove selected")
        self.btn_xs_remove_topo.clicked.connect(self.remove_xs_topo_panel_action)
        xs_topo_row.addWidget(self.btn_xs_remove_topo)
        form.addRow(xs_topo_row)
        xs_topo_reorder_row = QHBoxLayout()
        self.btn_xs_move_topo_up = QPushButton("▲ Move up")
        self.btn_xs_move_topo_up.clicked.connect(self.move_xs_topo_panel_up_action)
        xs_topo_reorder_row.addWidget(self.btn_xs_move_topo_up)
        self.btn_xs_move_topo_down = QPushButton("▼ Move down")
        self.btn_xs_move_topo_down.clicked.connect(self.move_xs_topo_panel_down_action)
        xs_topo_reorder_row.addWidget(self.btn_xs_move_topo_down)
        form.addRow(xs_topo_reorder_row)

        form.addRow(QLabel("<b>Annotation Point(s)</b> "
                           "<i>(drawn on a topo panel, e.g. volcano/town "
                           "markers)</i>"))
        self.xs_annotation_list = QListWidget()
        self.xs_annotation_list.setMaximumHeight(90)
        form.addRow(self.xs_annotation_list)
        xs_ann_row = QHBoxLayout()
        self.btn_xs_add_ann_layer = QPushButton("+ QGIS point layer…")
        self.btn_xs_add_ann_layer.clicked.connect(self.add_xs_annotation_qgis_layer_action)
        xs_ann_row.addWidget(self.btn_xs_add_ann_layer)
        self.btn_xs_add_ann_file = QPushButton("+ CSV file…")
        self.btn_xs_add_ann_file.clicked.connect(self.add_xs_annotation_file_action)
        xs_ann_row.addWidget(self.btn_xs_add_ann_file)
        self.btn_xs_remove_ann = QPushButton("Remove selected")
        self.btn_xs_remove_ann.clicked.connect(self.remove_xs_annotation_action)
        xs_ann_row.addWidget(self.btn_xs_remove_ann)
        form.addRow(xs_ann_row)

        xs_export_row = QHBoxLayout()
        self.btn_xs_export = QPushButton("Export line + search-width polygon to QGIS")
        self.btn_xs_export.clicked.connect(self.export_cross_section_geometry_action)
        xs_export_row.addWidget(self.btn_xs_export)
        form.addRow(xs_export_row)

        self._refresh_xs_waypoint_list()
        self._refresh_xs_extra_line_list()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setFrameShape(QScrollArea.NoFrame)
        self.tabs.addTab(scroll, "Cross-Section")

    def _build_receiver_faults_tab(self):
        """
        Individual receiver faults: rows in the Source Faults table with
        ZERO slip are treated as receivers, each resolved on its OWN
        strike/dip/rake — matching Coulomb's "specified faults" receiver
        mode. Distinct from the Receiver Fault tab, which defines a
        single SHARED orientation applied everywhere on the CFF grid.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel(
            "<b>Individual Receiver Faults</b><br><i>Rows in the Source "
            "Faults table with ZERO slip are treated as receivers here — "
            "each resolved on its OWN strike/dip/rake at its own centroid "
            "position and depth, rather than a single shared orientation. "
            "Add/edit those rows in the Source Faults tab, then compute "
            "below.</i>"))

        self.receiver_results_widget = ReceiverResultsWidget()
        layout.addWidget(self.receiver_results_widget)

        btn_row = QHBoxLayout()
        self.btn_compute_receiver_faults = QPushButton(
            "▶  Compute ΔCFF on Receiver Faults")
        self.btn_compute_receiver_faults.clicked.connect(
            self.compute_receiver_faults_action)
        btn_row.addWidget(self.btn_compute_receiver_faults)

        self.btn_export_receiver_faults = QPushButton(
            "Export → Colored Receiver Fault Layer")
        self.btn_export_receiver_faults.clicked.connect(
            self.export_receiver_faults_layer)
        btn_row.addWidget(self.btn_export_receiver_faults)
        layout.addLayout(btn_row)

        self.tabs.addTab(tab, "Receiver Faults")

    def _build_optimal_tab(self):
        """
        Coulomb stress CHANGE on optimally-oriented planes (King, Stein &
        Lin 1994; Coulomb 3.x/4.0's "Opt Faults"/"3D OOP" feature) — see
        ../core/optimal_plane.py's module docstring and
        `optimal_plane_solution()`'s docstring for the full derivation
        and its verification (cross-checked against Coulomb 3.4.2's own
        `calcOptPlanes` source and against AutoCoulomb's independent,
        peer-reviewed `find_3D_OOP.m`, both confirmed 2026-08-11).

        Requires a REGIONAL/tectonic stress tensor in addition to the
        Source Faults table -- every other receiver mode in this plugin
        ignores regional stress; this is the one place it matters, since
        the optimal orientation depends on the TOTAL (regional +
        coseismic) stress state even though the reported ΔCFF is the
        coseismic-only change.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel(
            "<b>Optimally-Oriented Planes</b><br><i>At every grid point, "
            "finds the (strike, dip, rake) that maximizes CFF given the "
            "TOTAL (regional + coseismic) stress, then reports the "
            "COSEISMIC-ONLY Coulomb stress CHANGE on that plane — "
            "matching Coulomb 3.4.2's own 'Opt Faults' output, not the "
            "total post-earthquake stress state. Uses the Source Faults "
            "table (Source Faults tab) and the regional stress specified "
            "below.</i>"))

        form = QFormLayout()
        layout.addLayout(form)

        form.addRow(QLabel("<hr><b>Regional (Tectonic) Stress</b>"))
        self.rs_s1 = spin(-10000, 10000, 100.0, 2, "bar")
        self.rs_s2 = spin(-10000, 10000, 0.0, 2, "bar")
        self.rs_s3 = spin(-10000, 10000, 0.0, 2, "bar")
        form.addRow("S1 (compression +):", self.rs_s1)
        form.addRow("S2 (compression +):", self.rs_s2)
        form.addRow("S3 (compression +):", self.rs_s3)
        form.addRow(QLabel(
            "<i>Coulomb's own convention: compression-positive, bars — "
            "opposite sign from every ΔCFF value elsewhere in this "
            "plugin (tension-positive); converted internally, see "
            "optimal_plane.py's module docstring.</i>"))

        form.addRow(QLabel("<hr><b>Principal Axis Orientation</b>"))
        self.rs_s1_strike = spin(0, 360, 0.0, 1, "°")
        self.rs_s1_plunge = spin(-90, 90, 0.0, 1, "°")
        self.rs_s2_strike = spin(0, 360, 90.0, 1, "°")
        self.rs_s2_plunge = spin(-90, 90, 0.0, 1, "°")
        form.addRow("S1 strike:", self.rs_s1_strike)
        form.addRow("S1 plunge:", self.rs_s1_plunge)
        form.addRow("S2 strike:", self.rs_s2_strike)
        form.addRow("S2 plunge:", self.rs_s2_plunge)
        form.addRow(QLabel(
            "<i>S3 is derived automatically as orthogonal to both (right-"
            "handed). S1/S2 should already be close to perpendicular — "
            "an orthogonality warning is shown after computing if they "
            "aren't.</i>"))

        form.addRow(QLabel("<hr><b>Friction</b>"))
        self.rs_friction = spin(0.0, 1.0, self.e_friction.value() if hasattr(self, 'e_friction') else 0.4, 2)
        form.addRow("Friction coefficient (override):", self.rs_friction)
        form.addRow(QLabel(
            "<i>Independent of the Elastic Params tab's friction — set "
            "once here; defaults to that tab's value at the time this "
            "tab is built.</i>"))

        form.addRow(QLabel("<hr><b>From Focal Mechanisms (optional)</b>"))
        self.btn_invert_stress = QPushButton(
            "Invert Regional Stress from Focal Mechanisms…")
        self.btn_invert_stress.clicked.connect(self.invert_regional_stress_action)
        form.addRow(self.btn_invert_stress)
        form.addRow(QLabel(
            "<i>Uses the catalog imported in the Focal Mechanisms tab "
            "(ILSI, Beaucé et al. 2022) to fill in the strike/plunge "
            "fields above from real data instead of a hand-entered "
            "guess. Still requires you to independently supply a "
            "differential-stress magnitude in that dialog — focal "
            "mechanisms alone can't determine it, see the dialog for "
            "why.</i>"))

        self.lbl_orthogonality = QLabel()
        self.lbl_orthogonality.setWordWrap(True)
        layout.addWidget(self.lbl_orthogonality)

        self.tabs.addTab(tab, "Optimal Faults")

    def _update_depth_status(self):
        if _has_okada_wrapper():
            self.lbl_depth_status.setText(
                "✅ External Python with okada-wrapper configured — exact depth CFF "
                "and deformation enabled (Okada 1992 DC3D)")
            self.lbl_depth_status.setStyleSheet("color: darkgreen;")
        else:
            self.lbl_depth_status.setText(
                "⚠️ No external Python with okada-wrapper configured — computations "
                "at depth fall back to the surface (z=0) formula. Sign pattern is "
                "correct at all depths; exact magnitudes at depth require an "
                "external Python. Plugins → Coulomb → Check / Configure "
                "Depth-Dependent CFF.")
            self.lbl_depth_status.setStyleSheet("color: darkorange;")

    # ── Parameter collection ────────────────────────────────────────────

    def _get_grid(self):
        lon_min, lon_max = self.g_lon_min.value(), self.g_lon_max.value()
        lat_min, lat_max = self.g_lat_min.value(), self.g_lat_max.value()

        mode = self.g_res_mode.currentText()
        if mode == "Grid point count":
            n_lon, n_lat = int(self.g_n_lon.value()), int(self.g_n_lat.value())
        else:
            units = "km" if mode == "Spacing (km)" else "deg"
            n_lon, n_lat, _clamped = grid_counts_from_spacing(
                lon_min, lon_max, lat_min, lat_max,
                self.g_spacing.value(), units=units,
                max_points_per_axis=int(self.g_max_points_per_axis.value()))

        return GridParameters(
            lon_min=lon_min, lon_max=lon_max,
            lat_min=lat_min, lat_max=lat_max,
            depth_km=self.g_depth.value(),
            n_lon=n_lon, n_lat=n_lat,
        )

    def _get_receiver(self):
        return FaultParameters(
            lon=0, lat=0, depth=self.g_depth.value(), length=1, width=1, slip=0,
            strike=self.r_strike.value(), dip=self.r_dip.value(), rake=self.r_rake.value(),
        )

    def _get_elastic(self):
        return ElasticParameters(
            mu=self.e_mu.value(), nu=self.e_nu.value(), friction=self.e_friction.value(),
        )

    def _get_regional(self):
        return RegionalStress(
            S1=self.rs_s1.value(), S2=self.rs_s2.value(), S3=self.rs_s3.value(),
            S1_strike=self.rs_s1_strike.value(), S1_plunge=self.rs_s1_plunge.value(),
            S2_strike=self.rs_s2_strike.value(), S2_plunge=self.rs_s2_plunge.value(),
        )

    def _get_sources(self):
        """Only rows with nonzero slip act as stress sources; rows with
        zero slip are receiver-only and handled separately via
        compute_cff_on_receiver_faults() (see the Receiver Faults tab)."""
        return self.fault_table.get_sources()

    def _get_table_receiver_faults(self):
        """Zero-slip rows in the fault table, each resolved on its OWN
        strike/dip/rake — the 'individual receiver faults' feature,
        distinct from the single shared receiver orientation used by
        the CFF grid/cross-section (Receiver Fault tab)."""
        return self.fault_table.get_receiver_faults()

    def _get_cross_section_params(self):
        vertices = ([(self.xs_lon1.value(), self.xs_lat1.value())]
                   + list(self.xs_waypoints)
                   + [(self.xs_lon2.value(), self.xs_lat2.value())])
        return dict(
            lon1=self.xs_lon1.value(), lat1=self.xs_lat1.value(),
            lon2=self.xs_lon2.value(), lat2=self.xs_lat2.value(),
            dist_increment_km=self.xs_dist_inc.value(),
            max_depth_km=self.xs_max_depth.value(),
            depth_increment_km=self.xs_depth_inc.value(),
            cff_source=("optimal" if self.xs_cff_source.currentIndex() == 1 else "receiver"),
            vertices=vertices,
        )

    def _refresh_xs_waypoint_list(self):
        self.xs_waypoint_list.clear()
        if not self.xs_waypoints:
            self.xs_waypoint_list.addItem("(none -- single straight segment)")
            return
        for i, (lon, lat) in enumerate(self.xs_waypoints):
            self.xs_waypoint_list.addItem(f"[{i}] {lon:.4f}, {lat:.4f}")

    def import_xs_profile_polyline_action(self):
        """
        2026-08-21 addition (request item 5): "option cross section
        profile line can be imported as a qgis polyline". Reads the
        first (selected, if any are selected; else first overall)
        feature of a chosen line layer's full vertex chain -- see
        utils.polyline_import.profile_vertices_from_line_layer() --
        and splits it into Start (first vertex) / Finish (last vertex)
        / waypoints (everything in between), so an existing bent/
        multi-part digitized line becomes a ready-to-compute
        multi-segment profile in one step.
        """
        from qgis.core import QgsProject, QgsMapLayer, QgsWkbTypes
        from ..utils.polyline_import import profile_vertices_from_line_layer

        layers = [lyr for lyr in QgsProject.instance().mapLayers().values()
                 if lyr.type() == QgsMapLayer.VectorLayer
                 and lyr.geometryType() == QgsWkbTypes.LineGeometry]
        if not layers:
            self.status_label.setText(
                "No line layers loaded in this QGIS project. Digitize the "
                "profile as a line layer first.")
            return
        names = [lyr.name() for lyr in layers]
        name, ok = QInputDialog.getItem(
            self, "Select profile line layer", "Line layer:", names, 0, False)
        if not ok:
            return
        layer = layers[names.index(name)]

        only_selected = layer.selectedFeatureCount() > 0
        vertices = profile_vertices_from_line_layer(layer, only_selected=only_selected)
        if vertices is None:
            self.status_label.setText(
                f"No usable line feature found in '{name}'.")
            return

        self.xs_lon1.setValue(vertices[0][0])
        self.xs_lat1.setValue(vertices[0][1])
        self.xs_lon2.setValue(vertices[-1][0])
        self.xs_lat2.setValue(vertices[-1][1])
        self.xs_waypoints = list(vertices[1:-1])
        self._refresh_xs_waypoint_list()
        n_seg = len(vertices) - 1
        self.status_label.setText(
            f"Profile imported from '{name}': {n_seg} segment(s), "
            f"{len(vertices)} vertices.")

    def add_xs_waypoint_action(self):
        """
        2026-08-22 addition: the Multi-Segment Profile label has always
        described waypoints as insertable directly ("optional waypoints
        inserted between Start and Finish above"), but until now the
        only ways to populate self.xs_waypoints were the polyline
        importer or clear-all -- there was no manual entry point. This
        prompts for one lon/lat pair with QInputDialog.getDouble() (same
        pattern already used for other numeric prompts in this file) and
        appends it to the end of the waypoint list, i.e. it becomes the
        new second-to-last vertex, just before Finish. To insert a
        waypoint in the MIDDLE of an existing multi-waypoint list, add it
        (it lands at the end) then use Remove selected / re-add as
        needed, or re-import from a corrected polyline.
        """
        lon, ok = QInputDialog.getDouble(
            self, "Add waypoint", "Longitude (°):",
            self.xs_lon2.value(), -180.0, 180.0, 4)
        if not ok:
            return
        lat, ok = QInputDialog.getDouble(
            self, "Add waypoint", "Latitude (°):",
            self.xs_lat2.value(), -90.0, 90.0, 4)
        if not ok:
            return
        self.xs_waypoints.append((lon, lat))
        self._refresh_xs_waypoint_list()

    def remove_xs_waypoint_action(self):
        """Remove the currently-selected row from the waypoint list."""
        row = self.xs_waypoint_list.currentRow()
        if row < 0 or row >= len(self.xs_waypoints):
            self.status_label.setText(
                "Select a waypoint in the list first, then click "
                "'Remove selected'.")
            return
        del self.xs_waypoints[row]
        self._refresh_xs_waypoint_list()

    def clear_xs_waypoints_action(self):
        self.xs_waypoints = []
        self._refresh_xs_waypoint_list()

    def _refresh_xs_extra_line_list(self):
        self.xs_extra_line_list.clear()
        for i, line_cfg in enumerate(self.xs_config.extra_lines):
            if line_cfg.source_kind == "vector_line":
                detail = f"{len(line_cfg.vertices)} vertices"
            else:
                kind_label = ("raster file" if line_cfg.source_kind == "raster_file"
                              else "QGIS raster layer")
                detail = f"{kind_label}, resampled along profile"
            self.xs_extra_line_list.addItem(
                f"[{i}] {line_cfg.label or '(unlabeled)'} ({detail})")

    def add_xs_extra_line_action(self):
        """
        2026-08-21 addition (request item 7): import an arbitrary
        depth-section element -- e.g. a digitized subducting slab
        interface, or another fault's known geometry from a published
        source -- from a QGIS line layer that carries each vertex's
        depth in an attribute field (depth positive-down, km; a
        negative/elevation field also works, just check "field is
        elevation" below to have the sign flipped for you).
        """
        from qgis.core import QgsProject, QgsMapLayer, QgsWkbTypes

        layers = [lyr for lyr in QgsProject.instance().mapLayers().values()
                 if lyr.type() == QgsMapLayer.VectorLayer
                 and lyr.geometryType() == QgsWkbTypes.LineGeometry]
        if not layers:
            self.status_label.setText("No line layers loaded in this QGIS project.")
            return
        names = [lyr.name() for lyr in layers]
        name, ok = QInputDialog.getItem(
            self, "Select line layer", "Line layer (with a depth field):",
            names, 0, False)
        if not ok:
            return
        layer = layers[names.index(name)]

        field_names = [f.name() for f in layer.fields()]
        if not field_names:
            self.status_label.setText(f"'{name}' has no attribute fields to use as depth.")
            return
        depth_field, ok = QInputDialog.getItem(
            self, "Depth field", "Field giving each vertex's depth (km):",
            field_names, 0, False)
        if not ok:
            return
        is_elevation, ok = QInputDialog.getItem(
            self, "Sign convention", "Is this field positive UP (elevation) "
            "or positive DOWN (depth)?", ["Positive down (depth)", "Positive up (elevation)"],
            0, False)
        if not ok:
            return
        sign = -1.0 if is_elevation.startswith("Positive up") else 1.0

        vertices = []
        features = layer.selectedFeatures() if layer.selectedFeatureCount() > 0 \
            else layer.getFeatures()
        for feat in features:
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            depth_val = feat[depth_field]
            if depth_val is None:
                continue
            depth_km = sign * float(depth_val)
            parts = geom.asMultiPolyline() if geom.isMultipart() else [geom.asPolyline()]
            for part in parts:
                for pt in part:
                    vertices.append((pt.x(), pt.y(), depth_km))

        if len(vertices) < 2:
            self.status_label.setText(
                f"No usable vertices found in '{name}' (need a valid "
                f"numeric '{depth_field}' value per feature).")
            return

        label, ok = QInputDialog.getText(self, "Label", "Label for this element:", text=name)
        if not ok:
            label = name

        self.xs_config.extra_lines.append(ExtraSectionLineConfig(
            label=label, source_kind="vector_line", vertices=vertices))
        self._refresh_xs_extra_line_list()
        self.status_label.setText(
            f"Added depth-section element '{label}' ({len(vertices)} vertices).")

    def _prompt_extra_line_raster_settings(self, default_label):
        """
        Shared prompts for a raster-sourced depth-section element (band,
        label, unit conversion, sign convention) -- used by both the
        raster-file and QGIS-raster-layer add actions below, the same
        way add_xs_extra_line_action() prompts for a depth field/sign
        convention on the vector-line path. Subducting-slab-interface
        rasters (e.g. Slab2) are commonly already in km, but a raster
        depth/elevation surface in metres (a DEM/bathymetry grid used
        as a horizon pick) is just as plausible, hence asking rather
        than assuming.

        Returns (label, band, unit_divisor, sign) or None if cancelled.
        """
        label, ok = QInputDialog.getText(
            self, "Label", "Label for this element:", text=default_label)
        if not ok:
            return None

        band, ok = QInputDialog.getInt(
            self, "Band", "Raster band to sample:", 1, 1, 999)
        if not ok:
            return None

        unit_choice, ok = QInputDialog.getItem(
            self, "Units", "Raster value units:",
            ["Kilometres (e.g. most Slab2 grids)", "Metres"], 0, False)
        if not ok:
            return None
        unit_divisor = 1.0 if unit_choice.startswith("Kilometres") else 1000.0

        sign_choice, ok = QInputDialog.getItem(
            self, "Sign convention",
            "Is this raster positive DOWN (depth) or positive UP "
            "(elevation)? Many Slab2 grids store NEGATIVE-down depth, "
            "which also needs flipping -- choose 'positive up' for "
            "that case too.",
            ["Positive down (depth)", "Positive up (elevation, or "
             "negative-down depth)"], 0, False)
        if not ok:
            return None
        sign = 1.0 if sign_choice.startswith("Positive down") else -1.0

        return label, band, unit_divisor, sign

    def add_xs_extra_line_raster_file_action(self):
        """
        Sample a raster FILE (rasterio-backed) along the current cross-
        section profile as an Extra Depth-Section Element -- e.g. a
        Slab2 slab-interface depth grid, or any other subducting-slab/
        seismic-horizon dataset distributed as a raster rather than a
        digitized line. Resampled fresh from `raster_source` every time
        the section is computed (core.raster_profile.
        sample_raster_along_polyline()), so it tracks edits to the
        profile's start/finish/waypoints automatically, unlike the
        fixed vertices imported by add_xs_extra_line_action() above.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Select raster file for depth-section element", "",
            "Rasters (*.tif *.tiff *.asc *.grd *.nc);;All files (*)")
        if not path:
            return
        settings = self._prompt_extra_line_raster_settings(
            os.path.splitext(os.path.basename(path))[0])
        if settings is None:
            return
        label, band, unit_divisor, sign = settings

        self.xs_config.extra_lines.append(ExtraSectionLineConfig(
            label=label, source_kind="raster_file", raster_source=path,
            raster_band=band, raster_unit_divisor=unit_divisor,
            raster_sign=sign))
        self._refresh_xs_extra_line_list()
        self.status_label.setText(
            f"Added raster depth-section element '{label}' ({os.path.basename(path)}).")

    def add_xs_extra_line_raster_qgis_layer_action(self):
        """QGIS-layer counterpart of add_xs_extra_line_raster_file_action()
        above -- needs no rasterio install, same trade-off as the topo
        panel's own file-vs-QGIS-layer raster source choice."""
        from qgis.core import QgsProject, QgsMapLayer

        layers = [lyr for lyr in QgsProject.instance().mapLayers().values()
                 if lyr.type() == QgsMapLayer.RasterLayer]
        if not layers:
            self.status_label.setText("No raster layers loaded in this QGIS project.")
            return
        names = [lyr.name() for lyr in layers]
        name, ok = QInputDialog.getItem(
            self, "Select raster layer", "Raster layer:", names, 0, False)
        if not ok:
            return
        layer = layers[names.index(name)]

        settings = self._prompt_extra_line_raster_settings(name)
        if settings is None:
            return
        label, band, unit_divisor, sign = settings

        # Store the layer ID, not the layer object itself -- resolved
        # back to the live QgsRasterLayer at gather time, same
        # "loaded layers can change between adding and recomputing"
        # reasoning as the topo panel's own QGIS-layer source.
        self.xs_config.extra_lines.append(ExtraSectionLineConfig(
            label=label, source_kind="qgis_layer", raster_source=layer.id(),
            raster_band=band, raster_unit_divisor=unit_divisor,
            raster_sign=sign))
        self._refresh_xs_extra_line_list()
        self.status_label.setText(f"Added raster depth-section element '{label}' ({name}).")

    def remove_xs_extra_line_action(self):
        row = self.xs_extra_line_list.currentRow()
        if row < 0 or row >= len(self.xs_config.extra_lines):
            return
        del self.xs_config.extra_lines[row]
        self._refresh_xs_extra_line_list()

    # ── Actions ──────────────────────────────────────────────────────────

    def compute_cff(self):
        sources = self._get_sources()
        if not sources:
            self.status_label.setText("Add at least one source fault first.")
            return
        self._run_worker("cff", sources, self._get_receiver(), self._get_elastic(),
                         grid=self._get_grid())

    def compute_displacement(self):
        sources = self._get_sources()
        if not sources:
            self.status_label.setText("Add at least one source fault first.")
            return
        self._run_worker("displacement", sources, self._get_receiver(),
                         self._get_elastic(), grid=self._get_grid())

    def show_cross_section_window_action(self):
        """
        Re-show the existing cross-section window without recomputing
        anything (2026-08-20 follow-up to item 13). `self._xs_window`
        holds its previously-built figure until the next recompute
        overwrites it (CrossSectionWindow reuses the same canvas/figure
        rather than rebuilding), so this is safe to call any time after
        at least one successful "Compute Cross-Section" -- it just calls
        show_and_raise() on what's already there.
        """
        if self._xs_window is None:
            self.status_label.setText(
                "No cross-section has been computed yet in this session -- "
                "click \u25b6 Compute Cross-Section first.")
            return
        self._xs_window.show_and_raise()

    def compute_cross_section_action(self):
        sources = self._get_sources()
        if not sources:
            self.status_label.setText("Add at least one source fault first.")
            return
        self._run_worker("cross_section", sources, self._get_receiver(),
                         self._get_elastic(),
                         cross_section_params=self._get_cross_section_params(),
                         regional=self._get_regional(),
                         opt_friction=self.rs_friction.value())

    def open_point_calculator_action(self):
        """
        2026-09-01 addition: open the Point Calculator dialog (predicted
        stress/displacement, and optional field-measurement validation,
        at arbitrary imported points). Non-modal-in-spirit but run with
        exec_() like every other tool dialog here (EQCatalogImportDialog,
        FocalMechanismImportDialog, etc.) -- the dialog does its own
        compute/export internally and doesn't need to hand anything back
        to main_dialog on close, unlike those two.
        """
        dlg = PointCalculatorDialog(
            self, get_sources=self._get_sources,
            get_receiver=self._get_receiver, get_elastic=self._get_elastic)
        dlg.exec_()

    def import_xs_eq_catalog_action(self):
        """
        Import an earthquake catalog for the cross-section's EQ overlay
        (2026-08-18b, point 1). Same EQCatalogImportDialog/EQCatalogEvent
        pipeline as the Aftershock/ΔCFF Monte Carlo test dialog uses --
        this dialog keeps its own self.eq_events rather than sharing that
        (modal, transient) dialog's copy, since there's no persistent
        handle back into it once it closes.
        """
        dlg = EQCatalogImportDialog(self)
        if dlg.exec_():
            self.eq_events = dlg.imported_events
            n = len(self.eq_events)
            self.status_label.setText(f"{n} earthquake(s) loaded for the cross-section overlay.")

    # ── Topo panel / annotation picker UI (2026-08-19, Phase 2 UI) ─────
    #
    # Deliberately minimal (add/remove + the fields needed to actually
    # locate and sample the source), NOT the full "colors, symbols,
    # size, placement" picker for every field TopoPanelConfig/
    # AnnotationOverlayConfig support -- that's the addendum's Phase 3
    # ("full symbology configuration dialog"). Editing an already-added
    # panel's cosmetic fields is Phase 3's job; this session's job was
    # getting a source onto the list at all.

    def _refresh_xs_topo_list(self):
        self.xs_topo_list.clear()
        for i, panel in enumerate(self.xs_topo_panels):
            self.xs_topo_list.addItem(
                f"[{i}] {panel.label} ({panel.source_kind}: {panel.source})")

    def _refresh_xs_annotation_list(self):
        self.xs_annotation_list.clear()
        for i, ann in enumerate(self.xs_annotations):
            self.xs_annotation_list.addItem(
                f"[{i}] {ann.label} ({ann.source_kind}: {ann.source}, "
                f"field={ann.label_field}, panel={ann.topo_panel_index})")

    def add_xs_topo_panel_file_action(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select raster file for topo panel", "",
            "Rasters (*.tif *.tiff *.asc *.grd *.nc);;All files (*)")
        if not path:
            return
        label, ok = QInputDialog.getText(
            self, "Topo panel label", "Panel label (y-axis title):",
            text="Elevation (km)")
        if not ok:
            return
        self.xs_topo_panels.append(
            TopoPanelConfig(label=label, source_kind="raster_file", source=path))
        self._refresh_xs_topo_list()
        self.status_label.setText(f"Topo panel added: {os.path.basename(path)}")

    def add_xs_topo_panel_qgis_layer_action(self):
        from qgis.core import QgsProject, QgsMapLayer

        layers = [lyr for lyr in QgsProject.instance().mapLayers().values()
                 if lyr.type() == QgsMapLayer.RasterLayer]
        if not layers:
            self.status_label.setText("No raster layers loaded in this QGIS project.")
            return
        names = [lyr.name() for lyr in layers]
        name, ok = QInputDialog.getItem(
            self, "Select raster layer", "Raster layer:", names, 0, False)
        if not ok:
            return
        layer = layers[names.index(name)]
        label, ok = QInputDialog.getText(
            self, "Topo panel label", "Panel label (y-axis title):",
            text=name)
        if not ok:
            return
        # Store the layer ID (a stable string), not the layer object
        # itself -- resolved back to the live QgsRasterLayer at gather
        # time via QgsProject.instance().mapLayer(id), same pattern as
        # core.raster_profile's own "already-resolved layer object"
        # contract, just deferred until the layer is actually needed
        # (the project's loaded layers can change between adding this
        # panel and recomputing the cross-section).
        self.xs_topo_panels.append(
            TopoPanelConfig(label=label, source_kind="qgis_layer", source=layer.id()))
        self._refresh_xs_topo_list()
        self.status_label.setText(f"Topo panel added: {name}")

    def remove_xs_topo_panel_action(self):
        row = self.xs_topo_list.currentRow()
        if row < 0 or row >= len(self.xs_topo_panels):
            return
        del self.xs_topo_panels[row]
        # Keep annotation.topo_panel_index pointing at the same
        # conceptual panel after a removal shifts every later panel's
        # position down by one (2026-08-19, same fix class as the
        # move-up/down remap below). An annotation that was pointing
        # AT the removed panel has nothing correct left to point to --
        # reset it to panel 0 rather than leaving a stale/out-of-range
        # index, and say so, since silently reassigning someone's
        # annotations to a different panel is exactly the kind of
        # thing that should be visible, not quiet.
        orphaned = 0
        for ann in self.xs_annotations:
            if ann.topo_panel_index == row:
                ann.topo_panel_index = 0
                orphaned += 1
            elif ann.topo_panel_index > row:
                ann.topo_panel_index -= 1
        self._refresh_xs_topo_list()
        if orphaned:
            self.status_label.setText(
                f"Removed topo panel; {orphaned} annotation source(s) that were "
                "pointing at it were reset to panel 0 -- reassign them via "
                "'Configure display…' if that's not what you want.")

    def _remap_annotation_topo_panel_indices_on_swap(self, i, j):
        """
        Keep AnnotationOverlayConfig.topo_panel_index pointing at the
        SAME conceptual panel across a topo-panel reorder (2026-08-19).
        Swapping self.xs_topo_panels[i] <-> [j] means any annotation
        previously pointing at i must now point at j and vice versa --
        otherwise "draw these annotations on the DEM panel" would
        silently start drawing them on whatever panel happens to land
        in that same list position after the reorder.
        """
        for ann in self.xs_annotations:
            if ann.topo_panel_index == i:
                ann.topo_panel_index = j
            elif ann.topo_panel_index == j:
                ann.topo_panel_index = i

    def move_xs_topo_panel_up_action(self):
        row = self.xs_topo_list.currentRow()
        if row <= 0 or row >= len(self.xs_topo_panels):
            return
        self.xs_topo_panels[row - 1], self.xs_topo_panels[row] = (
            self.xs_topo_panels[row], self.xs_topo_panels[row - 1])
        self._remap_annotation_topo_panel_indices_on_swap(row - 1, row)
        self._refresh_xs_topo_list()
        self.xs_topo_list.setCurrentRow(row - 1)

    def move_xs_topo_panel_down_action(self):
        row = self.xs_topo_list.currentRow()
        if row < 0 or row >= len(self.xs_topo_panels) - 1:
            return
        self.xs_topo_panels[row], self.xs_topo_panels[row + 1] = (
            self.xs_topo_panels[row + 1], self.xs_topo_panels[row])
        self._remap_annotation_topo_panel_indices_on_swap(row, row + 1)
        self._refresh_xs_topo_list()
        self.xs_topo_list.setCurrentRow(row + 1)

    def add_xs_annotation_qgis_layer_action(self):
        from qgis.core import QgsProject, QgsMapLayer

        layers = [lyr for lyr in QgsProject.instance().mapLayers().values()
                 if lyr.type() == QgsMapLayer.VectorLayer]
        if not layers:
            self.status_label.setText("No vector layers loaded in this QGIS project.")
            return
        names = [lyr.name() for lyr in layers]
        name, ok = QInputDialog.getItem(
            self, "Select point layer", "Vector layer:", names, 0, False)
        if not ok:
            return
        layer = layers[names.index(name)]
        field_names = [f.name() for f in layer.fields()]
        label_field = None
        if field_names:
            label_field, ok = QInputDialog.getItem(
                self, "Label field", "Field to use as the annotation label:",
                field_names, 0, False)
            if not ok:
                return
        panel_idx = 0
        if len(self.xs_topo_panels) > 1:
            panel_choices = [f"[{i}] {p.label}" for i, p in enumerate(self.xs_topo_panels)]
            choice, ok = QInputDialog.getItem(
                self, "Topo panel", "Draw annotations on which topo panel?",
                panel_choices, 0, False)
            if not ok:
                return
            panel_idx = panel_choices.index(choice)
        self.xs_annotations.append(AnnotationOverlayConfig(
            enabled=True, label=name, source_kind="qgis_layer", source=layer.id(),
            label_field=label_field, topo_panel_index=panel_idx))
        self._refresh_xs_annotation_list()
        self.status_label.setText(f"Annotation source added: {name}")

    def add_xs_annotation_file_action(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV of annotation points", "", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        label_field, ok = QInputDialog.getText(
            self, "Label column", "CSV column name to use as the annotation label:",
            text="label")
        if not ok:
            return
        panel_idx = 0
        if len(self.xs_topo_panels) > 1:
            panel_choices = [f"[{i}] {p.label}" for i, p in enumerate(self.xs_topo_panels)]
            choice, ok = QInputDialog.getItem(
                self, "Topo panel", "Draw annotations on which topo panel?",
                panel_choices, 0, False)
            if not ok:
                return
            panel_idx = panel_choices.index(choice)
        self.xs_annotations.append(AnnotationOverlayConfig(
            enabled=True, label=os.path.basename(path), source_kind="file", source=path,
            label_field=label_field, topo_panel_index=panel_idx))
        self._refresh_xs_annotation_list()
        self.status_label.setText(f"Annotation source added: {os.path.basename(path)}")

    def remove_xs_annotation_action(self):
        row = self.xs_annotation_list.currentRow()
        if row < 0 or row >= len(self.xs_annotations):
            return
        del self.xs_annotations[row]
        self._refresh_xs_annotation_list()

    def open_xs_config_dialog_action(self):
        """
        Full symbology configuration dialog (Phase 3, 2026-08-19):
        colors/sizes/cmaps/z-order for every overlay, topo-panel and
        annotation-source cosmetic fields, and legend placement.
        Deliberately separate from the quick checkboxes/spins already
        on this tab (show/hide + search width + vertical exaggeration),
        which remain the fast path and take precedence at compute time
        -- see the cross_section branch of _on_finished(). This dialog
        is for everything those quick controls don't expose.
        """
        dlg = CrossSectionConfigDialog(self, self.xs_config)
        dlg.exec_()

    def export_cross_section_geometry_action(self):
        """Export the cross-section profile line and its search-width
        swath polygon to QGIS (2026-08-18b, point 10)."""
        p = self._get_cross_section_params()
        vector_utils.create_cross_section_line_layer(
            p['lon1'], p['lat1'], p['lon2'], p['lat2'])
        vector_utils.create_cross_section_search_width_layer(
            p['lon1'], p['lat1'], p['lon2'], p['lat2'],
            self.xs_search_width.value())
        self.status_label.setText("Cross-section line and search-width "
                                   "polygon added to QGIS.")

    def compute_receiver_faults_action(self):
        sources = self._get_sources()
        receiver_faults = self._get_table_receiver_faults()
        if not sources:
            self.status_label.setText("Add at least one source fault (nonzero slip) first.")
            return
        if not receiver_faults:
            self.status_label.setText(
                "No receiver faults found. Add a row with ZERO slip in the "
                "Source Faults tab to use as an individual receiver.")
            return
        self._run_worker("receiver_faults", sources, self._get_receiver(),
                         self._get_elastic(), receiver_faults=receiver_faults)

    def compute_optimal_action(self):
        sources = self._get_sources()
        if not sources:
            self.status_label.setText("Add at least one source fault first.")
            return
        self._run_worker("optimal", sources, self._get_receiver(),
                         self._get_elastic(), grid=self._get_grid(),
                         regional=self._get_regional(),
                         opt_friction=self.rs_friction.value())

    def open_aftershock_mc_test_action(self):
        """
        Launch the Aftershock/ΔCFF Monte Carlo null test dialog
        (ui.aftershock_mc_dialog.AftershockMCTestDialog) with the
        current source/receiver/elastic/grid configuration -- same
        four objects _run_worker() already gathers for the "cff" mode,
        just handed to a separate modal dialog instead of the shared
        background ComputeWorker, since that dialog runs its own
        AftershockMCWorker(QThread) internally (building a multi-depth
        CFF volume + Monte Carlo sampling is a different, longer-running
        job than any single-grid computation this dialog's own
        ComputeWorker modes handle).

        grid's own .depth_km is irrelevant to the null test (only its
        lon/lat extent + resolution matter, and resolution gets capped
        further inside the dialog -- see
        aftershock_mc_dialog._MAX_VOLUME_GRID_POINTS_PER_AXIS) so
        reusing _get_grid() as-is is correct even though that grid was
        really configured with the raster-preview use case in mind.
        Constructed dialog is given this dialog's own current
        RegionalStress + friction (same values `compute_optimal_action()`
        already gathers via `_get_regional()`/`self.rs_friction`), so its
        "optimally-oriented fault" ΔCFF option is available -- that
        option is disabled inside the dialog itself if no regional
        stress has been configured (nothing enforced here).
        """
        sources = self._get_sources()
        if not sources:
            self.status_label.setText("Add at least one source fault first.")
            return
        dlg = AftershockMCTestDialog(sources, self._get_receiver(),
                                     self._get_elastic(), self._get_grid(),
                                     regional=self._get_regional(),
                                     friction=self.rs_friction.value(), parent=self)
        dlg.apply_settings(self._aftershock_dialog_settings)
        dlg.exec_()
        self._aftershock_dialog_settings = dlg.get_settings()

    def open_rate_state_forecast_action(self):
        """
        Launch the Rate-and-State Seismicity Forecast dialog
        (ui.rate_state_dialog.RateStateForecastDialog), mirroring
        open_aftershock_mc_test_action() above exactly -- same four
        objects (sources/receiver/elastic/grid) and the same
        regional/friction pass-through for the optimally-oriented-plane
        ΔCFF option, since this dialog is built the same way (own
        internal QThread worker for the CFF-volume-building step, not
        this dialog's shared ComputeWorker).
        """
        sources = self._get_sources()
        if not sources:
            self.status_label.setText("Add at least one source fault first.")
            return
        dlg = RateStateForecastDialog(sources, self._get_receiver(),
                                      self._get_elastic(), self._get_grid(),
                                      regional=self._get_regional(),
                                      friction=self.rs_friction.value(), parent=self)
        dlg.apply_settings(self._rate_state_dialog_settings)
        dlg.exec_()
        self._rate_state_dialog_settings = dlg.get_settings()

    def import_focal_mechanisms_action(self):
        dlg = FocalMechanismImportDialog(self)
        if dlg.exec_() == dlg.Accepted:
            self.focal_events = dlg.imported_events
            self.lbl_focal_status.setText(
                f"{len(self.focal_events)} focal mechanism(s) loaded.")
            # Populate the table right away (ΔCFF columns show "—" until
            # compute_focal_mechanisms_action() runs) so "Use as Source"/
            # "Source Plane" are available immediately -- designating a
            # source fault doesn't require computing ΔCFF on these events
            # as receivers first.
            self.focal_results_widget.set_events(self.focal_events)

    def invert_regional_stress_action(self):
        """
        Opens StressInversionDialog against whatever's currently in
        self.focal_events (Focal Mechanisms tab's imported catalog --
        NOT re-imported here, see that dialog's own docstring for why).
        On acceptance, pushes the resulting RegionalStress orientation
        (and ILSI's own used/found friction coefficient) into this tab's
        S1/S2/S3, strike/plunge, and friction-override fields, exactly
        as if the user had typed them in by hand.
        """
        dlg = StressInversionDialog(
            getattr(self, "focal_events", None) or [],
            default_friction=self.rs_friction.value(), parent=self)
        if dlg.exec_() == dlg.Accepted:
            regional, friction = dlg.get_result()
            if regional is not None:
                self.rs_s1.setValue(regional.S1)
                self.rs_s2.setValue(regional.S2)
                self.rs_s3.setValue(regional.S3)
                self.rs_s1_strike.setValue(regional.S1_strike)
                self.rs_s1_plunge.setValue(regional.S1_plunge)
                self.rs_s2_strike.setValue(regional.S2_strike)
                self.rs_s2_plunge.setValue(regional.S2_plunge)
                if friction is not None:
                    self.rs_friction.setValue(friction)
                self.status_label.setText(
                    "Regional stress orientation set from focal-"
                    "mechanism inversion (Optimal Faults tab).")

    def add_focal_mechanisms_as_sources_action(self):
        """
        Turn every checked row in the Focal Mechanisms table into a row in
        the Source Faults table (self.fault_table), using that row's own
        "Source Plane" choice for orientation and the relation/style
        picked here for length/width/slip. See
        core.focal_mechanism.build_source_fault_row() for the per-event
        conversion and why lonlat_mode is always "centroid" for these rows.
        """
        selections = self.focal_results_widget.get_source_selections()
        if not selections:
            self.status_label.setText(
                "No focal mechanisms selected — tick 'Use as Source' for "
                "one or more rows in the Focal Mechanisms table first.")
            return

        relation_name = self.focal_source_relation_combo.currentText()
        style = self.focal_source_style_combo.currentText()

        added = 0
        errors = []
        for ev, plane in selections:
            try:
                row, warnings = build_source_fault_row(
                    ev, plane, relation_name, style)
            except ValueError as e:
                errors.append(str(e))
                continue
            vals = [row["name"], row["lon"], row["lat"], row["depth"],
                    row["length"], row["width"], row["strike"], row["dip"],
                    row["rt_lateral_slip"], row["reverse_slip"], row["rake"],
                    row["subdiv_l"], row["subdiv_w"], ""]
            self.fault_table.add_row(vals, lonlat_mode=row["lonlat_mode"])
            added += 1

        msg = f"Added {added} focal-mechanism-derived source fault(s)."
        if errors:
            msg += f" Skipped {len(errors)}: {errors[0]}"
            if len(errors) > 1:
                msg += f" (+{len(errors) - 1} more)"
        self.status_label.setText(msg)

    def compute_focal_mechanisms_action(self):
        sources = self._get_sources()
        if not sources:
            self.status_label.setText("Add at least one source fault first.")
            return
        if not getattr(self, "focal_events", None):
            self.status_label.setText(
                "No focal mechanisms imported yet — use 'Import Focal "
                "Mechanisms…' above.")
            return
        mode = self.focal_mode_combo.currentData()
        self._run_worker("focal_mech", sources, self._get_receiver(),
                         self._get_elastic(), focal_events=self.focal_events,
                         focal_mode=mode)

    def show_focal_mechanisms_on_map_action(self):
        if self._last_result is None or self._last_mode != "focal_mech":
            self.status_label.setText(
                "Compute ΔCFF on focal mechanisms first.")
            return
        self.plot_widget.plot_focal_mechanisms(self._last_result['focal_results'])

    def _set_buttons_enabled(self, enabled):
        self.btn_compute.setEnabled(enabled)
        self.btn_compute_disp.setEnabled(enabled)
        self.btn_compute_xs.setEnabled(enabled)
        self.btn_compute_receiver_faults.setEnabled(enabled)
        self.btn_compute_optimal.setEnabled(enabled)
        self.btn_compute_focal.setEnabled(enabled)
        self.btn_show_focal_on_map.setEnabled(enabled)

    def _run_worker(self, mode, sources, receiver, elastic, grid=None,
                    cross_section_params=None, receiver_faults=None,
                    regional=None, opt_friction=None,
                    focal_events=None, focal_mode=None):
        self._set_buttons_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._last_mode = mode

        # Advisory-only proximity check (2026-08-09b): closely-spaced
        # faults -- e.g. adjacent segments of a multi-segment rupture, or
        # a receiver fault sitting near a source segment -- fall in the
        # Okada/DC3D near-field zone, where ΔCFF reflects the dislocation
        # singularity rather than a reliable physical estimate. This does
        # NOT block the computation; it only surfaces a warning.
        near_field_warnings = near_field_fault_pairs(sources, receiver)

        # Total seismic moment / Mw readout (2026-08-10), mirroring
        # Coulomb 3.4.2's own seis_moment() console+status-bar print,
        # computed from whatever is CURRENTLY in the source-fault table
        # for this run. Reporting only -- printed to the console (the
        # closest analog here to Coulomb's MATLAB console) via print(),
        # and prepended to the status label, so it's diffable by eye
        # against Coulomb's own printed value for the same input.
        amo, mw = total_seismic_moment(sources, elastic)
        moment_msg = format_seismic_moment_message(amo, mw)
        if moment_msg:
            print(moment_msg)

        self._pending_moment_msg = moment_msg  # carried through to _on_finished, see below

        status_lines = []
        if moment_msg:
            status_lines.append(moment_msg)
        if near_field_warnings:
            status_lines.append(
                f"⚠️ {len(near_field_warnings)} near-field fault pair(s) "
                f"detected — see warning below. Computing {mode}…")
            self._pending_near_field_warnings = near_field_warnings
        else:
            status_lines.append(f"Computing {mode}…")
            self._pending_near_field_warnings = []
        self.status_label.setText("\n".join(status_lines))

        self.compute_thread = ComputeWorker(
            mode, sources, receiver, elastic, grid=grid,
            cross_section_params=cross_section_params,
            receiver_faults=receiver_faults,
            regional=regional, opt_friction=opt_friction,
            focal_events=focal_events, focal_mode=focal_mode)
        self.compute_thread.progress.connect(self.progress_bar.setValue)
        self.compute_thread.finished.connect(self._on_finished)
        self.compute_thread.error.connect(self._on_error)
        self.compute_thread.start()

    def _on_error(self, msg):
        self._set_buttons_enabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {msg}")

    def _on_finished(self, result: dict):
        self._set_buttons_enabled(True)
        self.progress_bar.setVisible(False)
        self._last_result = result

        if self._last_mode == "cff":
            lon2d, lat2d, cff = result['lon2d'], result['lat2d'], result['cff']
            used_dc3d = result.get('used_dc3d', False)
            depth_km = self.g_depth.value()

            if used_dc3d:
                formula_msg = f"Okada (1992) DC3D (external Python) — exact stress at {depth_km:.1f} km depth"
                depth_label = f"{depth_km:.1f} km (DC3D)"
            elif depth_km > 0:
                formula_msg = (f"Okada (1985) surface formula (z=0) — no external Python "
                               f"with okada-wrapper configured. Pattern correct; magnitudes "
                               f"are surface values, not {depth_km:.1f} km.")
                depth_label = "surface (z=0)"
            else:
                formula_msg = "Okada (1985) surface formula (z=0) — exact vs Coulomb 3.4.2"
                depth_label = "surface (z=0)"

            near_field_mask = result.get('near_field_mask')
            if near_field_mask is not None:
                frac_caution = float(np.mean(near_field_mask >= 1))
                frac_untrusted = float(np.mean(near_field_mask == 2))
                if frac_caution > 0:
                    formula_msg += (
                        f" | ⚠️ {frac_caution*100:.1f}% of grid points are within the "
                        f"near-field zone of a source fault (Okada/DC3D "
                        f"singularity) — magnitude uncertain there.")
                    if frac_untrusted > 0:
                        formula_msg += (
                            f" {frac_untrusted*100:.1f}% are close enough that "
                            f"even ΔCFF SIGN should not be trusted.")

            self.status_label.setText(formula_msg)
            self.plot_widget.plot_cff(lon2d, lat2d, cff, depth_label=depth_label,
                                      near_field_mask=near_field_mask)

        elif self._last_mode == "displacement":
            lon2d, lat2d = result['lon2d'], result['lat2d']
            ux, uy, uz = result['ux'], result['uy'], result['uz']
            used_dc3d = result.get('used_dc3d', False)
            depth_km = self.g_depth.value()
            depth_label = f"{depth_km:.1f} km (DC3D)" if used_dc3d else "surface (z=0)"

            note = ""
            if depth_km > 0 and not used_dc3d:
                note = (f" (no external Python configured — showing surface "
                       f"values, not {depth_km:.1f} km)")
            self.status_label.setText(
                f"Surface deformation computed{note}. Max |uz| = "
                f"{np.abs(uz).max()*100:.2f} cm")
            self.plot_widget.plot_displacement(lon2d, lat2d, ux, uy, uz,
                                               depth_label=depth_label)

        elif self._last_mode == "cross_section":
            dist_km, depth_km, cff_2d, used_dc3d = (
                result['dist_km'], result['depth_km'], result['cff_2d'],
                result.get('used_dc3d', False))

            if used_dc3d:
                msg = "Cross-section computed with Okada (1992) DC3D (external Python)."
            else:
                msg = ("Cross-section: only the surface (z=0) row is populated — "
                      "no external Python with okada-wrapper configured for "
                      "depths below the surface.")
            self.status_label.setText(msg)

            # 2026-08-18b: cross-section now draws into its own popup
            # CrossSectionWindow (configurable, GMT-style multi-panel
            # composer) rather than the embedded PlotWidget -- see that
            # addendum for why (embedding-window-height layout bugs).
            p = self._get_cross_section_params()
            half_width = self.xs_search_width.value()
            # 2026-08-21: profile is now always a vertex list (see the
            # "cross_section" worker-mode note above) -- use whatever
            # the worker actually computed against (result['profile_
            # vertices']) rather than rebuilding it from the current
            # spinbox values, so a mid-computation edit to the Start/
            # Finish fields can't mismatch the displayed traces against
            # the displayed ΔCFF grid.
            vertices = result.get('profile_vertices') or [
                (p['lon1'], p['lat1']), (p['lon2'], p['lat2'])]
            segment_info = result.get('segment_info')

            cfg = self.xs_config
            cfg.layout.main_vertical_exaggeration = self.xs_exaggeration.value()
            cfg.mesh.enabled = self.xs_show_mesh.isChecked()
            cfg.eq.enabled = self.xs_show_eq.isChecked()
            cfg.fault.enabled = self.xs_show_faults.isChecked()
            cfg.contours.enabled = self.xs_show_contours.isChecked()
            # cfg.topo_panels / cfg.annotations are NOT reassigned here --
            # cfg IS self.xs_config, and self.xs_config.topo_panels /
            # .annotations were aliased to self.xs_topo_panels /
            # self.xs_annotations once in __init__ (2026-08-19), so
            # add/remove actions and CrossSectionConfigDialog's cosmetic
            # edits are already visible here with no copy-back step --
            # reassigning `cfg.topo_panels = list(...)` would silently
            # break that alias for CrossSectionConfigDialog instances
            # already holding a reference to the old list.
            cfg.focal_mechanisms.enabled = self.xs_show_focal_mechanisms.isChecked()
            cfg.focal_mechanisms.search_width_km = self.xs_focal_search_width.value()
            cfg.focal_mechanisms.diameter_km = self.xs_focal_diameter.value()

            eq_data = None
            if cfg.eq.enabled and self.eq_events:
                lons = np.array([e.lon for e in self.eq_events])
                lats = np.array([e.lat for e in self.eq_events])
                depths = np.array([e.depth for e in self.eq_events])
                mags = np.array([e.magnitude if e.magnitude is not None
                                 else np.nan for e in self.eq_events])
                d_along, d_perp, plen = project_points_to_polyline(lons, lats, vertices)
                mask = filter_within_search_width(d_along, d_perp, plen, half_width)
                eq_data = {
                    "dist_km": d_along[mask], "depth_km": depths[mask],
                    "magnitude": mags[mask] if np.isfinite(mags[mask]).any() else None,
                }

            fault_traces = None
            if cfg.fault.enabled:
                fault_traces = project_fault_traces_onto_polyline(
                    self._get_sources(), vertices, half_width_km=half_width)

            # ── Topo panel(s) (2026-08-19, Phase 2 UI) ──────────────────
            # "qgis_layer" panels were stored as a layer ID (a stable
            # string) when added, not the live layer object -- resolved
            # back here, at gather time, since the set of loaded QGIS
            # layers can change between adding a panel and recomputing
            # (same reasoning as the add-panel action's own comment).
            topo_data = None
            if cfg.topo_panels:
                from qgis.core import QgsProject
                topo_data = []
                for panel in cfg.topo_panels:
                    try:
                        if panel.source_kind == "qgis_layer":
                            layer = QgsProject.instance().mapLayer(panel.source)
                            if layer is None:
                                topo_data.append(None)
                                continue
                            src = layer
                        else:
                            src = panel.source
                        d, z = sample_raster_along_line(
                            src, panel.source_kind, p['lon1'], p['lat1'],
                            p['lon2'], p['lat2'], n_samples=panel.n_samples,
                            band=panel.band,
                            elevation_unit_divisor=panel.elevation_unit_divisor)
                        topo_data.append((d, z))
                    except Exception as exc:
                        self.status_label.setText(
                            f"Topo panel {panel.label!r} failed to sample: {exc}")
                        topo_data.append(None)

            # ── Annotation source(s) (2026-08-19, Phase 2 UI) ───────────
            annotation_data = None
            if cfg.annotations:
                from qgis.core import QgsProject
                annotation_data = []
                for ann in cfg.annotations:
                    if not ann.enabled:
                        annotation_data.append((np.array([]), [], None))
                        continue
                    try:
                        if ann.source_kind == "qgis_layer":
                            layer = QgsProject.instance().mapLayer(ann.source)
                            if layer is None:
                                annotation_data.append((np.array([]), [], None))
                                continue
                            src = layer
                        else:
                            src = ann.source
                        d, labels, z = gather_annotation_points(
                            ann.source_kind, src, ann.label_field,
                            p['lon1'], p['lat1'], p['lon2'], p['lat2'],
                            ann.search_width_km, z_field=ann.z_field)
                        annotation_data.append((d, labels, z))
                    except Exception as exc:
                        self.status_label.setText(
                            f"Annotation source {ann.label!r} failed to load: {exc}")
                        annotation_data.append((np.array([]), [], None))

            # ── Focal mechanisms, side view (2026-08-19, Phase 2 physics
            #    + UI; magnitude/label wiring added 2026-08-20). Orientations
            #    and magnitude/label come straight from self.focal_events
            #    (imported via the Focal Mechanisms tab); ΔCFF coloring
            #    and which nodal plane PLANE_MODES picked are pulled from
            #    self._focal_mech_results if that computation has been
            #    run (matched to events by identity/order) -- otherwise
            #    plane1 is drawn, and color_by only falls back away from
            #    "cff" (to "single") if it was set to "cff" with nothing
            #    to color by -- "type"/"depth" don't need cff_vals at all
            #    and are left alone.
            focal_mechanism_data = None
            profile_dir = None
            if cfg.focal_mechanisms.enabled and self.focal_events:
                half_width_fm = cfg.focal_mechanisms.search_width_km
                lons = np.array([e.lon for e in self.focal_events])
                lats = np.array([e.lat for e in self.focal_events])
                depths = np.array([e.depth for e in self.focal_events])
                s1 = np.array([e.strike1 for e in self.focal_events])
                d1 = np.array([e.dip1 for e in self.focal_events])
                r1 = np.array([e.rake1 for e in self.focal_events])
                s2 = np.array([e.strike2 if e.strike2 is not None else np.nan
                               for e in self.focal_events])
                d2 = np.array([e.dip2 if e.dip2 is not None else np.nan
                               for e in self.focal_events])
                r2 = np.array([e.rake2 if e.rake2 is not None else np.nan
                               for e in self.focal_events])
                mags = np.array([e.magnitude if e.magnitude is not None else np.nan
                                 for e in self.focal_events])
                fm_labels = [e.label if e.label else None for e in self.focal_events]

                cff_vals = None
                selected_vals = None
                fm_results = getattr(self, "_focal_mech_results", None)
                if fm_results is not None and len(fm_results) == len(self.focal_events):
                    cff_vals = np.array([r["cff_mpa"] for r in fm_results])
                    selected_vals = [r["selected"] for r in fm_results]

                d_along, d_perp, plen = project_points_to_polyline(lons, lats, vertices)
                mask = filter_within_search_width(d_along, d_perp, plen, half_width_fm)

                if mask.any():
                    focal_mechanism_data = {
                        "dist_km": d_along[mask], "depth_km": depths[mask],
                        "strike1": s1[mask], "dip1": d1[mask], "rake1": r1[mask],
                        "strike2": s2[mask], "dip2": d2[mask], "rake2": r2[mask],
                        "cff_mpa": cff_vals[mask] if cff_vals is not None else None,
                        "magnitude": mags[mask] if np.isfinite(mags[mask]).any() else None,
                        "label": [lbl for lbl, m in zip(fm_labels, mask) if m],
                        "selected": ([sv for sv, m in zip(selected_vals, mask) if m]
                                    if selected_vals is not None else None),
                    }
                    if cff_vals is None and cfg.focal_mechanisms.color_by == "cff":
                        cfg.focal_mechanisms.color_by = "single"
                    # 2026-08-21 known limitation: for a MULTI-segment
                    # profile, focal_side_view's nodal-plane rotation
                    # still uses ONE overall straight start->finish
                    # direction (not each event's own nearest leg's
                    # azimuth) -- correct for a single-segment profile,
                    # an approximation once a profile bends. Flagged in
                    # the 2026-08-21 addendum as a follow-up rather than
                    # solved here (per-event leg-aware rotation needs
                    # focal_side_view's own rotation math extended, a
                    # separate piece of physics work from this session's
                    # projection/geometry fixes).
                    profile_dir = profile_direction(
                        vertices[0][0], vertices[0][1], vertices[-1][0], vertices[-1][1])

            # ── Extra imported depth-section elements ──
            extra_line_data = None
            if cfg.extra_lines:
                from qgis.core import QgsProject
                extra_line_data = []
                for line_cfg in cfg.extra_lines:
                    if line_cfg.source_kind in ("raster_file", "qgis_layer"):
                        # Raster-sourced element: resampled fresh along
                        # the CURRENT profile every time (like a topo
                        # panel), so it tracks start/finish/waypoint
                        # edits rather than using fixed, one-time-
                        # imported points.
                        try:
                            if line_cfg.source_kind == "qgis_layer":
                                src = QgsProject.instance().mapLayer(line_cfg.raster_source)
                                if src is None:
                                    extra_line_data.append(None)
                                    continue
                            else:
                                src = line_cfg.raster_source
                            _lons, _lats, d_along, zs = sample_raster_along_polyline(
                                src, line_cfg.source_kind, vertices,
                                n_samples=line_cfg.raster_n_samples,
                                band=line_cfg.raster_band,
                                unit_divisor=line_cfg.raster_unit_divisor,
                                sign=line_cfg.raster_sign)
                            finite = np.isfinite(zs)
                            if finite.any():
                                extra_line_data.append((d_along[finite], zs[finite]))
                            else:
                                extra_line_data.append(None)
                        except Exception as exc:
                            self.status_label.setText(
                                f"Depth-section element {line_cfg.label!r} "
                                f"failed to sample: {exc}")
                            extra_line_data.append(None)
                        continue

                    if not line_cfg.vertices:
                        extra_line_data.append(None)
                        continue
                    lons = np.array([v[0] for v in line_cfg.vertices])
                    lats = np.array([v[1] for v in line_cfg.vertices])
                    zs = np.array([v[2] for v in line_cfg.vertices])
                    d_along, d_perp, plen = project_points_to_polyline(lons, lats, vertices)
                    mask = filter_within_search_width(
                        d_along, d_perp, plen, line_cfg.search_width_km)
                    if mask.any():
                        extra_line_data.append((d_along[mask], zs[mask]))
                    else:
                        extra_line_data.append(None)

            fig = build_cross_section_figure(
                cfg, dist_km, depth_km, cff_2d, eq_data=eq_data,
                fault_traces=fault_traces, topo_data=topo_data,
                annotation_data=annotation_data,
                focal_mechanism_data=focal_mechanism_data,
                profile_direction=profile_dir,
                extra_line_data=extra_line_data, segment_info=segment_info,
                title="Coulomb Stress Cross-Section")

            if self._xs_window is None:
                # Parent to QGIS's own main window, not this dialog --
                # see cross_section_window.py's module docstring for
                # why parenting it to `self` made the window disappear
                # after closing/reopening the plugin (Qt hides
                # window-flagged children along with their parent, even
                # though it doesn't resize/reposition them).
                xs_parent = self.iface.mainWindow() if self.iface is not None else None
                self._xs_window = CrossSectionWindow(xs_parent)
            self._xs_window.set_figure(fig)
            self._xs_window.show_and_raise()

        elif self._last_mode == "receiver_faults":
            receiver_results = result['receiver_results']
            n_dc3d = sum(1 for r in receiver_results if r['used_dc3d'])
            n_total = len(receiver_results)
            if n_dc3d == n_total:
                msg = f"Computed ΔCFF on {n_total} receiver fault(s) — all via DC3D."
            elif n_dc3d == 0:
                msg = (f"Computed ΔCFF on {n_total} receiver fault(s) — surface "
                      f"formula (z=0). Deep receivers used surface values since "
                      f"no external Python is configured.")
            else:
                msg = (f"Computed ΔCFF on {n_total} receiver fault(s): "
                      f"{n_dc3d} via DC3D, {n_total - n_dc3d} via surface formula.")
            self.status_label.setText(msg)
            self.receiver_results_widget.set_results(receiver_results)

        elif self._last_mode == "optimal":
            lon2d, lat2d = result['lon2d'], result['lat2d']
            cff_opt_mpa = result['cff_opt_mpa']
            strike1, dip1, rake1 = result['strike1'], result['dip1'], result['rake1']
            strike2, dip2, rake2 = result['strike2'], result['dip2'], result['rake2']
            cff1_mpa, cff2_mpa = result['cff1_mpa'], result['cff2_mpa']
            used_dc3d = result.get('used_dc3d', False)
            orth_err = result.get('orthogonality_error_deg', 0.0)
            depth_km = self.g_depth.value()

            if used_dc3d:
                msg = f"Optimal-plane ΔCFF: Okada (1992) DC3D — exact at {depth_km:.1f} km depth."
                depth_label = f"{depth_km:.1f} km (DC3D)"
            elif depth_km > 0:
                msg = (f"Optimal-plane ΔCFF: Okada (1985) surface formula (z=0) — no "
                      f"external Python configured. Pattern correct; magnitudes are "
                      f"surface values, not {depth_km:.1f} km.")
                depth_label = "surface (z=0)"
            else:
                msg = "Optimal-plane ΔCFF: Okada (1985) surface formula (z=0)."
                depth_label = "surface (z=0)"

            frac_plane2 = float(np.mean(cff2_mpa > cff1_mpa))
            msg += (f" Plane 2 has the larger ΔCFF at {frac_plane2*100:.0f}% of grid "
                   f"points (the two conjugate planes generally differ — see tick "
                   f"marks on the map).")

            self.status_label.setText(msg)
            if orth_err > 1.0:
                self.lbl_orthogonality.setText(
                    f"⚠️ S1/S2 axes are {orth_err:.2f}° from perpendicular — "
                    f"consider adjusting their strike/plunge to be closer to 90° apart.")
                self.lbl_orthogonality.setStyleSheet("color: darkorange;")
            else:
                self.lbl_orthogonality.setText(
                    f"S1/S2 orthogonality error: {orth_err:.3f}°")
                self.lbl_orthogonality.setStyleSheet("color: gray;")

            self.plot_widget.plot_optimal_cff(
                lon2d, lat2d, cff_opt_mpa, strike1, strike2, cff1_mpa, cff2_mpa,
                depth_label=depth_label)

        elif self._last_mode == "focal_mech":
            focal_results = result['focal_results']
            n_dc3d = sum(1 for r in focal_results if r['used_dc3d'])
            n_total = len(focal_results)
            self.status_label.setText(
                f"Computed ΔCFF on {n_total} focal mechanism(s) "
                f"({n_dc3d} via DC3D).")
            self.focal_results_widget.set_results(focal_results)
            # Kept independent of self._last_result (which the NEXT
            # compute action overwrites regardless of mode) so the
            # cross-section's focal-mechanism overlay can still color by
            # ΔCFF / highlight the PLANE_MODES-selected plane from the
            # most recent focal-mechanism run, even after switching tabs
            # and computing a cross-section afterward.
            self._focal_mech_results = focal_results

        # Total seismic moment / Mw readout (2026-08-10) -- prepended so
        # it survives the mode-specific status_label.setText() calls
        # above, same pattern as the near-field warnings below (which
        # were already solving the identical "don't get overwritten"
        # problem for a per-run, mode-independent message).
        moment_msg = getattr(self, "_pending_moment_msg", None)
        if moment_msg:
            self.status_label.setText(moment_msg + "\n\n" + self.status_label.text())

        # Pairwise near-field warnings (2026-08-09b) apply regardless of
        # mode -- appended after the mode-specific status text above so
        # they don't get overwritten by it.
        pending = getattr(self, "_pending_near_field_warnings", [])
        if pending:
            self.status_label.setText(
                self.status_label.text() + "\n\n" + "\n".join(
                    f"⚠️ {w}" for w in pending))

    # ── Export ───────────────────────────────────────────────────────────

    def _current_grid_values(self):
        """Return (lon2d, lat2d, values, value_name) for the last CFF or
        displacement result, or (None, None, None, None) if unavailable."""
        if self._last_result is None:
            return None, None, None, None
        if self._last_mode == "cff":
            return (self._last_result['lon2d'], self._last_result['lat2d'],
                   self._last_result['cff'], "Coulomb Stress Change")
        if self._last_mode == "displacement":
            return (self._last_result['lon2d'], self._last_result['lat2d'],
                   self._last_result['uz'], "Vertical Displacement")
        if self._last_mode == "optimal":
            return (self._last_result['lon2d'], self._last_result['lat2d'],
                   self._last_result['cff_opt_mpa'], "Optimal-Plane Coulomb Stress Change")
        return None, None, None, None

    def add_raster_to_project(self):
        lon2d, lat2d, values, name = self._current_grid_values()
        if lon2d is None:
            self.status_label.setText("Compute CFF or displacement first.")
            return
        from ..utils.raster_utils import add_raster_to_project
        layer = add_raster_to_project(lon2d, lat2d, values, layer_name=name)
        if layer is not None:
            self.status_label.setText(f"Raster layer added to project: {name}")
        else:
            self.status_label.setText("Failed to add raster layer.")

    def save_raster_to_file(self):
        lon2d, lat2d, values, name = self._current_grid_values()
        if lon2d is None:
            self.status_label.setText("Compute CFF or displacement first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save GeoTIFF", "", "GeoTIFF (*.tif)")
        if not path:
            return
        from ..utils.raster_utils import write_geotiff, load_raster_layer
        write_geotiff(path, lon2d, lat2d, values)
        load_raster_layer(path, name)
        self.status_label.setText(f"Raster saved and added: {os.path.basename(path)}")

    def save_plot_to_file(self):
        """
        Save whatever is currently shown in the preview panel (CFF map,
        displacement map, or cross-section) as an image file. Works for
        all plot types since it saves the matplotlib figure directly,
        regardless of which plot_* method last drew it.
        """
        if self._last_result is None:
            self.status_label.setText("Run a computation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;JPEG (*.jpg)")
        if not path:
            return
        try:
            self.plot_widget.save_to_file(path)
            self.status_label.setText(f"Plot saved: {os.path.basename(path)}")
        except Exception as e:
            self.status_label.setText(f"Failed to save plot: {type(e).__name__}: {e}")

    def export_csv(self):
        if self._last_mode == "optimal":
            if self._last_result is None:
                self.status_label.setText("Run a computation first.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV (*.csv)")
            if not path:
                return
            if not path.endswith(".csv"):
                path += ".csv"
            from ..utils.raster_utils import write_csv_multi
            r = self._last_result
            write_csv_multi(path, r['lon2d'], r['lat2d'], {
                "cff_opt_mpa": r['cff_opt_mpa'],
                "strike1_deg": r['strike1'], "dip1_deg": r['dip1'], "rake1_deg": r['rake1'],
                "cff1_mpa": r['cff1_mpa'],
                "strike2_deg": r['strike2'], "dip2_deg": r['dip2'], "rake2_deg": r['rake2'],
                "cff2_mpa": r['cff2_mpa'],
            })
            self.status_label.setText(f"Exported: {os.path.basename(path)}")
            return

        lon2d, lat2d, values, name = self._current_grid_values()
        if lon2d is None:
            self.status_label.setText("Run a computation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV/XYZ", "", "CSV (*.csv);;XYZ (*.xyz)")
        if not path:
            return
        from ..utils.raster_utils import write_csv, write_xyz
        value_name = "cff_mpa" if self._last_mode == "cff" else "uz_m"
        if path.endswith(".xyz"):
            write_xyz(path, lon2d, lat2d, values)
        else:
            write_csv(path, lon2d, lat2d, values, value_name=value_name)
        self.status_label.setText(f"Exported: {os.path.basename(path)}")

    def export_vectors(self):
        sources = self._get_sources()
        if not sources:
            self.status_label.setText("Add at least one source fault first.")
            return
        from ..utils.vector_utils import (
            create_fault_layer, create_surface_trace_layer,
            create_geological_surface_trace_layer,
            create_receiver_depth_layer, create_displacement_layer,
            create_near_field_mask_layer, create_optimal_plane_strike_layer,
        )
        create_fault_layer(sources, "Source Faults")
        create_surface_trace_layer(sources, "Fault Top Projection")
        create_geological_surface_trace_layer(sources, "Surface Trace (extrapolated to z=0)")

        z_recv = self.g_depth.value()
        if z_recv > 0:
            create_receiver_depth_layer(sources, z_recv)

        extra_msgs = []
        if self._last_result is not None and self._last_mode == "displacement":
            create_displacement_layer(
                self._last_result['lon2d'], self._last_result['lat2d'],
                self._last_result['ux'], self._last_result['uy'])

        # Near-field hatch overlay (whichever mode last drew it -- "cff"
        # and "optimal" both carry a near_field_mask in self._last_result).
        near_field_mask = (self._last_result.get('near_field_mask')
                          if self._last_result is not None else None)
        if near_field_mask is not None and np.any(np.asarray(near_field_mask) > 0):
            try:
                layer = create_near_field_mask_layer(
                    self._last_result['lon2d'], self._last_result['lat2d'],
                    near_field_mask)
                if layer is not None:
                    extra_msgs.append("near-field hatch layer")
            except ValueError as e:
                extra_msgs.append(f"near-field hatch layer skipped ({e})")

        # Optimal-plane strike tick-mark vectors (only meaningful after
        # an optimal-plane computation).
        if self._last_result is not None and self._last_mode == "optimal":
            r = self._last_result
            create_optimal_plane_strike_layer(
                r['lon2d'], r['lat2d'], r['strike1'], r['strike2'],
                r['cff1_mpa'], r['cff2_mpa'])
            extra_msgs.append("optimal-plane strike vectors")

        msg = "Vector layers added to project."
        if extra_msgs:
            msg += " Also added: " + ", ".join(extra_msgs) + "."
        self.status_label.setText(msg)

    def export_receiver_faults_layer(self):
        """
        Add a polygon layer of individual receiver faults, colored by
        their resolved ΔCFF (diverging red/blue), distinct from the solid
        amber source-fault layer. Requires having already run 'Compute
        ΔCFF on Receiver Faults' at least once.
        """
        if self._last_mode != "receiver_faults" or self._last_result is None:
            self.status_label.setText(
                "Compute ΔCFF on Receiver Faults first (Receiver Faults tab).")
            return
        try:
            from ..utils.vector_utils import create_receiver_fault_layer_colored
            create_receiver_fault_layer_colored(self._last_result['receiver_results'])
            self.status_label.setText("Colored receiver fault layer added to project.")
        except Exception as e:
            import traceback
            self.status_label.setText(
                f"Failed to create receiver fault layer: {type(e).__name__}: {e}")
            print(traceback.format_exc())  # full traceback to QGIS Python console/log

    def export_focal_mechanisms_layer(self):
        """
        Add a polygon layer of literal CFF-colored beachball glyphs (two
        polygons per event -- compressional lobe pair colored by ΔCFF,
        background lobe pair white) to the real QGIS map canvas. This is
        distinct from 'Preview Beachballs', which only renders into
        the plugin's own embedded preview panel, not the project map.
        Requires having already run 'Compute ΔCFF on Focal Mechanisms'
        at least once. Requires `obspy` (pip install obspy) in QGIS's
        Python environment.
        """
        if self._last_mode != "focal_mech" or self._last_result is None:
            self.status_label.setText(
                "Compute ΔCFF on Focal Mechanisms first (Focal Mechanisms tab).")
            return
        try:
            from ..utils.focal_mechanism_layer import create_focal_mechanism_beachball_layer
            create_focal_mechanism_beachball_layer(self._last_result['focal_results'])
            self.status_label.setText("Beachball layer added to project.")
        except ImportError as e:
            self.status_label.setText(
                f"Beachball layer export needs 'obspy' — pip install obspy "
                f"into QGIS's Python environment. ({e})")
        except Exception as e:
            import traceback
            self.status_label.setText(
                f"Failed to create beachball layer: {type(e).__name__}: {e}")
            print(traceback.format_exc())  # full traceback to QGIS Python console/log

    # ── Setup save/load (native JSON + Coulomb .inp bridge) ────────────────

    def save_setup_action(self):
        """Save every field in this dialog (fault table, receiver, grid,
        elastic params, cross-section) to a JSON file for later reload.
        Not meant to be opened by Coulomb itself -- see 'Export → Coulomb
        .inp…' for that."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Setup", "", "Coulomb Stress Transfer setup (*.json)")
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        from .project_io import save_setup
        try:
            save_setup(self, path)
            self.status_label.setText(f"Setup saved: {os.path.basename(path)}")
        except Exception as e:
            self.status_label.setText(f"Failed to save setup: {type(e).__name__}: {e}")

    def load_setup_action(self):
        """Load a setup previously written by 'Save Setup…', replacing
        every field in this dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Setup", "", "Coulomb Stress Transfer setup (*.json)")
        if not path:
            return
        from .project_io import load_setup
        try:
            load_setup(self, path)
            self.status_label.setText(f"Setup loaded: {os.path.basename(path)}")
        except Exception as e:
            self.status_label.setText(f"Failed to load setup: {type(e).__name__}: {e}")

    def export_inp_action(self):
        """Export the Source Faults table (plus grid/cross-section/elastic
        params) as a Coulomb-3.4.2-compatible ASCII .inp file. See
        project_io.export_inp()'s docstring for the format's scope limits
        (source-fault geometry round-trips exactly; Coulomb's own
        receiver-plane KODE types are not reproduced)."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export as Coulomb .inp", "", "Coulomb ASCII input (*.inp)")
        if not path:
            return
        if not path.endswith(".inp"):
            path += ".inp"
        from .project_io import export_inp
        try:
            export_inp(self, path)
            self.status_label.setText(f"Exported Coulomb .inp: {os.path.basename(path)}")
        except Exception as e:
            self.status_label.setText(f"Failed to export .inp: {type(e).__name__}: {e}")

    def import_inp_action(self):
        """Import fault geometry (and Poisson's ratio / shear modulus /
        friction, and grid bounds, where present) from a Coulomb-3.4.2
        ASCII .inp file, replacing the Source Faults table."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Coulomb .inp", "", "Coulomb ASCII input (*.inp *.inr)")
        if not path:
            return
        from .project_io import import_inp
        try:
            warnings = import_inp(self, path)
            msg = f"Imported Coulomb .inp: {os.path.basename(path)}"
            if warnings:
                msg += "\n\n" + "\n".join(f"⚠️ {w}" for w in warnings)
            self.status_label.setText(msg)
        except Exception as e:
            self.status_label.setText(f"Failed to import .inp: {type(e).__name__}: {e}")
