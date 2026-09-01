# -*- coding: utf-8 -*-
"""
Editable table widget for source fault parameters.

Each row carries its OWN "Lon/Lat mode" (a per-row combo box, not a
single table-wide toggle), describing what that row's (Lon, Lat, Depth)
means as a whole:

  "Top edge — start point"  — Lon/Lat is the STARTING point of the
    fault's TOP-EDGE surface trace ("Fault Top Projection"); Depth is
    the top-edge depth. The trace runs from there in the +strike
    direction for Length km.

  "Top edge — center point" — Lon/Lat is the MIDPOINT of the top-edge
    surface trace (the standard Aki-Richards/SRCMOD "top-center"
    reference point); Depth is the top-edge depth. (Default for new rows.)

  "Centroid (e.g. focal mechanism)" — (Lon, Lat, Depth) is the fault's
    VOLUMETRIC CENTROID directly, exactly as reported by a focal
    mechanism / moment-tensor solution. No geometric conversion is
    applied at all.

This is deliberately PER-ROW rather than a single table-wide toggle
(the plugin's earlier design): a table-wide toggle means switching it
for one purpose (e.g. importing a digitized polyline, which is always
"top edge — start point") silently reinterprets every OTHER row's
Lon/Lat/Depth under the new meaning too. Per-row mode means a
focal-mechanism-derived row and a digitized-polyline-derived row can
coexist in the same table safely, each carrying its own correct
convention independent of the others.

Note: Lon/Lat/Depth in ANY mode never refers to "Fault Top Projection"
or "Surface Trace" as separate quantities — those are always DERIVED,
display-only projections (see utils/vector_utils.py), never inputs.

Slip is entered as right-lateral + reverse components (Coulomb's own
convention), which decomposes into rake/scalar-slip internally.

Rows with ZERO total slip (rt-lateral == reverse == 0) act as
individual RECEIVER faults (see get_receiver_faults() below) rather
than stress sources. For a SOURCE row, rake is unambiguous: it falls
out of the rt-lateral/reverse decomposition (from_rt_lat_reverse()).
But a RECEIVER row has zero slip by definition, so that same
decomposition always collapses to rake=0 (atan2(0,0)) regardless of
what orientation the user actually wants resolved — there is no slip
vector to derive a rake from. The "Rake (receiver, °)" column exists
to supply that otherwise-unrecoverable value explicitly, in
Aki-Richards convention (0/180=strike-slip, +90=reverse, -90=normal,
matching the Receiver Fault tab's own Rake spinbox). It is read and
applied ONLY for zero-slip rows; for nonzero-slip (source) rows it is
ignored and the rake continues to come from rt-lateral/reverse as
before.

Two further per-row/per-group features, both built on the SAME
annex-letter naming convention ("-A", "-B", "-C", ...) used above for
subdivided sub-patches (see _annex_labels()):

  "Group" column — selecting two or more rows and clicking "🔗 Merge
    selected into group" (or just typing the same text into each row's
    Group cell by hand) makes those rows act as ONE named source fault
    for naming/organizational purposes: they are emitted as
    "GroupName-A", "GroupName-B", ... in table order. Each row KEEPS
    its own independent geometry (lon/lat/depth/length/width/strike/
    dip) and slip — grouping does not merge them into a single Okada
    rectangle (that would only be physically valid if every member
    were exactly collinear, same strike/dip/width, which digitized
    polyline segments across a bend are generally not). This is the
    intended use case: several imported polyline segments that trace
    out one bent/kinked geological fault can be grouped so they read,
    save, and export as one logically-named fault while each segment's
    own strike/slip stays physically correct.

  Distributed slip — for a row with Subdiv.(L)>1 or Subdiv.(W)>1,
    "🎚 Edit distributed slip…" opens a per-sub-patch editor (see
    ui/distributed_slip_dialog.py) so each sub-patch can be given its
    own right-lateral/reverse slip instead of all inheriting the
    parent row's single uniform value — a genuine variable-slip
    source, not just a display subdivision. Sub-patches left at the
    uniform default keep behaving exactly as before. Overrides are
    stored per-row (travel with the row; cleared by re-editing) and
    round-trip through get_raw_rows()/set_raw_rows() (JSON setup
    save/load), but are NOT used by Coulomb .inp export, which -- like
    subdivision itself -- exports one row per input row (Coulomb's own
    subdivision/variable-slip is a GUI/display-time feature there too).
"""

import json
import math

from qgis.PyQt.QtWidgets import (QTableWidget, QTableWidgetItem, QWidget,
                                  QVBoxLayout, QHBoxLayout, QPushButton,
                                  QHeaderView, QLabel, QComboBox,
                                  QMessageBox, QInputDialog)
from qgis.PyQt.QtCore import Qt

from ..core.okada_engine import FaultParameters

# (internal lon_lat_mode value, display label shown in the per-row combo box)
LONLAT_MODE_OPTIONS = [
    ("top_start", "Top edge — start point"),
    ("top_center", "Top edge — center point"),
    ("centroid", "Centroid (e.g. focal mechanism)"),
]
LONLAT_LABEL_TO_VALUE = {label: val for val, label in LONLAT_MODE_OPTIONS}
LONLAT_VALUE_TO_LABEL = {val: label for val, label in LONLAT_MODE_OPTIONS}
DEFAULT_LONLAT_MODE = "top_center"

_SETTINGS_KEY_DEFAULT_MODE = "CoulombStressTransfer/default_lonlat_mode"


def _get_default_lonlat_mode():
    """
    Return the persisted default Lon/Lat mode for NEWLY-ADDED rows (via
    "+ Add fault"), stored via QgsSettings so it's remembered across QGIS
    sessions/restarts -- mirrors the same QgsSettings pattern already used
    for the external-Python DC3D path (core/okada_engine.py,
    _get_external_python_path()/_set_external_python_path()). Falls back
    to DEFAULT_LONLAT_MODE if unset, or if an unrecognized value was
    somehow stored (e.g. an older/newer version of this plugin).

    This ONLY affects the default shown when a NEW row is added with no
    explicit lonlat_mode. Rows added by polyline import always explicitly
    pass lonlat_mode="top_start" regardless of this setting (see
    _open_polyline_import_dialog()), and existing rows' combo boxes are
    never touched by changing this default.
    """
    try:
        from qgis.core import QgsSettings
        value = QgsSettings().value(
            _SETTINGS_KEY_DEFAULT_MODE, DEFAULT_LONLAT_MODE, type=str)
    except Exception:
        return DEFAULT_LONLAT_MODE
    return value if value in LONLAT_VALUE_TO_LABEL else DEFAULT_LONLAT_MODE


def _set_default_lonlat_mode(value):
    """Persist the default Lon/Lat mode for newly-added rows via QgsSettings."""
    try:
        from qgis.core import QgsSettings
        QgsSettings().setValue(_SETTINGS_KEY_DEFAULT_MODE, value)
    except Exception:
        pass

COLUMNS = ["Name", "Lon/Lat mode", "Lon", "Lat", "Depth (km)", "Length (km)", "Width (km)",
          "Strike (°)", "Dip (°)", "Rt-lateral slip (m)", "Reverse slip (m)",
          "Rake (receiver, °)", "Subdiv. (L)", "Subdiv. (W)", "Group"]

# Full-length hover tooltip per column, in the SAME order as COLUMNS --
# lets the whole column name/meaning show on hover even when the
# Stretch-resized column itself is too narrow to display it (see
# module docstring for the underlying per-column semantics in full).
COLUMN_TOOLTIPS = [
    "Name",
    "Lon/Lat mode — what this row's Lon/Lat/Depth mean: fault top-edge "
    "start point, top-edge center point, or volumetric centroid.",
    "Longitude (°) — interpreted per this row's own Lon/Lat mode.",
    "Latitude (°) — interpreted per this row's own Lon/Lat mode.",
    "Depth (km) — interpreted per this row's own Lon/Lat mode.",
    "Length (km) — along-strike length of the fault plane.",
    "Width (km) — down-dip width of the fault plane.",
    "Strike (°) — fault strike, degrees clockwise from North.",
    "Dip (°) — fault dip angle, degrees from horizontal.",
    "Rt-lateral slip (m) — right-lateral slip component (Coulomb convention).",
    "Reverse slip (m) — reverse slip component (Coulomb convention).",
    "Rake (receiver, °) — Aki-Richards rake, used ONLY for zero-slip "
    "(receiver) rows to resolve stress on that row's own orientation; "
    "ignored for nonzero-slip (source) rows.",
    "Subdiv. (L) — number of equal sub-patches to split into along strike.",
    "Subdiv. (W) — number of equal sub-patches to split into down-dip.",
    "Group — rows sharing a non-empty Group name are treated/exported "
    "as one logically-named fault (\"Name-A\", \"Name-B\", ...) while "
    "each row keeps its own independent geometry and slip.",
]

# Column indices. COL_LONLAT_MODE is a QComboBox cell widget, not a
# QTableWidgetItem -- it is set/read separately from the other columns.
(COL_NAME, COL_LONLAT_MODE, COL_LON, COL_LAT, COL_DEPTH, COL_LENGTH,
 COL_WIDTH, COL_STRIKE, COL_DIP, COL_RTLAT, COL_REVERSE, COL_RAKE,
 COL_SUBL, COL_SUBW, COL_GROUP) = range(15)

# Defaults for every column EXCEPT COL_LONLAT_MODE (that one is set via
# the `lonlat_mode` parameter to add_row(), not through this list), in
# the same left-to-right order as the other columns.
DATA_COLUMNS = [COL_NAME, COL_LON, COL_LAT, COL_DEPTH, COL_LENGTH, COL_WIDTH,
                COL_STRIKE, COL_DIP, COL_RTLAT, COL_REVERSE, COL_RAKE,
                COL_SUBL, COL_SUBW, COL_GROUP]
DEFAULTS = ["Fault 1", 0.0, 0.0, 5.0, 20.0, 10.0, 0.0, 90.0, 1.0, 0.0, 0.0, 1, 1, ""]


class FaultTableWidget(QWidget):
    """Editable table for one or more source/receiver faults."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "<b>Source Faults</b> — each row's <b>Lon/Lat mode</b> column "
            "says what that row's Lon/Lat/Depth means: the fault's TOP "
            "EDGE (start or center point, with Depth = top-edge depth), "
            "or the fault's own volumetric CENTROID directly (Depth = "
            "centroid depth — use this for focal-mechanism/moment-tensor "
            "input). Each row is independent, so rows from different "
            "sources (manual entry, focal mechanisms, digitized "
            "polylines) can be mixed safely in the same table."))

        layout.addWidget(QLabel(
            "<i>Lon/Lat/Depth are always an input to the fault PLANE "
            "itself (per that row's Lon/Lat mode) — never the map-view "
            "\"Fault Top Projection\" or \"Surface Trace\" lines, which "
            "are always computed/derived for display, never separate "
            "inputs. Changing a row's Lon/Lat mode re-interprets ONLY "
            "that row's values, not any other row's.</i>"))

        default_mode_row = QHBoxLayout()
        default_mode_row.addWidget(QLabel("New rows (\"+ Add fault\") default to:"))
        self.default_mode_combo = QComboBox()
        self.default_mode_combo.addItems([label for _val, label in LONLAT_MODE_OPTIONS])
        self.default_mode_combo.setCurrentText(
            LONLAT_VALUE_TO_LABEL[_get_default_lonlat_mode()])
        self.default_mode_combo.currentIndexChanged.connect(self._on_default_mode_changed)
        default_mode_row.addWidget(self.default_mode_combo)
        default_mode_row.addWidget(QLabel(
            "<i>(remembered across QGIS sessions; polyline imports always "
            "use \"Top edge — start point\" regardless of this)</i>"))
        default_mode_row.addStretch()
        layout.addLayout(default_mode_row)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        # Column widths are stretched to fill the available space (see
        # setSectionResizeMode(Stretch) below), so with 15 columns many
        # headers get visually truncated -- hovering shows the full name
        # via tooltip as a lightweight fix (2026-08-15c).
        for col, tip in enumerate(COLUMN_TOOLTIPS):
            item = self.table.horizontalHeaderItem(col)
            if item is not None and tip:
                item.setToolTip(tip)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table)

        layout.addWidget(QLabel(
            "<i>Rows with zero total slip (highlighted) act as individual "
            "RECEIVER faults — Coulomb stress is resolved on each one's "
            "own strike/dip/<b>Rake (receiver, °)</b> column (Aki-Richards "
            "convention: 0/180=strike-slip, +90=reverse, -90=normal). That "
            "Rake column is used ONLY for zero-slip rows — a source row's "
            "rake instead comes from its rt-lateral/reverse slip, and the "
            "Rake column is ignored for it. Rows with nonzero slip act as "
            "stress SOURCES. 'Subdiv. (L)' / 'Subdiv. (W)' split a fault "
            "into that many equal-area patches along strike / down-dip.</i>"))

        layout.addWidget(QLabel(
            "<i>'Group': select ≥2 rows and click \"🔗 Merge selected into "
            "group\" (or just type the same name into each row's Group "
            "cell) to make them read/export as one logically-named fault "
            "— \"Name-A\", \"Name-B\", ... — while each row keeps its own "
            "independent geometry and slip. Useful for several digitized "
            "polyline segments that together trace one bent fault. "
            "'🎚 Edit distributed slip…' (for a row with Subdiv.(L)>1 or "
            "Subdiv.(W)>1) lets each sub-patch have its own slip instead "
            "of all inheriting the row's single uniform value.</i>"))

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("+ Add fault")
        self.btn_remove = QPushButton("- Remove selected")
        self.btn_scaling = QPushButton("📏 Estimate L/W/slip from magnitude…")
        self.btn_import_polyline = QPushButton("📍 Import from QGIS polyline…")
        self.btn_import_fault_table = QPushButton("📋 Import fault-patch table…")
        self.btn_construct_grid = QPushButton("🧱 Construct fault grid (variable dip)…")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_scaling)
        btn_row.addWidget(self.btn_import_polyline)
        btn_row.addWidget(self.btn_import_fault_table)
        btn_row.addWidget(self.btn_construct_grid)
        layout.addLayout(btn_row)

        btn_row2 = QHBoxLayout()
        self.btn_merge_group = QPushButton("🔗 Merge selected into group")
        self.btn_ungroup = QPushButton("✂ Ungroup selected")
        self.btn_distributed_slip = QPushButton("🎚 Edit distributed slip…")
        self.btn_slip_inversion = QPushButton("📡 Invert for slip…")
        btn_row2.addWidget(self.btn_merge_group)
        btn_row2.addWidget(self.btn_ungroup)
        btn_row2.addWidget(self.btn_distributed_slip)
        btn_row2.addWidget(self.btn_slip_inversion)
        layout.addLayout(btn_row2)

        self.btn_add.clicked.connect(lambda checked=False: self.add_row())
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_scaling.clicked.connect(self._open_scaling_dialog)
        self.btn_import_polyline.clicked.connect(self._open_polyline_import_dialog)
        self.btn_import_fault_table.clicked.connect(self._open_fault_table_import_dialog)
        self.btn_construct_grid.clicked.connect(self._open_fault_grid_builder_dialog)
        self.btn_merge_group.clicked.connect(self._merge_selected_into_group)
        self.btn_ungroup.clicked.connect(self._ungroup_selected)
        self.btn_distributed_slip.clicked.connect(self._open_distributed_slip_dialog)
        self.btn_slip_inversion.clicked.connect(self._open_slip_inversion_dialog)

        self._fault_counter = 1
        self._elastic_provider = None  # set via set_elastic_provider(); needed
                                        # only by _open_slip_inversion_dialog()
        self.add_row()  # start with one fault

    def set_elastic_provider(self, provider):
        """
        provider : zero-arg callable returning the CURRENT
                   core.okada_engine.ElasticParameters (e.g.
                   MainDialog._get_elastic) -- called fresh each time
                   the slip-inversion dialog is opened, so it always
                   uses the same mu/nu as the rest of the plugin's
                   Elastic tab rather than a stale/duplicated copy.
        """
        self._elastic_provider = provider

    # ── Row operations ───────────────────────────────────────────────────

    def add_row(self, values=None, lonlat_mode=None, distributed_slip=None):
        """
        Add one row.

        values      : list of values for DATA_COLUMNS (i.e. every column
                      EXCEPT Lon/Lat mode), in that column order. If None,
                      uses DEFAULTS with an auto-incremented fault name.
        lonlat_mode : internal lon_lat_mode value ("top_start",
                      "top_center", or "centroid") for this row's Lon/Lat
                      mode combo box. Defaults to DEFAULT_LONLAT_MODE
                      ("top_center") if not given or not recognized.
        distributed_slip : optional list of [i, j, rt_lateral_slip,
                      reverse_slip] entries (same shape as
                      get_raw_rows()'s "distributed_slip" field) to
                      pre-populate this row's per-sub-patch slip
                      overrides. None/empty = fully uniform (default).
        """
        row = self.table.rowCount()
        self.table.insertRow(row)
        if values is None:
            vals = list(DEFAULTS)
            vals[0] = f"Fault {self._fault_counter}"  # DATA_COLUMNS[0] == COL_NAME
            self._fault_counter += 1
        else:
            vals = values

        self.table.blockSignals(True)
        for col, v in zip(DATA_COLUMNS, vals):
            text = f"{v:.6g}" if isinstance(v, float) else str(v)
            item = QTableWidgetItem(text)
            if col != COL_NAME:
                item.setTextAlignment(Qt.AlignCenter)
            if col == COL_NAME and distributed_slip:
                item.setData(Qt.UserRole, json.dumps(distributed_slip))
            self.table.setItem(row, col, item)

        combo = QComboBox()
        combo.addItems([label for _val, label in LONLAT_MODE_OPTIONS])
        effective_mode = lonlat_mode if lonlat_mode is not None else _get_default_lonlat_mode()
        combo.setCurrentText(
            LONLAT_VALUE_TO_LABEL.get(effective_mode, LONLAT_VALUE_TO_LABEL[DEFAULT_LONLAT_MODE]))
        combo.currentIndexChanged.connect(lambda _idx=None: self._refresh_row_colors())
        self.table.setCellWidget(row, COL_LONLAT_MODE, combo)

        self.table.blockSignals(False)
        self._refresh_row_colors()

    def _on_default_mode_changed(self):
        """Persist the "New rows default to:" selection via QgsSettings.
        Does NOT touch any existing row's Lon/Lat mode combo box."""
        label = self.default_mode_combo.currentText()
        value = LONLAT_LABEL_TO_VALUE.get(label, DEFAULT_LONLAT_MODE)
        _set_default_lonlat_mode(value)

    def _on_item_changed(self, item):
        self._refresh_row_colors()

    def _refresh_row_colors(self):
        """Highlight receiver-only rows (zero total slip) with a distinct
        background so it's visually obvious which rows drive stress vs.
        receive it. Also lightly styles the Lon/Lat-mode combo box to
        match, since it's a cell widget (not a QTableWidgetItem) and
        setBackground() on the (nonexistent) item there is a no-op.

        A row's OWN uniform Rt-lateral/Reverse slip is not the only way
        it can be a source: '🎚 Edit distributed slip…' and '🧪 Invert
        for slip…' both write PER-PATCH overrides directly (see
        _set_distributed_slip()) while deliberately leaving the row's
        own uniform slip at 0 (see core.okada_engine.FaultParameters.
        subdivide()'s docstring -- overriding sub-patches is the whole
        point, the parent's own slip is irrelevant once every sub-patch
        has its own). Checking only rt_lat/reverse here (as this method
        used to) marked exactly this kind of row -- one with a
        perfectly good, nonzero per-patch slip distribution just
        applied by a slip inversion -- as a zero-slip RECEIVER, which
        reads as \"the inversion result wasn't applied\" even though
        get_faults()/get_sources() were already using it correctly.
        (2026-08-30 fix.)"""
        from qgis.PyQt.QtGui import QColor
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            try:
                rt_lat = float(self.table.item(row, COL_RTLAT).text())
                reverse = float(self.table.item(row, COL_REVERSE).text())
                slip = (rt_lat**2 + reverse**2) ** 0.5
            except (ValueError, AttributeError):
                slip = None

            has_distributed_slip = any(
                abs(rt) > 1e-9 or abs(rev) > 1e-9
                for (rt, rev) in self._get_distributed_slip(row).values())

            is_receiver = (slip is not None and abs(slip) <= 1e-9
                          and not has_distributed_slip)
            color = QColor(200, 220, 255) if is_receiver else QColor(255, 255, 255)
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is not None:
                    item.setBackground(color)
            combo = self.table.cellWidget(row, COL_LONLAT_MODE)
            if combo is not None:
                combo.setStyleSheet(f"background-color: {color.name()};")
        self.table.blockSignals(False)

    def remove_selected(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        if self.table.rowCount() == 0:
            self.add_row()

    # ── Distributed slip (per-sub-patch overrides) ─────────────────────────

    def _get_distributed_slip(self, row):
        """Return this row's stored overrides as {(i,j): (rt_lat, reverse)},
        or {} if none set. Stored as JSON on the COL_NAME item's UserRole
        data so it travels with the row through insert/remove."""
        item = self.table.item(row, COL_NAME)
        if item is None:
            return {}
        raw = item.data(Qt.UserRole)
        if not raw:
            return {}
        try:
            pairs = json.loads(raw)
            return {(int(i), int(j)): (float(rt), float(rev)) for i, j, rt, rev in pairs}
        except (ValueError, TypeError, json.JSONDecodeError):
            return {}

    def _set_distributed_slip(self, row, overrides):
        """Store {(i,j): (rt_lat, reverse)} overrides on the row (empty
        dict clears it back to fully uniform).

        Also sets/clears a tooltip on the Name cell summarizing the
        distributed slip (min/mean/max magnitude across patches) --
        this is now the ONLY place in the row itself that visibly says
        "this row has per-patch slip", since the row's own Rt-lateral/
        Reverse cells are deliberately left untouched (see
        _refresh_row_colors()'s docstring for why leaving them at 0 is
        correct, not a bug) and would otherwise look unchanged after a
        successful slip inversion. Caller (_open_slip_inversion_dialog's
        `dlg.exec_()` branch) is expected to call this AND then trigger a
        repaint (e.g. via any subsequent _refresh_row_colors() call) so
        the row's highlighting updates too -- see COL_NAME item.setData()
        already emitting itemChanged -> _on_item_changed ->
        _refresh_row_colors() in the live widget, so this happens
        automatically."""
        item = self.table.item(row, COL_NAME)
        if item is None:
            return
        if overrides:
            pairs = [[i, j, rt, rev] for (i, j), (rt, rev) in overrides.items()]
            item.setData(Qt.UserRole, json.dumps(pairs))
            mags = [math.hypot(rt, rev) for (rt, rev) in overrides.values()]
            item.setToolTip(
                f"Distributed slip active ({len(overrides)} sub-patches): "
                f"slip magnitude min={min(mags):.3f} m, "
                f"mean={(sum(mags) / len(mags)):.3f} m, max={max(mags):.3f} m.")
        else:
            item.setData(Qt.UserRole, None)
            item.setToolTip("")

    def _open_distributed_slip_dialog(self):
        selected_rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        if len(selected_rows) != 1:
            QMessageBox.information(
                self, "Distributed slip",
                "Select exactly one row (that has Subdiv.(L)>1 or "
                "Subdiv.(W)>1) to edit its per-sub-patch slip.")
            return
        row = selected_rows[0]
        try:
            n_l = max(1, int(round(float(self.table.item(row, COL_SUBL).text()))))
            n_w = max(1, int(round(float(self.table.item(row, COL_SUBW).text()))))
            rt_lat = float(self.table.item(row, COL_RTLAT).text())
            reverse = float(self.table.item(row, COL_REVERSE).text())
        except (ValueError, AttributeError):
            QMessageBox.warning(self, "Distributed slip", "This row has an invalid number.")
            return
        if n_l * n_w <= 1:
            QMessageBox.information(
                self, "Distributed slip",
                "Set this row's Subdiv.(L) and/or Subdiv.(W) to more than "
                "1 first, then re-open this dialog to give each sub-patch "
                "its own slip.")
            return
        name = self.table.item(row, COL_NAME).text().strip() or f"Fault {row + 1}"

        from .distributed_slip_dialog import DistributedSlipDialog
        dlg = DistributedSlipDialog(
            self, name, n_l, n_w, rt_lat, reverse,
            existing_overrides=self._get_distributed_slip(row))
        if dlg.exec_():
            self._set_distributed_slip(row, dlg.get_overrides())

    def _extract_fault_spec_for_row(self, row, require_subdivision=True):
        """
        Build one {"row":, "name":, "fault":, "n_length":, "n_width":}
        dict for _open_slip_inversion_dialog() from a single table row,
        or return (None, error_message) if the row isn't usable (bad
        number, or -- when require_subdivision is True -- Subdiv.(L)/(W)
        not both/either > 1).

        require_subdivision should be False when this row is one of
        several rows being inverted JOINTLY as a group: each row is
        already its own segment/sub-patch of the bent/kinked fault, so
        a row with Subdiv.(L)=1, Subdiv.(W)=1 is still a valid single
        patch and shouldn't be rejected -- the group as a whole still
        has >1 patches to solve for.
        """
        try:
            n_l = max(1, int(round(float(self.table.item(row, COL_SUBL).text()))))
            n_w = max(1, int(round(float(self.table.item(row, COL_SUBW).text()))))
            lon = float(self.table.item(row, COL_LON).text())
            lat = float(self.table.item(row, COL_LAT).text())
            depth = float(self.table.item(row, COL_DEPTH).text())
            length = float(self.table.item(row, COL_LENGTH).text())
            width = float(self.table.item(row, COL_WIDTH).text())
            strike = float(self.table.item(row, COL_STRIKE).text())
            dip = float(self.table.item(row, COL_DIP).text())
        except (ValueError, AttributeError):
            return None, f"Row {row + 1} has an invalid number."
        if require_subdivision and n_l * n_w <= 1:
            name = self.table.item(row, COL_NAME).text().strip() or f"Fault {row + 1}"
            return None, (f"'{name}' has Subdiv.(L)={n_l}, Subdiv.(W)={n_w} -- "
                          f"set at least one to more than 1 first.")

        combo = self.table.cellWidget(row, COL_LONLAT_MODE)
        mode_label = combo.currentText() if combo is not None else None
        lon_lat_mode = LONLAT_LABEL_TO_VALUE.get(mode_label, DEFAULT_LONLAT_MODE)

        # Geometry only -- this row's own uniform slip is irrelevant to
        # the inversion (it solves for each sub-patch's slip from
        # scratch), so rt_lateral_slip/reverse_slip are passed as 0.
        fault = FaultParameters.from_input(
            lon=lon, lat=lat, depth=depth, length=length, width=width,
            strike=strike, dip=dip, lon_lat_mode=lon_lat_mode,
            rt_lateral_slip=0.0, reverse_slip=0.0)

        name = self.table.item(row, COL_NAME).text().strip() or f"Fault {row + 1}"
        return dict(row=row, name=name, fault=fault, n_length=n_l, n_width=n_w), None

    def _rows_sharing_group(self, row):
        """All row indices (in table order) that share the SAME non-empty
        Group name as `row`, including `row` itself. Empty list if `row`
        has no Group name."""
        item = self.table.item(row, COL_GROUP)
        group = item.text().strip() if item is not None else ""
        if not group:
            return []
        out = []
        for r in range(self.table.rowCount()):
            g_item = self.table.item(r, COL_GROUP)
            if g_item is not None and g_item.text().strip() == group:
                out.append(r)
        return out

    def _open_slip_inversion_dialog(self):
        """Invert imported surface-displacement observations (GNSS
        and/or InSAR) for one or more rows' per-sub-patch slip -- an
        alternate way to POPULATE the same distributed-slip overrides
        _open_distributed_slip_dialog() lets the user edit by hand.

        Target rows: whatever is currently multi-selected in the table
        (≥2 rows = a joint "Group" inversion, see
        core.okada_engine.run_slip_inversion_group()). As a convenience,
        if exactly ONE row is selected and it carries a non-empty
        "Group" name shared with other rows, the user is offered to
        expand the selection to the whole group automatically -- this
        is the intended, easiest path for the common case of "select
        one row from a group I already merged, invert for the whole
        group" rather than requiring manual multi-select every time."""
        selected_rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        if len(selected_rows) == 0:
            QMessageBox.information(
                self, "Invert for slip",
                "Select one row (or several — e.g. all rows of one "
                "'Group' — for a joint inversion) that has "
                "Subdiv.(L)>1 or Subdiv.(W)>1.")
            return

        target_rows = selected_rows
        if len(selected_rows) == 1:
            group_rows = self._rows_sharing_group(selected_rows[0])
            if len(group_rows) > 1:
                group_name = self.table.item(selected_rows[0], COL_GROUP).text().strip()
                reply = QMessageBox.question(
                    self, "Invert for slip",
                    f"This row belongs to Group '{group_name}' "
                    f"({len(group_rows)} rows). Invert JOINTLY for the "
                    f"whole group (recommended for a bent/kinked fault), "
                    f"instead of just this one row?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    target_rows = group_rows

        if self._elastic_provider is None:
            QMessageBox.warning(
                self, "Invert for slip",
                "Elastic parameters are not available (this "
                "FaultTableWidget wasn't wired to the Elastic tab via "
                "set_elastic_provider()).")
            return

        # With >1 target rows this is a JOINT group inversion: each row
        # is already one segment/patch of the (typically bent/kinked)
        # fault, so a row need not individually have Subdiv.(L)/(W)>1
        # -- only a lone single-row inversion requires that, since then
        # the row's own subdivision is the only source of multiple
        # patches to solve for.
        require_subdivision = len(target_rows) == 1

        fault_specs = []
        for row in target_rows:
            spec, err = self._extract_fault_spec_for_row(row, require_subdivision=require_subdivision)
            if err:
                QMessageBox.warning(self, "Invert for slip", err)
                return
            fault_specs.append(spec)

        if len(target_rows) > 1 and sum(s["n_length"] * s["n_width"] for s in fault_specs) <= 1:
            # Degenerate case: a "group" of just one usable patch total
            # (shouldn't normally happen since target_rows has >=2 rows,
            # but guard anyway since a 1-patch inversion is underdetermined
            # for any observation set with the usual smoothing setup).
            QMessageBox.warning(
                self, "Invert for slip",
                "The selected rows together provide only one sub-patch "
                "total -- add more rows to the group, or raise a row's "
                "Subdiv.(L)/(W), before inverting.")
            return

        from .slip_inversion_dialog import SlipInversionDialog
        dlg = SlipInversionDialog(self, fault_specs, self._elastic_provider())
        if dlg.exec_():
            overrides_by_row = dlg.get_overrides()
            spec_by_row = {s["row"]: s for s in fault_specs}
            self.table.blockSignals(True)
            for row, overrides in overrides_by_row.items():
                spec = spec_by_row.get(row)
                if (spec is not None and spec["n_length"] == 1
                        and spec["n_width"] == 1 and overrides):
                    # This row is a single, NON-subdivided patch -- the
                    # normal shape for one row of a fault_grid_builder-
                    # style variable-dip grid, where each row already
                    # IS one final patch and a group inversion is run
                    # jointly across many such rows (see
                    # _extract_fault_spec_for_row(require_subdivision=
                    # False)). For a row like this, storing the result
                    # only as a hidden distributed-slip override (as
                    # done for a genuinely-subdivided row, where the
                    # override is the ONLY way to give sub-patches
                    # different slip) leaves the row's own visible
                    # Rt-lateral/Reverse cells at 0 -- correct for
                    # get_faults()'s physics (see the 2026-08-31 fix
                    # there) but reads as "the slip wasn't applied" to
                    # anyone just looking at the table, which is exactly
                    # what was reported. Write the inverted (0,0) patch
                    # value directly into the row's own cells instead,
                    # so a group-inverted fault_grid_builder row looks
                    # and behaves exactly like a manually-entered-slip
                    # row -- visible number, no override indirection,
                    # nothing to fall out of sync with the table's own
                    # zero-slip/receiver-row coloring logic.
                    rt, rev = overrides[(0, 0)]
                    self.table.item(row, COL_RTLAT).setText(f"{rt:.6g}")
                    self.table.item(row, COL_REVERSE).setText(f"{rev:.6g}")
                    self._set_distributed_slip(row, {})  # clear any stale override
                else:
                    self._set_distributed_slip(row, overrides)
            self.table.blockSignals(False)
            self._refresh_row_colors()

    # ── Merged fault groups ─────────────────────────────────────────────

    def _merge_selected_into_group(self):
        """Assign the same Group name to every selected row, so
        get_faults() emits them as one logically-named fault
        ("GroupName-A", "GroupName-B", ...) while each row keeps its
        own independent geometry and slip (see module docstring)."""
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        if len(rows) < 2:
            QMessageBox.information(
                self, "Merge into group",
                "Select two or more rows (click a row, then Ctrl/Shift-"
                "click more) to merge them into one named group.")
            return
        default_name = self.table.item(rows[0], COL_NAME).text().strip() or "Group 1"
        text, ok = QInputDialog.getText(
            self, "Merge into group",
            "Group name — selected rows will be emitted as one logically-"
            "named fault (\"Name-A\", \"Name-B\", ...), each keeping its "
            "own geometry and slip:",
            text=default_name)
        if not ok or not text.strip():
            return
        group_name = text.strip()
        self.table.blockSignals(True)
        for r in rows:
            self.table.setItem(r, COL_GROUP, QTableWidgetItem(group_name))
        self.table.blockSignals(False)
        self._refresh_row_colors()

    def _ungroup_selected(self):
        """Clear the Group cell for every selected row, restoring each
        one's own individual name."""
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        if not rows:
            QMessageBox.information(self, "Ungroup", "Select the row(s) to ungroup first.")
            return
        self.table.blockSignals(True)
        for r in rows:
            self.table.setItem(r, COL_GROUP, QTableWidgetItem(""))
        self.table.blockSignals(False)
        self._refresh_row_colors()

    # ── Reading faults out ───────────────────────────────────────────────

    def get_faults(self, expand_subdivisions=True):
        """
        Return list of (name, FaultParameters) tuples from the table,
        each interpreted using THAT ROW'S OWN Lon/Lat-mode combo box.
        Includes ALL rows regardless of slip; use get_sources() /
        get_receiver_faults() to split them.

        Subdivided rows (Subdiv.(L)>1 or Subdiv.(W)>1) are expanded into
        named sub-patches: "Fault 1" with 3 along-strike subdivisions
        becomes "Fault 1-A", "Fault 1-B", "Fault 1-C".

        Rows sharing a non-empty "Group" value are, before any of the
        above, first renamed to that SAME "GroupName-A", "GroupName-B",
        ... convention (in table order) so they read as one logically-
        named fault; each such row's own geometry/slip is otherwise
        untouched (see module docstring / _merge_selected_into_group()).
        A group with only one member is emitted as plain "GroupName"
        (no suffix). Subdivision, if also set on a grouped row, is then
        applied on top, e.g. "GroupName-A-a".

        expand_subdivisions : if False, returns one entry per row
            regardless of subdivision settings (e.g. for drawing a single
            polygon per input row).
        """
        # Pass 1: resolve "Group" -> effective base name per row.
        group_order = {}
        for row in range(self.table.rowCount()):
            item = self.table.item(row, COL_GROUP)
            gname = item.text().strip() if item is not None else ""
            if gname:
                group_order.setdefault(gname, []).append(row)
        group_base_name = {}
        for gname, rows in group_order.items():
            if len(rows) == 1:
                group_base_name[rows[0]] = gname
            else:
                letters = _annex_labels(len(rows))
                for r, letter in zip(rows, letters):
                    group_base_name[r] = f"{gname}-{letter}"

        results = []
        for row in range(self.table.rowCount()):
            try:
                name = self.table.item(row, COL_NAME).text().strip() or f"Fault {row+1}"
                lon = float(self.table.item(row, COL_LON).text())
                lat = float(self.table.item(row, COL_LAT).text())
                depth = float(self.table.item(row, COL_DEPTH).text())
                length = float(self.table.item(row, COL_LENGTH).text())
                width = float(self.table.item(row, COL_WIDTH).text())
                strike = float(self.table.item(row, COL_STRIKE).text())
                dip = float(self.table.item(row, COL_DIP).text())
                rt_lat = float(self.table.item(row, COL_RTLAT).text())
                reverse = float(self.table.item(row, COL_REVERSE).text())
                receiver_rake = float(self.table.item(row, COL_RAKE).text())
                n_sub_l = max(1, int(round(float(self.table.item(row, COL_SUBL).text()))))
                n_sub_w = max(1, int(round(float(self.table.item(row, COL_SUBW).text()))))
            except (ValueError, AttributeError):
                continue

            combo = self.table.cellWidget(row, COL_LONLAT_MODE)
            mode_label = combo.currentText() if combo is not None else None
            lon_lat_mode = LONLAT_LABEL_TO_VALUE.get(mode_label, DEFAULT_LONLAT_MODE)

            # A row can carry a per-patch slip override even when the row
            # itself is NOT subdivided (Subdiv.(L)=Subdiv.(W)=1). This is
            # exactly the shape produced by a GROUP "Invert for slip..."
            # run over fault_grid_builder-style rows: there, each row is
            # already one fixed patch of a larger variable-dip grid, not
            # a row the user additionally subdivides, so
            # _extract_fault_spec_for_row(require_subdivision=False)
            # accepts it as a valid 1-patch spec and the inversion's
            # single (i=0, j=0) result gets stored on the row via
            # _set_distributed_slip() -- deliberately leaving the row's
            # own uniform Rt-lateral/Reverse cells at 0 (see
            # _set_distributed_slip()'s docstring). Below, `subdivide()`
            # with n_length=n_width=1 just returns [self] and never even
            # looks at slip_overrides (see its docstring), so without
            # this, such a row's real inverted slip was silently dropped
            # here and get_faults()/get_sources() saw 0 slip forever
            # -- fixed by applying the lone override directly to the
            # row's own fault instead of only through subdivide().
            overrides = self._get_distributed_slip(row) or None
            if overrides and n_sub_l <= 1 and n_sub_w <= 1:
                rt_lat, reverse = next(iter(overrides.values()))

            fault = FaultParameters.from_input(
                lon=lon, lat=lat, depth=depth, length=length, width=width,
                strike=strike, dip=dip, lon_lat_mode=lon_lat_mode,
                rt_lateral_slip=rt_lat, reverse_slip=reverse,
            )

            # Zero-slip rows are individual RECEIVER faults (see
            # get_receiver_faults()). from_rt_lat_reverse() can only ever
            # derive rake=atan2(0,0)=0 from a zero slip vector, which is
            # not a real orientation -- it's just "no slip to decompose".
            # Substitute the row's own explicit "Rake (receiver, °)" value
            # instead, so compute_cff()/compute_cff_on_receiver_faults()
            # resolve stress on the orientation the user actually meant.
            # Nonzero-slip (source) rows are untouched: their rake keeps
            # coming from rt-lateral/reverse as before.
            if abs(fault.slip) <= 1e-9:   # matches get_receiver_faults()'s default threshold
                fault.rake = receiver_rake

            base_name = group_base_name.get(row, name)

            if expand_subdivisions and (n_sub_l > 1 or n_sub_w > 1):
                patches = fault.subdivide(n_sub_l, n_sub_w, slip_overrides=overrides)
                letters = _annex_labels(len(patches))
                for patch, suffix in zip(patches, letters):
                    results.append((f"{base_name}-{suffix}", patch))
            else:
                results.append((base_name, fault))
        return results

    def get_sources(self, slip_threshold=1e-9):
        """Rows (or sub-patches) with NONZERO slip act as stress SOURCES."""
        return [f for _, f in self.get_faults() if abs(f.slip) > slip_threshold]

    def get_receiver_faults(self, slip_threshold=1e-9):
        """Rows (or sub-patches) with ZERO slip act as individual RECEIVER
        faults, each resolved on its own strike/dip/rake."""
        return [f for _, f in self.get_faults() if abs(f.slip) <= slip_threshold]

    def get_named_sources(self, slip_threshold=1e-9):
        """Same as get_sources() but keeps each fault's name/annex label."""
        return [(n, f) for n, f in self.get_faults() if abs(f.slip) > slip_threshold]

    def get_named_receiver_faults(self, slip_threshold=1e-9):
        """Same as get_receiver_faults() but keeps each fault's name/annex label."""
        return [(n, f) for n, f in self.get_faults() if abs(f.slip) <= slip_threshold]

    # ── Setup save/load (raw row data, not converted FaultParameters) ─────

    def get_raw_rows(self):
        """
        Return every row's data EXACTLY as entered (name, lonlat_mode, and
        every DATA_COLUMNS value), unconverted -- unlike get_faults(),
        which resolves each row through FaultParameters.from_input() into
        centroid form. Used by project_io.py for setup save/load, where
        we want byte-for-byte round-tripping of what the user typed, not
        the derived centroid representation.
        """
        rows = []
        for row in range(self.table.rowCount()):
            try:
                name = self.table.item(row, COL_NAME).text().strip() or f"Fault {row+1}"
                lon = float(self.table.item(row, COL_LON).text())
                lat = float(self.table.item(row, COL_LAT).text())
                depth = float(self.table.item(row, COL_DEPTH).text())
                length = float(self.table.item(row, COL_LENGTH).text())
                width = float(self.table.item(row, COL_WIDTH).text())
                strike = float(self.table.item(row, COL_STRIKE).text())
                dip = float(self.table.item(row, COL_DIP).text())
                rt_lat = float(self.table.item(row, COL_RTLAT).text())
                reverse = float(self.table.item(row, COL_REVERSE).text())
                rake = float(self.table.item(row, COL_RAKE).text())
                sub_l = float(self.table.item(row, COL_SUBL).text())
                sub_w = float(self.table.item(row, COL_SUBW).text())
            except (ValueError, AttributeError):
                continue
            group_item = self.table.item(row, COL_GROUP)
            group = group_item.text().strip() if group_item is not None else ""
            combo = self.table.cellWidget(row, COL_LONLAT_MODE)
            mode_label = combo.currentText() if combo is not None else None
            lonlat_mode = LONLAT_LABEL_TO_VALUE.get(mode_label, DEFAULT_LONLAT_MODE)
            overrides = self._get_distributed_slip(row)
            distributed_slip = [[i, j, rt, rev] for (i, j), (rt, rev) in overrides.items()]
            rows.append(dict(
                name=name, lonlat_mode=lonlat_mode, lon=lon, lat=lat, depth=depth,
                length=length, width=width, strike=strike, dip=dip,
                rt_lateral_slip=rt_lat, reverse_slip=reverse, rake=rake,
                subdiv_l=sub_l, subdiv_w=sub_w, group=group,
                distributed_slip=distributed_slip,
            ))
        return rows

    def clear_all_rows(self):
        """Remove every row (used before loading a saved setup / imported
        fault set). Leaves the table empty -- caller is responsible for
        re-populating it, since an empty table is a valid transient state
        mid-load (unlike remove_selected(), which re-adds a blank row)."""
        self.table.blockSignals(True)
        while self.table.rowCount() > 0:
            self.table.removeRow(0)
        self.table.blockSignals(False)

    def set_raw_rows(self, rows):
        """Replace all rows with the given list of dicts (same shape as
        get_raw_rows() returns). Resets the fault-name counter so
        subsequently-added rows don't collide with loaded names."""
        self.clear_all_rows()
        for r in rows:
            vals = [r["name"], r["lon"], r["lat"], r["depth"], r["length"],
                   r["width"], r["strike"], r["dip"], r["rt_lateral_slip"],
                   r["reverse_slip"], r.get("rake", 0.0),
                   r.get("subdiv_l", 1), r.get("subdiv_w", 1), r.get("group", "")]
            self.add_row(vals, lonlat_mode=r.get("lonlat_mode", DEFAULT_LONLAT_MODE),
                        distributed_slip=r.get("distributed_slip"))
        if self.table.rowCount() == 0:
            self.add_row()
        self._fault_counter = self.table.rowCount() + 1

    # ── Scaling relations calculator ─────────────────────────────────────

    def _open_scaling_dialog(self):
        from .scaling_dialog import ScalingRelationsDialog
        dlg = ScalingRelationsDialog(self)
        if dlg.exec_():
            length, width, rt_lat, reverse = dlg.get_result()
            if length is None:
                return
            selected_rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
            row = selected_rows[0] if selected_rows else self.table.rowCount() - 1
            if row < 0:
                self.add_row()
                row = 0

            self.table.setItem(row, COL_LENGTH, QTableWidgetItem(f"{length:.4g}"))
            self.table.setItem(row, COL_WIDTH, QTableWidgetItem(f"{width:.4g}"))
            self.table.setItem(row, COL_RTLAT, QTableWidgetItem(f"{rt_lat:.4g}"))
            self.table.setItem(row, COL_REVERSE, QTableWidgetItem(f"{reverse:.4g}"))

    # ── Polyline import ───────────────────────────────────────────────────

    def _open_polyline_import_dialog(self):
        """
        Import fault segments from a QGIS line layer. The dialog itself
        asks whether the digitized line represents the Fault Top
        Projection or the (geological) Surface Trace, and applies the
        down-dip correction accordingly (see utils/polyline_import.py) --
        by the time rows come back here, lon1/lat1/lon2/lat2 are always
        already the TOP EDGE's own coordinates. So every imported row's
        Lon/Lat mode is set to "top_start" directly on that row (not via
        any table-wide toggle), leaving every other existing row's mode
        untouched.
        """
        from .polyline_import_dialog import PolylineImportDialog
        dlg = PolylineImportDialog(self)
        if not dlg.exec_():
            return

        rows = dlg.get_rows()
        if not rows:
            return

        for row in rows:
            vals = [f"Fault {self._fault_counter}", row["lon1"], row["lat1"],
                   row["top_depth_km"], row["length_km"], row["width_km"],
                   row["strike_deg"], row["dip_deg"],
                   row["rt_lateral_slip_m"], row["reverse_slip_m"], 0.0, 1, 1, ""]
            self._fault_counter += 1
            self.add_row(vals, lonlat_mode="top_start")

    def _open_fault_table_import_dialog(self):
        """
        Import fault patches from an external distributed-slip table
        (e.g. a GSI/geodetic-inversion .dat file, or any delimited
        table with column-mappable geometry+slip). Unlike polyline
        import, geometry here comes fully formed from the source table
        (no down-dip trace correction needed) and depth is ALWAYS
        already resolved to centroid depth by
        core.fault_table_import.build_fault_rows_from_mapped_rows()
        before it gets here -- regardless of whether the source table's
        own depth column was top-edge or centroid (that conversion, if
        needed, already happened via
        core.okada_engine.FaultParameters.from_rt_lat_reverse()). So
        every imported row's Lon/Lat mode is set to "centroid", not
        inherited from any table-wide default.
        """
        from .fault_table_import_dialog import FaultTableImportDialog
        dlg = FaultTableImportDialog(self)
        if not dlg.exec_():
            return

        rows = dlg.imported_rows
        if not rows:
            return

        for row in rows:
            vals = [row.get("name") or f"Fault {self._fault_counter}",
                   row["lon"], row["lat"], row["depth_km"],
                   row["length_km"], row["width_km"], row["strike"], row["dip"],
                   row["rt_lateral_slip_m"], row["reverse_slip_m"],
                   row["rake_deg"], 1, 1, row.get("group") or ""]
            self._fault_counter += 1
            self.add_row(vals, lonlat_mode=row.get("lonlat_mode", "centroid"))

    def _open_fault_grid_builder_dialog(self):
        """
        Generate a grid of independent, fixed-size fault sub-patches
        (e.g. 1 km x 1 km) with per-row-variable dip -- see
        ui.fault_grid_builder_dialog.FaultGridBuilderDialog and
        core.fault_grid_builder for the construction itself. Unlike
        "Import fault-patch table…", nothing is read from a file: the
        dialog builds the rows itself from an on-screen spec (start
        point, strike, fixed patch length, and one width+dip per
        down-dip row). Every generated row's Lon/Lat mode is set to
        "centroid" -- same reasoning as _open_fault_table_import_dialog
        above, since core.fault_grid_builder already resolves each
        patch's own centroid via FaultParameters.from_input() before
        it gets here.
        """
        from .fault_grid_builder_dialog import FaultGridBuilderDialog
        dlg = FaultGridBuilderDialog(self)
        if not dlg.exec_():
            return

        rows = dlg.built_rows
        if not rows:
            return

        for row in rows:
            vals = [row.get("name") or f"Fault {self._fault_counter}",
                   row["lon"], row["lat"], row["depth_km"],
                   row["length_km"], row["width_km"], row["strike"], row["dip"],
                   row["rt_lateral_slip_m"], row["reverse_slip_m"],
                   row["rake_deg"], 1, 1, row.get("group") or ""]
            self._fault_counter += 1
            self.add_row(vals, lonlat_mode=row.get("lonlat_mode", "centroid"))


def _annex_labels(n):
    """A, B, C, ..., Z, AA, AB, ... for n sub-patch labels."""
    labels = []
    for i in range(n):
        label = ""
        k = i
        while True:
            label = chr(ord('A') + k % 26) + label
            k = k // 26 - 1
            if k < 0:
                break
        labels.append(label)
    return labels
