# -*- coding: utf-8 -*-
"""
Read-only results table for ΔCFF resolved on focal-mechanism nodal
planes, plus the tab-building helper that wires it together with the
import dialog and a compute button — mirrors
receiver_results_widget.py's pattern, extended for two planes and a
mode selector.

Beachball rendering (Coulomb 3.4.2's color-coded focal-sphere display)
is NOT implemented here yet — see PROJECT_HANDOVER addendum for this
session for the plan (needs its own review of coulomb.m's `bb2`
function before implementation). This widget/tab is fully usable
without it: the results table is the primary deliverable, matching
what Coulomb itself shows in its "nodal plane" results list.
"""

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QLabel, QComboBox
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor

from ..core.focal_mechanism import PLANE_MODES
from ..core.scaling_relations import SCALING_RELATIONS, FAULT_STYLES

RESULT_COLUMNS = [
    "Event", "Lon", "Lat", "Depth (km)", "Mag",
    "Strike1", "Dip1", "Rake1", "ΔCFF1 (bar)",
    "Strike2", "Dip2", "Rake2", "ΔCFF2 (bar)",
    "Selected", "ΔCFF (bar)", "Method",
    "Use as Source", "Source Plane",
]

# Column indices for the two source-designation columns appended after the
# original 16 result columns (kept at the end so nothing above has to
# renumber; see build_source_fault_row() usage in main_dialog.py).
COL_USE_AS_SOURCE = 16
COL_SOURCE_PLANE = 17

MODE_LABELS = {
    "plane1": "Always nodal plane 1",
    "plane2": "Always nodal plane 2",
    "max": "Max ΔCFF (more destabilizing plane)",
    "min": "Min ΔCFF",
    "random": "Random (Monte Carlo ambiguity)",
}


class FocalMechanismResultsWidget(QTableWidget):
    """
    Displays one row per event. Before ΔCFF has been computed (right after
    import), the ΔCFF/Selected/Method columns show "—" (see set_events());
    after compute_focal_mechanisms_action() runs, set_results() fills them
    in. Either way, every row also carries a "Use as Source" checkbox and a
    "Source Plane" (Plane 1 / Plane 2) picker, independent of ΔCFF — an
    event can be designated a stress SOURCE (see
    core.focal_mechanism.build_source_fault_row() /
    main_dialog.add_focal_mechanisms_as_sources_action()) without ever
    computing ΔCFF on it as a receiver, since those are two unrelated uses
    of the same imported geometry.
    """

    def __init__(self, parent=None):
        super().__init__(0, len(RESULT_COLUMNS), parent)
        self.setHorizontalHeaderLabels(RESULT_COLUMNS)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self._events = []  # row index -> FocalMechanismEvent, parallel to table rows

    def set_events(self, events):
        """
        Populate the table directly from imported events, before any ΔCFF
        computation has run (ΔCFF/Selected/Method columns show "—"). Lets
        the user tick "Use as Source" and pick a nodal plane immediately
        after import, without first needing source faults of their own to
        compute a receiver ΔCFF against.
        """
        self._rebuild(events, results_by_event=None)

    def set_results(self, results):
        """
        results: list of dicts as returned by
        core.focal_mechanism.compute_focal_mechanism_cff().
        """
        self._rebuild([res["event"] for res in results],
                      results_by_event=results)

    def _rebuild(self, events, results_by_event):
        self.setRowCount(0)
        self._events = list(events)
        for i, ev in enumerate(self._events):
            res = results_by_event[i] if results_by_event is not None else None
            p1 = res["plane1"] if res else None
            p2 = res["plane2"] if res else None
            row = self.rowCount()
            self.insertRow(row)

            if res is not None:
                cff_bar = res["cff_mpa"] * 10
                cff1_bar = p1["cff_mpa"] * 10
                cff2_bar = p2["cff_mpa"] * 10 if p2 else None
                method = "DC3D" if res["used_dc3d"] else "Surface (z=0)"
                selected = res["selected"]
            else:
                cff_bar = cff1_bar = cff2_bar = None
                method = selected = "—"

            has_plane2 = ev.has_both_planes()
            vals = [
                ev.label or "—",
                f"{ev.lon:.5f}", f"{ev.lat:.5f}", f"{ev.depth:.3f}",
                f"{ev.magnitude:.2f}" if ev.magnitude is not None else "—",
                f"{ev.strike1:.1f}", f"{ev.dip1:.1f}", f"{ev.rake1:.1f}",
                (f"{cff1_bar:+.5f}" if cff1_bar is not None else "—"),
                (f"{ev.strike2:.1f}" if has_plane2 else "—"),
                (f"{ev.dip2:.1f}" if has_plane2 else "—"),
                (f"{ev.rake2:.1f}" if has_plane2 else "—"),
                (f"{cff2_bar:+.5f}" if cff2_bar is not None else "—"),
                selected,
                (f"{cff_bar:+.5f}" if cff_bar is not None else "—"),
                method,
            ]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 14 and cff_bar is not None:  # selected ΔCFF column: color by sign
                    if cff_bar > 0:
                        item.setBackground(QColor(255, 210, 210))
                    elif cff_bar < 0:
                        item.setBackground(QColor(210, 220, 255))
                self.setItem(row, col, item)

            use_item = QTableWidgetItem()
            use_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            use_item.setCheckState(Qt.Unchecked)
            use_item.setTextAlignment(Qt.AlignCenter)
            if ev.magnitude is None:
                use_item.setFlags(Qt.NoItemFlags)
                use_item.setToolTip(
                    "No magnitude on this event — required to estimate "
                    "length/width/slip via scaling relations, so it can't "
                    "be used as a source fault.")
            self.setItem(row, COL_USE_AS_SOURCE, use_item)

            plane_combo = QComboBox()
            plane_combo.addItem("Plane 1")
            if has_plane2:
                plane_combo.addItem("Plane 2")
            self.setCellWidget(row, COL_SOURCE_PLANE, plane_combo)

    def get_source_selections(self):
        """
        Returns a list of (FocalMechanismEvent, plane_int) for every row
        whose "Use as Source" checkbox is checked, plane_int being 1 or 2
        per that row's "Source Plane" combo. Used by
        main_dialog.add_focal_mechanisms_as_sources_action().
        """
        out = []
        for row, ev in enumerate(self._events):
            item = self.item(row, COL_USE_AS_SOURCE)
            if item is None or item.checkState() != Qt.Checked:
                continue
            combo = self.cellWidget(row, COL_SOURCE_PLANE)
            plane = 2 if (combo is not None and combo.currentText() == "Plane 2") else 1
            out.append((ev, plane))
        return out


def build_focal_mechanism_tab(main_dialog):
    """
    Builds the "Focal Mechanisms" tab and attaches it to main_dialog.tabs.
    Call from CoulombMainDialog.__init__ alongside the other _build_*_tab()
    calls. Stores state on main_dialog:
        main_dialog.focal_events        -- list[FocalMechanismEvent]
        main_dialog.focal_results_widget
        main_dialog.focal_mode_combo
    Wires two new buttons to main_dialog methods that don't exist yet --
    import_focal_mechanisms_action() and compute_focal_mechanisms_action()
    -- add those to CoulombMainDialog (see PROJECT_HANDOVER addendum for
    a drop-in implementation using ComputeWorker's existing "focal_mech"
    mode, which still needs to be added to ComputeWorker.run() alongside
    the existing "receiver_faults"/"optimal" branches).
    """
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.addWidget(QLabel(
        "<b>Stress on Focal Mechanisms</b><br><i>Import a catalog of "
        "earthquake focal mechanisms (nodal planes or moment tensors) "
        "and resolve ΔCFF from the Source Faults table onto each "
        "event's own nodal plane(s) — matching Coulomb 3.4.2's "
        "'Calc. stress on nodal planes' feature.</i>"))

    main_dialog.focal_events = []

    btn_row = QHBoxLayout()
    main_dialog.btn_import_focal = QPushButton("Import Focal Mechanisms…")
    main_dialog.btn_import_focal.clicked.connect(main_dialog.import_focal_mechanisms_action)
    btn_row.addWidget(main_dialog.btn_import_focal)

    btn_row.addWidget(QLabel("Nodal-plane mode:"))
    main_dialog.focal_mode_combo = QComboBox()
    for key in PLANE_MODES:
        main_dialog.focal_mode_combo.addItem(MODE_LABELS[key], userData=key)
    main_dialog.focal_mode_combo.setCurrentIndex(list(PLANE_MODES).index("max"))
    btn_row.addWidget(main_dialog.focal_mode_combo)
    layout.addLayout(btn_row)

    main_dialog.lbl_focal_status = QLabel("No focal mechanisms imported yet.")
    layout.addWidget(main_dialog.lbl_focal_status)

    main_dialog.focal_results_widget = FocalMechanismResultsWidget()
    layout.addWidget(main_dialog.focal_results_widget)

    layout.addWidget(QLabel(
        "<b>Use imported mechanisms as source faults</b><br><i>Tick "
        "'Use as Source' and pick a nodal plane above for one or more "
        "events (any number), then add them below as rows in the "
        "<b>Source Faults</b> table — this works whether or not you've "
        "computed ΔCFF on them as receivers. A focal mechanism only gives "
        "point geometry + orientation, never fault length/width/slip, so "
        "those are estimated from each event's own magnitude using the "
        "scaling relation and fault style chosen here (same calculation "
        "as the 'Estimate L/W/slip from magnitude' dialog); the slip's "
        "right-lateral/reverse split always comes from the chosen plane's "
        "own rake. Added rows use Lon/Lat mode 'Centroid' and are named "
        "'&lt;event&gt; (FM NP1/2)' so they're easy to spot in that "
        "table.</i>"))

    source_row = QHBoxLayout()
    source_row.addWidget(QLabel("Scaling relation:"))
    main_dialog.focal_source_relation_combo = QComboBox()
    main_dialog.focal_source_relation_combo.addItems(list(SCALING_RELATIONS.keys()))
    source_row.addWidget(main_dialog.focal_source_relation_combo)

    source_row.addWidget(QLabel("Fault style:"))
    main_dialog.focal_source_style_combo = QComboBox()
    main_dialog.focal_source_style_combo.addItems(FAULT_STYLES)
    source_row.addWidget(main_dialog.focal_source_style_combo)

    main_dialog.btn_add_focal_as_source = QPushButton(
        "→ Add Selected as Source Fault(s)")
    main_dialog.btn_add_focal_as_source.clicked.connect(
        main_dialog.add_focal_mechanisms_as_sources_action)
    source_row.addWidget(main_dialog.btn_add_focal_as_source)
    layout.addLayout(source_row)

    compute_row = QHBoxLayout()
    main_dialog.btn_compute_focal = QPushButton("▶  Compute ΔCFF on Focal Mechanisms")
    main_dialog.btn_compute_focal.clicked.connect(main_dialog.compute_focal_mechanisms_action)
    compute_row.addWidget(main_dialog.btn_compute_focal)

    main_dialog.btn_show_focal_on_map = QPushButton("Preview Beachballs")
    main_dialog.btn_show_focal_on_map.setToolTip(
        "Renders into this dialog's own preview panel only — not the "
        "QGIS project map. Use 'Export Beachballs to QGIS Map' to add "
        "a real map layer.")
    main_dialog.btn_show_focal_on_map.clicked.connect(
        main_dialog.show_focal_mechanisms_on_map_action)
    compute_row.addWidget(main_dialog.btn_show_focal_on_map)

    main_dialog.btn_export_focal_layer = QPushButton("Export Beachballs to QGIS Map")
    main_dialog.btn_export_focal_layer.setToolTip(
        "Adds a real polygon layer to the QGIS project — literal "
        "beachball glyphs, colored by ΔCFF, visible on the actual map "
        "canvas (not just this dialog). Requires the 'obspy' Python "
        "package.")
    main_dialog.btn_export_focal_layer.clicked.connect(
        main_dialog.export_focal_mechanisms_layer)
    compute_row.addWidget(main_dialog.btn_export_focal_layer)

    layout.addLayout(compute_row)

    main_dialog.tabs.addTab(tab, "Focal Mechanisms")
