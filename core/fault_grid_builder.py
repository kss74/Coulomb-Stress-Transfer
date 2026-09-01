# -*- coding: utf-8 -*-
"""
Construct a grid of INDEPENDENT fault sub-patches from a compact spec,
rather than from a table of individually-typed patches or from
FaultParameters.subdivide() (which only supports a single uniform dip
across the whole parent fault, splitting its existing length/width by
n_length/n_width).

This module supports the opposite construction direction: the user
fixes each sub-patch's own LENGTH and WIDTH (e.g. 1 km x 1 km, matching
typical geodetic-inversion fault-patch grids such as GSI/Kobayashi et
al. 2018's dataset), picks how many columns (along strike) and rows
(down-dip) to generate, and gives EACH ROW its own dip (and,
optionally, its own width) -- producing a segmented/listric fault
plane whose dip changes with depth, which FaultParameters.subdivide()
cannot represent.

Row stacking is CONTINUOUS: row i+1's top edge is exactly row i's
bottom edge (no gap, no overlap) -- i.e. depth and down-dip horizontal
position accumulate as
    top_depth_{i+1}    = top_depth_i    + width_i * sin(dip_i)
    down_dip_horiz_{i+1} = down_dip_horiz_i + width_i * cos(dip_i)
This matches (and was verified against) the row-to-row stepping in the
GSI Kobayashi et al. (2018) Northern Nagano fault-patch table: e.g.
successive along-dip rows step by exactly width*sin(dip) in centroid
depth (see PROJECT_HANDOVER_ADDENDUM_2026-08-28_insar_raster_import.md
/ fault_table_import.py's depth_convention="centroid" docstring for
the same observation from the import side).

Each patch is placed via FaultParameters.from_input(..., lon_lat_mode=
"top_start"), the SAME constructor used elsewhere in the plugin for
top-edge-start-point geometry (polyline import, fault_table_widget
rows) -- this module only computes each row's own top-edge start point
and hands the per-patch trig off to that single shared implementation,
rather than re-deriving centroid placement here.

down-dip horizontal direction convention (dip_dir = strike + 90 degrees,
E = horiz*sin(dip_dir), N = horiz*cos(dip_dir)) matches
FaultParameters.from_top_center()/.top_center()/.from_surface_trace().
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np

from .okada_engine import FaultParameters, km_to_geo


@dataclass
class FaultGridRowSpec:
    """One down-dip row's own width and dip. width_km applies to every
    patch in this row; dip_deg likewise. Rows are given top-to-bottom
    (row 0 = shallowest)."""
    width_km: float
    dip_deg: float


@dataclass
class FaultGridResult:
    rows: List[dict] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    bottom_depth_km: float = 0.0  # bottom edge depth of the deepest row, for a UI preview


def build_variable_dip_fault_grid(
    start_lon: float,
    start_lat: float,
    top_depth_km: float,
    strike_deg: float,
    n_cols: int,
    patch_length_km: float,
    row_specs: List[FaultGridRowSpec],
    rake_deg: float = 0.0,
    slip_m: float = 0.0,
    name_prefix: str = "Patch",
    group: Optional[str] = None,
) -> FaultGridResult:
    """
    Build n_cols * len(row_specs) independent fault-patch rows, ready
    for ui.fault_table_widget.FaultTableWidget.add_row() (same output
    dict shape as core.fault_table_import.build_fault_rows_from_mapped_rows()):
        {"name", "lon", "lat", "depth_km", "length_km", "width_km",
         "strike", "dip", "rt_lateral_slip_m", "reverse_slip_m",
         "rake_deg", "lonlat_mode", "group"}

    start_lon, start_lat : the TOP-EDGE START POINT of row 0, column 0
        (the shallowest, first-along-strike patch's top-left corner --
        same reference point convention as lon_lat_mode="top_start"
        elsewhere in the plugin).
    top_depth_km  : top-edge depth of row 0 (km, positive down).
    strike_deg    : constant along the whole grid (this module builds
        a straight-in-map-view, listric-in-cross-section fault; a
        along-strike-curving trace is out of scope -- digitize a
        polyline per bend instead and Group them, as elsewhere in the
        plugin).
    n_cols        : number of along-strike subdivisions (>=1).
    patch_length_km : fixed along-strike length of every patch.
    row_specs     : down-dip rows, top-to-bottom, each with its own
        width_km and dip_deg (>=1 row). A single-row list with uniform
        width/dip reduces to a plain rectangular grid.
    rake_deg, slip_m : uniform initial rake/slip applied to every
        patch (default 0/0 -- this tool builds GEOMETRY; slip is
        normally filled in afterward via the fault table itself or a
        slip inversion, matching the "Import fault-patch table" and
        "Import from QGIS polyline" tools' own default-then-edit
        pattern).
    name_prefix, group : naming. If `group` is given, every patch
        shares it, so the table's own "Merge selected into group" /
        Group-column mechanism reads/exports them as one logically-
        named fault ("group-A", "group-B", ...) while each patch keeps
        its own independent geometry.
    """
    n_cols = max(1, int(n_cols))
    if not row_specs:
        raise ValueError("row_specs must contain at least one row")
    if patch_length_km <= 0:
        raise ValueError("patch_length_km must be > 0")

    strike_rad = np.deg2rad(strike_deg)
    dip_dir_rad = np.deg2rad(strike_deg + 90.0)

    # Row 0's own top-edge start point, in local (E, N) km relative to
    # itself -- (0, 0) by construction; converted to lon/lat per row below.
    row_e, row_n = 0.0, 0.0
    row_top_depth = float(top_depth_km)

    out_rows = []
    for i, spec in enumerate(row_specs):
        width_km = float(spec.width_km)
        dip_deg = float(spec.dip_deg)
        if width_km <= 0:
            raise ValueError(f"row {i + 1}: width_km must be > 0")
        if not (0.0 < dip_deg <= 90.0):
            raise ValueError(f"row {i + 1}: dip_deg must be in (0, 90]")

        for j in range(n_cols):
            # This patch's own top-edge start point: the row's start
            # point, shifted along strike by patch_length_km * j.
            p_e = row_e + (patch_length_km * j) * np.sin(strike_rad)
            p_n = row_n + (patch_length_km * j) * np.cos(strike_rad)
            p_lon, p_lat = km_to_geo(p_e, p_n, start_lon, start_lat)

            fault = FaultParameters.from_input(
                lon=p_lon, lat=p_lat, depth=row_top_depth,
                length=patch_length_km, width=width_km,
                strike=strike_deg, dip=dip_deg,
                lon_lat_mode="top_start",
                rake=rake_deg, slip=slip_m,
            )

            out_rows.append({
                "name": f"{name_prefix} {i + 1}-{j + 1}",
                "lon": fault.lon, "lat": fault.lat, "depth_km": fault.depth,
                "length_km": patch_length_km, "width_km": width_km,
                "strike": strike_deg, "dip": dip_deg,
                "rt_lateral_slip_m": fault.rt_lateral_slip,
                "reverse_slip_m": fault.reverse_slip,
                "rake_deg": rake_deg,
                "lonlat_mode": "centroid",
                "group": group,
            })

        # Advance to the NEXT row's own top-edge start point: this row's
        # bottom edge becomes the next row's top edge (continuous, no
        # gap/overlap) -- see module docstring for the depth-stepping
        # derivation this was checked against.
        horiz_step = width_km * np.cos(np.deg2rad(dip_deg))
        row_e += horiz_step * np.sin(dip_dir_rad)
        row_n += horiz_step * np.cos(dip_dir_rad)
        row_top_depth += width_km * np.sin(np.deg2rad(dip_deg))

    return FaultGridResult(rows=out_rows, n_rows=len(row_specs), n_cols=n_cols,
                           bottom_depth_km=row_top_depth)
