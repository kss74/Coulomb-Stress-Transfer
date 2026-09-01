# -*- coding: utf-8 -*-
"""
Pure fault-footprint geometry, with no ui/ or qgis dependency.

fault_corners_geo() is the same computation as
ui.vector_utils._fault_corners_geo() (kept there, unchanged, for the
map-view polygon layer it draws) -- duplicated here rather than having
core.cross_section_faults import from ui.vector_utils, which would
invert this project's core/ (no ui dependency) -> ui/ (depends on
core) layering. Per project convention, existing validated modules
(vector_utils.py) are touched only when strictly necessary; this is a
new file, not a change to that one.
"""

import numpy as np

from .okada_engine import km_to_geo


def fault_corners_geo(fault, min_horiz_width_km=0.05):
    """
    4 corners (lon,lat) of a fault's horizontal-projection rectangle:
    top-left, top-right, bottom-right, bottom-left (top = up-dip edge
    at fault.top_depth, bottom = down-dip edge at fault.bottom_depth).
    See ui.vector_utils._fault_corners_geo() for the full derivation
    notes (near-vertical-dip minimum-width handling, corner ordering).
    """
    strike = np.deg2rad(fault.strike)
    dip = np.deg2rad(fault.dip)
    cs, ss = np.cos(strike), np.sin(strike)
    cd = np.cos(dip)

    half_L = fault.length / 2
    w_horiz = fault.width * cd
    if abs(w_horiz) < min_horiz_width_km:
        w_horiz = min_horiz_width_km if w_horiz >= 0 else -min_horiz_width_km
    half_w = w_horiz / 2.0

    corners_local = [(-half_L, -half_w), (half_L, -half_w), (half_L, half_w), (-half_L, half_w)]
    corners_geo = []
    for lx, ly in corners_local:
        e_km = lx * ss + ly * cs
        n_km = lx * cs - ly * ss
        lon, lat = km_to_geo(e_km, n_km, fault.lon, fault.lat)
        corners_geo.append((float(lon), float(lat)))
    return corners_geo
