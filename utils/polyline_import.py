
"""
2026-08-22 addition: faults_from_line_layer() was referenced by
ui.polyline_import_dialog.PolylineImportDialog._do_import() (and
documented in profile_vertices_from_line_layer()'s own docstring
below, as the contrast case) but was never actually defined in this
file -- a missing/stale-function bug of the same class flagged in
PROJECT_HANDOVER_ADDENDUM_2026-08-08c_stale_cache_stop.md, causing
"Import Fault from QGIS Polyline" to raise ImportError at runtime.
Implemented here now.
"""

import numpy as np

from ..core.okada_engine import geo_to_km, km_to_geo


def faults_from_line_layer(layer, default_top_depth=0.0, default_dip=90.0,
                            default_rt_lateral_slip=0.0, default_reverse_slip=0.0,
                            default_width=10.0, only_selected=True,
                            line_represents="top_edge"):
    """
    Turn EACH segment (pair of consecutive vertices) of a digitized
    line layer into one fault row dict, ready for
    ui.fault_table_widget.FaultTableWidget._open_polyline_import_dialog()
    to hand to add_row(). Unlike profile_vertices_from_line_layer()
    (which returns a single continuous vertex chain for a
    cross-section profile), every segment here becomes an
    INDEPENDENT fault row -- a digitized fault trace with several
    vertices is treated as several separate fault segments, each
    with its own length/strike computed from its own endpoints.

    layer : QGIS vector line layer.
    default_top_depth : km, applied to every row's "top_depth_km".
    default_dip : degrees (0-90), applied to every row's "dip_deg",
        and used to compute the down-dip correction when
        line_represents == "surface_trace".
    default_rt_lateral_slip, default_reverse_slip : m, applied to
        every row unchanged (this function only derives geometry
        from the digitized line -- slip is a UI default, edited
        afterward in the fault table).
    default_width : km, applied to every row's "width_km" (width is
        not recoverable from a 2D digitized line, only length/strike
        are).
    only_selected : use only selected features if True, else every
        feature in the layer.
    line_represents : "top_edge" (line already traces the fault's
        actual top edge -- no correction) or "surface_trace" (line
        traces where the fault plane extrapolates to z=0 -- shift
        each segment's start point down-dip by
        default_top_depth/tan(default_dip) to recover the true top
        edge, per this module's down-dip convention: for strike
        theta (degrees clockwise from North, matching
        core.geo_profile.polyline_segment_info's azimuth
        convention), the down-dip horizontal unit vector in
        (East, North) is (cos(theta), -sin(theta)) -- the same
        strike -> corner-offset convention used by
        core.fault_geometry.fault_corners_geo()).

    Returns a list of dicts, each with keys: lon1, lat1,
    top_depth_km, length_km, width_km, strike_deg, dip_deg,
    rt_lateral_slip_m, reverse_slip_m -- lon1/lat1 are always the
    fault's TOP EDGE start point (already down-dip-corrected if
    line_represents == "surface_trace"), so callers should treat
    the row's Lon/Lat mode as "top_start". Empty list if no usable
    segments were found (not None, unlike
    profile_vertices_from_line_layer -- a polyline import can
    legitimately yield zero, one, or many fault rows).
    """
    apply_correction = (line_represents == "surface_trace"
                        and default_top_depth > 0
                        and abs(default_dip - 90.0) > 1e-6)
    if apply_correction:
        shift_km = default_top_depth / np.tan(np.deg2rad(default_dip))
    else:
        shift_km = 0.0

    rows = []
    features = layer.selectedFeatures() if only_selected else layer.getFeatures()
    for feat in features:
        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            continue
        parts = geom.asMultiPolyline() if geom.isMultipart() else [geom.asPolyline()]
        for part in parts:
            vertices = [(pt.x(), pt.y()) for pt in part]
            for (lon1, lat1), (lon2, lat2) in zip(vertices[:-1], vertices[1:]):
                x2, y2 = geo_to_km(lon2, lat2, lon1, lat1)
                length_km = float(np.hypot(x2, y2))
                if length_km < 1e-6:
                    continue  # coincident vertices -- skip degenerate segment
                strike_deg = float(np.degrees(np.arctan2(x2, y2)) % 360.0)

                seg_lon1, seg_lat1 = lon1, lat1
                if shift_km != 0.0:
                    theta = np.deg2rad(strike_deg)
                    de_km = shift_km * np.cos(theta)
                    dn_km = -shift_km * np.sin(theta)
                    seg_lon1, seg_lat1 = km_to_geo(de_km, dn_km, lon1, lat1)

                rows.append({
                    "lon1": float(seg_lon1),
                    "lat1": float(seg_lat1),
                    "top_depth_km": float(default_top_depth),
                    "length_km": length_km,
                    "width_km": float(default_width),
                    "strike_deg": strike_deg,
                    "dip_deg": float(default_dip),
                    "rt_lateral_slip_m": float(default_rt_lateral_slip),
                    "reverse_slip_m": float(default_reverse_slip),
                })
    return rows


def profile_vertices_from_line_layer(layer, only_selected=True):
    """
    2026-08-21 addition, cross-section profile line import ("option
    cross section profile line can be imported as a qgis polyline" --
    request item 5). Unlike faults_from_line_layer() (which turns EACH
    segment into a separate fault row), a cross-section profile is ONE
    continuous polyline -- so this returns the vertex chain of a
    single feature as-is: [(lon, lat), (lon, lat), ...], length >= 2,
    ready to hand to core.okada_engine.compute_cross_section_multi() /
    core.optimal_plane.compute_cross_section_optimal_multi().

    Uses the FIRST usable feature only (first selected feature if
    only_selected, else the first feature in the layer) -- a
    cross-section profile is a single line, not a set of independent
    ones the way digitized fault traces are. If that feature is
    multipart, only its first part is used.

    Returns None if no usable feature/geometry was found.
    """
    features = layer.selectedFeatures() if only_selected else layer.getFeatures()
    for feat in features:
        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            continue
        part = geom.asMultiPolyline()[0] if geom.isMultipart() else geom.asPolyline()
        vertices = [(pt.x(), pt.y()) for pt in part]
        if len(vertices) >= 2:
            return vertices
    return None
