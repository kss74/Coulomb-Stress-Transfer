# -*- coding: utf-8 -*-
"""
Project source-fault rectangles onto a cross-section's vertical plane.

Replicates coulomb.m's CSC_fault_int_sec() (ground truth, see
PROJECT_HANDOVER.md convention): the fault's rectangular footprint,
in map view, has a TOP edge (up-dip, at fault.top_depth) and a BOTTOM
edge (down-dip, at fault.bottom_depth) -- exactly the horizontal-
projection rectangle already computed by ui.vector_utils._fault_corners_geo()
for the map-view fault polygon layer, reused here rather than
re-deriving it a second time. The cross-section TRACE of the fault is
found by intersecting the (infinite) cross-section line with each of
the rectangle's 4 edges in 2D map coordinates, keeping only
intersections that fall within both the section segment and the fault
edge segment, then assigning each kept intersection a depth: exact
fault.top_depth / fault.bottom_depth for the top/bottom edges, and a
linear interpolation between them for the two down-dip side edges.

A fault whose rectangle doesn't cross the section line at all (i.e.
fewer than 2 valid intersections) contributes no trace -- it's simply
too far from the profile to appear on it, same as coulomb.m's own
"select two proper points" screening.
"""

import numpy as np

from .okada_engine import geo_to_km


def _segment_intersection_2d(p0, p1, q0, q1):
    """
    Intersection of infinite line through p0->p1 with infinite line
    through q0->q1, returned as (t, u, point) where t/u are the
    parametric positions along each segment (0..1 = within the
    segment); point is (x, y) or None if the lines are parallel.
    """
    (x1, y1), (x2, y2) = p0, p1
    (x3, y3), (x4, y4) = q0, q1
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None, None, None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    return t, u, (px, py)


def project_fault_traces_onto_section(sources, lon1, lat1, lon2, lat2,
                                       half_width_km=None):
    """
    Returns a list of dicts, one per source fault that actually crosses
    the section line:
        {
          "fault": <FaultParameters>,
          "dist_km": [d0, d1],     # along-profile distance of the 2 trace points
          "depth_km": [z0, z1],    # depth (km, positive down) of the 2 trace points
          "label": "F1",           # 1-based index into `sources`, for labeling
        }

    half_width_km, if given, additionally drops faults whose CENTROID
    perpendicular distance from the profile exceeds it (a cheap
    pre-filter matching FaultOverlayConfig.search_width_km; the actual
    trace geometry above is unaffected by this -- it's map-view exact
    regardless of how far off-profile the fault sits).
    """
    from .fault_geometry import fault_corners_geo as _fault_corners_geo

    x0, y0 = 0.0, 0.0
    x1, y1 = geo_to_km(lon2, lat2, lon1, lat1)
    profile_len = float(np.hypot(x1 - x0, y1 - y0))

    traces = []
    for i, fault in enumerate(sources):
        if half_width_km is not None:
            fx, fy = geo_to_km(fault.lon, fault.lat, lon1, lat1)
            ux, uy = (x1 - x0) / profile_len, (y1 - y0) / profile_len
            perp = fx * uy - fy * ux
            if abs(perp) > half_width_km:
                continue

        corners_geo = _fault_corners_geo(fault)  # top-left, top-right, bottom-right, bottom-left
        corners_km = [geo_to_km(lon, lat, lon1, lat1) for lon, lat in corners_geo]
        top_left, top_right, bottom_right, bottom_left = corners_km

        edges = [
            (top_left, top_right, fault.top_depth, fault.top_depth),      # top edge
            (bottom_left, bottom_right, fault.bottom_depth, fault.bottom_depth),  # bottom edge
            (top_left, bottom_left, fault.top_depth, fault.bottom_depth),        # side 1
            (top_right, bottom_right, fault.top_depth, fault.bottom_depth),      # side 2
        ]

        hits = []
        for p_start, p_end, z_start, z_end in edges:
            t_sec, u_edge, pt = _segment_intersection_2d(
                (x0, y0), (x1, y1), p_start, p_end)
            if pt is None:
                continue
            if not (0.0 <= t_sec <= 1.0 and 0.0 <= u_edge <= 1.0):
                continue
            dist_along = t_sec * profile_len
            depth = z_start + u_edge * (z_end - z_start)
            hits.append((dist_along, depth))

        if len(hits) < 2:
            continue

        # A well-formed rectangle crossing a line gives exactly 2 hits
        # (possibly with duplicate near-corner hits from adjacent edges);
        # take the two most separated in along-profile distance.
        hits.sort(key=lambda h: h[0])
        p_a, p_b = hits[0], hits[-1]

        # 2026-08-21 fix: degenerate/near-edge-on case. When the profile
        # runs (nearly) PARALLEL to the fault's strike, the top/bottom
        # edges are themselves (nearly) parallel to the profile line, so
        # _segment_intersection_2d finds no valid crossing for them
        # (denom~0) and the only 2 hits come from the short SIDE edges
        # instead. Those side-edge hits both land near the fault's own
        # centroid depth (the profile passes near y=0 in the fault's
        # local frame), producing a near-zero vertical extent -- an
        # effectively invisible flat sliver -- instead of the fault's
        # true top-to-bottom depth range. Reproduces at ANY dip (not
        # dip-specific -- verified against dip=90 and dip=60 alike),
        # but is far more likely to be hit in practice at steep/vertical
        # dips: the near-vertical-dip minimum-horizontal-width clamp
        # (fault_geometry.fault_corners_geo) makes the true top/bottom
        # edges only ~50 m apart in map view, so even a profile that
        # isn't exactly strike-parallel can end up graze the side edges
        # instead of the top/bottom ones at these orientations.
        # Detected via a depth-span sanity check: a real top/bottom-edge
        # crossing always spans a meaningful fraction of the fault's
        # true top-to-bottom depth range; a side-edge-only crossing
        # collapses toward zero. When triggered, we report the fault's
        # TRUE vertical extent (top_depth -> bottom_depth) at the
        # along-profile position of the fault's own centroid -- the
        # physically meaningful thing a near-strike-parallel section
        # actually cuts through, instead of the degenerate sliver.
        true_span = abs(fault.bottom_depth - fault.top_depth)
        depth_span = abs(p_b[1] - p_a[1])
        if true_span > 1e-6 and depth_span < 0.05 * true_span:
            fx, fy = geo_to_km(fault.lon, fault.lat, lon1, lat1)
            ux, uy = (x1 - x0) / profile_len, (y1 - y0) / profile_len
            dist_at_centroid = fx * ux + fy * uy
            traces.append({
                "fault": fault,
                "dist_km": [dist_at_centroid, dist_at_centroid],
                "depth_km": [fault.top_depth, fault.bottom_depth],
                "label": f"F{i + 1}",
            })
            continue

        traces.append({
            "fault": fault,
            "dist_km": [p_a[0], p_b[0]],
            "depth_km": [p_a[1], p_b[1]],
            "label": f"F{i + 1}",
        })

    return traces


def project_fault_traces_onto_polyline(sources, vertices, half_width_km=None):
    """
    Multi-segment counterpart of project_fault_traces_onto_section()
    (2026-08-21; same "call the single-leg function once per leg and
    stitch, with cumulative distance offsets" strategy as
    okada_engine.compute_cross_section_multi() -- see that function's
    docstring). vertices: list of (lon, lat), length >= 2.

    A fault whose footprint is crossed by MULTIPLE legs of the polyline
    (possible near a sharp bend) will appear once per leg it crosses --
    each occurrence is still a geometrically valid trace of that fault
    against that particular leg, so this is left as-is rather than
    arbitrarily picking one.

    Returns the same list-of-dicts shape project_fault_traces_onto_section()
    returns, with "dist_km" already shifted into the polyline's own
    cumulative along-profile distance.
    """
    from .geo_profile import polyline_segment_info

    if len(vertices) < 2:
        raise ValueError("A profile polyline needs at least 2 vertices.")

    seg_info = polyline_segment_info(vertices)
    all_traces = []
    for leg_i, ((lon_a, lat_a), (lon_b, lat_b)) in enumerate(
            zip(vertices[:-1], vertices[1:])):
        leg_traces = project_fault_traces_onto_section(
            sources, lon_a, lat_a, lon_b, lat_b, half_width_km=half_width_km)
        offset = seg_info["cumulative_dist_km"][leg_i]
        for t in leg_traces:
            t["dist_km"] = [d + offset for d in t["dist_km"]]
            all_traces.append(t)
    return all_traces
