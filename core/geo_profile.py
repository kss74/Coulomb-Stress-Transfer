# -*- coding: utf-8 -*-
"""
Projection of geographic points onto a cross-section profile line.

Shared by every cross-section overlay (earthquake catalog, focal
mechanisms, annotation points) that needs to answer the same two
questions for a scatter of (lon, lat) points against a profile line
lon1,lat1 -> lon2,lat2:

    1. how far along the profile does this point fall (for the x-axis
       position in the cross-section), and
    2. how far off the profile line does it sit perpendicular to it
       (to decide whether it's within the "search width" swath that
       should be projected onto the section at all).

Uses the same local flat-Earth approximation (okada_engine.geo_to_km)
as the rest of the plugin's physics -- consistent, not a second
projection convention, and adequate at the fault/cross-section length
scales this plugin operates at.
"""

import numpy as np

from .okada_engine import geo_to_km


def project_points_to_profile(lons, lats, lon1, lat1, lon2, lat2):
    """
    Project (lons, lats) onto the profile line lon1,lat1 -> lon2,lat2.

    Returns (dist_along_km, perp_km, profile_length_km):
      dist_along_km : distance from the START point, measured along the
                       profile's direction (can be negative or exceed
                       profile_length_km for points that project outside
                       the segment -- callers filter on this as needed).
      perp_km        : signed perpendicular distance from the profile
                       line (positive = to the right of the start->finish
                       direction, i.e. clockwise 90 deg from the profile
                       azimuth; sign is only meaningful for symmetric
                       search-width filtering, not otherwise interpreted).
      profile_length_km : straight-line length of the profile itself.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    x, y = geo_to_km(lons, lats, lon1, lat1)
    x2, y2 = geo_to_km(lon2, lat2, lon1, lat1)
    profile_length_km = float(np.hypot(x2, y2))
    if profile_length_km < 1e-9:
        raise ValueError("Profile start and finish points coincide.")
    ux, uy = x2 / profile_length_km, y2 / profile_length_km

    dist_along_km = x * ux + y * uy
    perp_km = x * uy - y * ux
    return dist_along_km, perp_km, profile_length_km


def project_points_to_polyline(lons, lats, vertices):
    """
    Multi-segment counterpart of project_points_to_profile() (2026-08-21;
    needed for both "several segments with several strikes" profiles
    and importing arbitrary extra depth-section elements onto them --
    request items 5 and 7). vertices: list of (lon, lat), length >= 2.

    For each input point, projects it onto EVERY leg of the polyline
    (reusing project_points_to_profile() per leg -- no new projection
    math, same convention as the rest of this module) and keeps
    whichever leg gives the smallest perpendicular distance, clamping
    each leg's own along-leg distance to [0, leg_length] first so a
    point near a bend is attributed to the nearest ENDPOINT of a leg
    rather than an extrapolation past it.

    Returns (dist_along_km, perp_km, total_length_km):
      dist_along_km : cumulative distance from the polyline's start, to
                       the chosen closest point on the chosen leg.
      perp_km        : perpendicular distance (unsigned) from that point
                       to its chosen leg.
      total_length_km: polyline's total length (see
                       polyline_segment_info()).

    A 2-vertex polyline degenerates to a single call to
    project_points_to_profile() (with perp_km taken as absolute value,
    since sign is leg-relative and not meaningful once multiple legs
    are in play) -- so callers can use this uniformly for both single-
    and multi-segment profiles.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    seg_info = polyline_segment_info(vertices)

    best_dist = np.full(lons.shape, np.nan)
    best_perp = np.full(lons.shape, np.inf)

    for leg_i, ((lon_a, lat_a), (lon_b, lat_b)) in enumerate(
            zip(vertices[:-1], vertices[1:])):
        leg_len = seg_info["segment_length_km"][leg_i]
        if leg_len < 1e-9:
            continue
        d_along, d_perp, _ = project_points_to_profile(lons, lats, lon_a, lat_a, lon_b, lat_b)
        d_along_clamped = np.clip(d_along, 0.0, leg_len)
        d_perp_abs = np.abs(d_perp)
        # Distance to the CLAMPED nearest point on this leg (Pythagorean
        # combination of the along-leg overshoot, if any, and the
        # perpendicular offset) -- what actually decides "closest leg"
        # for points beyond a leg's own endpoints.
        overshoot = np.abs(d_along - d_along_clamped)
        dist_to_leg = np.hypot(overshoot, d_perp_abs)

        better = dist_to_leg < best_perp
        best_dist = np.where(better, seg_info["cumulative_dist_km"][leg_i] + d_along_clamped,
                             best_dist)
        best_perp = np.where(better, dist_to_leg, best_perp)

    return best_dist, best_perp, seg_info["total_length_km"]


def profile_direction(lon1, lat1, lon2, lat2):
    """
    (ux, uy): the profile's along-profile horizontal unit vector,
    (East, North) components -- i.e. sin(az), cos(az) for azimuth az
    measured clockwise from North. Shared with core.focal_side_view,
    which needs this same direction to rotate a focal mechanism's
    nodal planes into the cross-section's side-view frame.
    """
    x2, y2 = geo_to_km(lon2, lat2, lon1, lat1)
    length = float(np.hypot(x2, y2))
    if length < 1e-9:
        raise ValueError("Profile start and finish points coincide.")
    return float(x2) / length, float(y2) / length


def polyline_segment_info(vertices):
    """
    2026-08-21 addition, multi-segment cross-section profiles (a
    profile made of several straight legs with different strikes,
    instead of one lon1,lat1->lon2,lat2 segment).

    vertices : list of (lon, lat) tuples, length >= 2, in profile order.

    Returns a dict:
      cumulative_dist_km : list, length len(vertices) -- cumulative
                            along-profile distance (km) AT each vertex
                            (cumulative_dist_km[0] == 0.0).
      segment_length_km  : list, length len(vertices)-1 -- each leg's
                            own length.
      segment_azimuth_deg: list, length len(vertices)-1 -- each leg's
                            azimuth (degrees clockwise from North),
                            for labeling strike changes at segment
                            boundaries in the plot.
      total_length_km    : cumulative_dist_km[-1].

    A single-segment (2-vertex) profile is the degenerate case of this
    -- cumulative_dist_km == [0.0, total_length_km], one entry in the
    other two lists -- so callers can use this uniformly instead of
    special-casing "1 segment" vs "N segments".
    """
    if len(vertices) < 2:
        raise ValueError("A profile polyline needs at least 2 vertices.")
    cumulative = [0.0]
    seg_len = []
    seg_az = []
    for (lon_a, lat_a), (lon_b, lat_b) in zip(vertices[:-1], vertices[1:]):
        x, y = geo_to_km(lon_b, lat_b, lon_a, lat_a)
        length = float(np.hypot(x, y))
        seg_len.append(length)
        seg_az.append(float(np.degrees(np.arctan2(x, y)) % 360.0))
        cumulative.append(cumulative[-1] + length)
    return {
        "cumulative_dist_km": cumulative,
        "segment_length_km": seg_len,
        "segment_azimuth_deg": seg_az,
        "total_length_km": cumulative[-1],
    }


def filter_within_search_width(dist_along_km, perp_km, profile_length_km,
                                half_width_km, clip_to_segment=True):
    """
    Boolean mask: points within `half_width_km` of the profile line,
    and (if clip_to_segment) whose along-profile projection actually
    falls within [0, profile_length_km] rather than off one end.
    """
    mask = np.abs(perp_km) <= half_width_km
    if clip_to_segment:
        mask &= (dist_along_km >= 0.0) & (dist_along_km <= profile_length_km)
    return mask
