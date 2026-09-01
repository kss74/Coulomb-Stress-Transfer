# -*- coding: utf-8 -*-
"""
Focal-mechanism SIDE-VIEW projection for the cross-section tool
(PROJECT_HANDOVER_ADDENDUM_2026-08-18b_cross_section_overhaul.md, Phase 2
physics derivation -- this is the item that addendum deliberately left
out of core.cross_section_config pending its own derivation).

Problem
-------
core.beachball.draw_beachball() (used on the Focal Mechanisms tab and
by utils.focal_mechanism_layer) renders ObsPy's standard lower-
hemisphere projection as seen from ABOVE: picture axes = (East, North),
viewing axis = Down. A cross-section needs the mechanism as seen from
the SIDE instead: picture axes = (along-profile distance, depth),
viewing axis = horizontal, perpendicular to the profile.

This is a pure change of Cartesian frame, not a new radiation-pattern
algorithm, and it reuses -- rather than reimplements -- the exact
normal/slip-vector machinery already validated elsewhere in this
project: core.focal_mechanism._plane_normal_slip() turns
(strike, dip, rake) into a unit fault-normal n and unit slip vector l
in the project's standard (East, North, Down) frame, and
core.optimal_plane._traction_plane_to_strike_dip_rake() is its exact
inverse. Both are already exercised end-to-end by
core.focal_mechanism.aux_plane() and the optimal-plane solver.

Derivation
----------
If n and l are rotated into a DIFFERENT orthonormal frame (E'', N'',
D'') that plays the SAME structural role as (East, North, Down) --
i.e. E'' x N'' = D'', matching E x N = D in the original frame (see
`check()` below, which verifies this numerically rather than just
asserting it) -- then running the SAME
_traction_plane_to_strike_dip_rake() on the rotated (n'', l'') yields
an "apparent" (strike'', dip'', rake'') that, fed through the SAME
beach()-based rendering used for the map view, draws CORRECTLY for
that new frame: E'' becomes the picture's horizontal axis, N'' becomes
the picture's vertical axis, D'' becomes the new viewing axis.

Because R (with rows E'', N'', D'', each expressed in the original
East/North/Down components) is built from an explicit right-handed
cross product (D'' = E'' x N''), it is a PROPER rotation (det = +1),
not a reflection -- this is the property that actually matters
physically: a reflection would silently mirror the double-couple and
swap compressional/dilatational quadrants left-right, which is the
specific failure mode `check()` guards against.

Frame choice, and why N'' = Down (not Up)
------------------------------------------
    E'' = (ux, uy, 0)   -- the profile's own along-profile horizontal
                           unit vector, (East, North) components (e.g.
                           geo_profile.profile_direction()). Added
                           directly as a += dx offset to the plot's
                           dist_km coordinate.
    N'' = (0, 0, 1)     -- Down. Added directly as a += dy offset to
                           the plot's depth_km coordinate.
    D'' = E'' x N'' = (uy, -ux, 0)   -- horizontal, perpendicular to
                           the profile; the new viewing axis.

It is tempting to set N'' = Up (= -Down), by analogy with the map view
where beach()'s "north" offset (+dy) reads as visually "up" on a
normal, non-inverted axis. But core.cross_section_plot's main panel
plots events directly at (dist_km, depth_km) and then calls
ax_main.invert_yaxis() so that depth_km=0 renders at the TOP of the
figure (surface) and increasing depth_km renders further down --
exactly like core.beachball.draw_beachball()'s map-view usage, which
adds beach()'s (dx, dy) offsets straight to (cx, cy) with NO axes=
hack (see that module's docstring on why). Since cy IS depth_km
verbatim, "beach()'s +dy offset should mean physically deeper" is what
makes a compressional lobe drawn "above" the hypocenter in the
mechanism's own frame end up rendered shallower on the (inverted-axis)
screen, and a lobe drawn "below" end up deeper -- i.e. N'' = Down is
what keeps the DATA-coordinate offset consistent with what invert_yaxis()
then displays correctly. This is checked directly in `check()` by
confirming det(R) = +1 for the ACTUAL chosen (E'', N'', D''), not just
for a hypothetical alternative.

Two special-case sanity checks (also asserted in `check()`):
  * A profile running exactly perpendicular to a vertical fault's
    strike views that fault face-on (line of sight along the fault's
    own normal) -> apparent dip'' = 0, matching ordinary beachball
    convention for "viewed straight down the normal".
  * A profile running exactly parallel to a vertical fault's strike
    views it edge-on -> apparent dip'' = 90.
  * A perfectly horizontal fault (dip=0) is edge-on from ANY horizontal
    viewing direction, for any profile azimuth -> apparent dip'' = 90
    always, independent of az.

Caveat (documented, not swept under the rug): this is a true
orthographic side view along a SINGLE horizontal direction
(perpendicular to the profile's map-view azimuth). Like
core.eq_catalog_import's along-profile projection used for the EQ
overlay's PLOTTED POSITION, an event's own perpendicular offset from
the profile line affects only where it's placed along the x-axis
(handled separately by geo_profile.filter_within_search_width()), not
the mechanism's rendered ORIENTATION, which uses the profile's azimuth
only. This is the same simplification GMT's own psmeca cross-section
mode (-Q) makes, and is standard practice for swath cross-sections.
"""

import numpy as np

from .focal_mechanism import _plane_normal_slip
from .optimal_plane import _traction_plane_to_strike_dip_rake


def profile_frame_vectors(ux, uy):
    """
    Return (E'', N'', D'') as three length-3 numpy arrays, each
    expressed in the original (East, North, Down) frame, for a profile
    whose along-profile horizontal unit direction has (East, North)
    components (ux, uy) -- e.g. geo_profile.profile_direction(), or
    (sin(az), cos(az)) for azimuth az degrees clockwise from North.

    Normalizes (ux, uy) defensively so callers can pass an
    unnormalized (dx, dy) if that's more convenient.
    """
    norm = float(np.hypot(ux, uy))
    if norm < 1e-9:
        raise ValueError("Degenerate profile direction (ux, uy) ~ (0, 0).")
    ux, uy = ux / norm, uy / norm
    E2 = np.array([ux, uy, 0.0])
    N2 = np.array([0.0, 0.0, 1.0])           # Down -- see module docstring
    D2 = np.cross(E2, N2)                    # = (uy, -ux, 0)
    return E2, N2, D2


def _to_profile_frame(v, E2, N2, D2):
    v = np.asarray(v, dtype=float)
    return np.array([float(np.dot(E2, v)), float(np.dot(N2, v)), float(np.dot(D2, v))])


def apparent_side_view_sdr(strike_deg, dip_deg, rake_deg, ux, uy):
    """
    Rotate one nodal plane's (strike, dip, rake) into the "apparent"
    triple that reproduces the correct side-view radiation pattern
    when fed through core.beachball.draw_beachball() with cx=dist_km,
    cy=depth_km on a cross-section's main (inverted-y) panel.
    """
    E2, N2, D2 = profile_frame_vectors(ux, uy)
    n, l = _plane_normal_slip(strike_deg, dip_deg, rake_deg)
    n2 = _to_profile_frame(n, E2, N2, D2)
    l2 = _to_profile_frame(l, E2, N2, D2)
    # _traction_plane_to_strike_dip_rake() only negates n (not l) when
    # n[2] > 0. That's fine for its existing caller (which always hands
    # it a consistent nz<=0 pair already) but WRONG in general -- see
    # core.focal_mechanism.aux_plane()'s docstring for the same bug
    # class: negating the normal without also negating the slip vector
    # flips the physical rake sign, since (n, l) and (-n, -l) describe
    # the same plane but (-n, l) does not. Apply the correct fix here,
    # same as aux_plane() does, before calling it.
    if n2[2] > 0:
        n2, l2 = -n2, -l2
    return _traction_plane_to_strike_dip_rake(n2, l2)


def apparent_side_view_event(strike1, dip1, rake1, ux, uy,
                              strike2=None, dip2=None, rake2=None):
    """
    Convenience wrapper: side-view SDR for both nodal planes of one
    event (plane 2 optional, matching FocalMechanismEvent's own
    optional plane-2 fields). Returns
    (strike1'', dip1'', rake1'', strike2'', dip2'', rake2'') with the
    plane-2 entries None if strike2/dip2/rake2 weren't given.
    """
    s1, d1, r1 = apparent_side_view_sdr(strike1, dip1, rake1, ux, uy)
    if strike2 is None or dip2 is None or rake2 is None:
        return s1, d1, r1, None, None, None
    s2, d2, r2 = apparent_side_view_sdr(strike2, dip2, rake2, ux, uy)
    return s1, d1, r1, s2, d2, r2


def check():
    """Verification per project convention: numerical spot-checks."""
    # 1. (E'', N'', D'') is a proper rotation (orthonormal, det=+1) for
    #    a range of profile azimuths -- guards against an accidental
    #    reflection, which would mirror compressional/dilatational
    #    quadrants without raising any other visible error.
    for az_deg in [0.0, 30.0, 90.0, 137.0, 180.0, 271.0, 359.0]:
        az = np.radians(az_deg)
        ux, uy = np.sin(az), np.cos(az)
        E2, N2, D2 = profile_frame_vectors(ux, uy)
        R = np.vstack([E2, N2, D2])
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10), f"not orthonormal at az={az_deg}"
        assert abs(np.linalg.det(R) - 1.0) < 1e-10, f"reflection (det!=+1) at az={az_deg}"

    # 2. Face-on case: a vertical N-S fault (strike=0, dip=90) has its
    #    normal pointing East. The viewing axis is D''=(uy,-ux,0), so
    #    D''=East=(1,0,0) requires uy=1, ux=0 -- a profile heading due
    #    North. Viewing exactly along the fault's own normal -> dip''=0
    #    (matches ordinary beachball convention: looking straight down
    #    a plane's normal shows it "face on", dip=0).
    s, d, r = apparent_side_view_sdr(0.0, 90.0, 0.0, ux=0.0, uy=1.0)
    assert d < 1e-6, f"expected face-on dip''=0, got {d}"

    # 3. Edge-on case: the same N-S vertical fault, profile heading due
    #    East instead. Now the viewing axis D''=(0,-1,0) lies IN the
    #    fault plane itself (parallel to its strike) -> dip''=90.
    s, d, r = apparent_side_view_sdr(0.0, 90.0, 0.0, ux=1.0, uy=0.0)
    assert abs(d - 90.0) < 1e-6, f"expected edge-on dip''=90, got {d}"

    # 4. Horizontal fault (dip=0) is edge-on from ANY horizontal
    #    viewing direction, for any profile azimuth.
    for az_deg in [0.0, 47.0, 118.0, 260.0]:
        az = np.radians(az_deg)
        ux, uy = np.sin(az), np.cos(az)
        s, d, r = apparent_side_view_sdr(strike_deg=23.0, dip_deg=0.0, rake_deg=10.0,
                                          ux=ux, uy=uy)
        assert abs(d - 90.0) < 1e-6, f"horizontal fault should be edge-on at az={az_deg}, got dip={d}"

    # 5. Plane 2 optional-handling passthrough.
    s1, d1, r1, s2, d2, r2 = apparent_side_view_event(10, 80, -30, 0.3, 0.95)
    assert s2 is None and d2 is None and r2 is None
    s1, d1, r1, s2, d2, r2 = apparent_side_view_event(10, 80, -30, 0.3, 0.95,
                                                        strike2=100, dip2=60, rake2=-150)
    assert s2 is not None and 0.0 <= d2 <= 90.0

    # 6. Round trip sanity: rotating with (ux,uy) and its exact
    #    opposite (-ux,-uy) (profile reversed) must give the SAME dip
    #    (viewing from the other end of the same vertical plane is
    #    still perpendicular/parallel by the same amount) even though
    #    strike''/rake'' differ (mirrored left-right in the picture,
    #    which is the CORRECT behavior for walking the profile
    #    backwards, not a bug).
    for strike, dip, rake in [(45, 60, 90), (200, 35, -60), (10, 80, -30)]:
        d_ref = None
        for ux, uy in [(0.6, 0.8), (-0.6, -0.8)]:
            _, d, _ = apparent_side_view_sdr(strike, dip, rake, ux, uy)
            if d_ref is None:
                d_ref = d
            else:
                assert abs(d - d_ref) < 1e-6, (strike, dip, rake, d, d_ref)

    print("focal_side_view.check(): all assertions passed.")


if __name__ == "__main__":
    check()
