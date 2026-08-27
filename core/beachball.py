# -*- coding: utf-8 -*-
"""
Beachball (focal-sphere) rendering for the Focal Mechanisms tab, colored
by resolved ΔCFF instead of the traditional solid black/white
compressional-quadrant fill -- matching Coulomb 3.4.2's "Calc. stress
on nodal planes" display.

Uses ObsPy's obspy.imaging.beachball, NOT a hand-rolled port of
coulomb.m's `bb2`. A first pass at porting bb2 directly ran into a real
bug: bb2's polygon-construction loop actually needs to run TWICE per
mechanism (once per lobe-color -- see obspy's plot_dc(), which is
built from the exact same lineage as bb2: both trace back to Andy
Michael / Chen Ji / Oliver Boyd's public-domain `bb.m`), producing TWO
separate simple (non-self-intersecting) polygons. An initial port
collapsed this into a single self-intersecting "bowtie" curve, which
rendered visibly wrong for several mechanisms and failed a rasterized
area check (the compressional region of ANY double couple must cover
exactly half the focal sphere -- a hard physical invariant, verified
in tests/test_beachball.py). Rather than keep chasing bugs in a
hand-rolled port of an algorithm a mature, community-vetted library
(used by existing QGIS focal-mechanism plugins like QBeachball/GISfocal)
already implements correctly, this module wraps ObsPy instead.

⚠ NEW DEPENDENCY: this module requires `obspy` (pip install obspy),
which is not otherwise used anywhere in this plugin. Needs installing
into QGIS's own Python environment, not just the system Python --
flagged in the handover addendum for this session.

Cross-checked independently: ObsPy's documented GCMT<->Cartesian axis
relation (Mrr=Mzz, Mtt=Mxx, Mpp=Myy, Mrt=Mxz, Mrp=-Myz, Mtp=-Mxy) is an
EXACT match to core.focal_mechanism.gcmt_use_to_ned()'s formula, which
was itself derived independently (brute-force search + a separate
published source) in the previous session -- a third independent
confirmation of that conversion.
"""

import numpy as np


def cff_to_color(cff_values, cmap_name="RdBu_r"):
    """
    Map an array of ΔCFF values (MPa) to RGBA facecolors using the same
    symmetric-percentile-scaled diverging colormap plot_widget.py uses
    for the grid raster (RdBu_r, vmin=-vmax, vmax=+vmax from the 98th
    percentile of |ΔCFF|), so beachball colors and the grid map read
    consistently side by side. Returns (colors, norm, cmap) -- norm and
    cmap are also useful for drawing a shared colorbar.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    cff_values = np.asarray(cff_values, dtype=float)
    vmax = np.nanpercentile(np.abs(cff_values), 98) if len(cff_values) else 1.0
    vmax = max(vmax, 1e-6)
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    colors = [cmap(norm(v)) for v in cff_values]
    return colors, norm, cmap


def _selected_plane_arc(strike_deg, dip_deg, n_points=64):
    """
    Standalone trace of ONE nodal plane on the focal sphere, in the
    same normalized units ObsPy's beach() patches use (unit radius),
    for the "which plane was selected by PLANE_MODES" highlight overlay
    (matches coulomb.m bb2's iFlag=1 behavior, which overlays plane 1's
    own arc in a contrasting color). Uses the identical l(phi) formula
    ObsPy's plot_dc() uses internally for a single plane's trace, so
    this arc lines up exactly with the edge of that plane's lobe in the
    rendered beachball.
    """
    dip_deg = min(dip_deg, 89.9999) if dip_deg >= 90 else dip_deg
    phi = np.linspace(0.0, np.pi, n_points)
    d = 90.0 - dip_deg
    l = np.sqrt(d ** 2 / (np.sin(phi) ** 2 + np.cos(phi) ** 2 * d ** 2 / 8100.0))
    d2r = np.pi / 180.0
    x = l * np.cos(phi + strike_deg * d2r)
    y = l * np.sin(phi + strike_deg * d2r)
    return x / 90.0, y / 90.0


def draw_beachball(ax, cx, cy, diameter, strike1, dip1, rake1,
                    facecolor, highlight_plane=None, strike2=None, dip2=None,
                    bgcolor="white", edgecolor="black", zorder=3):
    """
    Draw one CFF-colored beachball onto a matplotlib Axes at data
    coordinates (cx, cy), sized `diameter` in the Axes' own data units
    (e.g. degrees for a lon/lat map -- pick a diameter that reads
    sensibly against typical map extents, similar to a symbol size).

    facecolor : color spec for the compressional (colored) quadrants --
                caller determines this from ΔCFF via cff_to_color().
    highlight_plane : None, "plane1", or "plane2" -- if given, overlay
                that plane's own trace as a dark arc (which nodal plane
                PLANE_MODES selected for this event's ΔCFF).
    strike2, dip2 : only needed when highlight_plane == "plane2".
    """
    from obspy.imaging.beachball import beach

    # NOTE: deliberately NOT passing axes=ax to beach(). ObsPy's axes=
    # hack keeps the beachball a fixed size in POINTS regardless of
    # data-to-pixel scaling (for figures where you don't want beachball
    # size tied to data units) -- confirmed by testing that width=2.0
    # with axes=ax rendered as a near-invisible ~2-point dot. Here we
    # WANT size in DATA units (degrees, matching diameter_deg on a
    # lon/lat map), so add the collection directly; its patches then
    # use the axes' default data transform.
    col = beach([strike1, dip1, rake1], xy=(cx, cy), width=diameter,
               facecolor=facecolor, bgcolor=bgcolor, edgecolor=edgecolor,
               linewidth=0.6, zorder=zorder)
    ax.add_collection(col)

    if highlight_plane is not None:
        if highlight_plane == "plane1":
            hx, hy = _selected_plane_arc(strike1, dip1)
        elif highlight_plane == "plane2":
            if strike2 is None or dip2 is None:
                raise ValueError("strike2/dip2 required when highlight_plane='plane2'")
            hx, hy = _selected_plane_arc(strike2, dip2)
        else:
            raise ValueError("highlight_plane must be None, 'plane1', or 'plane2'")
        r = diameter / 2.0
        ax.plot(cx + hx * r, cy + hy * r, color="0.15", linewidth=1.0, zorder=zorder + 3)


def draw_beachball_batch(ax, results, diameter_deg, cmap_name="RdBu_r",
                          highlight_selected=True):
    """
    Convenience: draw a full batch of beachballs from
    core.focal_mechanism.compute_focal_mechanism_cff()'s output onto a
    lon/lat map Axes, colored by ΔCFF with a shared, consistent color
    scale across the whole batch. Returns (norm, cmap) for drawing a
    shared colorbar (e.g. via plot_widget.py's existing colorbar setup).
    """
    cff_values = [r["cff_mpa"] for r in results]
    colors, norm, cmap = cff_to_color(cff_values, cmap_name=cmap_name)

    for res, color in zip(results, colors):
        ev = res["event"]
        highlight = res["selected"] if highlight_selected else None
        draw_beachball(ax, ev.lon, ev.lat, diameter_deg,
                       ev.strike1, ev.dip1, ev.rake1, facecolor=color,
                       highlight_plane=highlight, strike2=ev.strike2, dip2=ev.dip2)
    return norm, cmap
