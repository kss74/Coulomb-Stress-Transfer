# -*- coding: utf-8 -*-
"""
Create QGIS vector layers for fault polygons, surface traces, receiver-depth
intersection lines, and displacement arrows — mirroring the fault-geometry
overlays shown in Coulomb 3.x/4.0 map views.
"""

import numpy as np


def _fault_corners_geo(fault, min_horiz_width_km=0.05):
    """Compute the 4 corners (lon,lat) of a fault's surface projection
    (top-left, top-right, bottom-right, bottom-left, in the horizontal
    projection sense — i.e. the down-dip edges projected to the surface).

    For near-vertical faults (dip near 90°), the horizontal projection of
    the down-dip width approaches zero, which would otherwise produce a
    degenerate (zero-area) polygon that some renderers/exports may reject
    or fail to display. We enforce a minimum horizontal width purely for
    visualization; it does not affect any stress computation, which uses
    the fault's true (unprojected) geometry.
    """
    from ..core.okada_engine import km_to_geo

    strike = np.deg2rad(fault.strike)
    dip = np.deg2rad(fault.dip)
    cs, ss = np.cos(strike), np.sin(strike)
    cd = np.cos(dip)

    half_L = fault.length / 2
    w_horiz = fault.width * cd  # horizontal projection of down-dip width
    if abs(w_horiz) < min_horiz_width_km:
        w_horiz = min_horiz_width_km if w_horiz >= 0 else -min_horiz_width_km

    # Local fault-frame corners (top-left, top-right, bottom-right, bottom-left)
    # x = along-strike, y = across-strike, CENTERED ON THE CENTROID (y=0 at
    # fault.lon/fault.lat, which is the volumetric centroid's surface
    # projection -- NOT the top edge). This must match the same convention
    # used by surface_trace()/top_center()/_receiver_depth_trace_geo(),
    # where the top edge sits at y=-half_w (up-dip) and the bottom edge at
    # y=+half_w (down-dip), both measured from the centroid.
    #
    # FIXED (was a real bug): the previous version anchored the "top" edge
    # at y=0 (i.e. AT the centroid line) and the "bottom" edge at
    # y=+w_horiz (a full width_horiz below the centroid), which silently
    # shifts the entire drawn polygon down-dip by half_w relative to the
    # fault's true surface projection. This is invisible at dip=90 deg
    # (w_horiz=0) but for any dipping fault it means the polygon's "top"
    # edge does NOT coincide with surface_trace()'s top edge -- e.g. for a
    # fault with top_depth=0 (reaching the surface), the drawn polygon no
    # longer intersects the surface trace line at all. Purely a
    # visualization bug: no stress/CFF computation reads this function.
    half_w = w_horiz / 2.0
    corners_local = [(-half_L, -half_w), (half_L, -half_w), (half_L, half_w), (-half_L, half_w)]

    corners_geo = []
    for lx, ly in corners_local:
        # Rotate to geographic (x=E, y=N)
        e_km = lx * ss + ly * cs
        n_km = lx * cs - ly * ss
        lon, lat = km_to_geo(e_km, n_km, fault.lon, fault.lat)
        corners_geo.append((float(lon), float(lat)))
    return corners_geo


def _receiver_depth_trace_geo(fault, z_recv_km):
    """
    Compute the (lon,lat) line where the fault plane intersects a
    horizontal slice at receiver depth z_recv_km — i.e. the "receiver
    depth polyline" shown in Coulomb's map view. Returns None if the
    fault plane does not reach that depth (z_recv outside [top,bottom]).
    """
    from ..core.okada_engine import km_to_geo

    top = fault.top_depth
    bot = fault.bottom_depth
    if z_recv_km < min(top, bot) or z_recv_km > max(top, bot):
        return None

    strike = np.deg2rad(fault.strike)
    dip = np.deg2rad(fault.dip)
    cs, ss = np.cos(strike), np.sin(strike)
    cd = np.cos(dip)

    # Fraction of the way down-dip that this depth corresponds to
    if abs(bot - top) < 1e-9:
        frac = 0.0
    else:
        frac = (z_recv_km - top) / (bot - top)

    # fault.lon/fault.lat is the CENTROID's surface projection. The local
    # fault-frame y (across-strike, measured from the TOP edge) at this
    # depth is w_horiz_at_z; recentre on the centroid the same way
    # surface_trace() does, by subtracting the half-width horizontal offset.
    w_horiz = fault.width * cd
    w_horiz_at_z = w_horiz * frac
    y_from_centroid = w_horiz_at_z - w_horiz / 2.0

    half_L = fault.length / 2
    corners_local = [(-half_L, y_from_centroid), (half_L, y_from_centroid)]

    line_geo = []
    for lx, ly in corners_local:
        e_km = lx * ss + ly * cs
        n_km = lx * cs - ly * ss
        lon, lat = km_to_geo(e_km, n_km, fault.lon, fault.lat)
        line_geo.append((float(lon), float(lat)))
    return line_geo


def create_fault_layer(faults, layer_name="Source Faults"):
    """
    Create a QGIS polygon layer for stress-SOURCE faults (with slip),
    styled with a distinct solid amber/gold fill — visually separate from
    the diverging red/blue color ramp used for receiver-fault ΔCFF, so
    sources and receivers are never confused with "high positive stress"
    coloring on the same map.

    For individual receiver faults colored by their resolved ΔCFF, use
    create_receiver_fault_layer_colored() instead.
    """
    from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry,
                            QgsPointXY, QgsField, QgsProject,
                            QgsSingleSymbolRenderer, QgsFillSymbol)
    from qgis.PyQt.QtCore import QVariant

    layer = QgsVectorLayer("Polygon?crs=EPSG:4326", layer_name, "memory")
    provider = layer.dataProvider()
    provider.addAttributes([
        QgsField("strike", QVariant.Double),
        QgsField("dip", QVariant.Double),
        QgsField("rake", QVariant.Double),
        QgsField("slip", QVariant.Double),
        QgsField("depth_centroid", QVariant.Double),
        QgsField("depth_top", QVariant.Double),
        QgsField("depth_bottom", QVariant.Double),
        QgsField("rt_lateral_slip_m", QVariant.Double),
        QgsField("reverse_slip_m", QVariant.Double),
    ])
    layer.updateFields()

    for fault in faults:
        corners = _fault_corners_geo(fault)
        points = [QgsPointXY(lon, lat) for lon, lat in corners]
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromPolygonXY([points]))
        feat.setAttributes([
            fault.strike, fault.dip, fault.rake, fault.slip, fault.depth,
            fault.top_depth, fault.bottom_depth,
            fault.rt_lateral_slip, fault.reverse_slip,
        ])
        provider.addFeature(feat)

    layer.updateExtents()

    symbol = QgsFillSymbol.createSimple({
        "color": "255,170,0,90", "outline_color": "180,110,0,220", "outline_width": "0.6",
        "outline_style": "solid",
    })
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))

    QgsProject.instance().addMapLayer(layer)
    return layer


def create_receiver_fault_layer_colored(results, layer_name="Receiver Faults (ΔCFF)"):
    """
    Create a QGIS polygon layer for individual RECEIVER faults, colored by
    their resolved ΔCFF — red/blue diverging, matching the same color
    sense as the CFF raster (positive=red/stress-promoting, negative=
    blue/stress-relieving). This is the receiver-fault analogue of the
    raster's diverging color map, but per-fault-polygon instead of a
    continuous grid.

    results: list of dicts as returned by
    core.okada_engine.compute_cff_on_receiver_faults(), i.e.
    {"fault": FaultParameters, "cff_mpa": float, "shear_mpa": float,
     "normal_mpa": float, "used_dc3d": bool}
    """
    from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry,
                            QgsPointXY, QgsField, QgsProject,
                            QgsGraduatedSymbolRenderer, QgsFillSymbol,
                            QgsRendererRange)
    from qgis.PyQt.QtCore import QVariant
    from qgis.PyQt.QtGui import QColor

    layer = QgsVectorLayer("Polygon?crs=EPSG:4326", layer_name, "memory")
    provider = layer.dataProvider()
    provider.addAttributes([
        QgsField("strike", QVariant.Double),
        QgsField("dip", QVariant.Double),
        QgsField("rake", QVariant.Double),
        QgsField("depth_centroid", QVariant.Double),
        QgsField("cff_bar", QVariant.Double),
        QgsField("shear_bar", QVariant.Double),
        QgsField("normal_bar", QVariant.Double),
        QgsField("method", QVariant.String),
    ])
    layer.updateFields()

    cff_values = []
    for res in results:
        fault = res["fault"]
        corners = _fault_corners_geo(fault)
        points = [QgsPointXY(lon, lat) for lon, lat in corners]
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromPolygonXY([points]))
        cff_bar = res["cff_mpa"] * 10
        shear_bar = res["shear_mpa"] * 10
        normal_bar = res["normal_mpa"] * 10
        method = "DC3D" if res["used_dc3d"] else "Surface (z=0)"
        feat.setAttributes([
            fault.strike, fault.dip, fault.rake, fault.depth,
            float(cff_bar), float(shear_bar), float(normal_bar), method,
        ])
        provider.addFeature(feat)
        cff_values.append(cff_bar)

    layer.updateExtents()

    if cff_values:
        vmax = max(abs(min(cff_values)), abs(max(cff_values)), 1e-6)
    else:
        vmax = 1.0

    # Diverging classes: strongly negative -> blue, zero -> gray,
    # strongly positive -> red. Same visual sense as the CFF raster.
    # 5 upper bounds define 5 ranges; the first range's lower bound is -vmax.
    upper_bounds = [-vmax/2, -1e-9, 1e-9, vmax/2, vmax]
    colors = [
        QColor(33, 66, 160),    # strong negative (blue)
        QColor(120, 150, 210),  # mild negative
        QColor(230, 230, 230),  # ~zero
        QColor(220, 120, 100),  # mild positive
        QColor(178, 24, 43),    # strong positive (red)
    ]
    ranges = []
    lower = -vmax
    for i, upper in enumerate(upper_bounds):
        symbol = QgsFillSymbol.createSimple({
            "color": colors[i].name(), "outline_color": "60,60,60,200",
            "outline_width": "0.3",
        })
        label = f"{lower:.3f} to {upper:.3f} bar"
        ranges.append(QgsRendererRange(lower, upper, symbol, label))
        lower = upper

    renderer = QgsGraduatedSymbolRenderer("cff_bar", ranges)
    layer.setRenderer(renderer)

    QgsProject.instance().addMapLayer(layer)
    return layer


def create_surface_trace_layer(faults, layer_name="Fault Top Projection"):
    """
    Create a QGIS line layer showing each fault's TOP-EDGE horizontal
    projection ("Fault Top Projection") -- the actual (possibly buried)
    top edge's plan-view location, via FaultParameters.surface_trace().

    This is NOT the geological "Surface Trace" (the fault plane
    extrapolated up-dip to z=0 for a blind/buried fault) -- see
    create_geological_surface_trace_layer() for that. The two coincide
    exactly when top_depth=0.
    """
    from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry,
                            QgsPointXY, QgsField, QgsProject,
                            QgsSingleSymbolRenderer, QgsLineSymbol)
    from qgis.PyQt.QtCore import QVariant

    layer = QgsVectorLayer("LineString?crs=EPSG:4326", layer_name, "memory")
    provider = layer.dataProvider()
    provider.addAttributes([
        QgsField("strike", QVariant.Double),
        QgsField("dip", QVariant.Double),
        QgsField("depth_top", QVariant.Double),
    ])
    layer.updateFields()

    for fault in faults:
        (lon1, lat1), (lon2, lat2) = fault.surface_trace()
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromPolylineXY(
            [QgsPointXY(lon1, lat1), QgsPointXY(lon2, lat2)]))
        feat.setAttributes([fault.strike, fault.dip, fault.top_depth])
        provider.addFeature(feat)

    layer.updateExtents()

    symbol = QgsLineSymbol.createSimple({"color": "0,0,0", "width": "0.8"})
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))

    QgsProject.instance().addMapLayer(layer)
    return layer


def create_geological_surface_trace_layer(faults, layer_name="Surface Trace (extrapolated to z=0)"):
    """
    Create a QGIS line layer showing each fault's GEOLOGICAL Surface
    Trace -- the fault plane's intersection with z=0, extrapolated
    up-dip from the (possibly buried) top edge, via
    FaultParameters.geological_surface_trace(). This is Coulomb's own
    "surface trace" concept for a blind/buried fault: where the fault
    would break the surface if its dip continued, NOT the actual top
    edge's own horizontal projection (see create_surface_trace_layer(),
    "Fault Top Projection", for that).

    Faults for which geological_surface_trace() returns None (dip<=0)
    are skipped. Faults with top_depth<=0 are included but will coincide
    exactly with their Fault Top Projection line -- that's expected, not
    a bug: a fault already at or above the surface has no up-dip
    extrapolation left to apply.
    """
    from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry,
                            QgsPointXY, QgsField, QgsProject,
                            QgsSingleSymbolRenderer, QgsLineSymbol)
    from qgis.PyQt.QtCore import QVariant

    layer = QgsVectorLayer("LineString?crs=EPSG:4326", layer_name, "memory")
    provider = layer.dataProvider()
    provider.addAttributes([
        QgsField("strike", QVariant.Double),
        QgsField("dip", QVariant.Double),
        QgsField("depth_top", QVariant.Double),
    ])
    layer.updateFields()

    any_added = False
    for fault in faults:
        trace = fault.geological_surface_trace()
        if trace is None:
            continue
        (lon1, lat1), (lon2, lat2) = trace
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromPolylineXY(
            [QgsPointXY(lon1, lat1), QgsPointXY(lon2, lat2)]))
        feat.setAttributes([fault.strike, fault.dip, fault.top_depth])
        provider.addFeature(feat)
        any_added = True

    layer.updateExtents()

    symbol = QgsLineSymbol.createSimple({
        "color": "180,30,30", "width": "0.9", "line_style": "dash",
    })
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))

    QgsProject.instance().addMapLayer(layer)
    return layer if any_added else None


def create_receiver_depth_layer(faults, z_recv_km, layer_name=None):
    """
    Create a QGIS line layer showing where each fault plane intersects the
    receiver depth (the "receiver depth polyline" shown in Coulomb map
    views). Faults whose depth range does not include z_recv_km are skipped.
    """
    from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry,
                            QgsPointXY, QgsField, QgsProject,
                            QgsSingleSymbolRenderer, QgsLineSymbol)
    from qgis.PyQt.QtCore import QVariant

    if layer_name is None:
        layer_name = f"Receiver Depth Trace ({z_recv_km:.1f} km)"

    layer = QgsVectorLayer("LineString?crs=EPSG:4326", layer_name, "memory")
    provider = layer.dataProvider()
    provider.addAttributes([
        QgsField("depth_km", QVariant.Double),
        QgsField("strike", QVariant.Double),
        QgsField("dip", QVariant.Double),
    ])
    layer.updateFields()

    any_added = False
    for fault in faults:
        trace = _receiver_depth_trace_geo(fault, z_recv_km)
        if trace is None:
            continue
        (lon1, lat1), (lon2, lat2) = trace
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromPolylineXY(
            [QgsPointXY(lon1, lat1), QgsPointXY(lon2, lat2)]))
        feat.setAttributes([z_recv_km, fault.strike, fault.dip])
        provider.addFeature(feat)
        any_added = True

    layer.updateExtents()

    symbol = QgsLineSymbol.createSimple({
        "color": "0,120,255", "width": "0.8", "line_style": "dash",
    })
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))

    QgsProject.instance().addMapLayer(layer)
    return layer if any_added else None


def create_points_table_layer(rows, layer_name):
    """
    Generic point layer from a list of dicts each containing 'lon'/'lat'
    plus arbitrary numeric attribute columns -- used for slip-inversion
    augmented-results tables (core.slip_inversion_report's
    build_augmented_gnss_rows()/build_augmented_los_rows()). GNSS and
    LOS results have different column sets, so this infers fields from
    the row dicts themselves rather than hardcoding a schema; None
    values (e.g. an unused e/n/u component) become NULL attributes.
    """
    from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry,
                            QgsPointXY, QgsField, QgsProject)
    from qgis.PyQt.QtCore import QVariant

    if not rows:
        return None
    layer = QgsVectorLayer("Point?crs=EPSG:4326", layer_name, "memory")
    provider = layer.dataProvider()
    field_names = [k for k in rows[0].keys() if k not in ("lon", "lat")]
    provider.addAttributes([QgsField(name, QVariant.Double) for name in field_names])
    layer.updateFields()
    for row in rows:
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(row["lon"]), float(row["lat"]))))
        feat.setAttributes([None if row.get(name) is None else float(row[name]) for name in field_names])
        provider.addFeature(feat)
    layer.updateExtents()
    QgsProject.instance().addMapLayer(layer)
    return layer


def create_near_field_mask_layer(lon2d, lat2d, near_field_mask,
                                 layer_name="Near-field (Okada singularity)",
                                 max_cells=50000):
    """
    Export the near-field hatch overlay plot_widget.plot_cff()/
    plot_optimal_cff() draw over the map (see okada_engine.
    near_field_grid_mask()'s docstring for the 0/1/2 tier meaning) as a
    QGIS polygon layer -- one small rectangle per masked grid CELL
    (mask > 0), not a smoothed contour outline, so this needs no
    dependency on matplotlib's contour-path extraction (its public API
    has changed across versions) and always exactly matches the same
    grid the computation itself used. Each cell polygon carries a
    "tier" attribute (1 = magnitude caution, 2 = sign untrustworthy) so
    the two tiers can be styled/filtered independently in QGIS, mirroring
    the plot's two hatch densities.

    Returns None (with no layer added) if there is nothing to export
    (mask is all-zero) or if the masked-cell count exceeds `max_cells`
    (a defensive cap -- a full-resolution grid, e.g. the 2000x2000 cap
    noted elsewhere in this plugin, would otherwise create millions of
    tiny polygon features and could freeze QGIS; the caller should
    surface this as a message rather than silently doing nothing).
    """
    from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry,
                            QgsPointXY, QgsField, QgsProject)
    from qgis.PyQt.QtCore import QVariant

    mask = np.asarray(near_field_mask)
    idx_i, idx_j = np.nonzero(mask > 0)
    if idx_i.size == 0:
        return None
    if idx_i.size > max_cells:
        raise ValueError(
            f"{idx_i.size} near-field grid cells to export, exceeding "
            f"the {max_cells}-cell safety cap -- this would create too "
            f"many polygon features. Reduce the output grid resolution "
            f"(Grid Output tab) before exporting the hatch layer.")

    n_lat, n_lon = lon2d.shape
    layer = QgsVectorLayer(f"Polygon?crs=EPSG:4326", layer_name, "memory")
    provider = layer.dataProvider()
    provider.addAttributes([QgsField("tier", QVariant.Int)])
    layer.updateFields()

    feats = []
    for i, j in zip(idx_i, idx_j):
        # Half-cell extent in each direction (clamped at grid edges) so
        # adjacent cells' rectangles tile without gaps or overlaps.
        lon_lo = lon2d[i, j] - 0.5 * (lon2d[i, j] - lon2d[i, j - 1]) if j > 0 else \
            lon2d[i, j] - 0.5 * (lon2d[i, j + 1] - lon2d[i, j])
        lon_hi = lon2d[i, j] + 0.5 * (lon2d[i, j + 1] - lon2d[i, j]) if j + 1 < n_lon else \
            lon2d[i, j] + 0.5 * (lon2d[i, j] - lon2d[i, j - 1])
        lat_lo = lat2d[i, j] - 0.5 * (lat2d[i, j] - lat2d[i - 1, j]) if i > 0 else \
            lat2d[i, j] - 0.5 * (lat2d[i + 1, j] - lat2d[i, j])
        lat_hi = lat2d[i, j] + 0.5 * (lat2d[i + 1, j] - lat2d[i, j]) if i + 1 < n_lat else \
            lat2d[i, j] + 0.5 * (lat2d[i, j] - lat2d[i - 1, j])

        ring = [QgsPointXY(lon_lo, lat_lo), QgsPointXY(lon_hi, lat_lo),
               QgsPointXY(lon_hi, lat_hi), QgsPointXY(lon_lo, lat_hi),
               QgsPointXY(lon_lo, lat_lo)]
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromPolygonXY([ring]))
        feat.setAttributes([int(mask[i, j])])
        feats.append(feat)

    provider.addFeatures(feats)
    layer.updateExtents()
    QgsProject.instance().addMapLayer(layer)
    return layer


def create_optimal_plane_strike_layer(lon2d, lat2d, strike1, strike2,
                                      cff1_mpa, cff2_mpa, subsample=None,
                                      tick_len_deg=None,
                                      layer_name="Optimal Plane Strike Vectors"):
    """
    Export the short strike tick-marks plot_widget.plot_optimal_cff()
    draws (the WINNING conjugate plane -- whichever of the two attains
    the larger ΔCFF -- at each subsampled grid point) as a QGIS line
    layer, using the SAME subsampling/tick-length convention as that
    plot method so the exported layer visually matches what's on
    screen. Each line feature carries strike_deg, winning_plane (1 or
    2), and that point's own cff1_mpa/cff2_mpa as attributes.

    subsample     : grid-point stride; None = same default as
                    plot_widget.plot_optimal_cff() (max(1, n/15)).
    tick_len_deg   : half-length of each tick mark, in degrees; None =
                    same 2%-of-extent default plot_optimal_cff() uses.
    """
    from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry,
                            QgsPointXY, QgsField, QgsProject)
    from qgis.PyQt.QtCore import QVariant

    lon2d = np.asarray(lon2d); lat2d = np.asarray(lat2d)
    strike1 = np.asarray(strike1); strike2 = np.asarray(strike2)
    cff1_mpa = np.asarray(cff1_mpa); cff2_mpa = np.asarray(cff2_mpa)

    if subsample is None:
        subsample = max(1, lon2d.shape[0] // 15)
    if tick_len_deg is None:
        lon_ext = float(lon2d.max() - lon2d.min())
        lat_ext = float(lat2d.max() - lat2d.min())
        tick_len_deg = 0.02 * max(lon_ext, lat_ext, 1e-6)

    plane1_wins = cff1_mpa >= cff2_mpa
    winning_strike = np.where(plane1_wins, strike1, strike2)

    layer = QgsVectorLayer("LineString?crs=EPSG:4326", layer_name, "memory")
    provider = layer.dataProvider()
    provider.addAttributes([
        QgsField("strike_deg", QVariant.Double),
        QgsField("winning_plane", QVariant.Int),
        QgsField("cff1_mpa", QVariant.Double),
        QgsField("cff2_mpa", QVariant.Double),
    ])
    layer.updateFields()

    feats = []
    n_lat, n_lon = lon2d.shape
    for i in range(0, n_lat, subsample):
        for j in range(0, n_lon, subsample):
            strike_rad = np.radians(float(winning_strike[i, j]))
            dx = tick_len_deg * np.sin(strike_rad)
            dy = tick_len_deg * np.cos(strike_rad)
            lon0, lat0 = float(lon2d[i, j]), float(lat2d[i, j])
            line = [QgsPointXY(lon0 - dx, lat0 - dy), QgsPointXY(lon0 + dx, lat0 + dy)]
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPolylineXY(line))
            feat.setAttributes([
                float(winning_strike[i, j]),
                1 if plane1_wins[i, j] else 2,
                float(cff1_mpa[i, j]), float(cff2_mpa[i, j]),
            ])
            feats.append(feat)

    provider.addFeatures(feats)
    layer.updateExtents()
    QgsProject.instance().addMapLayer(layer)
    return layer


def create_displacement_layer(lon2d, lat2d, ux, uy, scale=1.0,
                               subsample=8, layer_name="Displacement Vectors"):
    """Create a QGIS point layer with displacement magnitude/azimuth attributes."""
    from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry,
                            QgsPointXY, QgsField, QgsProject)
    from qgis.PyQt.QtCore import QVariant

    layer = QgsVectorLayer("Point?crs=EPSG:4326", layer_name, "memory")
    provider = layer.dataProvider()
    provider.addAttributes([
        QgsField("ux_m", QVariant.Double),
        QgsField("uy_m", QVariant.Double),
        QgsField("magnitude_m", QVariant.Double),
        QgsField("azimuth_deg", QVariant.Double),
    ])
    layer.updateFields()

    n_lat, n_lon = lon2d.shape
    for i in range(0, n_lat, subsample):
        for j in range(0, n_lon, subsample):
            magnitude = float(np.hypot(ux[i, j], uy[i, j]))
            azimuth = float(np.degrees(np.arctan2(ux[i, j], uy[i, j])) % 360)
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(lon2d[i, j], lat2d[i, j])))
            feat.setAttributes([float(ux[i, j]), float(uy[i, j]), magnitude, azimuth])
            provider.addFeature(feat)

    layer.updateExtents()
    QgsProject.instance().addMapLayer(layer)
    return layer


def create_cross_section_line_layer(lon1, lat1, lon2, lat2,
                                     layer_name="Cross-Section Line"):
    """
    QGIS line layer with the single cross-section profile segment
    lon1,lat1 -> lon2,lat2 (2026-08-18b cross-section overhaul, point 10:
    "cross section line ... has an option to be exported to qgis").
    """
    from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry,
                            QgsPointXY, QgsField, QgsProject,
                            QgsSingleSymbolRenderer, QgsLineSymbol)
    from qgis.PyQt.QtCore import QVariant

    layer = QgsVectorLayer("LineString?crs=EPSG:4326", layer_name, "memory")
    provider = layer.dataProvider()
    provider.addAttributes([
        QgsField("lon1", QVariant.Double), QgsField("lat1", QVariant.Double),
        QgsField("lon2", QVariant.Double), QgsField("lat2", QVariant.Double),
    ])
    layer.updateFields()

    feat = QgsFeature()
    feat.setGeometry(QgsGeometry.fromPolylineXY(
        [QgsPointXY(lon1, lat1), QgsPointXY(lon2, lat2)]))
    feat.setAttributes([lon1, lat1, lon2, lat2])
    provider.addFeature(feat)
    layer.updateExtents()

    symbol = QgsLineSymbol.createSimple({"color": "255,0,0", "width": "1.0",
                                         "line_style": "dash"})
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))

    QgsProject.instance().addMapLayer(layer)
    return layer


def create_cross_section_search_width_layer(lon1, lat1, lon2, lat2, half_width_km,
                                             layer_name="Cross-Section Search Width"):
    """
    QGIS polygon layer for the rectangular swath (profile length x
    2*half_width_km) used to decide which earthquake-catalog/focal-
    mechanism points get projected onto the cross-section (2026-08-18b,
    point 10). A simple planar rectangle built from the same local-km
    projection used throughout the cross-section feature
    (core.geo_profile) -- not a geodesic buffer, consistent with the
    flat-Earth approximation the rest of the plugin's cross-section code
    already uses at these length scales.
    """
    from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry,
                            QgsPointXY, QgsField, QgsProject,
                            QgsSingleSymbolRenderer, QgsFillSymbol)
    from qgis.PyQt.QtCore import QVariant
    from ..core.okada_engine import geo_to_km, km_to_geo

    x1, y1 = geo_to_km(lon2, lat2, lon1, lat1)
    length_km = float(np.hypot(x1, y1))
    ux, uy = x1 / length_km, y1 / length_km
    # perpendicular unit vector (90 deg clockwise from profile direction,
    # matching core.geo_profile.project_points_to_profile's sign convention)
    px, py = uy, -ux

    corners_km = [
        (0.0 + px * half_width_km, 0.0 + py * half_width_km),
        (0.0 - px * half_width_km, 0.0 - py * half_width_km),
        (x1 - px * half_width_km, y1 - py * half_width_km),
        (x1 + px * half_width_km, y1 + py * half_width_km),
    ]
    corners_geo = [km_to_geo(x, y, lon1, lat1) for x, y in corners_km]

    layer = QgsVectorLayer("Polygon?crs=EPSG:4326", layer_name, "memory")
    provider = layer.dataProvider()
    provider.addAttributes([QgsField("half_width_km", QVariant.Double)])
    layer.updateFields()

    feat = QgsFeature()
    points = [QgsPointXY(float(lon), float(lat)) for lon, lat in corners_geo]
    feat.setGeometry(QgsGeometry.fromPolygonXY([points]))
    feat.setAttributes([half_width_km])
    provider.addFeature(feat)
    layer.updateExtents()

    symbol = QgsFillSymbol.createSimple({
        "color": "255,0,0,25", "outline_color": "255,0,0,150",
        "outline_width": "0.5", "outline_style": "dash",
    })
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))

    QgsProject.instance().addMapLayer(layer)
    return layer


def create_point_calc_layer(results, layer_name="Point Calculator Results"):
    """
    Create a QGIS point layer from core.point_calculation.compute_point_results()
    output (2026-09-01 addition, "Point Calculator" feature). One field
    per core.point_calculation.RESULT_COLUMNS entry (minus lon/lat, which
    become the point geometry instead), graduated-colored by cff_bar
    with the SAME diverging red/blue color sense as
    create_receiver_fault_layer_colored() above, so a point layer and a
    receiver-fault-polygon layer read consistently on the same map.

    results : list of dicts, exactly compute_point_results()'s return
              value (each dict has the keys in
              core.point_calculation.RESULT_COLUMNS, plus "lon"/"lat").
    """
    from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry,
                            QgsPointXY, QgsField, QgsProject,
                            QgsGraduatedSymbolRenderer, QgsMarkerSymbol,
                            QgsRendererRange)
    from qgis.PyQt.QtCore import QVariant
    from qgis.PyQt.QtGui import QColor
    from ..core.point_calculation import RESULT_COLUMNS

    # Field type per column -- everything is a Double except the string
    # label and the two boolean flags (stored as Int 0/1, since QGIS
    # attribute tables/most downstream consumers handle an Int more
    # predictably across providers than a Bool field).
    bool_fields = {"elevation_clamped", "used_dc3d"}
    str_fields = {"label"}
    attr_fields = [c for c in RESULT_COLUMNS if c not in ("lon", "lat")]

    layer = QgsVectorLayer("Point?crs=EPSG:4326", layer_name, "memory")
    provider = layer.dataProvider()
    qgs_fields = []
    for f in attr_fields:
        if f in str_fields:
            qgs_fields.append(QgsField(f, QVariant.String))
        elif f in bool_fields:
            qgs_fields.append(QgsField(f, QVariant.Int))
        else:
            qgs_fields.append(QgsField(f, QVariant.Double))
    provider.addAttributes(qgs_fields)
    layer.updateFields()

    cff_values = []
    for res in results:
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(res["lon"]), float(res["lat"]))))
        attrs = []
        for f in attr_fields:
            val = res.get(f)
            if f in str_fields:
                attrs.append("" if val is None else str(val))
            elif f in bool_fields:
                attrs.append(1 if val else 0)
            else:
                attrs.append(None if val is None else float(val))
        feat.setAttributes(attrs)
        provider.addFeature(feat)
        cff_values.append(res.get("cff_bar") or 0.0)

    layer.updateExtents()

    if cff_values:
        vmax = max(abs(min(cff_values)), abs(max(cff_values)), 1e-6)
    else:
        vmax = 1.0

    # Same 5-class diverging scheme as create_receiver_fault_layer_colored().
    upper_bounds = [-vmax / 2, -1e-9, 1e-9, vmax / 2, vmax]
    colors = [
        QColor(33, 66, 160), QColor(120, 150, 210), QColor(230, 230, 230),
        QColor(220, 120, 100), QColor(178, 24, 43),
    ]
    ranges = []
    lower = -vmax
    for i, upper in enumerate(upper_bounds):
        symbol = QgsMarkerSymbol.createSimple({
            "color": colors[i].name(), "outline_color": "60,60,60,200",
            "outline_width": "0.4", "size": "3.2",
        })
        label = f"{lower:.3f} to {upper:.3f} bar"
        ranges.append(QgsRendererRange(lower, upper, symbol, label))
        lower = upper

    renderer = QgsGraduatedSymbolRenderer("cff_bar", ranges)
    layer.setRenderer(renderer)

    QgsProject.instance().addMapLayer(layer)
    return layer
