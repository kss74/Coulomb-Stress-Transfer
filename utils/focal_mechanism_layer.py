# -*- coding: utf-8 -*-
"""
Export focal-mechanism results as literal beachball glyphs on the real
QGIS map canvas -- not just the plugin's embedded matplotlib preview
(core/beachball.py + PlotWidget.plot_focal_mechanisms()), which only
renders inline in the dialog and isn't part of the actual map/project.

Reuses ObsPy's plot_dc() geometry DIRECTLY (extracts the two lobe
polygons' vertex arrays from its returned PathPatch objects) rather
than re-deriving anything -- same validated geometry as
core/beachball.py, just repurposed as exportable polygon features
instead of rendered pixels. plot_dc() always returns exactly 2 patches
per mechanism: one tagged 'b' (the compressional lobe pair -- colored
by ΔCFF here) and one tagged 'w' (the complementary background lobe
pair -- always plain white/background; together they exactly tile the
full circle, confirmed in tests/test_focal_mechanism_layer.py via the
same rasterized-area check used for core/beachball.py).

Uses the SAME 5-class diverging color scheme as
utils/vector_utils.py's create_receiver_fault_layer_colored() (rather
than a data-defined-property continuous fill, which has QGIS-version-
dependent API naming) for visual and code consistency with that
existing layer, and because QgsRuleBasedRenderer with a small number of
fixed-color rules is more robust across QGIS versions than a
data-defined color expression.

⚠ Requires `obspy` (see core/beachball.py's dependency note).
"""

import numpy as np


# Same colors/bin structure as vector_utils.create_receiver_fault_layer_colored,
# duplicated rather than imported to keep this module's only QGIS-adjacent
# dependency being obspy + qgis.core, not a cross-import into vector_utils.py
# (which is a read-only mount at time of writing -- keeping this self-
# contained avoids needing to touch it).
_BIN_COLORS_HEX = ["#2142a0", "#7896d2", "#e6e6e6", "#dc7864", "#b2182b"]
_BIN_LABELS = ["strong negative", "mild negative", "~zero", "mild positive", "strong positive"]


def _color_bin(cff_bar, vmax):
    """5-class diverging bin index, matching create_receiver_fault_layer_colored()'s
    upper_bounds = [-vmax/2, -1e-9, 1e-9, vmax/2, vmax] scheme."""
    if cff_bar <= -vmax / 2:
        return 0
    elif cff_bar <= -1e-9:
        return 1
    elif cff_bar <= 1e-9:
        return 2
    elif cff_bar <= vmax / 2:
        return 3
    else:
        return 4


def create_focal_mechanism_beachball_layer(results, diameter_deg=None,
                                            layer_name="Focal Mechanisms (Beachballs)"):
    """
    Add a polygon layer to the QGIS project: one CFF-colored beachball
    per focal-mechanism event, built from two polygon features each
    (compressional lobe pair -- colored; background lobe pair --
    white), using ObsPy's own validated double-couple geometry.

    results : list of dicts from
              core.focal_mechanism.compute_focal_mechanism_cff().
    diameter_deg : beachball diameter in degrees. Auto-sized (~6% of
              the events' bounding-box span) if not given.
    """
    from obspy.imaging.beachball import plot_dc, NodalPlane
    from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY,
                            QgsField, QgsProject, QgsFillSymbol,
                            QgsRuleBasedRenderer)
    from qgis.PyQt.QtCore import QVariant

    if not results:
        raise ValueError("no focal mechanism results to export")

    if diameter_deg is None:
        lons = [r["event"].lon for r in results]
        lats = [r["event"].lat for r in results]
        span = max(max(lons) - min(lons), max(lats) - min(lats)) if len(results) > 1 else 1.0
        diameter_deg = max(span * 0.06, 1e-4)

    cff_bar_values = [r["cff_mpa"] * 10 for r in results]  # MPa -> bar, matching plugin convention
    vmax = max(np.nanpercentile(np.abs(cff_bar_values), 98), 1e-6)

    layer = QgsVectorLayer("Polygon?crs=EPSG:4326", layer_name, "memory")
    provider = layer.dataProvider()
    provider.addAttributes([
        QgsField("event_label", QVariant.String),
        QgsField("lobe", QVariant.String),        # "compressional" | "background"
        QgsField("color_bin", QVariant.Int),       # -1 for background
        QgsField("strike1", QVariant.Double),
        QgsField("dip1", QVariant.Double),
        QgsField("rake1", QVariant.Double),
        QgsField("strike2", QVariant.Double),
        QgsField("dip2", QVariant.Double),
        QgsField("rake2", QVariant.Double),
        QgsField("selected_plane", QVariant.String),
        QgsField("cff_bar", QVariant.Double),
        QgsField("magnitude", QVariant.Double),
    ])
    layer.updateFields()

    for res, cff_bar in zip(results, cff_bar_values):
        ev = res["event"]
        np1 = NodalPlane(ev.strike1, ev.dip1, ev.rake1)
        colors, patches = plot_dc(np1, size=200, xy=(0, 0),
                                  width=(diameter_deg, diameter_deg))
        color_bin = _color_bin(cff_bar, vmax)

        for tag, patch in zip(colors, patches):
            verts = patch.get_path().vertices  # (N,2), degree-units centered at (0,0)
            pts = [QgsPointXY(ev.lon + vx, ev.lat + vy) for vx, vy in verts]
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPolygonXY([pts]))
            is_compressional = (tag == "b")
            feat.setAttributes([
                ev.label or "", "compressional" if is_compressional else "background",
                color_bin if is_compressional else -1,
                ev.strike1, ev.dip1, ev.rake1,
                ev.strike2 if ev.strike2 is not None else None,
                ev.dip2 if ev.dip2 is not None else None,
                ev.rake2 if ev.rake2 is not None else None,
                res["selected"], float(cff_bar),
                ev.magnitude if ev.magnitude is not None else None,
            ])
            provider.addFeature(feat)

    layer.updateExtents()

    # Rule-based renderer: one rule for the (always white) background
    # lobes, one rule per color bin for the compressional lobes -- 6
    # fixed-color rules total, avoiding data-defined-property APIs that
    # have changed names across QGIS versions.
    root = QgsRuleBasedRenderer.Rule(None)

    bg_symbol = QgsFillSymbol.createSimple({
        "color": "255,255,255,255", "outline_color": "40,40,40,255", "outline_width": "0.3"})
    bg_rule = QgsRuleBasedRenderer.Rule(bg_symbol, label="Background", filterExp="\"lobe\" = 'background'")
    root.appendChild(bg_rule)

    for i, (hexcolor, label) in enumerate(zip(_BIN_COLORS_HEX, _BIN_LABELS)):
        symbol = QgsFillSymbol.createSimple({
            "color": hexcolor, "outline_color": "40,40,40,255", "outline_width": "0.3"})
        rule = QgsRuleBasedRenderer.Rule(
            symbol, label=label,
            filterExp=f'"lobe" = \'compressional\' AND "color_bin" = {i}')
        root.appendChild(rule)

    layer.setRenderer(QgsRuleBasedRenderer(root))
    QgsProject.instance().addMapLayer(layer)
    return layer
