# -*- coding: utf-8 -*-
"""
Gather point-feature annotations (core.cross_section_config.
AnnotationOverlayConfig) for the cross-section tool's topo panel(s) --
Phase 2 UI backend (PROJECT_HANDOVER_ADDENDUM_2026-08-18b, continued
2026-08-19: "topo/annotation picker UI").

Two source kinds, matching AnnotationOverlayConfig.source_kind,
deliberately mirroring core.raster_profile's own "file"/"qgis_layer"
split for topo panels:

  "qgis_layer" -- an already-loaded QGIS point vector layer (resolved
                  from a layer id/name to the actual QgsVectorLayer
                  object by the UI layer before calling this, same
                  convention as core.raster_profile.
                  sample_raster_along_line()'s `source`). Reuses
                  core.observation_import.read_qgis_layer_table(), the
                  same generic attribute+geometry reader the EQ-
                  catalog/observation importers already use, instead
                  of a second hand-rolled QGIS feature-iteration loop.
  "file"       -- a CSV/TSV with lon/lat columns (+ an optional label
                  column), fuzzy-matched the same lightweight way as
                  core.raster_profile's column handling.

Both paths reduce to the same (lons, lats, labels) triple, which is
then projected onto the profile and filtered to the search-width swath
using the exact same core.geo_profile helpers every other cross-
section overlay (EQ catalog, fault traces, focal mechanisms) already
uses -- one projection convention, not a second one invented for this
overlay.
"""

import csv

import numpy as np

from .geo_profile import project_points_to_profile, filter_within_search_width

_LON_ALIASES = ["lon", "lng", "long", "longitude", "x"]
_LAT_ALIASES = ["lat", "latitude", "y"]


def _find_column(fieldnames, aliases):
    lower = {f.lower(): f for f in fieldnames}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def _read_file_source(path, label_field=None, z_field=None):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        lon_col = _find_column(fieldnames, _LON_ALIASES)
        lat_col = _find_column(fieldnames, _LAT_ALIASES)
        if lon_col is None or lat_col is None:
            raise ValueError(
                f"Could not find lon/lat columns in {path!r} "
                f"(columns present: {fieldnames})")
        lbl_col = label_field if (label_field and label_field in fieldnames) else None
        z_col = z_field if (z_field and z_field in fieldnames) else None
        lons, lats, labels, zvals = [], [], [], []
        for row in reader:
            try:
                lon_val = float(row[lon_col])
                lat_val = float(row[lat_col])
            except (TypeError, ValueError):
                continue
            lons.append(lon_val)
            lats.append(lat_val)
            labels.append(row.get(lbl_col, "") if lbl_col else "")
            if z_col:
                try:
                    zvals.append(float(row[z_col]))
                except (TypeError, ValueError):
                    zvals.append(np.nan)
    z_arr = np.array(zvals, dtype=float) if z_col else None
    return np.array(lons, dtype=float), np.array(lats, dtype=float), labels, z_arr


def _read_qgis_layer_source(layer, label_field=None, z_field=None):
    from .observation_import import read_qgis_layer_table

    _fields, rows = read_qgis_layer_table(layer)
    lons, lats, labels, zvals = [], [], [], []
    for row in rows:
        gx, gy = row.get("__geom_x__"), row.get("__geom_y__")
        if gx is None or gy is None:
            continue
        try:
            lon_val = float(gx)
            lat_val = float(gy)
        except (TypeError, ValueError):
            continue
        lons.append(lon_val)
        lats.append(lat_val)
        labels.append(row.get(label_field, "") if label_field else "")
        if z_field:
            zraw = row.get(z_field)
            try:
                zvals.append(float(zraw))
            except (TypeError, ValueError):
                zvals.append(np.nan)
    z_arr = np.array(zvals, dtype=float) if z_field else None
    return np.array(lons, dtype=float), np.array(lats, dtype=float), labels, z_arr


def gather_annotation_points(source_kind, source, label_field,
                              lon1, lat1, lon2, lat2, half_width_km,
                              z_field=None):
    """
    Returns (dist_km, labels, z_vals): already filtered to the
    search-width swath and clipped to the profile segment, in exactly
    the shape core.cross_section_plot.build_cross_section_figure()'s
    `annotation_data` expects (one (dist_km_arr, labels_list, z_arr)
    tuple per AnnotationOverlayConfig entry -- z_arr is None when
    z_field is None, which build_cross_section_figure() treats as "no
    real vertical value, pin near the panel top" for backward
    compatibility with sources that don't have one).

    source_kind : "file" (source = path) or "qgis_layer" (source = an
                  already-resolved QgsVectorLayer instance).
    z_field     : optional column/attribute name giving each point's
                  own vertical value (elevation, depth -- whatever the
                  target topo panel's y-axis represents). Missing/
                  unparseable values become NaN, not a dropped row --
                  matplotlib simply won't draw a marker with a NaN
                  y-position, so one bad row doesn't lose the rest.
    """
    if source_kind == "file":
        lons, lats, labels, zvals = _read_file_source(source, label_field, z_field)
    elif source_kind == "qgis_layer":
        lons, lats, labels, zvals = _read_qgis_layer_source(source, label_field, z_field)
    else:
        raise ValueError(f"Unknown source_kind {source_kind!r} "
                          "(must be 'file' or 'qgis_layer')")

    if len(lons) == 0:
        return np.array([]), [], (np.array([]) if z_field else None)

    dist_along, perp, plen = project_points_to_profile(lons, lats, lon1, lat1, lon2, lat2)
    mask = filter_within_search_width(dist_along, perp, plen, half_width_km)
    labels_arr = np.array(labels, dtype=object)
    z_out = zvals[mask] if zvals is not None else None
    return dist_along[mask], list(labels_arr[mask]), z_out


def check():
    """Verification per project convention: numerical spot-check."""
    import tempfile
    import os as _os

    with tempfile.TemporaryDirectory() as d:
        path = _os.path.join(d, "annotations.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Longitude", "Latitude", "Name"])
            w.writerow([-1.0, -1.0, "on_profile"])       # sits on the profile start
            w.writerow([-1.0, -0.5, "on_profile_mid"])   # partway along, on-line
            w.writerow([5.0, 5.0, "far_away"])           # way off the profile

        dist_km, labels = gather_annotation_points(
            "file", path, "Name", lon1=-1.0, lat1=-1.0, lon2=-1.0, lat2=0.0,
            half_width_km=5.0)
        assert len(dist_km) == 2, f"expected 2 on-profile points, got {len(dist_km)}"
        assert set(labels) == {"on_profile", "on_profile_mid"}, labels
        assert dist_km.min() >= -1e-6

        # Column-alias robustness (lowercase, "lng").
        path2 = _os.path.join(d, "annotations2.csv")
        with open(path2, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lng", "lat", "label"])
            w.writerow([-1.0, -1.0, "aliased"])
        dist_km2, labels2 = gather_annotation_points(
            "file", path2, "label", lon1=-1.0, lat1=-1.0, lon2=-1.0, lat2=0.0,
            half_width_km=5.0)
        assert list(labels2) == ["aliased"], labels2

    print("annotation_gather.check(): all assertions passed.")


if __name__ == "__main__":
    check()
