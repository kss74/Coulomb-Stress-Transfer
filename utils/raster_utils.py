# -*- coding: utf-8 -*-
"""Export CFF grids to GeoTIFF/CSV/XYZ and load as QGIS raster layers."""

import numpy as np
import csv
import os
import tempfile
import uuid


def write_csv(path, lon2d, lat2d, values, value_name="cff"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lon", "lat", value_name])
        for i in range(lon2d.shape[0]):
            for j in range(lon2d.shape[1]):
                writer.writerow([lon2d[i, j], lat2d[i, j], values[i, j]])


def write_xyz(path, lon2d, lat2d, values):
    with open(path, "w") as f:
        for i in range(lon2d.shape[0]):
            for j in range(lon2d.shape[1]):
                f.write(f"{lon2d[i, j]:.6f} {lat2d[i, j]:.6f} {values[i, j]:.6f}\n")


def write_csv_multi(path, lon2d, lat2d, columns: dict):
    """
    Like write_csv(), but for grids with MULTIPLE value columns at each
    point -- e.g. the optimally-oriented-plane result, which has two
    conjugate planes' strike/dip/rake/CFF at every grid point (added
    2026-08-11 alongside the "Optimal Faults" UI tab; the two conjugate
    planes generally have DIFFERENT CFF since the coseismic-CFF-change
    fix in optimal_plane.py, so a single-column export would silently
    hide one plane's values).

    columns : dict[str, 2D array] -- every array must share lon2d's shape.
              Column order in the CSV follows dict insertion order.
    """
    names = list(columns.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lon", "lat"] + names)
        for i in range(lon2d.shape[0]):
            for j in range(lon2d.shape[1]):
                row = [lon2d[i, j], lat2d[i, j]] + [columns[name][i, j] for name in names]
                writer.writerow(row)


def write_geotiff(path, lon2d, lat2d, values):
    """Write a GeoTIFF using GDAL (available inside QGIS)."""
    from osgeo import gdal, osr

    n_lat, n_lon = values.shape
    lon_min, lon_max = lon2d.min(), lon2d.max()
    lat_min, lat_max = lat2d.min(), lat2d.max()
    px_w = (lon_max - lon_min) / n_lon
    px_h = (lat_max - lat_min) / n_lat

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, n_lon, n_lat, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((lon_min, px_w, 0, lat_max, 0, -px_h))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())

    band = ds.GetRasterBand(1)
    # values array is [lat, lon]; GDAL expects rows top-to-bottom (north first)
    band.WriteArray(np.flipud(values))
    band.SetNoDataValue(float("nan"))
    band.FlushCache()
    ds = None


def write_geotiff_multiband(path, lon2d, lat2d, values_3d, band_descriptions=None):
    """
    Write a multi-band GeoTIFF: one band per 2D slice of `values_3d`
    (shape (n_bands, n_lat, n_lon)) -- e.g. one band per forecast time
    step at a single fixed depth slice
    (core.rate_state_seismicity.RateStateForecast.rate[depth_idx] has
    shape (n_lat, n_lon, n_t); the caller transposes to
    (n_t, n_lat, n_lon) before calling this). Same georeferencing
    convention as write_geotiff() (each band array is [lat, lon],
    written north-first via np.flipud, same EPSG:4326 assumption).

    `band_descriptions`, if given, must have one string per band and is
    written as that band's GDAL band description -- visible in QGIS's
    Layer Properties > Information panel and in the band selector, so
    a user stepping through bands can tell which forecast time each one
    is (e.g. "t=12.5") without cross-referencing a separate manifest.
    """
    from osgeo import gdal, osr

    n_bands, n_lat, n_lon = values_3d.shape
    lon_min, lon_max = lon2d.min(), lon2d.max()
    lat_min, lat_max = lat2d.min(), lat2d.max()
    px_w = (lon_max - lon_min) / n_lon
    px_h = (lat_max - lat_min) / n_lat

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, n_lon, n_lat, n_bands, gdal.GDT_Float32)
    ds.SetGeoTransform((lon_min, px_w, 0, lat_max, 0, -px_h))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())

    for b in range(n_bands):
        band = ds.GetRasterBand(b + 1)   # GDAL bands are 1-indexed
        band.WriteArray(np.flipud(values_3d[b]))
        band.SetNoDataValue(float("nan"))
        if band_descriptions is not None and b < len(band_descriptions):
            band.SetDescription(band_descriptions[b])
        band.FlushCache()
    ds = None


def load_raster_layer(path, layer_name="CFF", sequential=False, band=1):
    """
    Load a GeoTIFF as a QGIS raster layer. `sequential=False` (default,
    unchanged behavior for every existing caller) uses the original
    diverging blue-white-red ramp centred on 0, appropriate for
    sign-varying fields like ΔCFF. `sequential=True` instead uses a
    manually-specified viridis-like ramp spanning the band's own
    min..max with NO zero-centering, for fields that are always
    non-negative by construction -- e.g.
    core.rate_state_seismicity.RateStateForecast.rate/.cumulative,
    which the d94() closed-form solution guarantees stay >= 0 (see that
    function's own docstring) -- a diverging ramp on such a field would
    waste half its range on a sign that never occurs, same reasoning
    ui.plot_widget.PlotWidget.plot_rate_state_map() already uses for its
    own on-screen preview (this function is the GeoTIFF/QGIS-layer
    equivalent of that same choice, not a separate one).

    The manual 5-stop ramp (not matplotlib's own viridis) is used for
    the same reason the existing diverging ramp is hand-specified
    rather than pulled from matplotlib: this module runs inside QGIS's
    own Python environment where importing matplotlib's colormap
    machinery just for 5 RGB triples isn't worth a new dependency here.

    `band`, default 1: which band's statistics/legend the pseudocolor
    renderer is built from -- relevant for a multi-band GeoTIFF from
    write_geotiff_multiband() (e.g. a rate-and-state forecast's
    multiple time steps), where QGIS's own band selector lets the user
    switch which band is displayed, but the initial style needs one
    band's stats to build against.
    """
    from qgis.core import (QgsRasterLayer, QgsProject,
                            QgsColorRampShader, QgsRasterShader,
                            QgsSingleBandPseudoColorRenderer)
    from qgis.PyQt.QtGui import QColor

    layer = QgsRasterLayer(path, layer_name)
    if not layer.isValid():
        return None

    provider = layer.dataProvider()
    stats = provider.bandStatistics(band)

    ramp = QgsColorRampShader()
    ramp.setColorRampType(QgsColorRampShader.Interpolated)

    if sequential:
        vmin = max(stats.minimumValue, 0.0)
        vmax = max(stats.maximumValue, vmin + 1e-12)
        stops = [
            (vmin, QColor(68, 1, 84)),
            (vmin + 0.25 * (vmax - vmin), QColor(59, 82, 139)),
            (vmin + 0.50 * (vmax - vmin), QColor(33, 145, 140)),
            (vmin + 0.75 * (vmax - vmin), QColor(94, 201, 98)),
            (vmax, QColor(253, 231, 37)),
        ]
        ramp.setColorRampItemList([
            QgsColorRampShader.ColorRampItem(v, c, f"{v:.3g}") for v, c in stops
        ])
    else:
        vmax = max(abs(stats.minimumValue), abs(stats.maximumValue), 1e-6)
        ramp.setColorRampItemList([
            QgsColorRampShader.ColorRampItem(-vmax, QColor(33, 66, 160), f"{-vmax:.3f}"),
            QgsColorRampShader.ColorRampItem(0, QColor(255, 255, 255), "0"),
            QgsColorRampShader.ColorRampItem(vmax, QColor(178, 24, 43), f"{vmax:.3f}"),
        ])

    shader = QgsRasterShader()
    shader.setRasterShaderFunction(ramp)
    renderer = QgsSingleBandPseudoColorRenderer(provider, band, shader)
    layer.setRenderer(renderer)

    QgsProject.instance().addMapLayer(layer)
    return layer


def _qgis_temp_dir():
    """
    A persistent-for-the-session temp directory for plugin-generated
    rasters that are added to the project without an explicit save
    dialog. Files here survive for the QGIS session (not deleted on
    each call) so the layer's underlying file remains valid, but live
    under the OS temp directory so they don't clutter the user's
    working folders.
    """
    d = os.path.join(tempfile.gettempdir(), "coulomb_stress_transfer")
    os.makedirs(d, exist_ok=True)
    return d


def add_raster_to_project(lon2d, lat2d, values, layer_name="Coulomb Stress Change",
                          sequential=False):
    """
    Write `values` to a GeoTIFF in a managed temp location and add it to
    the QGIS project directly — no save dialog. This is the "quick look"
    workflow; use write_geotiff()+load_raster_layer() (or the Save to
    File... UI action) when the user wants to choose a permanent path.

    `sequential`: see load_raster_layer()'s own docstring -- pass True
    for always-non-negative fields (e.g. a rate-and-state forecast
    snapshot) so the added layer doesn't get the default diverging
    ΔCFF-style ramp.

    Returns the created QgsRasterLayer (or None if it failed to load).
    """
    path = os.path.join(_qgis_temp_dir(), f"{layer_name.replace(' ', '_')}_{uuid.uuid4().hex[:8]}.tif")
    write_geotiff(path, lon2d, lat2d, values)
    return load_raster_layer(path, layer_name, sequential=sequential)


def add_multiband_raster_to_project(lon2d, lat2d, values_3d,
                                    layer_name="Rate-and-State Forecast",
                                    band_descriptions=None, sequential=True):
    """
    Multi-band equivalent of add_raster_to_project(): writes
    `values_3d` (shape (n_bands, n_lat, n_lon)) via
    write_geotiff_multiband() to a managed temp GeoTIFF and adds it to
    the QGIS project as a single multi-band raster layer -- e.g. one
    band per forecast time step of a rate-and-state seismicity
    snapshot at one depth slice, so a user can step through time using
    QGIS's own band selector instead of re-running an export per time
    step. `sequential` defaults to True here (unlike
    add_raster_to_project()'s False default) because this function's
    only current caller (RateStateForecastDialog's raster export) is
    always a non-negative rate/cumulative field, not a sign-varying
    ΔCFF field.

    The added layer's initial pseudocolor style is built from band 1's
    statistics only (see load_raster_layer()'s `band` parameter) --
    switching bands in QGIS keeps that same style/ramp rather than
    re-stretching per band, which is intentional: a fixed ramp across
    bands is what makes "the color at time t vs time t'" visually
    comparable, whereas per-band auto-stretch would not be.

    Returns the created QgsRasterLayer (or None if it failed to load).
    """
    path = os.path.join(_qgis_temp_dir(), f"{layer_name.replace(' ', '_')}_{uuid.uuid4().hex[:8]}.tif")
    write_geotiff_multiband(path, lon2d, lat2d, values_3d, band_descriptions=band_descriptions)
    return load_raster_layer(path, layer_name, sequential=sequential, band=1)
