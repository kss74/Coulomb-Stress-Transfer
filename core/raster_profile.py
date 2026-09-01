# -*- coding: utf-8 -*-
"""
Sample a raster along a lon/lat profile line, for the topographic-profile
cross-section panel(s) (core.cross_section_config.TopoPanelConfig).

Two source kinds, matching TopoPanelConfig.source_kind:

  "raster_file" -- any GDAL-readable file path. Uses rasterio (pip-
                   installable with a bundled GDAL, same "optional pip
                   dependency, checked at runtime" pattern as
                   okada_wrapper / obspy elsewhere in this project --
                   see _has_rasterio()).
  "qgis_layer"  -- an already-loaded QgsRasterLayer, sampled through its
                   own data provider (works even for formats/CRSs QGIS
                   handles but a bare rasterio install might not, and
                   needs no extra dependency when running inside QGIS).
                   qgis.core is imported lazily inside the function body,
                   matching vector_utils.py's convention, so this module
                   still imports cleanly outside QGIS.

Both paths assume the raster is in (or the QGIS layer reports) geographic
WGS84 lon/lat coordinates, consistent with every other lon/lat assumption
in this plugin (the CFF grid, fault positions, etc. are all WGS84 too).
Reprojecting a differently-CRS'd raster on the fly is future work, not
attempted here.
"""

import numpy as np

from .geo_profile import project_points_to_profile


def _has_rasterio():
    try:
        import rasterio  # noqa: F401
        return True
    except ImportError:
        return False


def _sample_rasterio_file(path, lons, lats, band=1):
    import rasterio
    with rasterio.open(path) as ds:
        coords = list(zip(lons.tolist(), lats.tolist()))
        values = np.array([v[band - 1] for v in ds.sample(coords)], dtype=float)
        nodata = ds.nodata
    if nodata is not None:
        values = np.where(values == nodata, np.nan, values)
    return values


def _sample_qgis_raster_layer(layer, lons, lats, band=1):
    from qgis.core import QgsPointXY

    provider = layer.dataProvider()
    values = np.full(len(lons), np.nan, dtype=float)
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        result = provider.sample(QgsPointXY(float(lon), float(lat)), band)
        # QGIS provider.sample() returns (value, ok)
        val, ok = result if isinstance(result, tuple) else (result, True)
        if ok and val is not None:
            values[i] = val
    return values


def sample_raster_along_line(source, source_kind, lon1, lat1, lon2, lat2,
                              n_samples=300, band=1,
                              elevation_unit_divisor=1000.0):
    """
    Sample `source` (file path if source_kind="raster_file", QgsRasterLayer
    instance if source_kind="qgis_layer") at n_samples evenly-spaced points
    along the profile lon1,lat1 -> lon2,lat2.

    Returns (dist_km, elevation_km):
      dist_km       : distance from the profile start (km), length n_samples
      elevation_km  : sampled raster values / elevation_unit_divisor
                      (default divisor 1000 converts metres -> km to match
                      the cross-section's depth axis units). NaN where the
                      raster has nodata or no coverage.

    Raises RuntimeError if source_kind="raster_file" and rasterio isn't
    installed, with a message pointing at how to install it -- same
    pattern as _has_okada_wrapper()'s guidance in okada_engine.py.
    """
    lons = np.linspace(lon1, lon2, n_samples)
    lats = np.linspace(lat1, lat2, n_samples)
    dist_km, _perp_km, _len_km = project_points_to_profile(
        lons, lats, lon1, lat1, lon2, lat2)

    if source_kind == "raster_file":
        if not _has_rasterio():
            raise RuntimeError(
                "Sampling a raster FILE for the topographic profile requires "
                "the 'rasterio' package in QGIS's own Python environment "
                "(pip install rasterio). Alternatively, load the raster as "
                "a QGIS layer first and use source_kind='qgis_layer', which "
                "needs no extra dependency.")
        elevation = _sample_rasterio_file(source, lons, lats, band=band)
    elif source_kind == "qgis_layer":
        elevation = _sample_qgis_raster_layer(source, lons, lats, band=band)
    else:
        raise ValueError(f"Unknown source_kind: {source_kind!r}")

    return dist_km, elevation / elevation_unit_divisor


def sample_raster_along_polyline(source, source_kind, vertices, n_samples=300,
                                  band=1, unit_divisor=1.0, sign=1.0):
    """
    Multi-segment counterpart of sample_raster_along_line(), for an
    "Extra Depth-Section Element" (core.cross_section_config.
    ExtraSectionLineConfig) sourced from a raster instead of a digitized
    vector line -- e.g. a Slab2 slab-interface depth grid, or any other
    raster-format subducting-slab/seismic-horizon dataset (these are
    frequently distributed as rasters, not vector picks).

    `vertices` is the SAME (lon, lat) list used everywhere else for a
    multi-segment cross-section profile (core.geo_profile.
    polyline_segment_info()) -- sampling directly off the profile's own
    current geometry (rather than a separately-digitized line's fixed
    points) means this element automatically tracks edits to the
    profile's start/finish/waypoints, the same way a topo panel
    resamples its raster fresh every time the section is recomputed.

    Implementation mirrors okada_engine.compute_cross_section_multi()'s
    "call the existing single-leg function once per leg, then stitch"
    approach -- no new projection math, just per-leg reuse of
    sample_raster_along_line()'s own rasterio/QGIS sampling helpers.
    n_samples is distributed across legs in proportion to each leg's
    share of the total profile length (at least 2 samples per leg, so
    a short leg still gets its own two endpoints); the shared vertex at
    each internal leg boundary is sampled once, not duplicated, same as
    compute_cross_section_multi()'s column de-duplication.

    sign: +1.0 if the raster's values are already positive-DOWN depth;
          -1.0 if the raster stores positive-UP elevation, or a
          negative-down depth (as many Slab2 grids do), and needs
          flipping to match this plugin's positive-down depth
          convention on the main cross-section panel.
    unit_divisor: raster units -> km (1.0 if already km, e.g. most
          Slab2 grids; 1000.0 for a metres-unit DEM/bathymetry raster).

    Returns (lons, lats, dist_km, depth_km), each length n_samples
    (approximately -- see the per-leg rounding above): lons/lats are
    each sample's own geographic position, ready to be zipped into
    (lon, lat, depth_km) vertices the same shape as a "vector_line"
    ExtraSectionLineConfig's, or projected directly with
    core.geo_profile.project_points_to_polyline() like any other
    extra-line source.

    Raises RuntimeError/ValueError under the same conditions as
    sample_raster_along_line() (missing rasterio for a file source, or
    an unrecognized source_kind).
    """
    from .geo_profile import polyline_segment_info

    if len(vertices) < 2:
        raise ValueError("A profile polyline needs at least 2 vertices.")
    if source_kind == "raster_file" and not _has_rasterio():
        raise RuntimeError(
            "Sampling a raster FILE for an extra depth-section element "
            "requires the 'rasterio' package in QGIS's own Python "
            "environment (pip install rasterio). Alternatively, load the "
            "raster as a QGIS layer first and use source_kind='qgis_layer', "
            "which needs no extra dependency.")
    if source_kind not in ("raster_file", "qgis_layer"):
        raise ValueError(f"Unknown source_kind: {source_kind!r}")

    seg_info = polyline_segment_info(vertices)
    total_len = seg_info["total_length_km"]
    n_legs = len(vertices) - 1

    lon_chunks, lat_chunks, dist_chunks = [], [], []
    for leg_i in range(n_legs):
        lon_a, lat_a = vertices[leg_i]
        lon_b, lat_b = vertices[leg_i + 1]
        leg_len = seg_info["segment_length_km"][leg_i]
        n_leg = max(2, int(round(n_samples * leg_len / total_len))) \
            if total_len > 1e-9 else 2
        leg_lons = np.linspace(lon_a, lon_b, n_leg)
        leg_lats = np.linspace(lat_a, lat_b, n_leg)
        leg_dist, _perp, _len = project_points_to_profile(
            leg_lons, leg_lats, lon_a, lat_a, lon_b, lat_b)
        offset = seg_info["cumulative_dist_km"][leg_i]
        if leg_i == 0:
            lon_chunks.append(leg_lons)
            lat_chunks.append(leg_lats)
            dist_chunks.append(leg_dist + offset)
        else:
            # Drop the first sample of every leg after the first -- it
            # coincides with the previous leg's last sample (both are
            # the shared waypoint between the two legs).
            lon_chunks.append(leg_lons[1:])
            lat_chunks.append(leg_lats[1:])
            dist_chunks.append(leg_dist[1:] + offset)

    lons = np.concatenate(lon_chunks)
    lats = np.concatenate(lat_chunks)
    dist_km = np.concatenate(dist_chunks)

    if source_kind == "raster_file":
        raw = _sample_rasterio_file(source, lons, lats, band=band)
    else:
        raw = _sample_qgis_raster_layer(source, lons, lats, band=band)

    depth_km = sign * (raw / unit_divisor)
    return lons, lats, dist_km, depth_km
