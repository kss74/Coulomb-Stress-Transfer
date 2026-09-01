# -*- coding: utf-8 -*-
"""
Ingest raw InSAR rasters (unwrapped LOS displacement + look-vector
geometry, from LiCSBAS/MintPy/GMTSAR/ISCE-style processing chains)
down into the same per-point schema observation_import.py's
"insar_los" path already produces from a hand-built table:
{"lon","lat","los","look_e","look_n","look_u","sigma"} -- exactly what
core.okada_engine.run_slip_inversion() / dc3d_worker.py's
"slip_inversion" mode expect (see that module's docstring).

This module fills the gap observation_import.py's docstring
deliberately left open ("Deliberately does NOT attempt raw InSAR
raster (GeoTIFF) ingestion or quadtree/uniform pixel downsampling --
that belongs upstream..."). This IS that upstream step. Output rows
from this module can be handed straight to
observation_import.build_observations_from_mapped_rows()'s "insar_los"
consumers, or straight into run_slip_inversion()'s los_observations
list -- the row schema is identical, deliberately.

──────────────────────────────────────────────────────────────────
LOOK-VECTOR CONVENTION -- read this before using load_los_with_angle_rasters()
──────────────────────────────────────────────────────────────────
Two supported raster inputs:

(1) ENU look-vector rasters (load_los_with_enu_rasters) -- three
    single-band rasters giving each pixel's already-resolved unit
    look vector (ground-to-satellite) east/north/up components
    directly. E.g. LiCSBAS's E.geo.tif / N.geo.tif / U.geo.tif.
    Sampled as-is: NO conversion, NO convention ambiguity. Prefer
    this path whenever your processing chain provides it.

(2) Incidence + azimuth/heading rasters (load_los_with_angle_rasters)
    -- two single-band rasters: incidence angle (from vertical,
    always positive) and EITHER
      angle_type="az"   -- LOS azimuth: ground-to-satellite,
                            measured from north, ANTI-CLOCKWISE
                            positive (ISCE/MintPy "az_angle"
                            convention; ISCE los.rdr band 2).
      angle_type="head" -- satellite heading/track angle, measured
                            from north, CLOCKWISE positive
                            (ROI_PAC/GAMMA "head_angle" convention --
                            this is what many raw processor products
                            actually ship, e.g. ~-12 deg for a
                            Sentinel-1 ascending track).

    Conversion formula (independently re-derived, then checked
    against MintPy's own reference implementation --
    mintpy.utils.utils0.get_unit_vector4component_of_interest(comp=
    "enu2los") / .enu2los(), https://github.com/insarlab/MintPy,
    src/mintpy/utils/utils0.py -- and against MintPy's own documented
    typical values table for near-polar sun-synchronous SAR missions;
    see verify_insar_raster_import.py for the numeric check):

        az = 90 - head              (right-looking radar; true for
                                      Sentinel-1, ALOS-2, TerraSAR-X,
                                      COSMO-SkyMed, ICEYE, RADARSAT-2
                                      -- pass look_direction="left"
                                      ONLY if you positively know your
                                      mission/mode is left-looking,
                                      which is rare)
        look_e = -sin(inc) * sin(az)
        look_n =  sin(inc) * cos(az)
        look_u =  cos(inc)

──────────────────────────────────────────────────────────────────
UNRESOLVED BY THIS MODULE, ON YOU TO VERIFY: LOS DISPLACEMENT SIGN
──────────────────────────────────────────────────────────────────
The sign of the LOS VALUES themselves (positive = motion toward the
satellite, vs positive = range increase / motion away from the
satellite) is processor-dependent and not standardized across chains.
dc3d_worker.py's slip-inversion dot product is:
    predicted_los = displacement_vector . look_vector
i.e. positive predicted LOS = motion TOWARD the satellite, using the
SAME ground-to-satellite look-vector sign convention this module
produces. If your processor's raw LOS raster uses the opposite sign,
negate it before/at import (`los_sign=-1`) -- this module has no way
to detect that mismatch for you, and getting it backwards silently
flips the recovered rake/slip sign in the inversion, not just its
magnitude.

──────────────────────────────────────────────────────────────────
NOT IMPLEMENTED (deliberately, scope decisions -- flag for a future
session if needed):
──────────────────────────────────────────────────────────────────
- Resampling mismatched-grid rasters onto a common grid. All rasters
  passed to one load_*() call must already share the same width,
  height, and geotransform (typical for products from a single
  processing run). Use `gdalwarp -te <xmin ymin xmax ymax> -tr <dx dy>`
  externally first if your LOS/incidence/heading/mask rasters differ
  in grid -- silently resampling here risked masking real registration
  errors between products from different runs.
- Coherence-weighted (as opposed to threshold-masked) downsampling.
  `mask_path`/`mask_valid_value` gives a hard include/exclude mask;
  turning coherence into a continuous per-point sigma is possible but
  not done here.
- Non-EPSG:4326 raster inputs are reprojected via gdal.Warp
  (bilinear) before sampling -- this is a real resampling step (unlike
  observation_import.py's point-layer path, which only transforms
  coordinates, not resamples data) and its accuracy is bounded by
  gdal.Warp's own quality, not independently re-verified here.

──────────────────────────────────────────────────────────────────
COMPONENT-DISPLACEMENT RASTERS (East/North/Vertical) -- a SEPARATE,
GNSS-SCHEMA path, not LOS
──────────────────────────────────────────────────────────────────
load_component_rasters() + downsample_components_uniform()/
downsample_components_quadtree() below read up to three already-
resolved DISPLACEMENT-COMPONENT rasters (East, North, Vertical/Up --
e.g. a decomposed multi-track InSAR product, or any other raster that
already carries physical displacement rather than a LOS/look-vector
pair), ANY ONE OR TWO of which may be omitted (a vertical-only product
is common and fully supported). Unlike the LOS path above, these are
independent physical quantities, not properties of one shared pixel
product, so partial per-pixel/per-raster availability is normal, not
an error.

Output rows are in observation_import.py's "gnss" schema
({"lon","lat","e","n","u","sigma_e","sigma_n","sigma_u"}), NOT
"insar_los" -- feed these into run_slip_inversion()'s `observations`
argument (component-wise), not `los_observations`, exactly like a
hand-built GNSS table/layer import would.
"""

import math
import uuid
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ─── Look-vector conversion (pure numpy/math -- no GDAL, testable here) ───

def heading_to_azimuth(head_deg, look_direction="right"):
    """Convert satellite orbit heading angle (clockwise-from-north,
    ROI_PAC/GAMMA convention) into LOS azimuth angle (anti-clockwise-
    from-north, ground-to-satellite, ISCE/MintPy convention).

    Formula matches mintpy.utils.utils0.heading2azimuth_angle() exactly
    (independently re-derived, then cross-checked against that
    reference): az = 90 - head for right-looking radar,
    az = -90 - head for left-looking.
    """
    if look_direction == "right":
        az = 90.0 - head_deg
    elif look_direction == "left":
        az = -90.0 - head_deg
    else:
        raise ValueError(f"look_direction must be 'right' or 'left', got {look_direction!r}")
    # wrap to (-180, 180]
    az = np.asarray(az, dtype=float)
    az = az - np.round(az / 360.0) * 360.0
    return az


def look_vector_from_incidence_azimuth(inc_deg, az_deg):
    """
    Ground-to-satellite unit look vector (E, N, U) from incidence angle
    (from vertical, degrees, always positive) and LOS azimuth angle
    (degrees, from north, anti-clockwise positive -- see module
    docstring for the "az" vs "head" distinction).

    look_e = -sin(inc) * sin(az)
    look_n =  sin(inc) * cos(az)
    look_u =  cos(inc)

    Matches mintpy.utils.utils0.get_unit_vector4component_of_interest(
    comp="enu2los") exactly -- see module docstring for citation and
    verify_insar_raster_import.py for the numeric cross-check against
    MintPy's own documented Sentinel-1 ascending/descending values.

    Accepts scalars or numpy arrays (any matching shape); returns
    (look_e, look_n, look_u) in the same shape.
    """
    inc = np.asarray(inc_deg, dtype=float)
    az = np.asarray(az_deg, dtype=float)
    inc_r = np.deg2rad(inc)
    az_r = np.deg2rad(az)
    look_e = -np.sin(inc_r) * np.sin(az_r)
    look_n = np.sin(inc_r) * np.cos(az_r)
    look_u = np.cos(inc_r)
    return look_e, look_n, look_u


def look_vector_from_incidence_heading(inc_deg, head_deg, look_direction="right"):
    """Convenience wrapper: heading_to_azimuth() then
    look_vector_from_incidence_azimuth()."""
    az_deg = heading_to_azimuth(head_deg, look_direction=look_direction)
    return look_vector_from_incidence_azimuth(inc_deg, az_deg)


# ─── GDAL raster I/O (only importable inside QGIS / an env with osgeo) ───

def check_gdal_available():
    """
    Verify `osgeo.gdal` is importable in THIS Python environment.
    Returns (ok: bool, message: str) -- same pattern as
    core.okada_engine.check_external_python()/_has_okada_wrapper().
    Unlike the DC3D external-Python check, GDAL is expected to already
    be present inside QGIS's OWN bundled Python (it ships GDAL for its
    own raster rendering), so this checks the current interpreter
    directly rather than a subprocess -- if this returns False while
    running inside real QGIS, something is unusually broken about that
    install, not just an optional extra that needs installing.
    """
    try:
        from osgeo import gdal  # noqa: F401
        return True, "osgeo.gdal is importable."
    except ImportError as e:
        return False, (f"osgeo.gdal is not importable in this Python "
                       f"environment ({e}). Raster InSAR import requires "
                       f"GDAL, normally bundled with QGIS itself -- if this "
                       f"is happening inside QGIS, the install may be "
                       f"unusually broken; the point-table/layer InSAR "
                       f"import (CSV/TSV or QGIS point layer) does not "
                       f"need GDAL and remains available regardless.")


def _read_band_as_array(path, band=1):
    """Read one band of a GDAL-supported raster as a float64 2D array
    (NaN where nodata), plus its geotransform, projection WKT, and
    (width, height). Only usable where `osgeo.gdal` is importable
    (QGIS's bundled Python, or a standalone install with GDAL) --
    same standing constraint as raster_utils.py's write_geotiff()."""
    from osgeo import gdal
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"GDAL could not open raster: {path!r}")
    b = ds.GetRasterBand(band)
    arr = b.ReadAsArray().astype(np.float64)
    nodata = b.GetNoDataValue()
    if nodata is not None:
        arr = np.where(np.isclose(arr, nodata), np.nan, arr)
    gt = ds.GetGeoTransform()
    proj_wkt = ds.GetProjection()
    shape = (ds.RasterYSize, ds.RasterXSize)
    ds = None
    return arr, gt, proj_wkt, shape


def _ensure_geographic(path, target_epsg=4326):
    """If `path`'s raster isn't already in EPSG:target_epsg, warp it
    (bilinear resample) into an in-memory (/vsimem/) copy and return
    that path instead; otherwise return `path` unchanged. See module
    docstring's "NOT IMPLEMENTED" note -- this IS a real resampling
    step, not just a coordinate relabeling."""
    from osgeo import gdal, osr
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"GDAL could not open raster: {path!r}")
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(ds.GetProjection())
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(target_epsg)
    if src_srs.IsSame(target_srs):
        ds = None
        return path
    vsi_path = f"/vsimem/insar_reproj_{uuid.uuid4().hex[:8]}.tif"
    gdal.Warp(vsi_path, ds, dstSRS=f"EPSG:{target_epsg}", resampleAlg="bilinear")
    ds = None
    return vsi_path


def _lonlat_grid_from_geotransform(gt, shape):
    """Pixel-CENTER lon/lat grid from a GDAL geotransform, shape
    (n_rows, n_cols). Matches the pixel-center convention used
    elsewhere in this codebase (see raster_profile.py / cff_volume.py
    grid conventions)."""
    n_rows, n_cols = shape
    gt0, gt1, gt2, gt3, gt4, gt5 = gt
    cols = np.arange(n_cols) + 0.5
    rows = np.arange(n_rows) + 0.5
    col_grid, row_grid = np.meshgrid(cols, rows)
    lon2d = gt0 + col_grid * gt1 + row_grid * gt2
    lat2d = gt3 + col_grid * gt4 + row_grid * gt5
    return lon2d, lat2d


def _check_same_grid(shapes, label_a, label_b):
    if shapes[0] != shapes[1]:
        raise RuntimeError(
            f"{label_a} raster shape {shapes[0]} does not match {label_b} raster "
            f"shape {shapes[1]}. This module requires all input rasters for one "
            f"load_*() call to already share the same grid -- resample them onto "
            f"a common grid first (e.g. `gdalwarp -te <xmin ymin xmax ymax> "
            f"-tr <dx dy>`); see module docstring's NOT IMPLEMENTED section.")


def _apply_mask(arr2d, mask_path, mask_valid_value, target_epsg):
    if mask_path is None:
        return arr2d
    mask_path_geo = _ensure_geographic(mask_path, target_epsg)
    mask_arr, _, _, mask_shape = _read_band_as_array(mask_path_geo)
    _check_same_grid((arr2d.shape, mask_shape), "data", "mask")
    if mask_valid_value is None:
        valid = mask_arr > 0
    else:
        valid = np.isclose(mask_arr, mask_valid_value)
    out = arr2d.copy()
    out[~valid] = np.nan
    return out


def load_los_with_enu_rasters(los_path, e_path, n_path, u_path,
                              sigma_path=None, mask_path=None,
                              mask_valid_value=None, target_epsg=4326):
    """
    Read LOS + already-resolved E/N/U look-vector rasters (e.g.
    LiCSBAS's E.geo.tif/N.geo.tif/U.geo.tif) sharing one grid.

    Returns (lon2d, lat2d, los2d, look_e2d, look_n2d, look_u2d, sigma2d)
    -- all 2D float arrays, NaN wherever any input is nodata/masked.
    sigma2d is None if sigma_path not given.
    """
    los_path = _ensure_geographic(los_path, target_epsg)
    los2d, gt, _, shape = _read_band_as_array(los_path)
    lon2d, lat2d = _lonlat_grid_from_geotransform(gt, shape)

    def _load_matching(path, label):
        p = _ensure_geographic(path, target_epsg)
        arr, _, _, s = _read_band_as_array(p)
        _check_same_grid((shape, s), "LOS", label)
        return arr

    look_e2d = _load_matching(e_path, "look_e")
    look_n2d = _load_matching(n_path, "look_n")
    look_u2d = _load_matching(u_path, "look_u")
    sigma2d = _load_matching(sigma_path, "sigma") if sigma_path else None

    los2d = _apply_mask(los2d, mask_path, mask_valid_value, target_epsg)
    return lon2d, lat2d, los2d, look_e2d, look_n2d, look_u2d, sigma2d


def load_los_with_angle_rasters(los_path, inc_path, angle_path,
                                angle_type="az", look_direction="right",
                                sigma_path=None, mask_path=None,
                                mask_valid_value=None, target_epsg=4326,
                                los_sign=1.0):
    """
    Read LOS + incidence + azimuth-or-heading rasters sharing one grid,
    converting incidence+angle to an ENU look vector via
    look_vector_from_incidence_azimuth()/_heading() -- see module
    docstring for the convention and its citation.

    angle_type: "az" (LOS azimuth, ISCE/MintPy convention) or "head"
        (satellite heading, ROI_PAC/GAMMA convention) -- see module
        docstring; get this wrong and every look vector is rotated
        180 degrees off the true one.
    los_sign: multiply raw LOS raster values by this (1.0 or -1.0) to
        match dc3d_worker.py's ground-to-satellite dot-product sign
        convention -- see module docstring's LOS SIGN section. This
        module cannot verify which sign your processor uses; default
        1.0 assumes it already matches.

    Returns (lon2d, lat2d, los2d, look_e2d, look_n2d, look_u2d, sigma2d)
    -- same shape as load_los_with_enu_rasters().
    """
    if angle_type not in ("az", "head"):
        raise ValueError(f"angle_type must be 'az' or 'head', got {angle_type!r}")

    los_path = _ensure_geographic(los_path, target_epsg)
    los2d, gt, _, shape = _read_band_as_array(los_path)
    lon2d, lat2d = _lonlat_grid_from_geotransform(gt, shape)

    def _load_matching(path, label):
        p = _ensure_geographic(path, target_epsg)
        arr, _, _, s = _read_band_as_array(p)
        _check_same_grid((shape, s), "LOS", label)
        return arr

    inc2d = _load_matching(inc_path, "incidence")
    angle2d = _load_matching(angle_path, "azimuth/heading")
    sigma2d = _load_matching(sigma_path, "sigma") if sigma_path else None

    if angle_type == "head":
        look_e2d, look_n2d, look_u2d = look_vector_from_incidence_heading(
            inc2d, angle2d, look_direction=look_direction)
    else:
        look_e2d, look_n2d, look_u2d = look_vector_from_incidence_azimuth(inc2d, angle2d)

    los2d = los2d * float(los_sign)
    los2d = _apply_mask(los2d, mask_path, mask_valid_value, target_epsg)
    return lon2d, lat2d, los2d, look_e2d, look_n2d, look_u2d, sigma2d


# ─── Downsampling (pure numpy -- no GDAL, fully testable here) ───

def _row_dict(lon, lat, los, le, ln, lu, sigma):
    return {"lon": float(lon), "lat": float(lat), "los": float(los),
            "look_e": float(le), "look_n": float(ln), "look_u": float(lu),
            "sigma": (float(sigma) if sigma is not None and np.isfinite(sigma) else None)}


def downsample_uniform(lon2d, lat2d, los2d, look_e2d, look_n2d, look_u2d,
                       stride, sigma2d=None):
    """
    Simple stride-based decimation: keep every `stride`-th pixel along
    both axes. Pixels where los/look_e/look_n/look_u is NaN are
    dropped (not substituted). Fast and deterministic, but spends the
    same density everywhere regardless of how much the LOS field
    actually varies there -- use downsample_quadtree() instead when
    fault-proximal resolution matters more than a flat/uniform point
    count.

    Returns a list of dicts in exactly observation_import.py's
    "insar_los" row schema -- feed straight to
    run_slip_inversion()'s los_observations or to
    observation_import.build_observations_from_mapped_rows()-shaped
    consumers.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    rows = []
    ny, nx = los2d.shape
    for i in range(0, ny, stride):
        for j in range(0, nx, stride):
            los, le, ln, lu = los2d[i, j], look_e2d[i, j], look_n2d[i, j], look_u2d[i, j]
            if not (np.isfinite(los) and np.isfinite(le) and np.isfinite(ln) and np.isfinite(lu)):
                continue
            sigma = sigma2d[i, j] if sigma2d is not None else None
            rows.append(_row_dict(lon2d[i, j], lat2d[i, j], los, le, ln, lu, sigma))
    return rows


@dataclass
class _Cell:
    r0: int
    r1: int  # exclusive
    c0: int
    c1: int  # exclusive


def _auto_std_threshold(los2d):
    """75th percentile of the field's local 3x3-window standard
    deviation, ignoring NaN -- a data-driven default split threshold
    (see downsample_quadtree() docstring)."""
    ny, nx = los2d.shape
    stds = []
    # sample every 2nd interior pixel for speed on large rasters
    for i in range(1, ny - 1, 2):
        row = los2d[i - 1:i + 2, :]
        for j in range(1, nx - 1, 2):
            win = row[:, j - 1:j + 2]
            if np.any(np.isfinite(win)):
                s = np.nanstd(win)
                if np.isfinite(s):
                    stds.append(s)
    if not stds:
        return 0.0
    return float(np.percentile(stds, 75))


def downsample_quadtree(lon2d, lat2d, los2d, look_e2d, look_n2d, look_u2d,
                        max_points=2000, min_cell_px=4, std_threshold=None,
                        sigma2d=None, sigma_fallback="none"):
    """
    Adaptive quadtree decimation on the LOS field (Simons et al.
    2002-style; standard practice for InSAR slip-inversion input
    reduction). A cell is split into 4 quadrants when its valid-pixel
    LOS standard deviation exceeds `std_threshold` AND it's still
    larger than `min_cell_px` on its shorter side; otherwise it
    becomes ONE output point at the centroid of its valid pixels, with
    los/look_e/look_n/look_u averaged over those pixels.

    Per-point sigma: `sigma2d`'s cell-mean if given. If NOT given,
    `sigma_fallback` decides what happens (2026-08-30 fix -- see below
    for why the old unconditional behaviour was a real bug):

      "none" (default) -- sigma is left as None for every leaf, i.e.
          run_slip_inversion() weights every point EQUALLY. This is
          the safe default: with no real uncertainty raster, there is
          no actual measurement-noise estimate to weight by.
      "local_std" -- the LEGACY behaviour: each leaf's own LOS
          standard deviation within its cell is used as a stand-in
          sigma. Kept available for anyone who explicitly wants
          density-adaptive weighting, but NOT the default any more --
          see the warning below.

    WHY "local_std" is not a safe default: for a smooth analytic
    displacement field (the normal case near a well-resolved fault),
    a quadtree cell's local std reflects the field's own CURVATURE/
    ROUGHNESS there, not measurement noise -- flat far-field cells get
    a near-zero std, hence an enormous implied 1/sigma weight (observed
    up to ~8000x in one real validation run against a published
    finite-fault model), while the steep, highly-informative near-fault
    cells get down-weighted. This isn't a hypothetical: it was the
    actual, confirmed root cause of scipy.optimize.lsq_linear failing
    to converge (hitting max_iter with an implausible recovered Mw)
    when this quadtree output was fed straight into
    run_slip_inversion() -- switching this fallback to "none" (equal
    weighting) was what made the SAME inversion converge cleanly and
    recover close to the correct moment/spatial slip pattern. See
    PROJECT_HANDOVER_ADDENDUM_2026-08-30_quadtree_sigma_fallback.md.

    `std_threshold`, if None (default): set automatically via
    _auto_std_threshold() to the 75th percentile of the full-
    resolution field's local 3x3 standard deviation, so cells subdivide
    roughly where the field is locally more variable than 3/4 of the
    scene. Override directly for denser/coarser output. (This still
    uses local std for the SPLIT decision regardless of
    sigma_fallback -- that use is fine, since it's a splitting
    heuristic, not a claimed noise estimate fed into a weighted solve.)

    `max_points` is a soft cap enforced breadth-first: cells are
    processed largest-first (a FIFO queue seeded with the whole
    raster, children appended to the back when split), so the budget
    is spent on the coarsest structure first and only the deepest few
    subdivisions are ever skipped once it's used up -- once
    len(leaves)+len(queue) reaches max_points, all remaining queued
    cells stop splitting and are emitted as-is (never silently
    dropped, just coarser than they might otherwise be).

    Returns a list of dicts in exactly observation_import.py's
    "insar_los" row schema.
    """
    if min_cell_px < 1:
        raise ValueError(f"min_cell_px must be >= 1, got {min_cell_px}")
    if sigma_fallback not in ("none", "local_std"):
        raise ValueError(
            f"sigma_fallback must be 'none' or 'local_std', got {sigma_fallback!r}")
    ny, nx = los2d.shape
    if std_threshold is None:
        std_threshold = _auto_std_threshold(los2d)

    def _valid_mask(cell):
        sub = los2d[cell.r0:cell.r1, cell.c0:cell.c1]
        sub_e = look_e2d[cell.r0:cell.r1, cell.c0:cell.c1]
        sub_n = look_n2d[cell.r0:cell.r1, cell.c0:cell.c1]
        sub_u = look_u2d[cell.r0:cell.r1, cell.c0:cell.c1]
        return np.isfinite(sub) & np.isfinite(sub_e) & np.isfinite(sub_n) & np.isfinite(sub_u)

    root = _Cell(0, ny, 0, nx)
    queue = deque([root])
    leaves = []

    while queue:
        cell = queue.popleft()
        h = cell.r1 - cell.r0
        w = cell.c1 - cell.c0
        vmask = _valid_mask(cell)
        if not np.any(vmask):
            continue  # no data in this cell at all -- drop, not a leaf

        sub_los = los2d[cell.r0:cell.r1, cell.c0:cell.c1]
        cell_std = float(np.nanstd(sub_los[vmask])) if np.sum(vmask) > 1 else 0.0

        can_split = (h >= 2 * min_cell_px) or (w >= 2 * min_cell_px)
        # only split along an axis that's still big enough; if one axis
        # is already at the floor, still allow splitting the other
        room_budget = (len(leaves) + len(queue)) < max_points
        if can_split and cell_std > std_threshold and room_budget:
            rmid = cell.r0 + max(1, h // 2)
            cmid = cell.c0 + max(1, w // 2)
            children = [
                _Cell(cell.r0, rmid, cell.c0, cmid),
                _Cell(cell.r0, rmid, cmid, cell.c1),
                _Cell(rmid, cell.r1, cell.c0, cmid),
                _Cell(rmid, cell.r1, cmid, cell.c1),
            ]
            # drop degenerate zero-area children (odd-sized cells at floor)
            queue.extend(ch for ch in children if ch.r1 > ch.r0 and ch.c1 > ch.c0)
            continue

        # emit as a leaf
        rs, cs = np.where(vmask)
        rr = rs + cell.r0
        cc = cs + cell.c0
        lon_c = float(np.mean(lon2d[rr, cc]))
        lat_c = float(np.mean(lat2d[rr, cc]))
        los_c = float(np.mean(los2d[rr, cc]))
        le_c = float(np.mean(look_e2d[rr, cc]))
        ln_c = float(np.mean(look_n2d[rr, cc]))
        lu_c = float(np.mean(look_u2d[rr, cc]))
        if sigma2d is not None:
            svals = sigma2d[rr, cc]
            svals = svals[np.isfinite(svals)]
            sigma_c = float(np.mean(svals)) if svals.size else None
        elif sigma_fallback == "local_std":
            sigma_c = cell_std if cell_std > 0 else None
        else:
            sigma_c = None
        leaves.append(_row_dict(lon_c, lat_c, los_c, le_c, ln_c, lu_c, sigma_c))

    return leaves


# ─── Component-displacement rasters (East/North/Vertical -- GNSS schema) ──
#
# Separate from everything above: these read up to three independent
# DISPLACEMENT-COMPONENT rasters (not LOS + look vector), any one or
# two of which may be omitted, and produce observation_import.py's
# "gnss" row schema instead of "insar_los" -- see module docstring's
# "COMPONENT-DISPLACEMENT RASTERS" section.

def load_component_rasters(e_path=None, n_path=None, u_path=None,
                           sigma_e_path=None, sigma_n_path=None,
                           sigma_u_path=None, mask_path=None,
                           mask_valid_value=None, target_epsg=4326):
    """
    Read up to three already-resolved displacement-component rasters
    (East, North, Vertical/Up), any ONE OR TWO of which may be None --
    unlike load_los_with_enu_rasters()/load_los_with_angle_rasters()
    (which always need LOS + a full 3-component look vector together,
    since those are properties of one shared physical pixel product),
    E/N/U displacement components are independent physical quantities
    and it is common to have, say, only a vertical (UD) product.

    At least one of e_path/n_path/u_path is required (raises
    ValueError otherwise). Grid alignment (_check_same_grid) is
    checked against whichever of e/n/u is loaded FIRST (in that
    order); each sigma raster is checked against its own matching
    component and requires that component to also be given.

    Returns (lon2d, lat2d, e2d, n2d, u2d, sigma_e2d, sigma_n2d, sigma_u2d)
    -- e2d/n2d/u2d and their sigmas are None wherever not REQUESTED at
    all (distinct from NaN, which means "requested but masked/nodata
    at this pixel") -- downstream code (downsample_components_uniform/
    _quadtree) relies on this None-vs-NaN distinction.
    """
    if not any([e_path, n_path, u_path]):
        raise ValueError("At least one of e_path/n_path/u_path is required.")

    ref_shape = None
    ref_gt = None

    def _load(path, label):
        nonlocal ref_shape, ref_gt
        if path is None:
            return None
        p = _ensure_geographic(path, target_epsg)
        arr, gt, _, shape = _read_band_as_array(p)
        if ref_shape is None:
            ref_shape = shape
            ref_gt = gt
        else:
            _check_same_grid((ref_shape, shape), "first-loaded component", label)
        return arr

    e2d = _load(e_path, "East displacement")
    n2d = _load(n_path, "North displacement")
    u2d = _load(u_path, "Vertical displacement")

    lon2d, lat2d = _lonlat_grid_from_geotransform(ref_gt, ref_shape)

    def _load_sigma(path, label, ref_arr):
        if path is None:
            return None
        if ref_arr is None:
            raise ValueError(
                f"{label} sigma raster given but its matching {label} "
                f"displacement raster was not -- a sigma raster without "
                f"its own component makes no sense here.")
        p = _ensure_geographic(path, target_epsg)
        arr, _, _, shape = _read_band_as_array(p)
        _check_same_grid((ref_shape, shape), "first-loaded component", f"{label} sigma")
        return arr

    sigma_e2d = _load_sigma(sigma_e_path, "East", e2d)
    sigma_n2d = _load_sigma(sigma_n_path, "North", n2d)
    sigma_u2d = _load_sigma(sigma_u_path, "Vertical", u2d)

    if e2d is not None:
        e2d = _apply_mask(e2d, mask_path, mask_valid_value, target_epsg)
    if n2d is not None:
        n2d = _apply_mask(n2d, mask_path, mask_valid_value, target_epsg)
    if u2d is not None:
        u2d = _apply_mask(u2d, mask_path, mask_valid_value, target_epsg)

    return lon2d, lat2d, e2d, n2d, u2d, sigma_e2d, sigma_n2d, sigma_u2d


def _row_dict_components(lon, lat, e, n, u, sigma_e, sigma_n, sigma_u):
    def _f(v):
        return float(v) if v is not None and np.isfinite(v) else None
    return {"lon": float(lon), "lat": float(lat),
            "e": _f(e), "n": _f(n), "u": _f(u),
            "sigma_e": _f(sigma_e), "sigma_n": _f(sigma_n), "sigma_u": _f(sigma_u)}


def downsample_components_uniform(lon2d, lat2d, e2d, n2d, u2d, stride,
                                  sigma_e2d=None, sigma_n2d=None, sigma_u2d=None):
    """
    Component-raster counterpart of downsample_uniform() -- stride
    decimation, but tolerant of PARTIAL per-pixel availability: a
    pixel is kept if AT LEAST ONE of the provided e2d/n2d/u2d channels
    is finite there (not ALL of them, unlike the LOS path, where
    los/look_e/look_n/look_u are properties of the same physical pixel
    and one being missing makes the whole row meaningless). Whichever
    channel(s) are NaN/absent at a kept pixel come out as None in the
    row -- observation_import.py's "gnss" schema already expects rows
    with only some of e/n/u present (e.g. leveling data).

    Returns a list of dicts in observation_import.py's "gnss" row
    schema -- NOT "insar_los".
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    channels = [c for c in (e2d, n2d, u2d) if c is not None]
    if not channels:
        raise ValueError("At least one of e2d/n2d/u2d is required.")
    ny, nx = channels[0].shape
    rows = []
    for i in range(0, ny, stride):
        for j in range(0, nx, stride):
            e = e2d[i, j] if e2d is not None else None
            n = n2d[i, j] if n2d is not None else None
            u = u2d[i, j] if u2d is not None else None
            if not any(v is not None and np.isfinite(v) for v in (e, n, u)):
                continue
            se = sigma_e2d[i, j] if sigma_e2d is not None else None
            sn = sigma_n2d[i, j] if sigma_n2d is not None else None
            su = sigma_u2d[i, j] if sigma_u2d is not None else None
            rows.append(_row_dict_components(lon2d[i, j], lat2d[i, j], e, n, u, se, sn, su))
    return rows


def downsample_components_quadtree(lon2d, lat2d, e2d, n2d, u2d,
                                   max_points=2000, min_cell_px=4,
                                   std_threshold=None,
                                   sigma_e2d=None, sigma_n2d=None, sigma_u2d=None,
                                   sigma_fallback="none"):
    """
    Adaptive quadtree decimation for component-displacement rasters --
    same breadth-first, max_points-capped splitting strategy as
    downsample_quadtree() above, generalized to whichever of e2d/n2d/
    u2d are provided (any subset) and to per-pixel PARTIAL
    availability (see downsample_components_uniform() docstring for
    why this differs from the LOS path's all-or-nothing validity mask).

    Split decision: each provided channel gets its OWN auto std
    threshold (via _auto_std_threshold(), unless std_threshold
    overrides all of them with one shared value); a cell splits if
    ANY one channel's local std exceeds ITS OWN threshold, even if the
    others are flat there -- so, e.g., a vertical-only import still
    subdivides sensibly using only the vertical field's own
    variability, and a joint E+N+U import subdivides wherever the
    NOISIEST of the three warrants it.

    Per-leaf output: lon/lat centroid is the UNION of pixels valid in
    ANY provided channel (not the intersection, unlike the LOS path,
    since these channels can have independently-shaped nodata/masks).
    Each channel's own value is then averaged only over ITS OWN valid
    pixels within that cell -- None (not a fabricated number) if that
    channel has zero valid pixels there.

    Per-channel sigma: that channel's own sigma raster cell-mean if
    given. If NOT given, `sigma_fallback` decides what happens
    (2026-08-30 fix -- SAME issue and SAME fix as
    downsample_quadtree() above; see that function's docstring for the
    full rationale, repeated only briefly here):

      "none" (default) -- sigma is left as None for every leaf/channel
          without its own sigma raster, i.e. equal weighting in
          run_slip_inversion(). Safe default.
      "local_std" -- LEGACY: that channel's own local std within the
          cell, used as a stand-in sigma. Confirmed to destabilize
          run_slip_inversion() (implied weight ratios up to ~8000x
          between a flat far-field cell and a genuinely informative
          near-fault cell, causing scipy.optimize.lsq_linear to fail
          to converge) -- kept available but opt-in only.

    Returns a list of dicts in observation_import.py's "gnss" row
    schema.
    """
    channels = {"e": e2d, "n": n2d, "u": u2d}
    provided = {k: v for k, v in channels.items() if v is not None}
    if not provided:
        raise ValueError("At least one of e2d/n2d/u2d is required.")
    if min_cell_px < 1:
        raise ValueError(f"min_cell_px must be >= 1, got {min_cell_px}")
    if sigma_fallback not in ("none", "local_std"):
        raise ValueError(
            f"sigma_fallback must be 'none' or 'local_std', got {sigma_fallback!r}")

    ny, nx = next(iter(provided.values())).shape

    if std_threshold is None:
        thresholds = {k: _auto_std_threshold(v) for k, v in provided.items()}
    else:
        thresholds = {k: std_threshold for k in provided}

    valid = {k: np.isfinite(v) for k, v in provided.items()}
    valid_any = np.zeros((ny, nx), dtype=bool)
    for v in valid.values():
        valid_any |= v

    sigma_map = {"e": sigma_e2d, "n": sigma_n2d, "u": sigma_u2d}

    root = _Cell(0, ny, 0, nx)
    queue = deque([root])
    leaves = []

    while queue:
        cell = queue.popleft()
        h = cell.r1 - cell.r0
        w = cell.c1 - cell.c0
        sub_any = valid_any[cell.r0:cell.r1, cell.c0:cell.c1]
        if not np.any(sub_any):
            continue  # no data from ANY channel in this cell -- drop

        # Worst-case (std / own-threshold) ratio across whichever
        # channels are provided -- ratio > 1 in any channel triggers a
        # split, so a flat channel never masks a noisy one.
        cell_std_ratio = 0.0
        for k, arr in provided.items():
            sub = arr[cell.r0:cell.r1, cell.c0:cell.c1]
            sub_valid = valid[k][cell.r0:cell.r1, cell.c0:cell.c1]
            if np.sum(sub_valid) > 1:
                s = float(np.nanstd(sub[sub_valid]))
                thr = thresholds[k] if thresholds[k] > 0 else 1e-12
                cell_std_ratio = max(cell_std_ratio, s / thr)

        can_split = (h >= 2 * min_cell_px) or (w >= 2 * min_cell_px)
        room_budget = (len(leaves) + len(queue)) < max_points
        if can_split and cell_std_ratio > 1.0 and room_budget:
            rmid = cell.r0 + max(1, h // 2)
            cmid = cell.c0 + max(1, w // 2)
            children = [
                _Cell(cell.r0, rmid, cell.c0, cmid),
                _Cell(cell.r0, rmid, cmid, cell.c1),
                _Cell(rmid, cell.r1, cell.c0, cmid),
                _Cell(rmid, cell.r1, cmid, cell.c1),
            ]
            queue.extend(ch for ch in children if ch.r1 > ch.r0 and ch.c1 > ch.c0)
            continue

        # emit as a leaf -- centroid over the UNION of valid pixels
        rs, cs = np.where(sub_any)
        rr = rs + cell.r0
        cc = cs + cell.c0
        lon_c = float(np.mean(lon2d[rr, cc]))
        lat_c = float(np.mean(lat2d[rr, cc]))

        vals, sigmas = {}, {}
        for k, arr in provided.items():
            sub_valid = valid[k][cell.r0:cell.r1, cell.c0:cell.c1]
            if np.any(sub_valid):
                rs_k, cs_k = np.where(sub_valid)
                rr_k, cc_k = rs_k + cell.r0, cs_k + cell.c0
                vals[k] = float(np.mean(arr[rr_k, cc_k]))
                s2d = sigma_map[k]
                if s2d is not None:
                    svals = s2d[rr_k, cc_k]
                    svals = svals[np.isfinite(svals)]
                    sigmas[k] = float(np.mean(svals)) if svals.size else None
                elif sigma_fallback == "local_std":
                    cell_std_k = float(np.nanstd(arr[rr_k, cc_k])) if rr_k.size > 1 else 0.0
                    sigmas[k] = cell_std_k if cell_std_k > 0 else None
                else:
                    sigmas[k] = None
            else:
                vals[k] = None
                sigmas[k] = None

        leaves.append(_row_dict_components(
            lon_c, lat_c,
            vals.get("e"), vals.get("n"), vals.get("u"),
            sigmas.get("e"), sigmas.get("n"), sigmas.get("u")))

    return leaves
