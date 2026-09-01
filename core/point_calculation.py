# -*- coding: utf-8 -*-
"""
Point Calculator (2026-09-01 addition).

Computes Coulomb stress (ΔCFF, shear, normal — resolved onto the
plugin's single shared receiver plane, same convention as the Receiver
Fault tab / core.okada_engine.compute_cff_on_receiver_faults()) AND
predicted coseismic displacement, at arbitrary observation points given
the plugin's current source faults.

Primary use case ("validation"): a point carries a field-measured
displacement/slip vector (e.g. a GNSS coseismic offset, a leveling
benchmark, or a measured surface-rupture slip vector decomposed into
East/North/Up) — the model's predicted displacement at that exact point
is compared against the observation, producing per-component residuals
and a combined misfit. This is the same "compare predicted vs measured
surface deformation" idea already used for slip INVERSION
(core.okada_engine.run_slip_inversion() consumes the same E/N/U
GNSS-style observations) — this module runs the FORWARD direction
instead: given a known/assumed fault (or set of faults), evaluate what
it predicts at a point and see how well that matches what was actually
measured. A point with no observed displacement mapped is still fully
useful — it just returns the predicted stress/displacement with no
residual columns populated.

Input: a delimited text file (CSV/TSV) or a QGIS point layer's
attribute table (via core.observation_import.read_qgis_layer_table(),
reused as-is — same flexible-column-mapping workflow already used for
slip-inversion GNSS/InSAR observations, aftershock catalogs, and focal
mechanisms elsewhere in this plugin).

Required columns: lon, lat, elev_m (elevation in METRES, POSITIVE UP /
NEGATIVE DOWN — i.e. a leveling benchmark 5 m below the ground surface
is elev_m=-5, and a GNSS antenna 2 m above a monument is elev_m=+2).
This is the opposite sign convention from the rest of the plugin's
internal depth_km (positive DOWN) -- deliberately, because "elevation"
is how field data is naturally reported (GNSS/leveling benchmarks,
DEMs), and converting it correctly (see elev_m_to_depth_km() below) is
this module's job, not the user's.

Optional columns: obs_e, obs_n, obs_u (observed displacement/slip
vector components, metres, geographic East/North/Up) — see the
"E/N/U, not lateral+vertical" design note below — plus optional
1-sigma uncertainties (sigma_e, sigma_n, sigma_u) and a point label.
"""

import csv
import difflib
import io
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

from .focal_mechanism_import import read_delimited_table  # generic CSV/TSV reader, reused as-is

# ─── Schema ──────────────────────────────────────────────────────────────
#
# DESIGN NOTE ("figure out if it should be N and E or it can be total"):
# observed slip/displacement is accepted as geographic East/North/Up
# components (obs_e, obs_n, obs_u), NOT as a "lateral total + azimuth"
# or "lateral + vertical" pair. Reasons:
#   1. This is the EXACT SAME "gnss" schema core.observation_import.py
#      already uses for slip-inversion input -- reusing it means a user
#      who already has a GNSS/leveling table for slip inversion can
#      point the SAME file at this tool with no reformatting, and any
#      future schema fix (units, sign convention) only has to happen
#      once.
#   2. "Lateral" is ambiguous for a field measurement that isn't aligned
#      with a specific fault's strike (which one, if multiple sources
#      are loaded?) -- East/North is unambiguous regardless of how many
#      source faults are involved or what orientation they have.
#   3. Nothing is lost: a "total lateral slip" figure is just
#      hypot(obs_e, obs_n), which this module computes anyway
#      (obs_horiz_mag_m/pred_horiz_mag_m) and reports alongside the
#      components -- so both views are available from the one input
#      schema, rather than forcing a choice at import time.
# A row needs at least one of obs_e/obs_n/obs_u to be treated as a
# validation point; a row with none of the three is still imported (for
# the predicted-stress/displacement-only use case) but reports no
# residuals.

POSITION_FIELDS = ("lon", "lat", "elev_m")
OPTIONAL_FIELDS = ("label", "obs_e", "obs_n", "obs_u", "sigma_e", "sigma_n", "sigma_u")
ALL_FIELDS = POSITION_FIELDS + OPTIONAL_FIELDS

FIELD_DISPLAY_NAMES = {
    "lon": "Longitude", "lat": "Latitude",
    "elev_m": "Elevation (m, +up / -down)",
    "label": "Point ID/label (optional)",
    "obs_e": "Observed East disp./slip (m, optional)",
    "obs_n": "Observed North disp./slip (m, optional)",
    "obs_u": "Observed Up disp./slip (m, optional)",
    "sigma_e": "1σ East uncertainty (m, optional)",
    "sigma_n": "1σ North uncertainty (m, optional)",
    "sigma_u": "1σ Up uncertainty (m, optional)",
}

ALIASES: Dict[str, List[str]] = {
    "lon": ["lon", "long", "longitude", "x", "easting"],
    "lat": ["lat", "latitude", "y", "northing"],
    "elev_m": ["elev", "elevm", "elevation", "elevationm", "z", "altitude", "alt",
              "height", "heightm", "elevationmasl", "gpsheight"],
    "label": ["label", "name", "id", "station", "site", "pointid", "pointname", "stationname"],
    "obs_e": ["obse", "e", "east", "ue", "de", "dx", "eastdisp", "dispe", "edisp",
             "veast", "slipe", "eslip", "lateralE"],
    "obs_n": ["obsn", "n", "north", "un", "dn", "dy", "northdisp", "dispn", "ndisp",
             "vnorth", "slipn", "nslip", "lateralN"],
    "obs_u": ["obsu", "u", "up", "uz", "du", "dz", "vertical", "verticaldisp", "dispu",
             "udisp", "vup", "slipu", "verticalslip", "uplift", "subsidence"],
    "sigma_e": ["sigmae", "sige", "se", "stde", "stde_m", "eerr", "erre", "esigma"],
    "sigma_n": ["sigman", "sign", "sn", "stdn", "stdn_m", "nerr", "errn", "nsigma"],
    "sigma_u": ["sigmau", "sigu", "su", "stdu", "stdu_m", "uerr", "erru", "usigma"],
}


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def suggest_column_mapping(columns: List[str], fields: List[str]) -> Dict[str, Optional[str]]:
    """Same alias-then-fuzzy matching strategy as
    core.observation_import.suggest_column_mapping() / this project's
    other importers -- see those docstrings. Never raises."""
    norm_cols = {c: _normalize(c) for c in columns}
    mapping: Dict[str, Optional[str]] = {}
    used = set()
    for f in fields:
        best = None
        for alias in ALIASES.get(f, [f]):
            for col, norm in norm_cols.items():
                if col in used:
                    continue
                if norm == alias:
                    best = col
                    break
            if best:
                break
        if best is None:
            candidates = [c for c in columns if c not in used]
            aliases = ALIASES.get(f, [f])
            scored = []
            for col in candidates:
                norm = norm_cols[col]
                score = max(difflib.SequenceMatcher(None, norm, a).ratio() for a in aliases)
                scored.append((score, col))
            scored.sort(reverse=True)
            if scored and scored[0][0] >= 0.6:
                best = scored[0][1]
        mapping[f] = best
        if best:
            used.add(best)
    return mapping


def detect_point_mapping(columns: List[str]) -> Dict[str, Optional[str]]:
    """Best-effort column guess across ALL_FIELDS, regardless of whether
    the required lon/lat/elev_m end up mapped -- the dialog decides
    whether the result is good enough to enable Compute."""
    return suggest_column_mapping(columns, list(ALL_FIELDS))


# ─── Row -> ObservationPoint ────────────────────────────────────────────

@dataclass
class ObservationPoint:
    lon: float
    lat: float
    elev_m: float
    label: str = ""
    obs_e: Optional[float] = None
    obs_n: Optional[float] = None
    obs_u: Optional[float] = None
    sigma_e: Optional[float] = None
    sigma_n: Optional[float] = None
    sigma_u: Optional[float] = None


@dataclass
class PointImportResult:
    points: List[ObservationPoint] = field(default_factory=list)
    n_skipped: int = 0
    errors: List[str] = field(default_factory=list)


def _get_float(row: Dict[str, str], column_map: Dict[str, Optional[str]], f: str):
    col = column_map.get(f)
    if not col:
        return None
    val = row.get(col, "")
    if val == "" or val is None:
        return None
    return float(val)


def build_points_from_mapped_rows(rows: List[Dict[str, str]],
                                  column_map: Dict[str, Optional[str]]) -> PointImportResult:
    """
    Convert mapped table rows into ObservationPoint objects. lon/lat/
    elev_m are required per row (a row missing any of these is skipped
    and recorded in .errors); every OPTIONAL_FIELDS column is read
    independently and defaults to None when unmapped or blank -- a row
    with none of obs_e/obs_n/obs_u is still imported (predicted-only
    use case), unlike core.observation_import's "gnss" schema, which
    requires at least one of e/n/u since an inversion row with none of
    them is otherwise useless. Here, a point with only lon/lat/elev_m
    is a perfectly valid "what does the model predict here" query.
    """
    out_points: List[ObservationPoint] = []
    errors: List[str] = []
    for i, row in enumerate(rows):
        row_num = i + 2  # 1-indexed + header row
        try:
            lon = _get_float(row, column_map, "lon")
            lat = _get_float(row, column_map, "lat")
            elev_m = _get_float(row, column_map, "elev_m")
            if lon is None or lat is None or elev_m is None:
                raise ValueError("missing lon/lat/elev_m")

            label_col = column_map.get("label")
            label = str(row.get(label_col, "")).strip() if label_col else ""
            if not label:
                label = f"P{i + 1}"

            out_points.append(ObservationPoint(
                lon=lon, lat=lat, elev_m=elev_m, label=label,
                obs_e=_get_float(row, column_map, "obs_e"),
                obs_n=_get_float(row, column_map, "obs_n"),
                obs_u=_get_float(row, column_map, "obs_u"),
                sigma_e=_get_float(row, column_map, "sigma_e"),
                sigma_n=_get_float(row, column_map, "sigma_n"),
                sigma_u=_get_float(row, column_map, "sigma_u"),
            ))
        except (ValueError, KeyError, TypeError) as e:
            errors.append(f"row {row_num}: {e}")

    return PointImportResult(points=out_points, n_skipped=len(errors), errors=errors)


# ─── Elevation/depth convention ─────────────────────────────────────────

def elev_m_to_depth_km(elev_m: float) -> Tuple[float, bool]:
    """
    Convert the user-facing Z convention (elev_m, metres, POSITIVE UP /
    NEGATIVE DOWN -- see module docstring) into this plugin's internal
    depth_km convention (km, POSITIVE DOWN) used throughout
    core.okada_engine / dc3d_worker.py.

    Points ABOVE the model's z=0 free surface (elev_m > 0, e.g. a GNSS
    antenna phase centre or a leveling rod a couple of metres up a
    monument) are CLAMPED to depth_km=0.0 (the surface) rather than
    passed through as a negative depth: the Okada/DC3D elastic
    half-space formulation has no defined solution for an observation
    point above the free surface (there is no medium there to have a
    stress/strain state at all) -- see okada_wrapper's own domain
    restriction. Field elevations of this kind are almost always small
    (metres) relative to fault dimensions and source depths (km), so
    evaluating at the surface instead introduces negligible error for
    realistic input; the `clamped` flag lets the caller flag/report it
    per point regardless, rather than silently discarding the
    information.

    Points below the surface (elev_m < 0, e.g. a borehole strainmeter
    or a slip measurement made on a fault scarp face at some depth) map
    directly and are NOT clamped.

    Returns (depth_km, clamped).
    """
    depth_km = -elev_m / 1000.0
    if depth_km < 0.0:
        return 0.0, True
    return depth_km, False


# ─── Output column order (shared by the QGIS layer + CSV export) ───────

RESULT_COLUMNS = [
    "label", "lon", "lat", "elev_m", "depth_km", "elevation_clamped", "used_dc3d",
    "sxx_pa", "syy_pa", "szz_pa", "sxy_pa", "sxz_pa", "syz_pa",
    "cff_bar", "shear_bar", "normal_bar",
    "pred_e_m", "pred_n_m", "pred_u_m", "pred_horiz_mag_m", "pred_azimuth_deg",
    "obs_e_m", "obs_n_m", "obs_u_m", "obs_horiz_mag_m", "obs_azimuth_deg",
    "resid_e_m", "resid_n_m", "resid_u_m", "resid_horiz_mag_m", "resid_3d_mag_m",
    "n_resid_components", "chi2_e", "chi2_n", "chi2_u",
]


# ─── Core computation ────────────────────────────────────────────────────

def compute_point_results(sources, points: List[ObservationPoint], receiver, elastic,
                          progress_callback=None) -> List[dict]:
    """
    For each ObservationPoint, evaluate the combined effect of all
    `sources` (the plugin's current stress-source faults) and report:

      - the full stress tensor (sxx_pa..syz_pa; geographic East/North/
        Down frame, Pa, tension-positive -- same convention as
        core.okada_engine's other stress outputs)
      - ΔCFF/shear/normal (bar) resolved onto `receiver`'s strike/dip/
        rake -- the SAME single shared receiver-plane convention as the
        Receiver Fault tab / compute_cff_on_receiver_faults(), not a
        per-point receiver orientation (see the handover addendum for
        this design choice and how to extend it later)
      - predicted coseismic displacement (pred_e_m/pred_n_m/pred_u_m,
        metres) and its horizontal magnitude/azimuth
      - IF the point carries an observed displacement/slip vector
        (obs_e/obs_n/obs_u -- any subset), the residual per available
        component (observed - predicted), the combined horizontal and
        full 3-D residual magnitude (using whichever components are
        available -- see n_resid_components), and per-component
        chi-square (only where BOTH the observation and its 1-sigma
        uncertainty are supplied)

    DEPTH HANDLING mirrors compute_cff_on_receiver_faults()'s existing
    strategy exactly (no new physics, just the same combination applied
    at points instead of receiver-fault centroids): elev_m<=0 (at or
    above the surface, clamped to depth_km=0) uses the validated z=0
    surface formula (exact, vectorized); elev_m<0 (below the surface)
    uses the external-Python DC3D worker (Okada 1992) if one is
    configured and working, falling back to the surface formula with
    `used_dc3d=False` per point otherwise -- exactly like every other
    depth-dependent computation in this plugin.

    Returns a list of dicts (see RESULT_COLUMNS for keys), one per
    point, in the same order as `points`.
    """
    from .okada_engine import (geo_to_km, _stress_from_surface_strain, okada85_surface,
                               resolve_cff_shear_normal, _has_okada_wrapper,
                               _run_dc3d_worker_points_full)

    n = len(points)
    if n == 0:
        return []

    depth_km = np.zeros(n)
    clamped = np.zeros(n, dtype=bool)
    for i, p in enumerate(points):
        d, c = elev_m_to_depth_km(p.elev_m)
        depth_km[i] = d
        clamped[i] = c

    sxx = np.zeros(n); syy = np.zeros(n); szz = np.zeros(n)
    sxy = np.zeros(n); sxz = np.zeros(n); syz = np.zeros(n)
    ue = np.zeros(n); un = np.zeros(n); uz = np.zeros(n)
    used_dc3d = np.zeros(n, dtype=bool)

    def _accumulate_surface(idx):
        """Fill sxx..syz/ue/un/uz at the given index array using the
        validated z=0 surface formula (Okada 1985), summed over every
        source. `used_dc3d` for these indices is left as-is (False by
        default, or unchanged if this is a DC3D-fallback call)."""
        if len(idx) == 0:
            return
        lons = np.array([points[i].lon for i in idx])
        lats = np.array([points[i].lat for i in idx])
        s_acc = dict(sxx=np.zeros(len(idx)), syy=np.zeros(len(idx)), szz=np.zeros(len(idx)),
                    sxy=np.zeros(len(idx)), sxz=np.zeros(len(idx)), syz=np.zeros(len(idx)))
        ue_acc = np.zeros(len(idx)); un_acc = np.zeros(len(idx)); uz_acc = np.zeros(len(idx))
        for src in sources:
            e_km, n_km = geo_to_km(lons, lats, src.lon, src.lat)
            s = _stress_from_surface_strain(e_km, n_km, src, elastic.mu, elastic.nu)
            for k in s_acc:
                s_acc[k] = s_acc[k] + s[k]
            ue_s, un_s, uz_s = okada85_surface(
                e_km, n_km, src.depth, src.strike, src.dip,
                src.length, src.width, src.rake, src.slip, 0., elastic.nu)
            ue_acc = ue_acc + ue_s; un_acc = un_acc + un_s; uz_acc = uz_acc + uz_s
        sxx[idx] = s_acc['sxx']; syy[idx] = s_acc['syy']; szz[idx] = s_acc['szz']
        sxy[idx] = s_acc['sxy']; sxz[idx] = s_acc['sxz']; syz[idx] = s_acc['syz']
        ue[idx] = ue_acc; un[idx] = un_acc; uz[idx] = uz_acc

    surface_idx = np.where(depth_km <= 0.0)[0]
    depth_idx = np.where(depth_km > 0.0)[0]

    _accumulate_surface(surface_idx)
    # used_dc3d already False (default) for surface_idx.

    if progress_callback: progress_callback(30)

    if len(depth_idx) > 0:
        if _has_okada_wrapper():
            try:
                pts = [(points[i].lon, points[i].lat, float(depth_km[i])) for i in depth_idx]
                (sxx_d, syy_d, szz_d, sxy_d, sxz_d, syz_d,
                 ue_d, un_d, uz_d) = _run_dc3d_worker_points_full(sources, elastic, pts)
                sxx[depth_idx] = sxx_d; syy[depth_idx] = syy_d; szz[depth_idx] = szz_d
                sxy[depth_idx] = sxy_d; sxz[depth_idx] = sxz_d; syz[depth_idx] = syz_d
                ue[depth_idx] = ue_d; un[depth_idx] = un_d; uz[depth_idx] = uz_d
                used_dc3d[depth_idx] = True
            except Exception:
                # Fall back to the surface formula for these points
                # (depth ignored), clearly flagged via used_dc3d=False --
                # same fallback semantics as compute_cff_on_receiver_faults()
                # and compute_surface_deformation_depth().
                _accumulate_surface(depth_idx)
        else:
            _accumulate_surface(depth_idx)

    if progress_callback: progress_callback(70)

    stress = dict(sxx=sxx, syy=syy, szz=szz, sxy=sxy, sxz=sxz, syz=syz)
    cff_pa, shear_pa, normal_pa = resolve_cff_shear_normal(stress, receiver, elastic.friction)
    # Pa -> MPa -> bar (1 MPa = 10 bar), matching this plugin's existing
    # display convention (see ui/receiver_results_widget.py / vector_utils.py).
    cff_bar = cff_pa / 1e6 * 10.0
    shear_bar = shear_pa / 1e6 * 10.0
    normal_bar = normal_pa / 1e6 * 10.0

    pred_horiz_mag = np.hypot(ue, un)
    pred_azimuth = np.degrees(np.arctan2(ue, un)) % 360.0  # 0=North, 90=East

    results = []
    for i, p in enumerate(points):
        row = {
            "label": p.label, "lon": p.lon, "lat": p.lat, "elev_m": p.elev_m,
            "depth_km": float(depth_km[i]), "elevation_clamped": bool(clamped[i]),
            "used_dc3d": bool(used_dc3d[i]),
            "sxx_pa": float(sxx[i]), "syy_pa": float(syy[i]), "szz_pa": float(szz[i]),
            "sxy_pa": float(sxy[i]), "sxz_pa": float(sxz[i]), "syz_pa": float(syz[i]),
            "cff_bar": float(cff_bar[i]), "shear_bar": float(shear_bar[i]),
            "normal_bar": float(normal_bar[i]),
            "pred_e_m": float(ue[i]), "pred_n_m": float(un[i]), "pred_u_m": float(uz[i]),
            "pred_horiz_mag_m": float(pred_horiz_mag[i]), "pred_azimuth_deg": float(pred_azimuth[i]),
            "obs_e_m": p.obs_e, "obs_n_m": p.obs_n, "obs_u_m": p.obs_u,
        }

        if p.obs_e is not None and p.obs_n is not None:
            row["obs_horiz_mag_m"] = float(np.hypot(p.obs_e, p.obs_n))
            row["obs_azimuth_deg"] = float(np.degrees(np.arctan2(p.obs_e, p.obs_n)) % 360.0)
        else:
            row["obs_horiz_mag_m"] = None
            row["obs_azimuth_deg"] = None

        row["resid_e_m"] = (p.obs_e - row["pred_e_m"]) if p.obs_e is not None else None
        row["resid_n_m"] = (p.obs_n - row["pred_n_m"]) if p.obs_n is not None else None
        row["resid_u_m"] = (p.obs_u - row["pred_u_m"]) if p.obs_u is not None else None

        if row["resid_e_m"] is not None and row["resid_n_m"] is not None:
            row["resid_horiz_mag_m"] = float(np.hypot(row["resid_e_m"], row["resid_n_m"]))
        else:
            row["resid_horiz_mag_m"] = None

        comps = [c for c in (row["resid_e_m"], row["resid_n_m"], row["resid_u_m"]) if c is not None]
        row["n_resid_components"] = len(comps)
        row["resid_3d_mag_m"] = float(np.sqrt(sum(c * c for c in comps))) if comps else None

        for comp, sigma in (("e", p.sigma_e), ("n", p.sigma_n), ("u", p.sigma_u)):
            resid = row[f"resid_{comp}_m"]
            row[f"chi2_{comp}"] = float((resid / sigma) ** 2) if (resid is not None and sigma) else None

        results.append(row)

    if progress_callback: progress_callback(100)
    return results


# ─── CSV export ──────────────────────────────────────────────────────────

def results_to_csv_text(results: List[dict]) -> str:
    """Serialize compute_point_results() output to CSV text (RESULT_COLUMNS
    order), suitable for writing straight to a file or presenting in a
    preview. None values are written as empty cells (not the string
    "None"), matching how a spreadsheet/GIS user expects a missing value
    to look."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(RESULT_COLUMNS)
    for row in results:
        writer.writerow(["" if row.get(c) is None else row.get(c) for c in RESULT_COLUMNS])
    return buf.getvalue()
