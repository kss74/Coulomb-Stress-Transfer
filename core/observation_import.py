# -*- coding: utf-8 -*-
"""
Flexible-column-mapping import for slip-inversion surface observations.

Two SCHEMAS are supported, matching the two observation types
core.okada_engine.run_slip_inversion() / dc3d_worker.py's "slip_inversion"
mode understand natively (see that module's docstring):

  "gnss"     -- component-wise (GNSS survey, leveling, tide-gauge, or any
                other point measurement of ground displacement). lon/lat
                required; e/n/u are each OPTIONAL per row (leveling data
                is u-only; a triangulation network might be horizontal-
                only), but at least one of the three must be present.
                Optional per-component 1-sigma uncertainty columns.

  "insar_los" -- one LOS-projected scalar per point plus its own unit
                look vector (ground-to-satellite, geographic E/N/U).
                lon/lat/los/look_e/look_n/look_u are all required (a LOS
                row without its look vector cannot be used at all,
                unlike gnss rows). Optional 1-sigma column.

Reads from a delimited text file (csv/tsv) or, when running inside QGIS,
directly from a loaded vector layer's attribute table -- geometry x/y is
read and, if the layer's CRS isn't already geographic WGS84, transformed
to lon/lat via QgsCoordinateTransform (slip-inversion physics is
correctness-sensitive to this in a way catalog display generally isn't,
so this reader does the transform itself rather than assuming the
layer's raw x/y are already lon/lat).

Deliberately does NOT attempt raw InSAR raster (GeoTIFF) ingestion or
quadtree/uniform pixel downsampling -- that belongs upstream, in
whatever InSAR processing chain (LiCSBAS, MintPy, GMTSAR, ...) produced
the point/vector product in the first place; this module just needs a
table of already-selected points with per-point LOS value + look vector,
however that table was produced.
"""

import csv
import difflib
import io
import datetime as _dt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from .focal_mechanism_import import read_delimited_table  # generic CSV/TSV reader, reused as-is

# ─── Schemas ─────────────────────────────────────────────────────────────

POSITION_FIELDS = ("lon", "lat")

SCHEMAS = {
    "gnss": ("e", "n", "u", "sigma_e", "sigma_n", "sigma_u"),        # all optional per row
    "insar_los": ("los", "look_e", "look_n", "look_u", "sigma"),      # los/look_* required, sigma optional
}
SCHEMA_LABELS = {
    "gnss": "GNSS / field measurements (E, N, U components)",
    "insar_los": "InSAR (LOS displacement + look vector)",
}
SCHEMA_REQUIRED_FIELDS = {
    # Fields that MUST be mapped for the schema to be usable at all
    # (distinct from "at least one of e/n/u", which is a per-ROW check
    # done in build_observations_from_mapped_rows(), not a per-COLUMN one).
    "gnss": (),
    "insar_los": ("los", "look_e", "look_n", "look_u"),
}

ALIASES: Dict[str, List[str]] = {
    "lon": ["lon", "long", "longitude", "x", "easting"],
    "lat": ["lat", "latitude", "y", "northing"],
    "e": ["e", "east", "ue", "de", "dx", "easting_disp", "disp_e", "e_disp", "veast"],
    "n": ["n", "north", "un", "dn", "dy", "northing_disp", "disp_n", "n_disp", "vnorth"],
    "u": ["u", "up", "uz", "du", "dz", "vertical", "vertical_disp", "disp_u", "u_disp", "vup"],
    "sigma_e": ["sigma_e", "sige", "se", "stde", "std_e", "e_err", "err_e", "e_sigma"],
    "sigma_n": ["sigma_n", "sign", "sn", "stdn", "std_n", "n_err", "err_n", "n_sigma"],
    "sigma_u": ["sigma_u", "sigu", "su", "stdu", "std_u", "u_err", "err_u", "u_sigma"],
    "los": ["los", "losdisp", "losdisplacement", "rangechange", "dlos", "los_m", "value"],
    "look_e": ["looke", "lookeast", "unite", "le", "elook", "eastlook", "incidence_e"],
    "look_n": ["lookn", "looknorth", "unitn", "ln", "nlook", "northlook", "incidence_n"],
    "look_u": ["looku", "lookup", "unitu", "lu", "ulook", "uplook", "incidence_u"],
    "sigma": ["sigma", "sig", "std", "stdlos", "error", "err", "los_err", "los_sigma"],
}


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def suggest_column_mapping(columns: List[str], fields: List[str]) -> Dict[str, Optional[str]]:
    """Same alias-then-fuzzy matching strategy as core.focal_mechanism_import
    .suggest_column_mapping() -- see that docstring. Never raises."""
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


def detect_schema(columns: List[str]) -> Tuple[Optional[str], Dict[str, Optional[str]]]:
    """Prefer "insar_los" only when its full required set (los + look
    vector) is confidently mapped; otherwise fall back to "gnss" as long
    as lon/lat are mapped (e/n/u are all-optional at the column-detection
    level -- the per-row "at least one" check happens later)."""
    for schema_name in ("insar_los", "gnss"):
        all_fields = list(POSITION_FIELDS) + list(SCHEMAS[schema_name])
        mapping = suggest_column_mapping(columns, all_fields)
        required = list(POSITION_FIELDS) + list(SCHEMA_REQUIRED_FIELDS[schema_name])
        if all(mapping[f] is not None for f in required):
            return schema_name, mapping
    return None, {}


def _stringify_attr(val):
    """
    Convert one QGIS feature attribute value to a string for the
    column-mapping pipeline, special-casing QDateTime/QDate/QTime (what
    a GeoPackage/OGR *DateTime*-typed field actually returns, as opposed
    to a plain text column that merely looks like a date) to ISO 8601
    via Qt.ISODate rather than falling through to `str()`'s default
    repr.

    Root cause this fixes (2026-08-23): `str(QDateTime(...))` does not
    produce a parseable date string (PyQt's QDateTime has no useful
    __str__, so Python falls back to repr-like output) -- every
    downstream strptime/fromisoformat-based parser
    (core.eq_catalog_import.parse_datetime, and anything else built on
    this same read_qgis_layer_table()) then silently fails to parse
    every row, returning time=None across the whole catalog. This is
    NOT the same as an unparseable *text* date -- a genuinely malformed
    date column still correctly fails and is counted in
    n_missing_time -- it only ever tripped when the source field's
    QGIS/OGR type was DateTime/Date/Time rather than String, which is
    exactly what "read the layer's actual DateTime column type" means
    for a GeoPackage: the field carries a real typed value, not text,
    and needs converting rather than stringifying blind. Symptom
    reported: a mainshock with thousands of real aftershocks in the
    catalog, correctly geolocated, still showing "0 observed events" in
    calibration/validation because every epoch_s came back None.
    """
    if val is None:
        return ""
    try:
        from qgis.PyQt.QtCore import QDateTime, QDate, QTime, Qt
        if isinstance(val, QDateTime):
            return val.toString(Qt.ISODate)
        if isinstance(val, QDate):
            return val.toString(Qt.ISODate)
        if isinstance(val, QTime):
            return val.toString(Qt.ISODate)
    except ImportError:
        pass  # not running inside QGIS's PyQt -- fall through below
    if isinstance(val, (_dt.datetime, _dt.date)):
        # Some OGR/GDAL provider paths hand back native Python
        # datetime/date objects instead of Qt ones for DateTime/Date
        # fields -- same fix, same reasoning, different object type.
        return val.isoformat()
    return str(val)


def read_qgis_layer_table(layer, target_epsg=4326):
    """
    Read attribute names/values plus geometry x/y (as "__geom_x__"/
    "__geom_y__", transformed to EPSG:target_epsg if the layer's own CRS
    differs) from a QGIS point vector layer. Only usable inside a real
    QGIS session (guarded import).

    Attribute values are stringified via _stringify_attr() (not a bare
    str()) so that native DateTime/Date/Time-typed fields (as opposed to
    plain text columns) come out as parseable ISO 8601 strings -- see
    that function's docstring for the bug this fixes.
    """
    from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject

    fields = [f.name() for f in layer.fields()]
    target_crs = QgsCoordinateReferenceSystem(f"EPSG:{target_epsg}")
    src_crs = layer.crs()
    transform = None
    if src_crs.isValid() and src_crs != target_crs:
        transform = QgsCoordinateTransform(src_crs, target_crs, QgsProject.instance())

    rows = []
    for feat in layer.getFeatures():
        row = {name: _stringify_attr(feat[name]) for name in fields}
        geom = feat.geometry()
        if geom and not geom.isEmpty():
            pt = geom.asPoint() if geom.type() == 0 else geom.centroid().asPoint()
            if transform is not None:
                pt = transform.transform(pt)
            row.setdefault("__geom_x__", str(pt.x()))
            row.setdefault("__geom_y__", str(pt.y()))
        rows.append(row)
    return fields, rows


# ─── Row -> observation dict ────────────────────────────────────────────

@dataclass
class ObservationImportResult:
    rows: List[dict] = field(default_factory=list)   # observation/los_observation dicts
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


def build_observations_from_mapped_rows(rows: List[Dict[str, str]],
                                        column_map: Dict[str, Optional[str]],
                                        schema: str) -> ObservationImportResult:
    """
    Convert mapped table rows into observation dicts in exactly the shape
    core.okada_engine.run_slip_inversion() expects:
      "gnss"      -> {"lon","lat","e","n","u","sigma_e","sigma_n","sigma_u"}
                     (unmapped/blank e/n/u/sigma_* become None; a row
                     with ALL of e/n/u missing is skipped -- it carries
                     no information)
      "insar_los" -> {"lon","lat","los","look_e","look_n","look_u","sigma"}
                     (sigma missing -> None; any of los/look_* missing
                     is skipped -- an incomplete LOS row is unusable)
    Rows that fail to parse are skipped and recorded in .errors rather
    than aborting the whole import.
    """
    if schema not in SCHEMAS:
        raise ValueError(f"unknown schema {schema!r}, must be one of {list(SCHEMAS)}")

    out_rows = []
    errors = []
    for i, row in enumerate(rows):
        row_num = i + 2  # 1-indexed + header row
        try:
            lon = _get_float(row, column_map, "lon")
            lat = _get_float(row, column_map, "lat")
            if lon is None or lat is None:
                raise ValueError("missing lon/lat")

            if schema == "gnss":
                e = _get_float(row, column_map, "e")
                n = _get_float(row, column_map, "n")
                u = _get_float(row, column_map, "u")
                if e is None and n is None and u is None:
                    raise ValueError("none of e/n/u present -- row carries no information")
                sigma_e = _get_float(row, column_map, "sigma_e")
                sigma_n = _get_float(row, column_map, "sigma_n")
                sigma_u = _get_float(row, column_map, "sigma_u")
                out_rows.append({"lon": lon, "lat": lat, "e": e, "n": n, "u": u,
                                 "sigma_e": sigma_e, "sigma_n": sigma_n, "sigma_u": sigma_u})

            elif schema == "insar_los":
                los = _get_float(row, column_map, "los")
                look_e = _get_float(row, column_map, "look_e")
                look_n = _get_float(row, column_map, "look_n")
                look_u = _get_float(row, column_map, "look_u")
                if None in (los, look_e, look_n, look_u):
                    raise ValueError("missing los and/or look_e/look_n/look_u")
                sigma = _get_float(row, column_map, "sigma")
                out_rows.append({"lon": lon, "lat": lat, "los": los,
                                 "look_e": look_e, "look_n": look_n, "look_u": look_u,
                                 "sigma": sigma})

        except (ValueError, KeyError, TypeError) as e:
            errors.append(f"row {row_num}: {e}")

    return ObservationImportResult(rows=out_rows, n_skipped=len(errors), errors=errors)
