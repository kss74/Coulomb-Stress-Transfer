# -*- coding: utf-8 -*-
"""
Flexible-column-mapping import for focal-mechanism catalogs.

Catalogs in the wild use wildly different column names for the same
quantities (GCMT psmeca dumps, ISC bulletin exports, HASH/FPFIT output,
ad-hoc CSVs from other software) and different SCHEMAS entirely (two
nodal planes given directly, one plane + derive the aux plane, or a full
moment tensor). This module handles both problems:

  1. SCHEMA detection/selection -- which of the supported field groups
     (two-plane / one-plane / moment-tensor-NED / moment-tensor-GCMT)
     the source data provides.
  2. COLUMN matching -- fuzzy-matching this project's canonical field
     names against whatever the source file/layer actually calls its
     columns, with the caller (UI layer) able to override any guess
     before committing.

Reads from a delimited text file (csv/tsv) or, when running inside QGIS,
directly from a loaded vector layer's attribute table. Deliberately does
NOT implement any network/download path -- see PROJECT_HANDOVER
addendum for this session's discussion of why (existing dedicated QGIS
plugins like QBeachball/GISfocal already cover that; import from
whatever layer they populate is covered here for free).
"""

import csv
import difflib
import io
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from .focal_mechanism import (
    FocalMechanismEvent, moment_tensor_ned_to_sdr, gcmt_use_to_sdr,
)


# ─── Schemas ─────────────────────────────────────────────────────────────

# Always required, regardless of schema.
POSITION_FIELDS = ("lon", "lat", "depth")
POSITION_OPTIONAL_FIELDS = ("magnitude", "label")

SCHEMAS = {
    "two_plane": ("strike1", "dip1", "rake1", "strike2", "dip2", "rake2"),
    "one_plane": ("strike1", "dip1", "rake1"),
    "moment_tensor_ned": ("mnn", "mee", "mdd", "mne", "mnd", "med"),
    "moment_tensor_gcmt": ("mrr", "mtt", "mpp", "mrt", "mrp", "mtp"),
}

SCHEMA_LABELS = {
    "two_plane": "Both nodal planes (strike/dip/rake x2)",
    "one_plane": "One nodal plane only (auxiliary plane derived)",
    "moment_tensor_ned": "Moment tensor, North/East/Down Cartesian convention",
    "moment_tensor_gcmt": "Moment tensor, GCMT/NDK convention (Mrr,Mtt,Mpp,Mrt,Mrp,Mtp)",
}

# Fuzzy-matching aliases. Keys are canonical field names; values are
# lowercase, alphanumeric-only strings the real column name is compared
# against (see _normalize()). Extend this list as new real-world catalog
# formats are encountered -- it is not exhaustive by construction.
ALIASES: Dict[str, List[str]] = {
    "lon": ["lon", "long", "longitude", "elon", "evlon", "x"],
    "lat": ["lat", "latitude", "elat", "evlat", "y"],
    "depth": ["depth", "edepth", "evdepth", "z", "depthkm", "hypodepth", "centroiddepth"],
    "magnitude": ["mag", "magnitude", "mw", "ms", "mb", "m"],
    "label": ["id", "eventid", "label", "name", "eventname", "cmtname", "code"],
    "strike1": ["strike1", "str1", "strk1", "s1", "strikenp1", "np1strike", "strikea"],
    "dip1": ["dip1", "dp1", "d1", "dipnp1", "np1dip", "dipa"],
    "rake1": ["rake1", "rk1", "r1", "rakenp1", "np1rake", "rakea"],
    "strike2": ["strike2", "str2", "strk2", "s2", "strikenp2", "np2strike", "strikeb"],
    "dip2": ["dip2", "dp2", "d2", "dipnp2", "np2dip", "dipb"],
    "rake2": ["rake2", "rk2", "r2", "rakenp2", "np2rake", "rakeb"],
    "mnn": ["mnn", "mxx"],
    "mee": ["mee", "myy"],
    "mdd": ["mdd", "mzz"],
    "mne": ["mne", "mxy"],
    "mnd": ["mnd", "mxz"],
    "med": ["med", "myz"],
    "mrr": ["mrr"],
    "mtt": ["mtt"],
    "mpp": ["mpp", "mff"],
    "mrt": ["mrt"],
    "mrp": ["mrp", "mrf"],
    "mtp": ["mtp", "mtf"],
}


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def suggest_column_mapping(columns: List[str], fields: List[str]) -> Dict[str, Optional[str]]:
    """
    For each canonical field name in `fields`, guess which entry of
    `columns` (actual source column/attribute names) it corresponds to,
    using alias matching first (exact, normalized) then fuzzy string
    matching as a fallback. Returns {field: best_guess_column_or_None}.
    Never raises -- an unmatched field just maps to None, and the UI
    layer is expected to let the user fill it in manually.
    """
    norm_cols = {c: _normalize(c) for c in columns}
    mapping: Dict[str, Optional[str]] = {}
    used = set()
    for field in fields:
        best = None
        # 1) exact alias match
        for alias in ALIASES.get(field, [field]):
            for col, norm in norm_cols.items():
                if col in used:
                    continue
                if norm == alias:
                    best = col
                    break
            if best:
                break
        # 2) fuzzy match against aliases if no exact hit
        if best is None:
            candidates = [c for c in columns if c not in used]
            aliases = ALIASES.get(field, [field])
            scored = []
            for col in candidates:
                norm = norm_cols[col]
                score = max(difflib.SequenceMatcher(None, norm, a).ratio() for a in aliases)
                scored.append((score, col))
            scored.sort(reverse=True)
            if scored and scored[0][0] >= 0.6:
                best = scored[0][1]
        mapping[field] = best
        if best:
            used.add(best)
    return mapping


def detect_schema(columns: List[str]) -> Tuple[Optional[str], Dict[str, Optional[str]]]:
    """
    Try each schema in a fixed preference order (two_plane is most
    informative, so prefer it when the columns support it) and return
    the first one where every field gets a confident (alias-exact or
    high-fuzzy-score) match, plus its suggested mapping. Returns
    (None, {}) if nothing matches well -- caller should then let the
    user pick a schema manually and map columns by hand.
    """
    for schema_name in ("two_plane", "moment_tensor_gcmt", "moment_tensor_ned", "one_plane"):
        required_fields = list(POSITION_FIELDS) + list(SCHEMAS[schema_name])
        all_fields = required_fields + list(POSITION_OPTIONAL_FIELDS)
        mapping = suggest_column_mapping(columns, all_fields)
        if all(mapping[f] is not None for f in required_fields):
            return schema_name, mapping
    return None, {}


# ─── File reading ───────────────────────────────────────────────────────

def read_delimited_table(path_or_text: str, is_path: bool = True,
                          max_preview_rows: int = 20):
    """
    Read a CSV/TSV/whitespace-delimited text table. Returns
    (columns, rows) where rows is a list of dicts {column: str_value}
    (ALL rows, not just the preview -- max_preview_rows only affects
    what's cheap to show in a UI preview table; the caller slices it).
    Delimiter is sniffed; falls back to comma.
    """
    if is_path:
        with open(path_or_text, "r", newline="", encoding="utf-8-sig") as f:
            text = f.read()
    else:
        text = path_or_text

    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
    except csv.Error:
        dialect = csv.excel  # comma default
    reader = csv.reader(io.StringIO(text), dialect)
    all_rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not all_rows:
        return [], []
    columns = [c.strip() for c in all_rows[0]]
    rows = []
    for raw in all_rows[1:]:
        row = {columns[i]: (raw[i].strip() if i < len(raw) else "")
               for i in range(len(columns))}
        rows.append(row)
    return columns, rows


def read_qgis_layer_table(layer):
    """
    Read column/attribute names and all feature attributes from a QGIS
    vector layer. Only usable inside a real QGIS session (guarded
    import). Point-geometry layers contribute lon/lat directly from the
    geometry if a matching attribute column isn't found -- handled by
    the caller via the usual column-mapping path (geometry x/y can be
    injected as synthetic "lon"/"lat" columns by the UI layer before
    calling build_events_from_mapped_rows(), left out of this function
    to keep it a pure attribute-table reader).
    """
    fields = [f.name() for f in layer.fields()]
    rows = []
    for feat in layer.getFeatures():
        row = {name: ("" if feat[name] is None else str(feat[name])) for name in fields}
        geom = feat.geometry()
        if geom and not geom.isEmpty():
            pt = geom.asPoint() if geom.type() == 0 else geom.centroid().asPoint()
            row.setdefault("__geom_x__", str(pt.x()))
            row.setdefault("__geom_y__", str(pt.y()))
        rows.append(row)
    return fields, rows


# ─── Row -> FocalMechanismEvent ─────────────────────────────────────────

@dataclass
class ImportResult:
    events: List[FocalMechanismEvent]
    n_skipped: int
    errors: List[str]  # one message per skipped row (row number + reason)


def _get_float(row: Dict[str, str], column_map: Dict[str, Optional[str]], field: str):
    col = column_map.get(field)
    if not col:
        return None
    val = row.get(col, "")
    if val == "" or val is None:
        return None
    return float(val)


def build_events_from_mapped_rows(rows: List[Dict[str, str]],
                                   column_map: Dict[str, Optional[str]],
                                   schema: str) -> ImportResult:
    """
    Convert mapped table rows into FocalMechanismEvent objects.
    column_map: {canonical_field: actual_column_name_or_None}, as
    produced/edited from suggest_column_mapping().
    schema: one of SCHEMAS' keys.
    Rows that fail to parse (missing required value, non-numeric,
    NaN/inf) are SKIPPED and recorded in .errors rather than aborting
    the whole import -- catalogs commonly have a handful of incomplete
    rows and the rest should still come through.
    """
    if schema not in SCHEMAS:
        raise ValueError(f"unknown schema {schema!r}, must be one of {list(SCHEMAS)}")

    events = []
    errors = []
    for i, row in enumerate(rows):
        row_num = i + 2  # 1-indexed + header row
        try:
            lon = _get_float(row, column_map, "lon")
            lat = _get_float(row, column_map, "lat")
            depth = _get_float(row, column_map, "depth")
            if lon is None or lat is None or depth is None:
                raise ValueError("missing lon/lat/depth")
            mag = _get_float(row, column_map, "magnitude")
            label_col = column_map.get("label")
            label = row.get(label_col, "") if label_col else ""

            if schema == "two_plane":
                s1 = _get_float(row, column_map, "strike1")
                d1 = _get_float(row, column_map, "dip1")
                r1 = _get_float(row, column_map, "rake1")
                s2 = _get_float(row, column_map, "strike2")
                d2 = _get_float(row, column_map, "dip2")
                r2 = _get_float(row, column_map, "rake2")
                if None in (s1, d1, r1, s2, d2, r2):
                    raise ValueError("missing plane 1/2 values")
                ev = FocalMechanismEvent(lon=lon, lat=lat, depth=depth,
                                          strike1=s1, dip1=d1, rake1=r1,
                                          strike2=s2, dip2=d2, rake2=r2,
                                          magnitude=mag, label=label)

            elif schema == "one_plane":
                s1 = _get_float(row, column_map, "strike1")
                d1 = _get_float(row, column_map, "dip1")
                r1 = _get_float(row, column_map, "rake1")
                if None in (s1, d1, r1):
                    raise ValueError("missing plane 1 values")
                ev = FocalMechanismEvent(lon=lon, lat=lat, depth=depth,
                                          strike1=s1, dip1=d1, rake1=r1,
                                          magnitude=mag, label=label)
                ev.fill_aux_plane()

            elif schema == "moment_tensor_ned":
                vals = [_get_float(row, column_map, f) for f in SCHEMAS["moment_tensor_ned"]]
                if any(v is None for v in vals):
                    raise ValueError("missing moment tensor component(s)")
                ev = FocalMechanismEvent.from_moment_tensor_ned(
                    lon, lat, depth, *vals, magnitude=mag, label=label)
                ev.fill_aux_plane()

            elif schema == "moment_tensor_gcmt":
                vals = [_get_float(row, column_map, f) for f in SCHEMAS["moment_tensor_gcmt"]]
                if any(v is None for v in vals):
                    raise ValueError("missing moment tensor component(s)")
                ev = FocalMechanismEvent.from_gcmt(
                    lon, lat, depth, *vals, magnitude=mag, label=label)
                ev.fill_aux_plane()

            events.append(ev)
        except (ValueError, KeyError, TypeError) as e:
            errors.append(f"row {row_num}: {e}")

    return ImportResult(events=events, n_skipped=len(errors), errors=errors)
