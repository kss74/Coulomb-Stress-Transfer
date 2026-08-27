# -*- coding: utf-8 -*-
"""
Flexible-column-mapping import for earthquake catalogs (aftershock/
seismicity data), feeding core.aftershock_mc_test and, later, any
period-comparison or rate-and-state module that needs event times.

Same two-stage approach as core.focal_mechanism_import and
core.observation_import (read this project's PROJECT_HANDOVER for the
rationale): SCHEMA detection for the one genuinely-variable structural
choice, then per-field COLUMN fuzzy-matching, with the caller (UI layer)
able to override any guess before committing.

Unlike the focal-mechanism/observation importers, the one structural
axis that actually varies across agency catalog exports is *time
representation*, not the physical quantities themselves (every agency
gives lon/lat/depth/magnitude one way or another):

  "iso_time"    -- one combined date+time column. Covers USGS ComCat
                   ("2019-07-06T03:19:53.040Z"), most modern CSV/QuakeML
                   -derived exports, and anything ISO8601-ish or using a
                   common "YYYY-MM-DD HH:MM:SS[.ffffff]" layout.
  "split_time"  -- separate year/month/day columns (JMA-style catalogs,
                   older ISC/NEIC dumps, and the column layout Coulomb's
                   own EQ_DATA format uses internally -- see coulomb.m's
                   plugin_seis_rate_change). hour/minute/second are
                   separate OPTIONAL columns under this schema; missing
                   ones default to 0.
  "no_time"     -- catalog carries no usable temporal info at all (or
                   the user doesn't want to bother mapping it). Valid on
                   its own: the aftershock/CFF Monte Carlo null test
                   (core.aftershock_mc_test) only needs lon/lat/depth,
                   nothing time-dependent. Any future period-comparison
                   or rate-and-state consumer will simply see time=None
                   on these rows and should handle that (e.g. by
                   excluding them from anything that needs a date).

lon/lat/depth are always required regardless of schema (depth is what
lets rows be interpolated against the 3D CFF volume from
core.cff_volume -- a catalog without depth is unusable for that even
though it might still be fine for a purely-2D use). magnitude/label are
always-optional scalar fields, independent of the time schema.

Reads from a delimited text file (csv/tsv) or, when running inside
QGIS, directly from a loaded point vector layer -- geometry x/y is read
and transformed to lon/lat WGS84 if the layer's CRS differs, reusing
core.observation_import.read_qgis_layer_table() as-is rather than
duplicating that CRS-transform logic a third time.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import difflib

from .focal_mechanism_import import read_delimited_table
from .observation_import import read_qgis_layer_table  # CRS-transforming version; reused as-is

# ─── Schemas ─────────────────────────────────────────────────────────────

POSITION_FIELDS = ("lon", "lat", "depth")
OPTIONAL_SCALAR_FIELDS = ("magnitude", "label")

SCHEMAS = {
    "iso_time": ("time",),
    "split_time": ("year", "month", "day"),   # hour/minute/second: see SPLIT_TIME_OPTIONAL_FIELDS
    "no_time": (),
}
SPLIT_TIME_OPTIONAL_FIELDS = ("hour", "minute", "second")

SCHEMA_LABELS = {
    "iso_time": "Combined date+time column (e.g. USGS ComCat, ISO8601)",
    "split_time": "Separate year/month/day columns (e.g. JMA-style catalogs)",
    "no_time": "No usable time information (spatial-only import)",
}
SCHEMA_REQUIRED_FIELDS = {
    # Fields that MUST be mapped for the schema to be usable at all.
    "iso_time": ("time",),
    "split_time": ("year", "month", "day"),
    "no_time": (),
}

ALIASES: Dict[str, List[str]] = {
    "lon": ["lon", "long", "longitude", "evlon", "x", "geomx"],
    "lat": ["lat", "latitude", "evlat", "y", "geomy"],
    "depth": ["depth", "depthkm", "evdepth", "z", "hypodepth", "depthinkm"],
    "magnitude": ["mag", "magnitude", "mw", "ms", "mb", "m", "magmw"],
    "label": ["id", "eventid", "label", "name", "code", "eventname"],
    "time": ["time", "datetime", "origintime", "date", "eventtime", "otime"],
    "year": ["year", "yr", "yyyy"],
    "month": ["month", "mon", "mm"],
    "day": ["day", "dy", "dd"],
    "hour": ["hour", "hr", "hh"],
    "minute": ["minute", "min", "mi"],
    "second": ["second", "sec", "ss"],
}


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def suggest_column_mapping(columns: List[str], fields: List[str]) -> Dict[str, Optional[str]]:
    """Same alias-then-fuzzy matching strategy as core.focal_mechanism_import
    .suggest_column_mapping() / core.observation_import's copy of it --
    duplicated (not imported) because each module's ALIASES table is its
    own; see those modules for the identical algorithm's rationale."""
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
    """Prefer "iso_time" over "split_time" when both would technically be
    satisfiable (a combined column is less error-prone than reassembling
    y/m/d), and only fall back to "no_time" if neither can be confidently
    mapped -- "no_time" is never auto-selected over a real match, only
    offered as a manual override in the UI."""
    for schema_name in ("iso_time", "split_time"):
        required = list(POSITION_FIELDS) + list(SCHEMA_REQUIRED_FIELDS[schema_name])
        all_fields = list(POSITION_FIELDS) + list(OPTIONAL_SCALAR_FIELDS) + list(SCHEMAS[schema_name])
        if schema_name == "split_time":
            all_fields = all_fields + list(SPLIT_TIME_OPTIONAL_FIELDS)
        mapping = suggest_column_mapping(columns, all_fields)
        if all(mapping[f] is not None for f in required):
            return schema_name, mapping
    # Neither time schema is confidently mappable -- still try to get
    # lon/lat/depth mapped so the UI opens with a useful starting guess.
    mapping = suggest_column_mapping(columns, list(POSITION_FIELDS) + list(OPTIONAL_SCALAR_FIELDS))
    if all(mapping.get(f) is not None for f in POSITION_FIELDS):
        return "no_time", mapping
    return None, {}


# ─── Time parsing ───────────────────────────────────────────────────────

# Tried in order against the whole string; covers the combined-column
# formats actually seen in the wild (USGS ComCat, ISC, GeoNet, JMA CSV
# exports, ad-hoc "space instead of T" variants). Kept as explicit
# strptime formats rather than pulling in dateutil, since this needs to
# keep working inside QGIS's bundled Python without extra pip installs.
_ISO_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
)


def parse_datetime(text: str) -> Optional[datetime]:
    """Best-effort parse of a combined date+time string into a naive UTC
    datetime. Returns None (never raises) if nothing matches -- the
    caller records that as a per-row error rather than aborting import.
    A trailing 'Z' is stripped before the non-Z formats are tried so a
    single format list covers both "...Z" and offset-less strings."""
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    s_stripped = re.sub(r"Z$", "", s)
    for fmt in _ISO_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
        try:
            return datetime.strptime(s_stripped, fmt)
        except ValueError:
            pass
    # Last resort: Python's own ISO8601 parser (handles e.g. numeric UTC
    # offsets like "+09:00" that the fixed-format list above doesn't).
    try:
        s2 = s_stripped.replace(" ", "T", 1) if "T" not in s_stripped else s_stripped
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        return None


def _epoch_seconds(dt: Optional[datetime]) -> Optional[float]:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc).timestamp()


# ─── Row -> EQCatalogEvent ──────────────────────────────────────────────

@dataclass
class EQCatalogEvent:
    lon: float
    lat: float
    depth: float           # km, positive down (this project's convention)
    time: Optional[datetime] = None
    epoch_s: Optional[float] = None   # POSIX seconds UTC; convenience for sorting/rate calcs
    magnitude: Optional[float] = None
    label: str = ""


@dataclass
class EQCatalogImportResult:
    events: List[EQCatalogEvent] = field(default_factory=list)
    n_skipped: int = 0
    errors: List[str] = field(default_factory=list)
    n_missing_time: int = 0   # rows imported OK but with time=None (schema=no_time, or unparseable time)


def _get_float(row: Dict[str, str], column_map: Dict[str, Optional[str]], f: str):
    col = column_map.get(f)
    if not col:
        return None
    val = row.get(col, "")
    if val == "" or val is None:
        return None
    return float(val)


def _get_str(row: Dict[str, str], column_map: Dict[str, Optional[str]], f: str) -> str:
    col = column_map.get(f)
    if not col:
        return ""
    return row.get(col, "") or ""


def build_events_from_mapped_rows(rows: List[Dict[str, str]],
                                   column_map: Dict[str, Optional[str]],
                                   schema: str) -> EQCatalogImportResult:
    """
    Convert mapped table rows into EQCatalogEvent objects. lon/lat/depth
    missing or non-numeric -> row skipped (recorded in .errors). Time
    missing or unparseable -> row still imported with time=None (counted
    in .n_missing_time), UNLESS schema == "no_time" was explicitly
    chosen, in which case that's simply expected and not worth counting
    separately.
    """
    if schema not in SCHEMAS:
        raise ValueError(f"unknown schema {schema!r}, must be one of {list(SCHEMAS)}")

    out_events = []
    errors = []
    n_missing_time = 0
    for i, row in enumerate(rows):
        row_num = i + 2  # 1-indexed + header row
        try:
            lon = _get_float(row, column_map, "lon")
            lat = _get_float(row, column_map, "lat")
            depth = _get_float(row, column_map, "depth")
            if lon is None or lat is None or depth is None:
                raise ValueError("missing lon/lat/depth")

            mag = _get_float(row, column_map, "magnitude")
            label = _get_str(row, column_map, "label")

            dt = None
            if schema == "iso_time":
                raw = _get_str(row, column_map, "time")
                dt = parse_datetime(raw) if raw else None
                if raw and dt is None:
                    n_missing_time += 1  # column was mapped but unparseable -- worth flagging
            elif schema == "split_time":
                y = _get_float(row, column_map, "year")
                mo = _get_float(row, column_map, "month")
                d = _get_float(row, column_map, "day")
                if y is None or mo is None or d is None:
                    n_missing_time += 1
                else:
                    hh = _get_float(row, column_map, "hour") or 0
                    mi = _get_float(row, column_map, "minute") or 0
                    ss = _get_float(row, column_map, "second") or 0
                    try:
                        dt = datetime(int(y), int(mo), int(d),
                                      int(hh), int(mi), int(ss))
                    except ValueError as e:
                        raise ValueError(f"invalid split date/time: {e}")
            # schema == "no_time": dt stays None, not counted as "missing"

            out_events.append(EQCatalogEvent(
                lon=lon, lat=lat, depth=depth, time=dt,
                epoch_s=_epoch_seconds(dt), magnitude=mag, label=label))

        except (ValueError, KeyError, TypeError) as e:
            errors.append(f"row {row_num}: {e}")

    return EQCatalogImportResult(events=out_events, n_skipped=len(errors),
                                  errors=errors, n_missing_time=n_missing_time)


def events_to_eq_array(events: List[EQCatalogEvent]):
    """
    Convenience packer: sorts events by time (undated events sorted
    last, stable order preserved among them) and returns a plain list of
    dicts ready for core.aftershock_mc_test / core.cff_volume consumers,
    which only care about lon/lat/depth (+ optionally epoch_s/magnitude)
    and shouldn't need to import the datetime-bearing dataclass itself.
    """
    ordered = sorted(events, key=lambda e: (e.epoch_s is None, e.epoch_s if e.epoch_s is not None else 0.0))
    return [{"lon": e.lon, "lat": e.lat, "depth": e.depth,
             "epoch_s": e.epoch_s, "magnitude": e.magnitude, "label": e.label}
            for e in ordered]
