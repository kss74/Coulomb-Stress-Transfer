# -*- coding: utf-8 -*-
"""
Flexible-column-mapping import for fault-patch tables (distributed-slip
models from external sources -- e.g. GSI/geodetic-inversion outputs,
SRCMOD-style finite-fault models, or a table exported from this
plugin's own ui.fault_table_widget elsewhere and re-imported).

Mirrors core.observation_import's / core.focal_mechanism_import's
architecture exactly (ALIASES + suggest_column_mapping + detect_schema
+ read_delimited_table), applied to fault-patch geometry+slip instead
of point observations or focal mechanisms. See those modules'
docstrings for the shared design rationale.

Two SCHEMAS, because slip shows up in the wild in two genuinely
different shapes:

  "rake_slip"      -- ONE scalar slip magnitude + one rake angle
                       (Aki-Richards convention: 0/180=strike-slip,
                       +90=reverse, -90=normal). This is what most
                       published finite-fault/geodetic-inversion
                       models actually report (e.g. the GSI dataset
                       this module was built against: Kobayashi et
                       al. 2018, Tectonophysics).

  "rt_lat_reverse" -- already decomposed into right-lateral + reverse
                       components (Coulomb's own convention -- this is
                       exactly how ui.fault_table_widget.FaultTableWidget
                       stores every row internally, so a CSV exported
                       from that widget and re-imported here round-
                       trips through this schema with no lossy
                       reconstruction).

Depth convention (top-edge vs centroid) and length/width/depth/slip
units are NOT auto-detected -- they're rarely stated unambiguously in
a bare column header ("Depth" alone doesn't say which), so the caller
(the import dialog) must ask the user rather than this module
guessing. The one place this module DOES have an opinion is the
actual top->centroid depth conversion arithmetic and the rake/slip ->
rt-lateral/reverse decomposition: both are done by handing off to
core.okada_engine.FaultParameters.from_rt_lat_reverse() -- the exact same
function ui.fault_table_widget already trusts for this -- rather than
re-deriving the trig here, so there is exactly one implementation of
that math in the whole plugin.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from .focal_mechanism_import import read_delimited_table  # generic CSV/TSV reader, reused as-is
from .okada_engine import FaultParameters

# ─── Schemas ─────────────────────────────────────────────────────────────

POSITION_FIELDS = ("lon", "lat", "depth", "length", "width", "strike", "dip")

SCHEMAS = {
    "rake_slip": ("rake", "slip"),
    "rt_lat_reverse": ("rt_lateral_slip", "reverse_slip"),
}
SCHEMA_LABELS = {
    "rake_slip": "Rake + slip magnitude (Aki-Richards)",
    "rt_lat_reverse": "Right-lateral + reverse slip (already decomposed, Coulomb convention)",
}
# every field in every schema is required -- a fault patch missing any
# one of geometry/slip is not a usable patch, unlike observation_import's
# gnss schema where individual e/n/u components can be optional
SCHEMA_REQUIRED_FIELDS = SCHEMAS

ALIASES: Dict[str, List[str]] = {
    "lon": ["lon", "long", "longitude", "x", "easting"],
    "lat": ["lat", "latitude", "y", "northing"],
    "depth": ["depth", "dep", "z", "centroiddepth", "topdepth", "depthkm", "hypocentraldepth"],
    "length": ["length", "leng", "len", "l", "lengthkm"],
    "width": ["width", "wid", "w", "widthkm"],
    "strike": ["strike", "str", "strk", "strikedeg"],
    "dip": ["dip", "dipdeg"],
    "rake": ["rake", "rak", "rakedeg"],
    "slip": ["slip", "sliptotal", "totalslip", "scalarslip", "slipamount", "slipm", "slipcm", "mag"],
    "rt_lateral_slip": ["rtlateralslip", "rtlat", "rightlateral", "rightlateralslip",
                        "strikeslip", "ss", "u1", "rtlateralslipm"],
    "reverse_slip": ["reverseslip", "reverse", "dipslip", "ds", "thrust", "u2", "reverseslipm"],
}


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def suggest_column_mapping(columns: List[str], fields: List[str]) -> Dict[str, Optional[str]]:
    """Same alias-then-fuzzy matching strategy as
    core.observation_import.suggest_column_mapping() -- see that
    docstring. Never raises."""
    import difflib
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
    """
    Prefer "rt_lat_reverse" only when BOTH its fields are confidently
    mapped (a table that genuinely ships decomposed slip almost always
    has clearly-named columns for both); otherwise fall back to
    "rake_slip". This order matters: a table with only a generic
    "Slip" + "Rake" pair (like the GSI dataset) must not accidentally
    satisfy "rt_lat_reverse" via a loose fuzzy match -- required-field
    confidence, not just field count, decides it.
    """
    for schema_name in ("rt_lat_reverse", "rake_slip"):
        all_fields = list(POSITION_FIELDS) + list(SCHEMAS[schema_name])
        mapping = suggest_column_mapping(columns, all_fields)
        required = list(POSITION_FIELDS) + list(SCHEMA_REQUIRED_FIELDS[schema_name])
        if all(mapping[f] is not None for f in required):
            return schema_name, mapping
    return None, {}


def read_fault_table(path: str):
    """
    Read a fault-patch table from disk, supporting TWO layouts:

    (1) GSI-style: first non-blank line starts with '#' and IS the
        header (column names after the '#'), fields separated by
        runs of whitespace, e.g.:
            # Lon[deg] Lat[deg] Dep[km] Leng[km] ...
            137.851 36.6425 0.251 1.0 ...
        csv.Sniffer-based parsing (what read_delimited_table() does)
        reliably misdetects this layout -- variable-width whitespace
        isn't a single-character delimiter and the '#' would otherwise
        be read as literal column-name text -- so this case is handled
        directly here.

    (2) Anything read_delimited_table() already handles: ordinary
        CSV/TSV with a plain (non-'#') header row -- e.g. a table
        exported from ui.fault_table_widget itself.

    Returns (columns, rows) -- same shape as read_delimited_table().
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read()

    lines = text.splitlines()
    first_nonblank = next((ln for ln in lines if ln.strip()), "")
    if first_nonblank.strip().startswith("#"):
        header_line = first_nonblank.strip().lstrip("#").strip()
        columns = header_line.split()
        rows = []
        for ln in lines:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            tokens = s.split()
            rows.append({columns[i]: (tokens[i] if i < len(tokens) else "")
                        for i in range(len(columns))})
        return columns, rows

    return read_delimited_table(path, is_path=True)


# ─── Row -> fault-patch dict ─────────────────────────────────────────────

LENGTH_UNIT_TO_KM = {"km": 1.0, "m": 0.001}
DEPTH_UNIT_TO_KM = {"km": 1.0, "m": 0.001}
SLIP_UNIT_TO_M = {"m": 1.0, "cm": 0.01, "mm": 0.001}


@dataclass
class FaultTableImportResult:
    rows: List[dict] = field(default_factory=list)
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


def build_fault_rows_from_mapped_rows(rows: List[Dict[str, str]],
                                      column_map: Dict[str, Optional[str]],
                                      schema: str,
                                      length_unit: str = "km",
                                      width_unit: Optional[str] = None,
                                      depth_unit: str = "km",
                                      slip_unit: str = "m",
                                      depth_convention: str = "centroid",
                                      name_prefix: str = "Patch",
                                      group: Optional[str] = None) -> FaultTableImportResult:
    """
    Convert mapped table rows into fault-patch dicts ready for
    ui.fault_table_widget.FaultTableWidget.add_row():
      {"name", "lon", "lat", "depth_km", "length_km", "width_km",
       "strike", "dip", "rt_lateral_slip_m", "reverse_slip_m",
       "rake_deg", "lonlat_mode", "group"}

    length_unit/width_unit : "km" or "m" (width_unit defaults to
        length_unit if not given -- most sources use the same unit for
        both, but not always, e.g. a table with length in km and a
        fixed uniform width column stated separately in m).
    depth_unit : "km" or "m".
    slip_unit : "m", "cm", or "mm" -- applies to `slip` (rake_slip
        schema) or to BOTH `rt_lateral_slip`/`reverse_slip`
        (rt_lat_reverse schema).
    depth_convention : "centroid" (depth column IS the patch's
        volumetric centroid depth -- verify against a source's own
        documentation or by checking whether successive along-dip
        rows step by width*sin(dip), as the GSI dataset does) or "top"
        (depth column is the patch's top-edge depth, Coulomb's own
        native convention).

    Depth conversion and rake/slip decomposition are NOT re-derived
    here -- every row is built via core.okada_engine.FaultParameters.
    from_rt_lat_reverse(), the exact same function
    ui.fault_table_widget already relies on, so there is one
    implementation of that math for the whole plugin (see module
    docstring).

    Rows that fail to parse are skipped and recorded in .errors rather
    than aborting the whole import.
    """
    if schema not in SCHEMAS:
        raise ValueError(f"unknown schema {schema!r}, must be one of {list(SCHEMAS)}")
    if length_unit not in LENGTH_UNIT_TO_KM:
        raise ValueError(f"length_unit must be one of {list(LENGTH_UNIT_TO_KM)}, got {length_unit!r}")
    if depth_unit not in DEPTH_UNIT_TO_KM:
        raise ValueError(f"depth_unit must be one of {list(DEPTH_UNIT_TO_KM)}, got {depth_unit!r}")
    if slip_unit not in SLIP_UNIT_TO_M:
        raise ValueError(f"slip_unit must be one of {list(SLIP_UNIT_TO_M)}, got {slip_unit!r}")
    if depth_convention not in ("centroid", "top"):
        raise ValueError(f"depth_convention must be 'centroid' or 'top', got {depth_convention!r}")
    width_unit = width_unit or length_unit
    if width_unit not in LENGTH_UNIT_TO_KM:
        raise ValueError(f"width_unit must be one of {list(LENGTH_UNIT_TO_KM)}, got {width_unit!r}")

    len_k = LENGTH_UNIT_TO_KM[length_unit]
    wid_k = LENGTH_UNIT_TO_KM[width_unit]
    dep_k = DEPTH_UNIT_TO_KM[depth_unit]
    slip_k = SLIP_UNIT_TO_M[slip_unit]
    depth_is_top = (depth_convention == "top")

    out_rows = []
    errors = []
    for i, row in enumerate(rows):
        row_num = i + 2  # 1-indexed + header row
        try:
            lon = _get_float(row, column_map, "lon")
            lat = _get_float(row, column_map, "lat")
            depth_raw = _get_float(row, column_map, "depth")
            length_raw = _get_float(row, column_map, "length")
            width_raw = _get_float(row, column_map, "width")
            strike = _get_float(row, column_map, "strike")
            dip = _get_float(row, column_map, "dip")
            required_geom = (lon, lat, depth_raw, length_raw, width_raw, strike, dip)
            if any(v is None for v in required_geom):
                errors.append(f"row {row_num}: missing required geometry field(s)")
                continue

            depth_km = depth_raw * dep_k
            length_km = length_raw * len_k
            width_km = width_raw * wid_k

            if schema == "rake_slip":
                rake = _get_float(row, column_map, "rake")
                slip_raw = _get_float(row, column_map, "slip")
                if rake is None or slip_raw is None:
                    errors.append(f"row {row_num}: missing rake or slip")
                    continue
                slip_m = slip_raw * slip_k
                # matches core.okada_engine.FaultParameters.U1/U2 exactly (slip*cos(rake),
                # slip*sin(rake)) and .rt_lateral_slip/.reverse_slip (-U1, U2)
                rt_lateral_slip_m = -(slip_m * math.cos(math.radians(rake)))
                reverse_slip_m = slip_m * math.sin(math.radians(rake))
            else:  # "rt_lat_reverse"
                rt_lat_raw = _get_float(row, column_map, "rt_lateral_slip")
                reverse_raw = _get_float(row, column_map, "reverse_slip")
                if rt_lat_raw is None or reverse_raw is None:
                    errors.append(f"row {row_num}: missing rt_lateral_slip or reverse_slip")
                    continue
                rt_lateral_slip_m = rt_lat_raw * slip_k
                reverse_slip_m = reverse_raw * slip_k

            fault = FaultParameters.from_rt_lat_reverse(
                lon=lon, lat=lat, depth=depth_km, length=length_km, width=width_km,
                strike=strike, dip=dip, rt_lateral_slip=rt_lateral_slip_m,
                reverse_slip=reverse_slip_m, depth_is_top=depth_is_top)

            out_rows.append({
                "name": f"{name_prefix} {i + 1}",
                "lon": fault.lon, "lat": fault.lat, "depth_km": fault.depth,
                "length_km": length_km, "width_km": width_km,
                "strike": strike, "dip": dip,
                "rt_lateral_slip_m": fault.rt_lateral_slip,
                "reverse_slip_m": fault.reverse_slip,
                "rake_deg": fault.rake,
                "lonlat_mode": "centroid",
                "group": group,
            })
        except (ValueError, TypeError) as e:
            errors.append(f"row {row_num}: {e}")

    return FaultTableImportResult(rows=out_rows, n_skipped=len(errors), errors=errors)
