# -*- coding: utf-8 -*-
"""
Save/load the setup entered in the main dialog.

Two independent mechanisms, both round-tripping the SAME underlying
data (fault table rows, receiver-fault orientation, grid, elastic
params, cross-section params):

  1. Native JSON "setup" file (save_setup / load_setup): exact,
     lossless round-trip of every field in this dialog. This is the
     one to use for "save my work, come back to it later" -- it is
     NOT meant to be opened by Coulomb 3.4.2 itself.

  2. Coulomb-3.4.2-compatible ASCII .inp file (export_inp / import_inp):
     mirrors Coulomb's own input-file format, so a setup created here
     can (in principle) be opened directly in Coulomb, and an existing
     Coulomb .inp can be loaded into this plugin.

The .inp format below was reverse-engineered directly from Coulomb's
own MATLAB source (coulomb.m), NOT from the user guide, per this
project's "source over documentation" principle:
  - Writer: the uiputfile/fprintf block that creates '*.inp' files.
  - Reader: local_read_faults_from_ascii_inp_inr() (the ASCII branch
    of the "add existing fault model as receiver" import helper).
Both were read from coulomb.m in this project's own files before this
module was written.

Known scope limits of the .inp bridge (documented rather than
silently guessed at):
  - Every row is written/read as KODE=100 (a "source" fault entry,
    right-lateral + reverse slip in meters -- the format Coulomb calls
    "rt.lat/reverse", not the alternate "rake/netslip" IRAKE=1 format).
    Zero-slip rows (this plugin's "individual receiver faults") are
    written the same way with slip=0.0, which Coulomb reads fine as an
    inert fault plane -- but Coulomb's OWN receiver-fault KODE values
    (0 = specified plane, etc.) are a separate, distinct convention
    that this bridge does not attempt to reproduce, since their exact
    column semantics were not present in the source excerpts read for
    this feature. If you need Coulomb's own receiver-plane types after
    export, set them up in Coulomb directly.
  - This plugin's own "Rake (receiver, °)" fault-table column (used to
    resolve CFF on individual zero-slip receiver rows -- see
    fault_table_widget.py) has no equivalent field in Coulomb's KODE=100
    ASCII format, so it is NOT written on export and always reads back
    as 0.0 on import. It DOES round-trip losslessly through the native
    JSON setup file (save_setup/load_setup), which is unaffected by
    this .inp limitation.
  - Regional stress (S1DR/S2DR/S3DR/... in the header) is not part of
    this plugin's UI, so it round-trips as zeros on export and is
    ignored (read but discarded) on import.
  - Young's modulus (Coulomb's E1/E2) is derived from this plugin's
    shear modulus mu and Poisson's ratio nu via the standard isotropic
    relation E = 2*mu*(1+nu) on export. On import, E1 is converted
    back the other way (mu = E1 / (2*(1+nu))) using whatever nu is
    already set in the Elastic Params tab (nu itself is not stored in
    the .inp fault-header block Coulomb writes, only PR1/PR2, which
    duplicate the SAME Poisson's ratio Coulomb itself uses -- PR1 is
    used for that).
  - The coordinate origin (zero_lon/zero_lat) needed to convert the
    file's local-km X/Y trace endpoints back to lon/lat is read from
    the file's own "Map info" section if present; if that section is
    absent, (0, 0) is used as the origin and a warning is returned to
    the caller. The SAME origin and km<->lon/lat conversion (geo_to_km/
    km_to_geo) applies to the "Grid Parameters" and "Cross section
    default" trailer blocks' Start-x/Start-y/Finish-x/Finish-y fields --
    per coulomb.m, these are local km too, not literal lon/lat (fixed
    2026-08-15c; previously both export_inp and import_inp treated them
    as literal degrees, which silently produced nonsensical grid/
    cross-section bounds when round-tripping through this plugin and
    outright wrong bounds when importing a genuine Coulomb-written
    .inp file).
  - The header's "#fixed=" fault count is Coulomb's own declared count,
    but is not always trustworthy (e.g. hand-edited/script-generated
    files where rows were added without updating it). coulomb.m treats
    a mismatch as a hard error and refuses to load; this importer
    instead scans forward from the fault-table header for as many
    contiguous, parseable fault lines as are actually present, uses
    that count, and returns a warning if it disagrees with "#fixed="
    (fixed 2026-08-15c; previously a low "#fixed=" value silently
    truncated the import to that many rows with no warning).
"""

import json
import re
import numpy as np

from ..core.okada_engine import geo_to_km, km_to_geo
from ..core.cross_section_config import config_to_dict, config_from_dict


# ─── Native JSON setup save/load ───────────────────────────────────────────

SETUP_FILE_VERSION = 1


def build_setup_dict(dialog):
    """Collect every field in the main dialog into a plain JSON-serializable dict."""
    return {
        "version": SETUP_FILE_VERSION,
        "fault_rows": dialog.fault_table.get_raw_rows(),
        "receiver": {
            "strike": dialog.r_strike.value(),
            "dip": dialog.r_dip.value(),
            "rake": dialog.r_rake.value(),
        },
        "grid": {
            "lon_min": dialog.g_lon_min.value(),
            "lon_max": dialog.g_lon_max.value(),
            "lat_min": dialog.g_lat_min.value(),
            "lat_max": dialog.g_lat_max.value(),
            "depth_km": dialog.g_depth.value(),
            "res_mode": dialog.g_res_mode.currentText(),
            "n_lon": int(dialog.g_n_lon.value()),
            "n_lat": int(dialog.g_n_lat.value()),
            "spacing": dialog.g_spacing.value(),
        },
        "elastic": {
            "mu": dialog.e_mu.value(),
            "nu": dialog.e_nu.value(),
            "friction": dialog.e_friction.value(),
        },
        "cross_section": {
            "lon1": dialog.xs_lon1.value(),
            "lat1": dialog.xs_lat1.value(),
            "lon2": dialog.xs_lon2.value(),
            "lat2": dialog.xs_lat2.value(),
            "dist_increment_km": dialog.xs_dist_inc.value(),
            "max_depth_km": dialog.xs_max_depth.value(),
            "depth_increment_km": dialog.xs_depth_inc.value(),
        },
        # Regional (tectonic) stress + friction override, Optimal Faults
        # tab -- previously missing from this dict entirely (2026-08-24).
        "regional_stress": {
            "s1": dialog.rs_s1.value(),
            "s2": dialog.rs_s2.value(),
            "s3": dialog.rs_s3.value(),
            "s1_strike": dialog.rs_s1_strike.value(),
            "s1_plunge": dialog.rs_s1_plunge.value(),
            "s2_strike": dialog.rs_s2_strike.value(),
            "s2_plunge": dialog.rs_s2_plunge.value(),
            "friction": dialog.rs_friction.value(),
        },
        # Full cross-section display/symbology config (colors, sizes,
        # cmaps, z-order, topo panels, annotation sources, extra lines,
        # legend -- everything CrossSectionConfigDialog edits), as
        # distinct from the "cross_section" key above which only covers
        # the quick coordinate/increment fields on the Cross-Section tab
        # itself. dialog.xs_config is the SAME object main_dialog.py
        # reuses across repeated "Configure display…" dialog openings
        # (see main_dialog.__init__); previously never written to disk
        # (2026-08-24).
        "cross_section_config": (config_to_dict(dialog.xs_config)
                                  if hasattr(dialog, "xs_config") else None),
        # Last-used configuration widgets for the Rate-and-State Forecast
        # and Aftershock/ΔCFF Monte Carlo Null Test dialogs -- see
        # main_dialog.py's own self._rate_state_dialog_settings /
        # self._aftershock_dialog_settings docstring for why these are
        # plain dicts rather than a live dialog reference (both dialogs
        # are reconstructed fresh on every open, and are None until the
        # first time each has been opened at least once in this session).
        "rate_state_dialog": getattr(dialog, "_rate_state_dialog_settings", None),
        "aftershock_mc_dialog": getattr(dialog, "_aftershock_dialog_settings", None),
    }


def save_setup(dialog, path):
    """Write the current dialog setup to a JSON file at `path`."""
    data = build_setup_dict(dialog)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def apply_setup_dict(dialog, data):
    """Populate the dialog's widgets from a setup dict (see build_setup_dict)."""
    rows = data.get("fault_rows", [])
    if rows:
        dialog.fault_table.set_raw_rows(rows)

    r = data.get("receiver", {})
    if "strike" in r: dialog.r_strike.setValue(r["strike"])
    if "dip" in r: dialog.r_dip.setValue(r["dip"])
    if "rake" in r: dialog.r_rake.setValue(r["rake"])

    g = data.get("grid", {})
    if "lon_min" in g: dialog.g_lon_min.setValue(g["lon_min"])
    if "lon_max" in g: dialog.g_lon_max.setValue(g["lon_max"])
    if "lat_min" in g: dialog.g_lat_min.setValue(g["lat_min"])
    if "lat_max" in g: dialog.g_lat_max.setValue(g["lat_max"])
    if "depth_km" in g: dialog.g_depth.setValue(g["depth_km"])
    if "n_lon" in g: dialog.g_n_lon.setValue(g["n_lon"])
    if "n_lat" in g: dialog.g_n_lat.setValue(g["n_lat"])
    if "spacing" in g: dialog.g_spacing.setValue(g["spacing"])
    if "res_mode" in g:
        idx = dialog.g_res_mode.findText(g["res_mode"])
        if idx >= 0:
            dialog.g_res_mode.setCurrentIndex(idx)

    e = data.get("elastic", {})
    if "mu" in e: dialog.e_mu.setValue(e["mu"])
    if "nu" in e: dialog.e_nu.setValue(e["nu"])
    if "friction" in e: dialog.e_friction.setValue(e["friction"])

    xs = data.get("cross_section", {})
    if "lon1" in xs: dialog.xs_lon1.setValue(xs["lon1"])
    if "lat1" in xs: dialog.xs_lat1.setValue(xs["lat1"])
    if "lon2" in xs: dialog.xs_lon2.setValue(xs["lon2"])
    if "lat2" in xs: dialog.xs_lat2.setValue(xs["lat2"])
    if "dist_increment_km" in xs: dialog.xs_dist_inc.setValue(xs["dist_increment_km"])
    if "max_depth_km" in xs: dialog.xs_max_depth.setValue(xs["max_depth_km"])
    if "depth_increment_km" in xs: dialog.xs_depth_inc.setValue(xs["depth_increment_km"])

    rs = data.get("regional_stress", {})
    if "s1" in rs: dialog.rs_s1.setValue(rs["s1"])
    if "s2" in rs: dialog.rs_s2.setValue(rs["s2"])
    if "s3" in rs: dialog.rs_s3.setValue(rs["s3"])
    if "s1_strike" in rs: dialog.rs_s1_strike.setValue(rs["s1_strike"])
    if "s1_plunge" in rs: dialog.rs_s1_plunge.setValue(rs["s1_plunge"])
    if "s2_strike" in rs: dialog.rs_s2_strike.setValue(rs["s2_strike"])
    if "s2_plunge" in rs: dialog.rs_s2_plunge.setValue(rs["s2_plunge"])
    if "friction" in rs: dialog.rs_friction.setValue(rs["friction"])

    # Cross-section display/symbology config -- reconstruct and re-alias
    # dialog.xs_topo_panels/xs_annotations to the NEW config's own lists
    # (main_dialog.__init__ aliases these as `dialog.xs_config.topo_panels
    # is dialog.xs_topo_panels`; replacing dialog.xs_config wholesale
    # without re-pointing these two would leave the tab's own "Topo
    # panels"/"Annotations" list widgets reading stale, orphaned lists
    # that config_from_dict() didn't populate). Only touches these
    # attributes if the dialog actually has them (keeps this function
    # usable against a minimal stub in a headless verify script).
    if "cross_section_config" in data and data["cross_section_config"] is not None \
            and hasattr(dialog, "xs_config"):
        dialog.xs_config = config_from_dict(data["cross_section_config"])
        if hasattr(dialog, "xs_topo_panels"):
            dialog.xs_topo_panels = dialog.xs_config.topo_panels
        if hasattr(dialog, "xs_annotations"):
            dialog.xs_annotations = dialog.xs_config.annotations
        if hasattr(dialog, "_refresh_xs_topo_list"):
            dialog._refresh_xs_topo_list()
        if hasattr(dialog, "_refresh_xs_annotation_list"):
            dialog._refresh_xs_annotation_list()

    # Last-used settings for the Rate-and-State Forecast / Aftershock MC
    # Test dialogs -- stored back onto the SAME main_dialog attributes
    # open_rate_state_forecast_action()/open_aftershock_mc_test_action()
    # read from, so the next time either dialog is opened (in this
    # session, after this load) it comes up with the loaded setup's
    # values rather than this session's prior state or hardcoded
    # defaults. `in data` (not truthiness) so an explicit `null` in an
    # older/hand-edited setup file is distinguished from the key being
    # absent -- either way nothing currently open changes retroactively,
    # since neither dialog is open while a setup is being loaded.
    if "rate_state_dialog" in data and hasattr(dialog, "_rate_state_dialog_settings"):
        dialog._rate_state_dialog_settings = data["rate_state_dialog"]
    if "aftershock_mc_dialog" in data and hasattr(dialog, "_aftershock_dialog_settings"):
        dialog._aftershock_dialog_settings = data["aftershock_mc_dialog"]


def load_setup(dialog, path):
    """Read a JSON setup file at `path` and populate the dialog from it."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    apply_setup_dict(dialog, data)


# ─── Coulomb 3.4.2-compatible .inp export ──────────────────────────────────

def export_inp(dialog, path):
    """
    Write the current Source Faults table to a Coulomb-3.4.2-compatible
    ASCII .inp file at `path`, following the exact format Coulomb's own
    MATLAB source writes (see module docstring). Grid/size/cross-section/
    map-info trailer blocks are included too, using the dialog's Grid
    Output and Cross-Section tabs, and a coordinate origin equal to the
    grid's own center (so the local-km fault coordinates and the grid
    bounds are mutually consistent).
    """
    rows = dialog.fault_table.get_raw_rows()
    if not rows:
        raise ValueError("No fault rows to export.")

    elastic = dialog._get_elastic()
    grid = dialog._get_grid()
    nu = elastic.nu
    young = 2.0 * elastic.mu * (1.0 + nu)  # E = 2*mu*(1+nu)

    lon0 = 0.5 * (grid.lon_min + grid.lon_max)
    lat0 = 0.5 * (grid.lat_min + grid.lat_max)

    # Resolve each raw row into a FaultParameters to get its surface
    # trace endpoints, exactly as get_faults() would (but keep the raw
    # row count/order, i.e. do NOT expand subdivisions -- Coulomb's own
    # subdivision is a display-time GUI feature, not stored per-row).
    from ..core.okada_engine import FaultParameters

    lines = []
    lines.append("Coulomb 3 input file exported by Coulomb Stress Transfer QGIS plugin")
    lines.append(" ")
    lines.append(f"#reg1=  0  #reg2=  0  #fixed={len(rows):3d}  sym=  1")
    lines.append(f" PR1={nu:12.3f}     PR2={nu:12.3f}   DEPTH={grid.depth_km:12.3f}")
    lines.append(f"  E1={young:15.3e}   E2={young:15.3e}")
    lines.append("XSYM=       .000     YSYM=       .000")
    lines.append(f"FRIC={elastic.friction:15.3f}")
    lines.append(f"S1DR={0.0:15.3f} S1DP={0.0:15.3f} S1IN={0.0:15.3f} S1GD={0.0:15.3f}")
    lines.append(f"S2DR={0.0:15.3f} S2DP={0.0:15.3f} S2IN={0.0:15.3f} S2GD={0.0:15.3f}")
    lines.append(f"S3DR={0.0:15.3f} S3DP={0.0:15.3f} S3IN={0.0:15.3f} S3GD={0.0:15.3f}")
    lines.append("")
    lines.append("  #   X-start    Y-start     X-fin      Y-fin   Kode  rt.lat    reverse   dip angle     top      bot")
    lines.append("xxx xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx xxx xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx")

    for i, r in enumerate(rows, start=1):
        fault = FaultParameters.from_input(
            lon=r["lon"], lat=r["lat"], depth=r["depth"],
            length=r["length"], width=r["width"],
            strike=r["strike"], dip=r["dip"], lon_lat_mode=r["lonlat_mode"],
            rt_lateral_slip=r["rt_lateral_slip"], reverse_slip=r["reverse_slip"],
        )
        (lon1, lat1), (lon2, lat2) = fault.surface_trace()
        xs, ys = geo_to_km(lon1, lat1, lon0, lat0)
        xf, yf = geo_to_km(lon2, lat2, lon0, lat0)
        comment = r["name"]
        lines.append(
            f"{i:3d} {xs:10.4f} {ys:10.4f} {xf:10.4f} {yf:10.4f} "
            f"{100:3d} {r['rt_lateral_slip']:10.4f} {r['reverse_slip']:10.4f} "
            f"{r['dip']:10.4f} {fault.top_depth:10.4f} {fault.bottom_depth:10.4f}   {comment}"
        )

    lines.append("  ")
    lines.append("    Grid Parameters")
    # Start-x/Start-y/Finish-x/Finish-y are local km (same frame as the
    # fault trace endpoints above), matching coulomb.m's own convention
    # -- NOT literal lon/lat. See the matching fix in import_inp() below.
    gx0, gy0 = geo_to_km(grid.lon_min, grid.lat_min, lon0, lat0)
    gx1, gy1 = geo_to_km(grid.lon_max, grid.lat_max, lon0, lat0)
    lines.append(f"  1  ----------------------------  Start-x = {gx0:16.7f}")
    lines.append(f"  2  ----------------------------  Start-y = {gy0:16.7f}")
    lines.append(f"  3  --------------------------   Finish-x = {gx1:16.7f}")
    lines.append(f"  4  --------------------------   Finish-y = {gy1:16.7f}")
    dlon = (gx1 - gx0) / max(grid.n_lon - 1, 1)
    dlat = (gy1 - gy0) / max(grid.n_lat - 1, 1)
    lines.append(f"  5  ------------------------  x-increment = {dlon:16.7f}")
    lines.append(f"  6  ------------------------  y-increment = {dlat:16.7f}")
    lines.append("     Size Parameters")
    lines.append(f"  1  --------------------------  Plot size = {1.0:16.7f}")
    lines.append(f"  2  --------------  Shade/Color increment = {1.0:16.7f}")
    lines.append(f"  3  ------  Exaggeration for disp.& dist. = {1.0:16.7f}")
    lines.append("  ")

    xs_params = dialog._get_cross_section_params()
    lines.append("     Cross section default")
    # Same local-km convention as the Grid Parameters block above (and as
    # coulomb.m's own SECTION(1:4), which it converts via xy2lonlat()).
    sx0, sy0 = geo_to_km(xs_params['lon1'], xs_params['lat1'], lon0, lat0)
    sx1, sy1 = geo_to_km(xs_params['lon2'], xs_params['lat2'], lon0, lat0)
    lines.append(f"  1  ----------------------------  Start-x = {sx0:16.7f}")
    lines.append(f"  2  ----------------------------  Start-y = {sy0:16.7f}")
    lines.append(f"  3  --------------------------   Finish-x = {sx1:16.7f}")
    lines.append(f"  4  --------------------------   Finish-y = {sy1:16.7f}")
    lines.append(f"  5  ------------------  Distant-increment = {xs_params['dist_increment_km']:16.7f}")
    lines.append(f"  6  ----------------------------  Z-depth = {xs_params['max_depth_km']:16.7f}")
    lines.append(f"  7  ------------------------  Z-increment = {xs_params['depth_increment_km']:16.7f}")

    lines.append("     Map info")
    lines.append(f"  1  ---------------------------- min. lon = {grid.lon_min:16.7f}")
    lines.append(f"  2  ---------------------------- max. lon = {grid.lon_max:16.7f}")
    lines.append(f"  3  ---------------------------- zero lon = {lon0:16.7f}")
    lines.append(f"  4  ---------------------------- min. lat = {grid.lat_min:16.7f}")
    lines.append(f"  5  ---------------------------- max. lat = {grid.lat_max:16.7f}")
    lines.append(f"  6  ---------------------------- zero lat = {lat0:16.7f}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ─── Coulomb 3.4.2-compatible .inp import ──────────────────────────────────

_NUM = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"


def _find_float(pattern, text, group=1):
    m = re.search(pattern, text)
    if not m:
        return None
    return float(m.group(group))


def import_inp(dialog, path):
    """
    Read a Coulomb-3.4.2-compatible ASCII .inp file at `path` and
    populate the dialog's Source Faults table (and Elastic Params /
    Grid Output tabs, where present in the file) from it.

    Returns a list of warning strings (empty if nothing to flag) --
    e.g. if the file uses the rake/netslip (IRAKE=1) format instead of
    rt.lat/reverse, or has no "Map info" section (origin defaults to
    (0, 0) in that case).
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    raw_lines = text.splitlines()

    warnings = []

    # #fixed= NUM
    n_fault = _find_float(r"#fixed=\s*(\d+)", text)
    if n_fault is None:
        raise ValueError("Could not find '#fixed=' fault count in file -- "
                         "not a recognizable Coulomb .inp file.")
    n_fault = int(n_fault)

    pois = _find_float(r"PR1=\s*(" + _NUM + r")", text)
    young = _find_float(r"\bE1=\s*(" + _NUM + r")", text)
    fric = _find_float(r"FRIC=\s*(" + _NUM + r")", text)

    is_irake = bool(re.search(r"\brake\b", text[:text.find("xxxxxxxxxx")] if "xxxxxxxxxx" in text else text))
    if is_irake:
        warnings.append(
            "This file uses Coulomb's rake/netslip (IRAKE) fault format, "
            "which this importer does not decode -- fault rows were NOT "
            "imported. Only rt.lat/reverse-format .inp files are supported.")

    # Map info (coordinate origin) -- optional block
    zero_lon = _find_float(r"zero lon\s*=\s*(" + _NUM + r")", text)
    zero_lat = _find_float(r"zero lat\s*=\s*(" + _NUM + r")", text)
    if zero_lon is None or zero_lat is None:
        zero_lon, zero_lat = 0.0, 0.0
        warnings.append(
            "No 'Map info' section (zero lon/zero lat) found in the file "
            "-- using (0, 0) as the coordinate origin. Fault positions may "
            "be offset if the file used a different origin.")

    fault_rows = []
    if not is_irake:
        # Fault data lines: find the dummy 'xxxxxxxxxx...' line, then
        # read fault lines after it as whitespace-separated fields:
        # ID xs ys xf yf KODE val1 val2 dip top bot [comment...]
        #
        # NOTE: `n_fault` (from the header's "#fixed=" field) is Coulomb's
        # own declared fault count, but real-world .inp files (hand-edited
        # or script-generated, e.g. checkerboard test files built by
        # appending rows without updating the header) can have this out
        # of sync with the number of data lines actually present.
        # coulomb.m itself treats this as a hard error ("The total number
        # of faults and the number of data are different... change #fixed
        # in the 3rd row") and refuses to load. This importer is more
        # forgiving: it scans forward from the dummy line for as many
        # contiguous, parseable fault rows as actually exist, uses THAT
        # count, and warns if it disagrees with the declared `#fixed`.
        start_idx = None
        for i, line in enumerate(raw_lines):
            if line.strip().startswith("xxx") and "xxxxxxxxxx" in line:
                start_idx = i + 1
                break
        if start_idx is None:
            raise ValueError(
                "Could not find the fault-table header ('xxx xxxxxxxxxx...') "
                "line -- not a recognizable Coulomb .inp file.")

        def _try_parse_fault_line(line):
            """Return parsed fields for a fault data line, or None if the
            line doesn't look like one (blank, trailer section, etc.)."""
            stripped = line.strip()
            if not stripped:
                return None
            parts = stripped.split(None, 11)
            if len(parts) < 11:
                return None
            try:
                fid = int(float(parts[0]))
                vals = [float(p) for p in parts[1:11]]
            except ValueError:
                return None
            comment = parts[11].strip() if len(parts) > 11 else None
            return (fid, vals, comment)

        k = 0
        while start_idx + k < len(raw_lines):
            parsed = _try_parse_fault_line(raw_lines[start_idx + k])
            if parsed is None:
                break
            _fid, vals, comment = parsed
            xs, ys, xf, yf = vals[0], vals[1], vals[2], vals[3]
            val1, val2 = vals[5], vals[6]
            dip = vals[7]
            top, bot = vals[8], vals[9]
            comment = comment if comment else f"Fault {k + 1}"

            lon1, lat1 = km_to_geo(xs, ys, zero_lon, zero_lat)
            lon2, lat2 = km_to_geo(xf, yf, zero_lon, zero_lat)
            length = float(np.hypot(xf - xs, yf - ys))
            dx, dy = xf - xs, yf - ys
            strike = float(np.degrees(np.arctan2(dx, dy)) % 360.0)
            sin_dip = np.sin(np.deg2rad(dip))
            width = (bot - top) / sin_dip if sin_dip > 1e-6 else 0.0

            fault_rows.append(dict(
                name=comment, lonlat_mode="top_start",
                lon=lon1, lat=lat1, depth=top,
                length=length, width=width, strike=strike, dip=dip,
                rt_lateral_slip=val1, reverse_slip=val2,
                subdiv_l=1, subdiv_w=1,
            ))
            k += 1

        if k != n_fault:
            warnings.append(
                f"Header declares '#fixed={n_fault}' but {k} fault data "
                f"line(s) were actually found and parsed -- imported all "
                f"{k}. If this is wrong, check the file's '#fixed=' value "
                f"on line 3 and the fault-table line count.")
        if k == 0:
            warnings.append("No parseable fault data lines were found "
                           "after the 'xxx xxxxxxxxxx...' header line.")

        if fault_rows:
            dialog.fault_table.set_raw_rows(fault_rows)

    if fric is not None:
        dialog.e_friction.setValue(fric)
    if young is not None and pois is not None:
        mu = young / (2.0 * (1.0 + pois))
        dialog.e_mu.setValue(mu)
    if pois is not None:
        dialog.e_nu.setValue(pois)

    # Grid Parameters: Start-x/Start-y/Finish-x/Finish-y are in the SAME
    # local-km coordinate frame as the fault trace endpoints above (see
    # coulomb.m's own usage: GRID(1)/GRID(2) are added to a km-scaled
    # lon/lat delta when converting back to geographic coords, e.g.
    # `a(:,1) = app.INPUT_VARS.GRID(1) + dlon .* x_per_lon`) -- they are
    # NOT literal lon/lat degrees, despite that being what this plugin's
    # own export_inp() writes (see the matching fix there). Convert via
    # km_to_geo() using the same (zero_lon, zero_lat) origin as the fault
    # traces so genuine Coulomb-produced .inp files load correctly.
    grid_vals = _parse_numbered_block(raw_lines, "Grid Parameters", 6)
    if grid_vals is not None:
        g_lon_min, g_lat_min = km_to_geo(grid_vals[0], grid_vals[1], zero_lon, zero_lat)
        g_lon_max, g_lat_max = km_to_geo(grid_vals[2], grid_vals[3], zero_lon, zero_lat)
        dialog.g_lon_min.setValue(g_lon_min)
        dialog.g_lat_min.setValue(g_lat_min)
        dialog.g_lon_max.setValue(g_lon_max)
        dialog.g_lat_max.setValue(g_lat_max)

    # Cross section default: Start-x/Start-y/Finish-x/Finish-y are also
    # local km (coulomb.m: `xy2lonlat(app, [SECTION(1), SECTION(2)])`),
    # same conversion. Distant-increment/Z-depth/Z-increment are plain
    # distances/depths in km already, no conversion needed. This block
    # was not read at all before this fix (cross-section params silently
    # never round-tripped through .inp import).
    xs_vals = _parse_numbered_block(raw_lines, "Cross section default", 7)
    if xs_vals is not None:
        xs_lon1, xs_lat1 = km_to_geo(xs_vals[0], xs_vals[1], zero_lon, zero_lat)
        xs_lon2, xs_lat2 = km_to_geo(xs_vals[2], xs_vals[3], zero_lon, zero_lat)
        if hasattr(dialog, "xs_lon1"):
            dialog.xs_lon1.setValue(xs_lon1)
            dialog.xs_lat1.setValue(xs_lat1)
            dialog.xs_lon2.setValue(xs_lon2)
            dialog.xs_lat2.setValue(xs_lat2)
            dialog.xs_dist_inc.setValue(xs_vals[4])
            dialog.xs_max_depth.setValue(xs_vals[5])
            dialog.xs_depth_inc.setValue(xs_vals[6])

    return warnings


def _parse_numbered_block(raw_lines, header_text, n_values):
    """
    Find a line containing `header_text` (e.g. "Grid Parameters"), then
    read the next n_values lines' trailing "= <float>" values. Returns a
    list of floats, or None if the block wasn't found.
    """
    start_idx = None
    for i, line in enumerate(raw_lines):
        if header_text in line:
            start_idx = i + 1
            break
    if start_idx is None:
        return None
    values = []
    for i in range(start_idx, min(start_idx + n_values, len(raw_lines))):
        m = re.search(r"=\s*(" + _NUM + r")\s*$", raw_lines[i])
        if not m:
            break
        values.append(float(m.group(1)))
    return values if len(values) == n_values else None
