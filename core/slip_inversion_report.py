# -*- coding: utf-8 -*-
"""
Post-run reporting for core.okada_engine.run_slip_inversion(): a
human-readable QA report, and per-observation-table "augmented" rows
(original input columns + model/predicted value, residual, point RMSE)
for export as CSV or a QGIS layer. Pure Python (only stdlib `math`) so
it's testable outside a real QGIS session, same as the rest of core/.

GNSS and InSAR/LOS observations have different native columns
(component-wise e/n/u vs. a single LOS scalar + look vector), so they
are always reported/augmented as two SEPARATE tables, not force-merged
into one -- mirrors how they're imported (core.observation_import's two
schemas) and how the worker keeps them as two input lists.
"""

import math


def _annex_labels(n):
    """A, B, C, ..., Z, AA, AB, ... -- same convention as
    ui.fault_table_widget._annex_labels(), duplicated here rather than
    imported so this module has no QGIS-importing dependency."""
    labels = []
    for i in range(n):
        label = ""
        k = i
        while True:
            label = chr(ord('A') + k % 26) + label
            k = k // 26 - 1
            if k < 0:
                break
        labels.append(label)
    return labels


def build_slip_inversion_report(fault_segments, elastic,
                                smoothing_factor, max_slip, target_mw,
                                gnss_points, los_points, diag,
                                fixed_rake_deg=None):
    """
    Plain-text QA report: run parameters, solver diagnostics, a
    per-observation fit table (predicted/observed/residual, labeled
    back to lon/lat/type), and a per-patch slip result table for EACH
    fault segment in the group (a single-fault run is just a
    one-element `fault_segments`). Returns a single string (caller
    writes it to disk).

    fault_segments : list of {"name": str, "n_length": int,
                              "n_width": int,
                              "overrides": {(i,j): (rt, rev)}} dicts,
                     one per fault row involved in this run, in the
                     SAME order the inversion's Green's matrix
                     concatenated their patches (only cosmetic here --
                     each segment's own table is self-contained).
    """
    lines = []
    names = ", ".join(seg["name"] for seg in fault_segments)
    total_patches = sum(seg["n_length"] * seg["n_width"] for seg in fault_segments)
    lines.append(f"Slip inversion QA report -- {names}")
    lines.append("=" * 64)
    if len(fault_segments) > 1:
        lines.append(f"Group of {len(fault_segments)} fault segments, "
                     f"{total_patches} total patches:")
        for seg in fault_segments:
            lines.append(f"  - {seg['name']}: {seg['n_length']} (along-strike) x "
                         f"{seg['n_width']} (down-dip) = "
                         f"{seg['n_length'] * seg['n_width']} patches")
    else:
        seg = fault_segments[0]
        lines.append(f"Subdivision: {seg['n_length']} (along-strike) x "
                     f"{seg['n_width']} (down-dip) = "
                     f"{seg['n_length'] * seg['n_width']} patches")
    lines.append(f"Elastic: mu={elastic.mu:.4g} Pa, nu={elastic.nu:.4g}")
    lines.append(f"Smoothing factor: {smoothing_factor:g}")
    lines.append(f"Max |slip| bound: {max_slip:g} m")
    lines.append("Target Mw constraint: " +
                 (f"{target_mw:g} (total, across all segments)" if target_mw is not None
                  else "(none -- unconstrained)"))
    lines.append("Rake constraint: " +
                 (f"FIXED at {fixed_rake_deg:g}° (1 unknown/patch -- signed slip "
                  f"magnitude only; rt_lateral/reverse below are exactly on this "
                  f"rake by construction)" if fixed_rake_deg is not None
                  else "(none -- free rt_lateral/reverse inversion, 2 unknowns/patch)"))
    lines.append(f"GNSS/field points: {len(gnss_points)}   InSAR/LOS points: {len(los_points)}")
    lines.append("")
    lines.append("Solver diagnostics (whole joint solve)")
    lines.append("-" * 64)
    lines.append(f"  success      : {diag.get('solver_success')}")
    lines.append(f"  message      : {diag.get('solver_message')}")
    lines.append(f"  n_iter       : {diag.get('n_iter')}")
    lines.append(f"  n_data       : {diag.get('n_data')}")
    lines.append(f"  rms_misfit   : {diag.get('rms_misfit'):.6g}")
    lines.append(f"  achieved_mw  : {diag.get('achieved_mw'):.4f}")
    lines.append("")
    lines.append("Per-observation fit (residual = predicted - observed)")
    lines.append("-" * 64)
    lines.append(f"{'idx':>4} {'type':>4} {'lon':>10} {'lat':>10} "
                 f"{'observed':>12} {'predicted':>12} {'residual':>12}")
    predicted = diag.get("predicted", [])
    observed = diag.get("observed", [])
    labels = diag.get("component_labels", [])
    for row_idx, (obs_idx, comp) in enumerate(labels):
        pt = los_points[obs_idx] if comp == "los" else gnss_points[obs_idx]
        pred, obs = predicted[row_idx], observed[row_idx]
        lines.append(f"{obs_idx:>4} {comp:>4} {pt['lon']:>10.5f} {pt['lat']:>10.5f} "
                     f"{obs:>12.5g} {pred:>12.5g} {pred - obs:>12.5g}")

    for seg in fault_segments:
        n_length, n_width, overrides = seg["n_length"], seg["n_width"], seg["overrides"]
        lines.append("")
        lines.append(f"Per-patch slip result -- {seg['name']} "
                     f"(Coulomb convention: U1=-rt_lateral, U2=reverse)")
        lines.append("-" * 64)
        lines.append(f"{'patch':>6} {'i':>3} {'j':>3} {'rt_lateral_m':>13} "
                     f"{'reverse_m':>11} {'magnitude_m':>12} {'rake_deg':>9}")
        patch_labels = _annex_labels(n_length * n_width)
        flat = 0
        for i in range(n_width):
            for j in range(n_length):
                rt, rev = overrides[(i, j)]
                mag = math.hypot(rt, rev)
                rake = math.degrees(math.atan2(rev, -rt)) if mag > 0 else 0.0
                lines.append(f"{patch_labels[flat]:>6} {i:>3} {j:>3} {rt:>13.5g} "
                             f"{rev:>11.5g} {mag:>12.5g} {rake:>9.2f}")
                flat += 1

    return "\n".join(lines) + "\n"


def build_augmented_gnss_rows(gnss_points, diag):
    """
    One dict per input GNSS/field point: original keys (lon, lat, e, n,
    u, sigma_e, sigma_n, sigma_u) PLUS pred_e/pred_n/pred_u (None for a
    component that point didn't provide -- it contributed no row to the
    solve, so there's nothing to predict against), resid_e/resid_n/
    resid_u (predicted - observed), point_rmse (RMS of that point's own
    used-component residuals, None if the point contributed nothing),
    and overall_rmse_m (diag['rms_misfit'], repeated on every row so a
    single exported table/layer is self-contained without the QA report).
    """
    predicted, observed = diag.get("predicted", []), diag.get("observed", [])
    labels = diag.get("component_labels", [])
    overall_rmse = diag.get("rms_misfit")

    pred_map, resid_map = {}, {}
    for row_idx, (obs_idx, comp) in enumerate(labels):
        if comp in ("e", "n", "u"):
            pred_map[(obs_idx, comp)] = predicted[row_idx]
            resid_map[(obs_idx, comp)] = predicted[row_idx] - observed[row_idx]

    out = []
    for i, pt in enumerate(gnss_points):
        row = dict(pt)
        point_residuals = []
        for comp in ("e", "n", "u"):
            key = (i, comp)
            row[f"pred_{comp}"] = pred_map.get(key)
            row[f"resid_{comp}"] = resid_map.get(key)
            if key in resid_map:
                point_residuals.append(resid_map[key])
        row["point_rmse"] = (math.sqrt(sum(r * r for r in point_residuals) / len(point_residuals))
                             if point_residuals else None)
        row["overall_rmse_m"] = overall_rmse
        out.append(row)
    return out


def build_augmented_los_rows(los_points, diag):
    """
    One dict per input InSAR/LOS point: original keys (lon, lat, los,
    look_e, look_n, look_u, sigma) PLUS pred_los, resid_los (predicted -
    observed), and overall_rmse_m (repeated, same reasoning as
    build_augmented_gnss_rows()). Every LOS row is always fully used (no
    optional components), so unlike the GNSS case there's no per-point
    "which components contributed" ambiguity -- pred_los/resid_los are
    always populated.
    """
    predicted, observed = diag.get("predicted", []), diag.get("observed", [])
    labels = diag.get("component_labels", [])
    overall_rmse = diag.get("rms_misfit")

    pred_map, resid_map = {}, {}
    for row_idx, (obs_idx, comp) in enumerate(labels):
        if comp == "los":
            pred_map[obs_idx] = predicted[row_idx]
            resid_map[obs_idx] = predicted[row_idx] - observed[row_idx]

    out = []
    for i, pt in enumerate(los_points):
        row = dict(pt)
        row["pred_los"] = pred_map.get(i)
        row["resid_los"] = resid_map.get(i)
        row["overall_rmse_m"] = overall_rmse
        out.append(row)
    return out
