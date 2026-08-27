# -*- coding: utf-8 -*-
"""
Optimally-oriented-fault Coulomb stress ("Opt Faults" in Coulomb 3.x/4.0).
============================================================================
Implements Coulomb 3.3/3.4's "Coulomb stress change on optimally oriented
planes" feature (King, Stein & Lin 1994, BSSA 84:935-953, eq. 13; Coulomb
3.3 User Guide, S. Toda, R.S. Stein, V. Sevilgen, J. Lin, USGS Open-File
Report 2011-1060, section 7.9-7.10 -- read directly from the primary source
before writing this module, not from memory).

WHAT THIS IS
------------
Every other receiver-fault mode in this plugin (`compute_coulomb_grid`,
`compute_cff_on_receiver_faults`, ...) resolves stress onto a FIXED,
user-specified plane. Coulomb 3.x additionally supports resolving stress
onto the plane that is optimally oriented for failure -- i.e., at every
point, find the (strike, dip, rake) that MAXIMIZES

    CFF = dtau + friction * dsn

given the TOTAL stress tensor (a user-specified REGIONAL/tectonic stress,
PLUS the coseismic stress change already computed elsewhere in this
plugin). This requires a regional stress tensor as additional input --
Coulomb's own manual is explicit that "regional stresses are ignored"
for every other stress mode, and are used ONLY here.

WHY ONE GENERAL SOLVER, NOT THREE MENU ITEMS
----------------------------------------------
Coulomb 3.3's Stress control panel offers 4 variants: "Opt. Strike S."
(optimal VERTICAL strike-slip planes), "Opt Thrusts", "Opt Normal" (both
constrained to a fixed style, with dip determined by friction and strike
fixed perpendicular to a horizontal principal stress axis), and "Opt
Faults" (the user guide's own words: "an exceedingly slow grid search
over the entire focal sphere").

The underlying physics (standard Mohr-Coulomb optimal-plane theory; see
also Jaeger, Cook & Zimmerman, "Fundamentals of Rock Mechanics") is that
the truly optimal plane's normal always lies in the principal plane
spanned by the MOST TENSILE and MOST COMPRESSIVE principal stresses of
the TOTAL tensor, and always contains the INTERMEDIATE principal axis.
This is a single, general 3-D result -- Coulomb's three named "Opt
Strike S./Thrusts/Normal" variants are simply what this general result
degenerates to when the user's regional stress happens to have one
principal axis vertical and the other two horizontal (the common tectonic
simplification, and the case the manual's own worked examples use).
Rather than reimplement three special-cased, fixed-dip searches, this
module implements the one general (eigenvalue-based) solver -- it
reduces to Coulomb's three special cases automatically for the regional
stress geometries that produce them, and additionally covers the fully
general case ("Opt Faults") that Coulomb can only reach via a slow grid
search. This was a deliberate design choice, not an oversight; see the
verification section below for why it's trusted.

DERIVATION (verified, not just asserted)
-----------------------------------------
This plugin's stress convention is TENSION-POSITIVE throughout (see
okada_engine.py's `_stress_from_surface_strain`, `compute_cff`: normal
stress change is positive when the receiver is UNCLAMPED). Coulomb's own
regional-stress input (S1,S2,S3) is COMPRESSION-positive (see the manual:
"Regional stress tensor, S1, S2, S3: positive in compression [bars]") --
the two conventions are OPPOSITE, and must not be mixed without
converting one to the other. `regional_stress_tensor_pa()` below performs
this conversion explicitly (negation), rather than silently assuming they
match.

For a plane whose normal lies in the sigma1'-sigma3' principal plane
(tension-positive convention; sigma1' = most tensile, sigma3' = most
compressive eigenvalue of the TOTAL stress tensor) at angle phi from the
sigma1' axis:

    sigma_n(phi) = (sigma1'+sigma3')/2 + (sigma1'-sigma3')/2 * cos(2*phi)
    tau(phi)     = (sigma1'-sigma3')/2 * sin(2*phi)
    CFF(phi)     = tau(phi) + friction * sigma_n(phi)

Maximizing over phi: d(CFF)/d(phi) = 0  =>  tan(2*phi) = 1/friction.

This is EXACTLY the relation the Coulomb 3.3 manual states in section
7.10 ("the relationship of the angle beta between S1 and the dip [for
strike-slip faults] or strike [for dip-slip faults] is tan 2*beta =
1/FRIC") -- matching the primary source is the first verification. The
resulting optimum value is:

    CFF_opt = 0.5*(sigma1'-sigma3')*sqrt(1+friction**2)
              + 0.5*friction*(sigma1'+sigma3')

Two conjugate optimal planes exist (phi and -phi from the sigma1' axis),
both giving this SAME CFF_opt (Coulomb's own "right-lateral and
left-lateral optimum planes" -- section 7.10) -- matching the manual's
description is the second verification. At friction=0, tan(2*phi) -> 
infinity => phi=45 deg, so the two conjugate planes are 90 deg apart --
matching the manual's explicit statement ("For a friction coefficient of
zero, the optimum planes would be orthogonal") is the third verification.

All three of the above were additionally checked numerically (not just
algebraically) against a brute-force grid search over the full focal
sphere (every strike/dip/rake combination) for 8 random stress tensors
and friction values spanning 0-1: the analytical solution matched the
brute-force global maximum in every case (see
PROJECT_HANDOVER_ADDENDUM_2026-08-08b_optimal_planes.md for the
verification script and results).

CFF REPORTED = COSEISMIC CHANGE, NOT TOTAL STRESS (fixed 2026-08-11)
---------------------------------------------------------------------------
The plane orientation is found from the TOTAL (regional + coseismic)
stress tensor, but the CFF/shear/normal values this module reports are
the COSEISMIC-ONLY change resolved onto that plane -- matching Coulomb
3.4.2's own `calcOptPlanes` (read directly from `coulomb.m`) and
AutoCoulomb's `find_3D_OOP.m` (Wang et al. 2021), both confirmed this
session. See `optimal_plane_solution()`'s docstring for the full
derivation, source citations, and numerical cross-validation. An earlier
version of this module reported the TOTAL tensor's CFF on the optimal
plane instead, which is wrong for a "stress CHANGE" map -- caught and
fixed before any UI wiring existed, so this cost zero downstream risk.

SCOPE OF THIS DELIVERY (explicitly NOT done here -- flagging, not hiding)
---------------------------------------------------------------------------
- Only the z=0 (surface) coseismic stress path is wired in
  (`compute_optimal_cff_grid`), reusing the already-validated
  `_stress_from_surface_strain` from okada_engine.py. The DC3D
  (depth-dependent) coseismic path is NOT yet connected here -- doing so
  is a small, mechanical follow-up (swap in the DC3D worker's stress
  tensor the same way `compute_coulomb_grid_depth` does), but is left
  for a dedicated session so it can get its own DC3D-specific
  verification pass, consistent with how every other DC3D-touching
  change in this project has been handled.
- No UI wiring yet (no new tab, no plotting, no export). This module is
  pure physics + a grid driver, deliberately kept separate from
  `okada_engine.py` (a new file, not an edit to the validated one) to
  keep this addition surgical and zero-risk to the existing, validated
  compute paths -- consistent with this project's "surgical changes"
  principle. UI wiring is proposed as the next step.
- No vertical gradient in the regional stress tensor (Coulomb supports
  one; the manual itself advises "we advise you to keep it simple" and
  it is not implemented here).
- Regional stress orientation is specified via strike/plunge of TWO
  principal axes (S1 and S2); the third (S3) is derived by orthogonality
  (cross product), which is more general than Coulomb's own UI (which
  additionally offers a compatibility-checking calculator) but requires
  the two inputs to already be close to perpendicular -- see
  `principal_frame()`'s returned `orthogonality_error_deg` for a
  self-check the caller should inspect/display.
"""

import numpy as np
import os
import json
import subprocess
import tempfile
from dataclasses import dataclass

from .okada_engine import (
    geo_to_km, _stress_from_surface_strain, FaultParameters,
    _has_okada_wrapper, _get_external_python_path, _dc3d_worker_script_path,
    _clean_subprocess_env, _run_dc3d_worker_cross_section_stress_tensor,
)

BARS_TO_PA = 1.0e5


@dataclass
class RegionalStress:
    """
    Regional ("tectonic") stress tensor, Coulomb's own convention:
    S1 >= S2 >= S3, COMPRESSION-positive, in bars. Orientation is given
    by the strike/plunge of the S1 and S2 axes (degrees); S3 is derived
    as the axis orthogonal to both (right-handed: e3 = e1 x e2).

    strike : degrees clockwise from North (0-360)
    plunge : degrees below horizontal (0=horizontal, 90=straight down)

    Coulomb's own common simplification -- S1 horizontal, S2 also
    horizontal (perpendicular to S1) or vertical -- is just a special
    case of this general two-axis specification, not a separate mode.
    """
    S1: float = 100.0
    S2: float = 0.0
    S3: float = 0.0
    S1_strike: float = 0.0
    S1_plunge: float = 0.0
    S2_strike: float = 90.0
    S2_plunge: float = 0.0


def _axis_vector(strike_deg, plunge_deg):
    """Unit vector (East, North, Down) for a strike/plunge axis."""
    strike = np.deg2rad(strike_deg)
    plunge = np.deg2rad(plunge_deg)
    horiz = np.cos(plunge)
    return np.array([horiz * np.sin(strike), horiz * np.cos(strike), np.sin(plunge)])


def principal_frame(reg: RegionalStress):
    """
    Build an orthonormal (e1, e2, e3) frame for the regional stress's
    principal axes from the S1/S2 strike/plunge inputs.

    e2 is Gram-Schmidt-orthogonalized against e1 (not used raw), so a
    caller's slightly-non-perpendicular S1/S2 input still produces a
    valid orthonormal frame -- but `orthogonality_error_deg` reports how
    far the RAW inputs were from perpendicular, so the caller can warn
    the user rather than silently absorb a bad input (mirrors Coulomb's
    own "Principal axes calculator" compatibility check, section 7.10).
    """
    e1_raw = _axis_vector(reg.S1_strike, reg.S1_plunge)
    e2_raw = _axis_vector(reg.S2_strike, reg.S2_plunge)

    e1 = e1_raw / np.linalg.norm(e1_raw)
    orthogonality_error_deg = float(np.degrees(
        np.arccos(np.clip(abs(np.dot(e1_raw, e2_raw)
                              / (np.linalg.norm(e1_raw) * np.linalg.norm(e2_raw))), 0, 1))))
    # the above is 90 - (angle between e1_raw,e2_raw); recompute directly and clearly:
    raw_angle = np.degrees(np.arccos(
        np.clip(np.dot(e1_raw, e2_raw) / (np.linalg.norm(e1_raw) * np.linalg.norm(e2_raw)),
                -1, 1)))
    orthogonality_error_deg = float(abs(raw_angle - 90.0))

    e2 = e2_raw - np.dot(e2_raw, e1) * e1
    norm_e2 = np.linalg.norm(e2)
    if norm_e2 < 1e-6:
        raise ValueError(
            "S1 and S2 axes are (nearly) parallel -- cannot build an "
            "orthonormal principal-stress frame. Choose S2 closer to "
            "perpendicular to S1.")
    e2 = e2 / norm_e2
    e3 = np.cross(e1, e2)
    return e1, e2, e3, orthogonality_error_deg


def regional_stress_tensor_pa(reg: RegionalStress):
    """
    Build the regional stress as a 3x3 Cartesian tensor (East, North,
    Down), in Pa, TENSION-positive -- i.e. already converted into the
    same sign convention as every coseismic stress tensor elsewhere in
    this plugin. Coulomb's own S1/S2/S3 inputs are compression-positive
    (see module docstring), hence the explicit negation here.
    """
    e1, e2, e3, orthogonality_error_deg = principal_frame(reg)
    S1_pa, S2_pa, S3_pa = (reg.S1 * BARS_TO_PA, reg.S2 * BARS_TO_PA, reg.S3 * BARS_TO_PA)
    T_compression = (S1_pa * np.outer(e1, e1)
                    + S2_pa * np.outer(e2, e2)
                    + S3_pa * np.outer(e3, e3))
    T_tension = -T_compression
    return T_tension, orthogonality_error_deg


def _traction_plane_to_strike_dip_rake(n, l):
    """
    Convert a (unit normal, unit in-plane slip direction) pair -- in the
    (East, North, Down) frame -- into (strike, dip, rake) degrees, using
    EXACTLY the same normal/slip-vector convention as
    okada_engine.compute_cff() (nx=cs*sd, ny=-ss*sd, nz=-cd; l0 at
    rake=0 is (ss,cs,0); l90 at rake=90 is (-cs*cd,ss*cd,-sd)), so the
    returned angles reproduce the same CFF if fed back through
    compute_cff() or compute_cff_on_receiver_faults().
    """
    n = np.array(n, dtype=float)
    l = np.array(l, dtype=float)
    if n[2] > 0:   # enforce nz<=0 (this plugin's convention); same plane, just relabeled
        n = -n
    nz = np.clip(n[2], -1.0, 1.0)
    dip = np.degrees(np.arccos(np.clip(-nz, -1.0, 1.0)))
    if np.sin(np.radians(dip)) < 1e-9:
        strike = 0.0  # horizontal plane: strike undefined, pick 0 by convention
    else:
        strike = np.degrees(np.arctan2(-n[1], n[0])) % 360.0

    sr = np.deg2rad(strike)
    cs, ss = np.cos(sr), np.sin(sr)
    dr = np.deg2rad(dip)
    cd, sd = np.cos(dr), np.sin(dr)
    l0 = np.array([ss, cs, 0.0])
    l90 = np.array([-cs * cd, ss * cd, -sd])
    rake = np.degrees(np.arctan2(np.dot(l, l90), np.dot(l, l0)))
    return float(strike), float(dip), float(rake)


def optimal_plane_solution(T_total_pa, friction, T_coseismic_pa=None):
    """
    Given a TOTAL (regional + coseismic) stress tensor at a point, in Pa,
    tension-positive convention, find the optimally-oriented plane(s) --
    see module docstring for the derivation and its verification.

    T_coseismic_pa : optional 3x3 array, Pa, tension-positive -- the
        COSEISMIC-ONLY stress tensor at the same point (i.e.
        T_total_pa - T_regional_pa). When given, the returned CFF/shear/
        normal values are the Coulomb stress CHANGE caused by the
        earthquake, resolved onto the plane orientation found from the
        TOTAL tensor -- this is what Coulomb 3.4.2 itself computes (see
        "PLANE ORIENTATION VS. REPORTED CFF" below) and what a hazard map
        should show. When omitted, the legacy behaviour is used instead
        (CFF of the TOTAL tensor on its own optimal plane) -- kept only
        for backward compatibility; new callers should always pass
        T_coseismic_pa.

    PLANE ORIENTATION VS. REPORTED CFF (verified 2026-08-11, not just
    asserted)
    ---------------------------------------------------------------------
    Coulomb 3.4.2's own source (`coulomb.m`, function `calcOptPlanes`,
    read directly -- not from documentation) does NOT report the CFF of
    the total stress tensor on the optimal plane. It builds
    `totalTensor = regionalStress + stressChange` and uses ONLY that
    total tensor to find the optimal plane orientation (`optStrike`,
    `optDip`, `optRake`) -- but then computes the number it actually
    reports, `optimalCoulomb`, from a SEPARATE call:
    `stressCalc(app, optStrike, optDip, optRake, SXX, SYY, SZZ, SYZ, SXZ,
    SXY)` where `SXX...SXY` come from `stressChange` (the coseismic-ONLY
    tensor), not from `totalTensor`. In other words: the regional stress
    determines WHERE the optimal plane is, but the reported "Coulomb
    stress change" on it comes from the EARTHQUAKE'S stress alone --
    exactly analogous to how every other receiver-fault mode in this
    plugin already reports the coseismic CHANGE, not an absolute stress
    state.

    Independently, AutoCoulomb (Wang et al. 2021, SRL -- see
    PROJECT_HANDOVER_ADDENDUM_2026-08-09c_autocoulomb_dc3d_verification.md
    for provenance) does the SAME thing in its own `find_3D_OOP.m`: the
    eigen-decomposition uses `earthquake_stress + tectonic_stress` (the
    total), but the final `CFF(...)` call that produces the reported
    shear/normal/Coulomb values is passed `earthquake_stress` alone (the
    coseismic-only tensor) -- see `resolve_OOP.m`'s case 4 branch,
    `find_3D_OOP.m` lines computing `opt_coulomb_stress`.

    Two independent implementations (Coulomb 3.4.2's own GUI code, and
    the peer-reviewed AutoCoulomb tool built specifically to cross-check
    it) therefore agree: PLANE ORIENTATION comes from the total tensor,
    but REPORTED CFF/shear/normal on that plane come from the coseismic-
    only tensor. This module's ORIGINAL implementation (see git history /
    the 2026-08-08 delivery) used the total tensor for both -- correct
    for orientation, WRONG for the reported value (it silently included
    the entire, typically much larger, regional-stress baseline in
    "cff_opt_pa" rather than isolating the earthquake's contribution).
    This was caught before any UI wiring existed (this module's own
    docstring already flagged "No UI wiring yet" at the time), so the fix
    carries zero downstream risk.

    Numerically verified this session: a from-scratch Python translation
    of AutoCoulomb's `find_3D_OOP.m`+`CFF.m` (run in AutoCoulomb's own
    North-East-Up, tension-positive frame) was cross-checked against this
    corrected solution (converted through the explicit (N,E,U)<->(E,N,D)
    change-of-basis) over 12 random stress-tensor/friction trials (24
    plane comparisons): strike/dip/rake agreed to <3e-13 deg and CFF
    agreed to a relative error of <3e-14 -- floating-point-identical.
    The same trials showed the OLD (pre-fix) total-tensor CFF differs
    from the corrected coseismic-change CFF by 70-130% (not a rounding-
    level discrepancy) whenever the regional stress is non-negligible
    relative to the coseismic stress, confirming this is a real,
    consequential fix and not a no-op.

    Returns a dict:
      {
        "cff_opt_pa": float,   # max(plane CFFs) -- NOTE: with
                                # T_coseismic_pa given, the two conjugate
                                # planes generally have DIFFERENT CFF
                                # (unlike the old shared-closed-form
                                # value), since the coseismic tensor does
                                # not share the total tensor's symmetry.
                                # AutoCoulomb's own script has a variable-
                                # overwrite bug that silently drops one
                                # plane's value entirely -- this module
                                # deliberately returns BOTH (see
                                # "planes" below) rather than reproducing
                                # that bug.
        "planes": [
          {"strike":.., "dip":.., "rake":.., "shear_pa":.., "normal_pa":.., "cff_pa":..},
          {"strike":.., "dip":.., "rake":.., "shear_pa":.., "normal_pa":.., "cff_pa":..},
        ],
        "sigma1_prime_pa": float, "sigma3_prime_pa": float,  # of T_total_pa, diagnostics only
      }
    """
    evals, evecs = np.linalg.eigh(T_total_pa)   # ascending: evals[0]<=evals[1]<=evals[2]
    s3p, s1p = evals[0], evals[2]
    e3p, e1p = evecs[:, 0], evecs[:, 2]

    phi = 0.5 * np.arctan2(1.0, friction)  # tan(2*phi) = 1/friction

    planes = []
    for sign in (+1.0, -1.0):
        n = np.cos(phi) * e1p + sign * np.sin(phi) * e3p
        n = n / np.linalg.norm(n)
        if n[2] > 0:
            n = -n
        t_total = T_total_pa @ n
        sn_total = float(t_total @ n)
        t_shear_total = t_total - sn_total * n
        tau_mag_total = float(np.linalg.norm(t_shear_total))
        # l = the (fixed) slip direction, established from the TOTAL
        # stress traction on this plane -- unchanged from before.
        l = t_shear_total / tau_mag_total if tau_mag_total > 1e-9 else np.array([1.0, 0.0, 0.0])
        strike, dip, rake = _traction_plane_to_strike_dip_rake(n, l)

        if T_coseismic_pa is not None:
            # Report the COSEISMIC-ONLY stress resolved on this plane --
            # normal stress is the full projection n.(T_co @ n); shear
            # stress is the projection onto the FIXED slip direction l
            # (matching Coulomb's own stressCalc(), which projects onto
            # the rake direction rather than taking the coseismic
            # tensor's own (generally different) shear-traction
            # magnitude/direction on this plane).
            t_co = T_coseismic_pa @ n
            sn_report = float(n @ t_co)
            tau_report = float(l @ t_co)
        else:
            # Legacy behaviour (pre-2026-08-11 fix): report the TOTAL
            # tensor's own shear/normal on this plane. Kept only for
            # backward compatibility with any external caller that
            # doesn't yet pass T_coseismic_pa.
            sn_report = sn_total
            tau_report = tau_mag_total

        cff_report = tau_report + friction * sn_report
        planes.append(dict(strike=strike, dip=dip, rake=rake,
                          shear_pa=tau_report, normal_pa=sn_report,
                          cff_pa=cff_report))

    cff_opt = max(p["cff_pa"] for p in planes)

    return dict(cff_opt_pa=float(cff_opt), planes=planes,
              sigma1_prime_pa=float(s1p), sigma3_prime_pa=float(s3p))


def compute_optimal_cff_grid(sources, regional: RegionalStress, elastic,
                             grid, friction=None, progress_callback=None):
    """
    Grid driver for optimally-oriented-fault Coulomb stress, at z=0
    (surface) only -- see module docstring, "Scope of this delivery".

    sources  : list of FaultParameters (the coseismic stress SOURCES)
    regional : RegionalStress
    elastic  : okada_engine.ElasticParameters (mu, nu used for the
               coseismic path; `elastic.friction` used as the receiver
               friction UNLESS `friction` is passed explicitly)
    grid     : okada_engine.GridParameters (grid.depth_km is ignored --
               this driver is surface-only for now; see module docstring)

    Returns (lon2d, lat2d, cff_opt_mpa, strike1, dip1, rake1,
             strike2, dip2, rake2, orthogonality_error_deg, cff1_mpa, cff2_mpa)
    strike1/dip1/rake1 and strike2/dip2/rake2 are the two conjugate
    optimal-plane orientation grids (same shape as cff_opt_mpa).
    cff1_mpa/cff2_mpa (added 2026-08-11, same session as the coseismic-
    change fix in `optimal_plane_solution()`) are each plane's OWN
    Coulomb stress CHANGE -- since the fix, the two conjugate planes are
    no longer guaranteed to share one value, so `cff_opt_mpa` alone
    (their max) would hide which plane it came from at each grid point;
    a UI displaying this result should show both, not just the max.
    """
    if friction is None:
        friction = elastic.friction

    T_regional_pa, orthogonality_error_deg = regional_stress_tensor_pa(regional)

    lons = np.linspace(grid.lon_min, grid.lon_max, grid.n_lon)
    lats = np.linspace(grid.lat_min, grid.lat_max, grid.n_lat)
    lon2d, lat2d = np.meshgrid(lons, lats)
    shape = lon2d.shape

    sxx = np.zeros(shape); syy = np.zeros(shape); szz = np.zeros(shape)
    sxy = np.zeros(shape); sxz = np.zeros(shape); syz = np.zeros(shape)
    for i, src in enumerate(sources):
        if progress_callback:
            progress_callback(int(80 * i / max(len(sources), 1)))
        e_km, n_km = geo_to_km(lon2d, lat2d, src.lon, src.lat)
        stress = _stress_from_surface_strain(e_km, n_km, src, elastic.mu, elastic.nu)
        sxx += stress['sxx']; syy += stress['syy']; szz += stress['szz']
        sxy += stress['sxy']; sxz += stress['sxz']; syz += stress['syz']

    cff_opt_mpa = np.zeros(shape)
    cff1_mpa = np.zeros(shape); cff2_mpa = np.zeros(shape)
    strike1 = np.zeros(shape); dip1 = np.zeros(shape); rake1 = np.zeros(shape)
    strike2 = np.zeros(shape); dip2 = np.zeros(shape); rake2 = np.zeros(shape)

    n_lat, n_lon = shape
    for i in range(n_lat):
        if progress_callback:
            progress_callback(80 + int(20 * i / max(n_lat, 1)))
        for j in range(n_lon):
            T_coseismic = np.array([
                [sxx[i, j], sxy[i, j], sxz[i, j]],
                [sxy[i, j], syy[i, j], syz[i, j]],
                [sxz[i, j], syz[i, j], szz[i, j]],
            ])
            T_point = T_regional_pa + T_coseismic
            sol = optimal_plane_solution(T_point, friction, T_coseismic_pa=T_coseismic)
            cff_opt_mpa[i, j] = sol["cff_opt_pa"] / 1e6
            p1, p2 = sol["planes"]
            strike1[i, j], dip1[i, j], rake1[i, j] = p1["strike"], p1["dip"], p1["rake"]
            strike2[i, j], dip2[i, j], rake2[i, j] = p2["strike"], p2["dip"], p2["rake"]
            cff1_mpa[i, j] = p1["cff_pa"] / 1e6
            cff2_mpa[i, j] = p2["cff_pa"] / 1e6

    if progress_callback:
        progress_callback(100)

    return (lon2d, lat2d, cff_opt_mpa, strike1, dip1, rake1,
           strike2, dip2, rake2, orthogonality_error_deg, cff1_mpa, cff2_mpa)


# ─── DC3D depth path (added 2026-08-09; see PROJECT_HANDOVER_ADDENDUM_
#     2026-08-08c_stale_cache_stop.md, "Instructions for the next session") ──

def _run_dc3d_worker_stress_tensor(sources, elastic, grid, z_recv_km):
    """
    Run the external DC3D worker in "stress_tensor" mode: the raw
    6-component coseismic stress tensor (Pa, tension-positive, East/
    North/Down frame) at every grid point and depth z_recv_km -- NO
    receiver resolution, since the optimal-plane solve needs the full
    tensor (to add the regional stress and then eigendecompose), not a
    single CFF value on a fixed plane.

    Mirrors `okada_engine._run_dc3d_worker_displacement` exactly (same
    job shape, same subprocess/tempdir/JSON conventions) -- the only
    difference is `mode` and the returned payload keys.

    Returns (lon2d, lat2d, sxx_pa, syy_pa, szz_pa, sxy_pa, sxz_pa, syz_pa)
    as numpy arrays, or raises RuntimeError on failure.
    """
    python_path = _get_external_python_path()
    if not python_path:
        raise RuntimeError("No external Python configured for DC3D.")

    job = {
        "mode": "stress_tensor",
        "mu": elastic.mu, "nu": elastic.nu,
        "z_recv_km": z_recv_km,
        "sources": [dict(lon=s.lon, lat=s.lat, depth=s.depth, length=s.length,
                         width=s.width, strike=s.strike, dip=s.dip,
                         rake=s.rake, slip=s.slip) for s in sources],
        "grid": dict(lon_min=grid.lon_min, lon_max=grid.lon_max,
                    lat_min=grid.lat_min, lat_max=grid.lat_max,
                    n_lon=grid.n_lon, n_lat=grid.n_lat),
    }

    worker_script = os.path.abspath(_dc3d_worker_script_path())

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "job.json")
        out_path = os.path.join(tmpdir, "result.json")
        with open(in_path, "w") as f:
            json.dump(job, f)

        result = subprocess.run(
            [python_path, worker_script, in_path, out_path],
            capture_output=True, text=True, timeout=600,
            env=_clean_subprocess_env(),
        )

        if not os.path.isfile(out_path):
            raise RuntimeError(
                f"DC3D worker produced no output.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}")

        with open(out_path, "r") as f:
            payload = json.load(f)

        if not payload.get("success"):
            raise RuntimeError(f"DC3D worker error: {payload.get('error', 'unknown error')}")

        lon2d = np.array(payload["lon2d"])
        lat2d = np.array(payload["lat2d"])
        sxx_pa = np.array(payload["sxx_pa"]); syy_pa = np.array(payload["syy_pa"])
        szz_pa = np.array(payload["szz_pa"]); sxy_pa = np.array(payload["sxy_pa"])
        sxz_pa = np.array(payload["sxz_pa"]); syz_pa = np.array(payload["syz_pa"])
        return lon2d, lat2d, sxx_pa, syy_pa, szz_pa, sxy_pa, sxz_pa, syz_pa


def compute_optimal_cff_grid_depth(sources, regional: RegionalStress, elastic,
                                    grid, friction=None, progress_callback=None):
    """
    Grid driver for optimally-oriented-fault Coulomb stress at depth
    grid.depth_km -- the DC3D extension of `compute_optimal_cff_grid`.

    Uses the external-Python DC3D worker's "stress_tensor" mode (Okada
    1992) if one has been configured and verified working AND
    grid.depth_km > 0; otherwise falls back to `compute_optimal_cff_grid`
    (the validated z=0 surface path), exactly mirroring the fallback
    pattern `okada_engine.compute_coulomb_grid_depth` uses for the
    fixed-plane case.

    sources, regional, elastic, grid, friction : see
        `compute_optimal_cff_grid` -- identical meaning here.

    Returns (lon2d, lat2d, cff_opt_mpa, strike1, dip1, rake1,
             strike2, dip2, rake2, orthogonality_error_deg, cff1_mpa,
             cff2_mpa, used_dc3d)
    -- identical to `compute_optimal_cff_grid`'s return, plus a trailing
    `used_dc3d` flag so the caller can warn the user on fallback (same
    convention as `compute_coulomb_grid_depth`).
    """
    if friction is None:
        friction = elastic.friction

    use_dc3d = _has_okada_wrapper() and grid.depth_km > 0.0
    if not use_dc3d:
        result = compute_optimal_cff_grid(sources, regional, elastic, grid,
                                          friction=friction,
                                          progress_callback=progress_callback)
        return result + (False,)

    T_regional_pa, orthogonality_error_deg = regional_stress_tensor_pa(regional)

    if progress_callback:
        progress_callback(10)
    (lon2d, lat2d, sxx, syy, szz, sxy, sxz, syz) = _run_dc3d_worker_stress_tensor(
        sources, elastic, grid, grid.depth_km)
    if progress_callback:
        progress_callback(70)

    shape = lon2d.shape
    cff_opt_mpa = np.zeros(shape)
    cff1_mpa = np.zeros(shape); cff2_mpa = np.zeros(shape)
    strike1 = np.zeros(shape); dip1 = np.zeros(shape); rake1 = np.zeros(shape)
    strike2 = np.zeros(shape); dip2 = np.zeros(shape); rake2 = np.zeros(shape)

    n_lat, n_lon = shape
    for i in range(n_lat):
        if progress_callback:
            progress_callback(70 + int(30 * i / max(n_lat, 1)))
        for j in range(n_lon):
            T_coseismic = np.array([
                [sxx[i, j], sxy[i, j], sxz[i, j]],
                [sxy[i, j], syy[i, j], syz[i, j]],
                [sxz[i, j], syz[i, j], szz[i, j]],
            ])
            T_point = T_regional_pa + T_coseismic
            sol = optimal_plane_solution(T_point, friction, T_coseismic_pa=T_coseismic)
            cff_opt_mpa[i, j] = sol["cff_opt_pa"] / 1e6
            p1, p2 = sol["planes"]
            strike1[i, j], dip1[i, j], rake1[i, j] = p1["strike"], p1["dip"], p1["rake"]
            strike2[i, j], dip2[i, j], rake2[i, j] = p2["strike"], p2["dip"], p2["rake"]
            cff1_mpa[i, j] = p1["cff_pa"] / 1e6
            cff2_mpa[i, j] = p2["cff_pa"] / 1e6

    if progress_callback:
        progress_callback(100)

    return (lon2d, lat2d, cff_opt_mpa, strike1, dip1, rake1,
           strike2, dip2, rake2, orthogonality_error_deg, cff1_mpa, cff2_mpa, True)


# ─── Cross-section path (2026-08-21 addition) ───────────────────────────
# Answers "what is the displayed CFF in the cross section -- receiver
# fault or optimal fault?" (it was always the fixed receiver fault; see
# okada_engine.compute_cross_section()) by adding the optimal-plane
# equivalent, reusing exactly the same per-point
# "raw tensor -> add regional stress -> optimal_plane_solution()"
# machinery compute_optimal_cff_grid_depth() already uses for the
# map-view case, applied along the cross-section's profile/depth points
# instead of a lon/lat grid.

def compute_cross_section_optimal(sources, regional: RegionalStress, elastic,
                                  lon1, lat1, lon2, lat2,
                                  dist_increment_km, max_depth_km, depth_increment_km,
                                  friction=None, progress_callback=None):
    """
    Optimal-plane counterpart of okada_engine.compute_cross_section() --
    same profile-line geometry and sampling (dist_km/depth_km output
    grid identical to that function's), but each grid point's ΔCFF is
    the OPTIMALLY-ORIENTED-PLANE value (regional + coseismic stress,
    eigendecomposed per point) rather than resolved onto one fixed
    receiver orientation.

    z=0 row uses the validated surface-strain formula directly (no
    DC3D needed), exactly mirroring compute_cross_section()'s own z=0
    special-case. Rows below the surface use the new
    "cross_section_stress_tensor" DC3D worker mode (raw tensor per
    point, no receiver resolution) -- the cross-section analogue of
    compute_optimal_cff_grid_depth()'s "stress_tensor" grid-mode call.

    Returns (dist_km, depth_km, cff_2d, used_dc3d) -- SAME shape/
    signature as compute_cross_section(), so callers (ui.main_dialog,
    core.cross_section_plot) don't need a second code path to consume
    it; just a different function to call based on the user's chosen
    CFF source.
    """
    if friction is None:
        friction = elastic.friction

    T_regional_pa, _orthogonality_error_deg = regional_stress_tensor_pa(regional)

    total_dist = float(np.hypot(*geo_to_km(lon2, lat2, lon1, lat1)))
    n_dist = max(2, int(round(total_dist / dist_increment_km)) + 1)
    dist_km = np.linspace(0.0, total_dist, n_dist)

    n_depth = max(2, int(round(max_depth_km / depth_increment_km)) + 1)
    depth_km = np.linspace(0.0, max_depth_km, n_depth)

    frac = dist_km / total_dist if total_dist > 0 else np.zeros_like(dist_km)
    profile_lons = lon1 + frac * (lon2 - lon1)
    profile_lats = lat1 + frac * (lat2 - lat1)

    cff_2d = np.full((n_depth, n_dist), np.nan)

    def _resolve_row(sxx, syy, szz, sxy, sxz, syz):
        row = np.empty(n_dist)
        for j in range(n_dist):
            T_coseismic = np.array([
                [sxx[j], sxy[j], sxz[j]],
                [sxy[j], syy[j], syz[j]],
                [sxz[j], syz[j], szz[j]],
            ])
            T_point = T_regional_pa + T_coseismic
            sol = optimal_plane_solution(T_point, friction, T_coseismic_pa=T_coseismic)
            row[j] = sol["cff_opt_pa"] / 1e6
        return row

    # z=0 row: surface-strain formula, no DC3D required (same
    # convention as compute_cross_section()'s own z=0 row).
    sxx0 = np.zeros(n_dist); syy0 = np.zeros(n_dist); szz0 = np.zeros(n_dist)
    sxy0 = np.zeros(n_dist); sxz0 = np.zeros(n_dist); syz0 = np.zeros(n_dist)
    for src in sources:
        e_km, n_km = geo_to_km(profile_lons, profile_lats, src.lon, src.lat)
        stress = _stress_from_surface_strain(e_km, n_km, src, elastic.mu, elastic.nu)
        sxx0 += stress['sxx']; syy0 += stress['syy']; szz0 += stress['szz']
        sxy0 += stress['sxy']; sxz0 += stress['sxz']; syz0 += stress['syz']
    cff_2d[0, :] = _resolve_row(sxx0, syy0, szz0, sxy0, sxz0, syz0)
    if progress_callback:
        progress_callback(10)

    used_dc3d = False
    if _has_okada_wrapper() and n_depth > 1:
        used_dc3d = True
        points = []
        for d_idx in range(1, n_depth):
            for x_idx in range(n_dist):
                points.append((float(profile_lons[x_idx]), float(profile_lats[x_idx]),
                              float(depth_km[d_idx])))
        if progress_callback:
            progress_callback(20)
        try:
            sxx, syy, szz, sxy, sxz, syz = _run_dc3d_worker_cross_section_stress_tensor(
                sources, elastic, points)
            for d_idx in range(1, n_depth):
                lo = (d_idx - 1) * n_dist
                hi = lo + n_dist
                cff_2d[d_idx, :] = _resolve_row(
                    sxx[lo:hi], syy[lo:hi], szz[lo:hi],
                    sxy[lo:hi], sxz[lo:hi], syz[lo:hi])
        except Exception:
            used_dc3d = False   # leave rows as NaN; caller can detect via used_dc3d
        if progress_callback:
            progress_callback(100)
    else:
        if progress_callback:
            progress_callback(100)

    return dist_km, depth_km, cff_2d, used_dc3d


def compute_cross_section_optimal_multi(sources, regional: RegionalStress, elastic,
                                        vertices, dist_increment_km, max_depth_km,
                                        depth_increment_km, friction=None,
                                        progress_callback=None):
    """
    Multi-segment counterpart of compute_cross_section_optimal(), same
    "call the single-leg function once per leg and stitch" strategy as
    okada_engine.compute_cross_section_multi() -- see that function's
    docstring for the concatenation convention (shared first vertex
    between adjacent legs is not duplicated). Returns the same 4-tuple
    compute_cross_section_optimal() returns, plus segment_info.
    """
    from .geo_profile import polyline_segment_info

    if len(vertices) < 2:
        raise ValueError("A profile polyline needs at least 2 vertices.")

    seg_info = polyline_segment_info(vertices)

    depth_km = None
    dist_chunks = []
    cff_chunks = []
    used_dc3d = True
    n_legs = len(vertices) - 1
    for leg_i in range(n_legs):
        lon_a, lat_a = vertices[leg_i]
        lon_b, lat_b = vertices[leg_i + 1]

        def _leg_progress(pct, leg_i=leg_i):
            if progress_callback:
                progress_callback(int((leg_i + pct / 100.0) / n_legs * 100))

        leg_dist, leg_depth, leg_cff, leg_used_dc3d = compute_cross_section_optimal(
            sources, regional, elastic, lon_a, lat_a, lon_b, lat_b,
            dist_increment_km, max_depth_km, depth_increment_km,
            friction=friction, progress_callback=_leg_progress,
        )
        if depth_km is None:
            depth_km = leg_depth
        used_dc3d = used_dc3d and leg_used_dc3d

        offset = seg_info["cumulative_dist_km"][leg_i]
        if leg_i == 0:
            dist_chunks.append(leg_dist + offset)
            cff_chunks.append(leg_cff)
        else:
            dist_chunks.append(leg_dist[1:] + offset)
            cff_chunks.append(leg_cff[:, 1:])

    dist_km = np.concatenate(dist_chunks)
    cff_2d = np.concatenate(cff_chunks, axis=1)

    if progress_callback:
        progress_callback(100)

    return dist_km, depth_km, cff_2d, used_dc3d, seg_info
