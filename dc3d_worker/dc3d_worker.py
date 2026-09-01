# -*- coding: utf-8 -*-
"""
Standalone DC3D worker script.

Runs under a SEPARATE Python interpreter that has okada_wrapper installed
(built with a real Fortran compiler + Python dev headers) — NOT under
QGIS's bundled Python, which typically lacks the dev headers needed to
build compiled extensions.

Usage:
    <external_python> dc3d_worker.py <input_json_path> <output_json_path>

Input JSON schema:
{
  "mode": "cff" | "displacement" | "cross_section" | "cross_section_stress_tensor"
          | "stress_tensor",                          # optional, default "cff"
  "mu": float, "nu": float,
  "z_recv_km": float,          # required for "cff"/"displacement" modes
  "sources": [
    {"lon":.., "lat":.., "depth":.., "length":.., "width":..,
     "strike":.., "dip":.., "rake":.., "slip":..},
    ...
  ],
  "receiver": {"strike":.., "dip":.., "rake":.., "friction":..},
      # required for mode="cff" and mode="cross_section"
  "grid": {"lon_min":.., "lon_max":.., "lat_min":.., "lat_max":..,
           "n_lon":.., "n_lat":..},
      # required for mode="cff"/"displacement"; a rectangular lon/lat grid
      # at the single shared depth z_recv_km
  "points": [{"lon":.., "lat":.., "z_km":..}, ...]
      # required for mode="cross_section"; an arbitrary list of points,
      # each with its OWN depth (e.g. a vertical profile line sampled at
      # many depths) — NOT a shared-depth rectangular grid
}

Output JSON schema (mode="cff"):
{
  "success": true,
  "cff_mpa": [[...], [...], ...],   # shape (n_lat, n_lon)
  "lon2d":   [[...], [...], ...],
  "lat2d":   [[...], [...], ...]
}

Output JSON schema (mode="displacement"):
{
  "success": true,
  "ux_m": [[...], ...], "uy_m": [[...], ...], "uz_m": [[...], ...],
  "lon2d": [[...], ...], "lat2d": [[...], ...]
}

Output JSON schema (mode="cross_section"):
{
  "success": true,
  "cff_mpa": [v0, v1, v2, ...],     # one value per input point, same order
  "shear_mpa": [v0, v1, v2, ...],   # Δτ per point (King et al. 1994 sign convention)
  "normal_mpa": [v0, v1, v2, ...]   # Δσn per point
}

Output JSON schema (mode="cross_section_stress_tensor"):
{
  "success": true,
  "sxx_pa": [...], "syy_pa": [...], "szz_pa": [...],
  "sxy_pa": [...], "sxz_pa": [...], "syz_pa": [...]   # one value per
                                                       # input point, Pa,
                                                       # tension-positive,
                                                       # (East,North,Down)
}

Input JSON schema (mode="slip_inversion") -- separate schema, does NOT
use "sources"/"receiver"/"grid" at all:
{
  "mode": "slip_inversion",
  "mu": float, "nu": float,
  "n_length": int, "n_width": int,     # subdivision shape of the parent
                                        # fault being inverted
  "patches": [ {"lon":.., "lat":.., "depth":.., "length":.., "width":..,
                "strike":.., "dip":..}, ... ],
      # length n_length*n_width, flat order i*n_length+j (i=down-dip,
      # j=along-strike) -- MUST match core.okada_engine.FaultParameters.
      # subdivide()'s own indexing exactly, since the returned "slip"
      # list is fed straight back into that same convention.
  "observations": [ {"lon":.., "lat":.., "e":float|null,
                     "n":float|null, "u":float|null,
                     "sigma_e":float|null, "sigma_n":float|null,
                     "sigma_u":float|null}, ... ],
      # component-wise (GNSS/leveling-style) surface (z=0) benchmarks;
      # any subset of e/n/u may be null/omitted per station (e.g.
      # leveling: u only). Optional per-component 1-sigma uncertainty
      # (same units as e/n/u, metres) weights that row as 1/sigma in
      # the solve; omitted/null sigma = unweighted (weight 1).
  "los_observations": [ {"lon":.., "lat":.., "los":float,
                         "look_e":float, "look_n":float, "look_u":float,
                         "sigma":float|null}, ... ],
      # InSAR-style: ONE scalar displacement per point, projected onto
      # a per-point unit look vector (ground-to-satellite, geographic
      # E/N/U -- caller is responsible for its sign convention, this
      # worker just takes the dot product as given). look_e/look_n/
      # look_u need not already be unit length; normalized here.
  "smoothing_factor": float,   # Laplacian roughness damping weight
  "max_slip": float,           # symmetric bound: unknowns in [-max_slip, +max_slip]
  "target_mw": float | null,   # optional hard total-moment equality constraint
  "fixed_rake_deg": float | list[float] | null
      # optional. None (default) = free inversion, 2 independent
      # unknowns/patch (rt_lateral, reverse), as above. If given, the
      # solve is reduced to ONE unknown per patch: a signed slip
      # magnitude s constrained to act along this rake direction
      # (Coulomb convention: rt_lateral=-s*cos(rake), reverse=s*sin(rake)).
      # A single float applies the SAME rake to every patch (the
      # intended use: a known rake from a focal mechanism, plate-motion
      # azimuth, or geologic slip vector for the whole fault). A list
      # of length n_length*n_width gives a per-patch rake instead (same
      # flat i*n_length+j order as "patches") -- supported by the
      # solver but not currently exposed in the dialog UI, which only
      # offers one uniform value. Halves the unknown count, which
      # directly improves conditioning for under-determined geometries
      # (vertical-only GNSS, single-track InSAR) at the cost of forcing
      # the rake the data is allowed to explain. Output "slip" is
      # STILL reported as [rt_lateral, reverse] pairs (see below) for
      # zero change to downstream consumers -- with fixed_rake_deg set,
      # every pair simply lies exactly on the requested rake by
      # construction (a useful sanity check: the report's atan2-derived
      # rake column will exactly reproduce the input value).
}

Output JSON schema (mode="slip_inversion"):
{
  "success": true,
  "solver_success": bool, "solver_message": str, "n_iter": int,
  "slip": [[rt_lateral_0, reverse_0], [rt_lateral_1, reverse_1], ...],
      # length n_length*n_width, same flat i*n_length+j order as "patches".
      # Always this [rt_lateral, reverse] pair shape regardless of
      # whether fixed_rake_deg was used (see input schema note above).
  "rms_misfit": float, "achieved_mw": float, "n_data": int,
  "predicted": [...], "observed": [...],   # same order, one per used
                                            # component/LOS row, weighted
  "component_labels": [[obs_idx, "e"|"n"|"u"|"los"], ...],
      # obs_idx indexes "observations" for e/n/u, "los_observations" for
      # "los" -- matches predicted/observed order
  "fixed_rake_deg": float | list[float] | null   # echoes the input, for
                                                  # report/diagnostic use
}

Input/output JSON schema (mode="slip_inversion_group") -- see
_run_slip_inversion_group()'s own docstring for the full schema; it is
the multi-fault-segment counterpart of "slip_inversion" above (jointly
inverts observations against the CONCATENATED patches of several fault
segments, e.g. a "Group" of source-fault rows with different strikes
tracing one bent fault), with each segment's own patches smoothed only
against its own neighbors (no cross-segment smoothing). Output schema
is the same as "slip_inversion" plus "segment_patch_counts" (list of
int, one per fault_segments entry, in order).

or on error:
{ "success": false, "error": "message" }
"""

import sys
import json
import math
import multiprocessing

EARTH_R = 6371.0  # km


def geo_to_km(lon, lat, lon0, lat0):
    lat0r = math.radians(lat0)
    x = (lon - lon0) * math.radians(1) * EARTH_R * math.cos(lat0r)
    y = (lat - lat0) * math.radians(1) * EARTH_R
    return x, y


def disp_from_dc3d_point(dc3dwrapper, e_km, n_km, src, mu, nu, z_recv_km):
    """
    Displacement (ue, un, uz; metres, geographic frame) at one observation
    point, at receiver depth z_recv_km, via Okada (1992) DC3D.
    """
    strike = math.radians(src["strike"])
    dip = math.radians(src["dip"])
    cs, ss = math.cos(strike), math.sin(strike)
    cd, sd = math.cos(dip), math.sin(dip)
    c_top = src["depth"] - sd * src["width"] / 2

    ec = e_km + cs * cd * src["width"] / 2
    nc = n_km - ss * cd * src["width"] / 2
    x_flt = cs * nc + ss * ec
    y_flt = ss * nc - cs * ec
    z_dc3d = -z_recv_km

    AL1, AL2 = -src["length"] / 2, src["length"] / 2
    # AW1/AW2 define the down-dip range of the fault relative to the
    # reference point at depth c_top (the TOP edge). From DC3D's own
    # internal geometry (D = c_top + Z; P = Y*cos(dip) + D*sin(dip);
    # ET = P - AW), a point on the fault plane at along-width parameter
    # eta has physical depth = c_top - eta*sin(dip). Since c_top is the
    # TOP (shallowest) edge, increasing eta moves SHALLOWER, not deeper.
    # The bottom edge (eta=-width) is therefore c_top + width*sin(dip),
    # i.e. the correct range is [-width, 0], NOT [0, width].
    # (Confirmed by direct comparison against Okada's DC3D.f source and
    # by continuity testing against the validated z=0 surface formula:
    # this fix reproduces it to 1e-5 relative precision.)
    AW1, AW2 = -src["width"], 0.0

    U1 = src["slip"] * math.cos(math.radians(src["rake"]))
    U2 = src["slip"] * math.sin(math.radians(src["rake"]))

    # alpha = (lambda + mu) / (lambda + 2*mu), computed from the ACTUAL
    # elastic parameters — this affects Okada's displacement formulas
    # directly (not just strain/derivative terms), so it must match the
    # same alpha used for stress, not a value hardcoded for nu=0.25.
    # (Fixed: earlier revision hardcoded alpha=2/3, silently ignoring the
    # user's actual Poisson's ratio for any nu != 0.25.)
    lam = 2 * mu * nu / (1 - 2 * nu)
    alpha = (lam + mu) / (lam + 2 * mu)

    ok, u, gu = dc3dwrapper(
        alpha, [x_flt, y_flt, z_dc3d], c_top, math.degrees(dip),
        [AL1, AL2], [AW1, AW2], [U1, U2, 0.0]
    )
    if ok != 0:
        return 0.0, 0.0, 0.0

    ux_flt, uy_flt, uz = u[0], u[1], u[2]
    # Rotate fault-frame (along-strike, across-strike) horizontal
    # displacement back to geographic (East, North)
    ue = ss * ux_flt - cs * uy_flt
    un = cs * ux_flt + ss * uy_flt
    return ue, un, uz


def stress_from_dc3d_point(dc3dwrapper, e_km, n_km, src, mu, nu, z_recv_km):
    """
    Full 3-D stress tensor (geographic frame, Pa) at one observation
    point, at receiver depth z_recv_km, via Okada (1992) DC3D.
    """
    strike = math.radians(src["strike"])
    dip = math.radians(src["dip"])
    cs, ss = math.cos(strike), math.sin(strike)
    cd, sd = math.cos(dip), math.sin(dip)
    c_top = src["depth"] - sd * src["width"] / 2

    ec = e_km + cs * cd * src["width"] / 2
    nc = n_km - ss * cd * src["width"] / 2
    x_flt = cs * nc + ss * ec
    y_flt = ss * nc - cs * ec
    z_dc3d = -z_recv_km

    AL1, AL2 = -src["length"] / 2, src["length"] / 2
    # See the identical fix (and derivation) in disp_from_dc3d_point()
    # above: the correct down-dip range relative to c_top is [-width, 0],
    # not [0, width].
    AW1, AW2 = -src["width"], 0.0

    U1 = src["slip"] * math.cos(math.radians(src["rake"]))
    U2 = src["slip"] * math.sin(math.radians(src["rake"]))

    lam = 2 * mu * nu / (1 - 2 * nu)
    alpha = (lam + mu) / (lam + 2 * mu)

    ok, u, gu = dc3dwrapper(
        alpha, [x_flt, y_flt, z_dc3d], c_top, math.degrees(dip),
        [AL1, AL2], [AW1, AW2], [U1, U2, 0.0]
    )
    if ok != 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # gu[i][j] = d(u_i)/d(x_j) with u_i in metres, x_j in km (since all
    # geometry above is in km): gradient units are [m/km] = 1e-3 *
    # dimensionless strain. Divide by 1000 to get true dimensionless strain.
    uxx, uxy, uxz = gu[0][0] / 1000., gu[0][1] / 1000., gu[0][2] / 1000.
    uyx, uyy, uyz = gu[1][0] / 1000., gu[1][1] / 1000., gu[1][2] / 1000.
    uzx, uzy, uzz = gu[2][0] / 1000., gu[2][1] / 1000., gu[2][2] / 1000.

    # DC3D's Z axis is upward-positive; our convention uses z downward
    # (depth) positive throughout the rest of this plugin. uz_dc3d (upward)
    # relates to our uz (downward) by uz_ours = -uz_dc3d, so:
    #   d(uz_ours)/d(z_ours) = d(-uz_dc3d)/d(-z_dc3d) = d(uz_dc3d)/d(z_dc3d) = uzz
    # i.e. ezz does NOT flip sign under this coordinate change (the two
    # negations cancel). exz/eyz involve ONE flipped axis (z) and one
    # not (x or y), so those DO flip sign relative to a naive same-frame
    # symmetric average.
    exx_f, eyy_f, ezz_f = uxx, uyy, uzz
    exy_f = (uxy + uyx) / 2
    exz_f = -(uxz + uzx) / 2
    eyz_f = -(uyz + uzy) / 2

    theta = exx_f + eyy_f + ezz_f
    sxx_f = lam * theta + 2 * mu * exx_f
    syy_f = lam * theta + 2 * mu * eyy_f
    szz_f = lam * theta + 2 * mu * ezz_f
    sxy_f = 2 * mu * exy_f
    sxz_f = 2 * mu * exz_f
    syz_f = 2 * mu * eyz_f

    # Rotate fault-frame stress back to geographic (x=E, y=N, z=down).
    #
    # Derived from the SAME (E,N)<->(x_flt,y_flt) rotation used to build
    # x_flt/y_flt above (x_flt=cs*nc+ss*ec, y_flt=ss*nc-cs*ec), i.e.
    # [x_flt;y_flt] = M[E;N] with M=[[ss,cs],[-cs,ss]] (a proper rotation,
    # det=1). For a rank-2 tensor, T_geo = M^T T_fault M, which gives:
    #   sxx_g = ss^2*Txx - s2*Txy + cs^2*Tyy
    #   syy_g = cs^2*Txx + s2*Txy + ss^2*Tyy
    #   sxy_g = -(cs^2-ss^2)*Txy + (Txx-Tyy)*ss*cs
    #   sxz_g = ss*Txz - cs*Tyz
    #   syz_g = cs*Txz + ss*Tyz
    # This independently matches okada85_surface_strain()'s own validated
    # rotation (its "uee"/"unn" formulas use -s2*(uxy+uyx)/2 and
    # +s2*(uxy+uyx)/2 respectively for the identical cross term). The
    # previous version of this function had the opposite sign on every
    # term involving the fault-frame shear components sxy_f/syz_f, which
    # is silent at strike=0/90/180/270 (where s2=0 and cs^2-ss^2=+-1
    # happens to only affect sxy_g) but wrong in general, and wrong for
    # sxy_g at EVERY strike including 0. Confirmed by continuity testing
    # against the validated surface formula (all 6 stress components
    # matched to <0.4% after this fix, vs. sign-flipped shear before) and
    # by an exact match (r=1.0000, sign agreement 100%) against Coulomb
    # 3.4.2's own dcff.cou reference output for test2.inp.
    s2 = math.sin(2 * strike)
    sxx_g = sxx_f * ss**2 - s2 * sxy_f + syy_f * cs**2
    syy_g = sxx_f * cs**2 + s2 * sxy_f + syy_f * ss**2
    sxy_g = -sxy_f * (cs**2 - ss**2) + (sxx_f - syy_f) * ss * cs
    sxz_g = sxz_f * ss - syz_f * cs
    syz_g = sxz_f * cs + syz_f * ss
    szz_g = szz_f

    return sxx_g, syy_g, szz_g, sxy_g, sxz_g, syz_g


def _greens_unit_matrices(dc3dwrapper, np, patches, points, mu, nu):
    """
    Unit-slip Okada displacement Green's functions for a list of surface
    (z=0) points against a list of patches, split into the SAME two
    "channels" the rest of this module and core.okada_engine's
    slip_overrides use: rt-lateral (rake=180, slip=1) and reverse
    (rake=90, slip=1). Returns 6 (n_points x n_patches) numpy arrays:
    (Ge_rt, Gn_rt, Gu_rt, Ge_rev, Gn_rev, Gu_rev).
    """
    n_p, n_pt = len(patches), len(points)
    Ge_rt = np.zeros((n_pt, n_p)); Gn_rt = np.zeros((n_pt, n_p)); Gu_rt = np.zeros((n_pt, n_p))
    Ge_rev = np.zeros((n_pt, n_p)); Gn_rev = np.zeros((n_pt, n_p)); Gu_rev = np.zeros((n_pt, n_p))
    for p_idx, patch in enumerate(patches):
        src_rt = dict(patch); src_rt["rake"] = 180.0; src_rt["slip"] = 1.0
        src_rev = dict(patch); src_rev["rake"] = 90.0; src_rev["slip"] = 1.0
        for o_idx, pt in enumerate(points):
            e_km, n_km = geo_to_km(pt["lon"], pt["lat"], patch["lon"], patch["lat"])
            ue, un, uz = disp_from_dc3d_point(dc3dwrapper, e_km, n_km, src_rt, mu, nu, 0.0)
            Ge_rt[o_idx, p_idx] = ue; Gn_rt[o_idx, p_idx] = un; Gu_rt[o_idx, p_idx] = uz
            ue, un, uz = disp_from_dc3d_point(dc3dwrapper, e_km, n_km, src_rev, mu, nu, 0.0)
            Ge_rev[o_idx, p_idx] = ue; Gn_rev[o_idx, p_idx] = un; Gu_rev[o_idx, p_idx] = uz
    return Ge_rt, Gn_rt, Gu_rt, Ge_rev, Gn_rev, Gu_rev


def _greens_chunk_worker(args):
    """
    Top-level (module-level, picklable) entry point for
    _greens_unit_matrices_mp()'s multiprocessing.Pool. Must stay a
    plain function -- not a lambda, closure, or bound method -- because
    Windows' "spawn" multiprocessing start method (the default there,
    and the platform this plugin targets -- see PROJECT_HANDOVER.md)
    pickles the target callable by reference and re-imports this module
    fresh in each child process; anything not reachable as a plain
    module-level name would fail to pickle.

    Re-imports okada_wrapper itself inside the child process rather
    than receiving the parent's already-imported dc3dwrapper function
    object -- f2py-wrapped Fortran callables are not reliably picklable
    across the process boundary, so this is the only universally
    correct approach (at the cost of the Fortran shared library being
    loaded again per worker process, a small one-time cost paid once
    per Pool, not per point).
    """
    patches, points_chunk, mu, nu = args
    from okada_wrapper import dc3dwrapper
    import numpy as np
    return _greens_unit_matrices(dc3dwrapper, np, patches, points_chunk, mu, nu)


def _greens_unit_matrices_mp(dc3dwrapper, np, patches, points, mu, nu, n_workers=None):
    """
    Parallel dispatcher for _greens_unit_matrices(). Every observation
    point's Green's-function row is independent of every other point
    (and computed against the SAME full patch list regardless), so
    splitting `points` into chunks across worker PROCESSES and
    concatenating the per-chunk results back in original order is an
    EXACT parallelization -- identical floating-point result to the
    serial path, just faster wall-clock. Processes (not threads) are
    used because each dc3dwrapper() call is a compiled Fortran call
    that does not release the GIL in a way threads could exploit here.

    This is the main lever for large slip-inversion jobs: cost scales
    as O(n_patches x n_points), and a dense point import (tens of
    thousands of scattered observations, e.g. GNSS/InSAR sampled onto a
    fine point grid) combined with a finely-subdivided fault (hundreds
    of patches) can otherwise take far longer than a typical subprocess
    timeout allows -- see PROJECT_HANDOVER_ADDENDUM for the specific
    19x15-patch / 100k-point case that motivated this.

    Falls back to the plain serial _greens_unit_matrices() untouched
    when the job is too small for process-spawn overhead (non-trivial
    on Windows, roughly hundreds of ms per process) to pay for itself.
    """
    n_pt = len(points)
    if n_workers is None:
        try:
            n_workers = multiprocessing.cpu_count()
        except NotImplementedError:
            n_workers = 1
    # Below ~2000 points, per-process spawn overhead likely exceeds any
    # savings; also guards n_pt == 0 and single-core machines.
    n_workers = max(1, min(n_workers, max(1, n_pt // 2000)))
    if n_workers <= 1 or n_pt < 2000:
        return _greens_unit_matrices(dc3dwrapper, np, patches, points, mu, nu)

    chunk_size = math.ceil(n_pt / n_workers)
    chunks = [points[i:i + chunk_size] for i in range(0, n_pt, chunk_size)]
    args = [(patches, chunk, mu, nu) for chunk in chunks]

    with multiprocessing.Pool(processes=len(chunks)) as pool:
        chunk_results = pool.map(_greens_chunk_worker, args)

    Ge_rt = np.vstack([r[0] for r in chunk_results])
    Gn_rt = np.vstack([r[1] for r in chunk_results])
    Gu_rt = np.vstack([r[2] for r in chunk_results])
    Ge_rev = np.vstack([r[3] for r in chunk_results])
    Gn_rev = np.vstack([r[4] for r in chunk_results])
    Gu_rev = np.vstack([r[5] for r in chunk_results])
    return Ge_rt, Gn_rt, Gu_rt, Ge_rev, Gn_rev, Gu_rev


def _run_slip_inversion(dc3dwrapper, job):
    """
    Build a 2-component-per-patch (rt-lateral, reverse) Okada Green's
    matrix against arbitrary scattered surface observations -- both
    component-wise (GNSS/leveling-style "observations") and
    LOS-projected (InSAR-style "los_observations") -- then solve a
    Laplacian-smoothed, bounded slip inversion, optionally with a hard
    total-moment (Mw) equality constraint. See module docstring
    "slip_inversion" schema for input/output.

    Requires numpy + scipy in THIS (external) Python environment, in
    addition to okada_wrapper. Raises on any problem; the caller (main())
    wraps this in the same try/except that already turns any exception
    into a {"success": false, "error": ...} payload.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            f"numpy not importable in this Python: {e}. This should "
            f"already be present as an okada_wrapper dependency.")

    mu = job["mu"]
    nu = job["nu"]
    patches = job["patches"]
    n_length = int(job["n_length"])
    n_width = int(job["n_width"])
    observations = job.get("observations", []) or []
    los_observations = job.get("los_observations", []) or []
    smoothing_factor = float(job.get("smoothing_factor", 0.05))
    max_slip = float(job.get("max_slip", 10.0))
    target_mw = job.get("target_mw", None)

    n_p = len(patches)
    if n_p != n_length * n_width:
        raise ValueError(
            f"patches length ({n_p}) != n_length*n_width "
            f"({n_length}*{n_width}={n_length * n_width})")
    if n_p == 0:
        raise ValueError("No patches given.")
    if not observations and not los_observations:
        raise ValueError("No observations or los_observations given -- nothing to invert.")

    fixed_rake_deg = job.get("fixed_rake_deg", None)
    L_base = _laplacian_base(np, [(n_length, n_width)])

    return _solve_slip_inversion(
        dc3dwrapper, np, patches, L_base, observations, los_observations,
        mu, nu, smoothing_factor, max_slip, target_mw, fixed_rake_deg)


def _run_slip_inversion_group(dc3dwrapper, job):
    """
    Multi-fault-segment counterpart to _run_slip_inversion(): jointly
    inverts the SAME kind of scattered surface observations for
    independent per-patch (rt-lateral, reverse) slip across the
    concatenated patches of SEVERAL fault segments at once (e.g. a
    "Group" of source-fault rows with different strikes tracing one
    bent/kinked fault) -- one combined linear system, one combined
    total-moment constraint if target_mw is given, but NO cross-segment
    Laplacian smoothing (each segment's patches are only smoothed
    against their OWN along-strike/down-dip neighbors -- there is no
    general, correct notion of "adjacent patch" across a strike bend
    without extra geometric info this schema doesn't carry, so this
    deliberately smooths each segment independently rather than
    guessing an adjacency across the group).

    Separate schema from "slip_inversion" -- does NOT use "patches"/
    "n_length"/"n_width" directly:
    {
      "mode": "slip_inversion_group",
      "mu": float, "nu": float,
      "fault_segments": [
        {"n_length": int, "n_width": int,
         "patches": [ {...}, ... ]},   # same per-patch dict shape as
                                        # "slip_inversion", length
                                        # n_length*n_width, flat
                                        # i*n_length+j order
        ...
      ],
      "observations": [...], "los_observations": [...],
      "smoothing_factor": float, "max_slip": float, "target_mw": float|null,
      "fixed_rake_deg": float|null
          # same meaning as "slip_inversion"'s fixed_rake_deg, applied
          # UNIFORMLY across every patch of every segment if given --
          # a per-segment-varying rake is not supported by this schema
          # (would need a list keyed to the concatenated patch order,
          # which the caller can already build and pass as a flat list
          # the same length as the total patch count if ever needed;
          # not exposed in the dialog UI, which offers one shared value
          # for the whole group).
    }

    Output: same shape as "slip_inversion", PLUS "segment_patch_counts"
    (list of int, one per fault_segments entry) so the caller can split
    the flat "slip" list back into each segment's own
    n_length*n_width-length block, in the SAME order as fault_segments.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            f"numpy not importable in this Python: {e}. This should "
            f"already be present as an okada_wrapper dependency.")

    mu = job["mu"]
    nu = job["nu"]
    fault_segments = job.get("fault_segments") or []
    observations = job.get("observations", []) or []
    los_observations = job.get("los_observations", []) or []
    smoothing_factor = float(job.get("smoothing_factor", 0.05))
    max_slip = float(job.get("max_slip", 10.0))
    target_mw = job.get("target_mw", None)

    if not fault_segments:
        raise ValueError("No fault_segments given.")
    if not observations and not los_observations:
        raise ValueError("No observations or los_observations given -- nothing to invert.")

    fixed_rake_deg = job.get("fixed_rake_deg", None)

    patches = []
    shapes = []
    segment_patch_counts = []
    for seg_idx, seg in enumerate(fault_segments):
        n_length = int(seg["n_length"])
        n_width = int(seg["n_width"])
        seg_patches = seg["patches"]
        if len(seg_patches) != n_length * n_width:
            raise ValueError(
                f"fault_segments[{seg_idx}]: patches length "
                f"({len(seg_patches)}) != n_length*n_width "
                f"({n_length}*{n_width}={n_length * n_width})")
        patches.extend(seg_patches)
        shapes.append((n_length, n_width))
        segment_patch_counts.append(n_length * n_width)

    if not patches:
        raise ValueError("No patches across any fault_segments.")

    L_base = _laplacian_base(np, shapes)

    result = _solve_slip_inversion(
        dc3dwrapper, np, patches, L_base, observations, los_observations,
        mu, nu, smoothing_factor, max_slip, target_mw, fixed_rake_deg)
    result["segment_patch_counts"] = segment_patch_counts
    return result


def _laplacian_base(np, shapes):
    """
    Laplacian roughness matrix (n_p_total columns, UNDOUBLED -- one row
    per along-strike/down-dip neighbor pair, one column per patch)
    across one or more (n_length, n_width) fault segments, each
    smoothed only against its OWN neighbors (same indexing as
    core.okada_engine.FaultParameters.subdivide(): i=down-dip,
    j=along-strike, flat i*n_length+j) -- no cross-segment terms.

    This is the shared "roughness on a scalar field over the patch
    grid" building block for BOTH solve paths in
    _solve_slip_inversion(): the free (rt_lateral, reverse) path block-
    doubles this into a 2*n_p_total-column matrix (one independent copy
    per channel, since the two channels are smoothed independently),
    while the fixed_rake_deg path applies it directly to the single
    per-patch slip-magnitude unknown -- there is exactly one physically
    sensible scalar field to smooth in that case, not two.
    """
    n_p_total = sum(n_length * n_width for n_length, n_width in shapes)
    L_rows = []
    offset = 0
    for n_length, n_width in shapes:
        for i in range(n_width):
            for j in range(n_length):
                p = offset + i * n_length + j
                if j + 1 < n_length:             # along-strike neighbor
                    row = np.zeros(n_p_total); row[p] = 1.0; row[p + 1] = -1.0
                    L_rows.append(row)
                if i + 1 < n_width:               # down-dip neighbor
                    row = np.zeros(n_p_total); row[p] = 1.0; row[p + n_length] = -1.0
                    L_rows.append(row)
        offset += n_length * n_width
    L_base = np.array(L_rows) if L_rows else np.zeros((0, n_p_total))
    return L_base


def _solve_slip_inversion(dc3dwrapper, np, patches, L_base, observations,
                          los_observations, mu, nu, smoothing_factor,
                          max_slip, target_mw, fixed_rake_deg=None):
    """
    The shared solve core of _run_slip_inversion()/_run_slip_inversion_group():
    Green's-matrix assembly, bounded/Laplacian-damped (optionally
    moment-constrained) least squares, and diagnostics -- everything
    that does NOT depend on whether `patches` came from one fault's
    subdivision or several concatenated fault segments. `L_base` is
    already fully built (see _laplacian_base()) by the caller; it is
    UNDOUBLED (n_p columns), and this function decides how to expand or
    apply it depending on fixed_rake_deg.

    fixed_rake_deg : None (default) -- free 2-unknowns/patch inversion,
      as originally implemented. Or a float/list of length n_p -- the
      inversion is reduced to ONE unknown per patch (signed slip
      magnitude s along that fixed rake), via a linear reparameterization
      x_free = P @ s where P encodes rt_lateral=-s*cos(rake),
      reverse=s*sin(rake) per patch. This is applied by projecting the
      SAME full (n_data x 2*n_p) Green's matrix G through P rather than
      rebuilding the Green's functions -- G is linear in the unknowns,
      so G_reduced = G @ P is exact, not an approximation.
    """
    import math

    n_p = len(patches)

    # ---- 1. Unit Green's functions, computed once per point list. ----
    # rt-lateral unit source: (rt_lateral=1, reverse=0) -> in the SAME
    # Coulomb-convention (rake, slip) mapping used everywhere else in
    # this plugin (U1=-rt_lat, U2=reverse -> slip=hypot, rake=atan2):
    # rake=180 deg, slip=1. Reverse unit source: rake=90 deg, slip=1.
    # These two columns per patch are exactly the two "channels" that
    # core.okada_engine.FaultParameters.subdivide()'s slip_overrides
    # already expects, so the solved (rt, reverse) pairs below need no
    # further conversion before being stored as overrides.
    if observations:
        Ge_rt, Gn_rt, Gu_rt, Ge_rev, Gn_rev, Gu_rev = _greens_unit_matrices_mp(
            dc3dwrapper, np, patches, observations, mu, nu)
    if los_observations:
        Ge_rt_l, Gn_rt_l, Gu_rt_l, Ge_rev_l, Gn_rev_l, Gu_rev_l = _greens_unit_matrices_mp(
            dc3dwrapper, np, patches, los_observations, mu, nu)

    # ---- 2. Assemble data vector + design matrix. ----
    #         Component rows (e/n/u) use one Green's column directly;
    #         LOS rows dot the per-point unit look vector against all
    #         three component columns first, giving one row per point
    #         (this is the only physics difference between the two
    #         observation types -- both feed the same 2*n_p unknowns).
    #         Optional per-row 1-sigma weights (1/sigma) let GNSS and
    #         InSAR data with very different noise levels be combined
    #         in one joint solve without one dataset silently dominating.
    d_rows, G_rows, comp_labels = [], [], []
    if observations:
        comp_map = {"e": (Ge_rt, Ge_rev), "n": (Gn_rt, Gn_rev), "u": (Gu_rt, Gu_rev)}
        for o_idx, obs in enumerate(observations):
            for comp in ("e", "n", "u"):
                val = obs.get(comp, None)
                if val is None:
                    continue
                sigma = obs.get(f"sigma_{comp}", None)
                weight = 1.0 / float(sigma) if sigma else 1.0
                Grt, Grev = comp_map[comp]
                G_rows.append(weight * np.concatenate([Grt[o_idx, :], Grev[o_idx, :]]))
                d_rows.append(weight * float(val))
                comp_labels.append((o_idx, comp))

    for o_idx, obs in enumerate(los_observations):
        look = np.array([obs["look_e"], obs["look_n"], obs["look_u"]], dtype=float)
        norm = float(np.linalg.norm(look))
        if norm <= 0:
            raise ValueError(f"los_observations[{o_idx}] has a zero-length look vector.")
        look = look / norm
        row_rt = look[0] * Ge_rt_l[o_idx, :] + look[1] * Gn_rt_l[o_idx, :] + look[2] * Gu_rt_l[o_idx, :]
        row_rev = look[0] * Ge_rev_l[o_idx, :] + look[1] * Gn_rev_l[o_idx, :] + look[2] * Gu_rev_l[o_idx, :]
        sigma = obs.get("sigma", None)
        weight = 1.0 / float(sigma) if sigma else 1.0
        G_rows.append(weight * np.concatenate([row_rt, row_rev]))
        d_rows.append(weight * float(obs["los"]))
        comp_labels.append((o_idx, "los"))

    if not d_rows:
        raise ValueError("No observation components (e/n/u/los) were provided -- nothing to invert.")

    G = np.array(G_rows)   # (n_data, 2*n_p)
    d = np.array(d_rows)

    # ---- 3. Patch areas (m^2), needed only for the moment constraint ----
    areas_m2 = np.array([p["length"] * 1000.0 * p["width"] * 1000.0 for p in patches])

    # ---- 3b. Reduce to the fixed-rake parameterization, if requested. ----
    #          x_free (length 2*n_p) = P @ s (length n_p), where
    #          rt_lateral_p = -cos(rake_p)*s_p, reverse_p = sin(rake_p)*s_p
    #          -- the same (rake, slip) -> (rt, reverse) mapping used
    #          everywhere else in this plugin, just applied in reverse.
    #          G is linear in x_free, so G @ P is an EXACT reduction of
    #          the design matrix, not a re-derivation -- no new Green's
    #          function evaluations needed. The roughness term uses
    #          L_base directly (n_p columns): smoothing the single
    #          scalar slip field, not two now-dependent channels.
    if fixed_rake_deg is not None:
        rake_arr = np.atleast_1d(np.asarray(fixed_rake_deg, dtype=float))
        if rake_arr.size == 1:
            rake_arr = np.full(n_p, float(rake_arr[0]))
        if rake_arr.size != n_p:
            raise ValueError(
                f"fixed_rake_deg length ({rake_arr.size}) != number of "
                f"patches ({n_p}).")
        rake_rad = np.radians(rake_arr)
        coef_rt = -np.cos(rake_rad)
        coef_rev = np.sin(rake_rad)
        P = np.zeros((2 * n_p, n_p))
        idx = np.arange(n_p)
        P[idx, idx] = coef_rt
        P[n_p + idx, idx] = coef_rev

        G_solve = G @ P                    # (n_data, n_p)
        L_solve = L_base                   # (n_roughness_rows, n_p)
        n_unknowns = n_p
        bounds_lo = -max_slip * np.ones(n_p)
        bounds_hi = max_slip * np.ones(n_p)
    else:
        zeros_block = np.zeros_like(L_base)
        L_solve = (np.block([[L_base, zeros_block], [zeros_block, L_base]])
                  if L_base.shape[0] > 0 else np.zeros((0, 2 * n_p)))
        G_solve = G
        n_unknowns = 2 * n_p
        bounds_lo = -max_slip * np.ones(2 * n_p)
        bounds_hi = max_slip * np.ones(2 * n_p)

    if target_mw is None:
        # ---- Plain bounded, Laplacian-damped least squares (the ----
        #      standard/default case) -- solved as one augmented
        #      linear least-squares system, bounds enforced directly.
        from scipy.optimize import lsq_linear
        if L_solve.shape[0] > 0:
            A = np.vstack([G_solve, smoothing_factor * L_solve])
            b = np.concatenate([d, np.zeros(L_solve.shape[0])])
        else:
            A, b = G_solve, d
        result = lsq_linear(A, b, bounds=(bounds_lo, bounds_hi), method="trf", max_iter=2000)
        x = result.x
        n_iter = int(result.nit) if result.nit is not None else 0
        solver_message, success = str(result.message), bool(result.success)
    else:
        # ---- Moment-constrained inversion. Unlike the original ----
        #      fixed-rake-ratio script (where slip WAS the single
        #      unknown, making the moment constraint linear), the total
        #      moment mu*sum(area_p * hypot(rt_p, rev_p)) is NONLINEAR
        #      when rt/reverse are independent per-patch unknowns, so
        #      this uses a NonlinearConstraint with an analytic
        #      Jacobian in that case. When fixed_rake_deg IS given, the
        #      unknown is already a signed scalar slip s_p and
        #      hypot(rt_p, rev_p) == |s_p| exactly (since coef_rt^2 +
        #      coef_rev^2 == 1 by construction), so the moment reduces
        #      to mu*sum(area_p*|s_p|) -- still nonlinear (abs, not
        #      linear), so this still uses the trust-constr path rather
        #      than reintroducing the original script's linear
        #      constraint, but with n_p rather than 2*n_p unknowns.
        #      For a multi-segment group, this is the TOTAL moment
        #      summed over every patch of every segment -- the physically
        #      correct target for one Mw applied to the whole rupture.
        from scipy.optimize import minimize, Bounds, NonlinearConstraint

        M0_target = 10.0 ** (1.5 * float(target_mw) + 9.1)
        eps = 1e-6  # keeps the magnitude's gradient finite at exactly zero slip

        if fixed_rake_deg is not None:
            def moment_of(x):
                mag = np.sqrt(x**2 + eps**2)
                return mu * float(np.sum(areas_m2 * mag))

            def moment_jac(x):
                mag = np.sqrt(x**2 + eps**2)
                jac = mu * areas_m2 * x / mag
                return jac.reshape(1, -1)

            max_possible = mu * float(np.sum(areas_m2 * max_slip))
        else:
            def moment_of(x):
                rt, rev = x[:n_p], x[n_p:]
                mag = np.sqrt(rt**2 + rev**2 + eps**2)
                return mu * float(np.sum(areas_m2 * mag))

            def moment_jac(x):
                rt, rev = x[:n_p], x[n_p:]
                mag = np.sqrt(rt**2 + rev**2 + eps**2)
                jac = np.zeros(2 * n_p)
                jac[:n_p] = mu * areas_m2 * rt / mag
                jac[n_p:] = mu * areas_m2 * rev / mag
                return jac.reshape(1, -1)

            max_possible = mu * float(np.sum(areas_m2 * math.sqrt(2.0) * max_slip))

        max_possible_mw = (2.0 / 3.0) * (math.log10(max_possible) - 9.1)
        if M0_target > max_possible:
            raise ValueError(
                f"Target Mw {target_mw:.2f} is not feasible with "
                f"max_slip={max_slip:.2f} m -- maximum achievable is "
                f"Mw {max_possible_mw:.2f}.")

        def objective(x):
            residual = G_solve @ x - d
            roughness = L_solve @ x if L_solve.shape[0] > 0 else np.zeros(0)
            return float(residual @ residual + (smoothing_factor**2) * (roughness @ roughness))

        GTG, GTd = G_solve.T @ G_solve, G_solve.T @ d
        LTL = (L_solve.T @ L_solve) if L_solve.shape[0] > 0 else np.zeros((n_unknowns, n_unknowns))
        lam2 = smoothing_factor**2
        H_const = 2.0 * (GTG + lam2 * LTL)

        def gradient(x):
            return 2.0 * (GTG @ x - GTd) + 2.0 * lam2 * (LTL @ x)

        def hessian(x):
            return H_const

        moment_constraint = NonlinearConstraint(moment_of, lb=M0_target, ub=M0_target, jac=moment_jac)

        mean_slip = M0_target / max(mu * float(np.sum(areas_m2)), 1e-30)
        mean_slip = float(np.clip(mean_slip, 0.01, max_slip))
        x0 = np.zeros(n_unknowns)
        if fixed_rake_deg is not None:
            x0[:] = mean_slip
        else:
            x0[n_p:] = mean_slip  # start on the "reverse" channel; optimizer is free to move it

        result = minimize(
            objective, x0, jac=gradient, hess=hessian, method="trust-constr",
            constraints=[moment_constraint], bounds=Bounds(bounds_lo, bounds_hi),
            options={"maxiter": 2000, "verbose": 0})
        x = result.x
        n_iter = int(result.nit) if getattr(result, "nit", None) is not None else 0
        solver_message, success = str(result.message), bool(result.success)

    # ---- 4. Diagnostics ----
    predicted = G_solve @ x
    residuals = predicted - d
    rms = float(np.sqrt(np.mean(residuals**2))) if len(residuals) else 0.0

    # Always express the result as (rt_lateral, reverse) pairs -- for
    # the free path these are x's two halves directly; for the
    # fixed-rake path they're recovered from the scalar slip via the
    # same P mapping used to build G_solve (exact, not re-derived).
    if fixed_rake_deg is not None:
        rt_arr = coef_rt * x
        rev_arr = coef_rev * x
    else:
        rt_arr = x[:n_p]
        rev_arr = x[n_p:]

    final_moment = mu * float(np.sum(areas_m2 * np.sqrt(rt_arr**2 + rev_arr**2)))
    final_mw = (2.0 / 3.0) * (math.log10(final_moment) - 9.1) if final_moment > 0 else float("-inf")

    return {
        "success": True,
        "solver_success": success,
        "solver_message": solver_message,
        "n_iter": n_iter,
        "slip": [[float(rt_arr[p]), float(rev_arr[p])] for p in range(n_p)],
        "rms_misfit": rms,
        "achieved_mw": final_mw,
        "n_data": int(len(d)),
        "predicted": [float(v) for v in predicted],
        "observed": [float(v) for v in d],
        "component_labels": [[int(o), c] for o, c in comp_labels],
        # Echoed straight from the input job (already a JSON-serializable
        # float, list of floats, or None -- no reprocessing needed).
        "fixed_rake_deg": fixed_rake_deg,
    }


def main():
    if len(sys.argv) != 3:
        print(json.dumps({"success": False,
                          "error": "Usage: dc3d_worker.py <input.json> <output.json>"}))
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        with open(in_path, "r") as f:
            job = json.load(f)
    except Exception as e:
        with open(out_path, "w") as f:
            json.dump({"success": False, "error": f"Failed to read input: {e}"}, f)
        sys.exit(1)

    try:
        from okada_wrapper import dc3dwrapper
    except ImportError as e:
        with open(out_path, "w") as f:
            json.dump({"success": False,
                      "error": f"okada_wrapper not importable in this Python: {e}"}, f)
        sys.exit(1)

    try:
        mode = job.get("mode", "cff")

        if mode == "slip_inversion":
            # Separate schema entirely ("patches"/"observations", not
            # "sources"/"receiver"/"grid") -- dispatched before the
            # job["sources"] lookup below, which this mode doesn't have.
            result = _run_slip_inversion(dc3dwrapper, job)
            with open(out_path, "w") as f:
                json.dump(result, f)
            return

        if mode == "slip_inversion_group":
            # Multi-fault-segment counterpart (see _run_slip_inversion_group()
            # docstring for schema) -- also dispatched before job["sources"].
            result = _run_slip_inversion_group(dc3dwrapper, job)
            with open(out_path, "w") as f:
                json.dump(result, f)
            return

        sources = job["sources"]

        if mode == "cross_section_stress_tensor":
            # 2026-08-21 addition: the "optimal" counterpart to
            # "cross_section". Same points-with-own-depth schema as
            # "cross_section", but returns the raw 6-component stress
            # tensor per point (no receiver resolution) -- exactly the
            # same "no receiver, tensor only" idea "stress_tensor" mode
            # already applies to a rectangular lon/lat grid, just over
            # arbitrary (lon, lat, z_km) points instead. Lets the caller
            # (core.optimal_plane.compute_cross_section_optimal) add the
            # regional stress and eigendecompose per point itself,
            # mirroring how compute_optimal_cff_grid_depth already uses
            # "stress_tensor" mode for the map-view optimal-plane path.
            mu = job["mu"]
            nu = job["nu"]
            points = job["points"]

            sxx_out = [0.0] * len(points)
            syy_out = [0.0] * len(points)
            szz_out = [0.0] * len(points)
            sxy_out = [0.0] * len(points)
            sxz_out = [0.0] * len(points)
            syz_out = [0.0] * len(points)
            for src in sources:
                for k, pt in enumerate(points):
                    e_km, n_km = geo_to_km(pt["lon"], pt["lat"], src["lon"], src["lat"])
                    sxx, syy, szz, sxy, sxz, syz = stress_from_dc3d_point(
                        dc3dwrapper, e_km, n_km, src, mu, nu, pt["z_km"])
                    sxx_out[k] += sxx; syy_out[k] += syy; szz_out[k] += szz
                    sxy_out[k] += sxy; sxz_out[k] += sxz; syz_out[k] += syz

            with open(out_path, "w") as f:
                json.dump({"success": True,
                          "sxx_pa": sxx_out, "syy_pa": syy_out, "szz_pa": szz_out,
                          "sxy_pa": sxy_out, "sxz_pa": sxz_out, "syz_pa": syz_out}, f)
            return

        if mode == "points_full":
            # 2026-09-01 addition (core.point_calculation / "point
            # calculator" feature): same points-with-own-depth schema as
            # "cross_section_stress_tensor" above, but returns BOTH the
            # raw 6-component stress tensor (Pa) AND the displacement
            # vector (m) per point in a single subprocess round-trip.
            # point_calculation.compute_point_results() always needs
            # both quantities at the exact same points (to report
            # predicted stress/CFF and predicted displacement side by
            # side, e.g. for comparison against a field-measured slip/
            # displacement observation) -- combining them here avoids
            # launching the external Python twice for what is otherwise
            # the identical source/point geometry loop, just calling
            # both stress_from_dc3d_point() and disp_from_dc3d_point()
            # once each instead of running two separate worker jobs.
            mu = job["mu"]
            nu = job["nu"]
            points = job["points"]

            sxx_out = [0.0] * len(points)
            syy_out = [0.0] * len(points)
            szz_out = [0.0] * len(points)
            sxy_out = [0.0] * len(points)
            sxz_out = [0.0] * len(points)
            syz_out = [0.0] * len(points)
            ue_out = [0.0] * len(points)
            un_out = [0.0] * len(points)
            uz_out = [0.0] * len(points)
            for src in sources:
                for k, pt in enumerate(points):
                    e_km, n_km = geo_to_km(pt["lon"], pt["lat"], src["lon"], src["lat"])
                    sxx, syy, szz, sxy, sxz, syz = stress_from_dc3d_point(
                        dc3dwrapper, e_km, n_km, src, mu, nu, pt["z_km"])
                    sxx_out[k] += sxx; syy_out[k] += syy; szz_out[k] += szz
                    sxy_out[k] += sxy; sxz_out[k] += sxz; syz_out[k] += syz
                    ue, un, uz = disp_from_dc3d_point(
                        dc3dwrapper, e_km, n_km, src, mu, nu, pt["z_km"])
                    ue_out[k] += ue; un_out[k] += un; uz_out[k] += uz

            with open(out_path, "w") as f:
                json.dump({"success": True,
                          "sxx_pa": sxx_out, "syy_pa": syy_out, "szz_pa": szz_out,
                          "sxy_pa": sxy_out, "sxz_pa": sxz_out, "syz_pa": syz_out,
                          "ue_m": ue_out, "un_m": un_out, "uz_m": uz_out}, f)
            return

        if mode == "cross_section":
            # Each point has its own (lon, lat, z_km) — a vertical profile,
            # not a rectangular lon/lat grid at one shared depth.
            mu = job["mu"]
            nu = job["nu"]
            receiver = job["receiver"]
            points = job["points"]   # list of {"lon":.., "lat":.., "z_km":..}

            sr = math.radians(receiver["strike"])
            dr = math.radians(receiver["dip"])
            rr = math.radians(receiver["rake"])
            cs, ss = math.cos(sr), math.sin(sr)
            cd, sd = math.cos(dr), math.sin(dr)
            cr, sr_ = math.cos(rr), math.sin(rr)
            nx, ny, nz = cs * sd, -ss * sd, -cd
            lx = cr * ss - sr_ * cs * cd
            ly = cr * cs + sr_ * ss * cd
            lz = -sr_ * sd
            friction = receiver["friction"]

            cff_out = [0.0] * len(points)
            shear_out = [0.0] * len(points)
            normal_out = [0.0] * len(points)
            for src in sources:
                for k, pt in enumerate(points):
                    e_km, n_km = geo_to_km(pt["lon"], pt["lat"], src["lon"], src["lat"])
                    sxx, syy, szz, sxy, sxz, syz = stress_from_dc3d_point(
                        dc3dwrapper, e_km, n_km, src, mu, nu, pt["z_km"])
                    tx = sxx * nx + sxy * ny + sxz * nz
                    ty = sxy * nx + syy * ny + syz * nz
                    tz = sxz * nx + syz * ny + szz * nz
                    dtau = tx * lx + ty * ly + tz * lz
                    dsn = tx * nx + ty * ny + tz * nz
                    cff_out[k] += (dtau + friction * dsn) / 1e6      # Pa -> MPa
                    shear_out[k] += dtau / 1e6
                    normal_out[k] += dsn / 1e6

            with open(out_path, "w") as f:
                json.dump({"success": True, "cff_mpa": cff_out,
                          "shear_mpa": shear_out, "normal_mpa": normal_out}, f)
            return

        # ── Rectangular lon/lat grid modes ("cff", "displacement") ──────────
        z_recv_km = job["z_recv_km"]
        grid = job["grid"]

        n_lon = int(grid["n_lon"])
        n_lat = int(grid["n_lat"])
        lon_vals = [grid["lon_min"] + (grid["lon_max"] - grid["lon_min"]) * i / (n_lon - 1)
                   for i in range(n_lon)]
        lat_vals = [grid["lat_min"] + (grid["lat_max"] - grid["lat_min"]) * i / (n_lat - 1)
                   for i in range(n_lat)]
        lon2d = [[lon_vals[j] for j in range(n_lon)] for _ in range(n_lat)]
        lat2d = [[lat_vals[i] for j in range(n_lon)] for i in range(n_lat)]

        if mode == "displacement":
            mu = job["mu"]
            nu = job["nu"]
            ux_grid = [[0.0] * n_lon for _ in range(n_lat)]
            uy_grid = [[0.0] * n_lon for _ in range(n_lat)]
            uz_grid = [[0.0] * n_lon for _ in range(n_lat)]

            for src in sources:
                for i in range(n_lat):
                    for j in range(n_lon):
                        e_km, n_km = geo_to_km(lon_vals[j], lat_vals[i], src["lon"], src["lat"])
                        ue, un, uz = disp_from_dc3d_point(
                            dc3dwrapper, e_km, n_km, src, mu, nu, z_recv_km)
                        ux_grid[i][j] += ue
                        uy_grid[i][j] += un
                        uz_grid[i][j] += uz

            with open(out_path, "w") as f:
                json.dump({"success": True, "ux_m": ux_grid, "uy_m": uy_grid,
                          "uz_m": uz_grid, "lon2d": lon2d, "lat2d": lat2d}, f)
            return

        if mode == "stress_tensor":
            # Raw 6-component stress tensor at each grid point, Pa,
            # geographic (East, North, Down) frame, tension-positive --
            # NO receiver resolution at all. Reuses stress_from_dc3d_point()
            # exactly as-is (same function the "cff" mode below calls) --
            # no new physics, only a different packaging of its output.
            mu = job["mu"]
            nu = job["nu"]
            sxx_grid = [[0.0] * n_lon for _ in range(n_lat)]
            syy_grid = [[0.0] * n_lon for _ in range(n_lat)]
            szz_grid = [[0.0] * n_lon for _ in range(n_lat)]
            sxy_grid = [[0.0] * n_lon for _ in range(n_lat)]
            sxz_grid = [[0.0] * n_lon for _ in range(n_lat)]
            syz_grid = [[0.0] * n_lon for _ in range(n_lat)]

            for src in sources:
                for i in range(n_lat):
                    for j in range(n_lon):
                        e_km, n_km = geo_to_km(lon_vals[j], lat_vals[i], src["lon"], src["lat"])
                        sxx, syy, szz, sxy, sxz, syz = stress_from_dc3d_point(
                            dc3dwrapper, e_km, n_km, src, mu, nu, z_recv_km)
                        sxx_grid[i][j] += sxx
                        syy_grid[i][j] += syy
                        szz_grid[i][j] += szz
                        sxy_grid[i][j] += sxy
                        sxz_grid[i][j] += sxz
                        syz_grid[i][j] += syz

            with open(out_path, "w") as f:
                json.dump({"success": True,
                          "sxx_pa": sxx_grid, "syy_pa": syy_grid, "szz_pa": szz_grid,
                          "sxy_pa": sxy_grid, "sxz_pa": sxz_grid, "syz_pa": syz_grid,
                          "lon2d": lon2d, "lat2d": lat2d}, f)
            return

        # mode == "cff" (default)
        mu = job["mu"]
        nu = job["nu"]
        receiver = job["receiver"]
        cff_grid = [[0.0] * n_lon for _ in range(n_lat)]

        # Receiver normal (n) and slip (l) unit vectors, geographic frame
        sr = math.radians(receiver["strike"])
        dr = math.radians(receiver["dip"])
        rr = math.radians(receiver["rake"])
        cs, ss = math.cos(sr), math.sin(sr)
        cd, sd = math.cos(dr), math.sin(dr)
        cr, sr_ = math.cos(rr), math.sin(rr)
        nx, ny, nz = cs * sd, -ss * sd, -cd
        lx = cr * ss - sr_ * cs * cd
        ly = cr * cs + sr_ * ss * cd
        lz = -sr_ * sd
        friction = receiver["friction"]

        for src in sources:
            for i in range(n_lat):
                for j in range(n_lon):
                    e_km, n_km = geo_to_km(lon_vals[j], lat_vals[i], src["lon"], src["lat"])
                    sxx, syy, szz, sxy, sxz, syz = stress_from_dc3d_point(
                        dc3dwrapper, e_km, n_km, src, mu, nu, z_recv_km)
                    tx = sxx * nx + sxy * ny + sxz * nz
                    ty = sxy * nx + syy * ny + syz * nz
                    tz = sxz * nx + syz * ny + szz * nz
                    dtau = tx * lx + ty * ly + tz * lz
                    dsn = tx * nx + ty * ny + tz * nz
                    cff_grid[i][j] += (dtau + friction * dsn) / 1e6  # Pa -> MPa

        with open(out_path, "w") as f:
            json.dump({"success": True, "cff_mpa": cff_grid,
                      "lon2d": lon2d, "lat2d": lat2d}, f)

    except Exception as e:
        import traceback
        with open(out_path, "w") as f:
            json.dump({"success": False,
                      "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}, f)
        sys.exit(1)


if __name__ == "__main__":
    main()
