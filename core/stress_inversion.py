# -*- coding: utf-8 -*-
"""
Regional stress-tensor orientation from a catalog of earthquake focal
mechanisms, via ILSI (Iterative Linear Stress Inversion).

    Eric Beaucé, Robert D. van der Hilst, Michel Campillo (2022). An
    Iterative Linear Method with Variable Shear Stress Magnitudes for
    Estimating the Stress Tensor from Earthquake Focal Mechanism Data:
    Method and Examples. Bulletin of the Seismological Society of
    America. DOI: 10.1785/0120210319
    https://github.com/ebeauce/ILSI  (GPL-3.0)

WHY THIS MODULE EXISTS
-----------------------
`optimal_plane.py`'s `RegionalStress` (the "Opt Faults" feature's
tectonic-stress input) currently requires the user to hand-enter the
strike/plunge of the S1/S2 principal axes -- a guess, or a value copied
from a paper. This module derives that orientation directly from a
catalog of focal mechanisms already imported into this plugin
(`focal_mechanism.FocalMechanismEvent`), using ILSI's implementation of
the instability-parameter method (Vavrycuk 2013, 2014; Lund & Slunga
1999), which simultaneously (a) resolves each event's nodal-plane
ambiguity (which of the two nodal planes is the real fault) and
(b) inverts for the stress tensor, iterating between the two -- the
same method the "iterative linear method with variable shear stress"
paper above is built on. ILSI was chosen over the more commonly-cited
MSATSI (Martinez-Garzon et al. 2014) specifically because MSATSI is
built around spatial/temporal BINNING of a large regional catalog
(its own stated use case), whereas ILSI operates on a single
user-supplied set of mechanisms with no minimum-area assumption --
matching this plugin's existing per-study-area, per-import workflow
(the same catalog already imported for aftershock/rate-and-state
work, see `eq_catalog_import.py`, or a dedicated focal-mechanism
import via `focal_mechanism_import.py`).

WHAT THIS MODULE DELIBERATELY DOES NOT DO
--------------------------------------------
Focal-mechanism-only stress inversion (the Wallace-Bott assumption:
slip direction on a fault parallels the resolved shear traction) is a
fundamentally SCALE-FREE and OFFSET-FREE problem: it constrains only
the deviatoric stress tensor's ORIENTATION and SHAPE RATIO
R=(S1-S2)/(S1-S3), never its absolute magnitude or its isotropic
(purely hydrostatic) offset -- there is no physical mechanism by which
slip directions alone could reveal how many bars of differential
stress are actually present, or what the mean stress level is. ILSI's
own `reduced_stress_tensor()` makes this explicit: it always returns
a tensor NORMALIZED to sigma1=-1, sigma3=+1 (tension-positive), never
absolute Pa/bars.

This module therefore returns ORIENTATION + SHAPE RATIO from ILSI,
and requires the CALLER (eventually: the UI dialog) to separately
supply a target differential-stress magnitude (S1-S3, in bars,
Coulomb's own compression-positive convention) to scale that into the
`RegionalStress` object `optimal_plane.py` actually needs for a
physical CFF calculation -- see `regional_stress_from_inversion()`.
This is a hard physical limitation of the method, not a shortcut taken
here; flagging it explicitly (rather than silently defaulting to some
arbitrary magnitude) is deliberate, matching this project's "known
limitations flagged in code comments, not silently deferred" rule.

COORDINATE CONVENTION -- VERIFIED, NOT ASSUMED
--------------------------------------------------------
ILSI works internally in a (North, West, Up) Cartesian frame, tension-
positive, and labels its eigen-ordering "sigma1 = most compressive .. 
sigma3 = least compressive" by VALUE ORDER (ascending eigenvalue under
its tension-positive sign, i.e. sigma1 is the most NEGATIVE number --
confirmed by reading `utils_stress.stress_tensor_eigendecomposition`'s
`order = np.argsort(principal_stresses)` with no reversal, and its own
comment "reorder from most compressive to most extensional with
tension positive convention").

Every other module in this plugin (`optimal_plane.py`,
`focal_mechanism.py`, `okada_engine.py`) works in (East, North, Down),
tension-positive (see `optimal_plane._axis_vector`'s own docstring:
"Unit vector (East, North, Down) for a strike/plunge axis", and
`regional_stress_tensor_pa`'s explicit tension-positive statement).

Converting a vector from ILSI's (N, W, U) frame to this plugin's
(E, N, D) frame:
    N_hat (ILSI) = (0, 1, 0) in (E,N,D)
    W_hat (ILSI) = (-1, 0, 0) in (E,N,D)   [West = -East]
    U_hat (ILSI) = (0, 0, -1) in (E,N,D)   [Up = -Down]
so a vector (vN, vW, vU) in ILSI's frame has (E,N,D) components:
    E = -vW,  N = vN,  D = -vU
This is `_nwu_to_end()` below. It is an orthogonal, right-handed,
determinant-+1 transform (a pure axis permutation + two sign flips),
so it preserves angles, handedness, and eigen-ordering -- confirmed
both algebraically and numerically in `verify_stress_inversion.py`
against a synthetic normal-faulting mechanism set (uniform strike,
dip=60, rake=-90), which is known on purely physical (Andersonian)
grounds to require sigma1 (most compressive) vertical: ILSI returns
that axis as (N,W,U)=(0,0,1) [straight up], and `_nwu_to_end()` maps
it to (E,N,D)=(0,0,-1) [straight up in E,N,D too, D=-1], which
`_end_vector_to_strike_plunge()` correctly reports as plunge=90 deg
(vertical) regardless of strike -- matching the known-correct answer.

AUXILIARY-PLANE CONSISTENCY
------------------------------
`focal_mechanism.py` already has its own `aux_plane()` (derived
independently via `_traction_plane_to_strike_dip_rake`, not borrowed
from ILSI). ILSI's own `utils_stress.aux_plane()` is a direct port of
the same Michael/Ji/Boyd `bb.m` formula (also used by ObsPy) that this
plugin's version was checked against. Rather than pass this plugin's
own already-computed plane-2 values into ILSI's
`inversion_one_set_instability(..., auxiliary_planes=...)` (which
would create a code path that diverges from
`inversion_bootstrap_instability`, which always derives its own
aux-plane internally and offers no `auxiliary_planes` override), this
module lets ILSI derive plane 2 itself throughout, for both the point
estimate and the bootstrap uncertainty -- keeping the two consistent
with each other. `verify_stress_inversion.py` numerically cross-checks
that ILSI's `aux_plane()` and this plugin's `focal_mechanism.aux_plane()`
agree to numerical precision on the same synthetic input, so this
substitution costs nothing.

SCOPE OF THIS DELIVERY
-------------------------
- Point-estimate inversion (`invert_regional_stress`) and bootstrap
  uncertainty (`bootstrap_regional_stress`) are implemented and
  verified here.
- Conversion to a `RegionalStress` object usable by `optimal_plane.py`
  is implemented (`regional_stress_from_inversion`), requiring an
  explicit user-supplied differential-stress magnitude (see above).
- NOT done here (left for the UI-wiring session): a dialog to pick a
  focal-mechanism catalog and drive this; the stereonet/instability
  plots ILSI can produce (`inversion_one_set_instability(..., plot=True)`
  needs `mplstereonet`, a NEW dependency not otherwise used in this
  plugin -- same "new dependency" flag as `beachball.py`'s `obspy`);
  QgsSettings persistence of the last-used friction/differential-stress
  values. All flagged for the next session per your instruction.
- ILSI itself (numpy/scipy only for the inversion math; `mplstereonet`
  is a further, separate, lazy import only inside ILSI's own
  `plot=True` code paths) needs `pip install git+https://github.com/
  ebeauce/ILSI` into QGIS's own Python environment -- unlike
  `okada_wrapper`, ILSI is pure Python (no compiled extension), so
  the external-interpreter/subprocess machinery `okada_engine.py` uses
  for DC3D is NOT needed here; a direct `import ILSI` (mirroring how
  `beachball.py` does `from obspy...` directly) is sufficient, see
  `_has_ilsi()`.
"""

import numpy as np

from .focal_mechanism import FocalMechanismEvent
from .optimal_plane import RegionalStress


# ─── Optional dependency ────────────────────────────────────────────────────

def _has_ilsi():
    """Return True if ILSI is importable in the current Python environment."""
    try:
        import ILSI  # noqa: F401
        return True
    except ImportError:
        return False


def check_ilsi():
    """
    Verify ILSI is importable. Returns (ok: bool, message: str) -- mirrors
    `okada_engine.check_external_python()`'s (ok, message) convention so a
    dialog can display the same kind of status line.
    """
    try:
        import ILSI
        return True, f"ILSI {getattr(ILSI, '__version__', '?')} is importable."
    except ImportError as e:
        return False, (
            "ILSI is not installed in this Python environment. Install with:\n"
            "    pip install git+https://github.com/ebeauce/ILSI\n"
            f"(import error: {e})"
        )


def check_mplstereonet():
    """
    Verify `mplstereonet` is importable -- a SECOND, separate optional
    dependency from ILSI itself (see module docstring "SCOPE OF THIS
    DELIVERY": ILSI's own inversion math needs only numpy/scipy; the
    stereonet plot is the only thing that needs mplstereonet). Same
    (ok, message) convention as `check_ilsi()`, kept as its own function
    rather than folded into `check_ilsi()` so a dialog can report "ILSI
    OK, plotting unavailable" as a distinct, non-fatal state -- the
    inversion itself works fine without mplstereonet installed, only
    `ui/plot_widget.py`'s `plot_stress_inversion_stereonet()` needs it.
    """
    try:
        import mplstereonet
        return True, f"mplstereonet {getattr(mplstereonet, '__version__', '?')} is importable."
    except ImportError as e:
        return False, (
            "mplstereonet is not installed in this Python environment "
            "(needed only for the stereonet plot, not the inversion "
            "itself). Install with:\n"
            "    pip install mplstereonet\n"
            f"(import error: {e})"
        )


# ─── Coordinate conversion: ILSI's (N,W,U) -> this plugin's (E,N,D) ────────
# See module docstring "COORDINATE CONVENTION" section for the derivation
# and its numeric verification.

def _nwu_to_end(v):
    """
    Convert a length-3 vector (or (3,) numpy array) from ILSI's
    (North, West, Up) frame to this plugin's (East, North, Down) frame.
        E = -W,  N = N,  D = -U
    """
    vN, vW, vU = v
    return np.array([-vW, vN, -vU])


def _end_vector_to_strike_plunge(v):
    """
    Inverse of `optimal_plane._axis_vector()`: given a (not necessarily
    unit, not necessarily downward-pointing) vector in the (East, North,
    Down) frame representing one end of a principal-stress AXIS (a line,
    not a directed vector -- so the antipodal vector represents the same
    axis), return (strike_deg, plunge_deg) with plunge in [0, 90]
    (downward-pointing convention, matching `RegionalStress`'s own
    strike/plunge fields and `optimal_plane._axis_vector`'s docstring).
    """
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("Zero-length axis vector -- cannot determine strike/plunge.")
    v = v / norm
    E, N, D = v
    if D < 0:
        # axis points "up" in this representative -- use the antipodal
        # (physically identical) representative that points down, so
        # plunge comes out in [0, 90] as RegionalStress expects.
        E, N, D = -E, -N, -D
    plunge = float(np.degrees(np.arcsin(np.clip(D, -1.0, 1.0))))
    strike = float(np.degrees(np.arctan2(E, N))) % 360.0
    return strike, plunge


# ─── Focal-mechanism-list -> plain arrays ───────────────────────────────────

def _events_to_arrays(events):
    """
    Extract (strikes1, dips1, rakes1) float64 numpy arrays from a list of
    `focal_mechanism.FocalMechanismEvent`. Plane 2 is intentionally NOT
    extracted here -- see module docstring "AUXILIARY-PLANE CONSISTENCY":
    ILSI derives its own plane 2 internally, for both the point estimate
    and the bootstrap uncertainty.
    """
    if len(events) < 4:
        # ILSI's stress tensor has 5 free model parameters (deviatoric,
        # symmetric, traceless); fewer than ~4 mechanisms is not usually
        # considered a meaningfully constrained inversion in this
        # literature (each event contributes 2 shear-traction-component
        # equations after nodal-plane selection). Not a hard ILSI
        # requirement, but worth surfacing to the caller/UI rather than
        # silently returning a numerically-unconstrained result.
        raise ValueError(
            f"Only {len(events)} focal mechanism(s) supplied -- stress "
            "inversion needs a reasonably sized catalog (a handful of "
            "events at minimum, more for a well-resolved shape ratio and "
            "usable uncertainty estimates)."
        )
    strikes = np.array([e.strike1 for e in events], dtype=float)
    dips = np.array([e.dip1 for e in events], dtype=float)
    rakes = np.array([e.rake1 for e in events], dtype=float)
    return strikes, dips, rakes


def _shape_ratio(principal_stresses):
    """
    R = (sigma1 - sigma2) / (sigma1 - sigma3), matching ILSI's own
    `utils_stress.R_()` exactly (principal_stresses ordered sigma1..sigma3,
    most compressive first, as ILSI always returns them). Computed inline
    here rather than imported from ILSI, since it is a one-line formula
    and avoids importing ILSI in contexts where the caller already has
    principal_stresses on hand (e.g. per-bootstrap-replica, where
    importing inside a loop would be wasteful).
    """
    sig1, sig2, sig3 = principal_stresses
    return (sig1 - sig2) / (sig1 - sig3)


# ─── Point-estimate inversion ───────────────────────────────────────────────

def invert_regional_stress(events, friction_coefficient=0.6, variable_shear=True,
                            n_averaging=1, n_stress_iter=10, n_random_selections=20,
                            signed_instability=False, weighted=False, verbose=0):
    """
    Invert a catalog of focal mechanisms for the regional stress tensor's
    ORIENTATION and SHAPE RATIO, using ILSI's instability-parameter method
    (simultaneously resolves nodal-plane ambiguity and the stress tensor).

    Parameters
    -----------
    events : list of focal_mechanism.FocalMechanismEvent
        The focal-mechanism catalog to invert. Only strike1/dip1/rake1 are
        used (plane 2 is re-derived by ILSI itself, see module docstring).
    friction_coefficient : float or None, default 0.6
        Fixed friction coefficient, or None to let ILSI grid-search for
        the value that maximizes instability (slower; see ILSI's own
        `friction_min`/`friction_max`/`friction_step`, left at ILSI's
        defaults here -- exposed via **kwargs is not needed for v1).
    variable_shear : bool, default True
        True = the iterative linear method of Beaucé et al. 2022
        (recommended, ILSI's own default). False = the classic Michael
        (1984) constant-shear method.
    n_averaging : int, default 1
        Repeat the inversion this many times and average (improves
        reproducibility against ILSI's own random subset selection at
        the cost of run time) -- see ILSI's own docstring.
    n_stress_iter, n_random_selections, signed_instability, weighted :
        Passed straight through to ILSI; see
        `ILSI.ilsi.inversion_one_set_instability`'s own docstring for
        full detail. Left at ILSI's recommended defaults.
    verbose : int, default 0
        0 = silent (this plugin's own convention elsewhere is a quiet
        library layer with the UI dialog deciding what to show, so this
        defaults to 0 unlike ILSI's own default of 1).

    Returns
    --------
    result : dict
        - "stress_tensor_nwu" : (3,3) ndarray, ILSI's raw (normalized,
          dimensionless) stress tensor in its native (N,W,U) frame --
          kept for anyone wanting to feed it back into further ILSI
          calls (e.g. `bootstrap_regional_stress` below).
        - "principal_stresses" : (3,) ndarray, normalized eigenvalues
          [sigma1 (most compressive) .. sigma3 (least compressive)],
          tension-positive, DIMENSIONLESS (not bars/Pa -- see module
          docstring).
        - "shape_ratio" : float, R = (sigma1-sigma2)/(sigma1-sigma3),
          in [0, 1].
        - "friction_coefficient" : float, as used (or found, if the
          input was None).
        - "axes_end" : dict with keys "S1", "S2", "S3", each a
          (strike_deg, plunge_deg) tuple in this plugin's (E,N,D)
          convention -- ready to drop into `RegionalStress`.
        - "n_events" : int.
    """
    ok, msg = check_ilsi()
    if not ok:
        raise ImportError(msg)
    import ILSI

    strikes, dips, rakes = _events_to_arrays(events)

    out = ILSI.ilsi.inversion_one_set_instability(
        strikes, dips, rakes,
        friction_coefficient=friction_coefficient,
        n_stress_iter=n_stress_iter,
        n_random_selections=n_random_selections,
        n_averaging=n_averaging,
        signed_instability=signed_instability,
        variable_shear=variable_shear,
        weighted=weighted,
        verbose=verbose,
        return_stats=False,
        plot=False,
    )

    principal_stresses = np.asarray(out["principal_stresses"], dtype=float)
    principal_directions_nwu = np.asarray(out["principal_directions"], dtype=float)

    shape_ratio = float(_shape_ratio(principal_stresses))

    axes_end = {}
    for i, name in enumerate(("S1", "S2", "S3")):
        v_nwu = principal_directions_nwu[:, i]
        v_end = _nwu_to_end(v_nwu)
        axes_end[name] = _end_vector_to_strike_plunge(v_end)

    return {
        "stress_tensor_nwu": np.asarray(out["stress_tensor"], dtype=float),
        "principal_stresses": principal_stresses,
        "principal_directions_nwu": principal_directions_nwu,
        "shape_ratio": shape_ratio,
        "friction_coefficient": float(out["friction_coefficient"]),
        "axes_end": axes_end,
        "n_events": len(events),
    }


# ─── Bootstrap uncertainty ──────────────────────────────────────────────────

def bootstrap_regional_stress(events, base_result, n_resamplings=200,
                               n_stress_iter=10, variable_shear=True,
                               signed_instability=False, weighted=False):
    """
    Bootstrap-resample `events` to estimate uncertainty on the stress
    tensor orientation/shape ratio, using `base_result` (the output of
    `invert_regional_stress`) as the prior/reference solution -- wraps
    `ILSI.ilsi.inversion_bootstrap_instability`.

    Returns
    --------
    result : dict
        - "boot_axes_end" : dict with keys "S1","S2","S3", each an
          (n_resamplings, 2) ndarray of (strike_deg, plunge_deg) pairs
          in this plugin's (E,N,D) convention -- e.g. for a later
          stereonet scatter/confidence-region plot.
        - "boot_shape_ratio" : (n_resamplings,) ndarray.
        - "boot_principal_directions_nwu" : (n_resamplings,3,3) ndarray,
          ILSI's raw output, kept for anyone wanting it directly.
    """
    ok, msg = check_ilsi()
    if not ok:
        raise ImportError(msg)
    import ILSI

    strikes, dips, rakes = _events_to_arrays(events)

    out = ILSI.ilsi.inversion_bootstrap_instability(
        base_result["principal_directions_nwu"],
        base_result["shape_ratio"],
        strikes, dips, rakes,
        base_result["friction_coefficient"],
        n_resamplings=n_resamplings,
        n_stress_iter=n_stress_iter,
        variable_shear=variable_shear,
        signed_instability=signed_instability,
        weighted=weighted,
    )

    boot_dirs_nwu = np.asarray(out["boot_principal_directions"], dtype=float)
    boot_stresses = np.asarray(out["boot_principal_stresses"], dtype=float)

    boot_R = np.array([_shape_ratio(boot_stresses[b]) for b in range(boot_stresses.shape[0])])

    boot_axes_end = {name: np.zeros((boot_dirs_nwu.shape[0], 2)) for name in ("S1", "S2", "S3")}
    for b in range(boot_dirs_nwu.shape[0]):
        for i, name in enumerate(("S1", "S2", "S3")):
            v_end = _nwu_to_end(boot_dirs_nwu[b, :, i])
            boot_axes_end[name][b, :] = _end_vector_to_strike_plunge(v_end)

    return {
        "boot_axes_end": boot_axes_end,
        "boot_shape_ratio": boot_R,
        "boot_principal_directions_nwu": boot_dirs_nwu,
    }


# ─── Convert to optimal_plane.RegionalStress ───────────────────────────────

def regional_stress_from_inversion(result, differential_stress_bars,
                                    isotropic_offset_bars=0.0):
    """
    Build an `optimal_plane.RegionalStress` from an `invert_regional_stress()`
    result, given a user-supplied differential-stress magnitude.

    THIS MAGNITUDE CANNOT BE RECOVERED FROM FOCAL MECHANISMS -- see module
    docstring. The caller (UI dialog) MUST surface this as a value the
    user chooses (e.g. from independent stress-magnitude studies, borehole
    data, or a simple sensitivity sweep), not a number this function
    invents.

    Parameters
    -----------
    result : dict
        Output of `invert_regional_stress()`.
    differential_stress_bars : float
        S1 - S3, in bars, Coulomb's own compression-positive convention
        (matches `RegionalStress.S1`/`S3`'s own units). Must be > 0.
    isotropic_offset_bars : float, default 0.0
        (S1+S3)/2, in bars -- the mean/hydrostatic level, ALSO
        unresolvable from focal mechanisms alone (defaults to 0, an
        arbitrary-but-standard reference; note this offset does NOT
        affect the optimal plane's ORIENTATION (tan(2*phi)=1/friction
        depends only on the deviatoric part), but DOES shift the
        reported CFF_opt magnitude by friction * isotropic_offset_bars
        -- see `optimal_plane.py`'s own CFF_opt formula).

    Returns
    --------
    RegionalStress
    """
    if differential_stress_bars <= 0:
        raise ValueError("differential_stress_bars must be > 0.")

    R = result["shape_ratio"]
    S1 = isotropic_offset_bars + differential_stress_bars / 2.0
    S3 = isotropic_offset_bars - differential_stress_bars / 2.0
    S2 = S1 - R * differential_stress_bars

    s1_strike, s1_plunge = result["axes_end"]["S1"]
    s2_strike, s2_plunge = result["axes_end"]["S2"]

    return RegionalStress(
        S1=S1, S2=S2, S3=S3,
        S1_strike=s1_strike, S1_plunge=s1_plunge,
        S2_strike=s2_strike, S2_plunge=s2_plunge,
    )
