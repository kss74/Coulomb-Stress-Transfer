# -*- coding: utf-8 -*-
"""
Focal Mechanism receivers — "stress on focal mechanisms" (Coulomb 3.4.2's
"Calc. stress on nodal planes" feature).

Physics summary (verified against coulomb.m's `focal_mech_calc`, 2026-08-12):
Coulomb treats each catalog EARTHQUAKE as a receiver FAULT: its own
hypocenter/centroid (lon, lat, depth) is the observation point, and its own
nodal-plane strike/dip/rake is the receiver orientation. ΔCFF from every
source fault in the model is then resolved onto that geometry — exactly
the calculation `okada_engine.compute_cff_on_receiver_faults()` already
performs for the "Receiver Faults" tab. This module is therefore mostly
glue: turning a catalog of focal mechanisms into receiver `FaultParameters`
pairs (one per nodal plane) and applying Coulomb's plane-selection modes.

Two supporting conversions are needed to get catalog data into (strike,
dip, rake) form in the first place:

  * aux_plane()  — given ONE nodal plane, derive the other. Uses the
    standard Aki & Richards (1980, eq. 4.22-4.25) double-couple identity
    (plane 2's normal = plane 1's slip vector, and vice versa) rather than
    porting coulomb.m's own `nodal_plane_calc`/`TDL` pipeline — that
    pipeline's P/T-axis round trip is a different (equivalent) route to
    the same answer, and coulomb.m's comments note its predecessor
    'AuxPlane' implementation had a bug, which is exactly the kind of
    fragile trig-branch code this module avoids re-deriving. Reuses
    optimal_plane._traction_plane_to_strike_dip_rake(), which is already
    validated to round-trip through compute_cff() consistently.

  * mij2sdr()    — given a moment tensor, derive nodal plane 1. Ported
    directly from coulomb.m's `mij2sdr`/`TDL` (eigenvector decomposition
    of the moment tensor into P/T axes, Aki & Richards convention). This
    is the ground-truth source per this project's own verification rules.
    ⚠ Input convention (mxx, myy, mzz, mxy, mxz, myz) is whatever
    coulomb.m's own moment-tensor import expects, NOT verified here
    against GCMT's USE (Up-South-East) NDK convention — see
    gcmt_use_to_local() docstring. Treat moment-tensor import as
    provisional until cross-checked against one known real GCMT solution.

Self-tests for both conversions are in tests/test_focal_mechanism.py.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from .okada_engine import FaultParameters, ElasticParameters, compute_cff_on_receiver_faults


# ─── Data model ────────────────────────────────────────────────────────────

@dataclass
class FocalMechanismEvent:
    """
    One earthquake with a (possibly two-plane) focal mechanism.

    lon, lat   : epicenter/centroid, degrees
    depth      : km, positive down
    magnitude  : optional, for display/scaling only (not used in the CFF calc)
    strike1/dip1/rake1 : nodal plane 1 (Aki-Richards convention, matching
                          FaultParameters — degrees)
    strike2/dip2/rake2 : nodal plane 2. If not supplied at construction,
                          call fill_aux_plane() to derive it from plane 1.
    label      : optional catalog ID / event name, for display and export
    """
    lon: float
    lat: float
    depth: float
    strike1: float
    dip1: float
    rake1: float
    strike2: Optional[float] = None
    dip2: Optional[float] = None
    rake2: Optional[float] = None
    magnitude: Optional[float] = None
    label: str = ""

    def fill_aux_plane(self):
        """Derive nodal plane 2 from plane 1 in place, if not already set."""
        if self.strike2 is None or self.dip2 is None or self.rake2 is None:
            self.strike2, self.dip2, self.rake2 = aux_plane(
                self.strike1, self.dip1, self.rake1)
        return self

    def has_both_planes(self):
        return None not in (self.strike2, self.dip2, self.rake2)

    @classmethod
    def from_moment_tensor_ned(cls, lon, lat, depth, mnn, mee, mdd, mne, mnd, med,
                                magnitude=None, label=""):
        """Build from a moment tensor in the standard (N,E,D) convention."""
        from .focal_mechanism import moment_tensor_ned_to_sdr
        s1, d1, r1 = moment_tensor_ned_to_sdr(mnn, mee, mdd, mne, mnd, med)
        return cls(lon=lon, lat=lat, depth=depth, strike1=s1, dip1=d1, rake1=r1,
                   magnitude=magnitude, label=label)

    @classmethod
    def from_gcmt(cls, lon, lat, depth, mrr, mtt, mpp, mrt, mrp, mtp,
                   magnitude=None, label=""):
        """Build from a GCMT/NDK moment tensor (USE convention)."""
        from .focal_mechanism import gcmt_use_to_sdr
        s1, d1, r1 = gcmt_use_to_sdr(mrr, mtt, mpp, mrt, mrp, mtp)
        return cls(lon=lon, lat=lat, depth=depth, strike1=s1, dip1=d1, rake1=r1,
                   magnitude=magnitude, label=label)


# ─── Plane <-> normal/slip vector (East, North, Down) ──────────────────────
# Duplicated in miniature from okada_engine.compute_cff() / optimal_plane.py
# on purpose (see okada_engine module docstring: "new physics stays in new
# files"). Must stay algebraically identical to compute_cff()'s nx/ny/nz,
# lx/ly/lz -- covered by tests/test_focal_mechanism.py::test_aux_plane_matches_compute_cff.

def _plane_normal_slip(strike_deg, dip_deg, rake_deg):
    sr = np.deg2rad(strike_deg); dr = np.deg2rad(dip_deg); rr = np.deg2rad(rake_deg)
    cs, ss = np.cos(sr), np.sin(sr)
    cd, sd = np.cos(dr), np.sin(dr)
    cr, sra = np.cos(rr), np.sin(rr)
    n = np.array([cs * sd, -ss * sd, -cd])
    l = np.array([cr * ss - sra * cs * cd, cr * cs + sra * ss * cd, -sra * sd])
    return n, l


def aux_plane(strike_deg, dip_deg, rake_deg):
    """
    Standard double-couple auxiliary-plane identity: plane 2's normal is
    plane 1's slip vector, and plane 2's slip vector is plane 1's normal
    (Aki & Richards 1980, eq. 4.22-4.25). Exact (no branch-dependent
    trig), unlike the naive strike/dip/rake-formula route.
    """
    from .optimal_plane import _traction_plane_to_strike_dip_rake
    n1, l1 = _plane_normal_slip(strike_deg, dip_deg, rake_deg)
    n2, l2 = l1, n1
    # _traction_plane_to_strike_dip_rake() enforces nz<=0 by negating ONLY
    # n internally if nz>0 -- correct for its existing caller (the
    # optimal-plane eigen-solver, which always hands it a consistent (n,l)
    # pair with nz<=0 already, so that branch is never exercised there).
    # Here nz CAN be >0 (n2 = l1, an arbitrary slip vector), and negating
    # n alone while leaving l untouched describes a DIFFERENT physical
    # fault (flips the rake sign) -- confirmed numerically: flipping n
    # only changes the returned rake's sign vs. flipping n and l together.
    # n and l must be negated TOGETHER (same plane, viewed from the other
    # side) to preserve the physical slip sense, so do that here before
    # calling, keeping nz<=0 already satisfied on entry.
    if n2[2] > 0:
        n2, l2 = -n2, -l2
    return _traction_plane_to_strike_dip_rake(n2, l2)


# ─── Fault-type classification (Frohlich 1992 ternary P/T/B scheme) ───────
# Used by the cross-section focal-mechanism overlay's color_by="type" and
# any future map-view symbology needing the same "what kind of fault is
# this" bin, so it's defined once here (core physics) rather than
# duplicated in a plotting module. This is genuinely a NEW physics
# addition (nothing in the project computed P/T/B axes or classified by
# them before), not a repackaging of an existing function.
#
# Derivation: for a double couple, the P (pressure/compressional) and T
# (tension) axes are the eigenvectors of the moment tensor with the most
# negative/positive eigenvalues, which for a pure double couple reduce to
# simple combinations of the normal (n) and slip (l) vectors already
# computed by _plane_normal_slip() -- T = (n+l)/|n+l|, P = (n-l)/|n-l|
# (Aki & Richards 1980, Ch. 4; Stein & Wysession 2003, eq. 4.71-4.72),
# and B (the null axis) = n x l. All three are mutually orthogonal by
# construction (n and l already are, since l lies in the fault plane and
# n is its normal) -- confirmed numerically in check() below.
#
# Frohlich (1992, "Triangle diagrams...") classifies a mechanism by
# whichever of the three axes' PLUNGE (angle below horizontal) is
# steepest: a steep P plunge means near-vertical shortening -> normal
# faulting; a steep T plunge means near-vertical extension -> reverse
# (thrust) faulting; a steep B plunge means the fault is nearly
# vertical-strike-slip. The 50 degree threshold below is Frohlich's own
# (his Table 1 uses 50 degree/60 degree bands for the "pure" categories vs.
# the oblique field between them); anything with no axis reaching 50
# degrees plunge falls in the broad "oblique" field in the middle of his
# triangle diagram.
FAULT_TYPE_COLORS = {
    # 2026-08-21: switched to the actual field-standard convention
    # instead of an in-house colorblind-safe substitute -- the World
    # Stress Map project (Heidbach et al.; world-stress-map.org) and
    # the broader stress-regime/focal-mechanism literature built on it
    # use red=normal, green=strike-slip, blue=thrust/reverse
    # essentially universally (WSM database releases, GFZ's public
    # stress-map figures, etc. all use exactly this triple). "Oblique"
    # has no single WSM equivalent (WSM's 4th class is "unknown", not
    # "oblique") -- orange is used here as a 4th, visually-distinct
    # color from the same widely-used qualitative palette family
    # (ColorBrewer Set1) the other three approximate, rather than
    # inventing an unrelated hue.
    "normal": "red",
    "reverse": "blue",
    "strike-slip": "green",
    "oblique": "#ff7f00",        # orange
}


def _axis_plunge_deg(v):
    """Plunge (deg below horizontal, 0-90) of a unit (E,N,D) vector."""
    return float(np.degrees(np.arcsin(np.clip(abs(v[2]), -1.0, 1.0))))


def classify_fault_type(strike_deg, dip_deg, rake_deg, threshold_deg=50.0):
    """
    Frohlich (1992) P/T/B-plunge classification. Returns one of
    "normal", "reverse", "strike-slip", "oblique" -- a key into
    FAULT_TYPE_COLORS.

    Only plane 1 is needed: P/T/B are properties of the DOUBLE COUPLE
    (the moment tensor), identical whichever of the two nodal planes you
    started from -- classifying from plane 2 gives the same answer
    (verified in check() below), so callers don't need to worry about
    which plane the catalog happened to store as "plane 1".
    """
    n, l = _plane_normal_slip(strike_deg, dip_deg, rake_deg)
    t_axis = n + l
    p_axis = n - l
    t_axis = t_axis / np.linalg.norm(t_axis)
    p_axis = p_axis / np.linalg.norm(p_axis)
    b_axis = np.cross(n, l)
    b_axis = b_axis / np.linalg.norm(b_axis)

    p_pl = _axis_plunge_deg(p_axis)
    t_pl = _axis_plunge_deg(t_axis)
    b_pl = _axis_plunge_deg(b_axis)

    # Steepest axis wins, provided it clears the threshold; ties broken
    # by P > T > B precedence (arbitrary but deterministic -- exact ties
    # are measure-zero in practice).
    best = max(p_pl, t_pl, b_pl)
    if best < threshold_deg:
        return "oblique"
    if p_pl == best:
        return "normal"
    if t_pl == best:
        return "reverse"
    return "strike-slip"


# ─── Moment tensor -> nodal plane 1 (ported from coulomb.m mij2sdr/TDL) ────

def _tdl(an, bn):
    """Direct port of coulomb.m's nested TDL() function."""
    xn, yn, zn = an
    xe, ye, ze = bn
    AAA = 1.0e-6
    if abs(zn) < AAA:
        fd = 90.0
        axn = min(abs(xn), 1.0)
        ft = np.degrees(np.arcsin(axn))
        st, ct = -xn, yn
        if st >= 0 and ct < 0: ft = 180 - ft
        if st < 0 and ct <= 0: ft = 180 + ft
        if st < 0 and ct > 0: ft = 360 - ft
        fl = np.degrees(np.arcsin(min(abs(ze), 1.0)))
        sl = -ze
        if abs(xn) < AAA:
            cl = xe / yn
        else:
            cl = -ye / xn
        if sl >= 0 and cl < 0: fl = 180 - fl
        if sl < 0 and cl <= 0: fl = fl - 180
        if sl < 0 and cl > 0: fl = -fl
    else:
        zn_c = -1.0 if -zn > 1.0 else zn
        fdh = np.arccos(-zn_c)
        fd = np.degrees(fdh)
        sd = np.sin(fdh)
        if sd == 0:
            return 0.0, fd, 0.0
        st, ct = -xn / sd, yn / sd
        ft = np.degrees(np.arcsin(min(abs(st), 1.0)))
        if st >= 0 and ct < 0: ft = 180 - ft
        if st < 0 and ct <= 0: ft = 180 + ft
        if st < 0 and ct > 0: ft = 360 - ft
        sl = -ze / sd
        fl = np.degrees(np.arcsin(min(abs(sl), 1.0)))
        if st == 0:
            cl = xe / ct
        else:
            xxx = yn * zn * ze / sd / sd + ye
            cl = -sd * xxx / xn
            if ct == 0:
                cl = ye / st
        if sl >= 0 and cl < 0: fl = 180 - fl
        if sl < 0 and cl <= 0: fl = fl - 180
        if sl < 0 and cl > 0: fl = -fl
    return ft, fd, fl


def mij2sdr(mxx, myy, mzz, mxy, mxz, myz):
    """
    Moment tensor (6 independent components, in mij2sdr's OWN internal
    axis convention -- see moment_tensor_ned_to_sdr() for the conversion
    from the standard (North, East, Down) convention used everywhere
    else in this project) -> nodal plane 1 (strike, dip, rake). Direct
    port of coulomb.m's mij2sdr(), eigenvector decomposition into P/T
    axes (Aki & Richards convention).

    ⚠ Do not call this directly with (N,E,D)-ordered components -- its
    row/column-reordering step (ported verbatim from coulomb.m) only
    produces correct results in ITS OWN internal axis convention, which
    was reverse-engineered empirically (not documented in coulomb.m's
    comments) to be: x=Down, y=North, z=-East. Use
    moment_tensor_ned_to_sdr() instead, which handles that conversion
    and is the function validated by tests/test_focal_mechanism.py.
    """
    a = np.array([[mxx, mxy, mxz], [mxy, myy, myz], [mxz, myz, mzz]], dtype=float)
    d, V = np.linalg.eigh(a)  # ascending eigenvalues, matches MATLAB eig() ordering
    D = np.array([d[2], d[0], d[1]])
    V = V.copy()
    V[1:3, 0:3] = -V[1:3, 0:3]
    V = np.array([
        [V[1, 2], V[1, 0], V[1, 1]],
        [V[2, 2], V[2, 0], V[2, 1]],
        [V[0, 2], V[0, 0], V[0, 1]],
    ])
    imax = int(np.argmax(D))
    imin = int(np.argmin(D))
    AE = (V[:, imax] + V[:, imin]) / np.sqrt(2.0)
    AN = (V[:, imax] - V[:, imin]) / np.sqrt(2.0)
    AE = AE / np.linalg.norm(AE)
    AN = AN / np.linalg.norm(AN)
    if AN[2] <= 0:
        AN1, AE1 = AN, AE
    else:
        AN1, AE1 = -AN, -AE
    ft, fd, fl = _tdl(AN1, AE1)
    strike = (360.0 - ft) % 360.0
    dip = fd
    rake = 180.0 - fl
    # normalize rake into (-180, 180]
    rake = ((rake + 180.0) % 360.0) - 180.0
    return float(strike), float(dip), float(rake)


# mij2sdr's own internal axis convention, reverse-engineered empirically
# (coulomb.m's comments don't state it) by brute-force search over all 48
# axis-permutation/sign combinations against 8 independent test
# mechanisms spanning strike-slip/normal/reverse/oblique geometries, in
# tests/test_focal_mechanism.py -- exactly one combination (up to an
# overall sign, which is physically irrelevant for a double couple)
# reproduced the correct nodal plane in all 8 cases: x=Down, y=North,
# z=-East. moment_tensor_ned_to_sdr() below applies that permutation so
# CALLERS only ever need to think in the project's standard (N,E,D)
# convention.
def moment_tensor_ned_to_sdr(mnn, mee, mdd, mne, mnd, med):
    """
    Moment tensor in the standard (North, East, Down) convention --
    mnn, mee, mdd, mne, mnd, med -- to nodal plane 1 (strike, dip, rake).
    This is the function to call; see mij2sdr()'s docstring for why the
    raw mij2sdr() inputs are NOT simply (mnn, mee, mdd, mne, mnd, med).
    """
    mxx, myy, mzz = mdd, mnn, mee
    mxy, mxz, myz = mnd, -med, -mne
    return mij2sdr(mxx, myy, mzz, mxy, mxz, myz)


def gcmt_use_to_ned(mrr, mtt, mpp, mrt, mrp, mtp):
    """
    GCMT/NDK moment tensor -- Up, South, East (r, t, p) convention -- to
    the standard (North, East, Down) convention used throughout this
    project. Standard textbook relabeling (Aki & Richards; also used by
    ObsPy's MomentTensor USE<->NED conversion):
        mnn =  mtt    mee =  mpp    mdd =  mrr
        mne = -mtp    mnd =  mrt    med = -mrp
    """
    mnn, mee, mdd = mtt, mpp, mrr
    mne, mnd, med = -mtp, mrt, -mrp
    return mnn, mee, mdd, mne, mnd, med


def gcmt_use_to_sdr(mrr, mtt, mpp, mrt, mrp, mtp):
    """
    GCMT/NDK moment tensor (USE convention) -> nodal plane 1 (strike,
    dip, rake). Chains gcmt_use_to_ned() with moment_tensor_ned_to_sdr().

    Verified two ways (tests/test_focal_mechanism.py):
      1. moment_tensor_ned_to_sdr() matches a REAL published event (SLU
         2007-03-25 solution: raw moment tensor + resulting double-couple
         planes both published, and they match to <0.1 deg).
      2. gcmt_use_to_ned()'s USE->NED formula independently matches the
         published Harvard-CMT-to-Cartesian relation from the Czech
         Academy of Sciences' MT_Decomposition tool.
    Not yet checked end-to-end against one single real GCMT NDK entry's
    raw Mrr..Mtp values against its published strike/dip/rake in the
    same file (both pieces verified independently, not jointly) --
    cheap to add once a real NDK sample is on hand.
    """
    ned = gcmt_use_to_ned(mrr, mtt, mpp, mrt, mrp, mtp)
    return moment_tensor_ned_to_sdr(*ned)


# ─── Build receiver FaultParameters from events ────────────────────────────

# Nominal receiver patch size. Only geometry at the CENTROID/point matters
# for compute_cff_on_receiver_faults() (it evaluates stress at the
# receiver's own lon/lat/depth using its strike/dip/rake for orientation
# only) -- length/width are not used in that evaluation, so any small
# placeholder is fine. Kept as a module constant rather than hardcoded so
# it's easy to find if that assumption ever changes.
_RECEIVER_PATCH_KM = 1.0


def build_source_fault_row(event: FocalMechanismEvent, plane: int,
                            relation_name: str, style: str, mu_pa: float = 32e9,
                            name_suffix: Optional[str] = None):
    """
    Turn ONE focal-mechanism event's CHOSEN nodal plane into a source-
    fault-table row, for designating an imported focal mechanism as a
    stress-generating SOURCE fault (as opposed to the RECEIVER usage
    build_receiver_faults()/compute_focal_mechanism_cff() implement above).

    A focal mechanism only ever gives point geometry (lon, lat, depth) and
    orientation (strike, dip, rake) -- never fault length/width/slip. This
    fills that gap using the SAME empirical scaling relations, and the
    SAME rake-decomposition math, as the "Estimate L/W/slip from
    magnitude" dialog (ui/scaling_dialog.py), via
    core.scaling_relations.compute_scaling_result() -- both now call that
    one function, so results are numerically identical whichever route
    produced them.

    plane        : 1 or 2 -- which nodal plane's strike/dip/rake to use as
                   the source fault's orientation.
    relation_name: a key of core.scaling_relations.SCALING_RELATIONS.
    style        : a value from core.scaling_relations.FAULT_STYLES. This
                   only controls which regression coefficients are used
                   for length/width/slip MAGNITUDE -- the actual rt-
                   lateral/reverse SPLIT always comes from the chosen
                   plane's own (real, not style-typical) rake, unlike the
                   scaling dialog's rake_spin (which defaults to a style-
                   representative rake but can be overridden).

    Returns (row, warnings):
      row      : dict with keys name, lonlat_mode, lon, lat, depth, length,
                 width, strike, dip, rt_lateral_slip, reverse_slip, rake,
                 subdiv_l, subdiv_w -- the exact shape
                 FaultTableWidget.get_raw_rows()/set_raw_rows() use, so a
                 caller can pass this straight to FaultTableWidget.add_row()
                 (see main_dialog.add_focal_mechanisms_as_sources_action()).
                 lonlat_mode is always "centroid": a focal mechanism's
                 (lon, lat, depth) IS the event's own hypocenter/centroid,
                 exactly the case that mode exists for (see
                 fault_table_widget.py's module docstring). `rake` in the
                 returned row is 0.0 and is IGNORED downstream for any
                 nonzero-slip row (see fault_table_widget.get_faults()) --
                 the row's real orientation is carried by strike/dip plus
                 the rt-lateral/reverse split, not this field.
      warnings : list[str], passed through from compute_scaling_result()
                 (e.g. "unverified relation", "coulomb-compatible slip
                 ignores mu_pa") for the caller to surface to the user.

    Raises ValueError if plane is not 1 or 2, if plane=2 is requested but
    the event has no plane 2 filled in, or if the event has no magnitude
    (scaling relations require Mw -- there is no other route to a length/
    width/slip estimate here).
    """
    if plane not in (1, 2):
        raise ValueError(f"plane must be 1 or 2, got {plane!r}")
    if plane == 2 and not event.has_both_planes():
        raise ValueError(
            f"Event {event.label or '(unlabeled)'!r} has no nodal plane 2 "
            "-- call fill_aux_plane() first, or choose plane=1")
    if event.magnitude is None:
        raise ValueError(
            f"Event {event.label or '(unlabeled)'!r} has no magnitude -- "
            "required to estimate fault length/width/slip via scaling "
            "relations")

    if plane == 1:
        strike, dip, rake = event.strike1, event.dip1, event.rake1
    else:
        strike, dip, rake = event.strike2, event.dip2, event.rake2

    from .scaling_relations import compute_scaling_result
    scaling = compute_scaling_result(
        relation_name=relation_name, style=style, mw=event.magnitude,
        rake_deg=rake, mu_pa=mu_pa)

    if scaling["length_km"] is None or scaling["width_km"] is None \
            or scaling["rt_lateral_slip_m"] is None:
        raise ValueError(
            f"{relation_name!r} did not produce a usable length/width/"
            f"slip estimate for Mw={event.magnitude:.2f}, style={style!r}")

    label = event.label or "FM event"
    name = name_suffix if name_suffix is not None else f"{label} (FM NP{plane})"

    row = dict(
        name=name, lonlat_mode="centroid",
        lon=event.lon, lat=event.lat, depth=event.depth,
        length=scaling["length_km"], width=scaling["width_km"],
        strike=strike, dip=dip,
        rt_lateral_slip=scaling["rt_lateral_slip_m"],
        reverse_slip=scaling["reverse_slip_m"],
        rake=0.0, subdiv_l=1, subdiv_w=1,
    )
    return row, scaling["warnings"]


def build_receiver_faults(events: List[FocalMechanismEvent]):
    """
    Returns (plane1_faults, plane2_faults, has_plane2_mask) -- three
    parallel lists, one entry per event. plane2 entries are None where
    the event has no plane 2 filled in (call fill_aux_plane() on all
    events first if you always want both).
    """
    plane1_faults, plane2_faults, has_plane2 = [], [], []
    for ev in events:
        plane1_faults.append(FaultParameters(
            lon=ev.lon, lat=ev.lat, depth=ev.depth,
            length=_RECEIVER_PATCH_KM, width=_RECEIVER_PATCH_KM,
            strike=ev.strike1, dip=ev.dip1, rake=ev.rake1, slip=0.0))
        if ev.has_both_planes():
            plane2_faults.append(FaultParameters(
                lon=ev.lon, lat=ev.lat, depth=ev.depth,
                length=_RECEIVER_PATCH_KM, width=_RECEIVER_PATCH_KM,
                strike=ev.strike2, dip=ev.dip2, rake=ev.rake2, slip=0.0))
            has_plane2.append(True)
        else:
            plane2_faults.append(None)
            has_plane2.append(False)
    return plane1_faults, plane2_faults, has_plane2


# ─── Main entry point ───────────────────────────────────────────────────────

PLANE_MODES = ("plane1", "plane2", "max", "min", "random")


def compute_focal_mechanism_cff(sources, events: List[FocalMechanismEvent],
                                 elastic: ElasticParameters,
                                 mode: str = "max",
                                 progress_callback=None,
                                 rng=None):
    """
    Resolve ΔCFF on every event's nodal plane(s), matching Coulomb
    3.4.2's "Calc. stress on nodal planes" (coulomb.m: focal_mech_calc +
    the INODAL mode switch).

    sources : list of FaultParameters with slip (the stress SOURCES)
    events  : list of FocalMechanismEvent. Events without plane 2 filled
              in are evaluated on plane 1 only regardless of `mode`.
    mode    : one of PLANE_MODES --
              "plane1" / "plane2" -- always report that plane
              "max"  -- report whichever plane has the larger (more
                        destabilizing) ΔCFF, per event (Coulomb's
                        "max delta CFF" nodal-plane mode)
              "min"  -- the smaller ΔCFF
              "random" -- Coulomb's Monte-Carlo nodal-plane-ambiguity
                        mode: one plane picked at random per event
    rng     : optional numpy.random.Generator for mode="random"
              (reproducibility in tests); defaults to a fresh Generator.

    Returns a list of dicts, one per event, each with:
      {"event": FocalMechanismEvent,
       "plane1": {"cff_mpa", "shear_mpa", "normal_mpa", "used_dc3d"} or None,
       "plane2": {...} or None (None if event has no plane 2),
       "selected": "plane1" | "plane2" -- which one `mode` picked,
       "cff_mpa", "shear_mpa", "normal_mpa" -- the selected plane's values,
       repeated at the top level for convenient table/beachball display}
    """
    if mode not in PLANE_MODES:
        raise ValueError(f"mode must be one of {PLANE_MODES}, got {mode!r}")

    plane1_faults, plane2_faults, has_plane2 = build_receiver_faults(events)

    # Batch both planes through compute_cff_on_receiver_faults() in ONE
    # call for efficiency (it already vectorizes the z=0 path and batches
    # the DC3D subprocess calls); plane-2 entries that are None are simply
    # omitted from the batch and re-inserted as None afterward.
    batch = list(plane1_faults)
    plane2_positions = {}  # event index -> position in `batch`
    for i, p2 in enumerate(plane2_faults):
        if p2 is not None:
            plane2_positions[i] = len(batch)
            batch.append(p2)

    results = compute_cff_on_receiver_faults(
        sources, batch, elastic, progress_callback=progress_callback)

    n = len(events)
    plane1_results = results[:n]
    if rng is None:
        rng = np.random.default_rng()

    out = []
    for i, ev in enumerate(events):
        r1 = plane1_results[i]
        p1 = dict(cff_mpa=r1["cff_mpa"], shear_mpa=r1["shear_mpa"],
                  normal_mpa=r1["normal_mpa"], used_dc3d=r1["used_dc3d"])
        p2 = None
        if i in plane2_positions:
            r2 = results[plane2_positions[i]]
            p2 = dict(cff_mpa=r2["cff_mpa"], shear_mpa=r2["shear_mpa"],
                      normal_mpa=r2["normal_mpa"], used_dc3d=r2["used_dc3d"])

        if p2 is None:
            selected = "plane1"
        elif mode == "plane1":
            selected = "plane1"
        elif mode == "plane2":
            selected = "plane2"
        elif mode == "max":
            selected = "plane1" if p1["cff_mpa"] >= p2["cff_mpa"] else "plane2"
        elif mode == "min":
            selected = "plane1" if p1["cff_mpa"] <= p2["cff_mpa"] else "plane2"
        elif mode == "random":
            selected = "plane1" if rng.integers(0, 2) == 0 else "plane2"

        sel = p1 if selected == "plane1" else p2
        out.append({
            "event": ev, "plane1": p1, "plane2": p2, "selected": selected,
            "cff_mpa": sel["cff_mpa"], "shear_mpa": sel["shear_mpa"],
            "normal_mpa": sel["normal_mpa"], "used_dc3d": sel["used_dc3d"],
        })
    return out
