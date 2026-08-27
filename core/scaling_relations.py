# -*- coding: utf-8 -*-
"""
Empirical earthquake scaling relations: estimate fault length, width, and
average slip from moment magnitude, by fault style.

Implements Wells & Coppersmith (1994) — BSSA 84(4), 974-1002 — the
relations used by Coulomb 3.x/4.0 (Toda, Stein, Sevilgen & Lin).

Coefficients verified against a known Mw 7.0 strike-slip worked example
(SRL~44km, RLD~62km, RW~13km per the paper's own summary tables); this
implementation reproduces those values within a few percent.

NOTE: An earlier draft of this module also included Leonard (2010,
erratum 2014) as a second option, but pulled it entirely: its regression
coefficients could not be independently verified against the original
paper and, on testing, produced physically nonsensical results (fault
lengths of order 10^7 km for a Mw 7 earthquake).

2026-08-17: Added Thingbaijam, Mai & Goda (2017) and Strasser, Arango &
Bommer (2010) as additional relations. Length/width coefficients for both
were cross-checked against openquake-engine's hazardlib.scalerel source
(github.com/gem/oq-engine, GEM Foundation, AGPL) -- a peer-reviewed,
independently-maintained, widely-used PSHA engine -- rather than typed in
from a secondhand table, for the same reason Leonard was rejected above.
Unlike Wells & Coppersmith's AD table, neither paper publishes a slip
regression this module could verify with the same rigor, so slip for
both is instead solved by MOMENT BALANCE: given the relation's L, W and
the model's actual mu_pa, avg_slip_m is chosen so that
M0 = mu*L*W*avg_slip_m exactly reproduces the entered Mw (same dyne-cm/
-10.7 Kanamori convention as okada_engine.total_seismic_moment(), so
"Total seismic moment" after applying one of these rows will read back
very close to the Mw entered here -- unlike the two W&C94 options, whose
mismatch with total_seismic_moment() is documented in
compute_scaling_result()'s warnings below).

Same date, later: Keanu supplied the actual Leonard (2010) PDF
(BSSA 100(5A), 1971-1988), which is what the first attempt lacked.
Table 4 (C1, C2 per category) and equations (6), (8), (12) were
transcribed directly from it (page 1980-1981) -- see the section below
for the exact values and page references. Unlike Thingbaijam/Strasser
above, Leonard's own equations are self-consistent by construction (mu
cancels out of mu*L*W*D-bar exactly, for any mu) -- no moment-balance
patching needed.

2026-08-17c: Keanu supplied Leonard (2014) (BSSA 104(6), 2953-2965,
"...Extension to Stable Continental Strike-Slip Faults"), whose Table 2
adds an SCR strike-slip category (previously unavailable -- the prior
session's implementation warned and fell back to SCR dip-slip whenever
strike-slip was requested against the SCR relation) and whose Table 3
gives the small-earthquake and width-limited-strike-slip regime
breakpoints that were previously skipped as "not fully specified." Both
gaps are now closed -- see the Leonard section below for the full
trilinear (small/main/width-limited) implementation, the derivation
used for the small/width-limited regime formulas (verified against
Table 3's own published constants), and the one exception (SCR
strike-slip, whose C1/C2 don't reconstruct Table 3 as cleanly as the
other three categories -- handled by using Table 3 directly instead,
fixed at the paper's reference mu rather than parametrically).

References
----------
Wells, D. L., & Coppersmith, K. J. (1994). New empirical relationships
    among magnitude, rupture length, rupture width, rupture area, and
    surface displacement. BSSA, 84(4), 974-1002.
Leonard, M. (2010). Earthquake fault scaling: Self-consistent relating
    of rupture length, width, average displacement, and moment release.
    BSSA, 100(5A), 1971-1988.
Leonard, M. (2014). Self-consistent earthquake fault-scaling relations:
    Update and extension to stable continental strike-slip faults.
    BSSA, 104(6), 2953-2965.
Thingbaijam, K. K. S., Mai, P. M., & Goda, K. (2017). New empirical
    earthquake source-scaling laws. BSSA, 107(5), 2225-2246.
Strasser, F. O., Arango, M. C., & Bommer, J. J. (2010). Scaling of the
    source dimensions of interface and intraslab subduction-zone
    earthquakes with moment magnitude. SRL, 81(6), 941-950.
"""

import math

FAULT_STYLES = ["strike-slip", "reverse", "normal", "all"]


# ─── Wells & Coppersmith (1994), Table 2A/2B ─────────────────────────────────
# Regression: log10(Y) = a + b * Mw   (Y = SRL, RLD, RW, RA, or MD/AD as noted)
# Coefficients keyed by (relation, style): (a, b)

_WC94_SRL = {  # Surface Rupture Length (km) vs Mw
    "strike-slip": (-3.55, 0.74),
    "reverse":     (-2.86, 0.63),
    "normal":      (-2.01, 0.50),
    "all":         (-3.22, 0.69),
}
_WC94_RLD = {  # Subsurface Rupture Length (km) vs Mw
    "strike-slip": (-2.57, 0.62),
    "reverse":     (-2.42, 0.58),
    "normal":      (-1.88, 0.50),
    "all":         (-2.44, 0.59),
}
_WC94_RW = {   # Downdip Rupture Width (km) vs Mw
    "strike-slip": (-0.76, 0.27),
    "reverse":     (-1.61, 0.41),
    "normal":      (-1.14, 0.35),
    "all":         (-1.01, 0.32),
}
_WC94_RA = {   # Rupture Area (km^2) vs Mw
    "strike-slip": (-3.42, 0.90),
    "reverse":     (-3.99, 0.98),
    "normal":      (-2.87, 0.82),
    "all":         (-3.49, 0.91),
}
_WC94_MD = {   # Maximum Displacement (m) vs Mw
    "strike-slip": (-7.03, 1.03),
    "reverse":     (-1.84, 0.29),
    "normal":      (-5.90, 0.89),
    "all":         (-5.46, 0.82),
}
_WC94_AD = {   # Average Displacement (m) vs Mw
    "strike-slip": (-6.32, 0.90),
    "reverse":     (-0.74, 0.08),
    "normal":      (-4.45, 0.63),
    "all":         (-4.80, 0.69),
}

# Inverse relations: Mw = a + b*log10(Y), used when solving M from a
# user-supplied length or area (Table 2B of Wells & Coppersmith 1994).
_WC94_M_FROM_SRL = {
    "strike-slip": (5.16, 1.12),
    "reverse":     (5.00, 1.22),
    "normal":      (4.86, 1.32),
    "all":         (5.08, 1.16),
}
_WC94_M_FROM_RA = {
    "strike-slip": (3.98, 1.02),
    "reverse":     (4.33, 0.90),
    "normal":      (3.93, 1.02),
    "all":         (4.07, 0.98),
}


def wells_coppersmith_1994(mw=None, style="all", from_area_km2=None,
                           from_length_km=None, mu_pa=None, **kwargs):
    """
    Wells & Coppersmith (1994) scaling relations.

    Provide exactly one of `mw`, `from_area_km2`, or `from_length_km`.

    Returns a dict with keys:
      mw, style, surface_rupture_length_km, subsurface_rupture_length_km,
      width_km, area_km2, max_slip_m, avg_slip_m, length_km
    (length_km is an alias for subsurface_rupture_length_km, provided for
    convenient use as the along-strike modeling length.)
    """
    style = style if style in FAULT_STYLES else "all"

    if mw is not None:
        a_srl, b_srl = _WC94_SRL[style]
        a_rld, b_rld = _WC94_RLD[style]
        a_rw, b_rw = _WC94_RW[style]
        a_ra, b_ra = _WC94_RA[style]
        a_md, b_md = _WC94_MD[style]
        a_ad, b_ad = _WC94_AD[style]

        srl = 10 ** (a_srl + b_srl * mw)
        rld = 10 ** (a_rld + b_rld * mw)
        rw = 10 ** (a_rw + b_rw * mw)
        ra = 10 ** (a_ra + b_ra * mw)
        md = 10 ** (a_md + b_md * mw)
        ad = 10 ** (a_ad + b_ad * mw)

        return dict(mw=mw, style=style,
                   surface_rupture_length_km=srl,
                   subsurface_rupture_length_km=rld,
                   width_km=rw, area_km2=ra,
                   max_slip_m=md, avg_slip_m=ad,
                   length_km=rld)

    if from_length_km is not None:
        a, b = _WC94_M_FROM_SRL[style]
        mw_est = a + b * math.log10(from_length_km)
        return wells_coppersmith_1994(mw=mw_est, style=style)

    if from_area_km2 is not None:
        a, b = _WC94_M_FROM_RA[style]
        mw_est = a + b * math.log10(from_area_km2)
        return wells_coppersmith_1994(mw=mw_est, style=style)

    raise ValueError("Provide exactly one of mw, from_area_km2, from_length_km.")


# ─── Coulomb 3.4 GUI-compatible variant ──────────────────────────────────────
# Coulomb 3.4's own "Wells & Coppersmith 1994" fault-dimension calculator
# (coulomb.m, BCF_EmpricalLaw_CalcButton_callback -> nested wells_coppersmith())
# does NOT use the forward Table 2A relations above. It solves the PAPER'S
# INVERSE regression (Table 2B, "Mw from L/W") backward for L and W instead:
#     wells_coppersmith(a, b, m) = 10 ** ((m - a) / b)
# Table 2A and Table 2B were fit as SEPARATE least-squares regressions in
# the original paper (not algebraic inverses of one another), so this
# produces systematically different L/W than the standard forward usage
# for the same Mw/style. Confirmed against Coulomb 3.4's own GUI output:
# Mw 7.0, reverse -> L=48.37 km, W=22.32 km, exact match.
#
# Coefficients copied verbatim from coulomb.m (both the live 4-style
# switch near BCF_EmpricalLaw_CalcButton_callback and the duplicate near
# the fault-classification table agree on these values).
_COULOMB_INVERSE_L = {  # (a, b) such that L_km = 10 ** ((Mw - a) / b)
    "all":         (4.38, 1.49),
    "strike-slip": (4.33, 1.49),
    "reverse":     (4.49, 1.49),
    "normal":      (4.34, 1.54),
}
_COULOMB_INVERSE_W = {  # (a, b) such that W_km = 10 ** ((Mw - a) / b)
    "all":         (4.06, 2.25),
    "strike-slip": (3.80, 2.59),
    "reverse":     (4.37, 1.95),
    "normal":      (4.04, 2.11),
}

# Coulomb also does NOT get slip from the W&C94 average-displacement (AD)
# table at all.
#
# CORRECTED 2026-08-10 (was previously wrong -- see below): this used to
# invert coulomb.m's seis_moment() (M0 = mu*L*W*D, using the model's real
# elastic parameters and a 16.05 dyne-cm exponent constant). That was a
# plausible-looking guess, not the actual code path. Grepping coulomb.m
# for every seis_moment/wells_coppersmith call site found that the fault-
# elements "Calc." button that actually produces the GUI's displayed slip
# (BCF_FE_Calcbutton_callback, the callback behind the "Build input from
# CMT" dialog's fault-table Calc. button) uses a DIFFERENT, hardcoded
# formula that does not touch the model's elastic parameters at all:
#
#     shr = 3.4e+11;                                          % hardcoded, NOT E1/PR1-derived
#     mo  = 10 ^ (1.5*Mw + 9.1) * 1.0e+7;                      % different constant, different units
#     slip = mo / (shr * length_km * width_km * 1.0e+10);      % length/width used directly in km
#     rlslip = -cos(rake) * slip / 100;
#     rvslip =  sin(rake) * slip / 100;
#
# Verified against Coulomb's own GUI screenshot for Mw 7.0 reverse,
# L=48.37 km, W=22.32 km, rake=89.0 deg: this reproduces rt-lat=-0.0189 m
# and reverse=1.0844 m to 5 significant figures -- an exact match, not an
# approximation. `seis_moment()` (the 16.05-constant, real-elastic-params
# formula) is genuine live code, but it's a FORWARD reporting function
# (slip -> moment, called after a fault is already built, to print
# "Total seismic moment = ... dyne cm (Mw = ...)" -- see
# okada_engine.total_seismic_moment(), added 2026-08-10 to mirror it) --
# it is not what estimates slip from Mw in this dialog. The two are
# easily confused because both live in the same file and both cite a
# similar-looking Kanamori-style Mw<->M0 relation with different
# constants; they are not the same formula and must not be conflated.
#
# NOTE ON UNITS: because slip here is genuinely independent of the
# model's elastic parameters, `mu_pa` is accepted for call-signature
# compatibility with the standard relation but is IGNORED -- see the
# docstring below.
_COULOMB_FE_SHEAR_MODULUS = 3.4e11  # hardcoded constant from coulomb.m; NOT a real shear modulus in Pa/bar -- see formula above, only meaningful combined with the 9.1/1e7/1e10 factors together
_COULOMB_FE_MW_M0_CONST = 9.1       # dyne-cm-ish exponent constant from coulomb.m's BCF_FE_Calcbutton_callback (distinct from seis_moment()'s 16.05 -- different formula entirely, not a rounding variant of it)

# Kanamori dyne-cm constant, matching okada_engine.total_seismic_moment()'s
# own -10.7 convention exactly (mw = (2/3)*log10(amo_dynecm) - 10.7  <=>
# amo_dynecm = 10**(1.5*mw + 16.05)). Used by both the Leonard (2010) and
# Thingbaijam/Strasser relations below, so it's defined once here.
_KANAMORI_DYNECM_EXPONENT = 16.05


def wells_coppersmith_1994_coulomb_compatible(mw=None, style="reverse",
                                              mu_pa=None, **kwargs):
    """
    Reproduces Coulomb 3.4's OWN "Wells & Coppersmith 1994" GUI
    calculator (coulomb.m) -- deliberately NOT the standard forward W&C94
    usage that `wells_coppersmith_1994()` implements above. Use this one
    when you want fault dimensions AND slip that match Coulomb 3.4's GUI
    numbers exactly; use the other one for the textbook/conventional
    forward relation. See the comments above this function for why they
    disagree, and for the exact formula (verified against Coulomb's GUI
    to 5 significant figures, both for L/W and for slip).

    mu_pa : accepted for call-signature compatibility with
        `wells_coppersmith_1994()`, but IGNORED. Coulomb's actual GUI
        slip calculation for this dialog does not read the model's
        elastic parameters (E1/PR1) at all -- it uses a hardcoded
        constant baked into coulomb.m. Passing a different mu_pa here
        has zero effect on the returned slip, by design: that mirrors
        Coulomb's own (arguably surprising, but verified) behavior. If
        you want a slip estimate that responds to your actual elastic
        parameters, use `wells_coppersmith_1994()`'s AD-table slip
        instead -- this function's slip is a faithful copy of Coulomb's
        specific dialog, not a general moment-balance estimate.
    """
    style = style if style in _COULOMB_INVERSE_L else "all"
    al, bl = _COULOMB_INVERSE_L[style]
    aw, bw = _COULOMB_INVERSE_W[style]
    length_km = 10 ** ((mw - al) / bl)
    width_km = 10 ** ((mw - aw) / bw)

    mo = 10 ** (1.5 * mw + _COULOMB_FE_MW_M0_CONST) * 1.0e7
    # coulomb.m's own "slip" intermediate here is NOT in meters -- it only
    # becomes meters after the /100 that coulomb.m applies when splitting
    # it into rlslip/rvslip by rake. Applying that same /100 here so this
    # function's avg_slip_m is directly comparable (meters) to the other
    # relation's avg_slip_m, exactly as Coulomb's GUI displays it.
    avg_slip_m = mo / (_COULOMB_FE_SHEAR_MODULUS * length_km * width_km * 1.0e10) / 100.0

    return dict(mw=mw, style=style,
               length_km=length_km, width_km=width_km,
               area_km2=length_km * width_km,
               avg_slip_m=avg_slip_m,
               coulomb_compatible=True)


# ─── Leonard (2010/2014) ──────────────────────────────────────────────────
# Coefficients (C1, C2) transcribed directly from PDFs Keanu supplied --
# NOT reconstructed from memory or a secondhand citation, which is why the
# earlier attempt at this relation (see module docstring) was pulled.
#   Leonard, M. (2010). BSSA 100(5A), 1971-1988. Table 4, p.1981.
#   Leonard, M. (2014). BSSA 104(6), 2953-2965 ("...Extension to Stable
#     Continental Strike-Slip Faults"). Table 2 (p.2957) confirms the
#     2010 C1/C2 for the three original categories (unchanged to the
#     precision given) and adds a fourth: SCR strike-slip -- the gap
#     flagged as unimplemented in the 2026-08-17 addendum.
# C1 is in m^(1/3) (paper's own units); C2 is dimensionless.
#
#   Category                C1 (m^1/3)   C2 (x1e-5)
#   Interplate Dip-Slip     17.5         3.8
#   Interplate Strike-Slip  15.0         3.7
#   SCR Dip-Slip            13.5         7.3
#   SCR Strike-Slip         11.3         5.8    (2014 only; see caveat below)
#
# Leonard's governing equations (SI: L, W, D-bar in meters; M0 in N*m; mu
# in Pa), self-consistent by construction (mu cancels out of
# mu*L*W*D-bar exactly, algebraically):
#   MAIN regime (beta=2/3, the paper's eq. 6, 8, 12):
#     W = C1 * L^(2/3)
#     log10(M0)    = 5/2*log10(L) + 3/2*log10(C1) + log10(C2*mu)
#     log10(D-bar) = 5/6*log10(L) + 1/2*log10(C1) + log10(C2)
#   SMALL regime (beta=1, crack-like, W=L, below the L=W transition
#   length L1=C1^3 -- this breakpoint is exact, not fit: it's just where
#   the main-regime formula W=C1*L^(2/3) evaluates to W=L. Not itself in
#   the 2010 paper as a usable formula, only as "faults act as cracks
#   with L^3 scaling" -- derived here by substituting W=L, D=C2*L
#   (Leonard's own displacement model D=C2*sqrt(A), with A=L*W=L^2 in
#   this regime) into M0=mu*L*W*D, giving M0=mu*C2*L^3):
#     W = L; D-bar = C2*L
#     log10(M0) = 3*log10(L) + log10(C2*mu)
#   WIDTH-LIMITED regime (beta=0, strike-slip only, above the second
#   corner length L2 -- also derived, not directly published as a
#   mu-parametric formula: width saturates at Wmax = C1*L2^(2/3) [i.e.
#   the main-regime W evaluated at the corner], and D=C2*sqrt(L*Wmax)
#   per the same displacement model, giving M0=mu*C2*Wmax^1.5*L^1.5):
#     W = Wmax = C1 * L2^(2/3);  D-bar = C2*sqrt(L*Wmax)
#     log10(M0) = 3/2*log10(L) + log10(mu*C2*Wmax^1.5)
#
# VERIFICATION of the small/width-limited derivations above: their
# predicted 'a' intercept (at the paper's own reference mu=3.3e10) was
# checked against Leonard (2014) Table 3's independently-published (a,b)
# regression constants for the same regimes. Match to <0.001 in log10(M0)
# (negligible, rounding-level) for Interplate Dip-Slip, Interplate
# Strike-Slip, and SCR Dip-Slip in every regime tested -- confirming this
# derivation is correct and these three categories can safely use the
# fully parametric (mu-adjustable) trilinear model below.
#
# SCR Strike-Slip is the exception: the same check comes out ~0.09-0.11
# off in log10(M0) (~0.06-0.07 Mw equivalent) in EVERY regime, including
# the main one -- meaning Table 2's published C1=11.3/C2=5.8e-5 do not
# cleanly reproduce Table 3's own tabulated (a,b) constants for this
# specific category, unlike the other three. This is very likely because
# SCR strike-slip was fit from only 9-10 earthquakes (Leonard 2014's own
# words) with the corner points adjusted somewhat independently rather
# than purely from C1^3/Wmax theory. Given a real, non-negligible
# mismatch between two numbers in the same paper, this module trusts the
# more directly empirical one: Table 3's own tabulated (a,b,range)
# triplets, taken and applied AS PUBLISHED, fixed at the paper's own
# reference mu=3.3e10 -- NOT mu-adjustable, unlike the other three
# categories. See leonard_2010_2014_scr_strike_slip's docstring.
_LEONARD10_CATEGORIES = {
    "interplate_dip_slip":    dict(C1=17.5, C2=3.8e-5, length_corner2_m=None),
    "interplate_strike_slip": dict(C1=15.0, C2=3.7e-5, length_corner2_m=40000.0),
    "scr_dip_slip":           dict(C1=13.5, C2=7.3e-5, length_corner2_m=None),
}

# Leonard (2014) Table 3 SCR strike-slip, taken directly and used as
# published (not re-derived) -- see the long comment above for why. Each
# entry is (b, a, L_min_m, L_max_m) for log10(Y) = a + b*log10(L).
_SCR_SS_TABLE3_M0_VS_L = [(3.0, 6.370, 0, 1600), (2.5, 7.972, 1600, 60000), (1.5, 12.750, 60000, None)]
_SCR_SS_TABLE3_W_VS_L  = [(1.0, 0.0, 0, 1600), (0.667, 1.068, 1600, 70000), (0.0, 4.298, 70000, None)]
_SCR_SS_TABLE3_D_VS_L  = [(1.0, -4.149, 0, 1600), (0.833, -3.615, 1600, 60000), (0.5, -2.022, 60000, None)]
_LEONARD_TABLE3_REFERENCE_MU_PA = 3.3e10  # baked into the intercepts above


def _leonard_2010_2014_solve(mw, category, mu_pa):
    """
    Trilinear solve for the three categories whose C1/C2 were verified to
    reproduce Leonard (2014) Table 3's own regression constants closely
    (see the long comment above) -- everything except SCR strike-slip.
    """
    p = _LEONARD10_CATEGORIES[category]
    c1, c2, length_corner2_m = p["C1"], p["C2"], p["length_corner2_m"]
    amo_dynecm = 10 ** (1.5 * mw + _KANAMORI_DYNECM_EXPONENT)
    m0_nm = amo_dynecm * 1.0e-7  # dyne-cm -> N*m

    length_corner1_m = c1 ** 3  # exact: where main-regime W=C1*L^(2/3) equals L

    # Small regime (beta=1, crack-like): M0 = mu*C2*L^3
    length_m = (m0_nm / (mu_pa * c2)) ** (1.0 / 3.0)
    regime = "small"
    if length_m > length_corner1_m:
        # Main regime (beta=2/3): eq. 8
        log_length_m = (2.0 / 5.0) * (
            math.log10(m0_nm) - 1.5 * math.log10(c1) - math.log10(c2 * mu_pa))
        length_m = 10 ** log_length_m
        regime = "main"
        if length_corner2_m is not None and length_m > length_corner2_m:
            # Width-limited regime (beta=0, strike-slip only)
            width_max_m = c1 * (length_corner2_m ** (2.0 / 3.0))
            length_m = (m0_nm / (mu_pa * c2 * width_max_m ** 1.5)) ** (2.0 / 3.0)
            regime = "width-limited"

    if regime == "small":
        width_m = length_m
        avg_slip_m = c2 * length_m
    elif regime == "main":
        width_m = c1 * (length_m ** (2.0 / 3.0))
        avg_slip_m = c2 * (c1 ** 0.5) * (length_m ** (5.0 / 6.0))
    else:  # width-limited
        width_m = c1 * (length_corner2_m ** (2.0 / 3.0))
        avg_slip_m = c2 * math.sqrt(length_m * width_m)

    return dict(length_km=length_m / 1000.0, width_km=width_m / 1000.0,
               avg_slip_m=avg_slip_m, regime=regime, self_consistent=True)


def _leonard_table3_eval(table, length_m):
    for b, a, lo, hi in table:
        if length_m >= lo and (hi is None or length_m <= hi):
            return a + b * math.log10(length_m)
    b, a, lo, hi = table[-1]
    return a + b * math.log10(length_m)


def _leonard_table3_invert_m0_vs_l(table, m0_nm):
    log_m0 = math.log10(m0_nm)
    for b, a, lo, hi in table:
        length_m = 10 ** ((log_m0 - a) / b)
        if length_m >= lo * 0.999 and (hi is None or length_m <= hi * 1.001):
            return length_m
    b, a, lo, hi = table[-1]
    return 10 ** ((log_m0 - a) / b)


def leonard_2010_2014_scr_strike_slip(mw, **kwargs):
    """
    Leonard (2014) SCR strike-slip, Table 3 constants applied exactly as
    published -- NOT the mu-adjustable parametric model used for the
    other three categories (see the long comment above _LEONARD10_
    CATEGORIES for why: the parametric reconstruction from Table 2's
    C1/C2 doesn't reproduce this specific category's own Table 3 numbers
    as closely as it does for the other three, likely a fitting artifact
    of this being the most data-poor category in the paper -- only 9-10
    earthquakes). Slip is therefore fixed at the paper's own reference
    mu=3.3e10 Pa regardless of this model's actual Elastic Parameters --
    same caveat as the Coulomb-GUI-compatible W&C94 relation.
    """
    amo_dynecm = 10 ** (1.5 * mw + _KANAMORI_DYNECM_EXPONENT)
    m0_nm = amo_dynecm * 1.0e-7
    length_m = _leonard_table3_invert_m0_vs_l(_SCR_SS_TABLE3_M0_VS_L, m0_nm)
    width_m = 10 ** _leonard_table3_eval(_SCR_SS_TABLE3_W_VS_L, length_m)
    avg_slip_m = 10 ** _leonard_table3_eval(_SCR_SS_TABLE3_D_VS_L, length_m)
    return dict(length_km=length_m / 1000.0, width_km=width_m / 1000.0,
               avg_slip_m=avg_slip_m, fixed_reference_mu=True)


def leonard_2010(mw=None, style="all", mu_pa=32e9, **kwargs):
    """
    Leonard (2010/2014) interplate (non-SCR) relations, full trilinear
    model (small/main/width-limited regimes -- see module comments).
    Strike-slip uses the Interplate Strike-Slip category; reverse and
    normal both map to the single Interplate Dip-Slip category (the
    paper does not distinguish normal from reverse). style="all"
    averages the two categories' (C1, C2) pairs (width-limited regime
    disabled for this case -- mixing a dip-slip category that has none
    with a strike-slip category that does doesn't have a coherent
    meaning) and is flagged approximate.
    """
    if style == "strike-slip":
        category = "interplate_strike_slip"
        approximate = False
    elif style in ("reverse", "normal"):
        category = "interplate_dip_slip"
        approximate = False
    else:
        c1 = (_LEONARD10_CATEGORIES["interplate_dip_slip"]["C1"] +
              _LEONARD10_CATEGORIES["interplate_strike_slip"]["C1"]) / 2.0
        c2 = (_LEONARD10_CATEGORIES["interplate_dip_slip"]["C2"] +
              _LEONARD10_CATEGORIES["interplate_strike_slip"]["C2"]) / 2.0
        _LEONARD10_CATEGORIES["_all_tmp"] = dict(
            C1=c1, C2=c2, length_corner2_m=None)
        category = "_all_tmp"
        approximate = True

    solved = _leonard_2010_2014_solve(mw, category, mu_pa)
    if category == "_all_tmp":
        del _LEONARD10_CATEGORIES["_all_tmp"]

    result = dict(mw=mw, style=style, length_km=solved["length_km"],
               width_km=solved["width_km"], avg_slip_m=solved["avg_slip_m"],
               area_km2=solved["length_km"] * solved["width_km"],
               self_consistent=True, regime=solved["regime"])
    if approximate:
        result["approximate"] = True
    return result


def leonard_2010_scr(mw=None, style="all", mu_pa=32e9, **kwargs):
    """
    Leonard (2010/2014) stable continental region (SCR) relations.
    Dip-slip (reverse/normal/all) uses the fully parametric, mu-
    adjustable trilinear model (main + small regime; SCR dip-slip has no
    width-limited regime, same as interplate dip-slip). Strike-slip uses
    Leonard (2014)'s newly-added SCR strike-slip category -- see
    leonard_2010_2014_scr_strike_slip()'s docstring for why THAT one is
    fixed at the paper's reference mu rather than adjustable.
    """
    if style == "strike-slip":
        solved = leonard_2010_2014_scr_strike_slip(mw)
        result = dict(mw=mw, style="SCR strike-slip",
                   length_km=solved["length_km"], width_km=solved["width_km"],
                   avg_slip_m=solved["avg_slip_m"],
                   area_km2=solved["length_km"] * solved["width_km"],
                   fixed_reference_mu=True)
        return result

    solved = _leonard_2010_2014_solve(mw, "scr_dip_slip", mu_pa)
    result = dict(mw=mw, style="SCR dip-slip", length_km=solved["length_km"],
               width_km=solved["width_km"], avg_slip_m=solved["avg_slip_m"],
               area_km2=solved["length_km"] * solved["width_km"],
               self_consistent=True, regime=solved["regime"])
    return result


# ─── Thingbaijam, Mai & Goda (2017), Table 1 ─────────────────────────────────
# Regression: log10(Y) = a + b*Mw  (Y = L or W, km). Coefficients verified
# against openquake.hazardlib.scalerel.thingbaijam2017 (github.com/gem/
# oq-engine, commit current as of 2026-08-17).
_TMG17_L = {
    "strike-slip": (-2.943, 0.681),
    "reverse":     (-2.693, 0.614),
    "normal":      (-1.722, 0.485),
}
_TMG17_W = {
    "strike-slip": (-0.543, 0.261),
    "reverse":     (-1.669, 0.435),
    "normal":      (-0.829, 0.323),
}
_TMG17_INTERFACE_L = (-2.412, 0.583)
_TMG17_INTERFACE_W = (-0.880, 0.366)

# ─── Strasser, Arango & Bommer (2010), subduction interface ─────────────────
# Same source/verification method as Thingbaijam above. The paper's
# intraslab class publishes only a magnitude-area relation (no separate
# length/width split), so only the interface (megathrust) relation is
# implemented here -- adding intraslab would require inventing an L/W
# split the paper doesn't provide, which is exactly the kind of
# unverifiable guess this module avoids (see Leonard note above).
_STRASSER10_INTERFACE_L = (-2.477, 0.585)
_STRASSER10_INTERFACE_W = (-0.882, 0.351)
# (_KANAMORI_DYNECM_EXPONENT is defined once, above the Leonard section,
# since both Leonard and this section need it.)


def _moment_balanced_slip_m(mw, length_km, width_km, mu_pa):
    """
    Solve avg_slip_m so that mu_pa*length_km*width_km*avg_slip_m (in the
    same dyne-cm/1e13 convention as okada_engine.total_seismic_moment())
    exactly reproduces the given Mw. Used by relations (Thingbaijam 2017,
    Strasser 2010) that publish L/W regressions but no independently-
    verifiable slip regression of their own -- see module docstring.
    """
    amo_dynecm = 10 ** (1.5 * mw + _KANAMORI_DYNECM_EXPONENT)
    return amo_dynecm / (mu_pa * length_km * width_km * 1.0e13)


def thingbaijam_2017(mw=None, style="all", mu_pa=32e9, **kwargs):
    """
    Thingbaijam, Mai & Goda (2017) crustal length/width relations
    (strike-slip, normal, reverse), with slip solved by moment balance
    against mu_pa rather than a paper-published AD regression -- see
    module docstring for why, and _moment_balanced_slip_m()'s docstring
    for the formula.

    style="all" is not published by the paper; this averages the three
    styles' (a, b) coefficient pairs, matching the pattern already used
    elsewhere in this codebase for an unpublished "all" case (see
    Leonard 2014's own SCR/Interplate rake=None handling for the same
    approach) and is flagged with an "approximate" warning.
    """
    if style not in _TMG17_L:
        a_l = sum(a for a, b in _TMG17_L.values()) / len(_TMG17_L)
        b_l = sum(b for a, b in _TMG17_L.values()) / len(_TMG17_L)
        a_w = sum(a for a, b in _TMG17_W.values()) / len(_TMG17_W)
        b_w = sum(b for a, b in _TMG17_W.values()) / len(_TMG17_W)
        approximate = True
        style = "all"
    else:
        a_l, b_l = _TMG17_L[style]
        a_w, b_w = _TMG17_W[style]
        approximate = False

    length_km = 10 ** (a_l + b_l * mw)
    width_km = 10 ** (a_w + b_w * mw)
    avg_slip_m = _moment_balanced_slip_m(mw, length_km, width_km, mu_pa)

    result = dict(mw=mw, style=style, length_km=length_km, width_km=width_km,
               area_km2=length_km * width_km, avg_slip_m=avg_slip_m,
               moment_balanced=True)
    if approximate:
        result["approximate"] = True
    return result


def thingbaijam_2017_interface(mw=None, style=None, mu_pa=32e9, **kwargs):
    """
    Thingbaijam, Mai & Goda (2017) subduction interface (megathrust)
    length/width relation. Rake/style is ignored -- the paper fits one
    relation for all interface events, matching openquake.hazardlib's
    own ThingbaijamInterface (rake parameter accepted but unused).
    Slip is moment-balanced -- see module docstring.
    """
    length_km = 10 ** (_TMG17_INTERFACE_L[0] + _TMG17_INTERFACE_L[1] * mw)
    width_km = 10 ** (_TMG17_INTERFACE_W[0] + _TMG17_INTERFACE_W[1] * mw)
    avg_slip_m = _moment_balanced_slip_m(mw, length_km, width_km, mu_pa)
    return dict(mw=mw, style="interface", length_km=length_km,
               width_km=width_km, area_km2=length_km * width_km,
               avg_slip_m=avg_slip_m, moment_balanced=True)


def strasser_2010_interface(mw=None, style=None, mu_pa=32e9, **kwargs):
    """
    Strasser, Arango & Bommer (2010) subduction interface (megathrust)
    length/width relation. Rake/style is ignored, matching the paper and
    openquake.hazardlib's own StrasserInterface. Slip is moment-balanced
    -- see module docstring.
    """
    length_km = 10 ** (_STRASSER10_INTERFACE_L[0] + _STRASSER10_INTERFACE_L[1] * mw)
    width_km = 10 ** (_STRASSER10_INTERFACE_W[0] + _STRASSER10_INTERFACE_W[1] * mw)
    avg_slip_m = _moment_balanced_slip_m(mw, length_km, width_km, mu_pa)
    return dict(mw=mw, style="interface", length_km=length_km,
               width_km=width_km, area_km2=length_km * width_km,
               avg_slip_m=avg_slip_m, moment_balanced=True)


SCALING_RELATIONS = {
    "Wells & Coppersmith (1994)": wells_coppersmith_1994,
    "Wells & Coppersmith (1994) — Coulomb 3.4 GUI-compatible": wells_coppersmith_1994_coulomb_compatible,
    "Leonard (2010/2014) — interplate": leonard_2010,
    "Leonard (2010/2014) — stable continental region (SCR)": leonard_2010_scr,
    "Thingbaijam, Mai & Goda (2017) — crustal": thingbaijam_2017,
    "Thingbaijam, Mai & Goda (2017) — subduction interface": thingbaijam_2017_interface,
    "Strasser, Arango & Bommer (2010) — subduction interface": strasser_2010_interface,
}

# Relation names whose slip actually depends on mu_pa. The two Leonard
# entries are genuinely SELF-consistent (mu cancels out of
# mu*L*W*D-bar algebraically -- see their section above) rather than
# moment-BALANCED after the fact like Thingbaijam/Strasser -- EXCEPT for
# the SCR strike-slip style specifically, which uses Leonard (2014)
# Table 3 fixed at the paper's own reference mu (see
# leonard_2010_2014_scr_strike_slip's docstring). Both entries still
# take mu_pa as an input for their other styles, so the field needs to
# be visible for all five names below. ui.scaling_dialog uses this set
# to decide whether to show its shear-modulus field -- unlike the two
# W&C94 options, where mu_pa is accepted for call-signature
# compatibility only and ignored.
MOMENT_BALANCED_RELATION_NAMES = {
    "Leonard (2010/2014) — interplate",
    "Leonard (2010/2014) — stable continental region (SCR)",
    "Thingbaijam, Mai & Goda (2017) — crustal",
    "Thingbaijam, Mai & Goda (2017) — subduction interface",
    "Strasser, Arango & Bommer (2010) — subduction interface",
}


def compute_scaling_result(relation_name, style, mw, rake_deg, mu_pa=32e9):
    """
    Run a named relation (see SCALING_RELATIONS) at a given Mw/style, then
    decompose the resulting scalar average slip into right-lateral +
    reverse components using rake_deg. This is exactly the calculation
    ui.scaling_dialog.ScalingRelationsDialog._calculate() performs for its
    "Apply to fault table" button -- factored out here so OTHER callers
    (e.g. core.focal_mechanism.build_source_fault_row(), which turns an
    imported focal-mechanism event into a source-fault-table row) get
    numerically identical length/width/slip without re-deriving the
    rake decomposition in a second place. See scaling_dialog.py's
    _calculate() for the UI-facing counterpart (result/warning labels);
    both now call this same function.

    mu_pa is accepted for call-signature compatibility with every
    relation in SCALING_RELATIONS -- some (e.g. the Coulomb-GUI-compatible
    variant) ignore it entirely; see that relation's own docstring.

    Returns a dict:
      length_km, width_km       -- None if the relation didn't produce them
      rt_lateral_slip_m,
      reverse_slip_m            -- None if the relation returned no slip
      raw                       -- the relation function's own full dict
      warnings                  -- list[str], plain-text (no HTML/emoji --
                                    UI callers format those themselves)
    Raises KeyError if relation_name isn't in SCALING_RELATIONS.
    """
    relation_fn = SCALING_RELATIONS[relation_name]
    result = relation_fn(mw=mw, style=style, mu_pa=mu_pa)

    length = result.get("length_km") or result.get("subsurface_rupture_length_km")
    width = result.get("width_km")
    slip = result.get("avg_slip_m")

    # Decompose the scalar slip magnitude into right-lateral + reverse
    # components using the rake, matching Coulomb's own slip convention.
    # Aki-Richards rake=0 is left-lateral; Coulomb's right-lateral is the
    # negative of that (consistent with FaultParameters.rt_lateral_slip).
    rt_lat = reverse = None
    if slip is not None:
        rake_rad = math.radians(rake_deg)
        u1_aki = slip * math.cos(rake_rad)   # Aki-Richards strike-slip component
        reverse = slip * math.sin(rake_rad)  # dip-slip component (reverse if +)
        rt_lat = -u1_aki                     # Coulomb rt-lateral = -Aki U1
        # Snap floating-point residuals near zero (e.g. cos(90 deg) is
        # ~6e-17, not exactly 0) so callers never see a spurious
        # scientific-notation value for what should read as exactly 0.
        if abs(rt_lat) < 1e-9 * max(abs(slip), 1.0):
            rt_lat = 0.0
        if abs(reverse) < 1e-9 * max(abs(slip), 1.0):
            reverse = 0.0

    warnings = []
    if result.get("approximate"):
        warnings.append(
            "Style 'all' is not published by this relation's source "
            "paper -- these coefficients are an average of the "
            "individual styles' own (verified) coefficients, not "
            "themselves an independently verified regression. Prefer "
            "picking a specific style if you can.")
    if style == "all":
        warnings.append(
            "Style 'all' has no single representative rake — the slip "
            "decomposition used whatever rake_deg was supplied by the "
            "caller.")
    if result.get("coulomb_compatible"):
        warnings.append(
            "Length, width, and slip are verified exact matches to "
            "Coulomb 3.4's GUI for this Mw/style. Slip does NOT depend on "
            "elastic parameters (mu_pa is ignored by this relation).")
        warnings.append(
            "This slip was solved using coulomb.m's own hardcoded internal "
            "constant, not your model's actual elastic mu. If you check "
            "'Total seismic moment' after applying this row (which DOES "
            "use your real mu), expect it to read a bit higher than the "
            "Mw entered here -- a small, consistent offset (~+0.02 Mw at "
            "mu=32 GPa), not an error.")
    elif result.get("self_consistent"):
        regime_note = {
            "small": " (this Mw solved into Leonard's small/crack-like "
                     "regime, W=L -- unusual for typical fault-modeling "
                     "magnitudes, double-check this is the intended style/Mw)",
            "main": "",
            "width-limited": " (this Mw solved into Leonard's width-"
                     "limited regime -- width is capped near the "
                     "seismogenic-depth limit and no longer grows with "
                     "length)",
        }.get(result.get("regime"), "")
        warnings.append(
            "Length, width, and slip are all solved from the SAME "
            "self-consistent model (Leonard's own design goal), using "
            "your current Elastic Parameters mu -- unlike the other "
            "relations in this dialog, no moment-balance patching was "
            "needed to make this one agree with itself. 'Total seismic "
            "moment' after applying this row should read the entered "
            "Mw exactly" + regime_note + ".")
    elif result.get("fixed_reference_mu"):
        warnings.append(
            "Leonard (2014)'s SCR strike-slip regression is used exactly "
            "as published (Table 3), fixed at the paper's own reference "
            "mu=33 GPa -- NOT your current Elastic Parameters mu. Unlike "
            "the other Leonard entries, slip here will NOT change if you "
            "change mu_pa, and 'Total seismic moment' will only read back "
            "the entered Mw exactly if your model's mu also happens to be "
            "33 GPa (this is deliberate -- see this relation's own "
            "docstring for why its C1/C2 aren't used parametrically here "
            "the way the other three Leonard categories are).")
    elif result.get("moment_balanced"):
        warnings.append(
            "Length and width come from a published regression; slip is "
            "solved by moment balance (M0 = mu*L*W*D) against your "
            "current Elastic Parameters mu, not from a paper-published "
            "slip regression. 'Total seismic moment' after applying this "
            "row should read very close to the Mw entered here -- change "
            "mu_pa and slip will change to compensate, unlike the two "
            "Wells & Coppersmith options above.")
    else:
        warnings.append(
            "Length/width and average slip come from separate least-"
            "squares regressions in Wells & Coppersmith (1994) -- they "
            "were never fit to jointly satisfy the seismic moment equation "
            "M0 = mu*L*W*D for a fixed mu. 'Total seismic moment' after "
            "applying this row can therefore read noticeably different "
            "from the Mw entered here (usually LOWER, and worst for "
            "'reverse' style, whose AD-vs-Mw regression is nearly flat). "
            "Treat this Mw as a scaling target, not a moment guarantee.")

    return dict(length_km=length, width_km=width,
               rt_lateral_slip_m=rt_lat, reverse_slip_m=reverse,
               raw=result, warnings=warnings)

# Short, GUI-facing explanation of why two entries exist for the "same"
# citation. Keyed by the SCALING_RELATIONS name so the dialog can look
# this up directly without duplicating the text.
SCALING_RELATION_NOTES = {
    "Wells & Coppersmith (1994)":
        "Standard forward usage of the paper (Table 2A: length/width "
        "regressed on Mw). This is the conventional way the relation is "
        "applied in the literature.",
    "Wells & Coppersmith (1994) — Coulomb 3.4 GUI-compatible":
        "Matches Coulomb 3.4's own 'Wells & Coppersmith' dialog instead "
        "of the paper's standard usage. Coulomb solves the paper's "
        "INVERSE table (Mw-from-length) backward for length/width, which "
        "is not the same regression as the forward table above -- the "
        "two give different numbers for the same Mw. Slip is also "
        "computed differently: Coulomb's own fault-builder uses a fixed "
        "internal constant for this, independent of your Elastic "
        "Parameters -- so slip here does NOT change with shear modulus. "
        "Verified exact (L/W and slip) against Coulomb 3.4's GUI. Pick "
        "this if you need numbers matching Coulomb 3.4's GUI; pick the "
        "other for the conventional (elastic-parameter-independent, "
        "AD-table-based) usage.",
    "Leonard (2010/2014) — interplate":
        "Length, width, and slip solved from Leonard's own self-"
        "consistent model (BSSA 100(5A), 1971-1988) for interplate "
        "(non-SCR) earthquakes -- strike-slip uses the paper's Interplate "
        "Strike-Slip category, reverse/normal both use its single "
        "Interplate Dip-Slip category (the paper doesn't distinguish "
        "normal from reverse). Genuinely self-consistent with your "
        "current Elastic Parameters mu, not moment-balanced after the "
        "fact -- see the warning below. Full trilinear model (small "
        "crack-like / main / width-limited-for-strike-slip regimes), so "
        "unlike most other relations here there's no length range this "
        "one falls outside of.",
    "Leonard (2010/2014) — stable continental region (SCR)":
        "Same Leonard self-consistent model as the interplate entry "
        "above, but using the paper's SCR (stable continental region) "
        "categories -- generally lower stress drop than interplate. "
        "Dip-slip (reverse/normal/all) is mu-adjustable like the "
        "interplate entry. Strike-slip uses Leonard (2014)'s newer SCR "
        "strike-slip regression applied exactly as published, fixed at "
        "the paper's own reference shear modulus rather than your "
        "current Elastic Parameters -- see the warning below for what "
        "that means and why (short version: this specific category was "
        "fit from only 9-10 earthquakes and doesn't reconstruct as "
        "cleanly from C1/C2 as the other three).",
    "Thingbaijam, Mai & Goda (2017) — crustal":
        "Newer (2017) length/width regression for strike-slip, normal, "
        "and reverse crustal earthquakes, generally considered an update "
        "to Wells & Coppersmith (1994) with a larger, more modern "
        "dataset. Slip is solved by moment balance against your current "
        "Elastic Parameters (mu), not a paper-published AD table -- see "
        "the warning below for what that means for 'Total seismic "
        "moment'.",
    "Thingbaijam, Mai & Goda (2017) — subduction interface":
        "Length/width regression for subduction interface (megathrust) "
        "earthquakes specifically -- use this instead of the crustal "
        "entry for subduction-zone source faults. Rake/style has no "
        "effect on length/width (one relation for all interface "
        "events); slip is moment-balanced, same as the crustal entry.",
    "Strasser, Arango & Bommer (2010) — subduction interface":
        "An older, independently-derived alternative to the Thingbaijam "
        "subduction interface entry above -- useful for sanity-checking "
        "one against the other, since the two were fit to different "
        "(overlapping) subduction earthquake datasets. Rake/style has "
        "no effect on length/width; slip is moment-balanced.",
}
