# -*- coding: utf-8 -*-
"""
Validate a run_slip_inversion()/run_slip_inversion_group() result
against an independently-known REFERENCE slip model on the same patch
grid -- e.g. a published finite-fault/geodetic-inversion solution
(such as the GSI/Kobayashi et al. 2018 Northern Nagano dataset this
was built against), a synthetic checkerboard test, or any other
already-imported fault-patch table.

This is deliberately a SEPARATE module from slip_inversion_report.py
(that one reports the fit to the OBSERVATIONS the inversion actually
used; this one compares the SOLVED slip distribution itself against an
independent ground truth that was never part of the inversion) --
different question, different consumer.

Pure Python + numpy only (no QGIS/UI dependency), so it's testable
outside a real QGIS session, same as the rest of core/.

──────────────────────────────────────────────────────────────────
SCOPE / REQUIRED CONVENTION -- read before use
──────────────────────────────────────────────────────────────────
The reference rows (as returned by
core.fault_table_import.build_fault_rows_from_mapped_rows(), e.g. via
ui.fault_table_import_dialog.FaultTableImportDialog.imported_rows)
MUST be:
  (1) exactly n_length*n_width rows -- one per patch, no more, no
      fewer -- since this is a cell-by-cell comparison, not a
      resampling/regridding step (deliberately not attempted here, in
      the same spirit as insar_raster_import.py declining to
      resample mismatched-grid rasters -- silently interpolating a
      mismatched reference model onto this grid would risk masking a
      real setup error rather than surfacing it).
  (2) already in the SAME flat (i=down-dip row, j=along-strike column,
      flat index i*n_length+j) order run_slip_inversion()'s own
      "patches"/"slip" arrays use -- i.e. the reference file's own
      row ordering must match the subdivided fault's own indexing
      convention. This is true by construction for the GSI-style
      dataset this module was validated against (see
      verify_slip_inversion_validation.py), and for any file that
      itself came from this plugin's own fault-patch export, but is
      NOT independently verified here (there is no reliable way to
      infer a patch's (i,j) position from lon/lat/depth alone once
      dip varies row-to-row, e.g. a listric fault -- see
      core.fault_grid_builder). Mismatched ordering will silently
      produce a wrong-looking but not obviously-invalid comparison,
      same class of risk as core.fault_table_import's own documented
      depth_convention ambiguity.
  ONLY the single-fault case (one segment) is supported. A group
  (multiple fault rows inverted jointly) would need one reference file
  PER segment, aligned to that segment's own patch order -- not
  implemented; see ui/slip_inversion_validation_dialog.py's own scope
  note.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class SlipValidationResult:
    n_length: int
    n_width: int
    rt_true: np.ndarray       # (n_p,) right-lateral slip, reference (m)
    rev_true: np.ndarray      # (n_p,) reverse slip, reference (m)
    rt_inv: np.ndarray        # (n_p,) right-lateral slip, inverted (m)
    rev_inv: np.ndarray       # (n_p,) reverse slip, inverted (m)
    mag_true: np.ndarray      # (n_p,) scalar slip magnitude, reference (m)
    mag_inv: np.ndarray       # (n_p,) scalar slip magnitude, inverted (m)
    magnitude_corr: float     # Pearson r, mag_inv vs mag_true (nan if degenerate)
    magnitude_rms_m: float    # RMS(mag_inv - mag_true), metres
    true_mw: float            # Mw implied by the reference model's own slip
    achieved_mw: float        # Mw implied by the inverted slip (mirrors diag["achieved_mw"])
    errors: List[str] = field(default_factory=list)

    def mag_true_grid(self):
        """Reshape to (n_width, n_length) -- i=row 0 (down-dip index 0)
        at the TOP, matching imshow's default row-0-at-top display."""
        return self.mag_true.reshape(self.n_width, self.n_length)

    def mag_inv_grid(self):
        return self.mag_inv.reshape(self.n_width, self.n_length)


def compare_slip_to_reference(diag, imported_rows, n_length, n_width, mu,
                              patch_areas_m2=None):
    """
    diag           : the diagnostics dict returned by
                     core.okada_engine.run_slip_inversion() (or
                     run_slip_inversion_group() with exactly one
                     segment) -- diag["slip"] is the (n_p, 2)
                     [rt_lateral, reverse] list this compares against.
    imported_rows  : list of dicts from
                     core.fault_table_import.build_fault_rows_from_mapped_rows()
                     (each has "rt_lateral_slip_m", "reverse_slip_m") --
                     see module docstring's SCOPE note on required
                     length/ordering.
    n_length,
    n_width        : this fault's own Subdiv.(L)/Subdiv.(W) -- same
                     values passed to run_slip_inversion().
    mu             : elastic shear modulus (Pa), for the true-Mw
                     calculation -- same value the inversion itself
                     used (core.okada_engine.ElasticParameters.mu).
    patch_areas_m2 : optional (n_p,) array of each patch's own area
                     (length_km*1000 * width_km*1000). If not given,
                     every patch is assumed the SAME area, computed
                     from imported_rows[0]'s length_km/width_km --
                     correct for a uniform subdivide() grid (the
                     normal case) but not for patches built with
                     per-row-varying size (e.g. core.fault_grid_builder
                     output) -- pass patch_areas_m2 explicitly in that
                     case.

    Returns a SlipValidationResult. Raises ValueError if
    len(imported_rows) != n_length*n_width (see module docstring --
    this is a deliberate hard stop, not a warning, since there is no
    safe partial-comparison fallback).
    """
    n_p = n_length * n_width
    if len(imported_rows) != n_p:
        raise ValueError(
            f"Reference model has {len(imported_rows)} patch row(s) but "
            f"this inversion's grid is {n_length}x{n_width}={n_p} patches "
            f"-- a reference model must be on the EXACT SAME patch grid "
            f"for a cell-by-cell comparison (see module docstring; "
            f"regridding/resampling a mismatched reference is not "
            f"attempted here).")

    slip = diag.get("slip")
    if slip is None or len(slip) != n_p:
        raise ValueError(
            f"diag['slip'] has {0 if slip is None else len(slip)} entries, "
            f"expected {n_p} -- pass the diagnostics dict from the run "
            f"whose n_length/n_width match the arguments given here.")

    rt_inv = np.array([s[0] for s in slip], dtype=float)
    rev_inv = np.array([s[1] for s in slip], dtype=float)
    mag_inv = np.hypot(rt_inv, rev_inv)

    rt_true = np.array([float(r["rt_lateral_slip_m"]) for r in imported_rows])
    rev_true = np.array([float(r["reverse_slip_m"]) for r in imported_rows])
    mag_true = np.hypot(rt_true, rev_true)

    errors = []
    if np.std(mag_inv) == 0 or np.std(mag_true) == 0:
        magnitude_corr = float("nan")
        errors.append("Correlation undefined: one of the two slip "
                      "distributions is uniform (zero variance).")
    else:
        magnitude_corr = float(np.corrcoef(mag_inv, mag_true)[0, 1])
    magnitude_rms_m = float(np.sqrt(np.mean((mag_inv - mag_true) ** 2)))

    # The REFERENCE model's own moment must use EACH REFERENCE PATCH'S
    # OWN area (imported_rows already carries per-row length_km/
    # width_km from core.fault_table_import) -- NOT the inversion's
    # patch_areas_m2 below. These can legitimately differ: a real
    # published fault model (e.g. the GSI/Kobayashi et al. 2018
    # dataset this was validated against) commonly has row-to-row
    # varying patch width with depth, while THIS inversion's own grid
    # (from FaultParameters.subdivide()) is uniform -- reusing one
    # area for both would silently mis-compute the reference Mw
    # whenever the two grids' patch sizes differ, even though the
    # patch COUNT still lines up 1:1 for the slip-magnitude comparison
    # above.
    try:
        ref_areas_m2 = np.array([
            float(r["length_km"]) * 1000.0 * float(r["width_km"]) * 1000.0
            for r in imported_rows
        ])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(
            f"Reference rows are missing usable length_km/width_km "
            f"(needed to compute the reference model's own seismic "
            f"moment): {e}")
    m0_true = mu * float(np.sum(ref_areas_m2 * mag_true))
    true_mw = (2.0 / 3.0) * (math.log10(m0_true) - 9.1) if m0_true > 0 else float("-inf")

    # patch_areas_m2 describes THIS INVERSION's own patches (uniform
    # for a single subdivide()d fault, hence the scalar-broadcast
    # convenience above) -- only used as a fallback if diag itself
    # didn't already carry achieved_mw (run_slip_inversion_group()
    # always does, in practice, but this keeps the function usable
    # against a hand-built diag too, e.g. in tests).
    if patch_areas_m2 is None:
        length_km = float(imported_rows[0].get("length_km", 0.0))
        width_km = float(imported_rows[0].get("width_km", 0.0))
        area_m2 = length_km * 1000.0 * width_km * 1000.0
        patch_areas_m2 = np.full(n_p, area_m2)
    elif np.isscalar(patch_areas_m2):
        # A single float is accepted directly for the common case of a
        # uniform subdivide() grid (every patch the same size) -- the
        # caller need not build a full n_p-length array just to repeat
        # one number.
        patch_areas_m2 = np.full(n_p, float(patch_areas_m2))
    else:
        patch_areas_m2 = np.asarray(patch_areas_m2, dtype=float)
        if patch_areas_m2.shape[0] != n_p:
            raise ValueError(
                f"patch_areas_m2 has {patch_areas_m2.shape[0]} entries, "
                f"expected {n_p}.")

    achieved_mw = diag.get("achieved_mw")
    if achieved_mw is None:
        m0_inv = mu * float(np.sum(patch_areas_m2 * mag_inv))
        achieved_mw = (2.0 / 3.0) * (math.log10(m0_inv) - 9.1) if m0_inv > 0 else float("-inf")

    return SlipValidationResult(
        n_length=n_length, n_width=n_width,
        rt_true=rt_true, rev_true=rev_true,
        rt_inv=rt_inv, rev_inv=rev_inv,
        mag_true=mag_true, mag_inv=mag_inv,
        magnitude_corr=magnitude_corr, magnitude_rms_m=magnitude_rms_m,
        true_mw=true_mw, achieved_mw=float(achieved_mw),
        errors=errors)
