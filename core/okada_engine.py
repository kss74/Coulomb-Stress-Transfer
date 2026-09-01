# -*- coding: utf-8 -*-
"""
Coulomb Stress Transfer — Core Physics Engine (v4, validated)
=============================================================
Python translation of Beauducel's okada85.m (IPGP deformation-lib),
implementing Okada (1985) BSSA 75:1135-1154 analytical surface dislocation.

Validated against Coulomb 3.4.2:
  • Displacement: exact match (4 decimal places)
  • CFF at z=0:  r=0.983, sign=99.6%, median ratio=1.01

Optional: okada_wrapper (pip install okada-wrapper, needs gfortran) enables
exact depth-dependent CFF via Okada (1992) DC3D.
"""

import numpy as np
import os
import sys
import json
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Tuple

EPS    = 1e-10
EARTH_R = 6371.0   # km

# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class FaultParameters:
    """
    Rectangular fault, stored internally in Aki & Richards / seismological
    convention:

        lon, lat  : geographic position of the fault CENTROID's surface
                    projection (i.e. directly above/below the centroid)
        depth     : CENTROID depth (km, positive down) — NOT the top edge.
        length    : along-strike length (km)
        width     : down-dip width (km)
        strike    : degrees clockwise from North (0-360)
        dip       : degrees from horizontal (0-90)
        rake      : degrees, Aki-Richards convention (-180 to 180)
                    0/180 = pure strike-slip, +90 = pure reverse, -90 = pure normal
        slip      : total scalar slip (m)

    All internal physics (okada_engine formulas) use this representation
    and are validated against Coulomb 3.4.2 in this convention.

    Coulomb 3.x instead defines faults by a SURFACE TRACE (two endpoints
    at the TOP edge) plus top/bottom depth, and decomposes slip into
    right-lateral + reverse/dip components rather than scalar slip + rake.
    Use the classmethods below to build a FaultParameters from that
    convention; the stored .depth is always converted to centroid depth.

    top_depth / bottom_depth are provided as read-only convenience
    properties for display purposes (e.g. showing "top: 2.9 km" in the UI).
    """
    lon:    float = 0.0
    lat:    float = 0.0
    depth:  float = 10.0
    length: float = 20.0
    width:  float = 10.0
    strike: float = 0.0
    dip:    float = 90.0
    rake:   float = 0.0
    slip:   float = 1.0

    @property
    def U1(self): return self.slip * np.cos(np.deg2rad(self.rake))
    @property
    def U2(self): return self.slip * np.sin(np.deg2rad(self.rake))

    @property
    def top_depth(self):
        """Depth to the fault's TOP edge (km, positive down)."""
        return self.depth - np.sin(np.deg2rad(self.dip)) * self.width / 2.0

    @property
    def bottom_depth(self):
        """Depth to the fault's BOTTOM edge (km, positive down)."""
        return self.depth + np.sin(np.deg2rad(self.dip)) * self.width / 2.0

    @property
    def rt_lateral_slip(self):
        """
        Right-lateral slip component (Coulomb convention, m).
        Positive = right-lateral. Coulomb's rake=0 is right-lateral in
        their sign convention, which is opposite Aki-Richards rake=0
        (left-lateral); hence the sign flip here.
        """
        return -self.U1

    @property
    def reverse_slip(self):
        """Reverse (dip-slip) component (Coulomb convention, m). Positive = reverse."""
        return self.U2

    @classmethod
    def from_centroid(cls, lon, lat, depth, length, width, strike, dip, rake, slip):
        """Build directly in Aki-Richards centroid convention (explicit alias)."""
        return cls(lon=lon, lat=lat, depth=depth, length=length, width=width,
                   strike=strike, dip=dip, rake=rake, slip=slip)

    @classmethod
    def from_rt_lat_reverse(cls, lon, lat, depth, length, width, strike, dip,
                            rt_lateral_slip, reverse_slip, depth_is_top=True):
        """
        Build from Coulomb-style slip components (right-lateral + reverse),
        given a centroid or top-edge depth.

        rt_lateral_slip : right-lateral slip (m, Coulomb sign convention;
                          positive = right-lateral)
        reverse_slip    : reverse/dip slip (m, Coulomb sign convention;
                          positive = reverse/thrust)
        depth_is_top    : if True, `depth` is interpreted as the TOP edge
                          depth (Coulomb's convention) and converted to
                          centroid depth internally. If False, `depth` is
                          already the centroid depth.
        """
        U1 = -rt_lateral_slip   # flip: Coulomb rt-lat+ -> Aki-Richards rake=0 is left-lat+
        U2 = reverse_slip
        slip = float(np.hypot(U1, U2))
        rake = float(np.degrees(np.arctan2(U2, U1))) if slip > 0 else 0.0

        if depth_is_top:
            centroid_depth = depth + np.sin(np.deg2rad(dip)) * width / 2.0
        else:
            centroid_depth = depth

        return cls(lon=lon, lat=lat, depth=centroid_depth, length=length,
                   width=width, strike=strike, dip=dip, rake=rake, slip=slip)

    @classmethod
    def from_surface_trace(cls, lon1, lat1, lon2, lat2, top_depth, bottom_depth,
                           dip, rt_lateral_slip, reverse_slip, strike_side="right"):
        """
        Build a fault EXACTLY as Coulomb 3.x specifies it: from a surface
        trace (two endpoints of the TOP edge), top/bottom depth, dip, and
        right-lateral/reverse slip components.

        lon1,lat1 -> lon2,lat2 : trace endpoints (top edge), in the
                                  along-strike direction (start -> finish)
        top_depth, bottom_depth: km, positive down
        dip                    : degrees from horizontal; dips to the
                                  `strike_side` of the start->finish direction
        strike_side            : "right" (Coulomb/Aki-Richards default) or
                                  "left" — which side of the trace the
                                  fault dips toward
        """
        e1, n1 = geo_to_km(lon1, lat1, lon1, lat1)  # = (0,0)
        e2, n2 = geo_to_km(lon2, lat2, lon1, lat1)
        length = float(np.hypot(e2 - e1, n2 - n1))
        strike = float(np.degrees(np.arctan2(e2 - e1, n2 - n1))) % 360.0
        if strike_side == "left":
            strike = (strike + 180.0) % 360.0

        width = float((bottom_depth - top_depth) / np.sin(np.deg2rad(dip)))
        centroid_depth = (top_depth + bottom_depth) / 2.0

        # Midpoint of the trace, projected to centroid position (shifted
        # down-dip by half the horizontal width)
        mid_e = (e1 + e2) / 2.0
        mid_n = (n1 + n2) / 2.0
        dip_dir = np.deg2rad(strike + 90.0)   # horizontal down-dip direction
        horiz_half_width = (width / 2.0) * np.cos(np.deg2rad(dip))
        cen_e = mid_e + horiz_half_width * np.sin(dip_dir)
        cen_n = mid_n + horiz_half_width * np.cos(dip_dir)
        cen_lon, cen_lat = km_to_geo(cen_e, cen_n, lon1, lat1)

        return cls.from_rt_lat_reverse(
            lon=cen_lon, lat=cen_lat, depth=centroid_depth, length=length,
            width=width, strike=strike, dip=dip,
            rt_lateral_slip=rt_lateral_slip, reverse_slip=reverse_slip,
            depth_is_top=False,
        )

    def top_center(self):
        """
        Return the (lon, lat) of the fault's TOP-EDGE, along-strike MIDPOINT
        — i.e. the top-center reference point used by the standard
        Aki & Richards / SRCMOD finite-fault convention (NOT the volumetric
        centroid used internally by FaultParameters.lon/.lat).

        This is the single reference point most commonly meant by "the
        fault's location" in seismological source catalogs: directly above
        the top edge, at the midpoint of the fault's length, at top_depth.
        """
        strike = np.deg2rad(self.strike)
        e_c, n_c = geo_to_km(self.lon, self.lat, self.lon, self.lat)  # (0,0)
        dip_dir = np.deg2rad(self.strike + 90.0)
        horiz_half_width = (self.width / 2.0) * np.cos(np.deg2rad(self.dip))
        top_e = e_c - horiz_half_width * np.sin(dip_dir)
        top_n = n_c - horiz_half_width * np.cos(dip_dir)
        return km_to_geo(top_e, top_n, self.lon, self.lat)

    @classmethod
    def from_input(cls, lon, lat, depth, length, width, strike, dip,
                   lon_lat_mode="top_center",
                   rake=None, slip=None, rt_lateral_slip=None, reverse_slip=None):
        """
        Unified fault constructor with a single 3-way `lon_lat_mode`
        selector, describing exactly what (lon, lat, depth) means as a
        whole:

        lon_lat_mode : "top_start", "top_center", or "centroid"
            "top_start"  — (lon, lat) is the STARTING point of the fault's
                           TOP-EDGE surface trace ("Fault Top Projection");
                           `depth` is the top-edge depth. The trace runs
                           from there in the +strike direction for `length`
                           km to reach the far end.
            "top_center" — (lon, lat) is the MIDPOINT of the top-edge
                           surface trace (the standard Aki-Richards/SRCMOD
                           "top-center" reference point); `depth` is the
                           top-edge depth.
            "centroid"   — (lon, lat, depth) is the fault's VOLUMETRIC
                           CENTROID directly — e.g. as reported by a focal
                           mechanism / moment-tensor solution. No geometric
                           conversion is applied to (lon, lat) at all: this
                           IS the plugin's internal representation already.

        (NOTE: this supersedes the plugin's earlier two-independent-toggle
        design — position_anchor={"start","midpoint"} crossed with
        depth_reference={"top","centroid"}. That design silently mishandled
        the centroid case: depth_reference="centroid" changed how the DEPTH
        number was interpreted but never changed where (lon, lat) sat
        horizontally — it was always forced onto the top-edge trace. That
        made it impossible to feed in a genuine (lon, lat, depth)=centroid
        triple, such as a focal-mechanism centroid, without first manually
        back-computing an equivalent top-edge trace point. The 3-way
        `lon_lat_mode` above fixes this: "centroid" is a real third case,
        not a combination of the other two toggles.)

        Provide EITHER (rake, slip) OR (rt_lateral_slip, reverse_slip) for
        the slip, matching the two slip-input styles used elsewhere.
        """
        if lon_lat_mode == "centroid":
            # (lon, lat, depth) IS the centroid -- no conversion at all.
            if rake is not None and slip is not None:
                return cls.from_centroid(
                    lon=lon, lat=lat, depth=depth, length=length, width=width,
                    strike=strike, dip=dip, rake=rake, slip=slip,
                )
            elif rt_lateral_slip is not None and reverse_slip is not None:
                return cls.from_rt_lat_reverse(
                    lon=lon, lat=lat, depth=depth, length=length, width=width,
                    strike=strike, dip=dip,
                    rt_lateral_slip=rt_lateral_slip, reverse_slip=reverse_slip,
                    depth_is_top=False,
                )
            else:
                raise ValueError(
                    "Provide either (rake, slip) or (rt_lateral_slip, reverse_slip).")

        if lon_lat_mode == "top_start":
            position_anchor = "start"
        elif lon_lat_mode == "top_center":
            position_anchor = "midpoint"
        else:
            raise ValueError(
                'lon_lat_mode must be "top_start", "top_center", or "centroid".')

        strike_rad = np.deg2rad(strike)

        # Resolve (lon, lat) to the trace MIDPOINT, regardless of whether
        # it was given as the start point or already the midpoint.
        if position_anchor == "start":
            e_start, n_start = geo_to_km(lon, lat, lon, lat)  # (0,0)
            e_mid = e_start + (length / 2.0) * np.sin(strike_rad)
            n_mid = n_start + (length / 2.0) * np.cos(strike_rad)
            mid_lon, mid_lat = km_to_geo(e_mid, n_mid, lon, lat)
        else:
            mid_lon, mid_lat = lon, lat

        # For both top_* modes, `depth` is always the TOP-edge depth.
        top_depth = depth

        # (mid_lon, mid_lat, top_depth) is now exactly the top-center
        # reference point — hand off to from_top_center().
        if rake is not None and slip is not None:
            return cls.from_top_center(
                lon=mid_lon, lat=mid_lat, top_depth=top_depth,
                length=length, width=width, strike=strike, dip=dip,
                rake=rake, slip=slip,
            )
        elif rt_lateral_slip is not None and reverse_slip is not None:
            return cls.from_top_center(
                lon=mid_lon, lat=mid_lat, top_depth=top_depth,
                length=length, width=width, strike=strike, dip=dip,
                rt_lateral_slip=rt_lateral_slip, reverse_slip=reverse_slip,
            )
        else:
            raise ValueError(
                "Provide either (rake, slip) or (rt_lateral_slip, reverse_slip).")

    @classmethod
    def from_top_center(cls, lon, lat, top_depth, length, width, strike, dip,
                        rake=None, slip=None, rt_lateral_slip=None, reverse_slip=None):
        """
        Build a fault from its TOP-EDGE, along-strike MIDPOINT — the
        standard Aki & Richards / SRCMOD reference point convention (e.g.
        SRCMOD: "the top-center of each fault segment... as reference").

        This differs from the plain constructor / from_centroid(), whose
        lon/lat is the fault's VOLUMETRIC centroid — a valid but different
        internal convention. Use this constructor whenever your fault
        location comes from a source that follows the standard top-center
        convention (most finite-fault catalogs, hand-digitized fault
        traces anchored at a single point, etc.).

        Provide EITHER (rake, slip) OR (rt_lateral_slip, reverse_slip),
        matching the two slip-input conventions used elsewhere in this
        class.

        lon, lat  : top-edge, along-strike midpoint position
        top_depth : depth to the TOP edge (km, positive down)
        length, width, strike, dip : as elsewhere (km / degrees)
        """
        strike_rad = np.deg2rad(strike)
        dip_rad = np.deg2rad(dip)
        e_top, n_top = geo_to_km(lon, lat, lon, lat)  # (0,0)

        # Shift from top-center DOWN-DIP by half the horizontal width to
        # reach the volumetric centroid's surface projection.
        dip_dir = np.deg2rad(strike + 90.0)
        horiz_half_width = (width / 2.0) * np.cos(dip_rad)
        cen_e = e_top + horiz_half_width * np.sin(dip_dir)
        cen_n = n_top + horiz_half_width * np.cos(dip_dir)
        cen_lon, cen_lat = km_to_geo(cen_e, cen_n, lon, lat)

        centroid_depth = top_depth + np.sin(dip_rad) * width / 2.0

        if rake is not None and slip is not None:
            return cls.from_centroid(
                lon=cen_lon, lat=cen_lat, depth=centroid_depth,
                length=length, width=width, strike=strike, dip=dip,
                rake=rake, slip=slip,
            )
        elif rt_lateral_slip is not None and reverse_slip is not None:
            return cls.from_rt_lat_reverse(
                lon=cen_lon, lat=cen_lat, depth=centroid_depth,
                length=length, width=width, strike=strike, dip=dip,
                rt_lateral_slip=rt_lateral_slip, reverse_slip=reverse_slip,
                depth_is_top=False,
            )
        else:
            raise ValueError(
                "Provide either (rake, slip) or (rt_lateral_slip, reverse_slip).")

    def surface_trace(self):
        """
        Return the fault's TOP-EDGE surface trace as two (lon, lat) points
        (start, finish), matching Coulomb's X-start/Y-start/X-fin/Y-fin.
        """
        strike = np.deg2rad(self.strike)
        half_L = self.length / 2.0
        e_c, n_c = geo_to_km(self.lon, self.lat, self.lon, self.lat)  # (0,0)
        # Shift from centroid back up-dip to the top edge (horizontal projection)
        dip_dir = np.deg2rad(self.strike + 90.0)
        horiz_half_width = (self.width / 2.0) * np.cos(np.deg2rad(self.dip))
        top_e = e_c - horiz_half_width * np.sin(dip_dir)
        top_n = n_c - horiz_half_width * np.cos(dip_dir)

        e1 = top_e - half_L * np.sin(strike)
        n1 = top_n - half_L * np.cos(strike)
        e2 = top_e + half_L * np.sin(strike)
        n2 = top_n + half_L * np.cos(strike)

        lon1, lat1 = km_to_geo(e1, n1, self.lon, self.lat)
        lon2, lat2 = km_to_geo(e2, n2, self.lon, self.lat)
        return (lon1, lat1), (lon2, lat2)

    def geological_surface_trace(self):
        """
        Return the fault plane's intersection with z=0 (the surface),
        extrapolated UP-DIP from the top edge if top_depth > 0 -- this is
        Coulomb's own "surface trace" concept for a blind/buried fault:
        where the fault plane would break the surface if its dip were
        continued, NOT the actual (possibly buried) top edge's horizontal
        projection (see surface_trace(), aka "Fault Top Projection", for
        that).

        Returns two (lon, lat) points (start, finish), same convention as
        surface_trace(). Returns None if dip <= 0 (a horizontal fault
        plane never intersects z=0 at a single well-defined location by
        up-dip extrapolation). If top_depth <= 0 (the fault already
        reaches or exceeds the surface), this coincides exactly with
        surface_trace().

        Derivation: moving a distance dW up-dip along the fault plane
        changes depth by -dW*sin(dip) and horizontal (down-dip-direction)
        position by -dW*cos(dip). To climb from top_depth to z=0 requires
        dW = top_depth/sin(dip), i.e. an additional horizontal (up-dip)
        shift of top_depth/sin(dip)*cos(dip) = top_depth/tan(dip) beyond
        the top edge's own up-dip offset from the centroid.
        """
        if self.dip <= EPS:
            return None
        top_depth = self.top_depth
        if top_depth <= 0:
            return self.surface_trace()

        strike = np.deg2rad(self.strike)
        dip_rad = np.deg2rad(self.dip)
        half_L = self.length / 2.0
        e_c, n_c = geo_to_km(self.lon, self.lat, self.lon, self.lat)  # (0,0)
        dip_dir = np.deg2rad(self.strike + 90.0)
        horiz_half_width = (self.width / 2.0) * np.cos(dip_rad)
        extra_updip = top_depth / np.tan(dip_rad)
        total_updip = horiz_half_width + extra_updip

        top_e = e_c - total_updip * np.sin(dip_dir)
        top_n = n_c - total_updip * np.cos(dip_dir)

        e1 = top_e - half_L * np.sin(strike)
        n1 = top_n - half_L * np.cos(strike)
        e2 = top_e + half_L * np.sin(strike)
        n2 = top_n + half_L * np.cos(strike)

        lon1, lat1 = km_to_geo(e1, n1, self.lon, self.lat)
        lon2, lat2 = km_to_geo(e2, n2, self.lon, self.lat)
        return (lon1, lat1), (lon2, lat2)

    def subdivide(self, n_length, n_width, slip_overrides=None):
        """
        Split this fault into n_length x n_width EQUAL-AREA sub-patches,
        matching Coulomb's fault-subdivision option. Each sub-patch is a
        full FaultParameters with the same strike/dip as the parent,
        positioned at its own sub-centroid, and (unless overridden --
        see `slip_overrides` below) the same rake/slip as the parent.

        For SOURCES: subdividing alone does not change the physics
        (Okada's rectangular-fault formula already integrates slip
        exactly over the whole patch), so a uniform-slip subdivision is
        only useful for visualization. Passing `slip_overrides` makes it
        physically meaningful: each sub-patch becomes an independent
        Okada dislocation, so per-patch slip genuinely changes the
        computed stress field ("distributed slip" / variable-slip
        source).

        For RECEIVERS: subdividing lets ΔCFF be evaluated at multiple
        points across what was previously a single receiver centroid,
        which better resolves near-field stress variation across a
        large receiver fault. (slip_overrides is meaningless for
        receivers, which have zero slip by definition, and should be
        left as None.)

        n_length : number of subdivisions along strike (>=1)
        n_width  : number of subdivisions down-dip (>=1)
        slip_overrides : optional dict {(i, j): (rt_lateral_slip,
            reverse_slip)}, Coulomb sign convention (rt-lateral+ =
            right-lateral, reverse+ = thrust), giving this specific
            sub-patch its own slip instead of inheriting the parent's
            uniform rake/slip. i is the down-dip (width) index
            (0..n_width-1), j is the along-strike (length) index
            (0..n_length-1) -- matching the (i, j) patch addressing
            used internally below and in the flat return order. Patches
            not present in the dict keep the parent's uniform slip.

        Returns a flat list of FaultParameters, length n_length*n_width,
        ordered along-strike-major (row = along-dip index, col = along-
        strike index), i.e. patch (i,j) is at flat index i*n_length+j.
        """
        n_length = max(1, int(n_length))
        n_width = max(1, int(n_width))
        if n_length == 1 and n_width == 1:
            return [self]

        sub_L = self.length / n_length
        sub_W = self.width / n_width

        strike = np.deg2rad(self.strike)
        dip = np.deg2rad(self.dip)
        cs, ss = np.cos(strike), np.sin(strike)
        cd = np.cos(dip)

        # Centroid-local along-strike offsets for each sub-patch column,
        # and down-dip (horizontal-projected) offsets for each row, measured
        # from the PARENT's centroid.
        half_L = self.length / 2.0
        half_W_horiz = (self.width / 2.0) * cd

        col_centers = [-half_L + sub_L * (j + 0.5) for j in range(n_length)]
        row_centers_horiz = [-half_W_horiz + (sub_W * cd) * (i + 0.5) for i in range(n_width)]

        patches = []
        for i, y_local in enumerate(row_centers_horiz):
            for j, x_local in enumerate(col_centers):
                # Rotate the (along-strike, across-strike-horizontal) local
                # offset into geographic (East, North) km, then to lon/lat.
                e_km = x_local * ss + y_local * cs
                n_km = x_local * cs - y_local * ss
                lon, lat = km_to_geo(e_km, n_km, self.lon, self.lat)

                # Depth: this sub-patch's centroid depth is the parent's
                # top depth plus the down-dip distance to this row's centre
                # (measured along the dip direction, not horizontally).
                sub_centroid_down_dip = sub_W * (i + 0.5)
                sub_depth = self.top_depth + np.sin(dip) * sub_centroid_down_dip

                override = slip_overrides.get((i, j)) if slip_overrides else None
                if override is not None:
                    rt_lateral_slip, reverse_slip = override
                    # Same Coulomb-convention -> (rake, slip) conversion
                    # used in from_rt_lat_reverse(), applied per-patch.
                    U1 = -rt_lateral_slip
                    U2 = reverse_slip
                    patch_slip = float(np.hypot(U1, U2))
                    patch_rake = float(np.degrees(np.arctan2(U2, U1))) if patch_slip > 0 else 0.0
                else:
                    patch_slip = self.slip
                    patch_rake = self.rake

                patches.append(FaultParameters(
                    lon=float(lon), lat=float(lat), depth=float(sub_depth),
                    length=sub_L, width=sub_W,
                    strike=self.strike, dip=self.dip, rake=patch_rake,
                    slip=patch_slip,
                ))
        return patches




@dataclass
class ElasticParameters:
    mu:       float = 3.2e10
    nu:       float = 0.25
    friction: float = 0.40


@dataclass
class GridParameters:
    lon_min:  float = -1.0
    lon_max:  float =  1.0
    lat_min:  float = -1.0
    lat_max:  float =  1.0
    depth_km: float =  0.0
    n_lon:    int   = 100
    n_lat:    int   = 100


# ─── Coordinate helpers ────────────────────────────────────────────────────────

def geo_to_km(lon, lat, lon0, lat0):
    lat0r = np.deg2rad(lat0)
    x = (lon - lon0) * np.deg2rad(1) * EARTH_R * np.cos(lat0r)
    y = (lat - lat0) * np.deg2rad(1) * EARTH_R
    return x, y

def km_to_geo(x, y, lon0, lat0):
    lat0r = np.deg2rad(lat0)
    lon = lon0 + x / (np.deg2rad(1) * EARTH_R * np.cos(lat0r))
    lat = lat0 + y / (np.deg2rad(1) * EARTH_R)
    return lon, lat


# ─── Near-field (Okada/DC3D singularity) proximity checks ─────────────────────
#
# Added 2026-08-09b, following a validation exercise (see
# PROJECT_HANDOVER_ADDENDUM_2026-08-09b_wells_coppersmith_scaling.md) that
# isolated and quantified how close a grid point or a second fault can get
# to a source fault's surface trace before ΔCFF stops being a physically
# meaningful estimate and starts reflecting the Okada/DC3D dislocation
# singularity instead (elastic displacement is discontinuous across the
# fault plane, so stress diverges as you approach it -- this is expected
# behavior in BOTH this engine and Coulomb 3.4.2, not a bug in either).
#
# Empirical calibration (single test case, Mw~7 reverse, L=48.4 km,
# W=22.3 km, dip 75 deg, receiver depth near the fault's top edge):
# agreement with Coulomb 3.4.2 was poor within ~10 km of the surface
# trace and excellent (amplitude ratio ~0.996, Pearson~1.0) beyond
# ~15-20 km -- roughly one down-dip WIDTH away. `near_field_threshold_km`
# uses width_km as a physically-motivated proxy (near-field distortion
# scales with source dimension) but this has only been validated against
# ONE geometry/depth combination -- treat it as a heuristic starting
# point, not a rigorously derived bound. Revisit if/when more test cases
# are run.

def _fault_trace_endpoints_km(fault, lon0, lat0):
    """
    The two (x_km, y_km) endpoints of `fault`'s TOP-EDGE surface trace,
    in local east/north km relative to (lon0, lat0). For near-field
    proximity checks only -- NOT a substitute for the full 3D fault-plane
    geometry the physics engine itself uses.

    `fault.lon`/`fault.lat` are the CENTROID's surface projection (see
    FaultParameters docstring), not the top edge's. For dip < 90 deg the
    top edge sits up-dip of the centroid by a horizontal offset of
    (width/2)*cos(dip), so that offset must be applied before extending
    along strike -- omitting it (as the previous version did) silently
    mislocates the trace by up to (width/2)*cos(dip) km, which can exceed
    near_field_threshold_km itself for shallow-dip faults. Verified
    against an independent full-4-corner 3D construction and against the
    existing (validated) top_depth/bottom_depth properties: both agree
    exactly, and dip=90 correctly collapses the offset to zero.
    """
    xc, yc = geo_to_km(fault.lon, fault.lat, lon0, lat0)
    strike_r = np.deg2rad(fault.strike)
    dip_r = np.deg2rad(fault.dip)
    ux, uy = np.sin(strike_r), np.cos(strike_r)           # along-strike unit vector
    updip_shift = (fault.width / 2.0) * np.cos(dip_r)     # horizontal centroid->top offset
    # up-dip horizontal unit vector = (-cos(strike), sin(strike))
    # (down-dip azimuth = strike+90 deg by convention, up-dip is its opposite)
    xc_top = xc - np.cos(strike_r) * updip_shift
    yc_top = yc + np.sin(strike_r) * updip_shift
    half = fault.length / 2.0
    return (xc_top - ux * half, yc_top - uy * half), (xc_top + ux * half, yc_top + uy * half)


def _point_segment_distance_km(px, py, p1, p2):
    """Perpendicular (clamped) distance from point(s) to a line segment; all km."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    seg_len2 = dx * dx + dy * dy
    if seg_len2 == 0:
        return np.hypot(px - p1[0], py - p1[1])
    t = np.clip(((px - p1[0]) * dx + (py - p1[1]) * dy) / seg_len2, 0.0, 1.0)
    projx, projy = p1[0] + t * dx, p1[1] + t * dy
    return np.hypot(px - projx, py - projy)


def _point_segment_distance_3d_km(px, py, pz, p1, p2):
    """3D perpendicular (clamped) distance from point(s) to a line segment;
    all km, depth positive down. p1/p2 are length-3 (x,y,depth) arrays."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    dx, dy, dz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    seg_len2 = dx * dx + dy * dy + dz * dz
    if seg_len2 == 0:
        return np.sqrt((px - p1[0]) ** 2 + (py - p1[1]) ** 2 + (pz - p1[2]) ** 2)
    t = np.clip(((px - p1[0]) * dx + (py - p1[1]) * dy + (pz - p1[2]) * dz) / seg_len2,
                0.0, 1.0)
    projx = p1[0] + t * dx
    projy = p1[1] + t * dy
    projz = p1[2] + t * dz
    return np.sqrt((px - projx) ** 2 + (py - projy) ** 2 + (pz - projz) ** 2)


def _fault_corners_3d_km(fault, lon0, lat0):
    """
    The 4 corners of `fault`'s rectangular plane in local (x_km, y_km,
    depth_km) relative to (lon0, lat0), ordered around the perimeter:
    top-start (up-dip end), top-end, bottom-end (down-dip end), bottom-start.

    `fault.lon`/`fault.lat` is the CENTROID's surface projection and
    `fault.depth` the CENTROID depth (see FaultParameters docstring); the
    top/bottom edges are offset from it by (width/2)*downdip_vector, same
    convention as the corrected _fault_trace_endpoints_km.
    """
    xc, yc = geo_to_km(fault.lon, fault.lat, lon0, lat0)
    strike_r = np.deg2rad(fault.strike)
    dip_r = np.deg2rad(fault.dip)
    along = np.array([np.sin(strike_r), np.cos(strike_r), 0.0])
    downdip = np.array([np.cos(strike_r) * np.cos(dip_r),
                         -np.sin(strike_r) * np.cos(dip_r),
                         np.sin(dip_r)])
    centroid = np.array([xc, yc, fault.depth])
    half_W_vec = downdip * (fault.width / 2.0)
    half_L_vec = along * (fault.length / 2.0)
    top_center = centroid - half_W_vec
    bottom_center = centroid + half_W_vec
    top_start = top_center - half_L_vec
    top_end = top_center + half_L_vec
    bottom_start = bottom_center - half_L_vec
    bottom_end = bottom_center + half_L_vec
    return top_start, top_end, bottom_end, bottom_start


def _fault_nearest_edge_distance_3d_km(fault, x_km, y_km, depth_km, lon0, lat0):
    """
    3D distance (km) from receiver point(s) at (x_km, y_km, depth_km) to
    the NEAREST point on `fault`'s full rectangular perimeter (all four
    edges: top, bottom, and both lateral ends) -- not just the top-edge
    surface trace. See PROJECT_HANDOVER_ADDENDUM near-field-threshold
    notes: the Okada/DC3D closed-form kernels diverge near any of the
    four edges (confirmed numerically, ~1/r), so a receiver close in 3D
    to the bottom or a lateral edge is just as much "near-field" as one
    close to the top trace, even if it's far from the top trace in 2D
    map view (the previous top-trace-only, depth-blind check missed
    this, especially for shallow-dip faults evaluated at receiver depths
    away from the surface).
    """
    top_start, top_end, bottom_end, bottom_start = _fault_corners_3d_km(fault, lon0, lat0)
    edges = ((top_start, top_end), (top_end, bottom_end),
             (bottom_end, bottom_start), (bottom_start, top_start))
    d = None
    for p1, p2 in edges:
        dd = _point_segment_distance_3d_km(x_km, y_km, depth_km, p1, p2)
        d = dd if d is None else np.minimum(d, dd)
    return d


_NEAR_FIELD_K = 0.6793  # calibrated so K*sqrt(L*W) == 22.32 km for the one
                         # Coulomb-validated test geometry (test3.inp,
                         # L=48.37, W=22.32 km) -- see near_field_threshold_km
                         # docstring. Provisional pending a second validated
                         # geometry; not yet re-derived from first principles.


def near_field_threshold_km(fault):
    """
    OUTER near-field radius (km): beyond this, both sign and magnitude of
    ΔCFF are considered reliable (~10% relative distortion from the edge
    singularity, by design/policy choice -- see near-field-threshold
    handover notes).

    Uses sqrt(length*width) rather than width alone. Derivation: the
    Okada/DC3D near-edge term diverges as C/r (confirmed numerically,
    C ~ mu*slip, both signed and magnitude-dependent on receiver
    orientation), while the smooth non-singular part of ΔCFF falls off
    over a length scale set by the fault's areal footprint, not width
    alone; balancing the two makes mu and slip cancel out of the
    crossover radius, leaving a scaling of sqrt(length*width). Reduces to
    ~width for near-square ruptures (like the validated test geometry,
    L/W ~ 2.2) but differs materially for elongated ones, where width
    alone would have under-estimated the near-field zone along the
    shorter (down-dip) extent and over-estimated it relative to the
    fault's true areal scale for very long ruptures.

    The functional FORM (sqrt(L*W)) is derived; the multiplicative
    constant _NEAR_FIELD_K is calibrated against the single Coulomb-
    validated geometry available (test3.inp) and floored at 5 km for
    small faults -- treat as provisional until validated against a
    second, differently-shaped geometry.
    """
    return max(_NEAR_FIELD_K * np.sqrt(fault.length * fault.width), 5.0)


def near_field_threshold_inner_km(fault):
    """
    INNER near-field radius (km): within this, ΔCFF sign itself is not
    reliable (~40% relative distortion), not just magnitude.

    Exactly outer/4. This ratio -- unlike the outer radius's absolute
    scale -- is robustly derived from the confirmed 1/r decay of the
    edge singularity alone: if the smooth (non-singular) part of ΔCFF is
    locally ~constant across the tier range, distortion(r) ~ 1/r, so
    going from the 10% (outer) to 40% (inner) distortion level requires
    exactly a 4x reduction in distance, independent of geometry, slip,
    or receiver coupling. See near-field-threshold handover notes for
    the numerical confirmation and for why a receiver-orientation-
    independent ABSOLUTE calibration of either radius could not be
    derived (worst-case coupling produces near-cancellation between
    shear and normal-stress contributions to CFF and never converges to
    a finite crossing within physically relevant distances).
    """
    return near_field_threshold_km(fault) / 4.0


def near_field_grid_mask(sources, lon2d, lat2d, depth_km=0.0):
    """
    Integer-coded array, same shape as lon2d/lat2d, one value per grid
    point (max/worst tier over all source faults):
        0 -- clear: ΔCFF sign and magnitude both reliable
        1 -- magnitude caution: within near_field_threshold_km of a
             source fault's nearest edge; sign should still be reliable
             but magnitude may be off by up to ~40% (the outer/inner
             tolerance band -- see near_field_threshold_km /
             near_field_threshold_inner_km docstrings)
        2 -- sign untrustworthy: within near_field_threshold_inner_km;
             do not trust sign OR magnitude here

    `depth_km` is the receiver depth this grid was evaluated at (0.0 for
    the z=0 surface path; pass grid.depth_km for compute_coulomb_grid_depth).
    Distance is full 3D distance to the nearest point on the fault's
    complete rectangular perimeter (all four edges), not just 2D distance
    to the top-edge surface trace -- see _fault_nearest_edge_distance_3d_km.

    Backward-compat note: this used to return a plain boolean (True where
    d <= near_field_threshold_km). `mask > 0` reproduces that behavior.
    """
    lon0, lat0 = float(np.mean(lon2d)), float(np.mean(lat2d))
    x_km, y_km = geo_to_km(lon2d, lat2d, lon0, lat0)
    depth_arr = np.full(np.shape(lon2d), float(depth_km))
    mask = np.zeros(np.shape(lon2d), dtype=np.int8)
    for src in sources:
        d = _fault_nearest_edge_distance_3d_km(src, x_km, y_km, depth_arr, lon0, lat0)
        outer = near_field_threshold_km(src)
        inner = near_field_threshold_inner_km(src)
        mask = np.maximum(mask, np.where(d <= inner, 2, np.where(d <= outer, 1, 0)).astype(np.int8))
    return mask


def near_field_fault_pairs(sources, receiver=None):
    """
    Pairwise check between source faults' surface traces (and, if given,
    the receiver fault) for the specific scenario flagged as a risk:
    closely-spaced faults, e.g. adjacent segments of a multi-segment
    rupture, or a receiver fault sitting close to a source segment.
    Returns a list of human-readable warning strings, one per flagged
    pair (empty list if none).

    Distances are surface-trace-to-surface-trace only (2D, ignoring
    depth/dip separation between the two planes) -- deliberately
    conservative/over-inclusive, since two faults that are close in map
    view but well-separated in depth would still be flagged here.
    """
    warnings = []
    labeled = [(f"Source fault {i + 1}", f) for i, f in enumerate(sources)]
    if receiver is not None:
        labeled.append(("Receiver fault", receiver))
    if len(labeled) < 2:
        return warnings

    lon0 = float(np.mean([f.lon for _, f in labeled]))
    lat0 = float(np.mean([f.lat for _, f in labeled]))
    endpoints = [(name, _fault_trace_endpoints_km(f, lon0, lat0), f)
                for name, f in labeled]

    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            name_i, (p1i, p2i), fi = endpoints[i]
            name_j, (p1j, p2j), fj = endpoints[j]
            # Approximate segment-segment distance via endpoint-to-opposite
            # -segment sampling (exact for the non-crossing case that
            # matters here; good enough for an advisory check).
            d = float(min(
                _point_segment_distance_km(p1i[0], p1i[1], p1j, p2j),
                _point_segment_distance_km(p2i[0], p2i[1], p1j, p2j),
                _point_segment_distance_km(p1j[0], p1j[1], p1i, p2i),
                _point_segment_distance_km(p2j[0], p2j[1], p1i, p2i),
            ))
            thresh = max(near_field_threshold_km(fi), near_field_threshold_km(fj))
            if d <= thresh:
                warnings.append(
                    f"{name_i} and {name_j} are ~{d:.1f} km apart (surface "
                    f"trace to surface trace) — within the ~{thresh:.0f} km "
                    f"near-field zone. ΔCFF between them may reflect the "
                    f"Okada/DC3D dislocation singularity rather than a "
                    f"reliable physical estimate; treat with caution, "
                    f"especially for closely-spaced multi-segment ruptures.")
    return warnings


def total_seismic_moment(sources, elastic: 'ElasticParameters'):
    """
    Total seismic moment (dyne-cm) and equivalent moment magnitude implied
    by the CURRENT source-fault set and elastic parameters -- mirroring
    Coulomb 3.4.2's own `seis_moment()` readout (coulomb.m, line ~6345,
    called automatically whenever a fault input is loaded and printed to
    both the MATLAB console and the status bar as "Total seismic moment =
    X.XXe+XX dyne cm (Mw = X.XX)").

    This is a REPORTING function only -- it does not feed back into or
    affect the DC3D/Okada stress calculation in any way. Its only purpose
    is letting a user sanity-check / diff this plugin's implied moment
    against Coulomb's own printed value for the same input, independent
    of however the slip values in the fault table were set (empirical
    scaling, manual edit, or otherwise).

    Formula, reproduced verbatim from coulomb.m's seis_moment() (this is
    a DIFFERENT formula from the one in scaling_relations.py's Coulomb-
    compatible mode, which estimates a NEW fault's slip FROM Mw using a
    separate, hardcoded-shear-modulus formula in coulomb.m's
    BCF_FE_Calcbutton_callback -- the two are easy to conflate since both
    live in coulomb.m and both cite a Kanamori-style Mw<->M0 relation,
    but they are not the same code path and use different constants):

        shearmod = mu (Pa)               # Coulomb: YOUNG/(2*(1+POIS)), in bar;
                                          # here: ElasticParameters.mu (Pa) directly
        smo[dyne-cm] = shearmod[bar] * length[km] * width[km] * slip[m] * 1e18
        (unit-equivalent using mu in Pa directly, since 1 Pa = 1e-5 bar):
        smo[dyne-cm] = mu[Pa] * length[km] * width[km] * slip[m] * 1e13
        amo = sum(smo) over all source faults
        mw  = (2/3) * log10(amo) - 10.7  # Coulomb's own (rounded)
                                          # Kanamori-style constant,
                                          # reproduced verbatim for
                                          # numeric parity with its
                                          # printed value -- not
                                          # "corrected" to 16.1.

    Uses `f.length`, `f.width`, `f.slip` directly from each FaultParameters
    (already the along-strike length, down-dip width, and net scalar slip
    in this plugin's internal representation -- no re-derivation from
    endpoint coordinates needed, unlike coulomb.m which reconstructs
    flength/wfault from xstart/ystart/xfinish/yfinish/top/bottom each time).

    Parameters
    ----------
    sources : list of FaultParameters
        The current SOURCE faults (nonzero slip), i.e. exactly what
        `_get_sources()` / `fault_table.get_sources()` returns in
        `main_dialog.py` -- NOT the receiver fault(s).
    elastic : ElasticParameters
        Uses `elastic.mu` (Pa).

    Returns
    -------
    (amo_dyne_cm, mw) : (float, float or None)
        `mw` is None if `sources` is empty or total moment is zero
        (log10(0) is undefined -- mirrors Coulomb's own guard, which
        simply skips the moment print in that case).
    """
    amo = 0.0
    for f in sources:
        amo += elastic.mu * f.length * f.width * f.slip * 1e13
    if amo <= 0.0:
        return 0.0, None
    mw = (2.0 / 3.0) * np.log10(amo) - 10.7
    return float(amo), float(mw)


def format_seismic_moment_message(amo_dyne_cm, mw):
    """
    Formats a (amo_dyne_cm, mw) pair from total_seismic_moment() into the
    same human-readable form Coulomb 3.4.2 prints to its console and
    status bar ('%6.2e' / '%4.2f' in coulomb.m), so the two can be
    diffed by eye. Returns None if `mw` is None (nothing to report).
    """
    if mw is None:
        return None
    return f"Total seismic moment = {amo_dyne_cm:.2e} dyne cm (Mw = {mw:.2f})"


def grid_counts_from_spacing(lon_min, lon_max, lat_min, lat_max, spacing,
                             units="km", max_points_per_axis=2000):
    """
    Compute (n_lon, n_lat, clamped) grid point counts that give an
    approximately uniform spacing between adjacent grid points, from a
    single spacing value in either kilometers or decimal degrees —
    instead of specifying n_lon/n_lat directly.

    lon_min, lon_max, lat_min, lat_max : grid bounds (degrees)
    spacing       : desired spacing between adjacent grid points
    units         : "km" or "deg"
    max_points_per_axis : safety cap (see `clamped` below)

    units="deg" is exact: n = round(span / spacing) + 1 on each axis.

    units="km" requires a km->degree conversion. One degree of latitude
    is ~constant (111.19 km), but one degree of LONGITUDE shrinks toward
    the poles by cos(latitude) — the same relationship used by
    geo_to_km()/km_to_geo() elsewhere in this module. This function uses
    the grid's CENTER latitude for that conversion, so the requested km
    spacing is exact at the center latitude and drifts slightly toward
    the grid's north/south edges — the same local flat-Earth
    approximation used throughout the rest of this engine, not a new
    inconsistency.

    Returns (n_lon, n_lat, clamped). `clamped` is True if the requested
    spacing would have produced more than `max_points_per_axis` points on
    either axis (e.g. a sub-meter spacing requested over a multi-degree
    extent) — in that case the corresponding axis is capped at
    `max_points_per_axis` rather than silently trying to compute (and
    likely hang on) an enormous grid.
    """
    if spacing <= 0:
        raise ValueError("Grid spacing must be positive.")
    if units not in ("km", "deg"):
        raise ValueError('units must be "km" or "deg".')

    lon_span = abs(lon_max - lon_min)
    lat_span = abs(lat_max - lat_min)

    if units == "deg":
        spacing_deg_lon = spacing
        spacing_deg_lat = spacing
    else:  # units == "km"
        lat0 = (lat_min + lat_max) / 2.0
        cos_lat0 = np.cos(np.deg2rad(lat0))
        deg_per_km_lat = 1.0 / (np.deg2rad(1) * EARTH_R)
        if abs(cos_lat0) < EPS:
            raise ValueError(
                "Cannot convert a km spacing to longitude degrees at "
                "latitude \u00b190\u00b0 (cos(lat)=0, so 1\u00b0 of longitude is 0 km "
                "there). Use degree-based spacing instead, or a grid that "
                "doesn't reach the pole.")
        deg_per_km_lon = 1.0 / (np.deg2rad(1) * EARTH_R * cos_lat0)
        spacing_deg_lat = spacing * deg_per_km_lat
        spacing_deg_lon = spacing * deg_per_km_lon

    n_lon = max(2, int(round(lon_span / spacing_deg_lon)) + 1)
    n_lat = max(2, int(round(lat_span / spacing_deg_lat)) + 1)

    clamped = False
    if n_lon > max_points_per_axis:
        n_lon = max_points_per_axis
        clamped = True
    if n_lat > max_points_per_axis:
        n_lat = max_points_per_axis
        clamped = True

    return n_lon, n_lat, clamped


# ─── I-functions (Okada 1985) ─────────────────────────────────────────────────

def _I1(xi, eta, q, dip, nu, R):
    db = eta*np.sin(dip) - q*np.cos(dip)
    cd = np.cos(dip)
    if cd > EPS:
        return ((1-2*nu)*(-xi/(cd*(R+db+EPS)))
                - np.sin(dip)/cd * _I5(xi,eta,q,dip,nu,R,db))
    return -(1-2*nu)/2 * xi*q/(R+db+EPS)**2

def _I2(eta, q, dip, nu, R):
    return (1-2*nu)*(-np.log(R+eta+EPS)) - _I3(eta,q,dip,nu,R)

def _I3(eta, q, dip, nu, R):
    yb = eta*np.cos(dip)+q*np.sin(dip)
    db = eta*np.sin(dip)-q*np.cos(dip)
    cd = np.cos(dip)
    if cd > EPS:
        return ((1-2*nu)*(yb/(cd*(R+db+EPS)) - np.log(R+eta+EPS))
                + np.sin(dip)/cd * _I4(db,eta,q,dip,nu,R))
    return (1-2*nu)/2*(eta/(R+db+EPS) + yb*q/(R+db+EPS)**2 - np.log(R+eta+EPS))

def _I4(db, eta, q, dip, nu, R):
    cd = np.cos(dip)
    if cd > EPS:
        return (1-2*nu)/cd*(np.log(R+db+EPS) - np.sin(dip)*np.log(R+eta+EPS))
    return -(1-2*nu)*q/(R+db+EPS)

def _I5(xi, eta, q, dip, nu, R, db):
    X  = np.sqrt(xi**2+q**2)+EPS
    cd = np.cos(dip)
    if cd > EPS:
        val = (1-2*nu)*2/cd * np.arctan(
            (eta*(X+q*cd)+X*(R+X)*np.sin(dip))/(xi*(R+X)*cd+EPS))
        return np.where(np.abs(xi)<EPS, 0.0, val)
    return -(1-2*nu)*xi*np.sin(dip)/(R+db+EPS)


# ─── K, J functions for strain (Okada 1985 Table 2) ──────────────────────────

def _A(x, R): return (2*R+x)/(R**3*(R+x)**2+EPS)

def _K1(xi, eta, q, dip, nu, R):
    db=eta*np.sin(dip)-q*np.cos(dip); cd=np.cos(dip)
    if cd>EPS: return (1-2*nu)*xi/cd*(1/(R*(R+db)+EPS)-np.sin(dip)/(R*(R+eta)+EPS))
    return (1-2*nu)*xi*q/(R*(R+db)**2+EPS)

def _K3(xi, eta, q, dip, nu, R):
    db=eta*np.sin(dip)-q*np.cos(dip); yb=eta*np.cos(dip)+q*np.sin(dip); cd=np.cos(dip)
    if cd>EPS: return (1-2*nu)/cd*(q/(R*(R+eta)+EPS)-yb/(R*(R+db)+EPS))
    return (1-2*nu)*np.sin(dip)/(R+db+EPS)*(xi**2/(R*(R+db)+EPS)-1)

def _J1(xi, eta, q, dip, nu, R):
    db=eta*np.sin(dip)-q*np.cos(dip); cd=np.cos(dip)
    if cd>EPS:
        return ((1-2*nu)/cd*(xi**2/(R*(R+db)**2+EPS)-1/(R+db+EPS))
                - np.sin(dip)/cd*_K3(xi,eta,q,dip,nu,R))
    return (1-2*nu)/2*q/(R+db+EPS)**2*(2*xi**2/(R*(R+db)+EPS)-1)

def _J2(xi, eta, q, dip, nu, R):
    db=eta*np.sin(dip)-q*np.cos(dip); yb=eta*np.cos(dip)+q*np.sin(dip); cd=np.cos(dip)
    if cd>EPS:
        return ((1-2*nu)/cd*xi*yb/(R*(R+db)**2+EPS)
                - np.sin(dip)/cd*_K1(xi,eta,q,dip,nu,R))
    return (1-2*nu)/2*xi*np.sin(dip)/(R+db+EPS)**2*(2*q**2/(R*(R+db)+EPS)-1)

def _J3(xi, eta, q, dip, nu, R):
    return (1-2*nu)*(-xi/(R*(R+eta)+EPS)) - _J2(xi,eta,q,dip,nu,R)

def _J4(xi, eta, q, dip, nu, R):
    return ((1-2*nu)*(-np.cos(dip)/R - q*np.sin(dip)/(R*(R+eta)+EPS))
            - _J1(xi,eta,q,dip,nu,R))


# ─── Displacement kernels (Okada 1985 eq.25-27) ───────────────────────────────

def _ux_ss(xi, eta, q, dip, nu):
    R=np.sqrt(xi**2+eta**2+q**2); sd=np.sin(dip)
    u=xi*q/(R*(R+eta)+EPS)+_I1(xi,eta,q,dip,nu,R)*sd
    return u+np.where(np.abs(q)>EPS, np.arctan(xi*eta/(q*R+EPS)), 0.)

def _uy_ss(xi, eta, q, dip, nu):
    R=np.sqrt(xi**2+eta**2+q**2); cd=np.cos(dip); sd=np.sin(dip)
    return ((eta*cd+q*sd)*q/(R*(R+eta)+EPS)+q*cd/(R+eta+EPS)+_I2(eta,q,dip,nu,R)*sd)

def _uz_ss(xi, eta, q, dip, nu):
    R=np.sqrt(xi**2+eta**2+q**2); cd=np.cos(dip); sd=np.sin(dip)
    db=eta*sd-q*cd
    return ((eta*sd-q*cd)*q/(R*(R+eta)+EPS)+q*sd/(R+eta+EPS)+_I4(db,eta,q,dip,nu,R)*sd)

def _ux_ds(xi, eta, q, dip, nu):
    R=np.sqrt(xi**2+eta**2+q**2)
    return q/R-_I3(eta,q,dip,nu,R)*np.sin(dip)*np.cos(dip)

def _uy_ds(xi, eta, q, dip, nu):
    R=np.sqrt(xi**2+eta**2+q**2); sd=np.sin(dip); cd=np.cos(dip)
    u=((eta*cd+q*sd)*q/(R*(R+xi)+EPS)-_I1(xi,eta,q,dip,nu,R)*sd*cd)
    return u+np.where(np.abs(q)>EPS, cd*np.arctan(xi*eta/(q*R+EPS)), 0.)

def _uz_ds(xi, eta, q, dip, nu):
    R=np.sqrt(xi**2+eta**2+q**2); sd=np.sin(dip); cd=np.cos(dip)
    db=eta*sd-q*cd
    u=db*q/(R*(R+xi)+EPS)-_I5(xi,eta,q,dip,nu,R,db)*sd*cd
    return u+np.where(np.abs(q)>EPS, sd*np.arctan(xi*eta/(q*R+EPS)), 0.)

def _chinnery(func, xi, p, L, W, q, dip, nu):
    return (func(xi,p,q,dip,nu)-func(xi,p-W,q,dip,nu)
           -func(xi-L,p,q,dip,nu)+func(xi-L,p-W,q,dip,nu))


# ─── Strain kernels (Okada 1985 Table 2) ──────────────────────────────────────

def _uxx_ss(xi, eta, q, dip, nu):
    R=np.sqrt(xi**2+eta**2+q**2)
    return xi**2*q*_A(eta,R)-_J1(xi,eta,q,dip,nu,R)*np.sin(dip)

def _uxy_ss(xi, eta, q, dip, nu):
    R=np.sqrt(xi**2+eta**2+q**2); db=eta*np.sin(dip)-q*np.cos(dip)
    return (xi**3*db/(R**3*(eta**2+q**2)+EPS)
            -(xi**3*_A(eta,R)+_J2(xi,eta,q,dip,nu,R))*np.sin(dip))

def _uyx_ss(xi, eta, q, dip, nu):
    R=np.sqrt(xi**2+eta**2+q**2)
    return (xi*q/R**3*np.cos(dip)
            +(xi*q**2*_A(eta,R)-_J2(xi,eta,q,dip,nu,R))*np.sin(dip))

def _uyy_ss(xi, eta, q, dip, nu):
    R=np.sqrt(xi**2+eta**2+q**2); yb=eta*np.cos(dip)+q*np.sin(dip)
    return (yb*q/R**3*np.cos(dip)
            +(q**3*_A(eta,R)*np.sin(dip)-2*q*np.sin(dip)/(R*(R+eta)+EPS)
              -(xi**2+eta**2)/R**3*np.cos(dip)-_J4(xi,eta,q,dip,nu,R))*np.sin(dip))

def _uxx_ds(xi, eta, q, dip, nu):
    R=np.sqrt(xi**2+eta**2+q**2)
    return xi*q/R**3+_J3(xi,eta,q,dip,nu,R)*np.sin(dip)*np.cos(dip)

def _uxy_ds(xi, eta, q, dip, nu):
    R=np.sqrt(xi**2+eta**2+q**2); yb=eta*np.cos(dip)+q*np.sin(dip)
    return (yb*q/R**3-np.sin(dip)/R+_J1(xi,eta,q,dip,nu,R)*np.sin(dip)*np.cos(dip))

def _uyx_ds(xi, eta, q, dip, nu):
    R=np.sqrt(xi**2+eta**2+q**2); yb=eta*np.cos(dip)+q*np.sin(dip)
    return (yb*q/R**3+q*np.cos(dip)/(R*(R+eta)+EPS)
            +_J1(xi,eta,q,dip,nu,R)*np.sin(dip)*np.cos(dip))

def _uyy_ds(xi, eta, q, dip, nu):
    R=np.sqrt(xi**2+eta**2+q**2); yb=eta*np.cos(dip)+q*np.sin(dip)
    return (yb**2*q*_A(xi,R)
            -(2*yb/(R*(R+xi)+EPS)+xi*np.cos(dip)/(R*(R+eta)+EPS))*np.sin(dip)
            +_J2(xi,eta,q,dip,nu,R)*np.sin(dip)*np.cos(dip))


# ─── Surface displacement (Okada 1985) ────────────────────────────────────────

def okada85_surface(e, n, depth, strike_deg, dip_deg, L, W, rake_deg, slip, U3, nu=0.25):
    """
    Okada (1985) surface displacement.
    e,n,depth,L,W in km; slip,U3 in m. Returns ue,un,uz in m.
    depth = CENTROID depth (km).
    """
    strike=np.deg2rad(strike_deg); dip=np.deg2rad(dip_deg); rake=np.deg2rad(rake_deg)
    cs,ss=np.cos(strike),np.sin(strike); cd,sd=np.cos(dip),np.sin(dip)
    U1=np.cos(rake)*slip; U2=np.sin(rake)*slip

    d=depth+sd*W/2
    ec=e+cs*cd*W/2; nc=n-ss*cd*W/2
    x=cs*nc+ss*ec+L/2; y=ss*nc-cs*ec+cd*W
    p=y*cd+d*sd; q=y*sd-d*cd

    ux=(-U1/(2*np.pi)*_chinnery(_ux_ss,x,p,L,W,q,dip,nu)
        -U2/(2*np.pi)*_chinnery(_ux_ds,x,p,L,W,q,dip,nu))
    uy=(-U1/(2*np.pi)*_chinnery(_uy_ss,x,p,L,W,q,dip,nu)
        -U2/(2*np.pi)*_chinnery(_uy_ds,x,p,L,W,q,dip,nu))
    uz=(-U1/(2*np.pi)*_chinnery(_uz_ss,x,p,L,W,q,dip,nu)
        -U2/(2*np.pi)*_chinnery(_uz_ds,x,p,L,W,q,dip,nu))
    ue=ss*ux-cs*uy; un=cs*ux+ss*uy
    return ue, un, uz


# ─── Surface strain (Okada 1985 Table 2) ─────────────────────────────────────

def okada85_surface_strain(e, n, depth, strike_deg, dip_deg, L, W, rake_deg, slip, nu=0.25):
    """
    Okada (1985) analytical surface strain.
    Returns (exx,exy,eyx,eyy) in geographic (x=E,y=N) frame.
    Standard tension-positive convention. Units: m/km (divide by 1000 before Hooke's law).
    """
    strike=np.deg2rad(strike_deg); dip=np.deg2rad(dip_deg); rake=np.deg2rad(rake_deg)
    cs,ss=np.cos(strike),np.sin(strike); cd,sd=np.cos(dip),np.sin(dip)
    U1=np.cos(rake)*slip; U2=np.sin(rake)*slip

    d=depth+sd*W/2
    ec=e+cs*cd*W/2; nc=n-ss*cd*W/2
    x=cs*nc+ss*ec+L/2; y=ss*nc-cs*ec+cd*W
    p=y*cd+d*sd; q=y*sd-d*cd

    uxx=(-U1/(2*np.pi)*_chinnery(_uxx_ss,x,p,L,W,q,dip,nu)
         -U2/(2*np.pi)*_chinnery(_uxx_ds,x,p,L,W,q,dip,nu))
    uxy=(-U1/(2*np.pi)*_chinnery(_uxy_ss,x,p,L,W,q,dip,nu)
         -U2/(2*np.pi)*_chinnery(_uxy_ds,x,p,L,W,q,dip,nu))
    uyx=(-U1/(2*np.pi)*_chinnery(_uyx_ss,x,p,L,W,q,dip,nu)
         -U2/(2*np.pi)*_chinnery(_uyx_ds,x,p,L,W,q,dip,nu))
    uyy=(-U1/(2*np.pi)*_chinnery(_uyy_ss,x,p,L,W,q,dip,nu)
         -U2/(2*np.pi)*_chinnery(_uyy_ds,x,p,L,W,q,dip,nu))

    # Rotate fault-local → geographic; flip sign (Beauducel +=compression → +=tension)
    s2=np.sin(2*strike)
    unn = cs**2*uxx + s2*(uxy+uyx)/2 + ss**2*uyy
    une = s2*(uxx-uyy)/2 + ss**2*uyx - cs**2*uxy
    uen = s2*(uxx-uyy)/2 - cs**2*uyx + ss**2*uxy
    uee = ss**2*uxx - s2*(uyx+uxy)/2 + cs**2*uyy
    return -uee, -uen, -une, -unn   # (exx,exy,eyx,eyy) tension-positive


# ─── CFF ─────────────────────────────────────────────────────────────────────

def compute_cff(stress: dict, receiver: FaultParameters, friction: float) -> np.ndarray:
    """ΔCFF = Δτ + μ'·Δσn (King et al. 1994). Stress in geographic frame."""
    sr=np.deg2rad(receiver.strike); dr=np.deg2rad(receiver.dip); rr=np.deg2rad(receiver.rake)
    cs,ss=np.cos(sr),np.sin(sr); cd,sd=np.cos(dr),np.sin(dr); cr,sr_=np.cos(rr),np.sin(rr)
    nx=cs*sd; ny=-ss*sd; nz=-cd
    lx=cr*ss-sr_*cs*cd; ly=cr*cs+sr_*ss*cd; lz=-sr_*sd
    sxx=stress['sxx']; syy=stress['syy']; szz=stress['szz']
    sxy=stress['sxy']; sxz=stress['sxz']; syz=stress['syz']
    tx=sxx*nx+sxy*ny+sxz*nz
    ty=sxy*nx+syy*ny+syz*nz
    tz=sxz*nx+syz*ny+szz*nz
    return (tx*lx+ty*ly+tz*lz) + friction*(tx*nx+ty*ny+tz*nz)


# ─── Surface stress ───────────────────────────────────────────────────────────

def _stress_from_surface_strain(e, n, src: FaultParameters, mu, nu):
    """
    Stress tensor at z=0 from Okada (1985) analytical surface strain.
    Exact: szz=sxz=syz=0 (free-surface BC), plane-stress closure.
    Units: e,n,depth,L,W in km; slip in m → strain m/km ÷1000 → dimensionless.
    """
    exx,exy,eyx,eyy=okada85_surface_strain(
        e,n,src.depth,src.strike,src.dip,src.length,src.width,src.rake,src.slip,nu)
    exx/=1000.; exy/=1000.; eyx/=1000.; eyy/=1000.
    exy_s=(exy+eyx)/2
    lam=2*mu*nu/(1-2*nu)
    ezz=-nu/(1-nu)*(exx+eyy); theta=exx+eyy+ezz
    sxx=lam*theta+2*mu*exx; syy=lam*theta+2*mu*eyy; sxy=2*mu*exy_s
    return dict(sxx=sxx,syy=syy,szz=np.zeros_like(sxx),
                sxy=sxy,sxz=np.zeros_like(sxx),syz=np.zeros_like(sxx))


# ─── External Python / DC3D worker (subprocess-based) ───────────────────────
#
# QGIS's bundled Python typically lacks Python.h / python3XX.lib, so
# okada-wrapper (a compiled Fortran extension) CANNOT be built or imported
# inside QGIS's own Python process on most systems. Instead, we call out to
# a separate, standalone Python installation (e.g. a conda/venv environment)
# where okada-wrapper has already been built successfully, via subprocess +
# a small worker script (dc3d_worker.py) that speaks JSON in/out.

def _get_external_python_path():
    """
    Return the path to the external Python interpreter (with okada_wrapper
    installed), as stored in QGIS settings, or None if not configured.
    """
    try:
        from qgis.core import QgsSettings
        path = QgsSettings().value("CoulombStressTransfer/external_python", "", type=str)
        return path if path else None
    except Exception:
        return None


def _set_external_python_path(path):
    """Store the external Python interpreter path in QGIS settings."""
    from qgis.core import QgsSettings
    QgsSettings().setValue("CoulombStressTransfer/external_python", path)


def _dc3d_worker_script_path():
    """Path to the bundled dc3d_worker.py script."""
    return os.path.join(os.path.dirname(__file__), "..", "dc3d_worker", "dc3d_worker.py")


def _clean_subprocess_env():
    """
    Build an environment dict for launching the EXTERNAL Python, with
    QGIS's own Python configuration stripped out.

    QGIS sets PYTHONHOME / PYTHONPATH (and sometimes other PYTHON* vars)
    to point at its own bundled Python installation. If those leak into
    a subprocess running a DIFFERENT Python (e.g. a conda env), that
    Python will try to load QGIS's standard library files instead of its
    own and crash immediately with something like:
        Fatal Python error: init_sys_streams: can't initialize sys
        standard streams ... ImportError: cannot import name
        'text_encoding' from 'io'
    Stripping these lets the external Python initialize using its own,
    self-contained installation as normal.
    """
    env = os.environ.copy()
    for var in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE",
                "PYTHONNOUSERSITE", "PYTHONSTARTUP"):
        env.pop(var, None)
    return env


def check_external_python(python_path):
    """
    Verify that the given Python interpreter exists and has okada_wrapper
    importable. Returns (ok: bool, message: str).
    """
    if not python_path or not os.path.isfile(python_path):
        return False, f"Python interpreter not found at: {python_path}"
    try:
        result = subprocess.run(
            [python_path, "-c", "import okada_wrapper; print('OK')"],
            capture_output=True, text=True, timeout=30,
            env=_clean_subprocess_env(),
        )
        if result.returncode == 0 and "OK" in (result.stdout or ""):
            return True, "okada_wrapper is importable in this Python environment."
        return False, ((result.stderr or "").strip()
                       or "okada_wrapper import failed (no error output).")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _has_okada_wrapper():
    """
    Return True if a configured external Python with okada_wrapper is
    available. This does NOT check QGIS's own Python (which cannot build
    compiled extensions on most systems) — it checks the external
    interpreter registered via the dependency dialog.
    """
    path = _get_external_python_path()
    if not path:
        return False
    ok, _ = check_external_python(path)
    return ok


# ─── DC3D stress via external-Python subprocess ──────────────────────────────

def _run_dc3d_worker(sources, receiver, elastic, grid, z_recv_km):
    """
    Run the external DC3D worker script as a subprocess, passing the job
    as JSON and reading back the resulting CFF grid.

    Returns (lon2d, lat2d, cff_mpa) as numpy arrays, or raises RuntimeError
    with a descriptive message on failure.
    """
    python_path = _get_external_python_path()
    if not python_path:
        raise RuntimeError("No external Python configured for DC3D.")

    job = {
        "mode": "cff",
        "mu": elastic.mu, "nu": elastic.nu, "z_recv_km": z_recv_km,
        "sources": [dict(lon=s.lon, lat=s.lat, depth=s.depth, length=s.length,
                         width=s.width, strike=s.strike, dip=s.dip,
                         rake=s.rake, slip=s.slip) for s in sources],
        "receiver": dict(strike=receiver.strike, dip=receiver.dip,
                         rake=receiver.rake, friction=elastic.friction),
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
        cff_mpa = np.array(payload["cff_mpa"])
        return lon2d, lat2d, cff_mpa


def _run_dc3d_worker_displacement(sources, elastic, grid, z_recv_km):
    """
    Run the external DC3D worker script in displacement mode.

    Returns (lon2d, lat2d, ux_m, uy_m, uz_m) as numpy arrays, or raises
    RuntimeError with a descriptive message on failure.
    """
    python_path = _get_external_python_path()
    if not python_path:
        raise RuntimeError("No external Python configured for DC3D.")

    job = {
        "mode": "displacement",
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
        ux_m = np.array(payload["ux_m"])
        uy_m = np.array(payload["uy_m"])
        uz_m = np.array(payload["uz_m"])
        return lon2d, lat2d, ux_m, uy_m, uz_m


# ─── Slip inversion via external-Python subprocess ────────────────────────

def _run_dc3d_worker_slip_inversion(patches, n_length, n_width, observations,
                                    los_observations, elastic,
                                    smoothing_factor, max_slip, target_mw,
                                    fixed_rake_deg=None, timeout_s=3600):
    """
    Run the external DC3D worker script in "slip_inversion" mode.

    patches           : list of FaultParameters, flat i*n_length+j order
                        (as returned by FaultParameters.subdivide()) --
                        only their geometry is used; slip is ignored.
    observations      : list of dicts {"lon","lat","e","n","u",
                        "sigma_e","sigma_n","sigma_u"} (component-wise,
                        GNSS/leveling-style; e/n/u/sigma_* may be None).
    los_observations  : list of dicts {"lon","lat","los","look_e",
                        "look_n","look_u","sigma"} (InSAR-style).
    target_mw         : float or None (None = no moment constraint).
    fixed_rake_deg    : float or None (None = free 2-unknowns/patch
                        inversion; a float constrains every patch's
                        slip to that single rake, solving for a signed
                        magnitude instead -- see dc3d_worker.py's
                        "slip_inversion" schema docstring for the full
                        rationale/derivation).
    timeout_s         : subprocess wall-clock timeout, seconds (default
                        3600 = 1 hour, raised from the original 900s --
                        Green's-matrix assembly is O(n_patches x
                        n_points); a finely-subdivided fault against a
                        dense point import (tens of thousands of
                        observations) can genuinely take this long even
                        WITH the worker's own internal multiprocessing
                        (dc3d_worker._greens_unit_matrices_mp) helping.
                        Downsampling observations (see
                        core.observation_import.downsample_rows_grid())
                        is the cheaper first lever; raise this only once
                        that's already been tried and the job is still
                        genuinely this large.

    Returns the raw worker payload dict (see dc3d_worker.py module
    docstring "slip_inversion" output schema), or raises RuntimeError
    with a descriptive message on failure.
    """
    python_path = _get_external_python_path()
    if not python_path:
        raise RuntimeError("No external Python configured for DC3D.")

    job = {
        "mode": "slip_inversion",
        "mu": elastic.mu, "nu": elastic.nu,
        "n_length": n_length, "n_width": n_width,
        "patches": [dict(lon=p.lon, lat=p.lat, depth=p.depth, length=p.length,
                         width=p.width, strike=p.strike, dip=p.dip) for p in patches],
        "observations": observations,
        "los_observations": los_observations,
        "smoothing_factor": smoothing_factor,
        "max_slip": max_slip,
        "target_mw": target_mw,
        "fixed_rake_deg": fixed_rake_deg,
    }

    worker_script = os.path.abspath(_dc3d_worker_script_path())

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "job.json")
        out_path = os.path.join(tmpdir, "result.json")
        with open(in_path, "w") as f:
            json.dump(job, f)

        result = subprocess.run(
            [python_path, worker_script, in_path, out_path],
            capture_output=True, text=True, timeout=timeout_s,
            env=_clean_subprocess_env(),
        )

        if not os.path.isfile(out_path):
            raise RuntimeError(
                f"Slip inversion worker produced no output.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}")

        with open(out_path, "r") as f:
            payload = json.load(f)

        if not payload.get("success"):
            raise RuntimeError(f"Slip inversion error: {payload.get('error', 'unknown error')}")

        return payload


def run_slip_inversion(parent_fault, n_length, n_width, observations,
                       los_observations, elastic, smoothing_factor=0.05,
                       max_slip=10.0, target_mw=None, fixed_rake_deg=None,
                       timeout_s=3600):
    """
    High-level driver: subdivide parent_fault (geometry only -- its own
    uniform slip is ignored) and invert scattered surface-displacement
    observations for per-patch slip.

    parent_fault : FaultParameters -- the un-subdivided source fault row.
    n_length,
    n_width      : this row's Subdiv.(L) / Subdiv.(W) values (must match
                   what the caller will pass to
                   FaultTableWidget._set_distributed_slip() afterwards).
    observations,
    los_observations : see _run_dc3d_worker_slip_inversion() above.
                   Either may be an empty list, but not both.
    target_mw    : float or None. None (default) = plain bounded,
                   Laplacian-damped least squares -- no constraint on
                   total moment.
    fixed_rake_deg : float or None. None (default) = independent
                   per-patch (rt-lateral, reverse) slip (2 unknowns/
                   patch). A float constrains every patch to that one
                   rake, solving for a single signed slip magnitude/
                   patch instead -- useful when an independent rake is
                   known (a focal mechanism, plate-motion azimuth, or
                   geologic slip vector on this fault), and directly
                   improves conditioning for otherwise under-determined
                   geometries (vertical-only GNSS, single-track InSAR).

    Returns (overrides, diagnostics):
      overrides    : {(i, j): (rt_lateral_slip, reverse_slip)} dict,
                     ready for FaultTableWidget._set_distributed_slip()
                     -- always this shape, even when fixed_rake_deg was
                     used (each pair then lies exactly on that rake).
      diagnostics  : the raw worker payload (rms_misfit, achieved_mw,
                     solver_success, solver_message, n_data, predicted,
                     observed, component_labels, fixed_rake_deg) for
                     display/QA.
    """
    patches = parent_fault.subdivide(n_length, n_width)  # geometry only
    payload = _run_dc3d_worker_slip_inversion(
        patches, n_length, n_width, observations, los_observations,
        elastic, smoothing_factor, max_slip, target_mw, fixed_rake_deg,
        timeout_s=timeout_s)

    overrides = {}
    flat = 0
    for i in range(n_width):
        for j in range(n_length):
            rt, rev = payload["slip"][flat]
            overrides[(i, j)] = (rt, rev)
            flat += 1

    return overrides, payload


def _run_dc3d_worker_slip_inversion_group(fault_segments, observations,
                                          los_observations, elastic,
                                          smoothing_factor, max_slip, target_mw,
                                          fixed_rake_deg=None, timeout_s=3600):
    """
    Multi-fault-segment counterpart of _run_dc3d_worker_slip_inversion()
    -- see dc3d_worker.py's "slip_inversion_group" mode docstring for
    the exact job/payload schema.

    fault_segments : list of {"n_length": int, "n_width": int,
                              "patches": list[FaultParameters]} dicts,
                     one per fault row in the group, each already
                     subdivided (geometry only -- slip is ignored).
    fixed_rake_deg : float or None -- same meaning as
                     run_slip_inversion(), applied UNIFORMLY to every
                     patch of every segment (see dc3d_worker.py's
                     "slip_inversion_group" schema note on why a
                     per-segment-varying rake isn't supported here).
    """
    python_path = _get_external_python_path()
    if not python_path:
        raise RuntimeError("No external Python configured for DC3D.")

    job = {
        "mode": "slip_inversion_group",
        "mu": elastic.mu, "nu": elastic.nu,
        "fault_segments": [
            {
                "n_length": seg["n_length"], "n_width": seg["n_width"],
                "patches": [dict(lon=p.lon, lat=p.lat, depth=p.depth, length=p.length,
                                 width=p.width, strike=p.strike, dip=p.dip)
                           for p in seg["patches"]],
            }
            for seg in fault_segments
        ],
        "observations": observations,
        "los_observations": los_observations,
        "smoothing_factor": smoothing_factor,
        "max_slip": max_slip,
        "target_mw": target_mw,
        "fixed_rake_deg": fixed_rake_deg,
    }

    worker_script = os.path.abspath(_dc3d_worker_script_path())

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "job.json")
        out_path = os.path.join(tmpdir, "result.json")
        with open(in_path, "w") as f:
            json.dump(job, f)

        result = subprocess.run(
            [python_path, worker_script, in_path, out_path],
            capture_output=True, text=True, timeout=timeout_s,
            env=_clean_subprocess_env(),
        )

        if not os.path.isfile(out_path):
            raise RuntimeError(
                f"Slip inversion (group) worker produced no output.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}")

        with open(out_path, "r") as f:
            payload = json.load(f)

        if not payload.get("success"):
            raise RuntimeError(f"Slip inversion (group) error: {payload.get('error', 'unknown error')}")

        return payload


def run_slip_inversion_group(fault_specs, observations, los_observations,
                             elastic, smoothing_factor=0.05, max_slip=10.0,
                             target_mw=None, fixed_rake_deg=None,
                             timeout_s=3600):
    """
    Multi-fault ("Group") counterpart of run_slip_inversion(): jointly
    inverts the SAME observations against several fault rows at once
    (e.g. rows sharing a non-empty "Group" name in FaultTableWidget --
    several digitized segments with different strikes tracing one bent
    fault), one combined linear system and one combined total-moment
    constraint if target_mw is given, but each fault's own patches are
    Laplacian-smoothed only against their OWN neighbors (no smoothing
    across the strike bend between faults -- see dc3d_worker.py's
    _laplacian_base() docstring for why).

    fault_specs : list of dicts, one per fault row in the group, each:
      {"key": <anything hashable identifying the row, e.g. the row
              index in FaultTableWidget's table -- returned as-is in
              the result dict, not interpreted here>,
       "fault": FaultParameters,   # the row's own un-subdivided geometry
       "n_length": int, "n_width": int}   # that row's Subdiv.(L)/(W)
    observations,
    los_observations,
    elastic, smoothing_factor,
    max_slip, target_mw : same meaning as run_slip_inversion().
    fixed_rake_deg : float or None -- same meaning as
                   run_slip_inversion(), applied uniformly across every
                   patch of every fault in the group (see
                   _run_dc3d_worker_slip_inversion_group() docstring).
    timeout_s      : same meaning as run_slip_inversion()'s timeout_s
                   (default 3600s) -- a joint group solve concatenates
                   every segment's patches into one Green's matrix, so
                   this is at least as slow as an equivalent single-
                   fault job with the same total patch count.

    Returns (overrides_by_key, diagnostics):
      overrides_by_key : {key: {(i, j): (rt_lateral_slip, reverse_slip)}}
                         -- one entry per fault_specs["key"], ready for
                         FaultTableWidget._set_distributed_slip() per row.
      diagnostics       : the raw worker payload (same fields as
                          run_slip_inversion(), plus
                          "segment_patch_counts") for display/QA --
                          rms_misfit/achieved_mw/predicted/observed are
                          for the WHOLE joint solve, not per-fault.
    """
    fault_segments = []
    for spec in fault_specs:
        patches = spec["fault"].subdivide(spec["n_length"], spec["n_width"])
        fault_segments.append({
            "n_length": spec["n_length"], "n_width": spec["n_width"],
            "patches": patches,
        })

    payload = _run_dc3d_worker_slip_inversion_group(
        fault_segments, observations, los_observations,
        elastic, smoothing_factor, max_slip, target_mw, fixed_rake_deg,
        timeout_s=timeout_s)

    overrides_by_key = {}
    flat = 0
    for spec, seg in zip(fault_specs, fault_segments):
        n_length, n_width = spec["n_length"], spec["n_width"]
        overrides = {}
        for i in range(n_width):
            for j in range(n_length):
                rt, rev = payload["slip"][flat]
                overrides[(i, j)] = (rt, rev)
                flat += 1
        overrides_by_key[spec["key"]] = overrides

    return overrides_by_key, payload


# ─── High-level grid drivers ─────────────────────────────────────────────────

def compute_coulomb_grid(sources, receiver: FaultParameters,
                         elastic: ElasticParameters, grid: GridParameters,
                         progress_callback=None) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """ΔCFF (MPa) at z=0. Exact vs Coulomb 3.4.2."""
    lons=np.linspace(grid.lon_min,grid.lon_max,grid.n_lon)
    lats=np.linspace(grid.lat_min,grid.lat_max,grid.n_lat)
    lon2d,lat2d=np.meshgrid(lons,lats)
    cff=np.zeros(lon2d.shape)
    for i,src in enumerate(sources):
        if progress_callback: progress_callback(int(100*i/len(sources)))
        e_km,n_km=geo_to_km(lon2d,lat2d,src.lon,src.lat)
        stress=_stress_from_surface_strain(e_km,n_km,src,elastic.mu,elastic.nu)
        cff+=compute_cff(stress,receiver,elastic.friction)
    if progress_callback: progress_callback(100)
    cff_mpa=cff/1e6
    # NOTE (2026-08-10, clip-removal fix): this used to return
    # np.clip(cff_mpa, -p995, p995) -- a 99.5th-percentile clip applied to
    # the ACTUAL returned/exported array, not just a display color scale.
    # plot_widget.py already computes its own independent 98th-percentile
    # vmin/vmax for color rendering from whatever array it's handed, so
    # this clip bought nothing for plotting and instead silently truncated
    # genuine near-fault ΔCFF values (confirmed against Coulomb 3.4.2 .dat
    # grids: up to ~7x underestimate at points inside the near-field zone)
    # in every downstream consumer of this array -- CSV export, raster
    # export, moment/statistics readouts. Removed; return the raw value.
    return lon2d,lat2d,cff_mpa


def compute_cff_on_receiver_faults(sources, receiver_faults,
                                   elastic: ElasticParameters,
                                   progress_callback=None):
    """
    Resolve ΔCFF on a set of INDIVIDUAL receiver faults, each using its OWN
    strike/dip/rake — matching Coulomb's "specified faults" receiver mode,
    where faults entered in the input table with no slip act as individual
    receivers rather than sources, and stress is resolved on each one's own
    geometry at its own centroid position and depth.

    sources         : list of FaultParameters with slip (the stress SOURCES)
    receiver_faults : list of FaultParameters representing receivers; their
                      .slip is ignored (should be 0) — only geometry
                      (lon, lat, depth, strike, dip, rake) is used.

    Each receiver's stress is evaluated at its own CENTROID position and
    depth. At depth=0 this uses the validated surface formula exactly; at
    depth>0 it requires an external Python with okada-wrapper (falls back
    to the surface formula, per-receiver, with a warning flag otherwise).

    Returns a list of dicts, one per receiver fault, each with:
      {"fault": FaultParameters, "cff_mpa": float, "shear_mpa": float,
       "normal_mpa": float, "used_dc3d": bool}
    """
    results = []
    n = len(receiver_faults)

    # Split receivers by depth: z=0 uses the fast vectorized surface path;
    # z>0 uses DC3D one point at a time (via the cross_section worker mode,
    # which already supports one-point-per-own-depth batches).
    surface_receivers = [(i, r) for i, r in enumerate(receiver_faults) if r.depth <= 0.0]
    deep_receivers = [(i, r) for i, r in enumerate(receiver_faults) if r.depth > 0.0]

    cff_by_index = {}
    shear_by_index = {}
    normal_by_index = {}
    used_dc3d_by_index = {}

    # ── z=0 receivers: validated surface formula ─────────────────────────
    for i, recv in surface_receivers:
        e_km = np.array([0.0])
        n_km = np.array([0.0])
        # receiver's OWN position is the observation point; sources are
        # offset relative to it by using geo_to_km with the receiver as origin
        stress_total = dict(sxx=np.zeros(1), syy=np.zeros(1), szz=np.zeros(1),
                           sxy=np.zeros(1), sxz=np.zeros(1), syz=np.zeros(1))
        for src in sources:
            e_src, n_src = geo_to_km(np.array([recv.lon]), np.array([recv.lat]),
                                     src.lon, src.lat)
            stress = _stress_from_surface_strain(e_src, n_src, src, elastic.mu, elastic.nu)
            for k in stress_total:
                stress_total[k] = stress_total[k] + stress[k]

        cff_val = compute_cff(stress_total, recv, elastic.friction)[0] / 1e6
        # Shear/normal for reporting (King et al. 1994 sign convention)
        sr=np.deg2rad(recv.strike); dr=np.deg2rad(recv.dip); rr=np.deg2rad(recv.rake)
        cs,ss=np.cos(sr),np.sin(sr); cd,sd=np.cos(dr),np.sin(dr); cr,sr_=np.cos(rr),np.sin(rr)
        nx,ny,nz = cs*sd, -ss*sd, -cd
        lx = cr*ss - sr_*cs*cd; ly = cr*cs + sr_*ss*cd; lz = -sr_*sd
        tx = stress_total['sxx']*nx + stress_total['sxy']*ny + stress_total['sxz']*nz
        ty = stress_total['sxy']*nx + stress_total['syy']*ny + stress_total['syz']*nz
        tz = stress_total['sxz']*nx + stress_total['syz']*ny + stress_total['szz']*nz
        dtau = (tx*lx + ty*ly + tz*lz)[0] / 1e6
        dsn = (tx*nx + ty*ny + tz*nz)[0] / 1e6

        cff_by_index[i] = float(cff_val)
        shear_by_index[i] = float(dtau)
        normal_by_index[i] = float(dsn)
        used_dc3d_by_index[i] = False

    # ── z>0 receivers: DC3D via cross_section-style point evaluation ─────
    if deep_receivers and _has_okada_wrapper():
        for i, recv in deep_receivers:
            points = [(recv.lon, recv.lat, recv.depth)]
            try:
                cff_arr, shear_arr, normal_arr = _run_dc3d_worker_cross_section(
                    sources, recv, elastic, points)
                cff_by_index[i] = float(cff_arr[0])
                shear_by_index[i] = float(shear_arr[0])
                normal_by_index[i] = float(normal_arr[0])
                used_dc3d_by_index[i] = True
            except Exception:
                # Fall back to surface formula for this receiver, at its own
                # (lon,lat), ignoring depth — clearly flagged via used_dc3d.
                stress_total = dict(sxx=np.zeros(1), syy=np.zeros(1), szz=np.zeros(1),
                                   sxy=np.zeros(1), sxz=np.zeros(1), syz=np.zeros(1))
                for src in sources:
                    e_src, n_src = geo_to_km(np.array([recv.lon]), np.array([recv.lat]),
                                             src.lon, src.lat)
                    stress = _stress_from_surface_strain(e_src, n_src, src, elastic.mu, elastic.nu)
                    for k in stress_total:
                        stress_total[k] = stress_total[k] + stress[k]
                cff_val = compute_cff(stress_total, recv, elastic.friction)[0] / 1e6
                cff_by_index[i] = float(cff_val)
                shear_by_index[i] = float('nan')
                normal_by_index[i] = float('nan')
                used_dc3d_by_index[i] = False
    else:
        for i, recv in deep_receivers:
            # No external Python: fall back to surface formula (depth ignored),
            # clearly flagged so the caller can warn the user.
            stress_total = dict(sxx=np.zeros(1), syy=np.zeros(1), szz=np.zeros(1),
                               sxy=np.zeros(1), sxz=np.zeros(1), syz=np.zeros(1))
            for src in sources:
                e_src, n_src = geo_to_km(np.array([recv.lon]), np.array([recv.lat]),
                                         src.lon, src.lat)
                stress = _stress_from_surface_strain(e_src, n_src, src, elastic.mu, elastic.nu)
                for k in stress_total:
                    stress_total[k] = stress_total[k] + stress[k]
            cff_val = compute_cff(stress_total, recv, elastic.friction)[0] / 1e6
            cff_by_index[i] = float(cff_val)
            shear_by_index[i] = float('nan')
            normal_by_index[i] = float('nan')
            used_dc3d_by_index[i] = False

    for i, recv in enumerate(receiver_faults):
        if progress_callback: progress_callback(int(100 * (i + 1) / max(n, 1)))
        results.append({
            "fault": recv,
            "cff_mpa": cff_by_index[i],
            "shear_mpa": shear_by_index[i],
            "normal_mpa": normal_by_index[i],
            "used_dc3d": used_dc3d_by_index[i],
        })

    return results


def compute_coulomb_grid_depth(sources, receiver: FaultParameters,
                                elastic: ElasticParameters, grid: GridParameters,
                                progress_callback=None):
    """
    ΔCFF (MPa) at depth grid.depth_km.

    Uses the external-Python DC3D worker (Okada 1992) if one has been
    configured and verified working; otherwise falls back to the
    validated z=0 surface formula (Okada 1985).

    Returns (lon2d, lat2d, cff_mpa, used_dc3d, near_field_mask).

    near_field_mask (added 2026-08-09b): boolean array, same shape as
    cff_mpa, True where the point is within near_field_threshold_km(...)
    of a source fault's surface trace -- see the near-field note above
    geo_to_km/km_to_geo for why these values are unreliable rather than
    wrong. BREAKING CHANGE: this function used to return 4 values.
    """
    use_dc3d = _has_okada_wrapper() and grid.depth_km > 0.
    if not use_dc3d:
        lon2d,lat2d,cff=compute_coulomb_grid(sources,receiver,elastic,grid,progress_callback)
        return lon2d,lat2d,cff,False,near_field_grid_mask(sources,lon2d,lat2d)

    if progress_callback: progress_callback(10)
    lon2d, lat2d, cff_mpa = _run_dc3d_worker(sources, receiver, elastic, grid, grid.depth_km)
    if progress_callback: progress_callback(100)

    # NOTE (2026-08-10, clip-removal fix): see matching note in
    # compute_coulomb_grid() above -- the former np.clip(cff_mpa, -p995,
    # p995) here clipped the actual returned DC3D-path array, not just a
    # plot color scale, and has been removed for the same reason.
    near_field_mask = near_field_grid_mask(sources, lon2d, lat2d, depth_km=grid.depth_km)
    return lon2d, lat2d, cff_mpa, True, near_field_mask


def compute_surface_deformation(sources, elastic: ElasticParameters,
                                 grid: GridParameters,
                                 progress_callback=None):
    """Returns lon2d, lat2d, ux(E), uy(N), uz in m. Always at z=0 (surface)."""
    lons=np.linspace(grid.lon_min,grid.lon_max,grid.n_lon)
    lats=np.linspace(grid.lat_min,grid.lat_max,grid.n_lat)
    lon2d,lat2d=np.meshgrid(lons,lats)
    ux=np.zeros(lon2d.shape); uy=np.zeros_like(ux); uz=np.zeros_like(ux)
    for i,src in enumerate(sources):
        if progress_callback: progress_callback(int(100*i/len(sources)))
        e_km,n_km=geo_to_km(lon2d,lat2d,src.lon,src.lat)
        ue,un,uz_=okada85_surface(e_km,n_km,src.depth,src.strike,src.dip,
                                   src.length,src.width,src.rake,src.slip,0.,elastic.nu)
        ux+=ue; uy+=un; uz+=uz_
    if progress_callback: progress_callback(100)
    return lon2d,lat2d,ux,uy,uz


def compute_surface_deformation_depth(sources, elastic: ElasticParameters,
                                       grid: GridParameters,
                                       progress_callback=None):
    """
    Displacement (ue, un, uz in m) at depth grid.depth_km.

    At z=0, uses the validated Okada (1985) surface formula directly
    (exact vs Coulomb 3.4.2). At z>0, requires the external-Python DC3D
    worker (Okada 1992); falls back to the z=0 formula (with a warning
    flag) if no working external Python is configured.

    Returns (lon2d, lat2d, ux, uy, uz, used_dc3d).
    """
    if grid.depth_km <= 0.0:
        lon2d, lat2d, ux, uy, uz = compute_surface_deformation(
            sources, elastic, grid, progress_callback)
        return lon2d, lat2d, ux, uy, uz, False

    if not _has_okada_wrapper():
        lon2d, lat2d, ux, uy, uz = compute_surface_deformation(
            sources, elastic, grid, progress_callback)
        return lon2d, lat2d, ux, uy, uz, False

    if progress_callback: progress_callback(10)
    lon2d, lat2d, ux, uy, uz = _run_dc3d_worker_displacement(
        sources, elastic, grid, grid.depth_km)
    if progress_callback: progress_callback(100)
    return lon2d, lat2d, ux, uy, uz, True


def _run_dc3d_worker_cross_section(sources, receiver, elastic, points):
    """
    Run the external DC3D worker in cross_section mode: each point has its
    own (lon, lat, z_km). Returns (cff_mpa, shear_mpa, normal_mpa), each a
    flat numpy array with one value per input point, in the same order.
    """
    python_path = _get_external_python_path()
    if not python_path:
        raise RuntimeError("No external Python configured for DC3D.")

    job = {
        "mode": "cross_section",
        "mu": elastic.mu, "nu": elastic.nu,
        "sources": [dict(lon=s.lon, lat=s.lat, depth=s.depth, length=s.length,
                         width=s.width, strike=s.strike, dip=s.dip,
                         rake=s.rake, slip=s.slip) for s in sources],
        "receiver": dict(strike=receiver.strike, dip=receiver.dip,
                         rake=receiver.rake, friction=elastic.friction),
        "points": [dict(lon=p[0], lat=p[1], z_km=p[2]) for p in points],
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

        cff_mpa = np.array(payload["cff_mpa"])
        shear_mpa = np.array(payload["shear_mpa"]) if "shear_mpa" in payload else np.full_like(cff_mpa, np.nan)
        normal_mpa = np.array(payload["normal_mpa"]) if "normal_mpa" in payload else np.full_like(cff_mpa, np.nan)
        return cff_mpa, shear_mpa, normal_mpa


def _run_dc3d_worker_cross_section_stress_tensor(sources, elastic, points):
    """
    Run the external DC3D worker in "cross_section_stress_tensor" mode
    (2026-08-21 addition): same points-with-own-depth schema as
    _run_dc3d_worker_cross_section() above, but returns the raw
    6-component stress tensor per point (Pa, tension-positive,
    East/North/Down frame) instead of a value resolved onto a fixed
    receiver -- the cross-section counterpart of
    optimal_plane._run_dc3d_worker_stress_tensor()'s map-view grid
    version, needed so core.optimal_plane.compute_cross_section_optimal()
    can add the regional stress and eigendecompose per point itself.

    Returns (sxx_pa, syy_pa, szz_pa, sxy_pa, sxz_pa, syz_pa), each a flat
    numpy array with one value per input point, in the same order.
    """
    python_path = _get_external_python_path()
    if not python_path:
        raise RuntimeError("No external Python configured for DC3D.")

    job = {
        "mode": "cross_section_stress_tensor",
        "mu": elastic.mu, "nu": elastic.nu,
        "sources": [dict(lon=s.lon, lat=s.lat, depth=s.depth, length=s.length,
                         width=s.width, strike=s.strike, dip=s.dip,
                         rake=s.rake, slip=s.slip) for s in sources],
        "points": [dict(lon=p[0], lat=p[1], z_km=p[2]) for p in points],
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

        return (np.array(payload["sxx_pa"]), np.array(payload["syy_pa"]),
               np.array(payload["szz_pa"]), np.array(payload["sxy_pa"]),
               np.array(payload["sxz_pa"]), np.array(payload["syz_pa"]))


def compute_cross_section(sources, receiver: FaultParameters,
                          elastic: ElasticParameters,
                          lon1, lat1, lon2, lat2,
                          dist_increment_km, max_depth_km, depth_increment_km,
                          progress_callback=None):
    """
    Compute a vertical Coulomb-stress cross-section below a surface profile
    line, mirroring Coulomb 3.x/4.0's cross-section tool.

    lon1,lat1 -> lon2,lat2 : profile line endpoints (start -> finish)
    dist_increment_km      : sampling step along the profile (km)
    max_depth_km           : maximum depth of the section (km, positive down)
    depth_increment_km     : sampling step in depth (km)

    Requires a working external Python (DC3D) for any row below the
    surface (z>0). The z=0 row always uses the validated surface formula.

    Returns (dist_km, depth_km, cff_2d, used_dc3d) where:
      dist_km  : 1D array, distance along profile (km), length n_dist
      depth_km : 1D array, depth (km, positive down), length n_depth
      cff_2d   : 2D array, shape (n_depth, n_dist), ΔCFF in MPa
      used_dc3d: True if rows below the surface used DC3D; False if the
                 section is surface-only (no external Python configured,
                 so only the z=0 row is populated and the rest are NaN)
    """
    total_dist = float(np.hypot(*geo_to_km(lon2, lat2, lon1, lat1)))
    n_dist = max(2, int(round(total_dist / dist_increment_km)) + 1)
    dist_km = np.linspace(0.0, total_dist, n_dist)

    n_depth = max(2, int(round(max_depth_km / depth_increment_km)) + 1)
    depth_km = np.linspace(0.0, max_depth_km, n_depth)

    # Profile points in (lon, lat) at each distance step
    frac = dist_km / total_dist if total_dist > 0 else np.zeros_like(dist_km)
    profile_lons = lon1 + frac * (lon2 - lon1)
    profile_lats = lat1 + frac * (lat2 - lat1)

    cff_2d = np.full((n_depth, n_dist), np.nan)

    # z=0 row: always available via the validated surface formula
    e_km0, n_km0 = geo_to_km(profile_lons, profile_lats, sources[0].lon, sources[0].lat) \
        if sources else (np.zeros(n_dist), np.zeros(n_dist))
    cff_row0 = np.zeros(n_dist)
    for src in sources:
        e_km, n_km = geo_to_km(profile_lons, profile_lats, src.lon, src.lat)
        stress = _stress_from_surface_strain(e_km, n_km, src, elastic.mu, elastic.nu)
        cff_row0 += compute_cff(stress, receiver, elastic.friction)
    cff_2d[0, :] = cff_row0 / 1e6

    used_dc3d = False
    if _has_okada_wrapper() and n_depth > 1:
        used_dc3d = True
        points = []
        for d_idx in range(1, n_depth):
            for x_idx in range(n_dist):
                points.append((float(profile_lons[x_idx]), float(profile_lats[x_idx]),
                              float(depth_km[d_idx])))
        if progress_callback: progress_callback(20)
        try:
            cff_flat, _shear_flat, _normal_flat = _run_dc3d_worker_cross_section(
                sources, receiver, elastic, points)
            cff_2d[1:, :] = cff_flat.reshape(n_depth - 1, n_dist)
        except Exception:
            used_dc3d = False   # leave rows as NaN; caller can detect via used_dc3d
        if progress_callback: progress_callback(100)
    else:
        if progress_callback: progress_callback(100)

    return dist_km, depth_km, cff_2d, used_dc3d


def compute_cross_section_multi(sources, receiver: FaultParameters,
                                elastic: ElasticParameters, vertices,
                                dist_increment_km, max_depth_km, depth_increment_km,
                                progress_callback=None):
    """
    Multi-segment counterpart of compute_cross_section() (2026-08-21,
    "cross section with several segments with several strikes" --
    request item 5). vertices is a list of (lon, lat) tuples, length
    >= 2, defining a profile made of one or more straight legs instead
    of a single lon1,lat1->lon2,lat2 segment.

    Implementation: calls the existing, already-validated
    compute_cross_section() once per leg (so no new physics -- same
    per-project convention as core.cross_section_faults reusing
    core.fault_geometry rather than re-deriving fault geometry a second
    time), then concatenates each leg's dist_km (shifted by that leg's
    cumulative start distance) and cff_2d columns end-to-end. depth_km
    is shared/identical across legs by construction (same
    max_depth_km/depth_increment_km passed to every leg).

    Returns (dist_km, depth_km, cff_2d, used_dc3d, segment_info) --
    the same 4-tuple compute_cross_section() returns, PLUS
    segment_info (core.geo_profile.polyline_segment_info()'s dict) so
    the caller/plot can mark segment boundaries and label each leg's
    strike. used_dc3d is True only if EVERY leg's own used_dc3d was
    True (a single leg falling back to surface-only should still warn
    the user, same as the single-segment path).

    A 2-vertex `vertices` list degenerates to one call to
    compute_cross_section() with a segment_info describing that single
    leg -- so callers can route ALL cross-section computations through
    this function uniformly rather than branching on segment count.
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

        leg_dist, leg_depth, leg_cff, leg_used_dc3d = compute_cross_section(
            sources, receiver, elastic, lon_a, lat_a, lon_b, lat_b,
            dist_increment_km, max_depth_km, depth_increment_km,
            progress_callback=_leg_progress,
        )
        if depth_km is None:
            depth_km = leg_depth
        used_dc3d = used_dc3d and leg_used_dc3d

        offset = seg_info["cumulative_dist_km"][leg_i]
        # Drop the first column of every leg AFTER the first one -- it
        # exactly coincides with the previous leg's last column (both
        # are the shared vertex between the two legs), so keeping both
        # would duplicate that sample instead of giving one continuous
        # profile.
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
