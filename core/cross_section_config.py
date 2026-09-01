# -*- coding: utf-8 -*-
"""
Configuration schema for the configurable cross-section tool
(PROJECT_HANDOVER_ADDENDUM_2026-08-18b_cross_section_overhaul.md).

Plain dataclasses, not tied to PyQt/QGIS -- importable and testable in
the sandbox with no qgis.PyQt available (see project constraint). The
UI layer (a future cross_section_config_dialog.py) reads/writes these;
core.cross_section_plot consumes them to build a matplotlib Figure.

Every field has a default so `CrossSectionConfig()` alone is a valid,
reasonable "topo + CFF panel only" configuration -- new overlays are
added by turning fields on, not by requiring the caller to fill out
the whole schema.

Focal-mechanism side-view overlay config (FocalMechanismOverlayConfig,
below) was deliberately withheld from the first pass of this module --
drawing a beachball "as seen from the side" (looking along the profile
azimuth into the vertical plane) is a moment-tensor rotation problem,
not a plotting-options problem, and per this project's physics-before-
code rule needed its own derivation before any config knobs were added
for it. That derivation now lives in core.focal_side_view (Phase 2,
2026-08-19 continuation of this addendum); this module just exposes
its plotting-facing knobs, the same way FaultOverlayConfig exposes
core.cross_section_faults's.
"""

from dataclasses import dataclass, field, asdict, fields
from typing import Optional, List, Tuple


@dataclass
class EQOverlayConfig:
    """Earthquake catalog overlay (core.eq_catalog_import.EQCatalogEvent)."""
    enabled: bool = False
    search_width_km: float = 10.0          # perpendicular half-width swath
    color_by: str = "depth"                # "depth" | "none" (single color)
    single_color: str = "0.4"
    cmap: str = "turbo_r"                  # reversed: shallow=warm, deep=cool
                                            # (matches the reference GMT figure)
    size_by: str = "magnitude"             # "magnitude" | "fixed"
    fixed_size_pt2: float = 14.0           # marker area (pt^2) if size_by="fixed"
    mag_size_min_pt2: float = 4.0          # area at the catalog's smallest Mw
    mag_size_max_pt2: float = 120.0        # area at the catalog's largest Mw
    marker: str = "o"
    alpha: float = 0.75
    edgecolor: str = "black"
    edge_linewidth: float = 0.25
    zorder: int = 5


@dataclass
class FaultOverlayConfig:
    """Source-fault trace overlay, projected from the 3D fault table."""
    enabled: bool = True
    search_width_km: float = 15.0          # faults whose centroid trace
                                            # passes within this of the
                                            # profile are drawn
    color: str = "black"
    linewidth: float = 1.8
    linestyle: str = "-"
    label_sources: bool = True             # annotate which is the source fault
    label_fontsize: float = 7.0
    zorder: int = 6


@dataclass
class ExtraSectionLineConfig:
    """
    A user-imported line to draw in the main depth-section panel,
    independent of the fault table (request item 7 -- "import other
    elements in the depth section, like the subducting slab interface
    or other faults"). Two source kinds, matching `source_kind`:

      "vector_line" -- each vertex carries its OWN depth (unlike a
                       fault trace, which is derived from a fault
                       rectangle's geometry), typically digitized from
                       a QGIS line layer with a depth/elevation
                       attribute field, e.g. a hand-picked slab-
                       interface trace or a previously-published
                       fault's reflection-seismic geometry. Uses
                       `vertices` (below); the raster_* fields are
                       unused.

      "raster_file" / "qgis_layer" -- the element is instead SAMPLED
                       from a raster (e.g. a Slab2 slab-interface depth
                       grid, or any other raster-format subducting-
                       slab/horizon dataset -- these are commonly
                       distributed as rasters rather than digitized
                       vector lines) along the cross-section's OWN
                       profile line, the same way a topo panel samples
                       elevation -- see TopoPanelConfig and
                       core.raster_profile.sample_raster_along_polyline().
                       Because it's resampled from `raster_source` at
                       compute time using whatever the CURRENT profile
                       is, it automatically tracks edits to the
                       start/finish/waypoints, unlike a "vector_line"
                       element's fixed, one-time-imported vertices.
                       `vertices` is unused for these kinds.

    vertices : list of (lon, lat, depth_km) tuples, length >= 2, in
              along-line order, used only when source_kind="vector_line".
              Projected onto the (possibly multi-segment) profile via
              core.geo_profile.project_points_to_polyline() -- each
              vertex's own depth_km is kept as-is; only its (lon, lat)
              determines its along-profile x-position.
    """
    label: str = ""
    source_kind: str = "vector_line"       # "vector_line" | "raster_file" | "qgis_layer"
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)

    # -- raster_file / qgis_layer fields (unused for "vector_line") --
    raster_source: Optional[str] = None    # file path, or QGIS layer id
                                            # (resolved by the UI layer,
                                            # same contract as
                                            # TopoPanelConfig.source)
    raster_band: int = 1
    raster_n_samples: int = 300
    raster_unit_divisor: float = 1.0       # raster units -> km (1.0 if
                                            # already km, e.g. most
                                            # Slab2 grids; 1000.0 for a
                                            # metres-unit raster)
    raster_sign: float = 1.0               # +1.0 if raster values are
                                            # already positive-DOWN
                                            # depth; -1.0 to flip a
                                            # positive-UP elevation (or
                                            # a negative-down depth,
                                            # e.g. many Slab2 grids)
                                            # to this plugin's positive-
                                            # down convention

    color: str = "saddlebrown"
    linewidth: float = 1.5
    linestyle: str = "--"
    search_width_km: float = 15.0          # vertices farther than this
                                            # perpendicular to the profile
                                            # are dropped (same idea as
                                            # FaultOverlayConfig.search_
                                            # width_km) -- applies only to
                                            # source_kind="vector_line";
                                            # raster-sampled points sit
                                            # exactly on the profile by
                                            # construction
    show_label: bool = True
    label_fontsize: float = 7.0
    zorder: int = 5


@dataclass
class CFFMeshConfig:
    """
    The ΔCFF color mesh itself (the pcolormesh), as distinct from
    CFFContourConfig's contour LINES drawn on top of it. Previously the
    mesh was drawn unconditionally with no config knob at all -- this
    is what lets the mesh be hidden (contour-lines-only view) or shown
    on its own (no contours), matching request item 1 of the
    2026-08-19 cross-section follow-up.

    `interpolate` is a DISPLAY-ONLY smoothing of the already-computed
    cff_2d grid -- it never changes dist_increment_km/depth_increment_km
    (the actual DC3D sampling resolution, set on the Cross-Section tab),
    only how densely the existing samples are re-gridded before
    pcolormesh draws them. Physics stays at whatever resolution the
    user computed; this just controls how coarse/blocky the picture
    looks. See core.cross_section_plot._maybe_interpolate_mesh().
    """
    enabled: bool = True
    cmap: str = "RdBu_r"
    vmin_mpa: Optional[float] = None       # None -> auto (98th-pct |CFF|,
    vmax_mpa: Optional[float] = None       # symmetric); set both to fix
                                            # the scale explicitly
    color_scale_percentile: float = 98.0   # percentile used for the
                                            # auto vmin/vmax when the
                                            # above are None
    interpolate: bool = False              # True -> smooth (spline)
                                            # upsampling of the display
                                            # mesh; False -> draw exactly
                                            # the computed grid ("auto"
                                            # shading)
    interpolation_factor: int = 4          # upsampling factor per axis
                                            # when interpolate=True
    zorder: int = 3


@dataclass
class CFFContourConfig:
    """Contour overlay on top of the ΔCFF color mesh."""
    enabled: bool = False
    levels: Optional[List[float]] = None   # None -> auto (baseline +
                                            # spacing, or n_levels if
                                            # neither is set)
    n_levels: int = 7
    baseline_mpa: Optional[float] = None   # None -> auto (0, i.e.
                                            # symmetric about zero);
                                            # set to offset the contour
                                            # ladder, e.g. start at a
                                            # specific ΔCFF value
    spacing_mpa: Optional[float] = None    # None -> auto (n_levels
                                            # divides the data range);
                                            # set to fix the contour
                                            # interval explicitly
                                            # (baseline_mpa + k*spacing_mpa)
    color: str = "black"
    linewidth: float = 0.5
    alpha: float = 0.8
    inline_labels: bool = True
    label_fontsize: float = 6.0
    fmt: str = "%.2f"
    zorder: int = 4


@dataclass
class TopoPanelConfig:
    """One stacked topographic-profile panel, drawn above the main section.

    Multiple instances of this in CrossSectionConfig.topo_panels give
    the "add another plot of profile above existing plot" / "two or
    more profiles" behavior requested -- e.g. one panel per raster
    (elevation vs. gravity anomaly vs. a second DEM).
    """
    label: str = "Elevation (km)"
    source_kind: str = "raster_file"       # "raster_file" | "qgis_layer"
    source: Optional[str] = None           # file path, or QGIS layer id/name
                                            # (resolved by the UI layer before
                                            # calling core.raster_profile)
    band: int = 1
    n_samples: int = 300
    elevation_unit_divisor: float = 1000.0  # raster units -> km (1000 for m)
    color: str = "#5b3a29"
    fill: bool = True
    fill_alpha: float = 0.25
    linewidth: float = 1.2
    height_ratio: float = 0.35             # relative to the main panel (=1.0)
    vertical_exaggeration: Optional[float] = None  # None -> auto (square-ish)


@dataclass
class AnnotationOverlayConfig:
    """Point-feature annotations on a topo panel (e.g. a volcano marker).

    `z_field` (new): an optional column/attribute giving each point's
    own vertical value (elevation, depth, whatever the target topo
    panel's y-axis represents) so the marker plots at its actual
    position instead of a fixed offset near the panel's top. When
    None (the default -- unchanged prior behavior), markers still draw
    pinned near the top of the panel's current y-range, since not
    every annotation source (e.g. a plain lon/lat point layer with no
    elevation attribute) has a meaningful vertical value to plot at.
    """
    enabled: bool = False
    label: str = "Annotations"
    source_kind: str = "qgis_layer"        # "file" | "qgis_layer"
    source: Optional[str] = None
    label_field: Optional[str] = None
    z_field: Optional[str] = None          # None -> pin near panel top
                                            # (old behavior); else read
                                            # this column/attribute as
                                            # the marker's y-position
    search_width_km: float = 5.0
    topo_panel_index: int = 0              # which topo panel to draw on
    marker: str = "^"
    color: str = "darkred"
    size_pt: float = 9.0
    label_fontsize: float = 7.0
    zorder: int = 7


@dataclass
class FocalMechanismOverlayConfig:
    """
    Focal-mechanism side-view overlay on the main depth section.

    See core.focal_side_view for the derivation that rotates each
    event's (strike, dip, rake) into the "apparent" triple that
    core.beachball.draw_beachball() renders correctly for this side
    view (picture axes = along-profile distance, depth; viewing axis =
    horizontal, perpendicular to the profile).
    """
    enabled: bool = False
    search_width_km: float = 15.0          # same swath-filtering role
                                            # as EQOverlayConfig's
    plane: str = "selected"                # "plane1" | "plane2" | "selected"
                                            # -- WHICH of the two
                                            # inherently-ambiguous nodal
                                            # planes to draw (a moment
                                            # tensor alone can't tell
                                            # fault plane from auxiliary
                                            # plane; PLANE_MODES is the
                                            # physical disambiguation,
                                            # same choice bb2/Coulomb
                                            # makes). This is INDEPENDENT
                                            # of profile_direction, which
                                            # only rotates whichever
                                            # plane is chosen here into
                                            # the side-view's 2D
                                            # projection (see
                                            # core.focal_side_view).
                                            # "selected" reads each
                                            # event's own resolved
                                            # PLANE_MODES choice, if
                                            # supplied by the caller;
                                            # falls back to plane1 if
                                            # not available.

    # -- Size --
    size_by: str = "fixed"                 # "fixed" | "magnitude"
    diameter_km: float = 3.0               # symbol diameter (km, main
                                            # panel data units) if
                                            # size_by="fixed"
    mag_diameter_min_km: float = 1.0       # diameter at the smallest Mw
    mag_diameter_max_km: float = 8.0       # diameter at the largest Mw

    # -- Color --
    color_by: str = "cff"                  # "cff" | "single" | "type" | "depth"
    single_color: str = "0.3"
    cmap: str = "RdBu_r"                   # used when color_by="cff"
                                            # (matches core.beachball.cff_to_color)
    vmin_mpa: Optional[float] = None       # None -> auto (98th-pct |CFF|
    vmax_mpa: Optional[float] = None       # across this overlay's own
                                            # events); set both to
                                            # narrow the scale so small
                                            # real ΔCFF values don't all
                                            # wash out near white on a
                                            # scale dominated by a few
                                            # large outliers
    depth_cmap: str = "turbo_r"            # used when color_by="depth"
                                            # (matches EQOverlayConfig's
                                            # default, for visual
                                            # consistency between the EQ
                                            # and focal-mechanism depth
                                            # colorings)
    bgcolor: str = "white"                 # complementary (dilatational)
                                            # lobe color -- avoid pure
                                            # white if color_by="cff" and
                                            # low-|CFF| events matter,
                                            # since the compressional
                                            # lobe will also read near-
                                            # white against it (a light
                                            # gray, e.g. "0.85", keeps
                                            # the outline/lobe split
                                            # visible even at ΔCFF≈0)
    edgecolor: str = "black"
    highlight_selected_plane: bool = True  # dark arc over whichever
                                            # plane was actually used,
                                            # same as the Focal
                                            # Mechanisms tab's beachballs

    # -- Labels --
    show_labels: bool = False
    label_source: str = "magnitude"        # "magnitude" | "custom" --
                                            # "magnitude" formats each
                                            # event's own Mw (via
                                            # label_fmt); "custom" uses
                                            # whatever string the caller
                                            # put in focal_mechanism_data
                                            # ["label"] (e.g. an event ID)
    label_fmt: str = "M%.1f"
    label_fontsize: float = 7.0
    label_offset_km: Optional[float] = None  # None -> auto (0.9x diameter);
                                            # gap between the symbol's
                                            # edge and its label, in the
                                            # main panel's own data units
                                            # (km) -- see
                                            # core.cross_section_plot's
                                            # 2026-08-21 label-placement
                                            # fix docstring for why this
                                            # must be data units, not
                                            # points.
    label_leader_line: bool = False        # draw a thin line from the
                                            # beachball to its label --
                                            # turn on once labels start
                                            # crowding/overlapping
                                            # neighboring symbols
    zorder: int = 8

    # "type" classification (strike-slip/normal/reverse/oblique,
    # Frohlich 1992 P/T/B-plunge scheme -- see core.focal_mechanism)
    type_colors: Optional[dict] = None     # None -> use
                                            # core.focal_mechanism.
                                            # FAULT_TYPE_COLORS (the WSM-
                                            # standard red/green/blue +
                                            # orange-for-oblique
                                            # default, 2026-08-21). Set
                                            # a dict with any of the
                                            # keys "normal"/"reverse"/
                                            # "strike-slip"/"oblique" to
                                            # override just those --
                                            # missing keys still fall
                                            # back to the default color
                                            # for that type. Same
                                            # per-overlay-configurable
                                            # idea as EQOverlayConfig's
                                            # cmap / this same config's
                                            # depth_cmap, just for a
                                            # discrete-category palette
                                            # instead of a continuous
                                            # colormap.


@dataclass
class LegendConfig:
    enabled: bool = True
    loc: str = "outside_right"             # "outside_right" or any matplotlib loc
    fontsize: float = 7.0


@dataclass
class LayoutConfig:
    figsize: Tuple[float, float] = (10.0, 7.5)
    title: Optional[str] = None
    main_vertical_exaggeration: float = 1.0   # depth-vs-distance panel (CFF/EQ/faults)
    main_height_ratio: float = 1.0
    hspace: float = 0.08
    dpi: int = 150


@dataclass
class CrossSectionConfig:
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    topo_panels: List[TopoPanelConfig] = field(default_factory=list)
    annotations: List[AnnotationOverlayConfig] = field(default_factory=list)
    mesh: CFFMeshConfig = field(default_factory=CFFMeshConfig)
    eq: EQOverlayConfig = field(default_factory=EQOverlayConfig)
    fault: FaultOverlayConfig = field(default_factory=FaultOverlayConfig)
    extra_lines: List[ExtraSectionLineConfig] = field(default_factory=list)
    contours: CFFContourConfig = field(default_factory=CFFContourConfig)
    focal_mechanisms: FocalMechanismOverlayConfig = field(
        default_factory=FocalMechanismOverlayConfig)
    legend: LegendConfig = field(default_factory=LegendConfig)
    cff_source: str = "receiver"           # "receiver" | "optimal" -- which
                                            # ΔCFF the main panel displays.
                                            # "receiver": resolved onto the
                                            # single fixed Receiver Fault
                                            # tab orientation (the ONLY
                                            # option before 2026-08-21;
                                            # okada_engine.compute_cross_
                                            # section()). "optimal":
                                            # resolved onto the
                                            # optimally-oriented plane at
                                            # each point given the
                                            # Regional Stress tab's
                                            # tensor (King/Stein/Lin
                                            # 1994; optimal_plane.
                                            # compute_cross_section_
                                            # optimal()) -- requires a
                                            # regional stress to be set,
                                            # same requirement as the
                                            # "Opt Faults" map-view tab.

    def add_topo_panel(self, **kwargs) -> TopoPanelConfig:
        panel = TopoPanelConfig(**kwargs)
        self.topo_panels.append(panel)
        return panel


# ─── Serialization (2026-08-24, "save setup" dialog-settings coverage) ─────
#
# Plain-dataclass round-trip to/from a JSON-serializable dict, so the
# full display-symbology config edited via CrossSectionConfigDialog
# (colors/sizes/cmaps/z-order for every overlay, topo-panel and
# annotation-source cosmetic fields, legend placement -- everything
# this module defines) can be captured by ui.project_io's native JSON
# "setup" file, not just the handful of coordinate/increment fields
# main_dialog.py's own tabs expose directly. Previously this config
# object was persistent for the LIFE OF THE DIALOG (main_dialog.py
# keeps one self.xs_config instance and reuses it across repeated
# CrossSectionConfigDialog openings) but was never written to disk,
# so a saved-and-reloaded setup silently reset it to all-defaults.
#
# Deliberately hand-rolled rather than a fully-generic recursive
# dataclass walker: every nested field here is one of (dataclass,
# list-of-dataclass, or a JSON-primitive/Optional-primitive), so an
# explicit per-field mapping is both simpler to read and easier to
# keep forward/backward-compatible (unrecognized keys are silently
# ignored via the `valid` filter in _mk(), matching project_io's own
# .get()-based tolerance for old/new setup files) than a generic
# walker would be to make equally tolerant.

def config_to_dict(cfg: "CrossSectionConfig") -> dict:
    """JSON-serializable dict for a CrossSectionConfig. Round-trips
    via config_from_dict(). Tuple fields (e.g. LayoutConfig.figsize,
    ExtraSectionLineConfig.vertices' (lon,lat,depth) tuples) come back
    out of asdict() as tuples; json.dump() below turns them into
    lists, which config_from_dict() converts back to tuples on load."""
    return asdict(cfg)


def _mk(cls, d):
    """Construct a dataclass `cls` from dict `d`, keeping only keys
    that are actually fields of `cls` (forward/backward-compatible:
    an older/newer setup file's extra or missing keys don't raise --
    missing keys just fall back to that dataclass's own defaults)."""
    if not d:
        return cls()
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in valid})


def config_from_dict(data: Optional[dict]) -> "CrossSectionConfig":
    """Reconstruct a CrossSectionConfig from config_to_dict() output
    (or from a hand-built/older dict missing some keys -- see _mk()).
    `data` of None/{} returns an all-defaults CrossSectionConfig()."""
    if not data:
        return CrossSectionConfig()

    cfg = CrossSectionConfig()

    cfg.layout = _mk(LayoutConfig, data.get("layout"))
    if cfg.layout.figsize is not None:
        cfg.layout.figsize = tuple(cfg.layout.figsize)

    cfg.topo_panels = [_mk(TopoPanelConfig, p) for p in (data.get("topo_panels") or [])]
    cfg.annotations = [_mk(AnnotationOverlayConfig, a) for a in (data.get("annotations") or [])]

    cfg.mesh = _mk(CFFMeshConfig, data.get("mesh"))
    cfg.eq = _mk(EQOverlayConfig, data.get("eq"))
    cfg.fault = _mk(FaultOverlayConfig, data.get("fault"))

    extra_lines = []
    for el in (data.get("extra_lines") or []):
        line = _mk(ExtraSectionLineConfig, el)
        line.vertices = [tuple(v) for v in (line.vertices or [])]
        extra_lines.append(line)
    cfg.extra_lines = extra_lines

    cfg.contours = _mk(CFFContourConfig, data.get("contours"))
    cfg.focal_mechanisms = _mk(FocalMechanismOverlayConfig, data.get("focal_mechanisms"))
    cfg.legend = _mk(LegendConfig, data.get("legend"))

    if "cff_source" in data:
        cfg.cff_source = data["cff_source"]

    return cfg
