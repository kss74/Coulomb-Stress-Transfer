# -*- coding: utf-8 -*-
"""
Build a configurable, GMT-style multi-panel cross-section Figure.

This module is pure matplotlib + numpy -- no PyQt/QGIS imports -- so it
is fully testable in the sandbox. All the "where is this data coming
from" concerns (EQ catalog import, focal mechanism import, raster
sampling, QGIS layer reads) are the CALLER's job (ui.cross_section_window
/ ui.main_dialog); this module only knows how to lay out and draw
already-extracted numpy arrays according to a CrossSectionConfig
(core.cross_section_config).

Why matplotlib and not PyGMT: PyGMT was evaluated for this feature
(2026-08-18) and rejected -- it's a thin wrapper around the GMT C
library (libgmt.so), which has no pip wheel and isn't installable in
this sandbox (no apt/conda access) or reliably in an end user's QGIS
Python environment either, unlike okada_wrapper/obspy/rasterio which
are all self-contained pip packages. The GMT-style LAYOUT (stacked
topo profile(s) above a depth section, configurable symbology per
overlay, per-panel vertical exaggeration) is reproduced here directly
in matplotlib instead, reusing the same fixed-axes-per-Figure approach
already used by plot_widget.PlotWidget elsewhere in this project.

Layout (top to bottom): one row per config.topo_panels entry (in
order), then the main ΔCFF depth section. All rows share the x-axis
(distance along profile, km) via `sharex`.

Panel-width alignment (2026-08-19 follow-up): every colorbar this
module adds is now requested with `ax=[*axes_topo, ax_main]` instead of
`ax=ax_main` alone. matplotlib's colorbar carves its reserved width out
of EVERY axes passed to `ax=`, not just the one the mappable was drawn
on -- passing the full panel stack keeps every row's plotting area the
same width regardless of how many colorbars end up living next to the
main panel. Previously colorbars only shrank ax_main, so the topo
panel(s) above it silently ended up wider than the main panel despite
sharing an x-axis -- the reported "profile panel and cross-section
panel not aligned in width" bug, and, since `layout="constrained"` was
then compensating for that mismatch by inserting extra vertical space,
also the reported "large gap between the cross-section and the
topographic profile" bug. Both were the same root cause.
"""

import numpy as np
from matplotlib.figure import Figure


def _eq_marker_sizes(magnitude, cfg):
    if magnitude is None or cfg.size_by != "magnitude":
        return np.full(0 if magnitude is None else len(magnitude), cfg.fixed_size_pt2)
    magnitude = np.asarray(magnitude, dtype=float)
    finite = magnitude[np.isfinite(magnitude)]
    if finite.size == 0:
        return np.full(len(magnitude), cfg.fixed_size_pt2)
    mmin, mmax = float(finite.min()), float(finite.max())
    if mmax - mmin < 1e-9:
        return np.full(len(magnitude), (cfg.mag_size_min_pt2 + cfg.mag_size_max_pt2) / 2.0)
    frac = np.clip((magnitude - mmin) / (mmax - mmin), 0.0, 1.0)
    return cfg.mag_size_min_pt2 + frac * (cfg.mag_size_max_pt2 - cfg.mag_size_min_pt2)


def _fm_diameters(magnitude, fm_cfg, n_fm):
    """Same min/max-linear-scaling convention as _eq_marker_sizes(), but
    in diameter (km, this panel's own data units) rather than marker
    area (pt^2) -- beachballs are drawn in data units, not points (see
    core.beachball.draw_beachball()'s own docstring on that choice)."""
    if magnitude is None or fm_cfg.size_by != "magnitude":
        return np.full(n_fm, fm_cfg.diameter_km)
    magnitude = np.asarray(magnitude, dtype=float)
    finite = magnitude[np.isfinite(magnitude)]
    if finite.size == 0:
        return np.full(n_fm, fm_cfg.diameter_km)
    mmin, mmax = float(finite.min()), float(finite.max())
    if mmax - mmin < 1e-9:
        return np.full(n_fm, (fm_cfg.mag_diameter_min_km + fm_cfg.mag_diameter_max_km) / 2.0)
    frac = np.clip((magnitude - mmin) / (mmax - mmin), 0.0, 1.0)
    return fm_cfg.mag_diameter_min_km + frac * (fm_cfg.mag_diameter_max_km - fm_cfg.mag_diameter_min_km)


def _mesh_vmax(cff_2d, mesh_cfg):
    """Resolve the mesh's vmin/vmax -- explicit config values win;
    otherwise fall back to the symmetric-percentile auto-scale."""
    if mesh_cfg.vmin_mpa is not None and mesh_cfg.vmax_mpa is not None:
        return float(mesh_cfg.vmin_mpa), float(mesh_cfg.vmax_mpa)
    vmax = (np.nanpercentile(np.abs(cff_2d), mesh_cfg.color_scale_percentile)
            if np.isfinite(cff_2d).any() else 1e-6)
    vmax = max(vmax, 1e-6)
    return -vmax, vmax


def _maybe_interpolate_mesh(dist_km, depth_km, cff_2d, mesh_cfg):
    """
    Display-only smoothing of the already-computed grid (see
    CFFMeshConfig's docstring -- this never touches the actual DC3D
    sampling resolution). Uses a bivariate spline, degree min(3, n-1)
    per axis so it degrades gracefully instead of raising on a thin
    (e.g. 2-row) grid. Falls back to returning the input unchanged if
    scipy isn't available or the grid is too small/degenerate to
    interpolate (e.g. all-NaN), since a display nicety should never be
    able to crash the plot.
    """
    if not mesh_cfg.interpolate or mesh_cfg.interpolation_factor <= 1:
        return dist_km, depth_km, cff_2d
    if len(dist_km) < 2 or len(depth_km) < 2 or not np.isfinite(cff_2d).any():
        return dist_km, depth_km, cff_2d
    try:
        from scipy.interpolate import RectBivariateSpline
    except ImportError:
        return dist_km, depth_km, cff_2d

    filled = np.where(np.isfinite(cff_2d), cff_2d, 0.0)
    kx = min(3, len(depth_km) - 1)
    ky = min(3, len(dist_km) - 1)
    if kx < 1 or ky < 1:
        return dist_km, depth_km, cff_2d
    try:
        spline = RectBivariateSpline(depth_km, dist_km, filled, kx=kx, ky=ky)
        dist_fine = np.linspace(dist_km.min(), dist_km.max(),
                                 max(2, len(dist_km) * mesh_cfg.interpolation_factor))
        depth_fine = np.linspace(depth_km.min(), depth_km.max(),
                                  max(2, len(depth_km) * mesh_cfg.interpolation_factor))
        cff_fine = spline(depth_fine, dist_fine)
        # NaN-ness doesn't survive a spline fit (we filled with 0 above)
        # -- reapply it via nearest-neighbor lookup on the ORIGINAL mask
        # so surface-only/no-DC3D rows/cols still show as gaps, not
        # spurious zero-CFF, after upsampling.
        nan_mask = ~np.isfinite(cff_2d)
        if nan_mask.any():
            di = np.clip(np.searchsorted(depth_km, depth_fine), 0, len(depth_km) - 1)
            dj = np.clip(np.searchsorted(dist_km, dist_fine), 0, len(dist_km) - 1)
            fine_nan = nan_mask[np.ix_(di, dj)]
            cff_fine = np.where(fine_nan, np.nan, cff_fine)
        return dist_fine, depth_fine, cff_fine
    except Exception:
        return dist_km, depth_km, cff_2d


def build_cross_section_figure(config, dist_km, depth_km, cff_2d,
                                eq_data=None, fault_traces=None,
                                topo_data=None, annotation_data=None,
                                focal_mechanism_data=None, profile_direction=None,
                                extra_line_data=None, segment_info=None,
                                title=None):
    """
    config         : core.cross_section_config.CrossSectionConfig
    dist_km        : 1D array, main-panel x axis (km along profile)
    depth_km       : 1D array, main-panel y axis (km, positive down)
    cff_2d         : 2D array (len(depth_km), len(dist_km)), delta-CFF MPa
    eq_data        : optional dict {"dist_km":arr,"depth_km":arr,"magnitude":arr or None}
                      -- already filtered to the search-width swath by the caller.
                      Depth is NOT pre-clipped to depth_km's own range by
                      this function's caller convention -- events deeper
                      than the computed section are still passed through,
                      but this function locks the main panel's y-limits
                      to depth_km's own range (see near the end) so they
                      don't silently stretch the axis past the section
                      that was actually computed.
    fault_traces   : optional list from core.cross_section_faults
                      .project_fault_traces_onto_section()
    topo_data      : optional list of (dist_km_arr, elevation_km_arr), one
                      per entry in config.topo_panels, same order
    annotation_data: optional list of (dist_km_arr, labels_list) OR
                      (dist_km_arr, labels_list, z_km_arr_or_None) --
                      the 2-tuple form is still accepted for backward
                      compatibility (pins markers near the panel top,
                      the old behavior); the 3-tuple form's z_km_arr, if
                      not None, plots each marker at its own vertical
                      position on the target topo panel instead. One
                      tuple per entry in config.annotations, same order.
    focal_mechanism_data : optional dict, already filtered to the
                      search-width swath by the caller (same convention
                      as eq_data), with keys:
                        "dist_km", "depth_km" : arrays
                        "strike1", "dip1", "rake1" : arrays (plane 1,
                              required)
                        "strike2", "dip2", "rake2" : arrays or None
                              (plane 2, optional -- use NaN entries for
                              events without a plane 2, not a shorter
                              array)
                        "cff_mpa"      : array or None (used when
                              config.focal_mechanisms.color_by=="cff")
                        "magnitude"    : array or None (used when
                              size_by=="magnitude" and/or label_source==
                              "magnitude")
                        "selected"     : list of "plane1"/"plane2" or
                              None (used when
                              config.focal_mechanisms.plane=="selected")
                        "label"        : list of str or None (used when
                              label_source=="custom")
                      Orientations in this dict are the RAW (strike,
                      dip, rake) in the standard (East, North, Down)
                      convention -- this function does the side-view
                      rotation itself (core.focal_side_view), given
                      `profile_direction`, so callers don't repeat that
                      physics at every call site.
    profile_direction : (ux, uy) -- the profile's along-profile unit
                      direction, (East, North) components (e.g.
                      core.geo_profile.profile_direction()). Required
                      when focal_mechanism_data is given.
    extra_line_data : optional list of (dist_km_arr, depth_km_arr)
                      tuples, one per entry in config.extra_lines, same
                      order (2026-08-21; request item 7 -- imported
                      slab interfaces, other faults' known geometry,
                      etc.). Already projected/filtered by the caller
                      via core.geo_profile.project_points_to_polyline(),
                      same convention as fault_traces/eq_data.
    segment_info    : optional dict from
                      core.geo_profile.polyline_segment_info() -- when
                      given AND it describes more than one segment,
                      draws a vertical marker + strike label at each
                      internal profile vertex (2026-08-21; request item
                      5, "segment/strike change must be indicated in
                      the plot"). Omit (or a single-segment profile)
                      draws nothing extra.
    title          : overrides config.layout.title if given

    Returns a matplotlib Figure (caller embeds it in a FigureCanvas, or
    calls fig.savefig() directly).
    """
    topo_panels = config.topo_panels
    n_topo = len(topo_panels)
    height_ratios = [p.height_ratio for p in topo_panels] + [config.layout.main_height_ratio]

    fig = Figure(figsize=config.layout.figsize, dpi=config.layout.dpi,
                layout="constrained")
    gs = fig.add_gridspec(n_topo + 1, 1, height_ratios=height_ratios,
                          hspace=config.layout.hspace)

    axes_topo = []
    ax_main = None
    for i in range(n_topo):
        ax = fig.add_subplot(gs[i], sharex=axes_topo[0] if axes_topo else None)
        axes_topo.append(ax)
    ax_main = fig.add_subplot(gs[n_topo], sharex=axes_topo[0] if axes_topo else None)
    all_row_axes = [*axes_topo, ax_main]  # passed to every colorbar's ax=
                                           # so all rows shrink together
                                           # (see module docstring)

    legend_handles = []
    legend_labels = []

    # ── Topo panel(s) ───────────────────────────────────────────────
    for i, panel_cfg in enumerate(topo_panels):
        ax = axes_topo[i]
        if topo_data and i < len(topo_data) and topo_data[i] is not None:
            d, z = topo_data[i]
            ax.plot(d, z, color=panel_cfg.color, linewidth=panel_cfg.linewidth)
            if panel_cfg.fill:
                ax.fill_between(d, z, np.nanmin(z), color=panel_cfg.color,
                                alpha=panel_cfg.fill_alpha, linewidth=0)
        ax.set_ylabel(panel_cfg.label, fontsize=8)
        ax.tick_params(labelsize=7)
        if panel_cfg.vertical_exaggeration is not None:
            ax.set_aspect(panel_cfg.vertical_exaggeration)
        ax.grid(True, alpha=0.25)

        if annotation_data:
            for ann_cfg, entry in zip(config.annotations, annotation_data):
                if ann_cfg.topo_panel_index != i or not ann_cfg.enabled:
                    continue
                if entry is None:
                    continue
                if len(entry) == 3:
                    a_dist, a_labels, a_z = entry
                else:
                    a_dist, a_labels = entry
                    a_z = None
                if len(a_dist) == 0:
                    continue
                if a_z is not None:
                    y_at = np.asarray(a_z, dtype=float)
                    clip_on = True
                else:
                    ylim = ax.get_ylim()
                    y_at = np.full(len(a_dist), ylim[1] - 0.08 * (ylim[1] - ylim[0]))
                    clip_on = False
                ax.scatter(a_dist, y_at, marker=ann_cfg.marker,
                          color=ann_cfg.color, s=ann_cfg.size_pt ** 2,
                          zorder=ann_cfg.zorder, clip_on=clip_on)
                for x, y, lbl in zip(a_dist, y_at, a_labels):
                    if lbl:
                        ax.annotate(lbl, (x, y), fontsize=ann_cfg.label_fontsize,
                                   ha="center", va="bottom", color=ann_cfg.color)

    # ── Main ΔCFF depth section ────────────────────────────────────
    mesh_cfg = config.mesh if hasattr(config, "mesh") else None
    if mesh_cfg is not None:
        vmin, vmax = _mesh_vmax(cff_2d, mesh_cfg)
    else:
        vmax = np.nanpercentile(np.abs(cff_2d), 98) if np.isfinite(cff_2d).any() else 1e-6
        vmax = max(vmax, 1e-6)
        vmin = -vmax

    # 2026-08-21 fix ("setting Min/Max MPa for the CFF Mesh made the
    # contours stop displaying"): contour default levels (used whenever
    # config.contours.spacing_mpa is unset) were computed as
    # np.linspace(vmin, vmax, n_levels) using THIS mesh's vmin/vmax --
    # including any manual override the user set for the MESH COLOR
    # SCALE. If that manual range doesn't span the actual cff_2d data
    # (e.g. the user set +-50 MPa to tame a color-scale outlier, but the
    # real ΔCFF values are all ~+-0.01 MPa), every auto contour level
    # landed far outside the data's actual range, so ax_main.contour()
    # found no crossings anywhere -- contours silently vanished with no
    # error. The mesh's manual vmin/vmax should only ever control mesh
    # COLORING; contour level defaults must always be derived from the
    # DATA's own extent, independent of that override.
    if np.isfinite(cff_2d).any():
        data_vmax = np.nanpercentile(np.abs(cff_2d),
                                     mesh_cfg.color_scale_percentile if mesh_cfg else 98.0)
        data_vmax = max(float(data_vmax), 1e-6)
    else:
        data_vmax = 1e-6
    contour_vmin, contour_vmax = -data_vmax, data_vmax

    mesh_enabled = mesh_cfg.enabled if mesh_cfg is not None else True
    if mesh_enabled and np.isfinite(cff_2d).any():
        dist_plot, depth_plot, cff_plot = (
            _maybe_interpolate_mesh(dist_km, depth_km, cff_2d, mesh_cfg)
            if mesh_cfg is not None else (dist_km, depth_km, cff_2d))
        im = ax_main.pcolormesh(dist_plot, depth_plot, cff_plot,
                                cmap=(mesh_cfg.cmap if mesh_cfg else "RdBu_r"),
                                vmin=vmin, vmax=vmax, shading="auto",
                                zorder=(mesh_cfg.zorder if mesh_cfg else 3))
        cbar = fig.colorbar(im, ax=all_row_axes, pad=0.02, fraction=0.05)
        cbar.set_label("ΔCFF (MPa)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    if config.contours.enabled and np.isfinite(cff_2d).any():
        c_cfg = config.contours
        levels = c_cfg.levels
        if not levels:
            if c_cfg.spacing_mpa:
                baseline = c_cfg.baseline_mpa if c_cfg.baseline_mpa is not None else 0.0
                spacing = c_cfg.spacing_mpa
                # Same fix as above: derive the level RANGE from the
                # actual data extent (contour_vmin/vmax), not the
                # mesh's possibly-manual color-scale override.
                k_lo = np.floor((contour_vmin - baseline) / spacing)
                k_hi = np.ceil((contour_vmax - baseline) / spacing)
                levels = baseline + np.arange(k_lo, k_hi + 1) * spacing
            else:
                levels = np.linspace(contour_vmin, contour_vmax, c_cfg.n_levels)
            levels = levels[np.abs(levels) > 1e-9]
        if len(levels) > 0:
            cs = ax_main.contour(dist_km, depth_km, cff_2d, levels=sorted(set(levels)),
                                 colors=c_cfg.color, linewidths=c_cfg.linewidth,
                                 alpha=c_cfg.alpha, zorder=c_cfg.zorder)
            if c_cfg.inline_labels:
                ax_main.clabel(cs, fmt=c_cfg.fmt, fontsize=c_cfg.label_fontsize)

    if config.fault.enabled and fault_traces:
        f_cfg = config.fault
        for trace in fault_traces:
            ax_main.plot(trace["dist_km"], trace["depth_km"], color=f_cfg.color,
                        linewidth=f_cfg.linewidth, linestyle=f_cfg.linestyle,
                        zorder=f_cfg.zorder)
            if f_cfg.label_sources:
                mx = float(np.mean(trace["dist_km"]))
                my = float(np.mean(trace["depth_km"]))
                ax_main.annotate(trace["label"], (mx, my), fontsize=f_cfg.label_fontsize,
                                color=f_cfg.color, ha="center", va="center",
                                fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                         ec=f_cfg.color, alpha=0.75, linewidth=0.5))
        if fault_traces:
            legend_handles.append(
                ax_main.plot([], [], color=f_cfg.color, linewidth=f_cfg.linewidth)[0])
            legend_labels.append("Source fault trace")

    # ── Extra imported depth-section elements (2026-08-21, item 7) ──
    if config.extra_lines and extra_line_data:
        for line_cfg, line_pts in zip(config.extra_lines, extra_line_data):
            if line_pts is None:
                continue
            d_arr, z_arr = line_pts
            if len(d_arr) == 0:
                continue
            # Sort by along-profile distance so an out-of-order digitized
            # line (or one that folds back) still draws as a sensible
            # left-to-right polyline rather than a scribble.
            order = np.argsort(d_arr)
            ax_main.plot(np.asarray(d_arr)[order], np.asarray(z_arr)[order],
                        color=line_cfg.color, linewidth=line_cfg.linewidth,
                        linestyle=line_cfg.linestyle, zorder=line_cfg.zorder)
            if line_cfg.show_label and line_cfg.label:
                mi = order[len(order) // 2]
                ax_main.annotate(line_cfg.label, (float(d_arr[mi]), float(z_arr[mi])),
                                 fontsize=line_cfg.label_fontsize, color=line_cfg.color,
                                 ha="center", va="bottom", fontweight="bold",
                                 bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                          ec=line_cfg.color, alpha=0.75, linewidth=0.5))
            legend_handles.append(
                ax_main.plot([], [], color=line_cfg.color, linewidth=line_cfg.linewidth,
                            linestyle=line_cfg.linestyle)[0])
            legend_labels.append(line_cfg.label or "Imported line")

    # ── Multi-segment profile boundary markers (2026-08-21, item 5) ──
    if segment_info and len(segment_info.get("segment_azimuth_deg", [])) > 1:
        internal_vertices = segment_info["cumulative_dist_km"][1:-1]
        for k, x_boundary in enumerate(internal_vertices):
            ax_main.axvline(x_boundary, color="0.4", linewidth=0.8,
                            linestyle=":", zorder=20)
            az_before = segment_info["segment_azimuth_deg"][k]
            az_after = segment_info["segment_azimuth_deg"][k + 1]
            ax_main.annotate(f"bend\n{az_before:.0f}°→{az_after:.0f}°",
                             (x_boundary, depth_km[0]), fontsize=6.5, color="0.3",
                             ha="center", va="top", rotation=0, zorder=21,
                             annotation_clip=False,
                             bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                      ec="0.4", alpha=0.85, linewidth=0.4))
        for ax in axes_topo:
            for x_boundary in internal_vertices:
                ax.axvline(x_boundary, color="0.4", linewidth=0.8, linestyle=":", zorder=20)

    if config.eq.enabled and eq_data and len(eq_data.get("dist_km", [])) > 0:
        e_cfg = config.eq
        sizes = _eq_marker_sizes(eq_data.get("magnitude"), e_cfg)
        if len(sizes) == 0:
            sizes = np.full(len(eq_data["dist_km"]), e_cfg.fixed_size_pt2)
        if e_cfg.color_by == "depth":
            sc = ax_main.scatter(eq_data["dist_km"], eq_data["depth_km"], s=sizes,
                                c=eq_data["depth_km"], cmap=e_cfg.cmap, marker=e_cfg.marker,
                                alpha=e_cfg.alpha, edgecolors=e_cfg.edgecolor,
                                linewidths=e_cfg.edge_linewidth, zorder=e_cfg.zorder)
            eq_cbar = fig.colorbar(sc, ax=all_row_axes, pad=0.09, fraction=0.05)
            eq_cbar.set_label("Earthquake depth (km)", fontsize=8)
            eq_cbar.ax.tick_params(labelsize=7)
        else:
            ax_main.scatter(eq_data["dist_km"], eq_data["depth_km"], s=sizes,
                           color=e_cfg.single_color, marker=e_cfg.marker,
                           alpha=e_cfg.alpha, edgecolors=e_cfg.edgecolor,
                           linewidths=e_cfg.edge_linewidth, zorder=e_cfg.zorder)
        legend_handles.append(ax_main.scatter([], [], s=40, color="0.5",
                                              marker=e_cfg.marker,
                                              edgecolors=e_cfg.edgecolor))
        legend_labels.append("Earthquake catalog")

    if (config.focal_mechanisms.enabled and focal_mechanism_data
            and len(focal_mechanism_data.get("dist_km", [])) > 0):
        if profile_direction is None:
            raise ValueError(
                "profile_direction=(ux, uy) is required when focal_mechanism_data "
                "is given -- see core.geo_profile.profile_direction().")
        from .focal_side_view import apparent_side_view_event
        from .beachball import draw_beachball, cff_to_color
        from .focal_mechanism import classify_fault_type, FAULT_TYPE_COLORS

        fm_cfg = config.focal_mechanisms
        ux, uy = profile_direction
        fmd = focal_mechanism_data
        n_fm = len(fmd["dist_km"])
        strike2_arr = fmd.get("strike2")
        dip2_arr = fmd.get("dip2")
        rake2_arr = fmd.get("rake2")
        has_plane2 = strike2_arr is not None
        cff_arr = fmd.get("cff_mpa")
        mag_arr = fmd.get("magnitude")
        selected_arr = fmd.get("selected")
        labels = fmd.get("label")

        fm_norm = None
        type_legend_entries = []
        if fm_cfg.color_by == "cff" and cff_arr is not None and np.isfinite(cff_arr).any():
            if fm_cfg.vmin_mpa is not None and fm_cfg.vmax_mpa is not None:
                import matplotlib.cm as _cm0
                import matplotlib.colors as _mc0
                fm_norm = _mc0.Normalize(vmin=fm_cfg.vmin_mpa, vmax=fm_cfg.vmax_mpa)
                _cmap0 = _cm0.get_cmap(fm_cfg.cmap)
                facecolors = [_cmap0(fm_norm(v)) for v in np.asarray(cff_arr, dtype=float)]
            else:
                facecolors, fm_norm, _ = cff_to_color(cff_arr, fm_cfg.cmap)
        elif fm_cfg.color_by == "depth":
            depth_arr = np.asarray(fmd["depth_km"], dtype=float)
            import matplotlib.cm as _cm1
            import matplotlib.colors as _mc1
            dmin, dmax = float(np.nanmin(depth_arr)), float(np.nanmax(depth_arr))
            if dmax - dmin < 1e-9:
                dmax = dmin + 1e-9
            fm_norm = _mc1.Normalize(vmin=dmin, vmax=dmax)
            _cmap1 = _cm1.get_cmap(fm_cfg.depth_cmap)
            facecolors = [_cmap1(fm_norm(v)) for v in depth_arr]
        elif fm_cfg.color_by == "type":
            # 2026-08-21: per-overlay type_colors override, same
            # "None -> shared default, dict -> override" convention as
            # every other color knob in this config (cmap, depth_cmap,
            # single_color) -- missing keys in a partial override still
            # fall back to the FAULT_TYPE_COLORS default for that type.
            type_color_map = dict(FAULT_TYPE_COLORS)
            if fm_cfg.type_colors:
                type_color_map.update(fm_cfg.type_colors)
            types = [classify_fault_type(float(fmd["strike1"][i]), float(fmd["dip1"][i]),
                                          float(fmd["rake1"][i])) for i in range(n_fm)]
            facecolors = [type_color_map[t] for t in types]
            for t in ("normal", "reverse", "strike-slip", "oblique"):
                if t in types:
                    type_legend_entries.append(t)
        else:
            facecolors = [fm_cfg.single_color] * n_fm

        diameters = _fm_diameters(mag_arr, fm_cfg, n_fm)

        for i in range(n_fm):
            s2 = (float(strike2_arr[i]) if has_plane2 and np.isfinite(strike2_arr[i]) else None)
            d2 = (float(dip2_arr[i]) if has_plane2 and np.isfinite(dip2_arr[i]) else None)
            r2 = (float(rake2_arr[i]) if has_plane2 and np.isfinite(rake2_arr[i]) else None)

            s1a, d1a, r1a, s2a, d2a, r2a = apparent_side_view_event(
                float(fmd["strike1"][i]), float(fmd["dip1"][i]), float(fmd["rake1"][i]),
                ux, uy, strike2=s2, dip2=d2, rake2=r2)

            which = "plane1"
            if fm_cfg.plane == "plane2" and s2a is not None:
                which = "plane2"
            elif fm_cfg.plane == "selected" and selected_arr is not None:
                sel = selected_arr[i]
                if sel == "plane2" and s2a is not None:
                    which = "plane2"

            if which == "plane2":
                plot_s, plot_d, plot_r = s2a, d2a, r2a
            else:
                plot_s, plot_d, plot_r = s1a, d1a, r1a

            highlight = None
            if fm_cfg.highlight_selected_plane and s2a is not None:
                highlight = which

            diam = float(diameters[i])
            draw_beachball(
                ax_main, float(fmd["dist_km"][i]), float(fmd["depth_km"][i]),
                diam, plot_s, plot_d, plot_r,
                facecolor=facecolors[i], highlight_plane=highlight,
                strike2=s2a, dip2=d2a,
                bgcolor=fm_cfg.bgcolor, edgecolor=fm_cfg.edgecolor,
                zorder=fm_cfg.zorder)

            if fm_cfg.show_labels:
                if fm_cfg.label_source == "magnitude" and mag_arr is not None \
                        and np.isfinite(mag_arr[i]):
                    lbl = fm_cfg.label_fmt % float(mag_arr[i])
                else:
                    lbl = labels[i] if labels else None
                if lbl:
                    # 2026-08-21 fix ("label is behind the beachball"):
                    # the offset was `xytext=(0, diam*0.6)` with
                    # `textcoords="offset points"` -- but `diam` is in
                    # DATA units (km, same convention as
                    # draw_beachball()'s own diameter argument; see its
                    # docstring), while "offset points" is interpreted
                    # in PRINTER POINTS (1/72"). For a typical diam of
                    # 1-8 km that's an offset of ~1-5 points -- a couple
                    # of screen pixels -- so the label rendered almost
                    # exactly on top of the beachball it was meant to
                    # sit above. Fixed by placing the label directly in
                    # DATA coordinates instead, offset by a multiple of
                    # the symbol's own diameter (so the gap scales with
                    # symbol size, same as the old intent) rather than
                    # a fixed point offset that doesn't share units with
                    # the thing it's offset from. Note depth increases
                    # DOWNWARD and ax_main.invert_yaxis() is called
                    # later, so "above" the beachball == a SMALLER depth
                    # value here.
                    cx = float(fmd["dist_km"][i])
                    cy = float(fmd["depth_km"][i])
                    label_gap = (fm_cfg.label_offset_km if fm_cfg.label_offset_km is not None
                                else diam * 0.9)
                    lx, ly = cx, cy - (diam / 2.0 + label_gap)
                    if fm_cfg.label_leader_line:
                        # Optional thin leader connecting the label back
                        # to the beachball -- useful once labels start
                        # crowding each other and no longer sit right
                        # next to their own symbol.
                        ax_main.plot([cx, lx], [cy - diam / 2.0, ly],
                                    color="0.35", linewidth=0.5, alpha=0.8,
                                    zorder=fm_cfg.zorder + 1, solid_capstyle="round")
                    ax_main.annotate(lbl, (lx, ly), fontsize=fm_cfg.label_fontsize,
                                     ha="center", va="bottom", zorder=fm_cfg.zorder + 2)

        if fm_norm is not None:
            import matplotlib.cm as _cm
            _cmap_used = fm_cfg.depth_cmap if fm_cfg.color_by == "depth" else fm_cfg.cmap
            sm = _cm.ScalarMappable(norm=fm_norm, cmap=_cmap_used)
            sm.set_array([])
            fm_cbar = fig.colorbar(sm, ax=all_row_axes, pad=0.16, fraction=0.05)
            fm_cbar.set_label(
                "Focal mechanism depth (km)" if fm_cfg.color_by == "depth"
                else "Focal mechanism ΔCFF (MPa)", fontsize=8)
            fm_cbar.ax.tick_params(labelsize=7)
        if type_legend_entries:
            for t in type_legend_entries:
                legend_handles.append(ax_main.scatter([], [], s=60, facecolor=type_color_map[t],
                                                       edgecolor=fm_cfg.edgecolor, marker="o"))
                legend_labels.append(f"Focal mechanism ({t})")
        else:
            legend_handles.append(ax_main.scatter([], [], s=60, facecolor="0.6",
                                                  edgecolor=fm_cfg.edgecolor, marker="o"))
            legend_labels.append("Focal mechanism (side view)")

    ax_main.invert_yaxis()
    ax_main.set_aspect(config.layout.main_vertical_exaggeration)
    ax_main.set_xlabel("Distance along profile (km)", fontsize=8)
    ax_main.set_ylabel("Depth (km)", fontsize=8)
    ax_main.tick_params(labelsize=7)

    # Lock the main panel's vertical extent to the COMPUTED section
    # (depth_km), regardless of what got scattered on top of it -- fixes
    # "the EQ catalog's vertical range stretches past the CFF section's
    # own depth range" (matplotlib autoscales to include every artist by
    # default, so an earthquake or focal mechanism deeper than
    # max_depth_km would otherwise silently expand the axis).
    if len(depth_km) > 0:
        ax_main.set_ylim(float(np.nanmax(depth_km)), float(np.nanmin(depth_km)))

    fig_title = title if title is not None else config.layout.title
    if fig_title:
        fig.suptitle(fig_title, fontsize=11, fontweight="bold")
    elif axes_topo:
        axes_topo[0].set_title("Cross-Section", fontsize=10, fontweight="bold")
    else:
        ax_main.set_title("Coulomb Stress Cross-Section", fontsize=10, fontweight="bold")

    for ax in axes_topo:
        for label in ax.get_xticklabels():
            label.set_visible(False)

    if config.legend.enabled and legend_handles:
        if config.legend.loc == "outside_right":
            ax_main.legend(legend_handles, legend_labels, fontsize=config.legend.fontsize,
                          loc="upper left", bbox_to_anchor=(1.12, 1.0),
                          borderaxespad=0.0, frameon=True)
        else:
            ax_main.legend(legend_handles, legend_labels, fontsize=config.legend.fontsize,
                          loc=config.legend.loc, frameon=True)

    return fig
