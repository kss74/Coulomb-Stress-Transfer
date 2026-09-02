# Coulomb Stress Transfer

A QGIS 3.x plugin for Coulomb Failure Function (ΔCFF) stress transfer analysis,
rate-and-state seismicity forecasting, and related static-stress-triggering
workflows — built on the Okada (1985/1992) analytical dislocation model.

It is designed to reproduce and extend the core functionality of the
widely-used **Coulomb 3.4 / 4.0** desktop application (Toda, Stein, Sevilgen &
Lin) directly inside QGIS, so results can be combined with other GIS layers
without switching tools.

> **Status:** exploratory tool, not actively developed. Needs more validation. Read [Validation](#validation)
> and [Known limitations](#known-limitations) before relying on it for
> publication-quality results, and always sanity-check output against an
> independent reference for your specific fault geometry.

> **A note on how this was built:** this plugin is "vibecoded" — the
> vast majority of the code, verification scripts, and documentation
> (including this README) were written by [Claude Sonnet
> 5](https://www.anthropic.com/claude) (Anthropic) in an extended,
> session-by-session collaboration with the maintainer, who directed the
> work, reviewed the physics and design decisions, and ran the actual
> QGIS integration testing. Every physics module was checked against an
> independent reference (Coulomb 3.4.2 output, Dieterich 1994's own
> equations, symbolic re-derivation, etc.) before being trusted — see
> [Validation](#validation) — but the code was not hand-written line by
> line, and the usual caveats about AI-assisted code apply: read it
> before you rely on it, and please open an issue if something looks
> off. If you want to add more features or debug the plugin, you can
> ask Claude for help. Just upload the plugin files to a Claude Project
> or provide the GitHub repository link.

---

## Features

- **Coulomb stress transfer (ΔCFF)** from one or more rectangular source
  faults onto a receiver plane, using Okada (1985) surface strain (exact at
  z = 0) and optional Okada (1992) DC3D depth-dependent stress
- **Receiver fault modes:** a single shared receiver plane, individually
  oriented per-cell receiver faults, or receiver orientation resolved from
  imported focal mechanisms
- **Fault subdivision** into equal-area sub-patches for extended ruptures;
  multi-segment fault support
- **Distributed slip inversion** (uniform-rake and free-rake) from geodetic
  surface displacement observations
- **Rate-and-state seismicity forecasting** (Dieterich, 1994): forecast
  construction, calibration against real earthquake catalogs (background
  rate, ta/asig fitting with bounded optimization and parameter
  uncertainty), CSEP-style forecast scoring, near-field singularity
  exclusion
- **Aftershock Monte Carlo null testing** against a ΔCFF field, with
  enrichment-ratio significance reporting
- **Regional stress inversion** from a focal mechanism catalog via
  [ILSI](https://github.com/ebeauce/ILSI) (Beaucé et al., 2022), feeding the
  optimal-fault-orientation calculation
- **Focal mechanism import and analysis**, beachball rendering (via ObsPy),
  focal-mechanism-as-source-fault support
- **Earthquake catalog import** (CSV / GeoPackage), with catalog QA/QC
  timeline view
- **Cross-section tool**: multi-fault projection, mesh/contour ΔCFF display,
  topographic profile overlay, DEM raster sampling
- **Empirical magnitude–geometry scaling relations**: Wells & Coppersmith
  (1994), Leonard (2010, 2014), Thingbaijam et al. (2017), Strasser et al.
  (2010)
- **QGIS-native import** of fault traces from digitized polylines
- **Raster / vector export** of all computed fields (GeoTIFF, CSV/XYZ,
  QGIS fault and receiver-fault vector layers), text reports for
  rate-and-state and slip-inversion results
- **Project save/load**, including full dialog-state persistence across
  Source Faults, Receiver Fault, Grid Output, Elastic Params, Cross-Section,
  Rate-and-State, and Aftershock MC dialogs

## Installation

### Requirements

- QGIS ≥ 3.16 (QGIS 3.40 LTR or newer recommended)
- Python 3.9+ (whichever ships with your QGIS install)

### From a release zip

1. Download the latest release zip from this repository's
   [Releases](../../releases) page.
2. In QGIS: **Plugins → Manage and Install Plugins → Install from ZIP**,
   select the downloaded file, and click **Install Plugin**.
3. Enable **Coulomb Stress Transfer** in the plugin list if it isn't
   already active.

### From source

```bash
git clone https://github.com/kss74/Coulomb-Stress-Transfer.git
```

Copy (or symlink) the cloned folder into your QGIS profile's `python/plugins`
directory, then enable the plugin from **Plugins → Manage and Install
Plugins**. Typical profile plugin paths:

- **Windows:** `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
- **macOS:** `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
- **Linux:** `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`

### Python dependencies

Most of the plugin runs on packages already bundled with QGIS
(`numpy`, `scipy`, `matplotlib`, `PyQt`, `osgeo`/GDAL). A few features
require additional packages installed **into QGIS's own Python
environment** (use the OSGeo4W Shell on Windows, or QGIS's bundled
`pip`/`python3` on macOS/Linux):

| Package | Required for | Install |
|---|---|---|
| `obspy` | Focal mechanism beachball rendering | `pip install obspy` |
| `ILSI` | Regional stress inversion from focal mechanisms | see [github.com/ebeauce/ILSI](https://github.com/ebeauce/ILSI) |
| `mplstereonet` | Stereonet/instability plots for stress inversion | `pip install mplstereonet` |
| `okada-wrapper` | Exact depth-dependent (Okada 1992 DC3D) stress and displacement | `pip install okada-wrapper` (requires a Fortran compiler, see below) |

All four are **optional**: the plugin runs without them, with the
corresponding feature disabled or falling back to the validated z = 0
surface formula. The plugin's **Check / Install Dependencies…** menu item
reports current status.

`okada-wrapper` compiles a Fortran extension and needs `gfortran` on the
system first:

- **Windows:** install [MinGW-w64](https://www.mingw-w64.org/) and add
  `gfortran.exe` to your PATH
- **macOS:** `brew install gcc`
- **Linux:** `sudo apt install gfortran` (or `sudo yum install gcc-gfortran`)

Because QGIS's bundled Python often lacks the headers needed to build
compiled extensions in-place, the plugin can instead be pointed at any
external Python interpreter (e.g. a conda/venv environment) that already
has `okada-wrapper` installed; it is invoked via subprocess for
depth-dependent calculations. Configure this under **Check / Install
Dependencies…**.

---

## Usage

1. **Source Faults** — add one or more rectangular source faults (lon, lat,
   centroid depth, length, width, strike, dip, rake, slip), or import them
   from a digitized polyline / focal mechanism catalog.
2. **Receiver Fault** — specify a shared receiver orientation, or use
   per-cell / focal-mechanism-derived receivers.
3. **Grid Output** — define the map extent, resolution, and receiver depth.
4. **Elastic Params** — shear modulus, Poisson's ratio, friction coefficient.
5. Compute ΔCFF, surface deformation, a cross-section, a slip inversion, a
   rate-and-state forecast, or an aftershock Monte Carlo test from the
   relevant tab.
6. Export results as a QGIS raster layer, CSV/XYZ, fault/receiver vector
   layers, or a text report.

---

## Validation

The Okada (1985) surface engine (`core/okada_engine.py`) has been checked
against **Coulomb 3.4.2** reference output for a single-fault benchmark
(reverse fault, dip = 45°, 10 km centroid depth, 2.34 m slip):

| Metric | Value |
|---|---|
| ΔCFF spatial correlation (z = 0) | 0.983 |
| Sign agreement | 99.6% |
| Median magnitude ratio | 1.01 |
| Surface displacement vs. Coulomb's own output | exact to 4 decimal places |

The quadrant sign pattern (which lobes are positive/negative) is the same
at all receiver depths; only exact magnitudes at depth differ, which is why
depth-dependent runs use the full Okada (1992) DC3D solution rather than
extrapolating the surface formula.

### Known limitations

- Depth-dependent (DC3D) output has not been validated end-to-end against a
  real multi-depth Coulomb 3.4.2 reference case — only the plumbing has
  been exercised.
- Near-field cells close to a source fault's edges are subject to the
  well-known Okada/DC3D near-source singularity; the plugin flags and can
  exclude these cells, but values there should always be treated with
  caution regardless.
- Regional stress inversion (ILSI) returns stress **orientation and shape
  ratio only** — it cannot recover absolute stress magnitude from focal
  mechanisms alone (a physical limitation of the method, not an
  implementation gap). You must supply a differential-stress magnitude to
  use the result in a CFF calculation.
- This is an exploratory tool; treat any single run as
  provisional until cross-checked, especially for geometries far from the
  validated benchmark above.

## References

**Core dislocation theory**
- Okada, Y. (1985). Surface deformation due to shear and tensile faults in
  a half-space. *Bulletin of the Seismological Society of America*, 75(4),
  1135–1154.
- Okada, Y. (1992). Internal deformation due to shear and tensile faults in
  a half-space. *Bulletin of the Seismological Society of America*, 82(2),
  1018–1040.

**Coulomb stress transfer**
- King, G. C. P., Stein, R. S., & Lin, J. (1994). Static stress changes and
  the triggering of earthquakes. *Bulletin of the Seismological Society of
  America*, 84(3), 935–953.
- Toda, S., Stein, R. S., Sevilgen, V., & Lin, J. (2011). *Coulomb 3.3
  graphic-rich deformation and stress-change software for earthquake,
  tectonic, and volcano research and teaching — user guide.* U.S.
  Geological Survey Open-File Report 2011–1060.
  [pubs.usgs.gov/of/2011/1060](https://pubs.usgs.gov/of/2011/1060/)
- Wang, J. et al. (2021). AutoCoulomb — used as a secondary
  cross-validation reference for optimal fault plane / DC3D behavior.
  [github.com/jjwangw/CoulombAnalysis](https://github.com/jjwangw/CoulombAnalysis)

**Rate-and-state seismicity**
- Dieterich, J. (1994). A constitutive law for rate of earthquake
  production and its application to earthquake clustering. *Journal of
  Geophysical Research*, 99(B2), 2601–2618.
  [doi.org/10.1029/93JB02581](https://doi.org/10.1029/93JB02581)
- Cattania, C. `d94-mtmod` — MATLAB tutorial implementation of Dieterich
  (1994) seismicity-rate forecasting (`d94.m`, `coulomb2forecast.m`),
  developed for the MTMOD summer school. This plugin's
  `core/rate_state_seismicity.py::d94()` was independently re-derived
  symbolically from Dieterich (1994) eqs. 12/13 rather than ported from
  `d94.m`; `d94-mtmod` is referenced only as a cross-check on the
  functional form, not as a source of copied code. See
  [Third-party code, data, and dependencies](#third-party-code-data-and-dependencies)
  for the full provenance note.
  [github.com/camcat/d94-mtmod](https://github.com/camcat/d94-mtmod)

**Regional stress inversion**
- Beaucé, E., van der Hilst, R. D., & Campillo, M. (2022). An iterative
  linear method with variable shear stress magnitudes for estimating the
  stress tensor from earthquake focal mechanism data: method and examples.
  *Bulletin of the Seismological Society of America*.
  [doi.org/10.1785/0120210319](https://doi.org/10.1785/0120210319) —
  [github.com/ebeauce/ILSI](https://github.com/ebeauce/ILSI) (GPL-3.0)
- Vavrycuk, V. (2013, 2014). Iterative joint inversion for stress and fault
  orientations from focal mechanisms.
- Lund, B., & Slunga, R. (1999). Stress tensor inversion using detailed
  microearthquake information and stability constraints: application to
  Ölfus in southwest Iceland. *Journal of Geophysical Research*, 104(B7).

**Magnitude–geometry scaling relations**
- Wells, D. L., & Coppersmith, K. J. (1994). New empirical relationships
  among magnitude, rupture length, rupture width, rupture area, and
  surface displacement. *Bulletin of the Seismological Society of
  America*, 84(4), 974–1002.
- Leonard, M. (2010). Earthquake fault scaling: self-consistent relating
  of rupture length, width, average displacement, and moment release.
  *Bulletin of the Seismological Society of America*, 100(5A), 1971–1988.
- Leonard, M. (2014). Self-consistent earthquake fault-scaling relations:
  update and extension to stable continental strike-slip faults.
  *Bulletin of the Seismological Society of America*, 104(6), 2953–2965.
- Thingbaijam, K. K. S., Martin Mai, P., & Goda, K. (2017). New empirical
  earthquake source-scaling laws. *Bulletin of the Seismological Society
  of America*, 107(5), 2225–2246.
- Strasser, F. O., Arango, M. C., & Bommer, J. J. (2010). Scaling of the
  source dimensions of interface and intraslab subduction-zone earthquakes
  with moment magnitude. *Seismological Research Letters*, 81(6), 941–950.

**Focal mechanisms**
- Aki, K., & Richards, P. G. (1980/2002). *Quantitative Seismology.*
  University Science Books.

---

## Third-party code, data, and dependencies

This project builds on, wraps, or was cross-validated against the
following third-party software. If you redistribute this repository, please
review each item's own license/terms independently — this list is provided
for attribution and does not substitute for that review.

| Component | Role in this plugin | Source | License (as distributed by upstream) |
|---|---|---|---|
| `okada_wrapper` | Compiled Fortran DC3D subroutine (Okada 1992) for exact depth-dependent stress/displacement; installed as a pip dependency, not vendored in this repo | [github.com/cutde-org/okada_wrapper](https://github.com/cutde-org/okada_wrapper) | MIT |
| ObsPy | Beachball / focal-mechanism plotting | [github.com/obspy/obspy](https://github.com/obspy/obspy) | LGPL-3.0 |
| ILSI | Regional stress tensor inversion from focal mechanisms | [github.com/ebeauce/ILSI](https://github.com/ebeauce/ILSI) | GPL-3.0 |
| `mplstereonet` | Stereonet plotting for stress inversion results | PyPI / GitHub | MIT |
| Beauducel, F., `okada85.m` (IPGP `deformation-lib`) | Cross-reference implementation used while validating this plugin's own Okada (1985) strain kernels; not vendored here | IPGP | see upstream repository |
| Cattania, C., `d94-mtmod` (`d94.m`, `coulomb2forecast.m`) | Cross-check reference for Dieterich (1994) seismicity-rate forecasting. `core/rate_state_seismicity.py::d94()` was independently re-derived (symbolically, via sympy) from Dieterich (1994) eqs. 12/13 | [github.com/camcat/d94-mtmod](https://github.com/camcat/d94-mtmod) |
| Coulomb ver 3.4, ver. 4.x (Yoshizawa, Toda, Stein, Sevilgen, Lin) | Used as the ground-truth benchmark in [Validation](#validation) | [temblor.net/coulomb](https://temblor.net/coulomb/), [github.com/YoshKae/Coulomb_ver4](https://github.com/YoshKae/Coulomb_ver4) | |

This plugin's own source code — the files under `core/` and `ui/` — is
original work, including `d94()`, which was independently re-derived
from Dieterich (1994) rather than ported from any third-party script.

---

## License
This project is licensed under the **GNU General Public License v3.0 or
later (GPL-3.0-or-later)**.

This choice is not arbitrary: [ILSI](https://github.com/ebeauce/ILSI)
(GPL-3.0), used for regional stress inversion, is imported directly
in-process (`import ILSI` in `stress_inversion.py`) rather than isolated
behind a subprocess boundary the way `okada_wrapper` is. Under the GPL,
a program that links or imports GPL-3.0 code like this forms a combined
work when distributed, which must itself be released under GPL-3.0 (or
a later version, per the "-or-later" grant) — GPL-2.0-only would not be
compatible. See
[Third-party code, data, and dependencies](#third-party-code-data-and-dependencies)
for the full breakdown of every dependency's license and how each is
used.

See the [`LICENSE`](LICENSE) file in this repository for the full
license text.

```
Copyright (C) 2026  KSarmiento

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```

