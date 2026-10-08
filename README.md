# GSPWD — Generalized Spherical Plane-Wave Decomposition

Direction-of-arrival (DOA) and amplitude estimation of several plane waves from
spherical-harmonic-domain signals (e.g. ambisonics / spherical microphone arrays), using

1. **optimised zonal kernels** — a closed-form, equality-constrained quadratic design that keeps
   a unit peak, minimises energy outside a mainlobe of half-width θ₀ and nulls the back lobe, in
   place of the standard maximum-directivity (Dirichlet) kernel of plane-wave decomposition (SPWD); and
2. a **joint orthogonal matching pursuit** that picks one shared set of directions across all
   harmonic orders 1…N.

> **License and patent notice.** Free for **noncommercial** use (PolyForm Strict 1.0.0): run it
> for research, teaching and other noncommercial purposes. Commercial use, and modifying or
> redistributing the software, need separate permission; see [License](#license). The methods
> implemented here are the subject of a pending patent application.

## Install

```bash
pip install -e .              # numpy, scipy
pip install -e ".[plot]"      # + matplotlib   (figures, examples)
pip install -e ".[dev]"       # + pytest, ruff, pandas
```

Python ≥ 3.10. HEALPix pixelisation and spherical-harmonic analysis are implemented in
`gspwd._healpix`; `healpy` (GPL) is **not** required (only used as an optional test oracle).

## Quick start

```python
import numpy as np
from gspwd import (generalized_srf, gridangs, maxdfcoeff, scene_shd,
                   zonal_multiobjective_coeff)

N = 4
shd = scene_shd(amplitudes=[1.0, 0.8], thetas=[np.pi/2, np.pi/3], phis=[0.6, -1.6], NORDER=N)

# optimised kernel (Legendre coefficients) vs. the standard SPWD kernel
k, info = zonal_multiobjective_coeff(N, np.pi/(N + 0.5), lam_side=N**2,
                                     hard_backnull=True, normalize="energy")
th, ph = gridangs(3072)                       # nested HEALPix grid, nside 16
srf_gspwd = generalized_srf(k, shd, th, ph, N)
srf_spwd = generalized_srf(maxdfcoeff(N), shd, th, ph, N)
```

Runnable scripts are in [`examples/`](examples):

| script | shows |
|---|---|
| `01_kernel_design.py` | Dirichlet vs optimised kernels |
| `02_single_scene_srf.py` | SPWD vs GSPWD steering-response maps of a two-source scene |
| `03_multiscale_omp.py` | joint multi-order OMP on a random scene, with metrics |
| `04_monte_carlo_mini.py` | a miniature version of the Monte Carlo study |

## Package layout

| module | content |
|---|---|
| `gspwd.kernels` | `maxdfcoeff`, `zonal_multiobjective_coeff`, `gpulse` |
| `gspwd.srf` | `generalized_srf`, `generalizeddict`, `pwd_dictionaries` |
| `gspwd.omp` | `omp_multi_dict_shared` (joint OMP), `selectscores` |
| `gspwd.scenes` | `randomscene_rejection`, `planewave`, `scene_shd` |
| `gspwd.metrics` | `le`, `lr`, `ecm`, `detectionmetrics`, `support_aware_amplitude_error` |
| `gspwd.experiments` | `ExperimentConfig`, `run_trial`, `run_experiment` |
| `gspwd.plotting` | optional matplotlib helpers |

## Relation to the paper

The defaults reproduce the setup of the revised paper:

- **Metrics** (`gspwd.metrics`): localisation error `le`, localisation recall `lr`, cardinality
  mismatch `ecm = 1 - M/N` (positive means *under*-estimation), precision/recall/F1/FDR with
  `FP = M - TP` and `FN = max(0, N - M)` (DCASE-2019 style: mislocalised estimates count once, as
  false positives), and the support-aware amplitude error `saae` plus the matched-only `aerror`.
- **Noise**: isotropic diffuse field; the SNR is relative to the *weakest* source and exact for
  every realisation. OMP uses a noise-aware stopping threshold (`snr=`).
- **Estimators**: GSPWD vs SPWD, multi-order (joint, harmonic-mean selection) vs single-order.

`tests/golden/golden_v1.npz` was produced by the original research code; the test suite checks
that this package reproduces it. Two caveats:

- The research script drew its noise from an *unseeded* generator, so its noisy runs cannot be
  repeated bit-for-bit; the noise statistics are reproduced and tested.
- The multi-order joint OMP is numerically chaotic once one harmonic order is fitted to machine
  precision (its per-order correlations are renormalised by their maximum), so atoms selected
  after that point are not reproducible across machines. This also holds for the original code.

```python
from gspwd import ExperimentConfig, run_experiment
rows = run_experiment(ExperimentConfig(legacy_rng=True))  # global-stream scenes, as the script did
```

See [`CHANGELOG.md`](CHANGELOG.md) for everything that differs from the research code.

## Data and results

The raw simulation output (CSV) and the statistical analysis files of the study are archived
separately and are not part of this repository. <!-- TODO: add Zenodo DOI -->

## Citation

See [`CITATION.cff`](CITATION.cff). <!-- TODO: add the paper reference once accepted -->

## License

Licensed under the [PolyForm Strict License 1.0.0](LICENSE). In short, you may use the software
for any **noncommercial** purpose, including research, experiments, personal study, and use by
educational institutions, public research organisations, charities and government bodies. You may
**not** use it commercially, distribute it, or make changes or new works based on it. The license
text is authoritative; this summary is not.

**Research permission.** Noncommercial researchers may, on written request, receive permission to
modify the software and to share their modified versions for noncommercial research purposes, for
example to reproduce or extend the results of a publication, to compare against it in a paper, or
for a student thesis. Tell us what you plan to do and we will normally reply with a written
permission. Requests: **hhuseyin@metu.edu.tr**.

**Patents.** The methods implemented here are the subject of a pending patent application. The
licensor's patent license under the PolyForm Strict License is granted for the permitted
noncommercial purposes only; using the software or the methods commercially requires a separate
commercial license.

**Commercial use** (including use inside a company, a product or a paid service) requires a
separate commercial license. Contact the METU Technology Transfer Office: Mr. Yusuf Dudu,
**yusuf.dudu@odtuteknokent.com.tr**.

Copyright 2026 Huseyin Hacihabiboglu. This is a source-available license, not an OSI-approved
open-source license.
