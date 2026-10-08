"""Generate golden regression data from the ORIGINAL (healpy-based) GSPWD code.

``legacy_src`` must hold the revised research sources (the ``genspwd`` working copy that
produced the r5 results). Run with the pinned legacy environment, from the repository root::

    .venv-legacy/bin/python scripts/make_golden.py legacy_src tests/golden/golden_v1.npz

The original sources are deliberately not part of the public repository; this script
exists so the provenance of ``tests/golden/golden_v0.npz`` is documented and repeatable.

The loop below mirrors the main block of the original ``legpulse.py`` exactly (same seeding
per (SNR, NSOURCE, N) cell, same call order), restricted to a handful of cells.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

legacy = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
sys.path.insert(0, str(legacy))

import healpy as hp  # noqa: E402
from gspdw import generalized_srf, gridangs, scene_shd  # noqa: E402
from jointomp import omp_multi_dict_shared  # noqa: E402
from legpulse import pwd_dictionaries  # noqa: E402
from metrics import detectionmetrics, ecm, le, lr, support_aware_amplitude_error  # noqa: E402
from utility import randomscene_rejection  # noqa: E402

NUMPIX, NUMPEAK, MAXN, THR = 48, 768 * 4, 6, 1e-2
MAXCOMP = NUMPIX
RESOL = hp.nside2resol(hp.npix2nside(NUMPIX))
THETA0 = [np.pi / (N + 0.5) for N in range(MAXN + 1)]
CELLS = [  # (snr, nsource, N)
    (np.inf, 1, 1),
    (np.inf, 2, 3),
    (np.inf, 4, 6),
    (30, 3, 4),
    (20, 3, 4),
    (10, 5, 6),
]
REPS = 3

ths, phs = gridangs(NUMPIX)
Dd, Ds, optlist, mdflist = pwd_dictionaries(MAXN, THETA0, NUMPIX, NUMPEAK, True, hardbacknull=True)

data: dict[str, np.ndarray] = {}
data["meta_cells"] = np.array([[c[0], c[1], c[2]] for c in CELLS], dtype=float)
data["meta_reps"] = np.array(REPS)
for i, k in enumerate(optlist):
    data[f"kernel_gspwd_N{i + 1}"] = np.asarray(k)
# one full dictionary (order 2, optimised kernel) for the dictionary-equality test
data["dict_gspwd_N2"] = Dd[1].astype(np.float64)
data["dict_spwd_N2"] = Ds[1].astype(np.float64)


def metrics_row(S, X, vecscene, amps):
    vx, vy, vz = hp.pix2vec(hp.npix2nside(NUMPEAK), S, nest=True)
    vecres = np.concatenate([vx.reshape(-1, 1), vy.reshape(-1, 1), vz.reshape(-1, 1)], axis=1)
    d = detectionmetrics(vecres, vecscene, THETA=RESOL)
    row = [
        le(vecres, vecscene, THETA=RESOL),
        lr(vecres, vecscene, THETA=RESOL),
        ecm(vecres, vecscene, THETA=RESOL),
    ]
    row += [d[k] for k in sorted(d)]
    row.extend(
        support_aware_amplitude_error(vecres, X, vecscene, amps.flatten(), THETA=RESOL)
    )  # (saae, aerror)
    return np.array(row, dtype=float), sorted(d)


PERTURB_EPS, PERTURB_TRIALS = 1e-14, 8
_prng = np.random.default_rng(12345)  # independent of the legacy global stream


def stable_prefix_len(call, Ys, Ds_, base_support):
    """Longest prefix of ``base_support`` that survives relative perturbations of the inputs.

    The original joint OMP is numerically chaotic once some order's residual is at machine
    precision (its per-order correlations are renormalised by their max, which amplifies
    rounding noise). Atoms chosen in that regime are not reproducible across machines/BLAS,
    so regression tests only demand exactness on the stable prefix.
    """
    stable = len(base_support)
    for _ in range(PERTURB_TRIALS):
        Yp = [y * (1 + PERTURB_EPS * _prng.standard_normal(np.shape(y))) for y in Ys]
        Dp = [d * (1 + PERTURB_EPS * _prng.standard_normal(d.shape)) for d in Ds_]
        S = list(call(Yp, Dp)[1])
        k = 0
        while k < min(len(S), len(base_support)) and S[k] == base_support[k]:
            k += 1
        stable = min(stable, k if (len(S) == len(base_support) or k < len(base_support)) else k)
        if len(S) != len(base_support):
            stable = min(stable, k)
    return stable


for ci, (SNR, NSOURCE, N) in enumerate(CELLS):
    np.random.seed(42)  # same stream policy as the original: reseed once per cell
    for ind in range(REPS):
        tag = f"c{ci}_r{ind}"
        amps, thes, phis = randomscene_rejection(NSOURCE, AMIN=0.2, ANGMIN=(np.pi / max(3 * N, 4)))
        vecscene = hp.ang2vec(thes, phis)
        truepix = hp.ang2pix(hp.npix2nside(NUMPEAK), thes, phis, nest=True)
        data[f"{tag}_amps"], data[f"{tag}_thes"], data[f"{tag}_phis"] = amps, thes, phis
        data[f"{tag}_truepix"] = truepix

        dsrf, ssrf = [], []
        # The research script left the noise generator unseeded; seed it so the file is repeatable.
        noise_rng = np.random.default_rng([100 + ci, ind])
        for NORDER in range(1, N + 1):
            s = scene_shd(amps, thes, phis, NORDER, SNR, rng=noise_rng)
            data[f"{tag}_shd{NORDER}"] = s
            dsrf.append(generalized_srf(optlist[NORDER - 1], s, ths, phs, NORDER))
            ssrf.append(generalized_srf(mdflist[NORDER - 1], s, ths, phs, NORDER))
        data[f"{tag}_dsrf_last"], data[f"{tag}_ssrf_last"] = dsrf[-1], ssrf[-1]

        D_d, D_s = Dd[:N], Ds[:N]
        runs = {
            "gm": omp_multi_dict_shared(
                dsrf,
                D_d,
                dsrf,
                D_d,
                max_iters=MAXCOMP,
                return_support=True,
                aggregate="harmonic_mean",
                threshold=THR,
                snrflag=True,
                snr=SNR,
            ),
            "sm": omp_multi_dict_shared(
                ssrf,
                D_s,
                ssrf,
                D_s,
                max_iters=MAXCOMP,
                threshold=THR,
                return_support=True,
                aggregate="harmonic_mean",
                snrflag=True,
                snr=SNR,
            ),
            "gs": omp_multi_dict_shared(
                [dsrf[-1]],
                [D_d[-1]],
                [dsrf[-1]],
                [D_d[-1]],
                max_iters=MAXCOMP,
                threshold=THR,
                return_support=True,
                aggregate="mean_abs",
                snrflag=True,
                snr=SNR,
            ),
            "ss": omp_multi_dict_shared(
                [ssrf[-1]],
                [D_s[-1]],
                [ssrf[-1]],
                [D_s[-1]],
                max_iters=MAXCOMP,
                threshold=THR,
                return_support=True,
                aggregate="mean_abs",
                snrflag=True,
                snr=SNR,
            ),
        }
        for name, (X, S, _R) in runs.items():
            data[f"{tag}_{name}_X"] = np.asarray(X)
            data[f"{tag}_{name}_S"] = np.asarray(S, dtype=np.int64)
            m, keys = metrics_row(S, X, vecscene, amps)
            data[f"{tag}_{name}_metrics"] = m
            if name in ("gm", "sm"):
                agg = "harmonic_mean"
                Ys_, Ds_ = (dsrf, D_d) if name == "gm" else (ssrf, D_s)
            else:
                agg = "mean_abs"
                Ys_, Ds_ = ([dsrf[-1]], [D_d[-1]]) if name == "gs" else ([ssrf[-1]], [D_s[-1]])

            def call(Y, D, agg=agg, SNR=SNR):
                return omp_multi_dict_shared(
                    Y,
                    D,
                    Y,
                    D,
                    max_iters=MAXCOMP,
                    threshold=THR,
                    return_support=True,
                    aggregate=agg,
                    snrflag=True,
                    snr=SNR,
                )

            data[f"{tag}_{name}_stable"] = np.array(stable_prefix_len(call, Ys_, Ds_, list(S)))
        print("done", tag, flush=True)

data["meta_detection_keys"] = np.array(keys)
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, **data)
print("wrote", out, out.stat().st_size / 1e6, "MB")
