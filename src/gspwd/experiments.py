"""Monte Carlo comparison of GSPWD and SPWD, single-order and multi-order.

This is the library form of the original experiment script: for every
(SNR, number of sources, maximum SH order N) cell it draws random scenes, runs the four
estimators and scores them. Nothing here touches the file system; use
:func:`run_experiment` and write the returned rows yourself (``pandas.DataFrame(rows)``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from . import _healpix as hpx
from .metrics import amplitude_errors, detectionmetrics, ecm, le, lr
from .omp import omp_multi_dict_shared
from .scenes import randomscene_rejection, scene_shd
from .sphere import gridangs, pix2vec
from .srf import generalized_srf, pwd_dictionaries

METHODS = ("GSPWD Multiple", "SPWD Multiple", "GSPWD Single", "SPWD Single")


@dataclass
class ExperimentConfig:
    """Parameters of the experiment (defaults are those of the original study)."""

    numpix: int = 48  #: look-grid pixels (nested HEALPix, nside 2)
    numpeak: int = 768 * 4  #: candidate-direction pixels (nside 16)
    minn: int = 1
    maxn: int = 6
    thr: float = 1e-2  #: OMP stopping threshold on the joint residual
    snr_list: tuple = (np.inf, 40, 30, 20, 10)
    numsources: tuple = tuple(range(1, 11))
    numsim: int = 50
    amin: float = 0.2  #: minimum source amplitude modulus
    seed: int = 42
    amplitude_error: bool = True
    snr_aware_stop: bool = True  #: noise-aware OMP stopping threshold (see omp_multi_dict_shared)
    unitflag: bool = False  #: unit-modulus source amplitudes
    legacy_rng: bool = False  #: reseed the global numpy stream per cell as the original did
    maxdf_baseline: bool = True  #: also run the SPWD (maximum-directivity) baselines
    single: bool = True  #: also run the single-order estimators

    @property
    def resol(self) -> float:
        """Match threshold: HEALPix pixel size of the look grid (radians)."""
        return hpx.nside2resol(hpx.npix2nside(self.numpix))

    @property
    def theta0(self) -> list[float]:
        """Mainlobe half-widths used for kernel design, ``pi / (N + 1/2)``."""
        return [np.pi / (n + 0.5) for n in range(self.maxn + 1)]


@dataclass
class Dictionaries:
    """Kernels and per-order dictionaries shared by all trials of an experiment."""

    gspwd: list  #: one (numpix, numpeak) dictionary per order, optimised kernel
    spwd: list  #: same with the maximum-directivity kernel
    kernels: list  #: optimised kernel coefficients per order
    maxdf: list  #: maximum-directivity kernel coefficients per order


def build_dictionaries(cfg: ExperimentConfig) -> Dictionaries:
    Dd, Ds, opt, mdf = pwd_dictionaries(
        cfg.maxn, cfg.theta0, cfg.numpix, cfg.numpeak, cfg.maxdf_baseline, hardbacknull=True
    )
    return Dictionaries(Dd, Ds, opt, mdf)


def cell_rng(cfg: ExperimentConfig, snr: float, nsource: int, n: int):
    """Random source for one (SNR, nsource, N) cell.

    With ``legacy_rng`` the legacy global stream is reseeded (``np.random.seed(seed)``) and
    ``None`` is returned, exactly as the original script did. Otherwise an independent
    ``numpy.random.Generator`` is derived from ``(seed, snr, nsource, n)``.
    """
    if cfg.legacy_rng:
        np.random.seed(cfg.seed)
        return None
    snr_key = -1 if np.isinf(snr) else int(round(snr))
    return np.random.default_rng([cfg.seed, snr_key + 1, nsource, n])


def _score(cfg, method, support, X, vecscene, amps, extra, elapsed):
    vecres = pix2vec(cfg.numpeak, support).T
    row = {"type": method, **extra}
    row.update(
        le=le(vecres, vecscene, THETA=cfg.resol),
        lr=lr(vecres, vecscene, THETA=cfg.resol),
        ecm=ecm(vecres, vecscene, THETA=cfg.resol),
    )
    row.update(detectionmetrics(vecres, vecscene, THETA=cfg.resol))
    row["time"] = elapsed
    if cfg.amplitude_error:
        saae, aerror = amplitude_errors(vecres, X, vecscene, amps.flatten(), THETA=cfg.resol)
        row["saae"] = 20 * np.log10(saae)
        row["aerror"] = 20 * np.log10(aerror)
    return row


def run_trial(
    cfg, dicts, snr, nsource, n, rng=None, rep=0, scene=None, ths_phs=None, noise_rng=None
):
    """Draw one scene and score every estimator. Returns a list of result rows (dicts).

    ``rng`` draws the scene (and the noise, unless ``noise_rng`` is given). ``scene`` may supply
    ``(amps, thetas, phis)`` instead of drawing one.
    """
    noise_rng = rng if noise_rng is None else noise_rng
    ths, phs = ths_phs if ths_phs is not None else gridangs(cfg.numpix)
    if scene is None:
        amps, thes, phis = randomscene_rejection(
            nsource,
            AMIN=cfg.amin,
            ANGMIN=(np.pi / max(3 * n, 4)),
            unitflag=cfg.unitflag,
            rng=rng,
        )
    else:
        amps, thes, phis = scene
    vecscene = np.stack(
        [np.sin(thes) * np.cos(phis), np.sin(thes) * np.sin(phis), np.cos(thes)], axis=1
    )
    extra = {
        "numpix": cfg.numpix,
        "numpeak": cfg.numpeak,
        "numsource": nsource,
        "maxcomp": cfg.numpix,
        "maxn": n,
        "thr": cfg.thr,
        "snr": snr,
        "resol": cfg.resol,
        "sceneid": f"sceneid_snr{snr}_numsource_{nsource}_rep{rep}",
    }

    dsrf, ssrf = [], []
    for order in range(1, n + 1):
        s = scene_shd(amps, thes, phis, order, snr, rng=noise_rng)
        dsrf.append(generalized_srf(dicts.kernels[order - 1], s, ths, phs, order))
        if cfg.maxdf_baseline:
            ssrf.append(generalized_srf(dicts.maxdf[order - 1], s, ths, phs, order))

    D_d, D_s = dicts.gspwd[:n], dicts.spwd[:n]
    common = dict(max_iters=cfg.numpix, threshold=cfg.thr, return_support=True)
    if cfg.snr_aware_stop:
        common["snr"] = snr
    rows = []

    def run(method, Y, D, aggregate):
        t0 = time.time()
        X, S, _ = omp_multi_dict_shared(Y, D, Y, D, aggregate=aggregate, **common)
        rows.append(_score(cfg, method, S, X, vecscene, amps, extra, time.time() - t0))

    run("GSPWD Multiple", dsrf, D_d, "harmonic_mean")
    if cfg.maxdf_baseline:
        run("SPWD Multiple", ssrf, D_s, "harmonic_mean")
    if cfg.single:
        run("GSPWD Single", [dsrf[-1]], [D_d[-1]], "mean_abs")
        if cfg.maxdf_baseline:
            run("SPWD Single", [ssrf[-1]], [D_s[-1]], "mean_abs")
    return rows


def iter_experiment(cfg: ExperimentConfig, dicts: Dictionaries | None = None) -> Iterator[dict]:
    """Yield result rows for the whole experiment grid (SNR x sources x N x repetitions)."""
    dicts = dicts or build_dictionaries(cfg)
    ths_phs = gridangs(cfg.numpix)
    for snr in cfg.snr_list:
        for nsource in cfg.numsources:
            for n in range(cfg.minn, cfg.maxn + 1):
                rng = cell_rng(cfg, snr, nsource, n)
                for rep in range(cfg.numsim):
                    yield from run_trial(cfg, dicts, snr, nsource, n, rng, rep, ths_phs=ths_phs)


def run_experiment(
    cfg: ExperimentConfig | None = None, dicts: Dictionaries | None = None
) -> list[dict]:
    """Run the full experiment and return all result rows as a list of dicts."""
    return list(iter_experiment(cfg or ExperimentConfig(), dicts))
