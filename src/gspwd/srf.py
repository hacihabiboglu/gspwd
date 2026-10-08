"""Steering-response functions and dictionaries built from zonal kernels."""

import numpy as np
from scipy.special import eval_legendre, sph_harm_y

from . import _healpix as hpx
from .kernels import maxdfcoeff, zonal_multiobjective_coeff


def generalized_srf(a, shd, th_grid, ph_grid, n_order):
    npix = len(th_grid)
    sall = np.zeros((npix, 0), dtype=complex)
    for n in range(n_order + 1):
        for m in range(-n, n + 1):
            s = a[n] * sph_harm_y(n, m, th_grid, ph_grid).reshape(npix, 1) / (2 * n + 1) * 4 * np.pi
            sall = np.column_stack((sall, s))
    gsrf = sall @ shd
    return gsrf


def generalizeddict(a, npix_look, npix_peak):
    """Dictionary ``G[look, peak] = k(angle between look and peak pixel centres)``.

    Both grids are nested HEALPix grids; ``a`` are the Legendre coefficients of the kernel.
    Returns ``(G, column_norms)``.
    """
    look = hpx.pix2vec(hpx.npix2nside(npix_look), np.arange(npix_look), nest=True)
    peak = hpx.pix2vec(hpx.npix2nside(npix_peak), np.arange(npix_peak), nest=True)
    look = look / np.linalg.norm(look, axis=0)
    peak = peak / np.linalg.norm(peak, axis=0)
    cx = look.T @ peak  # (npix_look, npix_peak) cosines of the angles
    gdict = np.zeros_like(cx)
    for n, an in enumerate(np.asarray(a)):
        gdict += an * eval_legendre(n, cx)
    return gdict, list(np.linalg.norm(gdict, axis=0))


def pwd_dictionaries(MAXN, THETA0, numpix, numpeak, maxdfflag, hardbacknull=True):
    Dcompound_d = []
    Dcompound_s = []
    mdflist = []
    optlist = []
    for NORDER in range(1, MAXN + 1):
        maxdf = maxdfcoeff(NORDER)
        mdflist.append(maxdf)
        k, info = zonal_multiobjective_coeff(
            NORDER,
            THETA0[NORDER],
            lam_side=NORDER**2,
            mu_back=1.0,
            hard_backnull=hardbacknull,
            normalize="energy",
        )

        optlist.append(k)
        # Dc, _ = generalizeddict(c, numpix, numpeak)
        Dd, _ = generalizeddict(k, numpix, numpeak)
        Dcompound_d.append(Dd)
        if maxdfflag:
            Ds, _ = generalizeddict(maxdf, numpix, numpeak)
            Dcompound_s.append(Ds)
    return Dcompound_d, Dcompound_s, optlist, mdflist
