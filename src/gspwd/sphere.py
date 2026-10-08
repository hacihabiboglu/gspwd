"""Sphere grids and spherical-harmonic channel ordering."""

import numpy as np

from . import _healpix as hpx


def acn_index(l: int, m: int) -> int:
    """Ambisonics Channel Number (ACN) index for degree l and order m."""
    return l * (l + 1) + m


def reorderalms(alms, N):
    """Convert ``m >= 0`` HEALPix-ordered coefficients to full ACN ordering up to order ``N``.

    Negative orders follow from the real-field symmetry ``a_{l,-m} = (-1)^m conj(a_{l,m})``.
    """
    acnarr = np.zeros((N + 1) ** 2, dtype=complex)
    for n in range(N + 1):  # pass 1: m >= 0 from the stored half
        for m in range(n + 1):
            acnarr[acn_index(n, m)] = alms[hpx.alm_index(N, n, m)]
    for n in range(N + 1):  # pass 2: m < 0 by Hermitian symmetry
        for m in range(-n, 0):
            acnarr[acn_index(n, m)] = (-1) ** m * np.conj(acnarr[acn_index(n, abs(m))])
    return acnarr


def gridangs(npix):
    """``(theta, phi)`` of all pixel centres of the nested HEALPix grid with ``npix`` pixels."""
    nside = hpx.npix2nside(npix)
    return hpx.pix2ang(nside, np.arange(0, npix), nest=True)


def pix2vec(npix, pix):
    """Unit vectors (shape ``(3,)`` or ``(3, n)``) of nested pixels on the ``npix``-pixel grid."""
    return hpx.pix2vec(hpx.npix2nside(npix), pix, nest=True)
