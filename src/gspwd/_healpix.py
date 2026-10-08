"""Minimal pure-numpy implementation of the HEALPix subset used by GSPWD.

HEALPix is an equal-area sphere pixelisation (Gorski et al., ApJ 622, 759, 2005).
This module is written from the published pixelisation geometry and the standard
spherical-harmonic quadrature; it does not use or copy the ``healpy`` package
(GPL-2.0+). ``healpy`` is only used as a test oracle in ``tests/test_healpix_oracle.py``.

Provided:

* ``nside2npix``, ``npix2nside``, ``order2nside``, ``order2npix``, ``nside2resol``
* ``pix2ang`` / ``pix2vec`` / ``ang2pix`` / ``ang2vec`` for RING and NESTED ordering
* ``nest2ring`` / ``ring2nest``
* ``alm_index`` (healpy-compatible ``m >= 0`` storage index)
* ``alm_lm`` (degree/order of each stored coefficient)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "nside2npix",
    "npix2nside",
    "order2nside",
    "order2npix",
    "nside2resol",
    "pix2ang",
    "pix2vec",
    "ang2pix",
    "ang2vec",
    "nest2ring",
    "ring2nest",
    "alm_index",
    "alm_lm",
]

# Base-pixel face geometry (Gorski et al. 2005, Sec. 4): ring index and longitude
# offset of each of the 12 base faces.
_JRLL = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
_JPLL = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])


# --------------------------------------------------------------------------- #
# nside bookkeeping
# --------------------------------------------------------------------------- #
def _check_nside(nside: int) -> int:
    nside = int(nside)
    if nside < 1 or (nside & (nside - 1)) != 0:
        raise ValueError(f"nside must be a positive power of two, got {nside}")
    return nside


def nside2npix(nside: int) -> int:
    return 12 * _check_nside(nside) ** 2


def npix2nside(npix: int) -> int:
    nside = int(round(np.sqrt(npix / 12.0)))
    if nside < 1 or 12 * nside * nside != npix or (nside & (nside - 1)) != 0:
        raise ValueError(f"{npix} is not a valid HEALPix pixel count")
    return nside


def order2nside(order: int) -> int:
    return 1 << int(order)


def order2npix(order: int) -> int:
    return nside2npix(order2nside(order))


def nside2resol(nside: int) -> float:
    """Approximate pixel size in radians, ``sqrt(4*pi/npix)``."""
    return float(np.sqrt(np.pi / 3.0) / _check_nside(nside))


# --------------------------------------------------------------------------- #
# bit interleaving for the NESTED scheme
# --------------------------------------------------------------------------- #
def _spread_bits(v: np.ndarray) -> np.ndarray:
    """Place the bits of ``v`` on the even bit positions."""
    v = np.asarray(v, dtype=np.int64)
    out = np.zeros_like(v)
    for b in range(32):
        out |= ((v >> b) & 1) << (2 * b)
    return out


def _compress_bits(v: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_spread_bits`."""
    v = np.asarray(v, dtype=np.int64)
    out = np.zeros_like(v)
    for b in range(32):
        out |= ((v >> (2 * b)) & 1) << b
    return out


def _nest2xyf(nside: int, pix: np.ndarray):
    pix = np.asarray(pix, dtype=np.int64)
    npface = nside * nside
    face = pix // npface
    local = pix % npface
    ix = _compress_bits(local)
    iy = _compress_bits(local >> 1)
    return ix, iy, face


def _xyf2nest(nside: int, ix, iy, face):
    return face * nside * nside + _spread_bits(ix) + (_spread_bits(iy) << 1)


def _xyf2ring(nside: int, ix, iy, face):
    ix, iy, face = (np.asarray(a, dtype=np.int64) for a in (ix, iy, face))
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    jr = _JRLL[face] * nside - ix - iy - 1

    north = jr < nside
    south = jr > 3 * nside
    nr = np.where(north, jr, np.where(south, 4 * nside - jr, nside))
    n_before = np.where(
        north,
        2 * nr * (nr - 1),
        np.where(south, npix - 2 * (nr + 1) * nr, ncap + (jr - nside) * 4 * nside),
    )
    kshift = np.where(north | south, 0, (jr - nside) & 1)

    jp = (_JPLL[face] * nr + ix - iy + 1 + kshift) // 2
    jp = np.where(jp > 4 * nside, jp - 4 * nside, jp)
    jp = np.where(jp < 1, jp + 4 * nside, jp)
    return n_before + jp - 1


def nest2ring(nside: int, pix) -> np.ndarray:
    nside = _check_nside(nside)
    ix, iy, face = _nest2xyf(nside, pix)
    return _xyf2ring(nside, ix, iy, face)


def _ring2xyf_search(nside: int, pix: np.ndarray):
    """Exact RING -> (ix, iy, face) by inverting :func:`_xyf2ring` over all faces.

    Each RING pixel belongs to exactly one (face, ix, iy). Building the full
    forward table once per ``nside`` and inverting it is exact and simple; the
    table is cached.
    """
    table = _ring_inverse_table(nside)
    return table[0][pix], table[1][pix], table[2][pix]


_INV_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _ring_inverse_table(nside: int):
    if nside not in _INV_CACHE:
        npix = 12 * nside * nside
        nest = np.arange(npix, dtype=np.int64)
        ix, iy, face = _nest2xyf(nside, nest)
        ring = _xyf2ring(nside, ix, iy, face)
        ix_r = np.empty(npix, dtype=np.int64)
        iy_r = np.empty(npix, dtype=np.int64)
        face_r = np.empty(npix, dtype=np.int64)
        ix_r[ring], iy_r[ring], face_r[ring] = ix, iy, face
        _INV_CACHE[nside] = (ix_r, iy_r, face_r)
    return _INV_CACHE[nside]


def ring2nest(nside: int, pix) -> np.ndarray:
    nside = _check_nside(nside)
    ix, iy, face = _ring2xyf_search(nside, np.asarray(pix, dtype=np.int64))
    return _xyf2nest(nside, ix, iy, face)


# --------------------------------------------------------------------------- #
# pixel <-> angles
# --------------------------------------------------------------------------- #
def _ring_pix2zphi(nside: int, pix: np.ndarray):
    """z = cos(theta) and phi of RING-ordered pixel centres (Gorski et al. 2005, Eq. 3-5)."""
    pix = np.asarray(pix, dtype=np.int64)
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    fn = float(nside)

    z = np.empty(pix.shape, dtype=float)
    phi = np.empty(pix.shape, dtype=float)

    north = pix < ncap
    equat = (pix >= ncap) & (pix < npix - ncap)
    south = pix >= npix - ncap

    if north.any():
        ip = pix[north]
        iring = (1 + np.floor(np.sqrt(1 + 2 * ip) + 0.5).astype(np.int64)) >> 1
        iring = np.where(2 * iring * (iring - 1) > ip, iring - 1, iring)
        iring = np.where(2 * (iring + 1) * iring <= ip, iring + 1, iring)
        iphi = ip + 1 - 2 * iring * (iring - 1)
        z[north] = 1.0 - iring.astype(float) ** 2 / (3.0 * fn * fn)
        phi[north] = (iphi - 0.5) * np.pi / (2.0 * iring)
    if equat.any():
        ip = pix[equat] - ncap
        iring = ip // (4 * nside) + nside
        iphi = ip % (4 * nside) + 1
        fodd = np.where(((iring + nside) & 1) == 1, 1.0, 0.5)
        z[equat] = (2 * nside - iring) * 2.0 / (3.0 * fn)
        phi[equat] = (iphi - fodd) * np.pi / (2.0 * fn)
    if south.any():
        ip = npix - pix[south]
        iring = (1 + np.floor(np.sqrt(2 * ip - 1) + 0.5).astype(np.int64)) >> 1
        iring = np.where(2 * iring * (iring - 1) >= ip, iring - 1, iring)
        iring = np.where(2 * (iring + 1) * iring < ip, iring + 1, iring)
        iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1))
        z[south] = -1.0 + iring.astype(float) ** 2 / (3.0 * fn * fn)
        phi[south] = (iphi - 0.5) * np.pi / (2.0 * iring)
    return z, phi


def pix2ang(nside: int, pix, nest: bool = False):
    """Colatitude ``theta`` and longitude ``phi`` (radians) of pixel centres."""
    nside = _check_nside(nside)
    pix = np.asarray(pix, dtype=np.int64)
    npix = 12 * nside * nside
    if np.any(pix < 0) or np.any(pix >= npix):
        raise ValueError("pixel index out of range")
    ring = nest2ring(nside, pix) if nest else pix
    z, phi = _ring_pix2zphi(nside, ring)
    return np.arccos(np.clip(z, -1.0, 1.0)), phi


def pix2vec(nside: int, pix, nest: bool = False):
    """Unit vectors ``(x, y, z)`` of pixel centres, stacked along axis 0."""
    theta, phi = pix2ang(nside, pix, nest=nest)
    s = np.sin(theta)
    return np.stack([s * np.cos(phi), s * np.sin(phi), np.cos(theta)])


def ang2vec(theta, phi):
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    s = np.sin(theta)
    return np.stack([s * np.cos(phi), s * np.sin(phi), np.cos(theta)], axis=-1)


def ang2pix(nside: int, theta, phi, nest: bool = False):
    """Index of the pixel containing the direction ``(theta, phi)``."""
    nside = _check_nside(nside)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    z = np.cos(theta)
    za = np.abs(z)
    tt = np.mod(phi, 2.0 * np.pi) / (0.5 * np.pi)  # in [0, 4)
    fn = float(nside)

    # equatorial zone
    temp1 = fn * (0.5 + tt)
    temp2 = fn * z * 0.75
    jp_e = (temp1 - temp2).astype(np.int64)
    jm_e = (temp1 + temp2).astype(np.int64)
    ifp = jp_e // nside
    ifm = jm_e // nside
    face_e = np.where(ifp == ifm, ifp | 4, np.where(ifp < ifm, ifp, ifm + 8))
    ix_e = jm_e & (nside - 1)
    iy_e = nside - (jp_e & (nside - 1)) - 1

    # polar caps
    ntt = np.minimum(tt.astype(np.int64), 3)
    tp = tt - ntt
    tmp = fn * np.sqrt(3.0 * (1.0 - za))
    jp_p = np.minimum((tp * tmp).astype(np.int64), nside - 1)
    jm_p = np.minimum(((1.0 - tp) * tmp).astype(np.int64), nside - 1)
    north = z >= 0
    face_p = np.where(north, ntt, ntt + 8)
    ix_p = np.where(north, nside - jm_p - 1, jp_p)
    iy_p = np.where(north, nside - jp_p - 1, jm_p)

    equatorial = za <= 2.0 / 3.0
    ix = np.where(equatorial, ix_e, ix_p)
    iy = np.where(equatorial, iy_e, iy_p)
    face = np.where(equatorial, face_e, face_p)

    if nest:
        return _xyf2nest(nside, ix, iy, face)
    return _xyf2ring(nside, ix, iy, face)


# --------------------------------------------------------------------------- #
# a_lm storage layout
# --------------------------------------------------------------------------- #
def alm_index(lmax: int, l, m):
    """Storage index of ``a_lm`` (``m >= 0`` only, ``m``-major) as used by HEALPix."""
    l = np.asarray(l)
    m = np.asarray(m)
    return m * (2 * lmax + 1 - m) // 2 + l


def alm_lm(lmax: int):
    """Degrees ``l`` and orders ``m`` of every stored coefficient, in storage order."""
    m = np.concatenate([np.full(lmax - mm + 1, mm) for mm in range(lmax + 1)])
    l = np.concatenate([np.arange(mm, lmax + 1) for mm in range(lmax + 1)])
    return l, m
