"""Random scene generation: plane-wave sources in the spherical-harmonic domain."""

import warnings

import numpy as np
from scipy.special import sph_harm_y

from . import _healpix as hpx
from .sphere import reorderalms


def _rng_or_global(rng):
    """``rng``: a ``numpy.random.Generator``/``RandomState``; ``None`` is the global stream."""
    return np.random if rng is None else rng


def box_muller(rng=None):
    """One circular complex Gaussian sample (shape ``(1,)``) via Box-Muller."""
    rng = _rng_or_global(rng)
    u1 = rng.random(1)
    u2 = rng.random(1)
    t = np.sqrt((-2) * np.log(u1))
    v = 2 * np.pi * u2
    return t * (np.cos(v) + 1j * np.sin(v))


def randomscene_rejection(
    nsource, AMIN=0.3, ANGMIN=np.pi / 20, max_tries=1000, unitflag=False, rng=None
):
    """Random plane-wave scene with a minimum angular separation between sources.

    Directions are uniform on the sphere (area-uniform) and rejected if closer than ``ANGMIN``
    to an accepted source. Amplitudes are complex Gaussian with modulus clipped to
    ``[AMIN, 1]``.

    Parameters
    ----------
    nsource : int
        Number of sources.
    max_tries : int
        Maximum number of *rejected* candidates before giving up. When exhausted, a
        ``RuntimeWarning`` is issued and fewer than ``nsource`` sources are returned.
    unitflag : bool
        Give every source unit amplitude modulus (keeping its random phase).
    rng : numpy.random.Generator, optional
        Random source. ``None`` uses the legacy global ``numpy.random`` stream.

    Returns
    -------
    amps (nsource, 1) complex, thetas (nsource,), phis (nsource,)
    """
    rng = _rng_or_global(rng)
    amps, thes, phis = [], [], []
    vectors = []
    rejections = 0

    while len(thes) < nsource:
        the = np.arccos(1 - 2 * rng.random())  # arccos(1-2u): uniform in area
        phi = rng.random() * 2 * np.pi
        new_vec = np.array([np.sin(the) * np.cos(phi), np.sin(the) * np.sin(phi), np.cos(the)])

        too_close = any(np.arccos(np.clip(np.dot(new_vec, v), -1.0, 1.0)) < ANGMIN for v in vectors)
        if too_close:
            rejections += 1
            if rejections >= max_tries:
                warnings.warn(
                    f"Could only place {len(thes)} of {nsource} sources "
                    f"with ANGMIN={ANGMIN:.3g} rad.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
            continue

        amp = box_muller(rng)
        if np.abs(amp) < AMIN:
            amp = (amp / np.abs(amp)) * AMIN
        if np.abs(amp) > 1.0:
            amp = amp / np.abs(amp)
        thes.append(the)
        phis.append(phi)
        amps.append(amp)
        vectors.append(new_vec)

    if unitflag:
        amps = [a / np.abs(a) for a in amps]
    return np.array(amps), np.array(thes), np.array(phis)


def planewave(A, theta, phi, n_order):
    """SHD coefficients (ACN order) of a plane wave with amplitude ``A`` from ``(theta, phi)``."""
    shpw = []
    for n in range(n_order + 1):
        for m in range(-n, n + 1):
            s = sph_harm_y(n, m, theta, phi)
            shpw.append(np.conj(s))
    return A * np.array(shpw)


def diffuse_alm_unit_power(lmax, rng=None):
    """Isotropic diffuse field with unit power per mode, in HEALPix real-map ``a_lm`` storage.

    ``E|a_lm|^2 = 1`` for every physical ``(l, m)``, i.e. a flat power spectrum
    ``E{C_l} = 1``. Only ``m >= 0`` is stored for a real field: ``m = 0`` is real with unit
    variance, ``m > 0`` is circular complex with unit variance (its implicit ``(l, -m)``
    partner carries the same power).

    ``rng`` may be a ``numpy.random.Generator``; ``None`` uses the global ``numpy.random`` stream.
    """
    rng = _rng_or_global(rng)
    _, m_arr = hpx.alm_lm(lmax)
    alm = np.zeros(m_arr.size, dtype=np.complex128)
    m0 = m_arr == 0
    mp = ~m0
    alm[m0] = rng.standard_normal(m0.sum())
    alm[mp] = (rng.standard_normal(mp.sum()) + 1j * rng.standard_normal(mp.sum())) / np.sqrt(2.0)
    return alm


def scene_shd(amplitudes, thetas, phis, NORDER, snr=np.inf, rng=None):
    """Spherical-harmonic-domain (ACN-ordered) coefficients of a plane-wave scene plus noise.

    The noise is an isotropic diffuse field (:func:`diffuse_alm_unit_power`) scaled for this
    realisation so that its total energy in the ``(NORDER+1)**2`` ACN channels equals the
    energy of the **weakest source** divided by ``10**(snr/10)``. ``snr`` (dB) is therefore the
    SNR of the weakest source and exact for every draw. ``snr=np.inf`` is noiseless.
    """
    s = 0.0
    for ind in range(len(amplitudes)):
        s += planewave(amplitudes[ind], thetas[ind], phis[ind], NORDER)

    if snr != np.inf:
        imin = np.argmin(np.abs(amplitudes))
        smin = planewave(amplitudes[imin], thetas[imin], phis[imin], NORDER)
        nss = reorderalms(diffuse_alm_unit_power(NORDER, rng), NORDER)
        spower = np.sum(np.abs(smin) ** 2)
        npower = np.sum(np.abs(nss) ** 2)
        ngain = np.sqrt(spower / npower) / 10 ** (snr / 20)
        s = s + nss * ngain
    return s
