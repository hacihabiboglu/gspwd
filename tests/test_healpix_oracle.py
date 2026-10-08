"""Equivalence of gspwd._healpix with healpy (dev-only oracle; skipped without healpy)."""

import numpy as np
import pytest
from scipy.special import sph_harm_y

from gspwd import _healpix as gh

hp = pytest.importorskip("healpy")
pytestmark = pytest.mark.oracle

NSIDES = [1, 2, 4, 8, 16]


@pytest.mark.parametrize("nside", NSIDES)
@pytest.mark.parametrize("nest", [True, False])
def test_pix2ang_all_pixels(nside, nest):
    pix = np.arange(12 * nside**2)
    th, ph = gh.pix2ang(nside, pix, nest=nest)
    th0, ph0 = hp.pix2ang(nside, pix, nest=nest)
    np.testing.assert_allclose(th, th0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(ph, ph0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        gh.pix2vec(nside, pix, nest=nest), np.array(hp.pix2vec(nside, pix, nest=nest)), atol=1e-12
    )


@pytest.mark.parametrize("nside", NSIDES)
def test_nest_ring_conversion(nside):
    pix = np.arange(12 * nside**2)
    np.testing.assert_array_equal(gh.nest2ring(nside, pix), hp.nest2ring(nside, pix))
    np.testing.assert_array_equal(gh.ring2nest(nside, pix), hp.ring2nest(nside, pix))


def _hard_angles(rng, n=20000):
    z = rng.uniform(-1, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    # classic trouble spots: poles, equatorial-cap boundary z=+-2/3, phi seam
    zs = np.array(
        [1.0, -1.0, 2 / 3, -2 / 3, 2 / 3 + 1e-9, 2 / 3 - 1e-9, -2 / 3 + 1e-9, -2 / 3 - 1e-9, 0.0]
    )
    ps = np.array([0.0, 2 * np.pi - 1e-12, np.pi / 2, np.pi, 3 * np.pi / 2, 1e-12])
    zz, pp = np.meshgrid(zs, ps)
    z = np.concatenate([z, zz.ravel()])
    phi = np.concatenate([phi, pp.ravel()])
    return np.arccos(np.clip(z, -1, 1)), phi


@pytest.mark.parametrize("nside", NSIDES)
@pytest.mark.parametrize("nest", [True, False])
def test_ang2pix_random_and_edges(nside, nest):
    th, ph = _hard_angles(np.random.default_rng(0))
    mine = gh.ang2pix(nside, th, ph, nest=nest)
    ref = hp.ang2pix(nside, th, ph, nest=nest)
    # points exactly on a pixel boundary may fall on either side in floating point
    assert np.mean(mine == ref) > 0.9999
    c = gh.ang2vec(*gh.pix2ang(nside, mine, nest=nest))
    v = gh.ang2vec(th, ph)
    ang = np.arccos(np.clip(np.sum(c * v, axis=-1), -1, 1))
    assert ang.max() < 2.0 * hp.nside2resol(nside)


@pytest.mark.parametrize("nside", NSIDES)
@pytest.mark.parametrize("nest", [True, False])
def test_centres_roundtrip(nside, nest):
    pix = np.arange(12 * nside**2)
    th, ph = gh.pix2ang(nside, pix, nest=nest)
    np.testing.assert_array_equal(gh.ang2pix(nside, th, ph, nest=nest), pix)


@pytest.mark.parametrize("nside", NSIDES)
def test_resol_and_counts(nside):
    assert gh.nside2resol(nside) == pytest.approx(hp.nside2resol(nside), rel=1e-14)
    assert gh.nside2npix(nside) == hp.nside2npix(nside)
    assert gh.npix2nside(12 * nside**2) == hp.npix2nside(12 * nside**2)


@pytest.mark.parametrize("lmax", range(1, 8))
def test_alm_index(lmax):
    for m in range(lmax + 1):
        for l in range(m, lmax + 1):
            assert gh.alm_index(lmax, l, m) == hp.Alm.getidx(lmax, l, m)


@pytest.mark.parametrize("lmax", range(0, 8))
def test_alm_lm(lmax):
    l, m = gh.alm_lm(lmax)
    l0, m0 = hp.Alm.getlm(lmax)
    np.testing.assert_array_equal(l, l0)
    np.testing.assert_array_equal(m, m0)


def test_sph_harm_convention_matches_healpy():
    """scipy sph_harm_y(l, m, theta, phi) is the Y_lm healpy synthesises with."""
    nside, lmax = 8, 5
    rng = np.random.default_rng(3)
    n = hp.Alm.getsize(lmax)
    alm = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    alm[: lmax + 1] = alm[: lmax + 1].real  # m=0 coefficients are real for a real map
    f_hp = hp.alm2map(alm, nside, lmax=lmax)
    th, ph = hp.pix2ang(nside, np.arange(12 * nside**2))
    f = np.zeros_like(th)
    for m in range(lmax + 1):
        for l in range(m, lmax + 1):
            a = alm[hp.Alm.getidx(lmax, l, m)]
            f += (1 if m == 0 else 2) * np.real(a * sph_harm_y(l, m, th, ph))
    np.testing.assert_allclose(f, f_hp, atol=1e-10)
