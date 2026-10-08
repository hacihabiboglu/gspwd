import numpy as np
import pytest

from gspwd import (
    generalized_srf,
    generalizeddict,
    gridangs,
    maxdfcoeff,
    planewave,
    zonal_multiobjective_coeff,
)
from gspwd.kernels import gpulse
from gspwd.sphere import pix2vec


def _kernels(N):
    opt, _ = zonal_multiobjective_coeff(
        N, np.pi / (N + 0.5), lam_side=N**2, mu_back=1.0, hard_backnull=True, normalize="energy"
    )
    return {"maxdf": maxdfcoeff(N), "optimised": opt}


@pytest.mark.parametrize("N", [2, 4, 6])
@pytest.mark.parametrize("name", ["maxdf", "optimised"])
def test_srf_peak_equals_kernel_peak(N, name):
    """Addition theorem: a unit plane wave from pixel p gives srf[p] = sum_n a[n] = g(0).

    This ties the SHD (planewave) scale to the zonal-kernel (dictionary) scale, so the
    coefficients of the joint fit are the true source amplitudes.
    """
    a = _kernels(N)[name]
    th, ph = gridangs(192)
    g0 = gpulse(a, 0.0)
    assert g0 == pytest.approx(a.sum())
    for p in np.linspace(0, 191, 8).astype(int):
        srf = generalized_srf(a, planewave(1.0, th[p], ph[p], N), th, ph, N)
        assert srf[p] == pytest.approx(g0, rel=1e-9)


def test_srf_is_linear_in_amplitude_and_superposition():
    N, a = 3, maxdfcoeff(3)
    th, ph = gridangs(48)
    s1, s2 = planewave(1.0, 0.7, 1.1, N), planewave(0.5j, 2.0, -0.4, N)
    f = lambda s: generalized_srf(a, s, th, ph, N)  # noqa: E731
    np.testing.assert_allclose(f(2 * s1 + s2), 2 * f(s1) + f(s2), atol=1e-12)


@pytest.mark.parametrize("name", ["maxdf", "optimised"])
def test_dictionary_atom_equals_srf_of_unit_source(name):
    """Column ``p`` of the dictionary is the SRF (on the look grid) of a unit source at peak
    pixel ``p``: the dictionary and the SRF are the same operator, so OMP can invert it."""
    N, npix_look, npix_peak = 4, 48, 192
    a = _kernels(N)[name]
    D, norms = generalizeddict(a, npix_look, npix_peak)
    th_l, ph_l = gridangs(npix_look)
    th_p, ph_p = gridangs(npix_peak)
    assert D.shape == (npix_look, npix_peak) and len(norms) == npix_peak
    for p in (0, 17, 101, 191):
        srf = generalized_srf(a, planewave(1.0, th_p[p], ph_p[p], N), th_l, ph_l, N)
        np.testing.assert_allclose(srf.real, D[:, p], atol=1e-9)
        np.testing.assert_allclose(srf.imag, 0.0, atol=1e-9)


def test_dictionary_columns_symmetric_in_look_and_peak_grids():
    a = maxdfcoeff(3)
    D, _ = generalizeddict(a, 48, 48)
    np.testing.assert_allclose(D, D.T, atol=1e-12)  # same grid on both axes -> symmetric
    v = pix2vec(48, np.arange(48))
    np.testing.assert_allclose(np.diag(D), gpulse(a, 0.0), rtol=1e-12)
    cos = np.clip(v.T @ v, -1, 1)
    np.testing.assert_allclose(D[3, 20], gpulse(a, np.arccos(cos[3, 20])), rtol=1e-10)
