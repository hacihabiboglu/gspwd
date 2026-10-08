import numpy as np
import pytest

from gspwd.kernels import (
    gpulse,
    maxdfcoeff,
    zonal_multiobjective_coeff,
)


@pytest.mark.parametrize("N", range(1, 7))
def test_maxdf_peak_value(N):
    # sum_n (2n+1) = (N+1)^2 and P_n(1) = 1
    assert gpulse(maxdfcoeff(N), 0.0) == pytest.approx((N + 1) ** 2 / (4 * np.pi))


@pytest.mark.parametrize("N", range(1, 7))
def test_maxdf_is_dirichlet_kernel(N):
    # Christoffel-Darboux: sum_{n<=N} (2n+1) P_n(x) = (N+1) (P_{N+1}(x) - P_N(x)) / (x - 1)
    from scipy.special import eval_legendre

    theta = np.linspace(0.1, 3.0, 25)
    x = np.cos(theta)
    expected = (N + 1) / (4 * np.pi) * (eval_legendre(N + 1, x) - eval_legendre(N, x)) / (x - 1)
    got = np.array([gpulse(maxdfcoeff(N), t) for t in theta])
    np.testing.assert_allclose(got, expected, rtol=1e-10)


@pytest.mark.parametrize("N", [2, 4, 6])
def test_multiobjective_constraints(N):
    c, info = zonal_multiobjective_coeff(
        N, np.pi / (N + 0.5), lam_side=N**2, mu_back=1.0, hard_backnull=True, normalize="peak"
    )
    assert c.shape == (N + 1,)
    assert np.sum(c) == pytest.approx(1.0, abs=1e-9)  # unit peak
    assert gpulse(c, np.pi) == pytest.approx(0.0, abs=1e-9)  # hard back-lobe null
    assert info["sidelobe_energy"] >= 0


@pytest.mark.parametrize("N", [2, 4, 6])
def test_multiobjective_energy_normalisation(N):
    c, _ = zonal_multiobjective_coeff(
        N, np.pi / (N + 0.5), lam_side=N**2, hard_backnull=True, normalize="energy"
    )
    energy = np.sum(c**2 * 2.0 / (2 * np.arange(N + 1) + 1))
    assert energy == pytest.approx(1.0)


def test_multiobjective_beats_dirichlet_sidelobes():
    N, theta0 = 4, np.pi / 4.5
    c, _ = zonal_multiobjective_coeff(
        N, theta0, lam_side=N**2, hard_backnull=True, normalize="peak"
    )
    d = maxdfcoeff(N) / maxdfcoeff(N).sum()
    theta = np.linspace(theta0, np.pi, 200)

    def side(k):
        return np.sum([gpulse(k, t) ** 2 * np.sin(t) for t in theta])

    assert side(c) < side(d)


def test_multiobjective_invalid_theta0():
    with pytest.raises(ValueError):
        zonal_multiobjective_coeff(3, 0.0)
    with pytest.raises(ValueError):
        zonal_multiobjective_coeff(3, 4.0)
