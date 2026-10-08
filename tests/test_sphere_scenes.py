import warnings

import numpy as np
import pytest
from scipy.special import sph_harm_y

from gspwd import _healpix as hpx
from gspwd.scenes import diffuse_alm_unit_power, planewave, randomscene_rejection, scene_shd
from gspwd.sphere import acn_index, gridangs, pix2vec, reorderalms


def _angles(vecs):
    return np.arccos(np.clip(vecs @ vecs.T, -1, 1))


def test_acn_index():
    assert [acn_index(l, m) for l in range(3) for m in range(-l, l + 1)] == list(range(9))


def test_gridangs_and_vectors():
    th, ph = gridangs(48)
    assert th.shape == ph.shape == (48,)
    assert np.all((th >= 0) & (th <= np.pi)) and np.all((ph >= 0) & (ph < 2 * np.pi))
    v = pix2vec(48, np.arange(48))
    np.testing.assert_allclose(np.linalg.norm(v, axis=0), 1.0)
    # equal-area pixelisation: centres have zero mean
    np.testing.assert_allclose(v.mean(axis=1), 0.0, atol=1e-12)


@pytest.mark.parametrize("N", [1, 3, 5])
def test_reorderalms_gives_real_field(N):
    """ACN reordering must encode a REAL field: synthesis from all (l, m) is real and equals
    the synthesis from the stored m >= 0 half (which counts m > 0 twice)."""
    alm = diffuse_alm_unit_power(N, np.random.default_rng(N))
    acn = reorderalms(alm, N)
    th, ph = gridangs(48)
    full = np.zeros(48, dtype=complex)
    half = np.zeros(48)
    for l in range(N + 1):
        for m in range(-l, l + 1):
            full += acn[acn_index(l, m)] * sph_harm_y(l, m, th, ph)
        for m in range(l + 1):
            half += (1 if m == 0 else 2) * np.real(
                alm[hpx.alm_index(N, l, m)] * sph_harm_y(l, m, th, ph)
            )
    np.testing.assert_allclose(full.imag, 0.0, atol=1e-12)
    np.testing.assert_allclose(full.real, half, atol=1e-12)


def test_planewave_shape_and_addition_theorem():
    N, A = 4, 0.7 - 0.2j
    pw = planewave(A, 0.9, 2.1, N)
    assert pw.shape == ((N + 1) ** 2,)
    for l in range(N + 1):  # sum_m |Y_lm|^2 = (2l+1)/(4 pi)
        block = pw[l**2 : (l + 1) ** 2]
        assert np.sum(np.abs(block) ** 2) == pytest.approx(abs(A) ** 2 * (2 * l + 1) / (4 * np.pi))


# ---------------------------------------------------------------- scenes
def test_scene_constraints():
    rng = np.random.default_rng(1)
    angmin, amin = np.pi / 12, 0.2
    amps, th, ph = randomscene_rejection(6, AMIN=amin, ANGMIN=angmin, rng=rng)
    assert amps.shape == (6, 1) and th.shape == ph.shape == (6,)
    v = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
    ang = _angles(v)
    np.fill_diagonal(ang, np.inf)
    assert ang.min() >= angmin
    mod = np.abs(amps)
    assert mod.min() >= amin - 1e-12 and mod.max() <= 1 + 1e-12


def test_scene_is_reproducible_with_generator():
    a = randomscene_rejection(4, rng=np.random.default_rng(5))
    b = randomscene_rejection(4, rng=np.random.default_rng(5))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_scene_directions_are_area_uniform():
    rng = np.random.default_rng(2)
    z = np.concatenate(
        [np.cos(randomscene_rejection(5, ANGMIN=0.0, rng=rng)[1]) for _ in range(400)]
    )
    # uniform area <=> cos(theta) uniform on [-1, 1]: mean 0, variance 1/3
    assert abs(z.mean()) < 0.05
    assert z.var() == pytest.approx(1 / 3, abs=0.03)


def test_scene_warns_when_infeasible():
    with pytest.warns(RuntimeWarning, match="Could only place"):
        amps, th, ph = randomscene_rejection(
            5, ANGMIN=np.pi * 0.9, max_tries=50, rng=np.random.default_rng(0)
        )
    assert len(th) < 5


def test_max_tries_counts_only_rejections():
    # 10 well-separated sources need far fewer than 10 rejections allowed to be tried: with the
    # legacy semantics (every iteration counted) max_tries=12 would stop at <= 12 sources anyway,
    # but max_tries=1 must still succeed when nothing is ever rejected (ANGMIN=0).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, th, _ = randomscene_rejection(10, ANGMIN=0.0, max_tries=1, rng=np.random.default_rng(0))
    assert len(th) == 10


def test_scene_shd_noiseless_is_sum_of_planewaves():
    amps, th, ph = randomscene_rejection(3, rng=np.random.default_rng(3))
    s = scene_shd(amps, th, ph, 3)
    expected = sum(planewave(a, t, p, 3) for a, t, p in zip(amps, th, ph))
    np.testing.assert_allclose(s, expected)
    assert s.shape == (16,)


def test_noise_energy_is_exact_relative_to_weakest_source():
    amps, th, ph = randomscene_rejection(4, rng=np.random.default_rng(3))
    s0 = scene_shd(amps, th, ph, 4)
    weakest = np.argmin(np.abs(amps))
    e_weak = np.sum(np.abs(planewave(amps[weakest], th[weakest], ph[weakest], 4)) ** 2)
    for snr in (30, 10, 0):
        d = scene_shd(amps, th, ph, 4, snr, rng=np.random.default_rng(1)) - s0
        # exact for every draw, not just on average
        assert np.sum(np.abs(d) ** 2) == pytest.approx(e_weak / 10 ** (snr / 10), rel=1e-12)


def test_noise_is_a_real_field_with_both_signs_of_m():
    amps, th, ph = randomscene_rejection(2, rng=np.random.default_rng(0))
    s0 = scene_shd(amps, th, ph, 3)
    d = scene_shd(amps, th, ph, 3, 10, rng=np.random.default_rng(1)) - s0
    for l in range(4):
        for m in range(1, l + 1):  # a_{l,-m} = (-1)^m conj(a_{l,m})
            assert d[acn_index(l, -m)] == pytest.approx((-1) ** m * np.conj(d[acn_index(l, m)]))
            assert abs(d[acn_index(l, -m)]) > 0
    for l in range(4):
        assert abs(d[acn_index(l, 0)].imag) < 1e-15


def test_diffuse_field_has_unit_power_per_mode():
    lmax, n = 4, 4000
    rng = np.random.default_rng(0)
    alm = np.array([diffuse_alm_unit_power(lmax, rng) for _ in range(n)])
    l, m = hpx.alm_lm(lmax)
    power = np.mean(np.abs(alm) ** 2, axis=0)
    np.testing.assert_allclose(power, 1.0, atol=0.08)  # E|a_lm|^2 = 1 for every stored mode
    assert np.all(alm[:, m == 0].imag == 0)  # m = 0 coefficients are real


def test_noise_is_independent_between_calls_and_reproducible_by_seed():
    amps, th, ph = randomscene_rejection(2, rng=np.random.default_rng(0))
    a = scene_shd(amps, th, ph, 3, 10, rng=np.random.default_rng(1))
    b = scene_shd(amps, th, ph, 3, 10, rng=np.random.default_rng(1))
    c = scene_shd(amps, th, ph, 3, 10, rng=np.random.default_rng(2))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_unitflag_gives_unit_modulus_amplitudes():
    amps, _, _ = randomscene_rejection(5, unitflag=True, rng=np.random.default_rng(0))
    np.testing.assert_allclose(np.abs(amps), 1.0)
