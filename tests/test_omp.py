import numpy as np
import pytest

from gspwd.omp import omp_multi_dict_shared, selectscores


def _problem(seed=0, n_orders=3, M=30, K=80, support=(5, 41), coefs=(1.0 + 0.5j, -0.7j)):
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((n_orders, M, K))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    x = np.zeros(K, dtype=complex)
    x[list(support)] = coefs
    Y = np.einsum("imk,k->im", D, x)
    return Y, D, list(support), np.array(coefs)


def test_recovers_noiseless_support_and_coefficients():
    Y, D, support, coefs = _problem()
    Xi, S, Ri = omp_multi_dict_shared(
        Y, D, Y, D, max_iters=10, return_support=True, aggregate="mean_abs", threshold=1e-12
    )
    assert sorted(S) == sorted(support)
    order = [S.index(k) for k in support]
    np.testing.assert_allclose(np.asarray(Xi)[order], coefs, atol=1e-9)
    assert Ri < 1e-9


@pytest.mark.parametrize("aggregate", ["mean_abs", "harmonic_mean", "geometric_mean", "mean_sq"])
def test_aggregates_find_the_support(aggregate):
    Y, D, support, _ = _problem(seed=1)
    _, S, _ = omp_multi_dict_shared(
        Y, D, Y, D, max_iters=2, return_support=True, aggregate=aggregate
    )
    assert sorted(S) == sorted(support)


def test_single_signal_and_single_dictionary_forms():
    Y, D, support, _ = _problem(seed=2)
    # 1-D signal + 2-D dictionary (the "single-order" form) must work and agree with the list form
    _, S1, _ = omp_multi_dict_shared(
        Y[0], D[0], Y[0], D[0], max_iters=2, return_support=True, aggregate="mean_abs"
    )
    _, S2, _ = omp_multi_dict_shared(
        [Y[0]], [D[0]], [Y[0]], [D[0]], max_iters=2, return_support=True, aggregate="mean_abs"
    )
    assert S1 == S2 and sorted(S1) == sorted(support)


def test_stopping_rules():
    Y, D, *_ = _problem(seed=3)
    _, S, _ = omp_multi_dict_shared(
        Y,
        D,
        Y,
        D,
        n_nonzero_coefs=1,
        max_iters=10,
        return_support=True,
        aggregate="mean_abs",
        threshold=0.0,
    )
    assert len(S) == 1
    _, S, _ = omp_multi_dict_shared(
        Y, D, Y, D, max_iters=3, return_support=True, aggregate="mean_abs", threshold=0.0
    )
    assert len(S) == 3
    # eps rule (mean residual norm) stops as soon as the signal is explained
    _, S, _ = omp_multi_dict_shared(
        Y, D, Y, D, max_iters=10, eps=1e-8, return_support=True, aggregate="mean_abs", threshold=0.0
    )
    assert len(S) == 2


def test_inputs_are_not_modified():
    Y, D, *_ = _problem(seed=4)
    Y0, D0 = Y.copy(), D.copy()
    omp_multi_dict_shared(Y, D, Y, D, max_iters=3, return_support=True)
    np.testing.assert_array_equal(Y, Y0)
    np.testing.assert_array_equal(D, D0)


def test_atoms_are_never_reselected_when_exhausted():
    D = np.random.default_rng(0).standard_normal((1, 6, 4))
    # all-zero correlations must not raise (division guarded); selection simply proceeds
    Xi, S, _ = omp_multi_dict_shared(
        np.ones((1, 6)),
        D,
        np.ones((1, 6)),
        D,
        max_iters=10,
        threshold=-1.0,
        n_nonzero_coefs=10,
        return_support=True,
        aggregate="mean_abs",
    )
    assert len(S) == len(set(S)) == 4  # exhausted all atoms exactly once


def test_input_validation():
    Y, D, *_ = _problem()
    with pytest.raises(ValueError):
        omp_multi_dict_shared(Y[:2], D, Y[:2], D)
    with pytest.raises(ValueError):
        omp_multi_dict_shared(Y[:, :10], D, Y[:, :10], D)
    with pytest.raises(ValueError):
        selectscores(np.ones((2, 3)), "nope")


def test_selectscores_values():
    C = np.array([[1.0, 2.0], [3.0, 2.0]])
    np.testing.assert_allclose(selectscores(C, "mean_abs"), [2.0, 2.0])
    np.testing.assert_allclose(selectscores(C, "harmonic_mean"), [1.5, 2.0], rtol=1e-12)
    np.testing.assert_allclose(selectscores(C, "geometric_mean"), [np.sqrt(3), 2.0])
    # harmonic mean punishes an order that does not correlate at all
    assert selectscores(np.array([[1.0], [0.0]]), "harmonic_mean")[0] < 1e-10
