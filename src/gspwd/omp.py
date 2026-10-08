"""Joint (simultaneous) orthogonal matching pursuit over several dictionaries."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

_EPS = np.finfo(np.float64).eps


def selectscores(C, aggregate):
    """Aggregate per-signal correlations ``C`` (shape ``(N, K)``) into one score per atom.

    ``aggregate`` is one of ``mean_abs``, ``geometric_mean``, ``harmonic_mean``,
    ``mean_sq`` (mean squared modulus) or ``mean_raw`` (signed mean; meaningful for real
    correlations only). The harmonic mean (used for multi-order GSPWD) is large only when
    *all* orders correlate with the atom.
    """
    C = np.asarray(C)
    if aggregate == "mean_abs":
        return np.mean(np.abs(C), axis=0)
    if aggregate == "geometric_mean":
        with np.errstate(divide="ignore"):
            return np.exp(np.mean(np.log(np.abs(C)), axis=0))
    if aggregate == "harmonic_mean":
        return C.shape[0] / np.sum(1.0 / (np.abs(C) + _EPS), axis=0)
    if aggregate == "mean_sq":
        return np.mean(np.abs(C) ** 2, axis=0)
    if aggregate == "mean_raw":
        return np.mean(C, axis=0)
    raise ValueError(f"Unknown aggregate='{aggregate}'")


def omp_multi_dict_shared(
    Ya,
    Danalysis,
    Yb,
    Dsynthesis,
    n_nonzero_coefs=None,
    eps=None,
    max_iters=None,
    aggregate="harmonic_mean",
    return_support=False,
    threshold=0.01,
    snr=None,
):
    """Joint OMP: one shared support, one analysis dictionary per signal (e.g. per SH order).

    Parameters
    ----------
    Ya : (N, M) array-like
        Signals used to *select* atoms (a single signal of shape ``(M,)`` is accepted).
    Danalysis : (N, M, K) array-like
        Analysis dictionaries, ``Danalysis[i]`` belongs to ``Ya[i]`` (a single ``(M, K)``
        dictionary is accepted).
    Yb, Dsynthesis :
        Signals and dictionaries used for the joint least-squares fit that defines the
        stopping residual (usually identical to ``Ya``/``Danalysis``).
    n_nonzero_coefs : int, optional
        Maximum support size. Defaults to ``M`` (the signal dimension).
    eps : float, optional
        Stop when the mean per-signal residual norm (analysis fit) is at most ``eps``.
    max_iters : int, optional
        Hard iteration limit. Defaults to ``K``.
    aggregate : str
        Correlation aggregation across signals, see :func:`selectscores`.
    return_support : bool
        Also return the selected atom indices.
    threshold : float
        Stop when ``(relative joint residual)**2 / N`` is at most ``threshold``.
    snr : float, optional
        Nominal SNR in dB. If finite, the stopping threshold is raised to a noise-aware level,
        evaluated at every iteration ``it`` (0-based)::

            thr = max(threshold, chi2.ppf(0.95, dof) / (2 * M) / (1 + 10**(snr / 10)))
            dof = 2 * D * ((N + 1)**2 - it),   M = D * (N + 1)**2

        with ``D = N`` the number of signals, which for the multi-order estimators is the maximum
        harmonic order. (A single-signal call therefore uses ``N = 1`` and, once ``it`` reaches 4,
        ``dof <= 0``, where the noise term is undefined and ``threshold`` applies unchanged.)
        This is the stopping rule of the published experiments.

    Returns
    -------
    Xi : (|S|,) ndarray
        Joint least-squares coefficients on the final support.
    support : list of int
        Selected atom indices in selection order (only if ``return_support``).
    Ri : float
        Final relative joint residual norm.

    Notes
    -----
    Per-signal correlations are divided by their own maximum before aggregation. When a
    signal is fitted to machine precision, that division amplifies rounding noise, so atoms
    selected after that point are not numerically reproducible.
    """
    Ya = np.asarray(Ya)
    if Ya.ndim != 2:  # single-signal case
        Ya = Ya.reshape(1, -1)
    N, M = Ya.shape

    Danalysis = np.asarray(Danalysis)
    if Danalysis.ndim != 3:  # single-dictionary case
        Danalysis = Danalysis[np.newaxis]
    Yb = np.asarray(Yb)
    if Yb.ndim != 2:
        Yb = Yb.reshape(1, -1)
    Dsynthesis = np.asarray(Dsynthesis)
    if Dsynthesis.ndim != 3:
        Dsynthesis = Dsynthesis[np.newaxis]
    N_d, M_d, K = Danalysis.shape
    if N != N_d:
        raise ValueError("Ya and Danalysis must hold the same number of signals.")
    if M != M_d:
        raise ValueError("Each dictionary must have as many rows as its signal has samples.")

    if max_iters is None:
        max_iters = K
    if n_nonzero_coefs is None:
        n_nonzero_coefs = M_d

    D = N  # number of signals
    thr_base = threshold
    R = Ya.astype(complex)  # per-signal residuals of the analysis fit
    support: list[int] = []
    X = np.zeros((N, K), dtype=complex)
    Xi = np.zeros(0, dtype=complex)
    Ri = np.inf

    for it in range(max_iters):
        if snr is not None and np.isfinite(snr):
            dof = 2 * D * ((N + 1) ** 2 - it)
            if dof > 0:
                thr_noise = (
                    chi2.ppf(0.95, dof) / (2.0 * D * (N + 1) ** 2) / (1.0 + 10.0 ** (snr / 10.0))
                )
                threshold = max(thr_base, thr_noise)
            else:
                threshold = thr_base
        # 1) correlate each residual with its dictionary, normalised per signal
        C = np.empty((N, K), dtype=complex)
        for i in range(N):
            c = Danalysis[i].T @ R[i]
            peak = np.max(np.abs(c))
            C[i] = c / peak if peak > 0 else c
        scores = selectscores(C, aggregate)

        # 2) pick the best not-yet-selected atom
        if support:
            scores[np.array(support, dtype=int)] = -np.inf
        k_star = int(np.argmax(scores))
        if scores[k_star] == -np.inf:
            break  # every atom already selected
        support.append(k_star)

        # 3) refit each signal on the support; update residuals
        for i in range(N):
            D_i_S = Danalysis[i][:, support]
            a_i, *_ = np.linalg.lstsq(D_i_S, Ya[i], rcond=None)
            X[i, :] = 0.0
            X[i, support] = a_i
            R[i] = Ya[i] - D_i_S @ X[i, support]

        # 4) joint synthesis fit and stopping residual
        Dall = np.concatenate([Dsynthesis[i][:, support] for i in range(N)])
        srfall = np.concatenate([Yb[i] for i in range(N)])
        Xi, *_ = np.linalg.lstsq(Dall, srfall, rcond=None)
        Ri = np.linalg.norm(srfall - Dall @ Xi) / np.linalg.norm(srfall)

        if len(support) >= n_nonzero_coefs:
            break
        if eps is not None and np.mean(np.linalg.norm(R, axis=1)) <= eps:
            break
        if Ri**2 / N <= threshold:
            break

    if return_support:
        return Xi, support, Ri
    return Xi, Ri
