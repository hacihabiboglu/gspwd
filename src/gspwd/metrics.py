"""Evaluation metrics for DOA estimation (Hungarian matching, angular threshold ``THETA``)."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _match(vecres, vecscene, THETA):
    """Angular distances ``D``, within-threshold mask ``T`` and Hungarian assignment ``A``."""
    D = np.arccos(np.clip(vecres @ vecscene.T, -1.0, 1.0))
    T = np.zeros_like(D)
    T[np.where(D <= THETA)] = 1.0
    r, c = linear_sum_assignment(D)
    A = np.zeros_like(D)
    A[r, c] = 1.0
    return D, T, A


def le(vecres, vecscene, THETA=np.pi):
    """Localization error (radians): mean angular error of matched pairs within ``THETA``."""
    D, T, A = _match(vecres, vecscene, THETA)
    return np.sum(np.abs((T * A * D)).flatten()) / np.sum(np.abs(T * A).flatten() + 1e-16)


def lr(vecres, vecscene, THETA=np.pi):
    """Localization recall: fraction of true sources matched within ``THETA``."""
    _, T, A = _match(vecres, vecscene, THETA)
    return np.sum(np.abs((T * A)).flatten()) / vecscene.shape[0]


def ecm(vecres, vecscene, THETA=np.pi):
    """Estimated-cardinality mismatch ``1 - M/N`` (negative if over-selecting; ``THETA`` unused)."""
    return 1 - float(vecres.shape[0]) / float(vecscene.shape[0])


def detectionmetrics(vecres, vecscene, THETA=np.pi):
    """Detection precision, recall, F1 and false-discovery rate (DCASE-2019-style counting).

    With ``M`` estimates, ``N`` true sources and ``TP`` estimates assigned (Hungarian
    matching) to a true source within ``THETA`` radians:

    * ``FP = M - TP``: every estimate that is not a true positive (spurious *or* mislocalised);
    * ``FN = max(0, N - M)``: missed sources, counted by cardinality only (as in the DCASE 2019
      sound-event-localisation-and-detection F-score), so a mislocalised estimate is penalised
      once, as a false positive, and not again as a false negative.

    ``precision = TP / (TP + FP)``, ``recall = TP / (TP + FN)``, ``F1`` their harmonic mean and
    ``FDR = FP / (TP + FP)``.
    """
    M = vecres.shape[0]
    N = vecscene.shape[0]
    _, T, A = _match(vecres, vecscene, THETA)
    TP = np.sum(np.abs(T * A).flatten())
    FP = M - TP
    FN = max(0, N - M)
    precision = TP / (TP + FP + 1e-16)
    recall = TP / (TP + FN + 1e-16)
    F1 = 2 * precision * recall / (precision + recall + 1e-16)
    FDR = FP / (TP + FP + 1e-16)
    return {"precision": precision, "recall": recall, "F1": F1, "FDR": FDR}


def amplitude_errors(vecrec, amprec, vecgt, ampgt, THETA=np.pi):
    """
    Support-aware total amplitude error for plane-wave decomposition.

    Charges three error sources against the true plane-wave model, all
    normalized by total true energy:
      (1) matched amplitude error  - survivors within THETA, complex difference
      (2) spurious recovered energy - unmatched/over-threshold recovered atoms,
                                      charged against zero (penalizes over-selection)
      (3) missed true energy        - true sources with no recovered match within
                                      THETA, charged against zero (penalizes under-selection)

    Reduces to amplitude accuracy when cardinality is correct.
    Monotone in quality; over-selection raises it, unlike matched-only AE.

    Parameters
    ----------
    vecrec : (Q, 3)  unit DOA vectors of recovered plane waves
    amprec : (Q,)    complex recovered amplitudes
    vecgt  : (S, 3)  unit DOA vectors of true plane waves
    ampgt  : (S,)    complex true amplitudes
    THETA  : float   angular threshold (rad); pairs beyond it are not "matched"

    Returns
    -------
    (saae, matched) : tuple of float
        ``saae = sqrt((E_matched + E_spurious + E_missed) / E_true)`` and the matched-only
        error ``matched = sqrt(E_matched / E_true + 1e-10)`` (the ``1e-10`` keeps its dB value
        finite when the matched amplitudes are exact).
    """
    amprec = amprec.ravel()
    ampgt = ampgt.ravel()
    Q, S = amprec.shape[0], ampgt.shape[0]

    E_true = np.sum(np.abs(ampgt) ** 2)
    if E_true <= 0:
        return np.nan, np.nan

    # Degenerate: nothing recovered -> all true energy is missed.
    if Q == 0:
        return 1.0, np.sqrt(1e-10)  # all true energy missed

    # Direction-only cosine-distance assignment.
    D = np.arccos(np.clip(vecrec @ vecgt.T, -1.0, 1.0))  # (Q, S)
    r, c = linear_sum_assignment(D)  # len = min(Q, S)

    # Split assigned pairs into survivors (within THETA) and over-threshold.
    within = D[r, c] <= THETA
    rk, ck = r[within], c[within]  # matched survivors

    # (1) Matched amplitude error: complex difference over survivors.
    E_matched = np.sum(np.abs(amprec[rk] - ampgt[ck]) ** 2)

    # (2) Spurious recovered energy: every recovered atom NOT a survivor,
    #     charged against zero. This includes:
    #       - atoms matched but beyond THETA
    #       - atoms left unassigned by LSA (when Q > S)
    matched_rec_mask = np.zeros(Q, dtype=bool)
    matched_rec_mask[rk] = True
    E_spurious = np.sum(np.abs(amprec[~matched_rec_mask]) ** 2)

    # (3) Missed true energy: every true source with no survivor match,
    #     charged against zero. Includes true sources left unassigned
    #     (when S > Q) and those whose match fell beyond THETA.
    matched_gt_mask = np.zeros(S, dtype=bool)
    matched_gt_mask[ck] = True
    E_missed = np.sum(np.abs(ampgt[~matched_gt_mask]) ** 2)
    # E_missed = 0.

    return (
        np.sqrt((E_matched + E_spurious + E_missed) / E_true),
        np.sqrt(E_matched / E_true + 1e-10),
    )


def support_aware_amplitude_error(vecrec, amprec, vecgt, ampgt, THETA=np.pi):
    """The total (support-aware) amplitude error only; see :func:`amplitude_errors`."""
    return amplitude_errors(vecrec, amprec, vecgt, ampgt, THETA)[0]
