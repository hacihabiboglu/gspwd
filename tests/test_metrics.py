import numpy as np
import pytest

from gspwd.metrics import detectionmetrics, ecm, le, lr, support_aware_amplitude_error

TRUTH = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
FAR = np.array([[0.0, 1.0, 0.0]])  # 90 deg from both sources
THETA = 0.1


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_perfect_estimate():
    est = TRUTH[::-1]  # order must not matter (Hungarian matching)
    assert le(est, TRUTH, THETA) == pytest.approx(0, abs=1e-6)
    assert lr(est, TRUTH, THETA) == 1
    assert ecm(est, TRUTH) == 0
    m = detectionmetrics(est, TRUTH, THETA)
    assert (m["precision"], m["recall"], m["F1"], m["FDR"]) == pytest.approx((1, 1, 1, 0))


def test_localisation_error_is_mean_matched_angle():
    off = 0.04
    est = _unit(TRUTH + np.array([[off, 0, 0], [0, off, 0]]))
    expected = np.mean([np.arctan(off), np.arctan(off)])
    assert le(est, TRUTH, THETA) == pytest.approx(expected, rel=1e-6)


def test_over_selection():
    est = np.vstack([TRUTH, FAR])
    m = detectionmetrics(est, TRUTH, THETA)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(1.0)
    assert m["FDR"] == pytest.approx(1 / 3)
    assert m["F1"] == pytest.approx(0.8)
    assert ecm(est, TRUTH) == pytest.approx(1 - 3 / 2)


def test_under_selection():
    est = TRUTH[:1]
    m = detectionmetrics(est, TRUTH, THETA)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(0.5)
    assert m["F1"] == pytest.approx(2 / 3)
    assert lr(est, TRUTH, THETA) == pytest.approx(0.5)


def test_all_wrong():
    m = detectionmetrics(FAR, TRUTH, THETA)
    assert m["precision"] == pytest.approx(0, abs=1e-12)
    assert m["recall"] == pytest.approx(0, abs=1e-12)
    assert m["F1"] == pytest.approx(0, abs=1e-12)
    assert m["FDR"] == pytest.approx(1.0)


def test_dcase_false_negatives_count_cardinality_only():
    # M == N but one estimate is mislocalised: FP = 1, FN = max(0, N - M) = 0, so recall stays 1
    est = np.vstack([TRUTH[0], FAR])
    m = detectionmetrics(est, TRUTH, THETA)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1.0)
    assert m["F1"] == pytest.approx(2 / 3)
    assert m["FDR"] == pytest.approx(0.5)
    # ... while the localisation recall (lr) does expose the missed source
    assert lr(est, TRUTH, THETA) == pytest.approx(0.5)


def test_coincident_directions_do_not_produce_nan():
    # unit vectors whose dot product is 1 + 1e-16 used to make arccos return NaN
    v = _unit([[0.3, 0.4, 0.5]])
    assert np.isfinite(le(v, v * (1 + 2e-16), THETA))
    assert np.isfinite(detectionmetrics(v, v, THETA)["F1"])


# ---- support-aware amplitude error
AMPS = np.array([1.0 + 0j, 0.5j])


def test_amplitude_error_perfect_and_empty():
    assert support_aware_amplitude_error(TRUTH, AMPS, TRUTH, AMPS, THETA) == pytest.approx(
        0, abs=1e-12
    )
    # nothing recovered: all true energy is missed
    assert support_aware_amplitude_error(np.zeros((0, 3)), np.zeros(0), TRUTH, AMPS, THETA) == 1.0


def test_amplitude_error_penalises_spurious_and_missed_energy():
    est = np.vstack([TRUTH, FAR])
    amps = np.array([1.0, 0.5j, 0.3])
    e_true = np.sum(np.abs(AMPS) ** 2)
    assert support_aware_amplitude_error(est, amps, TRUTH, AMPS, THETA) == pytest.approx(
        np.sqrt(0.3**2 / e_true)
    )
    only_first = support_aware_amplitude_error(TRUTH[:1], AMPS[:1], TRUTH, AMPS, THETA)
    assert only_first == pytest.approx(np.sqrt(0.25 / e_true))


def test_amplitude_errors_returns_total_and_matched():
    from gspwd.metrics import amplitude_errors

    est = np.vstack([TRUTH, FAR])
    amps = np.array([1.0, 0.5j, 0.3])
    total, matched = amplitude_errors(est, amps, TRUTH, AMPS, THETA)
    e_true = np.sum(np.abs(AMPS) ** 2)
    assert total == pytest.approx(np.sqrt(0.09 / e_true))
    assert matched == pytest.approx(
        np.sqrt(1e-10)
    )  # matched amplitudes exact: only the floor remains
    assert support_aware_amplitude_error(est, amps, TRUTH, AMPS, THETA) == pytest.approx(total)
    assert amplitude_errors(np.zeros((0, 3)), np.zeros(0), TRUTH, AMPS, THETA) == (1.0, 1e-5)


def test_amplitude_error_matched_difference():
    got = support_aware_amplitude_error(TRUTH, AMPS * 1.1, TRUTH, AMPS, THETA)
    assert got == pytest.approx(0.1, rel=1e-9)  # relative amplitude error of the matched pair
