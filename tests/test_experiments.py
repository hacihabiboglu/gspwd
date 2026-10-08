"""The library experiment driver reproduces the golden data of the original script."""

from pathlib import Path

import numpy as np
import pytest

from gspwd.experiments import (
    METHODS,
    ExperimentConfig,
    build_dictionaries,
    cell_rng,
    run_trial,
)

GOLDEN = np.load(Path(__file__).parent / "golden" / "golden_v1.npz")
CELLS = [tuple(c) for c in GOLDEN["meta_cells"]]
REPS = int(GOLDEN["meta_reps"])
KEYS = ["le", "lr", "ecm"] + [str(k) for k in GOLDEN["meta_detection_keys"]]
SHORT = {"GSPWD Multiple": "gm", "SPWD Multiple": "sm", "GSPWD Single": "gs", "SPWD Single": "ss"}


@pytest.fixture(scope="module")
def setup():
    cfg = ExperimentConfig(legacy_rng=True)
    return cfg, build_dictionaries(cfg)


@pytest.mark.parametrize("ci", range(len(CELLS)))
def test_run_trial_matches_golden(ci, setup):
    cfg, dicts = setup
    snr, nsource, n = CELLS[ci]
    nsource, n = int(nsource), int(n)
    rng = cell_rng(cfg, snr, nsource, n)
    for rep in range(REPS):
        rows = run_trial(
            cfg,
            dicts,
            snr,
            nsource,
            n,
            rng,
            rep,
            noise_rng=np.random.default_rng([100 + ci, rep]),
        )
        assert [r["type"] for r in rows] == list(METHODS)
        for row in rows:
            tag = f"c{ci}_r{rep}_{SHORT[row['type']]}"
            if int(GOLDEN[f"{tag}_stable"]) < len(GOLDEN[f"{tag}_S"]):
                continue  # tail of the support is rounding-noise driven
            gold = dict(zip(KEYS + ["saae_lin", "aerror_lin"], GOLDEN[f"{tag}_metrics"]))
            for k in KEYS:
                assert row[k] == pytest.approx(gold[k], abs=1e-9, rel=1e-7), (tag, k)
            assert row["saae"] == pytest.approx(20 * np.log10(gold["saae_lin"]), abs=1e-7)
            assert row["aerror"] == pytest.approx(20 * np.log10(gold["aerror_lin"]), abs=1e-7)
            assert row["time"] >= 0


def test_generator_rng_is_reproducible_and_independent():
    cfg = ExperimentConfig(maxn=2, numsources=(2,), snr_list=(20,), numsim=1)
    a = cell_rng(cfg, 20, 2, 2).random(3)
    b = cell_rng(cfg, 20, 2, 2).random(3)
    c = cell_rng(cfg, 20, 3, 2).random(3)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
