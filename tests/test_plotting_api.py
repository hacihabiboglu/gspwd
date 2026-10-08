import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

import gspwd  # noqa: E402
from gspwd import plotting  # noqa: E402


def test_public_api_is_importable():
    for name in gspwd.__all__:
        assert hasattr(gspwd, name), name


def test_plot_helpers_smoke():
    ax = plotting.plot_kernels(
        {
            "SPWD": gspwd.maxdfcoeff(3),
            "GSPWD": gspwd.zonal_multiobjective_coeff(
                3, np.pi / 3.5, lam_side=9, hard_backnull=True
            )[0],
        }
    )
    assert len(ax.lines) >= 2
    th, ph = gspwd.gridangs(48)
    ax2 = plotting.plot_sphere_map(np.cos(th), th, ph, title="x")
    plotting.mark_directions(ax2, [1.0], [2.0])


def test_experiment_smoke_and_metric_grid():
    import pandas as pd

    cfg = gspwd.ExperimentConfig(
        maxn=2, numsources=(1, 2), snr_list=(np.inf, 20), numsim=1, numpeak=768
    )
    rows = gspwd.run_experiment(cfg)
    assert len(rows) == 2 * 2 * 2 * 1 * 4  # snr x sources x N x reps x methods
    df = pd.DataFrame(rows)
    assert {"type", "le", "lr", "F1", "saae", "aerror", "time", "snr", "maxn", "numsource"} <= set(
        df.columns
    )
    fig = plotting.plot_metric_grid(df, "F1")
    assert fig is not None
