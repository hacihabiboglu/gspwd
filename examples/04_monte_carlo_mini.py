"""A small version of the paper's Monte Carlo study (about a minute).

Compares GSPWD and SPWD, single-order and multi-order, over SNR and number of sources.
The full study uses ``ExperimentConfig()`` defaults (50 repetitions, SNR in {inf,40,30,20,10},
1-10 sources, N in 1..6). Writes ``monte_carlo_f1.png`` and prints a summary table.
"""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from _common import parse_out

from gspwd import ExperimentConfig, run_experiment
from gspwd.plotting import plot_metric_grid


def main():
    out = parse_out()
    cfg = ExperimentConfig(maxn=6, numsources=(2, 6), snr_list=(np.inf, 30), numsim=10)
    df = pd.DataFrame(run_experiment(cfg))
    print(
        df.groupby(["snr", "type"])[["F1", "precision", "recall", "le", "saae", "time"]]
        .mean()
        .round(3)
    )
    plot_metric_grid(df, "F1").savefig(out / "monte_carlo_f1.png", dpi=120)
    print("wrote", out / "monte_carlo_f1.png")


if __name__ == "__main__":
    main()
