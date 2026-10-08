"""Kernel design: maximum-directivity (Dirichlet) vs optimised zonal kernels.

The optimised kernel trades a little mainlobe width for much lower sidelobes and a null at the
back (theta = pi). Writes ``kernels.png``.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _common import parse_out

from gspwd import maxdfcoeff, zonal_multiobjective_coeff
from gspwd.plotting import plot_kernels


def main():
    out = parse_out()
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
    for ax, N in zip(axes, (2, 4, 6)):
        theta0 = np.pi / (N + 0.5)  # mainlobe half-width used in the experiments
        gspwd, info = zonal_multiobjective_coeff(
            N, theta0, lam_side=N**2, mu_back=1.0, hard_backnull=True, normalize="peak"
        )
        plot_kernels(
            {"SPWD (Dirichlet)": maxdfcoeff(N), "GSPWD (optimised)": gspwd},
            ax=ax,
        )
        ax.set_title(f"order N={N}: back-lobe k(pi)={info['kpi']:.1e}")
    fig.tight_layout()
    fig.savefig(out / "kernels.png", dpi=150)
    print("wrote", out / "kernels.png")


if __name__ == "__main__":
    main()
