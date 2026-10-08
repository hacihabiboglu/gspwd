"""Steering response of a two-source scene: SPWD vs GSPWD kernel.

Both maps are computed from the same spherical-harmonic signal of order N. The optimised
kernel concentrates the response around the sources with lower sidelobes. Writes ``srf.png``.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _common import parse_out

from gspwd import generalized_srf, gridangs, maxdfcoeff, scene_shd, zonal_multiobjective_coeff
from gspwd.plotting import mark_directions, plot_sphere_map


def main():
    out = parse_out()
    N = 4
    thetas = np.array([np.pi / 2, np.pi / 3])
    phis = np.array([np.pi / 5, -np.pi / 2])
    amps = np.array([1.0, 0.8])

    shd = scene_shd(amps, thetas, phis, N)  # noiseless
    th, ph = gridangs(3072)  # fine nested HEALPix grid (nside 16)
    gspwd_k, _ = zonal_multiobjective_coeff(
        N, np.pi / (N + 0.5), lam_side=N**2, hard_backnull=True, normalize="energy"
    )
    maps = {
        "SPWD": generalized_srf(maxdfcoeff(N), shd, th, ph, N),
        "GSPWD": generalized_srf(gspwd_k, shd, th, ph, N),
    }

    fig = plt.figure(figsize=(12, 3.6))
    for i, (name, srf) in enumerate(maps.items(), start=1):
        ax = fig.add_subplot(1, 2, i, projection="mollweide")
        mag = np.abs(srf)
        plot_sphere_map(mag / mag.max(), th, ph, ax=ax, title=f"{name} (order {N})")
        mark_directions(ax, thetas, phis)
    fig.savefig(out / "srf.png", dpi=150)
    print("wrote", out / "srf.png")


if __name__ == "__main__":
    main()
