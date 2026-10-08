"""Optional matplotlib helpers (``pip install gspwd[plot]``)."""

from __future__ import annotations

import numpy as np

from .kernels import gpulse


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plotting needs matplotlib: pip install gspwd[plot]") from exc
    return plt


def plot_kernels(kernels: dict, ax=None, ntheta: int = 721, normalize: bool = True):
    """Plot zonal kernels ``{label: legendre_coefficients}`` against angle (degrees).

    With ``normalize`` each kernel is scaled to unit peak so shapes are comparable.
    """
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    theta = np.linspace(0, np.pi, ntheta)
    for label, c in kernels.items():
        c = np.asarray(c, dtype=float)
        k = np.array([gpulse(c, t) for t in theta])
        if normalize:
            k = k / k[0]
        ax.plot(np.degrees(theta), k, label=label)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set(xlabel="angle from look direction (deg)", ylabel="kernel", xlim=(0, 180))
    ax.legend()
    return ax


def plot_sphere_map(values, theta, phi, ax=None, title: str | None = None, cmap: str = "viridis"):
    """Mollweide scatter map of ``values`` sampled at ``(theta, phi)`` (colatitude, longitude)."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.2), subplot_kw={"projection": "mollweide"})
    lon = np.mod(np.asarray(phi) + np.pi, 2 * np.pi) - np.pi
    lat = np.pi / 2 - np.asarray(theta)
    sc = ax.scatter(lon, lat, c=values, s=14, cmap=cmap, marker="s", linewidths=0)
    ax.grid(True, lw=0.3)
    if title:
        ax.set_title(title)
    plt.colorbar(sc, ax=ax, shrink=0.7)
    return ax


def mark_directions(ax, theta, phi, **kwargs):
    """Overlay directions on a map made by :func:`plot_sphere_map`."""
    lon = np.mod(np.asarray(phi) + np.pi, 2 * np.pi) - np.pi
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("facecolors", "none")
    kwargs.setdefault("edgecolors", "red")
    ax.scatter(lon, np.pi / 2 - np.asarray(theta), **kwargs)


def plot_metric_grid(df, metric: str, snr_col: str = "snr"):
    """Mean of ``metric`` per method over (number of sources x maximum order), one panel per SNR.

    ``df`` is a ``pandas.DataFrame`` of experiment rows (``pandas.DataFrame(run_experiment())``).
    """
    plt = _plt()
    snrs = sorted(df[snr_col].unique(), key=lambda v: (np.isinf(v), v))
    methods = list(df["type"].unique())
    fig, axes = plt.subplots(
        len(methods),
        len(snrs),
        figsize=(2.6 * len(snrs), 2.4 * len(methods)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for i, m in enumerate(methods):
        for j, snr in enumerate(snrs):
            sub = df[(df["type"] == m) & (df[snr_col] == snr)]
            pivot = sub.pivot_table(index="numsource", columns="maxn", values=metric)
            axes[i, j].imshow(pivot.values, origin="lower", aspect="auto")
            if i == 0:
                axes[i, j].set_title(f"SNR {snr} dB")
            if j == 0:
                axes[i, j].set_ylabel(m)
    fig.suptitle(metric)
    fig.tight_layout()
    return fig
