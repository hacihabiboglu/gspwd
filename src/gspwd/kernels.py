"""Zonal (axisymmetric) kernel design in the Legendre domain.

A kernel is stored as Legendre coefficients ``c`` so that
``k(theta) = sum_n c[n] * P_n(cos(theta))``.
"""

import numpy as np
from scipy.special import eval_legendre


def zonal_multiobjective_coeff(
    N,
    theta0,
    lam_side=1.0,
    mu_back=0.0,
    hard_backnull=False,
    quad_order=None,
    normalize="peak",
    reg=1e-12,
):
    """
    Multi-objective axisymmetric (zonal) kernel design in Legendre domain.

    We design k(theta) = sum_{n=0}^N c_n P_n(cos theta) with competing goals:
      - narrow mainlobe: enforced indirectly by choosing a small theta0 (mainlobe boundary)
      - low sidelobes: minimize energy outside [0, theta0]
      - low backlobe: penalize or null k(pi)

    Objective (convex quadratic):
        minimize   c^T Q c
        subject to 1^T c = 1            (unit peak at theta=0)
                  (optional) s^T c = 0  (hard backlobe null)

    where
        Q = lam_side * S(theta0) + mu_back * (s s^T) + reg * I,
        s_n = (-1)^n,
        S_nm(theta0) = ∫_{theta0}^{pi} P_n(cosθ) P_m(cosθ) sinθ dθ
                     = ∫_{-1}^{cos(theta0)} P_n(x) P_m(x) dx,  x=cosθ

    Parameters
    ----------
    N : int
        Max Legendre degree.
    theta0 : float
        Mainlobe boundary in radians (0 < theta0 <= pi).
        Smaller theta0 typically yields narrower mainlobe but makes sidelobe suppression harder.
    lam_side : float
        Weight for sidelobe energy outside [0, theta0].
    mu_back : float
        Weight for soft backlobe penalty (k(pi))^2. Ignored if hard_backnull=True.
    hard_backnull : bool
        If True, enforce k(pi)=0 exactly via linear constraint s^T c = 0.
    quad_order : int or None
        Gauss-Legendre quadrature order. If None uses max(8*N, 200).
    normalize : {"peak", "energy", None}
        "peak": enforce sum(c)=1 (already enforced; kept for compatibility)
        "energy": additionally scale so ∫_0^pi k(θ)^2 sinθ dθ = 1
        None: return coefficients as solved (peak constraint still holds).
    reg : float
        Small diagonal regularization added to Q for numerical stability.

    Returns
    -------
    c : (N+1,) ndarray
        Legendre coefficients.
    info : dict
        Diagnostics: kpi, sidelobe_energy, total_energy, cond_Q, etc.
    """
    if not (0.0 < theta0 <= np.pi):
        raise ValueError("theta0 must satisfy 0 < theta0 <= pi.")
    if quad_order is None:
        quad_order = max(8 * N, 200)

    # --- Build S(theta0): integral over x in [-1, cos(theta0)] of P_n(x)P_m(x) dx
    a = -1.0
    b = float(np.cos(theta0))

    # Gauss-Legendre nodes/weights on [-1,1]
    xg, wg = np.polynomial.legendre.leggauss(quad_order)
    # Map to [a,b]
    x = 0.5 * (b - a) * xg + 0.5 * (b + a)
    w = 0.5 * (b - a) * wg

    # Vandermonde V_{i,n} = P_n(x_i), shape (quad_order, N+1)
    V = np.polynomial.legendre.legvander(x, N)

    # S = V^T diag(w) V
    S = (V.T * w) @ V  # (N+1, N+1), PSD

    # Backlobe selector s: k(pi) = sum c_n P_n(-1) = sum c_n (-1)^n
    n = np.arange(N + 1)
    s = ((-1.0) ** n).astype(float)  # (N+1,)

    # Build Q
    Q = lam_side * S
    if not hard_backnull and mu_back != 0.0:
        Q = Q + float(mu_back) * np.outer(s, s)
    Q = Q + float(reg) * np.eye(N + 1)

    # Peak constraint: 1^T c = 1 (since P_n(1)=1)
    ones = np.ones(N + 1, dtype=float)

    # Solve constrained quadratic minimization in closed form:
    #   min c^T Q c  s.t. C c = d
    # gives c = Q^{-1} C^T (C Q^{-1} C^T)^{-1} d
    if hard_backnull:
        C = np.vstack([ones, s])  # shape (2, N+1)
        d = np.array([1.0, 0.0], float)  # shape (2,)
    else:
        C = ones[None, :]  # shape (1, N+1)
        d = np.array([1.0], float)  # shape (1,)

    # Compute Q^{-1} C^T via solves (avoid forming Q^{-1})
    # Solve Q X = C^T
    X = np.linalg.solve(Q, C.T)  # (N+1, n_constraints)
    G = C @ X  # (n_constraints, n_constraints)
    y = np.linalg.solve(G, d)  # (n_constraints,)
    c = X @ y  # (N+1,)

    # Optional sign convention: make k(0)=1 already; keep "front" positive by ensuring sum(c)=1
    # (This is already enforced; but numerical noise might create tiny deviation)
    if normalize == "peak":
        c = c / np.sum(c)

    # Diagnostics: total energy uses Bdiag = 2/(2n+1)
    Bdiag = 2.0 / (2.0 * n + 1.0)
    total_energy = float(np.sum((c**2) * Bdiag))
    sidelobe_energy = float(c @ (S @ c))
    k0 = float(np.sum(c))
    kpi = float(s @ c)
    if normalize == "energy":
        # Scale to make total energy = 1 (does NOT preserve k(0)=1; re-impose peak after if needed)
        if total_energy <= 0:
            raise RuntimeError("Nonpositive total energy; cannot normalize.")
        c = c / np.sqrt(total_energy)
        # If you truly need BOTH energy=1 and peak=1, that's two constraints (doable),
        # but typically you want peak=1 for correlation kernels.
        total_energy = 1.0
        k0 = float(np.sum(c))
        kpi = float(s @ c)
        sidelobe_energy = float(c @ (S @ c))

    elif normalize is None:
        pass
    else:
        if normalize not in ("peak", "energy", None):
            raise ValueError("normalize must be 'peak', 'energy', or None.")

    # Condition number (rough) for transparency
    try:
        cond_Q = float(np.linalg.cond(Q))
    except Exception:
        cond_Q = np.nan

    info = {
        "k0": k0,
        "kpi": kpi,
        "sidelobe_energy": sidelobe_energy,
        "total_energy": total_energy,
        "cond_Q": cond_Q,
        "theta0": float(theta0),
        "lam_side": float(lam_side),
        "mu_back": float(mu_back),
        "hard_backnull": bool(hard_backnull),
        "quad_order": int(quad_order),
        "reg": float(reg),
    }
    return c, info


def maxdfcoeff(n_order):
    """Maximum-directivity (Dirichlet) kernel coefficients ``(2n+1)/(4*pi)``."""
    return (2 * np.arange(n_order + 1) + 1) / (4 * np.pi)


def gpulse(weights, x):
    """Evaluate the zonal kernel with Legendre ``weights`` at angle(s) ``x`` (radians)."""
    n = np.arange(0, len(weights))
    return weights @ eval_legendre(n, np.cos(x))
