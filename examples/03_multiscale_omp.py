"""Estimate directions and amplitudes of three plane waves with the joint multi-order OMP.

Prints the estimated pixel directions next to the truth together with the evaluation metrics.
"""

import numpy as np

from gspwd import ExperimentConfig, gridangs
from gspwd.experiments import build_dictionaries, run_trial
from gspwd.scenes import randomscene_rejection


def main():
    cfg = ExperimentConfig(maxn=6, numsources=(8,), snr_list=(30,), numsim=1)
    dicts = build_dictionaries(cfg)  # kernels and (numpix x numpeak) dictionaries, built once
    rng = np.random.default_rng(7)
    snr, nsource, N = 30, 8, 6

    scene = randomscene_rejection(nsource, AMIN=cfg.amin, ANGMIN=np.pi / (3 * N), rng=rng)
    rows = run_trial(cfg, dicts, snr, nsource, N, rng, scene=scene, ths_phs=gridangs(cfg.numpix))

    amps, thes, phis = scene
    print("true sources (theta, phi in degrees, |amp|):")
    for t, p, a in zip(np.degrees(thes), np.degrees(phis), np.abs(amps.ravel())):
        print(f"  {t:7.2f} {p:7.2f}  {a:.2f}")
    print(f"\nSNR {snr} dB, maximum order N={N}")
    print(f"{'method':16s} {'F1':>5s} {'prec':>5s} {'recall':>6s} {'LE(deg)':>8s}")
    for r in rows:
        print(
            f"{r['type']:16s} {r['F1']:5.2f} {r['precision']:5.2f} {r['recall']:6.2f} "
            f"{np.degrees(r['le']):8.2f}"
        )


if __name__ == "__main__":
    main()
