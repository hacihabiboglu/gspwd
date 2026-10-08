"""Compare the package with an archived long-format result CSV of the original study.

    python scripts/compare_to_archive.py path/to/rm_longlist.r5.csv [--numsim 20]

Noiseless scenes are drawn from the same seeded stream as the original script, so the
single-order estimators must agree row by row. The multi-order estimators agree up to the
documented numerical chaos of the joint OMP, so they are compared by their means.
"""

import argparse

import numpy as np
import pandas as pd

from gspwd import ExperimentConfig, run_experiment

KEY = ["type", "snr", "numsource", "maxn", "sceneid"]
METRICS = ["le", "lr", "ecm", "precision", "recall", "F1", "FDR", "saae", "aerror"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--numsim", type=int, default=20)
    args = ap.parse_args()

    ref = pd.read_csv(args.csv)
    ref["snr"] = ref["snr"].astype(float)
    cfg = ExperimentConfig(
        legacy_rng=True, snr_list=(np.inf,), numsources=(2, 4, 6), numsim=args.numsim
    )
    merged = pd.DataFrame(run_experiment(cfg)).merge(ref, on=KEY, suffixes=("", "_ref"))
    print(f"{len(merged)} noiseless rows matched against the archive")

    single = merged[merged["type"].str.contains("Single")]
    print("\nsingle-order (expected identical): fraction of rows equal to 1e-6")
    for c in METRICS:
        print(f"  {c:10s} {np.mean(np.abs(single[c] - single[c + '_ref']) < 1e-6):.1%}")

    multi = merged[merged["type"].str.contains("Multiple")]
    print("\nmulti-order (expected equal in mean): mine vs archive, mean difference / s.e.")
    for c in METRICS:
        d = multi[c] - multi[c + "_ref"]
        print(
            f"  {c:10s} {multi[c].mean():9.4f} {multi[c + '_ref'].mean():9.4f}  "
            f"{d.mean() / (d.std() / np.sqrt(len(d))):+.2f} s.e."
        )


if __name__ == "__main__":
    main()
