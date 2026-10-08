"""Shared helpers for the examples."""

import argparse
from pathlib import Path


def parse_out() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="out", help="directory for generated figures")
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    return out
