#!/usr/bin/env python3
"""Backward-compatible single-budget wrapper around run_small_to_large_ladder_v3.py."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-budget small-to-large ablation v3")
    ap.add_argument("--simulations", type=int, default=12)
    ap.add_argument("--max-depth", type=int, default=24)
    known, rest = ap.parse_known_args()
    budget = f"{known.simulations}:{known.max_depth}"
    cmd = [
        sys.executable,
        "run_small_to_large_ladder_v3.py",
        "--budget-ladder",
        budget,
        *rest,
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
