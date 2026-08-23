#!/usr/bin/env python3
"""Run all three paper experiments sequentially."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("mnist_data"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    scripts = [
        ("exp1_trained_models.py", "exp1"),
        ("exp2_estimator_accuracy.py", "exp2"),
        ("exp3_fisher_alignment.py", "exp3"),
    ]
    for script, subdir in scripts:
        cmd = [
            sys.executable,
            str(root / script),
            "--data-dir", str(args.data_dir),
            "--outdir", str(args.results_dir / subdir),
        ]
        if args.quick:
            cmd.append("--quick")
        print("\n>>>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
