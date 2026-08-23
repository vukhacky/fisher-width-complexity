#!/usr/bin/env python3
"""Experiment 1: Fisher width in trained models.

Paper role
----------
This script supports Section 7.2 of the manuscript. It compares Fisher width
for three simple model classes on MNIST:

  A. Binary logistic regression (digits 0 vs 1), using the FULL conditional
     model Fisher averaged over the observed training covariates,

         G_hat = (1/n) sum_i p_i(1-p_i) x_i x_i^T.

  B. Ten-class softmax regression, using the DIAGONAL of the conditional
     model Fisher averaged over the observed training covariates,

         gamma_{k,j} = (1/n) sum_i p_{ik}(1-p_{ik}) x_{ij}^2.

  C. Ridge regression, using the analytic Gaussian-regression Fisher matrix

         G = sigma^{-2} Sigma_X.

Important terminology
---------------------
The logistic and softmax matrices used here are conditional/model Fisher
matrices averaged over the observed covariates. They are NOT observed-label
score outer-product matrices.

The parameterizations have no intercept. This matches the dimensions reported
in the paper: p=784 for binary logistic/ridge and p=7840 for 10-class softmax.

Outputs
-------
  <outdir>/exp1_figure.pdf
  <outdir>/exp1_figure.png
  <outdir>/exp1_results.npz
  <outdir>/exp1_summary.txt
  <outdir>/exp1_table.csv

Run
---
  python exp1_trained_models.py
  python exp1_trained_models.py --quick   # smoke test only
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import struct
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "t10k-labels-idx1-ubyte.gz",
}

MASTER_SEED = 42
SIGMA2_RIDGE = 1.0
DAMP = 1e-6
LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0]
LAMBDA_LABELS = [r"$0$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$1$", r"$5$"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("mnist_data"))
    p.add_argument("--outdir", type=Path, default=Path("results/exp1"))
    p.add_argument("--quick", action="store_true", help="Small smoke-test run; do not use its numbers in the paper.")
    return p.parse_args()


def ensure_mnist(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in MNIST_FILES:
        path = data_dir / filename
        if path.exists():
            continue
        url = f"{MNIST_BASE_URL}/{filename}"
        print(f"Downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        n = struct.unpack(">I", f.read(4))[0]
        if magic == 2051:
            rows, cols = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), np.uint8).reshape(n, rows * cols).astype(np.float64)
        if magic == 2049:
            return np.frombuffer(f.read(), np.uint8).astype(int)
    raise ValueError(f"Unknown IDX file: {path}")


def load_mnist(data_dir: Path):
    ensure_mnist(data_dir)
    X_train = read_idx(data_dir / "train-images-idx3-ubyte.gz")
    y_train = read_idx(data_dir / "train-labels-idx1-ubyte.gz")
    X_test = read_idx(data_dir / "t10k-images-idx3-ubyte.gz")
    y_test = read_idx(data_dir / "t10k-labels-idx1-ubyte.gz")
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    return X_all, y_all


def euclidean_ball_width(p: int) -> float:
    return float(np.exp(0.5 * np.log(2.0) + gammaln((p + 1) / 2.0) - gammaln(p / 2.0)))


def logistic_C(lam: float, n: int) -> float:
    # sklearn's L2 objective is equivalent, after division by n, to
    # mean NLL + (1/(2 C n)) ||w||^2. Thus C = 1/(lam*n).
    return 1.0 / (lam * n) if lam > 0 else 1e12


def mc_width_from_spectrum(eigvals, B: int, rng, batch: int = 256) -> float:
    eigvals = np.clip(np.asarray(eigvals, dtype=np.float64), 0.0, None)
    if eigvals.size == 0 or eigvals.max(initial=0.0) == 0.0:
        return 0.0
    total = 0.0
    done = 0
    while done < B:
        m = min(batch, B - done)
        z = rng.standard_normal((m, eigvals.size))
        total += np.sqrt((z * z) @ eigvals).sum()
        done += m
    return float(total / B)


def spectral_stats(eigvals, wE: float, damp: float = DAMP):
    eigvals = np.clip(np.asarray(eigvals), 0.0, None)
    lmax = float(eigvals.max(initial=0.0))
    lmin = float(eigvals.min(initial=0.0))
    lower = np.sqrt(lmin) * wE
    upper = np.sqrt(lmax) * wE
    kappa = lmax / max(lmin, damp)
    trace_upper = np.sqrt(eigvals.sum())
    return lower, upper, kappa, trace_upper, lmax, lmin


def conditional_fisher_binary(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    z = X @ theta
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    q = p * (1.0 - p)
    G = (X * q[:, None]).T @ X / len(X)
    return 0.5 * (G + G.T)


def diagonal_conditional_fisher_softmax(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    W = P * (1.0 - P)
    # shape (K,d), then flatten to Kd parameters.
    dG = (W[:, :, None] * (X[:, None, :] ** 2)).mean(axis=0)
    return dG.ravel()


def run(args):
    args.outdir.mkdir(parents=True, exist_ok=True)
    X_all, y_all = load_mnist(args.data_dir)
    mask = (y_all == 0) | (y_all == 1)
    X_bin, y_bin = X_all[mask], y_all[mask]
    d = X_all.shape[1]
    K = 10

    if args.quick:
        n_seeds, n_train, n_test = 2, 1500, 500
        B_A, B_B, B_C = 600, 700, 600
        B_reference = 4000
        lambdas = [0.0, 1e-2, 1.0]
        lambda_labels = [r"$0$", r"$10^{-2}$", r"$1$"]
        print("QUICK MODE: outputs are only for a smoke test, not for the manuscript.")
    else:
        n_seeds, n_train, n_test = 10, 10_000, 2_000
        B_A, B_B, B_C = 5_000, 10_000, 5_000
        B_reference = 50_000
        lambdas = LAMBDAS
        lambda_labels = LAMBDA_LABELS

    pA, pB, pC = d, d * K, d
    wE = {"A": euclidean_ball_width(pA), "B": euclidean_ball_width(pB), "C": euclidean_ball_width(pC)}
    nL = len(lambdas)

    def blank():
        return np.zeros((nL, n_seeds), dtype=float)

    R = {
        key: {name: blank() for name in ["wF", "trace_upper", "lb", "ub", "kappa", "metric"]}
        for key in ["A", "B", "C"]
    }

    scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Model A: binary logistic, full conditional model Fisher.
    # ------------------------------------------------------------------
    print("\nModel A: binary logistic, full conditional model Fisher")
    for li, lam in enumerate(lambdas):
        for si in range(n_seeds):
            rng = np.random.default_rng(MASTER_SEED + 100 * si)
            idx = rng.permutation(len(X_bin))
            Xtr = scaler.fit_transform(X_bin[idx[:n_train]])
            Xte = scaler.transform(X_bin[idx[n_train:n_train + n_test]])
            ytr = y_bin[idx[:n_train]]
            yte = y_bin[idx[n_train:n_train + n_test]]

            clf = LogisticRegression(
                C=logistic_C(lam, n_train),
                max_iter=2000,
                solver="lbfgs",
                fit_intercept=False,
                random_state=0,
                tol=1e-6,
            )
            clf.fit(Xtr, ytr)
            theta = clf.coef_.ravel()
            G = conditional_fisher_binary(Xtr, theta)
            eigvals = np.linalg.eigvalsh(G)

            R["A"]["wF"][li, si] = mc_width_from_spectrum(eigvals, B_A, rng)
            lb, ub, kap, trub, _, _ = spectral_stats(eigvals, wE["A"])
            R["A"]["lb"][li, si] = lb
            R["A"]["ub"][li, si] = ub
            R["A"]["kappa"][li, si] = kap
            R["A"]["trace_upper"][li, si] = trub
            R["A"]["metric"][li, si] = clf.score(Xte, yte)

    # ------------------------------------------------------------------
    # Model B: softmax, diagonal conditional model Fisher.
    # ------------------------------------------------------------------
    print("Model B: 10-class softmax, diagonal conditional model Fisher")
    for li, lam in enumerate(lambdas):
        for si in range(n_seeds):
            rng = np.random.default_rng(MASTER_SEED + 100 * si + 1)
            idx = rng.permutation(len(X_all))
            Xtr = scaler.fit_transform(X_all[idx[:n_train]])
            Xte = scaler.transform(X_all[idx[n_train:n_train + n_test]])
            ytr = y_all[idx[:n_train]]
            yte = y_all[idx[n_train:n_train + n_test]]

            clf = LogisticRegression(
                C=logistic_C(lam, n_train),
                max_iter=2000,
                solver="lbfgs",
                fit_intercept=False,
                random_state=0,
                tol=1e-5,
            )
            clf.fit(Xtr, ytr)
            P = clf.predict_proba(Xtr)
            dG = diagonal_conditional_fisher_softmax(Xtr, P)

            R["B"]["wF"][li, si] = mc_width_from_spectrum(dG, B_B, rng, batch=128)
            lb, ub, kap, trub, _, _ = spectral_stats(dG, wE["B"])
            R["B"]["lb"][li, si] = lb
            R["B"]["ub"][li, si] = ub
            R["B"]["kappa"][li, si] = kap
            R["B"]["trace_upper"][li, si] = trub
            R["B"]["metric"][li, si] = clf.score(Xte, yte)

    # ------------------------------------------------------------------
    # Model C: Gaussian/ridge Fisher. G does not depend on theta/lambda.
    # We compute the Fisher geometry once per seed and reuse it over lambda.
    # ------------------------------------------------------------------
    print("Model C: ridge/Gaussian regression, analytic Fisher matrix")
    C_seed_stats = []
    for si in range(n_seeds):
        rng = np.random.default_rng(MASTER_SEED + 100 * si + 2)
        idx = rng.permutation(len(X_bin))
        Xtr = scaler.fit_transform(X_bin[idx[:n_train]])
        Sigma = Xtr.T @ Xtr / n_train
        G = Sigma / SIGMA2_RIDGE
        eigvals = np.linalg.eigvalsh(G)
        width = mc_width_from_spectrum(eigvals, B_C, rng)
        stats = spectral_stats(eigvals, wE["C"])
        C_seed_stats.append((width, *stats))

    for li in range(nL):
        for si, vals in enumerate(C_seed_stats):
            width, lb, ub, kap, trub, _, _ = vals
            R["C"]["wF"][li, si] = width
            R["C"]["lb"][li, si] = lb
            R["C"]["ub"][li, si] = ub
            R["C"]["kappa"][li, si] = kap
            R["C"]["trace_upper"][li, si] = trub
            R["C"]["metric"][li, si] = np.nan

    # High-accuracy spectral Monte Carlo reference for one fixed Model-C matrix.
    rng0 = np.random.default_rng(MASTER_SEED + 2)
    idx0 = rng0.permutation(len(X_bin))
    Xtr0 = scaler.fit_transform(X_bin[idx0[:n_train]])
    G0 = (Xtr0.T @ Xtr0 / n_train) / SIGMA2_RIDGE
    eig0 = np.linalg.eigvalsh(G0)
    wC_reference = mc_width_from_spectrum(eig0, B_reference, np.random.default_rng(999), batch=256)

    # MC convergence for the same fixed spectrum.
    B_values = [100, 200, 500, 1000, 2000, 5000, 10000] if not args.quick else [100, 300, 700]
    mc_errors = np.zeros((len(B_values), n_seeds))
    for bi, B in enumerate(B_values):
        for si in range(n_seeds):
            est = mc_width_from_spectrum(eig0, B, np.random.default_rng(MASTER_SEED + 1000 + 17 * si + B))
            mc_errors[bi, si] = abs(est - wC_reference) / wC_reference * 100.0

    # Save numerical results.
    np.savez(
        args.outdir / "exp1_results.npz",
        lambdas=np.asarray(lambdas),
        A_wF=R["A"]["wF"], B_wF=R["B"]["wF"], C_wF=R["C"]["wF"],
        A_trace_upper=R["A"]["trace_upper"], B_trace_upper=R["B"]["trace_upper"], C_trace_upper=R["C"]["trace_upper"],
        A_kappa=R["A"]["kappa"], B_kappa=R["B"]["kappa"], C_kappa=R["C"]["kappa"],
        A_lb=R["A"]["lb"], B_lb=R["B"]["lb"], C_lb=R["C"]["lb"],
        A_ub=R["A"]["ub"], B_ub=R["B"]["ub"], C_ub=R["C"]["ub"],
        wE_A=wE["A"], wE_B=wE["B"], wE_C=wE["C"],
        wC_reference=wC_reference, B_values=np.asarray(B_values), mc_errors=mc_errors,
        n_train=n_train, n_test=n_test, n_seeds=n_seeds,
    )

    rows = []
    for key, label in [("A", "Binary logistic"), ("B", "Softmax 10-class"), ("C", "Ridge/Gaussian")]:
        for li, lam in enumerate(lambdas):
            rows.append({
                "model": key,
                "description": label,
                "lambda": lam,
                "wF_mean": R[key]["wF"][li].mean(),
                "wF_std": R[key]["wF"][li].std(),
                "wF_over_wE": R[key]["wF"][li].mean() / wE[key],
                "trace_upper_mean": R[key]["trace_upper"][li].mean(),
                "trace_ratio": R[key]["trace_upper"][li].mean() / R[key]["wF"][li].mean(),
                "kappa_mean": R[key]["kappa"][li].mean(),
            })
    with open(args.outdir / "exp1_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary designed to be pasted back into the manuscript revision workflow.
    summary = []
    summary.append("Experiment 1: Fisher Width in Trained Models")
    summary.append(f"n_train={n_train}, n_test={n_test}, seeds={n_seeds}")
    summary.append("Model A = full conditional model Fisher; Model B = diagonal conditional model Fisher.")
    summary.append("Model C Fisher is lambda-invariant by construction.")
    summary.append("")
    for key in ["A", "B", "C"]:
        summary.append(f"Model {key}:")
        for li, lam in enumerate(lambdas):
            summary.append(
                f"  lambda={lam:g}: wF={R[key]['wF'][li].mean():.6f} +- {R[key]['wF'][li].std():.6f}; "
                f"wF/wE={R[key]['wF'][li].mean()/wE[key]:.6f}; "
                f"traceUB={R[key]['trace_upper'][li].mean():.6f}; "
                f"traceUB/wF={R[key]['trace_upper'][li].mean()/R[key]['wF'][li].mean():.6f}"
            )
        summary.append("")
    summary.append(f"Model-C fixed-spectrum high-accuracy MC reference: {wC_reference:.6f} (B={B_reference})")
    (args.outdir / "exp1_summary.txt").write_text("\n".join(summary))

    # Figure: 3 rows x 3 columns.
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.50, wspace=0.40)
    x = np.arange(nL)

    plot_specs = [
        ("A", r"Model A: Binary Logistic ($p=784$, full conditional Fisher)", wE["A"]),
        ("B", r"Model B: Softmax 10-class ($p=7840$, diagonal conditional Fisher)", wE["B"]),
        ("C", r"Model C: Ridge/Gaussian ($p=784$, analytic $G$)", wE["C"]),
    ]

    for row, (key, title, wE_key) in enumerate(plot_specs):
        wF_m = R[key]["wF"].mean(axis=1)
        wF_s = R[key]["wF"].std(axis=1)
        tr_m = R[key]["trace_upper"].mean(axis=1)
        tr_s = R[key]["trace_upper"].std(axis=1)
        lb_m = R[key]["lb"].mean(axis=1)
        ub_m = R[key]["ub"].mean(axis=1)
        kp_m = R[key]["kappa"].mean(axis=1)
        kp_s = R[key]["kappa"].std(axis=1)

        ax = axes[row, 0]
        ax.fill_between(x, lb_m, ub_m, alpha=0.15, label="Spectral sandwich")
        ax.errorbar(x, wF_m, yerr=wF_s, fmt="o-", capsize=3, lw=1.6, label=r"$\widehat w_G$ (MC)")
        ax.errorbar(x, tr_m, yerr=tr_s, fmt="s--", capsize=3, lw=1.4, label=r"Trace upper scale $\sqrt{\mathrm{Tr}(G)}$")
        ax.axhline(wE_key, ls=":", lw=1.1, label=r"$w(B_2^p)$")
        if key == "C":
            ax.axhline(wC_reference, ls="--", lw=1.2, label="Fixed-spectrum MC reference")
        ax.set_xticks(x)
        ax.set_xticklabels(lambda_labels, fontsize=7)
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\widehat w_G(B_2^p)$")
        ax.set_title(title + "\n(a) Absolute Fisher width", fontsize=8)
        ax.legend(fontsize=6)
        ax.set_ylim(bottom=0)

        ax = axes[row, 1]
        ratio = wF_m / wE_key
        ax.errorbar(x, ratio, yerr=wF_s / wE_key, fmt="o-", capsize=3, lw=1.6)
        ax.axhline(1.0, ls=":", lw=1.1, label="Euclidean = 1")
        ax.set_xticks(x)
        ax.set_xticklabels(lambda_labels, fontsize=7)
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\widehat w_G / w(B_2^p)$")
        ax.set_title("(b) Normalized Fisher width", fontsize=8)
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.15)
        if key == "C":
            ax.text(0.5, 0.40, r"$G=\sigma^{-2}\Sigma_X$" + "\n" + r"(no $\lambda$ dependence)", transform=ax.transAxes, ha="center", fontsize=8)

        ax = axes[row, 2]
        if key in ["A", "B"]:
            ax.errorbar(x, kp_m, yerr=kp_s, fmt="D-", capsize=3, lw=1.6)
            ax.set_xticks(x)
            ax.set_xticklabels(lambda_labels, fontsize=7)
            ax.set_xlabel(r"$\lambda$")
            ax.set_ylabel(r"$\kappa_\varepsilon(G)$")
            ax.set_title(r"(c) Spectral anisotropy $\kappa_\varepsilon(G)$", fontsize=8)
            ax.set_yscale("log")
        else:
            Bv = np.asarray(B_values, dtype=float)
            err_m = mc_errors.mean(axis=1)
            err_s = mc_errors.std(axis=1)
            ax.errorbar(Bv, err_m, yerr=err_s, fmt="o-", capsize=3, lw=1.6, label="MC error")
            ref = err_m[0] * np.sqrt(Bv[0] / Bv)
            ax.plot(Bv, ref, "--", lw=1.2, label=r"$O(B^{-1/2})$ reference")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$B$ (Gaussian samples)")
            ax.set_ylabel("Relative MC error (%)")
            ax.set_title("(c) Monte Carlo convergence", fontsize=8)
            ax.legend(fontsize=7)

    fig.suptitle("Experiment 1: Fisher Width in Trained Models", fontsize=11, y=1.01)
    fig.savefig(args.outdir / "exp1_figure.pdf", bbox_inches="tight")
    fig.savefig(args.outdir / "exp1_figure.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "\n".join(summary))
    print(f"\nSaved outputs to {args.outdir.resolve()}")


if __name__ == "__main__":
    run(parse_args())
