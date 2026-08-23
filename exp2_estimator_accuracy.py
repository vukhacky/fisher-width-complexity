#!/usr/bin/env python3
"""Experiment 2: estimator accuracy and stability.

Paper role
----------
This script supports Section 7.3. It computes three sub-experiments, while the main-paper figure displays 2.1 and 2.3:

  2.1 Rank-k approximation on the SAME Fisher matrix.
      For each seed s, form G^(s), truncate that very matrix to G_k^(s), and
      compare

          w_{G^(s)}(T) - w_{G_k^(s)}(T)

      with sqrt(lambda_{k+1}(G^(s))) w(T). This directly matches the matrix
      pairing in Proposition 6.1.

  2.2 Fixed-base-point empirical convergence diagnostic.
      Fit one theta_ref once, keep it fixed, define a large-sample conditional
      Fisher reference G_ref at theta_ref, and estimate G_n on subsamples at
      the same theta_ref. This removes the earlier confounding from refitting
      theta at every n. Because the matrices may still be rank deficient, this
      is presented as a convergence diagnostic rather than a literal test of
      the positive-definite corollary.

  2.3 Structured approximation.
      Compare the full conditional Fisher with its diagonal approximation and
      with the trace upper scale sqrt(Tr(G)). Also compute the deterministic
      Theorem 3.1 stability bound for the diagonal approximation.

The binary logistic model has NO intercept, matching p=784 in the paper.

Outputs
-------
  <outdir>/exp2_figure.pdf   # main-paper two-panel figure
  <outdir>/exp2_figure.png   # main-paper two-panel figure
  <outdir>/exp2_results.npz
  <outdir>/exp2_summary.txt
  <outdir>/exp2_rankk.csv
  <outdir>/exp2_convergence.csv
  <outdir>/exp2_structured.csv

Run
---
  python exp2_estimator_accuracy.py
  python exp2_estimator_accuracy.py --quick   # smoke test only
"""

from __future__ import annotations

import argparse
import csv
import gzip
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
MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

MASTER_SEED = 42
LAM_FIX = 0.01
RANKS = [1, 2, 5, 10, 20, 30, 50, 100, 200, 392]
N_VALUES = [100, 200, 500, 1000, 2000, 5000, 10_000]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("mnist_data"))
    p.add_argument("--outdir", type=Path, default=Path("results/exp2"))
    p.add_argument("--quick", action="store_true", help="Small smoke-test run; do not use its numbers in the paper.")
    return p.parse_args()


def ensure_mnist(data_dir: Path):
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


def load_binary_mnist(data_dir: Path):
    ensure_mnist(data_dir)
    X = np.concatenate([
        read_idx(data_dir / "train-images-idx3-ubyte.gz"),
        read_idx(data_dir / "t10k-images-idx3-ubyte.gz"),
    ])
    y = np.concatenate([
        read_idx(data_dir / "train-labels-idx1-ubyte.gz"),
        read_idx(data_dir / "t10k-labels-idx1-ubyte.gz"),
    ])
    mask = (y == 0) | (y == 1)
    return X[mask], y[mask]


def euclidean_ball_width(d: int) -> float:
    return float(np.exp(0.5 * np.log(2.0) + gammaln((d + 1) / 2.0) - gammaln(d / 2.0)))


def logistic_C(lam: float, n: int) -> float:
    return 1.0 / (lam * n) if lam > 0 else 1e12


def fit_logistic(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    clf = LogisticRegression(
        C=logistic_C(lam, len(X)),
        max_iter=2000,
        solver="lbfgs",
        fit_intercept=False,
        random_state=0,
        tol=1e-6,
    )
    clf.fit(X, y)
    return clf.coef_.ravel()


def conditional_fisher_binary(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    z = X @ theta
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    q = p * (1.0 - p)
    G = (X * q[:, None]).T @ X / len(X)
    return 0.5 * (G + G.T)


def sqrt_psd(G: np.ndarray, eigvals=None, eigvecs=None) -> np.ndarray:
    if eigvals is None or eigvecs is None:
        eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.clip(eigvals, 0.0, None)
    return (eigvecs * np.sqrt(eigvals)[None, :]) @ eigvecs.T


def width_from_spectrum(eigvals: np.ndarray, z2: np.ndarray) -> float:
    eigvals = np.clip(np.asarray(eigvals, dtype=float), 0.0, None)
    return float(np.sqrt(z2 @ eigvals).mean())


def rankk_widths_from_desc_spectrum(eig_desc: np.ndarray, ranks, B: int, rng, batch: int = 1000):
    """Use common Gaussian draws to estimate full and nested top-k widths."""
    eig_desc = np.clip(np.asarray(eig_desc, dtype=float), 0.0, None)
    ranks = list(ranks)
    sums = {k: 0.0 for k in ranks}
    full_sum = 0.0
    done = 0
    while done < B:
        m = min(batch, B - done)
        z2 = rng.standard_normal((m, eig_desc.size)) ** 2
        cum = np.cumsum(z2 * eig_desc[None, :], axis=1)
        full_sum += np.sqrt(cum[:, -1]).sum()
        for k in ranks:
            sums[k] += np.sqrt(cum[:, k - 1]).sum()
        done += m
    return full_sum / B, {k: sums[k] / B for k in ranks}


def mc_width_from_spectrum(eigvals: np.ndarray, B: int, rng, batch: int = 1000) -> float:
    eigvals = np.clip(np.asarray(eigvals, dtype=float), 0.0, None)
    total = 0.0
    done = 0
    while done < B:
        m = min(batch, B - done)
        z2 = rng.standard_normal((m, eigvals.size)) ** 2
        total += np.sqrt(z2 @ eigvals).sum()
        done += m
    return float(total / B)


def operator_norm_symmetric(A: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(0.5 * (A + A.T))
    return float(np.max(np.abs(vals)))


def write_csv(path: Path, rows):
    rows = list(rows)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    args.outdir.mkdir(parents=True, exist_ok=True)
    X_bin, y_bin = load_binary_mnist(args.data_dir)
    d = X_bin.shape[1]
    wE = euclidean_ball_width(d)

    if args.quick:
        n_seeds = 2
        n_ref = 1500
        ranks = [1, 5, 20, 100]
        n_values = [100, 300, 800, 1500]
        B_rank = 2500
        B_width = 1200
        print("QUICK MODE: outputs are only for a smoke test, not for the manuscript.")
    else:
        n_seeds = 10
        n_ref = 10_000
        ranks = RANKS
        n_values = N_VALUES
        B_rank = 20_000
        B_width = 5_000

    scaler = StandardScaler()

    # ================================================================
    # 2.1 Rank-k approximation: SAME G within each seed.
    # ================================================================
    print("\n--- 2.1 Rank-k approximation on the same matrix ---")
    rank_errors = np.zeros((len(ranks), n_seeds))
    rank_bounds = np.zeros((len(ranks), n_seeds))
    rank_width_full = np.zeros(n_seeds)
    rank_width_k = np.zeros((len(ranks), n_seeds))
    rank_lambda_kp1 = np.zeros((len(ranks), n_seeds))

    seed_G = []
    seed_eigh = []

    for si in range(n_seeds):
        rng_data = np.random.default_rng(MASTER_SEED + 100 * si)
        idx = rng_data.permutation(len(X_bin))
        Xtr = scaler.fit_transform(X_bin[idx[:n_ref]])
        ytr = y_bin[idx[:n_ref]]
        theta = fit_logistic(Xtr, ytr, LAM_FIX)
        G = conditional_fisher_binary(Xtr, theta)
        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals = np.clip(eigvals, 0.0, None)
        eig_desc = eigvals[::-1]
        seed_G.append(G)
        seed_eigh.append((eigvals, eigvecs))

        full_w, k_widths = rankk_widths_from_desc_spectrum(
            eig_desc, ranks, B_rank, np.random.default_rng(MASTER_SEED + 10_000 + si)
        )
        rank_width_full[si] = full_w
        for ki, k in enumerate(ranks):
            wk = k_widths[k]
            lam_kp1 = eig_desc[k] if k < d else 0.0
            bound = np.sqrt(max(lam_kp1, 0.0)) * wE
            rank_width_k[ki, si] = wk
            rank_errors[ki, si] = full_w - wk  # nonnegative for Euclidean balls/truncation
            rank_lambda_kp1[ki, si] = lam_kp1
            rank_bounds[ki, si] = bound

    rank_rows = []
    for ki, k in enumerate(ranks):
        rank_rows.append({
            "k": k,
            "error_mean": rank_errors[ki].mean(),
            "error_std": rank_errors[ki].std(),
            "bound_mean": rank_bounds[ki].mean(),
            "bound_std": rank_bounds[ki].std(),
            "sqrt_lambda_kp1_mean": np.sqrt(rank_lambda_kp1[ki]).mean(),
            "max_seed_error_minus_bound": np.max(rank_errors[ki] - rank_bounds[ki]),
        })
    write_csv(args.outdir / "exp2_rankk.csv", rank_rows)
    max_rank_violation = float(np.max(rank_errors - rank_bounds))
    print(f"Max seed-wise MC(error - theoretical bound) = {max_rank_violation:.6g}")

    # ================================================================
    # 2.2 Fixed-base-point empirical convergence.
    # ================================================================
    print("\n--- 2.2 Fixed-base-point convergence diagnostic ---")
    # Fit theta_ref once on one fixed reference training subset.
    rng_ref = np.random.default_rng(MASTER_SEED + 500_000)
    perm_ref = rng_ref.permutation(len(X_bin))
    train_idx = perm_ref[:n_ref]
    X_train_raw = X_bin[train_idx]
    y_train = y_bin[train_idx]
    scaler_ref = StandardScaler().fit(X_train_raw)
    X_train = scaler_ref.transform(X_train_raw)
    theta_ref = fit_logistic(X_train, y_train, LAM_FIX)

    # Keep theta_ref and the coordinate system fixed. Use all available binary
    # covariates as a large-sample empirical target for the conditional Fisher.
    X_pool = scaler_ref.transform(X_bin)
    G_target = conditional_fisher_binary(X_pool, theta_ref)
    eig_target = np.linalg.eigvalsh(G_target)
    rng_width_common = np.random.default_rng(MASTER_SEED + 600_000)
    z2_common = rng_width_common.standard_normal((B_width, d)) ** 2
    w_target = width_from_spectrum(eig_target, z2_common)

    conv_widths = np.zeros((len(n_values), n_seeds))
    conv_errors = np.zeros((len(n_values), n_seeds))
    conv_metric_errors = np.zeros((len(n_values), n_seeds))

    for ni, n in enumerate(n_values):
        for si in range(n_seeds):
            rng = np.random.default_rng(MASTER_SEED + 700_000 + 1000 * ni + si)
            idx = rng.choice(len(X_pool), size=n, replace=False)
            G_n = conditional_fisher_binary(X_pool[idx], theta_ref)
            eig_n = np.linalg.eigvalsh(G_n)
            w_n = width_from_spectrum(eig_n, z2_common)
            conv_widths[ni, si] = w_n
            conv_errors[ni, si] = abs(w_n - w_target)
            conv_metric_errors[ni, si] = operator_norm_symmetric(G_n - G_target)

    # Log-log descriptive slope.
    mean_err = conv_errors.mean(axis=1)
    positive = mean_err > 0
    slope = float(np.polyfit(np.log(np.asarray(n_values, float)[positive]), np.log(mean_err[positive]), 1)[0])

    conv_rows = []
    for ni, n in enumerate(n_values):
        conv_rows.append({
            "n": n,
            "w_mean": conv_widths[ni].mean(),
            "w_std": conv_widths[ni].std(),
            "abs_width_error_mean": conv_errors[ni].mean(),
            "abs_width_error_std": conv_errors[ni].std(),
            "metric_op_error_mean": conv_metric_errors[ni].mean(),
            "metric_op_error_std": conv_metric_errors[ni].std(),
            "n_over_d": n / d,
        })
    write_csv(args.outdir / "exp2_convergence.csv", conv_rows)
    print(f"Fixed-base-point width-error log-log slope = {slope:.3f}")

    # ================================================================
    # 2.3 Structured approximation: full vs diagonal + stability bound.
    # ================================================================
    print("\n--- 2.3 Structured approximation ---")
    full_widths = np.zeros(n_seeds)
    diag_widths = np.zeros(n_seeds)
    trace_uppers = np.zeros(n_seeds)
    diag_errors = np.zeros(n_seeds)
    stability_bounds = np.zeros(n_seeds)

    for si in range(n_seeds):
        G = seed_G[si]
        eigvals, eigvecs = seed_eigh[si]
        # Reuse a common isotropic Gaussian table for full and diagonal widths.
        rng = np.random.default_rng(MASTER_SEED + 800_000 + si)
        z2 = rng.standard_normal((B_width, d)) ** 2
        full_w = width_from_spectrum(eigvals, z2)
        dG = np.clip(np.diag(G), 0.0, None)
        diag_w = width_from_spectrum(dG, z2)
        trub = float(np.sqrt(np.trace(G)))

        Ghalf = sqrt_psd(G, eigvals, eigvecs)
        Dhalf = np.diag(np.sqrt(dG))
        stab = operator_norm_symmetric(Ghalf - Dhalf) * wE

        full_widths[si] = full_w
        diag_widths[si] = diag_w
        trace_uppers[si] = trub
        diag_errors[si] = abs(diag_w - full_w)
        stability_bounds[si] = stab

    struct_rows = []
    for si in range(n_seeds):
        struct_rows.append({
            "seed": si,
            "full_conditional_fisher_width": full_widths[si],
            "diagonal_conditional_fisher_width": diag_widths[si],
            "trace_upper_scale": trace_uppers[si],
            "diagonal_abs_error": diag_errors[si],
            "theorem_3_1_stability_bound": stability_bounds[si],
            "bound_satisfied": bool(diag_errors[si] <= stability_bounds[si] + 1e-10),
        })
    write_csv(args.outdir / "exp2_structured.csv", struct_rows)

    np.savez(
        args.outdir / "exp2_results.npz",
        ranks=np.asarray(ranks),
        rank_errors=rank_errors,
        rank_bounds=rank_bounds,
        rank_width_full=rank_width_full,
        rank_width_k=rank_width_k,
        rank_lambda_kp1=rank_lambda_kp1,
        n_values=np.asarray(n_values),
        conv_widths=conv_widths,
        conv_errors=conv_errors,
        conv_metric_errors=conv_metric_errors,
        conv_slope=slope,
        w_target=w_target,
        full_widths=full_widths,
        diag_widths=diag_widths,
        trace_uppers=trace_uppers,
        diag_errors=diag_errors,
        stability_bounds=stability_bounds,
        w_euclidean=wE,
        lambda_fixed=LAM_FIX,
        n_ref=n_ref,
        n_seeds=n_seeds,
    )

    summary = [
        "Experiment 2: Estimator Accuracy and Stability",
        f"Binary logistic, lambda={LAM_FIX}, d={d}, seeds={n_seeds}",
        "Model matrix = full conditional model Fisher averaged over observed covariates.",
        "",
        "2.1 Rank-k approximation (same G within every comparison):",
        f"  max seed-wise MC(error - theoretical bound) = {max_rank_violation:.8f}",
    ]
    for row in rank_rows:
        summary.append(
            f"  k={row['k']}: error={row['error_mean']:.6f} +- {row['error_std']:.6f}; "
            f"bound={row['bound_mean']:.6f} +- {row['bound_std']:.6f}"
        )
    summary += [
        "",
        "2.2 Fixed-base-point convergence diagnostic:",
        f"  target width (large-sample empirical conditional Fisher) = {w_target:.6f}",
        f"  log-log slope of mean |w_n-w_ref| vs n = {slope:.4f}",
    ]
    for row in conv_rows:
        summary.append(
            f"  n={row['n']}: width error={row['abs_width_error_mean']:.6f} +- {row['abs_width_error_std']:.6f}; "
            f"metric op error={row['metric_op_error_mean']:.6f}"
        )
    summary += [
        "",
        "2.3 Structured approximation:",
        f"  full conditional Fisher width = {full_widths.mean():.6f} +- {full_widths.std():.6f}",
        f"  diagonal conditional Fisher width = {diag_widths.mean():.6f} +- {diag_widths.std():.6f}",
        f"  trace upper scale = {trace_uppers.mean():.6f} +- {trace_uppers.std():.6f}",
        f"  mean diagonal abs error = {diag_errors.mean():.6f}",
        f"  mean Theorem 3.1 stability bound = {stability_bounds.mean():.6f}",
        f"  all seed-wise stability bounds satisfied = {bool(np.all(diag_errors <= stability_bounds + 1e-10))}",
    ]
    (args.outdir / "exp2_summary.txt").write_text("\n".join(summary))

    # ================================================================
    # Figure 2 for the main paper: two panels.
    #
    # We intentionally omit the fixed-base-point convergence diagnostic
    # from the main figure. The diagnostic is still computed and saved in
    # exp2_convergence.csv / exp2_results.npz for reproducibility, but its
    # observed slope is not used as a main-paper claim.
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    plt.subplots_adjust(wspace=0.34)

    # ---------------------------------------------------------------
    # (a) Rank-k approximation: observed error vs Proposition 6.1 bound.
    # Each comparison uses the same G and its own rank-k truncation.
    # ---------------------------------------------------------------
    ax = axes[0]
    x_rank = np.sqrt(rank_lambda_kp1).mean(axis=1)
    err_m = rank_errors.mean(axis=1)
    err_s = rank_errors.std(axis=1)
    bnd_m = rank_bounds.mean(axis=1)
    bnd_s = rank_bounds.std(axis=1)

    ax.errorbar(
        x_rank,
        err_m,
        yerr=err_s,
        fmt="o-",
        capsize=3,
        lw=1.4,
        markersize=4.5,
        label=r"Observed $|w_{\widehat G}-w_{\widehat G_k}|$",
    )
    ax.errorbar(
        x_rank,
        bnd_m,
        yerr=bnd_s,
        fmt="s--",
        capsize=3,
        lw=1.3,
        markersize=4.2,
        label=r"Bound $\sqrt{\lambda_{k+1}(\widehat G)}\,w(T)$",
    )

    for ki, k in enumerate(ranks):
        if k in [1, 10, 100, 392]:
            ax.annotate(
                f"$k={k}$",
                xy=(x_rank[ki], err_m[ki]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Mean $\sqrt{\lambda_{k+1}(\widehat G)}$")
    ax.set_ylabel(r"Rank-$k$ width error / upper bound")
    ax.set_title("(a) Low-rank approximation\n(Proposition 6.1; same $G$)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.18)

    # ---------------------------------------------------------------
    # (b) Structured approximation: full / diagonal / trace upper scale.
    # Theorem 3.1 stability bounds are checked numerically and saved in
    # exp2_structured.csv, but are too loose to display on this scale.
    # ---------------------------------------------------------------
    ax = axes[1]
    labels = [
        "Full conditional\nFisher",
        "Diagonal\nconditional Fisher",
        "Trace upper\nscale",
    ]
    means = np.array([
        full_widths.mean(),
        diag_widths.mean(),
        trace_uppers.mean(),
    ])
    stds = np.array([
        full_widths.std(),
        diag_widths.std(),
        trace_uppers.std(),
    ])
    x = np.arange(3)

    ax.bar(x, means, yerr=stds, capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(r"$\widehat w_G(B_2^d)$ or upper scale")
    ax.set_title("(b) Structured approximation", fontsize=9)

    diag_pct = 100.0 * diag_errors.mean() / full_widths.mean()
    trace_pct = 100.0 * (trace_uppers.mean() - full_widths.mean()) / full_widths.mean()
    pad = 0.025 * means.max()

    ax.text(
        1,
        means[1] + stds[1] + pad,
        f"{diag_pct:.2f}% error",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    ax.text(
        2,
        means[2] + stds[2] + pad,
        f"{trace_pct:.2f}% above full",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    ax.set_ylim(0, np.max(means + stds) * 1.20)
    ax.grid(True, axis="y", alpha=0.18)

    fig.suptitle(
        rf"Experiment 2: Estimator Accuracy and Stability ($\lambda={LAM_FIX}$)",
        fontsize=10,
        y=1.02,
    )
    fig.savefig(args.outdir / "exp2_figure.pdf", bbox_inches="tight")
    fig.savefig(args.outdir / "exp2_figure.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "\n".join(summary))
    print(f"\nSaved outputs to {args.outdir.resolve()}")


if __name__ == "__main__":
    run(parse_args())
