#!/usr/bin/env python3
"""Experiment 3: non-spherical perturbation sets and Fisher alignment.

Paper role
----------
This script supports Section 7.4. It asks why one needs the full set-dependent
Fisher width rather than only Euclidean width or the global Fisher trace.

For a fitted binary logistic model, let

    T_V = V \\cap B_2^d

for three k-dimensional subspaces V:
  - the top-k eigenspace of the conditional Fisher,
  - a random k-dimensional subspace,
  - the bottom-k eigenspace.

At fixed k all T_V have the same ordinary Euclidean Gaussian width, while their
Fisher widths depend on alignment with the Fisher operator. We also measure
held-out finite-epsilon logistic-loss sensitivity on the same subspaces.

The model has no intercept and uses the official MNIST train/test split for
0-vs-1, matching p=784 in the paper.

Outputs
-------
  <outdir>/exp3_figure.pdf
  <outdir>/exp3_figure.png
  <outdir>/exp3_results.csv
  <outdir>/exp3_summary.csv
  <outdir>/exp3_run.txt

Run
---
  python exp3_fisher_alignment.py
  python exp3_fisher_alignment.py --quick   # smoke test only
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
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from sklearn.preprocessing import StandardScaler


MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

SEED = 20260818
LAMBDA = 1.0
EPS = 0.02
K_VALUES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 784]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("mnist_data"))
    p.add_argument("--outdir", type=Path, default=Path("results/exp3"))
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
            return np.frombuffer(f.read(), np.uint8).reshape(n, rows * cols).astype(np.float64) / 255.0
        if magic == 2049:
            return np.frombuffer(f.read(), np.uint8).astype(int)
    raise ValueError(f"Unknown IDX file: {path}")


def load_official_binary_split(data_dir: Path):
    ensure_mnist(data_dir)
    Xtr = read_idx(data_dir / "train-images-idx3-ubyte.gz")
    ytr = read_idx(data_dir / "train-labels-idx1-ubyte.gz")
    Xte = read_idx(data_dir / "t10k-images-idx3-ubyte.gz")
    yte = read_idx(data_dir / "t10k-labels-idx1-ubyte.gz")
    mtr = (ytr == 0) | (ytr == 1)
    mte = (yte == 0) | (yte == 1)
    return Xtr[mtr], ytr[mtr].astype(float), Xte[mte], yte[mte].astype(float)


def euclidean_width(k: int) -> float:
    return float(np.sqrt(2.0) * np.exp(gammaln((k + 1) / 2.0) - gammaln(k / 2.0)))


def logistic_loss_from_logit(z, y):
    return np.logaddexp(0.0, z) - y * z


def fit_logistic_l2(X, y, lam):
    n, d = X.shape

    def fg(w):
        z = X @ w
        p = expit(z)
        loss = np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * np.dot(w, w)
        grad = X.T @ (p - y) / n + lam * w
        return loss, grad

    res = minimize(
        lambda w: fg(w)[0],
        np.zeros(d),
        jac=lambda w: fg(w)[1],
        method="L-BFGS-B",
        options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not res.success:
        raise RuntimeError(res.message)
    return res.x, res


def mc_width_from_eigenvalues(mu, rng, B=10_000, batch=1000):
    mu = np.clip(np.asarray(mu, dtype=np.float64), 0.0, None)
    if mu.size == 0 or mu.max(initial=0.0) == 0.0:
        return 0.0
    total = 0.0
    done = 0
    while done < B:
        m = min(batch, B - done)
        z2 = rng.standard_normal((m, mu.size)) ** 2
        total += np.sqrt(z2 @ mu).sum()
        done += m
    return float(total / B)


def write_rows(path: Path, rows):
    rows = list(rows)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    args.outdir.mkdir(parents=True, exist_ok=True)
    X_train_raw, y_train, X_test_raw, y_test = load_official_binary_split(args.data_dir)
    d = X_train_raw.shape[1]

    if args.quick:
        # Keep the same fitted model/data but reduce MC and random-subspace counts.
        MC_SAMPLES = 1000
        N_RANDOM = 3
        k_values = [2, 8, 32, 128, 512, d]
        print("QUICK MODE: outputs are only for a smoke test, not for the manuscript.")
    else:
        MC_SAMPLES = 10_000
        N_RANDOM = 20
        k_values = K_VALUES

    # Training-only standardization, with zero-variance coordinates left at scale 1.
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    theta, opt = fit_logistic_l2(X_train, y_train, LAMBDA)
    z_train = X_train @ theta
    z_test = X_test @ theta
    p_train = expit(z_train)
    p_test = expit(z_test)

    # Conditional model Fisher averaged over observed training covariates.
    q = p_train * (1.0 - p_train)
    G = (X_train.T * q) @ X_train / X_train.shape[0]
    G = 0.5 * (G + G.T)
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.clip(eigvals, 0.0, None)
    eig_desc = eigvals[::-1]
    U_top = eigvecs[:, ::-1]
    U_bottom = eigvecs

    base_test_loss = logistic_loss_from_logit(z_test, y_test)

    def sensitivity_from_radius(radius):
        # For fixed y, binary logistic NLL is monotone in the logit. Over the
        # symmetric interval [-eps*r, eps*r], the maximum absolute deviation
        # from the base loss is therefore attained at one of the endpoints.
        z_plus = z_test + EPS * radius
        z_minus = z_test - EPS * radius
        l_plus = logistic_loss_from_logit(z_plus, y_test)
        l_minus = logistic_loss_from_logit(z_minus, y_test)
        return float(np.mean(np.maximum(np.abs(l_plus - base_test_loss), np.abs(l_minus - base_test_loss))))

    # Nested projections for top/bottom spaces.
    proj_top = X_test @ U_top
    proj_bottom = X_test @ U_bottom
    cum_top = np.cumsum(proj_top ** 2, axis=1)
    cum_bottom = np.cumsum(proj_bottom ** 2, axis=1)

    # Common draws for smooth nested top/bottom Fisher-width curves.
    rng_common = np.random.default_rng(SEED)
    z2 = rng_common.standard_normal((MC_SAMPLES, d)) ** 2
    cum_mc_top = np.cumsum(z2 * eig_desc[None, :], axis=1)
    cum_mc_bottom = np.cumsum(z2 * eigvals[None, :], axis=1)

    records = []
    for kind, eig_order, cum_proj, cum_mc in [
        ("top", eig_desc, cum_top, cum_mc_top),
        ("bottom", eigvals, cum_bottom, cum_mc_bottom),
    ]:
        for k in k_values:
            records.append({
                "subspace": kind,
                "replicate": 0,
                "k": k,
                "euclidean_width": euclidean_width(k),
                "fisher_width": float(np.sqrt(cum_mc[:, k - 1]).mean()),
                "trace_upper": float(np.sqrt(eig_order[:k].sum())),
                "heldout_sensitivity": sensitivity_from_radius(np.sqrt(cum_proj[:, k - 1])),
            })

    # Random nested Haar frames. For each replicate, the first k columns form a
    # Haar-distributed k-dimensional subspace; nesting is only used to reduce
    # simulation noise across k within that replicate.
    max_random_k = max(k for k in k_values if k < d)
    for rep in range(N_RANDOM):
        rng = np.random.default_rng(SEED + 1000 + rep)
        A = rng.standard_normal((d, max_random_k))
        Q, _ = np.linalg.qr(A, mode="reduced")
        GQ = 0.5 * (Q.T @ G @ Q + (Q.T @ G @ Q).T)
        proj = X_test @ Q
        cum_proj = np.cumsum(proj ** 2, axis=1)

        for k in [kk for kk in k_values if kk < d]:
            mu = np.linalg.eigvalsh(GQ[:k, :k])
            records.append({
                "subspace": "random",
                "replicate": rep,
                "k": k,
                "euclidean_width": euclidean_width(k),
                "fisher_width": mc_width_from_eigenvalues(mu, np.random.default_rng(SEED + 100_000 + 1000 * rep + k), MC_SAMPLES),
                "trace_upper": float(np.sqrt(max(mu.sum(), 0.0))),
                "heldout_sensitivity": sensitivity_from_radius(np.sqrt(cum_proj[:, k - 1])),
            })

    # At k=d all orientations coincide.
    if d in k_values:
        full_fw = mc_width_from_eigenvalues(eigvals, np.random.default_rng(SEED + 999_999), MC_SAMPLES)
        full_trace = float(np.sqrt(eigvals.sum()))
        full_sens = sensitivity_from_radius(np.linalg.norm(X_test, axis=1))
        for rep in range(N_RANDOM):
            records.append({
                "subspace": "random",
                "replicate": rep,
                "k": d,
                "euclidean_width": euclidean_width(d),
                "fisher_width": full_fw,
                "trace_upper": full_trace,
                "heldout_sensitivity": full_sens,
            })

    write_rows(args.outdir / "exp3_results.csv", records)

    # Aggregate by k.
    def values(kind, k, field):
        return np.asarray([r[field] for r in records if r["subspace"] == kind and r["k"] == k], dtype=float)

    summary_rows = []
    for k in k_values:
        top_fw = values("top", k, "fisher_width")[0]
        bot_fw = values("bottom", k, "fisher_width")[0]
        rand_fw = values("random", k, "fisher_width")
        top_s = values("top", k, "heldout_sensitivity")[0]
        bot_s = values("bottom", k, "heldout_sensitivity")[0]
        rand_s = values("random", k, "heldout_sensitivity")
        top_tr = values("top", k, "trace_upper")[0]
        bot_tr = values("bottom", k, "trace_upper")[0]
        rand_tr = values("random", k, "trace_upper")
        summary_rows.append({
            "k": k,
            "euclidean_width": euclidean_width(k),
            "top_fisher_width": top_fw,
            "random_fisher_width_mean": rand_fw.mean(),
            "random_fisher_width_std": rand_fw.std(),
            "bottom_fisher_width": bot_fw,
            "top_trace_upper": top_tr,
            "random_trace_upper_mean": rand_tr.mean(),
            "random_trace_upper_std": rand_tr.std(),
            "bottom_trace_upper": bot_tr,
            "top_heldout_sensitivity": top_s,
            "random_heldout_sensitivity_mean": rand_s.mean(),
            "random_heldout_sensitivity_std": rand_s.std(),
            "bottom_heldout_sensitivity": bot_s,
        })
    write_rows(args.outdir / "exp3_summary.csv", summary_rows)

    # Figure: one 1x3 panel, no PDF/image post-processing dependencies.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    plt.subplots_adjust(wspace=0.34)
    ks = np.asarray(k_values)

    top_fw = np.asarray([r["top_fisher_width"] for r in summary_rows])
    rand_fw = np.asarray([r["random_fisher_width_mean"] for r in summary_rows])
    rand_fw_std = np.asarray([r["random_fisher_width_std"] for r in summary_rows])
    bot_fw = np.asarray([r["bottom_fisher_width"] for r in summary_rows])
    euc = np.asarray([r["euclidean_width"] for r in summary_rows])

    ax = axes[0]
    ax.plot(ks, top_fw, marker="o", label="Top eigenspace")
    ax.errorbar(ks, rand_fw, yerr=rand_fw_std, marker="o", capsize=2, label="Random subspace")
    ax.plot(ks, bot_fw, marker="o", label="Bottom eigenspace")
    ax.plot(ks, euc, "--", label="Common Euclidean width")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], rotation=45, fontsize=7)
    ax.set_xlabel("Subspace dimension $k$")
    ax.set_ylabel("Width")
    ax.set_title("(a) Same Euclidean width, different Fisher widths", fontsize=9)
    ax.legend(fontsize=7)

    top_tr = np.asarray([r["top_trace_upper"] for r in summary_rows])
    rand_tr = np.asarray([r["random_trace_upper_mean"] for r in summary_rows])
    rand_tr_std = np.asarray([r["random_trace_upper_std"] for r in summary_rows])
    bot_tr = np.asarray([r["bottom_trace_upper"] for r in summary_rows])
    ax = axes[1]
    ax.plot(top_tr, top_fw, "o", label="Top eigenspace")
    ax.errorbar(rand_tr, rand_fw, xerr=rand_tr_std, yerr=rand_fw_std, fmt="o", capsize=2, label="Random subspace")
    ax.plot(bot_tr, bot_fw, "o", label="Bottom eigenspace")
    mx = max(top_tr.max(), rand_tr.max(), bot_tr.max())
    ax.plot([0, mx], [0, mx], "--", label=r"$y=x$")
    ax.set_xlabel(r"Restricted trace upper scale $\sqrt{\mathrm{Tr}(P_V\widehat G P_V)}$")
    ax.set_ylabel(r"$\widehat w_G(T_V)$")
    ax.set_title("(b) Fisher width versus restricted trace upper scale", fontsize=9)
    ax.legend(fontsize=7)

    top_s = np.asarray([r["top_heldout_sensitivity"] for r in summary_rows])
    rand_s = np.asarray([r["random_heldout_sensitivity_mean"] for r in summary_rows])
    rand_s_std = np.asarray([r["random_heldout_sensitivity_std"] for r in summary_rows])
    bot_s = np.asarray([r["bottom_heldout_sensitivity"] for r in summary_rows])
    ax = axes[2]
    ax.plot(ks, top_s, marker="o", label="Top eigenspace")
    ax.errorbar(ks, rand_s, yerr=rand_s_std, marker="o", capsize=2, label="Random subspace")
    ax.plot(ks, bot_s, marker="o", label="Bottom eigenspace")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], rotation=45, fontsize=7)
    ax.set_xlabel("Subspace dimension $k$")
    ax.set_ylabel("Held-out loss sensitivity")
    ax.set_title(r"(c) Held-out local sensitivity, $\varepsilon=0.02$", fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle("Experiment 3: Non-Spherical Perturbation Sets and Fisher Alignment", fontsize=10, y=1.02)
    fig.savefig(args.outdir / "exp3_figure.pdf", bbox_inches="tight")
    fig.savefig(args.outdir / "exp3_figure.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    train_acc = float(np.mean((p_train >= 0.5) == y_train))
    test_acc = float(np.mean((p_test >= 0.5) == y_test))
    meta = [
        "Experiment 3: Fisher alignment",
        f"MNIST binary train n={len(y_train)}; test n={len(y_test)}; d={d}",
        f"lambda={LAMBDA}; epsilon={EPS}; MC samples={MC_SAMPLES}; random subspaces={N_RANDOM}",
        f"optimizer iterations={opt.nit}",
        f"train accuracy={train_acc:.6f}",
        f"test accuracy={test_acc:.6f}",
        f"Tr(Ghat)={eigvals.sum():.10f}",
        f"lambda_max(Ghat)={eigvals.max():.10f}",
        f"numerical rank(eig>1e-10)={int((eigvals > 1e-10).sum())}",
    ]
    if 8 in k_values:
        r8 = next(r for r in summary_rows if r["k"] == 8)
        meta += [
            "",
            "k=8 manuscript-ready values:",
            f"  common Euclidean width={r8['euclidean_width']:.6f}",
            f"  top Fisher width={r8['top_fisher_width']:.6f}",
            f"  random Fisher width={r8['random_fisher_width_mean']:.6f} +- {r8['random_fisher_width_std']:.6f}",
            f"  bottom Fisher width={r8['bottom_fisher_width']:.6f}",
            f"  top sensitivity={r8['top_heldout_sensitivity']:.8f}",
            f"  random sensitivity={r8['random_heldout_sensitivity_mean']:.8f} +- {r8['random_heldout_sensitivity_std']:.8f}",
            f"  bottom sensitivity={r8['bottom_heldout_sensitivity']:.8f}",
        ]
    (args.outdir / "exp3_run.txt").write_text("\n".join(meta))

    print("\n" + "\n".join(meta))
    print(f"\nSaved outputs to {args.outdir.resolve()}")


if __name__ == "__main__":
    run(parse_args())
