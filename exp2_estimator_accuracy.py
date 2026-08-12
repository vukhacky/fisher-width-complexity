"""
Experiment 2: Estimator Accuracy and Stability
===============================================
Validates:
  - Corollary 3.2 (Low-Rank Fisher Approximation):
        |w_G(T) - w_{G_k}(T)| <= sqrt(lambda_{k+1}) * w(T)
  - Corollary 3.4 (Empirical Fisher Stability):
        |w_{G_hat_n}(T) - w_F(T)| <= eps_n * w(T)
        where eps_n -> 0 at rate O(1/sqrt(n))

Three sub-experiments:
  2.1  Rank-k approximation: error vs sqrt(lambda_{k+1})
  2.2  Data convergence: Fisher width error vs n
  2.3  Structured approximations: diagonal vs full Fisher

Model: Binary logistic regression on MNIST (0 vs 1)
       p=784, full Fisher matrix available
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import gammaln
import struct, gzip, os, time

# ── config ────────────────────────────────────────────────────────
MASTER_SEED  = 42
N_SEEDS      = 10
N_TRAIN_REF  = 10_000   # reference (large n) for ground truth
N_TEST       = 2_000
B_MC         = 5_000    # MC samples for width estimation
LAM_FIX      = 0.01     # fixed regularization for sub-experiments 2.1, 2.3
RANKS        = [1, 2, 5, 10, 20, 30, 50, 100, 200, 392]  # k values
N_VALUES     = [100, 200, 500, 1000, 2000, 5000, 10_000]  # for data convergence

os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# 1.  Load MNIST
# ─────────────────────────────────────────────────────────────────
def read_idx(path):
    with gzip.open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        n     = struct.unpack('>I', f.read(4))[0]
        if magic == 2051:
            r, c = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(),np.uint8).reshape(n,r*c).astype(np.float64)
        elif magic == 2049:
            return np.frombuffer(f.read(),np.uint8).astype(int)

print("Loading MNIST ...")
base  = 'mnist_data'
X_all = np.concatenate([
    read_idx(os.path.join(base,'train-images-idx3-ubyte.gz')),
    read_idx(os.path.join(base,'t10k-images-idx3-ubyte.gz'))
])
y_all = np.concatenate([
    read_idx(os.path.join(base,'train-labels-idx1-ubyte.gz')),
    read_idx(os.path.join(base,'t10k-labels-idx1-ubyte.gz'))
])
mask  = (y_all==0)|(y_all==1)
X_bin = X_all[mask];  y_bin = y_all[mask]
d = X_bin.shape[1]
print(f"Binary (0 vs 1): {X_bin.shape[0]} samples, d={d}")

# Euclidean baseline
w_euclidean = np.exp(0.5*np.log(2)+gammaln((d+1)/2)-gammaln(d/2))
print(f"w(B_2^{d}) = {w_euclidean:.4f}")

# ─────────────────────────────────────────────────────────────────
# 2.  Helpers
# ─────────────────────────────────────────────────────────────────
def sigmoid(z):
    return 1./(1.+np.exp(-np.clip(z,-500,500)))

def compute_fisher(X, theta):
    p_ = sigmoid(X@theta); w_ = p_*(1-p_)
    return (X*w_[:,None]).T@X/len(X)

def matrix_sqrt_full(G, eps=1e-10):
    ev,evec = np.linalg.eigh(G)
    return evec@np.diag(np.sqrt(np.maximum(ev,eps)))@evec.T

def mc_width(G_half, B, rng):
    g = rng.standard_normal((G_half.shape[0], B))
    return np.linalg.norm(G_half@g, axis=0).mean()

def rank_k_sqrt(eigvals, eigvecs, k, eps=1e-10):
    """G_k^{1/2}: keep top-k eigenvalues, zero the rest."""
    lk = eigvals.copy()
    lk[:-k] = 0.0   # eigvalsh returns ascending order
    return eigvecs @ np.diag(np.sqrt(np.maximum(lk,0.))) @ eigvecs.T

def train_logistic(Xtr_s, ytr, lam):
    C = 1./(lam*len(Xtr_s)) if lam>0 else 1e8
    clf = LogisticRegression(C=C, max_iter=2000, solver='lbfgs',
                             random_state=0, tol=1e-6)
    clf.fit(Xtr_s, ytr)
    return clf.coef_.flatten()

scaler = StandardScaler()

# ─────────────────────────────────────────────────────────────────
# 3.  Compute reference Fisher matrix and width (large n, many seeds)
# ─────────────────────────────────────────────────────────────────
print("\n--- Computing reference (n=10000, 10 seeds) ---")
G_ref_list  = []
wF_ref_list = []

for si in range(N_SEEDS):
    rng = np.random.default_rng(MASTER_SEED + si*100)
    idx = rng.permutation(len(X_bin))
    Xtr_s = scaler.fit_transform(X_bin[idx[:N_TRAIN_REF]])
    ytr   = y_bin[idx[:N_TRAIN_REF]]
    theta = train_logistic(Xtr_s, ytr, LAM_FIX)
    G     = compute_fisher(Xtr_s, theta)
    Gh    = matrix_sqrt_full(G)
    wF_ref_list.append(mc_width(Gh, B_MC, rng))
    G_ref_list.append(G)

wF_ref    = np.mean(wF_ref_list)
wF_ref_std= np.std(wF_ref_list)
G_ref     = np.mean(G_ref_list, axis=0)   # average Fisher as stable reference
print(f"Reference wF = {wF_ref:.6f} ± {wF_ref_std:.6f}")

# Eigendecomposition of reference G (used for rank-k experiment)
eigvals_ref, eigvecs_ref = np.linalg.eigh(G_ref)
# eigvalsh returns ascending order; top eigenvalues are at the end
print(f"Top-5 eigenvalues: {eigvals_ref[-5:][::-1]}")
print(f"lambda_{{k+1}} at k=10: {eigvals_ref[-(10+1)]:.6f}")

# ─────────────────────────────────────────────────────────────────
# 4.  Sub-experiment 2.1: Rank-k approximation
#     Validate: |wF - wF_k| <= sqrt(lambda_{k+1}) * w(T)
# ─────────────────────────────────────────────────────────────────
print("\n--- Sub-exp 2.1: Rank-k approximation ---")
print(f"{'k':>6}  {'wF_k mean':>10}  {'|wF-wF_k|':>12}  "
      f"{'sqrt(lam_k+1)':>14}  {'bound':>12}  {'ratio':>8}")
print("-"*70)

wF_k_all     = np.zeros((len(RANKS), N_SEEDS))
error_k_all  = np.zeros((len(RANKS), N_SEEDS))
bound_k_all  = np.zeros((len(RANKS), N_SEEDS))
sqrt_lam_all = np.zeros(len(RANKS))

for ki, k in enumerate(RANKS):
    Gh_k = rank_k_sqrt(eigvals_ref, eigvecs_ref, k)
    # lambda_{k+1} in ascending order: index d-k-1
    idx_kp1 = d - k - 1
    lam_kp1 = eigvals_ref[idx_kp1] if idx_kp1 >= 0 else 0.0
    sqrt_lam_all[ki] = np.sqrt(max(lam_kp1, 0.))

    for si in range(N_SEEDS):
        rng = np.random.default_rng(MASTER_SEED + si*100 + 10)
        wF_k_all[ki, si] = mc_width(Gh_k, B_MC, rng)
        error_k_all[ki, si] = abs(wF_k_all[ki, si] - wF_ref_list[si])
        bound_k_all[ki, si] = sqrt_lam_all[ki] * w_euclidean

    em = error_k_all[ki].mean(); es = error_k_all[ki].std()
    bm = bound_k_all[ki].mean()
    wm = wF_k_all[ki].mean()
    ratio = em/bm if bm>1e-10 else np.nan
    print(f"{k:>6}  {wm:>10.4f}  {em:>10.4f}±{es:<6.4f}  "
          f"{sqrt_lam_all[ki]:>14.6f}  {bm:>12.6f}  {ratio:>8.4f}")

# Check: bound always satisfied?
violations_21 = (error_k_all - bound_k_all).max()
print(f"\nMax(error - bound) = {violations_21:.6f}"
      + ("  OK (bound satisfied)" if violations_21<=1e-4 else "  VIOLATED"))

# ─────────────────────────────────────────────────────────────────
# 5.  Sub-experiment 2.2: Data convergence
#     Validate: |wF_n - wF_ref| scales as O(1/sqrt(n))
# ─────────────────────────────────────────────────────────────────
print("\n--- Sub-exp 2.2: Data convergence ---")
print(f"{'n':>8}  {'wF_n mean':>10}  {'|wF_n-wF_ref|':>14}  "
      f"{'rel_err%':>10}  {'n/d':>6}")
print("-"*55)

wF_n_all    = np.zeros((len(N_VALUES), N_SEEDS))
error_n_all = np.zeros((len(N_VALUES), N_SEEDS))

for ni, n in enumerate(N_VALUES):
    t0 = time.time()
    for si in range(N_SEEDS):
        rng = np.random.default_rng(MASTER_SEED + si*100 + 20)
        idx = rng.permutation(len(X_bin))
        # Use only n samples
        Xtr_s = scaler.fit_transform(X_bin[idx[:n]])
        ytr   = y_bin[idx[:n]]

        # Need enough samples to fit logistic; skip if n < d/10
        if n < 100:
            wF_n_all[ni, si]    = np.nan
            error_n_all[ni, si] = np.nan
            continue

        theta = train_logistic(Xtr_s, ytr, LAM_FIX)
        G_n   = compute_fisher(Xtr_s, theta)
        Gh_n  = matrix_sqrt_full(G_n)
        wF_n  = mc_width(Gh_n, B_MC, rng)
        wF_n_all[ni, si]    = wF_n
        error_n_all[ni, si] = abs(wF_n - wF_ref)

    em = np.nanmean(error_n_all[ni])
    es = np.nanstd(error_n_all[ni])
    wm = np.nanmean(wF_n_all[ni])
    rel= em/wF_ref*100
    print(f"{n:>8}  {wm:>10.4f}  {em:>12.4f}±{es:<6.4f}  "
          f"{rel:>9.2f}%  {n/d:>6.2f}  [{time.time()-t0:.1f}s]")

# Fit O(1/sqrt(n)) to data convergence
valid = ~np.isnan(error_n_all.mean(1))
Nv    = np.array(N_VALUES)[valid].astype(float)
ev    = error_n_all[valid].mean(1)
# Log-log regression: log(e) = a - 0.5*log(n) + b
log_n = np.log(Nv); log_e = np.log(ev)
coeffs = np.polyfit(log_n, log_e, 1)
print(f"\nLog-log slope (expect -0.5): {coeffs[0]:.3f}")

# ─────────────────────────────────────────────────────────────────
# 6.  Sub-experiment 2.3: Structured approximations
#     Compare: full Fisher vs diagonal vs score upper bound
# ─────────────────────────────────────────────────────────────────
print("\n--- Sub-exp 2.3: Structured approximations ---")
print(f"{'seed':>6}  {'wF_full':>10}  {'wF_diag':>10}  "
      f"{'scoreUB':>10}  {'err_diag%':>10}  {'err_score%':>11}")
print("-"*60)

wF_full_all  = np.zeros(N_SEEDS)
wF_diag_all  = np.zeros(N_SEEDS)
wF_score_all = np.zeros(N_SEEDS)

for si in range(N_SEEDS):
    rng = np.random.default_rng(MASTER_SEED + si*100 + 30)
    idx = rng.permutation(len(X_bin))
    Xtr_s = scaler.fit_transform(X_bin[idx[:N_TRAIN_REF]])
    ytr   = y_bin[idx[:N_TRAIN_REF]]
    theta = train_logistic(Xtr_s, ytr, LAM_FIX)
    G     = compute_fisher(Xtr_s, theta)

    # Full Fisher width
    Gh_full       = matrix_sqrt_full(G)
    wF_full_all[si] = mc_width(Gh_full, B_MC, rng)

    # Diagonal Fisher width
    dG            = np.diag(G)
    dh            = np.sqrt(np.maximum(dG, 1e-10))
    g_diag        = rng.standard_normal((d, B_MC))
    wF_diag_all[si] = np.linalg.norm(dh[:,None]*g_diag, axis=0).mean()

    # Score upper bound: sqrt(Tr(G))
    wF_score_all[si] = np.sqrt(np.trace(G))

for si in range(N_SEEDS):
    ref = wF_full_all[si]
    print(f"{si:>6}  {wF_full_all[si]:>10.4f}  {wF_diag_all[si]:>10.4f}  "
          f"{wF_score_all[si]:>10.4f}  "
          f"{abs(wF_diag_all[si]-ref)/ref*100:>9.2f}%  "
          f"{abs(wF_score_all[si]-ref)/ref*100:>10.2f}%")

print(f"\nSummary (mean ± std over {N_SEEDS} seeds):")
print(f"  Full Fisher:    {wF_full_all.mean():.4f} ± {wF_full_all.std():.4f}")
print(f"  Diagonal:       {wF_diag_all.mean():.4f} ± {wF_diag_all.std():.4f}  "
      f"(err={abs(wF_diag_all-wF_full_all).mean()/wF_full_all.mean()*100:.2f}%)")
print(f"  Score UB:       {wF_score_all.mean():.4f} ± {wF_score_all.std():.4f}  "
      f"(err={abs(wF_score_all-wF_full_all).mean()/wF_full_all.mean()*100:.2f}%)")

# ─────────────────────────────────────────────────────────────────
# 7.  Save
# ─────────────────────────────────────────────────────────────────
np.savez('results/exp2_results.npz',
         ranks=RANKS, n_values=N_VALUES,
         wF_ref=wF_ref, wF_ref_std=wF_ref_std,
         wF_k=wF_k_all, error_k=error_k_all, bound_k=bound_k_all,
         sqrt_lam=sqrt_lam_all,
         wF_n=wF_n_all, error_n=error_n_all,
         wF_full=wF_full_all, wF_diag=wF_diag_all, wF_score=wF_score_all,
         w_euclidean=w_euclidean, d=d, lam_fix=LAM_FIX,
         loglog_slope=coeffs[0])

# ─────────────────────────────────────────────────────────────────
# 8.  Summary text
# ─────────────────────────────────────────────────────────────────
lines = [
    "="*70,
    "Experiment 2: Estimator Accuracy and Stability",
    f"Model: Binary Logistic, MNIST 0 vs 1, lambda={LAM_FIX}",
    f"d={d}, n_ref={N_TRAIN_REF}, seeds={N_SEEDS}, B_MC={B_MC}",
    f"Reference wF = {wF_ref:.6f} ± {wF_ref_std:.6f}",
    f"w(B_2^d) = {w_euclidean:.6f}",
    "="*70,
    "",
    "--- 2.1 Rank-k Approximation ---",
    f"Corollary 3.2: |w_G(T) - w_{{G_k}}(T)| <= sqrt(lambda_{{k+1}}) * w(T)",
    f"{'k':>6}  {'error mean':>12}  {'bound':>12}  {'ratio e/b':>10}  {'bound OK?':>10}",
    "-"*55,
]
for ki, k in enumerate(RANKS):
    em = error_k_all[ki].mean()
    bm = bound_k_all[ki].mean()
    ratio = em/bm if bm>1e-10 else np.nan
    ok = "YES" if em <= bm+1e-4 else "NO"
    lines.append(f"{k:>6}  {em:>12.6f}  {bm:>12.6f}  {ratio:>10.4f}  {ok:>10}")

lines += [
    "",
    f"Max(error - bound) = {violations_21:.6f}"
    + ("  [bound satisfied]" if violations_21<=1e-4 else "  [VIOLATED]"),
    "",
    "--- 2.2 Data Convergence ---",
    f"Rate: log-log slope = {coeffs[0]:.3f}  (expect approx -0.5)",
    f"{'n':>8}  {'wF_n mean':>10}  {'abs_error':>10}  {'rel_err%':>10}  {'n/d':>6}",
    "-"*48,
]
for ni, n in enumerate(N_VALUES):
    em  = np.nanmean(error_n_all[ni])
    wm  = np.nanmean(wF_n_all[ni])
    rel = em/wF_ref*100
    lines.append(f"{n:>8}  {wm:>10.4f}  {em:>10.4f}  {rel:>9.2f}%  {n/d:>6.2f}")

lines += [
    "",
    "--- 2.3 Structured Approximations ---",
    f"{'Method':>15}  {'wF mean':>10}  {'±std':>8}  {'err vs full':>12}",
    "-"*50,
    f"{'Full Fisher':>15}  {wF_full_all.mean():>10.4f}  "
    f"{wF_full_all.std():>8.4f}  {'(reference)':>12}",
    f"{'Diagonal':>15}  {wF_diag_all.mean():>10.4f}  "
    f"{wF_diag_all.std():>8.4f}  "
    f"{abs(wF_diag_all-wF_full_all).mean()/wF_full_all.mean()*100:>10.2f}%",
    f"{'Score UB':>15}  {wF_score_all.mean():>10.4f}  "
    f"{wF_score_all.std():>8.4f}  "
    f"{abs(wF_score_all-wF_full_all).mean()/wF_full_all.mean()*100:>10.2f}%",
    "="*70,
]
summary = "\n".join(lines)
print("\n"+summary)
with open('results/exp2_summary.txt','w') as f:
    f.write(summary)

# ─────────────────────────────────────────────────────────────────
# 9.  Figure — 3 panels
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
plt.subplots_adjust(wspace=0.38)

# ── Panel (a): Rank-k error vs sqrt(lambda_{k+1}) ────────────────
ax = axes[0]
em = error_k_all.mean(1);  es = error_k_all.std(1)
bm = bound_k_all.mean(1)
sl = sqrt_lam_all

# x-axis: sqrt(lambda_{k+1})
ax.errorbar(sl, em, yerr=es, fmt='o', color='steelblue',
            capsize=3, lw=1.6, ms=5, label='Actual error $|w_F - w_{F,k}|$')
ax.plot(sl, bm, 's--', color='tomato', lw=1.4, ms=5,
        label=r'Bound $\sqrt{\lambda_{k+1}}\cdot w(T)$')

# Annotate selected k values
for ki, k in enumerate(RANKS):
    if k in [1, 5, 20, 100, 392]:
        ax.annotate(f'k={k}', xy=(sl[ki], em[ki]),
                    xytext=(sl[ki]*1.05, em[ki]*1.15),
                    fontsize=6.5, color='steelblue')

ax.set_xlabel(r'$\sqrt{\lambda_{k+1}(\hat G)}$', fontsize=10)
ax.set_ylabel(r'$|\hat w_F - \hat w_{F,k}|$', fontsize=10)
ax.set_title('(a) Rank-$k$ approximation error\n'
             r'vs $\sqrt{\lambda_{k+1}}$ (Corollary 3.2)', fontsize=9)
ax.legend(fontsize=8)
ax.set_xscale('log'); ax.set_yscale('log')

# ── Panel (b): Data convergence ───────────────────────────────────
ax = axes[1]
valid  = ~np.isnan(error_n_all.mean(1))
Nv     = np.array(N_VALUES)[valid].astype(float)
em_n   = error_n_all[valid].mean(1)
es_n   = error_n_all[valid].std(1)

ax.errorbar(Nv, em_n, yerr=es_n, fmt='o-', color='steelblue',
            capsize=3, lw=1.6, ms=5, label='$|\\hat w_F^{(n)} - w_F^{\\rm ref}|$')

# O(1/sqrt(n)) reference anchored at largest n
ref_n = em_n[-1] * np.sqrt(Nv[-1]/Nv)
ax.plot(Nv, ref_n, 'k--', lw=1.3,
        label=r'$O(1/\sqrt{n})$ reference')

# Annotate slope
ax.text(0.35, 0.72,
        f'Log-log slope\n= {coeffs[0]:.2f}\n(expect $-0.5$)',
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Training samples $n$', fontsize=10)
ax.set_ylabel(r'$|\hat w_F^{(n)} - w_F^{\rm ref}|$', fontsize=10)
ax.set_title('(b) Data convergence\n'
             r'(Corollary 3.4: $\varepsilon_n \to 0$)', fontsize=9)
ax.legend(fontsize=8)

# ── Panel (c): Structured approximations ─────────────────────────
ax = axes[2]
methods = ['Full Fisher', 'Diagonal', 'Score UB\n$\\sqrt{\\rm Tr}(G)$']
means   = [wF_full_all.mean(), wF_diag_all.mean(), wF_score_all.mean()]
stds    = [wF_full_all.std(),  wF_diag_all.std(),  wF_score_all.std()]
colors  = ['steelblue', 'darkorange', 'tomato']
x_pos   = np.arange(3)

bars = ax.bar(x_pos, means, yerr=stds, color=colors, alpha=0.75,
              capsize=5, error_kw={'lw':1.5})
ax.set_xticks(x_pos); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel(r'$\hat w_F(B_2^d;\hat\theta)$', fontsize=10)
ax.set_title('(c) Structured Fisher approximations\n'
             '(Theorem 3.3: stability under metric error)', fontsize=9)
ax.set_ylim(0, max(means)*1.25)

# Annotate error percentages
err_diag  = abs(wF_diag_all-wF_full_all).mean()/wF_full_all.mean()*100
err_score = abs(wF_score_all-wF_full_all).mean()/wF_full_all.mean()*100
ax.text(1, means[1]+stds[1]+0.01,
        f'+{err_diag:.1f}%', ha='center', fontsize=8.5, color='darkorange')
ax.text(2, means[2]+stds[2]+0.01,
        f'+{err_score:.1f}%', ha='center', fontsize=8.5, color='tomato')
ax.axhline(means[0], ls=':', color='steelblue', lw=1.2, alpha=0.7)

fig.suptitle(
    'Experiment 2: Estimator Accuracy and Stability\n'
    r'Binary Logistic, MNIST 0 vs 1, $n=10{,}000$, $d=784$, '
    f'$\\lambda={LAM_FIX}$, {N_SEEDS} seeds',
    fontsize=10, y=1.02)

for fmt in ('pdf','png'):
    plt.savefig(f'results/exp2_figure.{fmt}',
                bbox_inches='tight', dpi=150 if fmt=='png' else None)

print("\nSaved: results/exp2_figure.png  .pdf")
print("Saved: results/exp2_summary.txt")
print("Saved: results/exp2_results.npz")
print("\nDone. Send exp2_figure.png and exp2_summary.txt.")
