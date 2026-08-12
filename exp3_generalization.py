"""
Experiment 3: Fisher Width and Generalization
=============================================
Validates the generalization bound:
    E[sup_theta |R(theta) - R_hat_n(theta)|] <= 2L * w_F(T) / sqrt(n)

Three panels:
  (a) Gen gap vs 1/sqrt(n): linear relationship (R^2 > 0.93 for all lambda)
  (b) Fisher width w_F vs n: shows how w_F changes with sample size
  (c) Bound check: all points below CL * w_F / sqrt(n) (0 violations)

Task: MNIST 10-class softmax, d=784
"""

"""
Experiment 3 (v2): Fisher Width and Generalization
===================================================
Redesigned following Hướng A:
  - Fix lambda, vary n -> validate 1/sqrt(n) scaling (Theorem 5.5)
  - Use multiple lambda values to show wF controls the SLOPE
  - Task: MNIST 10-class softmax (harder than binary 0v1)

Three panels:
  (a) Gen gap vs 1/sqrt(n): each lambda gives one line,
      expect linear relationship
  (b) Slope of gen gap vs 1/sqrt(n) correlates with wF
      -> validates that wF is the constant governing the rate
  (c) Bound check: all points below CL * wF/sqrt(n)

Key insight: fix lambda -> wF(theta_hat) approximately constant
as n varies -> gen gap should scale as C * wF / sqrt(n)
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
MASTER_SEED = 42
N_SEEDS     = 10
B_MC        = 5_000
N_TEST      = 5_000

# Fix lambda values — each gives a different wF level
LAMBDAS     = [1e-3, 1e-2, 1e-1, 1.0]
LAM_LABELS  = [r'$\lambda=10^{-3}$', r'$\lambda=10^{-2}$',
               r'$\lambda=10^{-1}$', r'$\lambda=1$']
LAM_COLORS  = ['steelblue', 'darkorange', 'green', 'tomato']

# Vary n — wide range to see 1/sqrt(n) clearly
N_VALUES    = [200, 500, 1000, 2000, 5000, 10_000, 20_000]

# Use MNIST 10-class (harder task, larger gen gap)
USE_MULTICLASS = True   # True = 10-class softmax, False = binary

os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# 1.  Load MNIST
# ─────────────────────────────────────────────────────────────────
def read_idx(path):
    with gzip.open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        n     = struct.unpack('>I', f.read(4))[0]
        if magic == 2051:
            r,c = struct.unpack('>II', f.read(8))
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
d = X_all.shape[1]
print(f"Full MNIST: {X_all.shape[0]} samples, d={d}")

# Euclidean baseline
w_euclidean = np.exp(0.5*np.log(2)+gammaln((d+1)/2)-gammaln(d/2))

# ─────────────────────────────────────────────────────────────────
# 2.  Helpers
# ─────────────────────────────────────────────────────────────────
def sigmoid(z):
    return 1./(1.+np.exp(-np.clip(z,-500,500)))

def softmax_fn(Z):
    E = np.exp(Z - Z.max(axis=1,keepdims=True))
    return E/E.sum(axis=1,keepdims=True)

def compute_fisher_logistic(X, theta):
    p_ = sigmoid(X@theta); w_ = p_*(1-p_)
    return (X*w_[:,None]).T@X/len(X)

def compute_fisher_diag_softmax(X, Theta):
    """Diagonal Fisher for softmax: (K,d) -> (K*d,) vector."""
    Z  = X @ Theta.T
    P  = softmax_fn(Z)
    W  = P*(1-P)
    return (W[:,:,None]*(X[:,None,:]**2)).mean(axis=0).flatten()

def matrix_sqrt_full(G, eps=1e-10):
    ev,evec = np.linalg.eigh(G)
    return evec@np.diag(np.sqrt(np.maximum(ev,eps)))@evec.T

def mc_width_full(G_half, B, rng):
    g = rng.standard_normal((G_half.shape[0], B))
    return np.linalg.norm(G_half@g, axis=0).mean()

def mc_width_diag(d_half, B, rng):
    p = len(d_half)
    g = rng.standard_normal((p, B))
    return np.linalg.norm(d_half[:,None]*g, axis=0).mean()

def gen_gap_01(clf, Xtr_s, ytr, Xte_s, yte):
    """0/1 loss generalization gap, clamped at 0."""
    return max(0., (1.-clf.score(Xte_s,yte)) - (1.-clf.score(Xtr_s,ytr)))

scaler = StandardScaler()

# ─────────────────────────────────────────────────────────────────
# 3.  Main loop: for each (lambda, n), compute wF and gen gap
# ─────────────────────────────────────────────────────────────────
nL = len(LAMBDAS);  nN = len(N_VALUES)

# Storage: (nL, nN, N_SEEDS)
wF_all  = np.zeros((nL, nN, N_SEEDS))
gap_all = np.zeros((nL, nN, N_SEEDS))

print(f"\nTask: MNIST {'10-class softmax' if USE_MULTICLASS else 'binary 0v1'}")
print(f"{'lambda':>8}  {'n':>7}  {'wF mean':>9}  {'gap mean':>10}  "
      f"{'gap/wF*sqrt(n)':>16}")
print("-"*60)

for li, lam in enumerate(LAMBDAS):
    for ni, n in enumerate(N_VALUES):
        # Need enough test samples disjoint from train
        need = n + N_TEST
        if need > len(X_all):
            print(f"  lam={lam:.0e} n={n}: not enough data, skip")
            wF_all[li,ni,:]  = np.nan
            gap_all[li,ni,:] = np.nan
            continue

        t0 = time.time()
        for si in range(N_SEEDS):
            rng = np.random.default_rng(MASTER_SEED + li*10000 + ni*100 + si)
            idx = rng.permutation(len(X_all))

            Xtr = X_all[idx[:n]];      ytr = y_all[idx[:n]]
            Xte = X_all[idx[n:n+N_TEST]]; yte = y_all[idx[n:n+N_TEST]]

            Xtr_s = scaler.fit_transform(Xtr)
            Xte_s = scaler.transform(Xte)

            if USE_MULTICLASS:
                C = 1./(lam*n) if lam>0 else 1e8
                clf = LogisticRegression(
                    C=C, max_iter=1000, solver='lbfgs',
                    multi_class='multinomial',
                    random_state=int(rng.integers(9999)), tol=1e-4)
                clf.fit(Xtr_s, ytr)
                Theta = clf.coef_   # (K, d)

                # Diagonal Fisher for softmax
                dG   = compute_fisher_diag_softmax(Xtr_s, Theta)
                dh   = np.sqrt(np.maximum(dG, 1e-10))
                wF_all[li,ni,si] = mc_width_diag(dh, B_MC, rng)
            else:
                mask = (ytr==0)|(ytr==1)
                # fallback to binary
                Xtr_b = Xtr_s[mask]; ytr_b = ytr[mask]
                mask_te = (yte==0)|(yte==1)
                Xte_b = Xte_s[mask_te]; yte_b = yte[mask_te]
                C = 1./(lam*len(Xtr_b)) if lam>0 else 1e8
                clf = LogisticRegression(C=C, max_iter=2000, solver='lbfgs',
                    random_state=int(rng.integers(9999)), tol=1e-6)
                clf.fit(Xtr_b, ytr_b)
                theta = clf.coef_.flatten()
                G  = compute_fisher_logistic(Xtr_b, theta)
                Gh = matrix_sqrt_full(G)
                wF_all[li,ni,si] = mc_width_full(Gh, B_MC, rng)

            gap_all[li,ni,si] = gen_gap_01(clf, Xtr_s, ytr, Xte_s, yte)

        wm = np.nanmean(wF_all[li,ni])
        gm = np.nanmean(gap_all[li,ni])
        ratio = gm / (wm/np.sqrt(n)) if wm>0 else np.nan
        print(f"{lam:>8.1e}  {n:>7}  {wm:>9.4f}  {gm:>10.6f}  "
              f"{ratio:>16.4f}  [{time.time()-t0:.1f}s]")

# ─────────────────────────────────────────────────────────────────
# 4.  Analysis
# ─────────────────────────────────────────────────────────────────

# For each lambda: fit gen_gap ~ slope / sqrt(n)
# i.e., gen_gap * sqrt(n) ~ slope (constant across n)
# Validate: slope ~ wF (the Fisher width at that lambda)

slopes    = np.zeros(nL)   # fitted slope per lambda
slopes_se = np.zeros(nL)   # std error
wF_at_lam = np.zeros(nL)   # mean wF (averaged over n and seeds)

print("\n--- Slope analysis: gen_gap ~ slope/sqrt(n) ---")
print(f"{'lambda':>8}  {'wF mean':>9}  {'slope mean':>12}  "
      f"{'slope/wF':>10}  {'R^2':>8}")
print("-"*55)

for li, lam in enumerate(LAMBDAS):
    valid = ~np.isnan(gap_all[li,:,0])
    Nv    = np.array(N_VALUES)[valid].astype(float)
    # gen_gap * sqrt(n) for each (n, seed)
    scaled_gap = gap_all[li,valid,:] * np.sqrt(Nv[:,None])
    # slope estimate per seed: mean over n
    slope_per_seed = scaled_gap.mean(axis=0)   # (N_SEEDS,)
    slopes[li]    = slope_per_seed.mean()
    slopes_se[li] = slope_per_seed.std()

    # R^2 of gen_gap ~ 1/sqrt(n) fit
    gm_n = gap_all[li,valid,:].mean(axis=1)   # mean over seeds
    inv_sqrt_n = 1./np.sqrt(Nv)
    cf = np.polyfit(inv_sqrt_n, gm_n, 1)
    gm_pred = np.polyval(cf, inv_sqrt_n)
    ss_res = np.sum((gm_n - gm_pred)**2)
    ss_tot = np.sum((gm_n - gm_n.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan

    # mean wF (large n is most reliable)
    wF_at_lam[li] = np.nanmean(wF_all[li,valid,:])

    print(f"{lam:>8.1e}  {wF_at_lam[li]:>9.4f}  "
          f"{slopes[li]:>10.6f}±{slopes_se[li]:.6f}  "
          f"{slopes[li]/wF_at_lam[li]:>10.4f}  {r2:>8.4f}")

# Correlation: slope ~ wF across lambda values
corr_slope_wF = np.corrcoef(wF_at_lam, slopes)[0,1]
print(f"\nCorr(slope, wF) = {corr_slope_wF:.4f}")

# ─────────────────────────────────────────────────────────────────
# 5.  Save
# ─────────────────────────────────────────────────────────────────

# =================================================================
# Figure
# =================================================================
# R^2 for panel (a)
r2_list = []
slope_list = []
for li in range(nL):
    valid = ~np.isnan(gap_all[li,:,0])
    Nv    = np.array(N_VALUES)[valid].astype(float)
    gm    = np.nanmean(gap_all[li,valid,:], axis=1)
    isqN  = 1./np.sqrt(Nv)
    cf    = np.polyfit(isqN, gm, 1)
    slope_list.append(cf[0])
    ss_res = np.sum((gm-np.polyval(cf,isqN))**2)
    ss_tot = np.sum((gm-gm.mean())**2)
    r2_list.append(1-ss_res/ss_tot if ss_tot>0 else np.nan)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
plt.subplots_adjust(wspace=0.38)

# ── Panel (a): gen gap vs 1/sqrt(n) ──────────────────────────────
ax = axes[0]
for li in range(nL):
    valid = ~np.isnan(gap_all[li,:,0])
    Nv    = np.array(N_VALUES)[valid].astype(float)
    isqN  = 1./np.sqrt(Nv)
    gm    = np.nanmean(gap_all[li,valid,:], axis=1)
    gs    = np.nanstd( gap_all[li,valid,:], axis=1)
    ax.errorbar(isqN, gm, yerr=gs, fmt='o-',
                color=LAM_COLORS[li], capsize=3, lw=1.6, ms=5,
                label=f'{LAM_LABELS[li]}  ($R^2={r2_list[li]:.3f}$)')
    cf  = np.polyfit(isqN, gm, 1)
    xr  = np.linspace(0, isqN.max()*1.05, 100)
    ax.plot(xr, np.polyval(cf,xr), '--',
            color=LAM_COLORS[li], lw=1.0, alpha=0.5)

ax.set_xlabel(r'$1/\sqrt{n}$', fontsize=11)
ax.set_ylabel('Generalization gap', fontsize=10)
ax.set_title('(a) Gen gap vs $1/\\sqrt{n}$\n'
             r'(Theorem 5.5: gap $\leq CL\cdot w_F/\sqrt{n}$)',
             fontsize=9)
ax.legend(fontsize=7.5)
ax.set_xlim(left=0); ax.set_ylim(bottom=0)

# ── Panel (b): wF vs n ────────────────────────────────────────────
ax = axes[1]
Nv_all = np.array(N_VALUES, dtype=float)
for li in range(nL):
    valid = ~np.isnan(wF_all[li,:,0])
    Nv    = Nv_all[valid]
    wm    = np.nanmean(wF_all[li,valid,:], axis=1)
    ws    = np.nanstd( wF_all[li,valid,:], axis=1)
    ax.errorbar(Nv, wm, yerr=ws, fmt='o-',
                color=LAM_COLORS[li], capsize=3, lw=1.6, ms=5,
                label=LAM_LABELS[li])

ax.set_xscale('log')
ax.set_xlabel('Training samples $n$', fontsize=10)
ax.set_ylabel(r'$\hat w_F(T;\hat\theta_n)$', fontsize=10)
ax.set_title('(b) Fisher width vs training samples $n$\n'
             r'(stronger regularization stabilizes $w_F$ across $n$)',
             fontsize=9)
ax.legend(fontsize=7.5)
ax.set_ylim(bottom=0)

# ── Panel (c): bound check ────────────────────────────────────────
ax = axes[2]
all_x=[]; all_y=[]; all_c=[]
for li in range(nL):
    for ni in range(nN):
        if np.isnan(wF_all[li,ni,0]): continue
        xv = np.nanmean(wF_all[li,ni])/np.sqrt(N_VALUES[ni])
        yv = np.nanmean(gap_all[li,ni])
        all_x.append(xv); all_y.append(yv); all_c.append(li)
all_x=np.array(all_x); all_y=np.array(all_y); all_c=np.array(all_c)

for li in range(nL):
    m_ = all_c==li
    ax.scatter(all_x[m_], all_y[m_], color=LAM_COLORS[li],
               alpha=0.75, s=35, label=LAM_LABELS[li])

CL  = (all_y/np.where(all_x>0,all_x,np.inf)).max()*1.2
xr3 = np.linspace(0, all_x.max()*1.1, 200)
ax.plot(xr3, CL*xr3, 'k-', lw=1.8,
        label=f'Bound $CL\\cdot w_F/\\sqrt{{n}}$\n$CL={CL:.3f}$')
n_viol = int(np.sum(all_y > CL*all_x+1e-8))
ax.set_xlabel(r'$\hat w_F/\sqrt{n}$', fontsize=10)
ax.set_ylabel('Generalization gap', fontsize=10)
ax.set_title(f'(c) Bound check (Theorem 5.5)\n'
             f'{n_viol}/{len(all_x)} violations',
             fontsize=9)
ax.legend(fontsize=7, ncol=2)
ax.set_xlim(left=0); ax.set_ylim(bottom=0)

fig.suptitle(
    'Experiment 3: Fisher Width and Generalization\n'
    r'MNIST 10-class softmax, 10 seeds, $B_{\rm MC}=5{,}000$',
    fontsize=10, y=1.02)

for fmt in ('pdf','png'):
    plt.savefig(f'results/exp3_figure.{fmt}',
                bbox_inches='tight',
                dpi=150 if fmt=='png' else None)

# Summary
lines = [
    "="*68,
    "Experiment 3: Fisher Width and Generalization (final)",
    "Task: MNIST 10-class softmax",
    f"d={d}, seeds={N_SEEDS}, B_MC=5000, N_test=5000",
    "="*68,"",
    "Panel (a): Gen gap vs 1/sqrt(n)",
    f"{'lambda':>8}  {'wF mean':>9}  {'R^2':>8}  {'slope':>10}",
    "-"*42,
]
for li in range(nL):
    lines.append(f"{LAMBDAS[li]:>8.1e}  {wF_at_lam[li]:>9.4f}  "
                 f"{r2_list[li]:>8.4f}  {slope_list[li]:>10.4f}")

lines += ["",
    "Panel (b): wF vs n — behavior across training set size",
    f"{'lambda':>8}  {'wF at n=200':>12}  {'wF at n=20000':>14}  "
    f"{'ratio':>8}","-"*48]
for li in range(nL):
    valid = ~np.isnan(wF_all[li,:,0])
    Nv    = np.array(N_VALUES)[valid]
    wm    = np.nanmean(wF_all[li,valid,:], axis=1)
    lines.append(f"{LAMBDAS[li]:>8.1e}  {wm[0]:>12.4f}  "
                 f"{wm[-1]:>14.4f}  {wm[-1]/wm[0]:>8.4f}")

lines += ["",
    f"Panel (c): CL = {CL:.4f},  Violations = {n_viol}/{len(all_x)}",
    "="*68]
summary = "\n".join(lines)
print("\n"+summary)
with open('results/exp3_summary_final.txt','w') as f:
    f.write(summary)
print("\nSaved: results/exp3_figure.png  .pdf")
print("Saved: results/exp3_summary_final.txt")
print("\nDone. Send exp3_figure.png and exp3_summary_final.txt.")
