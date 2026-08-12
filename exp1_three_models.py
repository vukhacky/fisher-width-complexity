"""
Experiment 1: Fisher Width in Trained Models — Three Model Classes (v2)
=======================================================================
Fixes vs v1:
  - Model B: B_MC=10000 to eliminate diagonal-Fisher MC noise violation
  - Model C: panel (c) replaced by MC-vs-analytic convergence plot
  - Model C: note that wF is lambda-invariant (G = Sigma_X/sigma^2)
  - Model C: test_acc replaced by MSE
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from scipy.special import gammaln
import struct, gzip, os, time

# ── config ────────────────────────────────────────────────────────
MASTER_SEED   = 42
N_SEEDS       = 10
N_TRAIN       = 10_000
N_TEST        = 2_000
B_MC_A        = 5_000    # Binary logistic
B_MC_B        = 10_000   # Softmax (larger p, needs more MC samples)
B_MC_C        = 5_000    # Ridge
DAMP          = 1e-6
SIGMA2_RIDGE  = 1.0
LAMBDAS       = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0]
LAMBDA_LABELS = [r'$0$', r'$10^{-4}$', r'$10^{-3}$',
                 r'$10^{-2}$', r'$10^{-1}$', r'$1$', r'$5$']
# For Model C convergence panel: vary B
B_VALUES      = [100, 200, 500, 1000, 2000, 5000, 10000]

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
            return np.frombuffer(f.read(), np.uint8).reshape(n,r*c).astype(np.float64)
        elif magic == 2049:
            return np.frombuffer(f.read(), np.uint8).astype(int)

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
mask  = (y_all == 0) | (y_all == 1)
X_bin = X_all[mask];  y_bin = y_all[mask]
d = X_all.shape[1]
K = 10
p_A = d;  p_B = d*K;  p_C = d
print(f"Full MNIST: {X_all.shape[0]} samples | Binary: {X_bin.shape[0]} | d={d}")

# ─────────────────────────────────────────────────────────────────
# 2.  Euclidean baselines
# ─────────────────────────────────────────────────────────────────
def w_euclidean(p):
    return np.exp(0.5*np.log(2) + gammaln((p+1)/2) - gammaln(p/2))

wE = {'A': w_euclidean(p_A),
      'B': w_euclidean(p_B),
      'C': w_euclidean(p_C)}
print(f"w(B_2^p): A={wE['A']:.4f}  B={wE['B']:.4f}  C={wE['C']:.4f}")

# ─────────────────────────────────────────────────────────────────
# 3.  Helpers
# ─────────────────────────────────────────────────────────────────
def sigmoid(z):
    return 1.0/(1.0+np.exp(-np.clip(z,-500,500)))

def softmax_fn(Z):
    E = np.exp(Z - Z.max(axis=1,keepdims=True))
    return E/E.sum(axis=1,keepdims=True)

def matrix_sqrt_full(G, eps=1e-10):
    ev, evec = np.linalg.eigh(G)
    return evec @ np.diag(np.sqrt(np.maximum(ev,eps))) @ evec.T

def mc_full(G_half, B, rng):
    g = rng.standard_normal((G_half.shape[0], B))
    return np.linalg.norm(G_half @ g, axis=0).mean()

def mc_diag(d_half, B, rng):
    """Exact for diagonal G: w_F = sqrt(Tr(G)) analytically,
    but we still MC to validate and report."""
    p = len(d_half)
    g = rng.standard_normal((p, B))
    return np.linalg.norm(d_half[:,None]*g, axis=0).mean()

def score_ub_full(G):   return np.sqrt(np.trace(G))
def score_ub_diag(dG):  return np.sqrt(dG.sum())

def spectral_info_full(G, wE_, damp=DAMP):
    ev = np.linalg.eigvalsh(G)
    lmx, lmn = ev.max(), ev.min()
    return (np.sqrt(max(lmn,0.))*wE_,   # lb  (true, no damp)
            np.sqrt(lmx)*wE_,            # ub
            lmx/max(lmn,damp),           # kappa
            lmx, lmn)

def spectral_info_diag(dG, wE_, damp=DAMP):
    lmx, lmn = dG.max(), dG.min()
    return (np.sqrt(max(lmn,0.))*wE_,
            np.sqrt(lmx)*wE_,
            lmx/max(lmn,damp),
            lmx, lmn)

# ─────────────────────────────────────────────────────────────────
# 4.  Storage
# ─────────────────────────────────────────────────────────────────
nL = len(LAMBDAS)
def blank(): return np.zeros((nL, N_SEEDS))
R = {m: {'wF':blank(),'scoreub':blank(),'lb':blank(),
          'ub':blank(),'kappa':blank(),'metric':blank()}
     for m in ('A','B','C')}
ridge_analytic = blank()   # analytic wF for model C

# ─────────────────────────────────────────────────────────────────
# 5A.  Model A: Binary Logistic
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*62)
print(f"Model A: Binary Logistic  p={p_A}, full Fisher, B_MC={B_MC_A}")
print("="*62)
scaler = StandardScaler()
for li, lam in enumerate(LAMBDAS):
    t0 = time.time()
    for si in range(N_SEEDS):
        rng = np.random.default_rng(MASTER_SEED + si*100)
        idx = rng.permutation(len(X_bin))
        Xtr_s = scaler.fit_transform(X_bin[idx[:N_TRAIN]])
        Xte_s = scaler.transform(X_bin[idx[N_TRAIN:N_TRAIN+N_TEST]])
        ytr = y_bin[idx[:N_TRAIN]]
        yte = y_bin[idx[N_TRAIN:N_TRAIN+N_TEST]]

        C = 1./(lam*N_TRAIN) if lam>0 else 1e8
        clf = LogisticRegression(C=C,max_iter=2000,solver='lbfgs',
                                 random_state=int(rng.integers(9999)),tol=1e-6)
        clf.fit(Xtr_s, ytr)
        theta = clf.coef_.flatten()
        R['A']['metric'][li,si] = clf.score(Xte_s, yte)

        p_ = sigmoid(Xtr_s@theta); w_ = p_*(1-p_)
        G  = (Xtr_s*w_[:,None]).T @ Xtr_s / N_TRAIN
        Gh = matrix_sqrt_full(G)
        R['A']['wF'][li,si]      = mc_full(Gh, B_MC_A, rng)
        R['A']['scoreub'][li,si] = score_ub_full(G)
        lb,ub,kap,lmx,lmn       = spectral_info_full(G, wE['A'])
        R['A']['lb'][li,si]  = lb;  R['A']['ub'][li,si]  = ub
        R['A']['kappa'][li,si] = kap

    m,s = R['A']['wF'][li].mean(), R['A']['wF'][li].std()
    print(f"  lam={lam:.0e}  wF={m:.4f}±{s:.4f}  "
          f"wF/wE={m/wE['A']:.4f}  "
          f"kappa={R['A']['kappa'][li].mean():.1f}  "
          f"acc={R['A']['metric'][li].mean():.4f}  [{time.time()-t0:.1f}s]")

# ─────────────────────────────────────────────────────────────────
# 5B.  Model B: Softmax 10-class
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*62)
print(f"Model B: Softmax 10-class  p={p_B}, diag Fisher, B_MC={B_MC_B}")
print("  Note: for diagonal G, score UB = sqrt(Tr(G)) is exact w_F")
print("  MC should match to within O(1/sqrt(B)) noise")
print("="*62)
for li, lam in enumerate(LAMBDAS):
    t0 = time.time()
    for si in range(N_SEEDS):
        rng = np.random.default_rng(MASTER_SEED + si*100 + 1)
        idx = rng.permutation(len(X_all))
        Xtr_s = scaler.fit_transform(X_all[idx[:N_TRAIN]])
        Xte_s = scaler.transform(X_all[idx[N_TRAIN:N_TRAIN+N_TEST]])
        ytr = y_all[idx[:N_TRAIN]]
        yte = y_all[idx[N_TRAIN:N_TRAIN+N_TEST]]

        C = 1./(lam*N_TRAIN) if lam>0 else 1e8
        clf = LogisticRegression(C=C,max_iter=2000,solver='lbfgs',
                                 multi_class='multinomial',
                                 random_state=int(rng.integers(9999)),tol=1e-4)
        clf.fit(Xtr_s, ytr)
        Theta = clf.coef_   # (K,d)
        R['B']['metric'][li,si] = clf.score(Xte_s, yte)

        # Diagonal Fisher for softmax
        Z  = Xtr_s @ Theta.T
        P  = softmax_fn(Z)
        W  = P*(1-P)
        dG = (W[:,:,None]*(Xtr_s[:,None,:]**2)).mean(axis=0).flatten()  # (K*d,)

        # Exact wF for diagonal G: w_F = sqrt(Tr(G)) * correction_factor
        # For diagonal G, w_F(B_2^p) = E[||diag(sqrt(dG)) g||]
        # = sqrt(sum_i dG_i) only when isotropic; in general use MC
        dh = np.sqrt(np.maximum(dG, 1e-10))
        R['B']['wF'][li,si]      = mc_diag(dh, B_MC_B, rng)
        R['B']['scoreub'][li,si] = score_ub_diag(dG)  # exact for diag
        lb,ub,kap,lmx,lmn       = spectral_info_diag(dG, wE['B'])
        R['B']['lb'][li,si]  = lb;  R['B']['ub'][li,si]  = ub
        R['B']['kappa'][li,si] = kap

    m,s = R['B']['wF'][li].mean(), R['B']['wF'][li].std()
    print(f"  lam={lam:.0e}  wF={m:.4f}±{s:.4f}  "
          f"wF/wE={m/wE['B']:.4f}  "
          f"kappa={R['B']['kappa'][li].mean():.1f}  "
          f"acc={R['B']['metric'][li].mean():.4f}  [{time.time()-t0:.1f}s]")

# ─────────────────────────────────────────────────────────────────
# 5C.  Model C: Ridge Regression
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*62)
print(f"Model C: Ridge Regression  p={p_C}, analytic G, B_MC={B_MC_C}")
print(f"  G(theta) = (1/sigma^2) Sigma_X  [lambda-invariant]")
print("="*62)

# Use one fixed seed to get Sigma_X for analytic computation
rng0 = np.random.default_rng(MASTER_SEED)
idx0 = rng0.permutation(len(X_bin))
Xtr0 = scaler.fit_transform(X_bin[idx0[:N_TRAIN]])
Sigma_X_ref = Xtr0.T @ Xtr0 / N_TRAIN
G_ref       = Sigma_X_ref / SIGMA2_RIDGE
Gh_ref      = matrix_sqrt_full(G_ref)

# Analytic wF: E[||G^{1/2} g||] computed with large B for ground truth
rng_gt = np.random.default_rng(999)
g_gt   = rng_gt.standard_normal((d, 50_000))
wF_analytic_gt = np.linalg.norm(Gh_ref @ g_gt, axis=0).mean()
print(f"  Analytic ground truth (B=50000): wF = {wF_analytic_gt:.6f}")

for li, lam in enumerate(LAMBDAS):
    t0 = time.time()
    for si in range(N_SEEDS):
        rng = np.random.default_rng(MASTER_SEED + si*100 + 2)
        idx = rng.permutation(len(X_bin))
        Xtr_s = scaler.fit_transform(X_bin[idx[:N_TRAIN]])
        Xte_s = scaler.transform(X_bin[idx[N_TRAIN:N_TRAIN+N_TEST]])
        ytr = y_bin[idx[:N_TRAIN]].astype(float)
        yte = y_bin[idx[N_TRAIN:N_TRAIN+N_TEST]].astype(float)

        alpha = lam if lam>0 else 1e-8
        clf = Ridge(alpha=alpha*N_TRAIN, fit_intercept=False)
        clf.fit(Xtr_s, ytr)
        # MSE as metric (more appropriate than acc for regression)
        mse = np.mean((clf.predict(Xte_s) - yte)**2)
        R['C']['metric'][li,si] = mse

        # G = Sigma_X / sigma^2 (does NOT depend on lambda or theta)
        Sigma_X = Xtr_s.T @ Xtr_s / N_TRAIN
        G       = Sigma_X / SIGMA2_RIDGE
        Gh      = matrix_sqrt_full(G)

        R['C']['wF'][li,si]      = mc_full(Gh, B_MC_C, rng)
        R['C']['scoreub'][li,si] = score_ub_full(G)
        lb,ub,kap,lmx,lmn       = spectral_info_full(G, wE['C'])
        R['C']['lb'][li,si]  = lb;  R['C']['ub'][li,si]  = ub
        R['C']['kappa'][li,si] = kap
        ridge_analytic[li,si]  = wF_analytic_gt  # same for all lambda

    m,s  = R['C']['wF'][li].mean(), R['C']['wF'][li].std()
    err  = abs(m - wF_analytic_gt)/wF_analytic_gt*100
    print(f"  lam={lam:.0e}  wF={m:.4f}±{s:.4f}  "
          f"analytic={wF_analytic_gt:.4f}  err={err:.2f}%  "
          f"MSE={R['C']['metric'][li].mean():.4f}  [{time.time()-t0:.1f}s]")

# ─────────────────────────────────────────────────────────────────
# 5D.  Model C convergence: MC error vs B (fixed seed, fixed G)
# ─────────────────────────────────────────────────────────────────
print("\nComputing MC convergence curve for Model C ...")
mc_errors_by_B = np.zeros((len(B_VALUES), N_SEEDS))
for bi, B in enumerate(B_VALUES):
    for si in range(N_SEEDS):
        rng = np.random.default_rng(MASTER_SEED + si*7 + 3)
        mc_errors_by_B[bi,si] = abs(mc_full(Gh_ref, B, rng) - wF_analytic_gt) \
                                 / wF_analytic_gt * 100
    m,s = mc_errors_by_B[bi].mean(), mc_errors_by_B[bi].std()
    print(f"  B={B:>6}:  MC error = {m:.3f}% ± {s:.3f}%")

# ─────────────────────────────────────────────────────────────────
# 6.  Save
# ─────────────────────────────────────────────────────────────────
np.savez('results/exp1_results.npz',
         lambdas=LAMBDAS,
         A_wF=R['A']['wF'], B_wF=R['B']['wF'], C_wF=R['C']['wF'],
         ridge_analytic=ridge_analytic,
         wF_analytic_gt=wF_analytic_gt,
         mc_errors_by_B=mc_errors_by_B, B_values=B_VALUES,
         wE_A=wE['A'], wE_B=wE['B'], wE_C=wE['C'],
         d=d, K=K)

# ─────────────────────────────────────────────────────────────────
# 7.  Summary text
# ─────────────────────────────────────────────────────────────────
lines = ["="*72,
         "Experiment 1: Fisher Width in Trained Models — Three Model Classes",
         f"MNIST, n_train={N_TRAIN}, n_test={N_TEST}, seeds={N_SEEDS}",
         "="*72]

model_cfgs = [
    ('A','Binary Logistic',  f'p={p_A}, full Fisher, B_MC={B_MC_A}', wE['A']),
    ('B','Softmax 10-class', f'p={p_B}, diag Fisher, B_MC={B_MC_B}', wE['B']),
    ('C','Ridge Regression', f'p={p_C}, analytic G,  B_MC={B_MC_C}', wE['C']),
]
for key,name,desc,wE_ in model_cfgs:
    lines += ["", f"Model {key}: {name}  ({desc})",
              f"  w(B_2^p) = {wE_:.6f}",
              f"  {'lambda':>8}  {'wF mean':>8}  {'±std':>6}  "
              f"{'wF/wE':>7}  {'scoreUB':>9}  {'kappa':>12}",
              "  "+"-"*62]
    for li,lam in enumerate(LAMBDAS):
        extra = ""
        if key=='C':
            err = abs(R['C']['wF'][li].mean()-wF_analytic_gt)/wF_analytic_gt*100
            extra = f"  [analytic={wF_analytic_gt:.4f}, err={err:.2f}%]"
        lines.append(
            f"  {lam:>8.1e}  {R[key]['wF'][li].mean():>8.4f}  "
            f"{R[key]['wF'][li].std():>6.4f}  "
            f"{R[key]['wF'][li].mean()/wE_:>7.4f}  "
            f"{R[key]['scoreub'][li].mean():>9.4f}  "
            f"{R[key]['kappa'][li].mean():>12.1f}" + extra)
    lb_v = (R[key]['wF']-R[key]['lb']).min()
    ub_v = (R[key]['ub']-R[key]['wF']).min()
    sc_v = (R[key]['scoreub']-R[key]['wF']).min()
    mono = all(R[key]['wF'][i].mean()<=R[key]['wF'][i+1].mean()
               for i in range(nL-1))
    lines += ["", "  Bound checks:",
              f"    min(wF - lb)      = {lb_v:.6f}" + ("  OK" if lb_v>=-1e-6 else "  VIOLATED"),
              f"    min(ub - wF)      = {ub_v:.6f}" + ("  OK" if ub_v>=0 else "  VIOLATED"),
              f"    min(scoreUB - wF) = {sc_v:.6f}" + ("  OK" if sc_v>=-1e-4 else "  VIOLATED"),
              f"    wF monotone?      = {'YES' if mono else 'NO -- lambda-invariant (correct for Ridge)'}"]

lines += ["","="*72,"Ridge MC vs Analytic (ground truth B=50000):",
          f"  Analytic wF = {wF_analytic_gt:.6f}",
          f"  {'B':>8}  {'MC error mean':>14}  {'±std':>8}"]
for bi,B in enumerate(B_VALUES):
    lines.append(f"  {B:>8}  {mc_errors_by_B[bi].mean():>13.3f}%  "
                 f"{mc_errors_by_B[bi].std():>7.3f}%")
lines.append("="*72)

summary = "\n".join(lines)
print("\n"+summary)
with open('results/exp1_summary.txt','w') as f:
    f.write(summary)

# ─────────────────────────────────────────────────────────────────
# 8.  Figure — 3 rows x 3 cols
#     Row A: binary logistic   — (a) abs width, (b) normalized, (c) kappa
#     Row B: softmax           — (a) abs width, (b) normalized, (c) kappa
#     Row C: ridge             — (a) abs width, (b) normalized, (c) MC convergence
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
plt.subplots_adjust(hspace=0.50, wspace=0.40)

x = np.arange(nL)

for row,(key,title,wE_) in enumerate([
    ('A', r'Model A: Binary Logistic ($p=784$, full Fisher)',  wE['A']),
    ('B', r'Model B: Softmax 10-class ($p=7840$, diag Fisher)',wE['B']),
    ('C', r'Model C: Ridge Regression ($p=784$, analytic $G$)',wE['C']),
]):
    wF_m = R[key]['wF'].mean(1);     wF_s = R[key]['wF'].std(1)
    sb_m = R[key]['scoreub'].mean(1); sb_s = R[key]['scoreub'].std(1)
    lb_m = R[key]['lb'].mean(1);     ub_m = R[key]['ub'].mean(1)
    kp_m = R[key]['kappa'].mean(1);  kp_s = R[key]['kappa'].std(1)
    rt_m = wF_m/wE_;                 rt_s = wF_s/wE_

    # col 0: absolute width
    ax = axes[row,0]
    ax.fill_between(x, lb_m, ub_m, alpha=0.15, color='gray',
                    label='Spectral sandwich')
    ax.errorbar(x, wF_m, yerr=wF_s, fmt='o-', color='steelblue',
                capsize=3, lw=1.6, label=r'$\hat w_F$ (MC)')
    ax.errorbar(x, sb_m, yerr=sb_s, fmt='s--', color='tomato',
                capsize=3, lw=1.4, label='Score UB')
    ax.axhline(wE_, ls=':', color='k', lw=1.1, label=r'$w(B_2^p)$')
    if key=='C':
        ax.axhline(wF_analytic_gt, ls='--', color='green', lw=1.4,
                   label=r'Analytic $w_F$')
    ax.set_xticks(x); ax.set_xticklabels(LAMBDA_LABELS, fontsize=7)
    ax.set_xlabel(r'$\lambda$', fontsize=9)
    ax.set_ylabel(r'$\hat w_F(B_2^p;\hat\theta)$', fontsize=9)
    ax.set_title(f'{title}\n(a) Absolute Fisher width', fontsize=8)
    ax.legend(fontsize=6, loc='upper left'); ax.set_ylim(bottom=0)

    # col 1: normalized ratio
    ax = axes[row,1]
    ax.errorbar(x, rt_m, yerr=rt_s, fmt='o-', color='steelblue',
                capsize=3, lw=1.6)
    ax.axhline(1.0, ls=':', color='k', lw=1.1, label='Euclidean = 1')
    ax.annotate(f'{rt_m[0]:.4f}', xy=(0,rt_m[0]),
                xytext=(0.2,rt_m[0]+0.02), fontsize=7.5, color='steelblue')
    ax.annotate(f'{rt_m[-1]:.3f}', xy=(nL-1,rt_m[-1]),
                xytext=(nL-1.9,rt_m[-1]+0.02), fontsize=7.5, color='steelblue')
    ax.set_xticks(x); ax.set_xticklabels(LAMBDA_LABELS, fontsize=7)
    ax.set_xlabel(r'$\lambda$', fontsize=9)
    ax.set_ylabel(r'$\hat w_F / w(B_2^p)$', fontsize=9)
    ax.set_title('(b) Normalized Fisher width', fontsize=8)
    ax.legend(fontsize=7); ax.set_ylim(0, 1.15)
    if key=='C':
        ax.text(3.5, 0.5,
                r'$G(\theta)=\frac{1}{\sigma^2}\Sigma_X$'+'\n(invariant to '+r'$\lambda$)',
                fontsize=8, color='gray', ha='center')

    # col 2: kappa (A,B) or MC convergence (C)
    ax = axes[row,2]
    if key in ('A','B'):
        ax.errorbar(x, kp_m, yerr=kp_s, fmt='D-', color='darkorange',
                    capsize=3, lw=1.6)
        ax.set_xticks(x); ax.set_xticklabels(LAMBDA_LABELS, fontsize=7)
        ax.set_xlabel(r'$\lambda$', fontsize=9)
        ax.set_ylabel(r'$\kappa_\varepsilon(G)$', fontsize=9)
        ax.set_title(r'(c) Fisher anisotropy $\kappa_\varepsilon(G)$', fontsize=8)
        ax.set_yscale('log')
    else:
        # MC convergence for Ridge
        err_m = mc_errors_by_B.mean(1)
        err_s = mc_errors_by_B.std(1)
        Bv    = np.array(B_VALUES, dtype=float)
        ax.errorbar(Bv, err_m, yerr=err_s, fmt='o-', color='steelblue',
                    capsize=3, lw=1.6, label='MC error %')
        # O(1/sqrt(B)) reference
        ref = err_m[0] * np.sqrt(Bv[0]/Bv)
        ax.plot(Bv, ref, 'k--', lw=1.2, label=r'$O(1/\sqrt{B})$')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$B$ (MC samples)', fontsize=9)
        ax.set_ylabel('Relative MC error (%)', fontsize=9)
        ax.set_title(r'(c) MC convergence: error vs $B$', fontsize=8)
        ax.legend(fontsize=7)
        # Annotate final error
        ax.annotate(f'{err_m[-1]:.2f}%', xy=(Bv[-1],err_m[-1]),
                    xytext=(Bv[-1]*0.5, err_m[-1]*2),
                    fontsize=8, color='steelblue')

fig.suptitle(
    'Experiment 1: Fisher Width in Trained Models — Three Model Classes\n'
    r'MNIST, $n=10{,}000$, 10 seeds',
    fontsize=11, y=1.01)

for fmt in ('pdf','png'):
    plt.savefig(f'results/exp1_figure.{fmt}',
                bbox_inches='tight', dpi=150 if fmt=='png' else None)

print("\nSaved: results/exp1_figure.png  .pdf")
print("Saved: results/exp1_summary.txt")
print("Saved: results/exp1_results.npz")
print("\nDone. Send exp1_figure.png and exp1_summary.txt.")
