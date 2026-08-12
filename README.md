# Fisher Width Complexity — Experiments

Reproducibility code for the paper:

> **Fisher Width: A Geometric Measure of Complexity on Statistical Manifolds**  
> Vu Khac Ky, FPT University, 2026.

---

## Structure

```
experiments/
├── exp1_three_models.py        # Experiment 1: Fisher width across model classes
├── exp2_estimator_accuracy.py  # Experiment 2: Estimator accuracy and stability
├── exp3_generalization.py      # Experiment 3: Generalization bound scaling
└── results/                    # Created automatically on first run (not tracked)
```

---

## Requirements

```bash
pip install numpy scipy matplotlib scikit-learn
```

Python 3.9+.

---

## Data

All experiments use MNIST. Download the raw files into `mnist_data/`:

```bash
mkdir mnist_data && cd mnist_data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

Place `mnist_data/` in the same directory as the scripts before running.

---

## Experiments

### Experiment 1 — Fisher Width in Trained Models

```bash
python exp1_three_models.py
```

Estimates $\hat{w}_F(B_2^d;\,\hat\theta)$ across regularization strengths $\lambda$
for three model classes, and validates the spectral sandwich bounds.

| Model | Description | Fisher matrix |
|-------|-------------|---------------|
| A | Binary logistic regression, MNIST 0 vs 1, $d=784$ | Full |
| B | 10-class softmax, $d=7840$ | Diagonal |
| C | Ridge regression, $d=784$ | Analytic (compared to MC) |

Key parameters at the top of the script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_TRAIN` | 10 000 | Training samples |
| `N_SEEDS` | 10 | Seeds for error bars |
| `B_MC_A/C` | 5 000 | MC samples (Models A, C) |
| `B_MC_B` | 10 000 | MC samples (Model B) |
| `LAMBDAS` | `[0, 1e-4, ..., 5]` | Regularization sweep |

Output: `results/exp1_figure.pdf`, `results/exp1_summary.txt`  
Runtime: ~15 minutes.

---

### Experiment 2 — Estimator Accuracy and Stability

```bash
python exp2_estimator_accuracy.py
```

Validates the error bounds for the three Fisher width estimators
on binary logistic regression ($d=784$, MNIST 0 vs 1).

Three sub-experiments:

- **2.1 Rank-$k$ approximation.** Plots $|\hat{w}_F^{(k)} - \hat{w}_F|$ vs
  $\sqrt{\lambda_{k+1}(\hat{G})}$, validating the bound from the paper.
- **2.2 Data convergence.** Plots Fisher width error vs $n$.
  Log-log slope expected near $-0.5$ (observed: $-0.598$).
- **2.3 Structured approximations.** Compares full Fisher, diagonal, and score norm.

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_TRAIN_REF` | 10 000 | Reference sample size for ground truth |
| `N_SEEDS` | 10 | Seeds for error bars |
| `B_MC` | 5 000 | MC samples per width estimate |
| `RANKS` | `[1, 2, 5, ..., 392]` | Rank-$k$ values for sub-exp 2.1 |
| `N_VALUES` | `[100, ..., 10 000]` | Sample sizes for sub-exp 2.2 |

Output: `results/exp2_figure.pdf`, `results/exp2_summary.txt`  
Runtime: ~20 minutes.

---

### Experiment 3 — Fisher Width and Generalization

```bash
python exp3_generalization.py
```

Validates the generalization bound
$\mathbb{E}[\sup_\theta |R(\theta) - \hat{R}_n(\theta)|] \leq 2L\,\hat{w}_F(T)/\sqrt{n}$
on 10-class softmax, MNIST, $d=784$.

Three panels:

- **(a)** Gen gap vs $1/\sqrt{n}$: linear fits with $R^2 > 0.93$ for all $\lambda$.
- **(b)** $\hat{w}_F$ vs $n$: Fisher width across training sizes.
- **(c)** Bound check: 0/28 violations of $CL \cdot \hat{w}_F/\sqrt{n}$ ($CL = 2.67$).

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_SEEDS` | 10 | Seeds for error bars |
| `B_MC` | 5 000 | MC samples per width estimate |
| `N_TEST` | 5 000 | Test set size for generalization gap |
| `LAMBDAS` | `[1e-3, 1e-2, 1e-1, 1.0]` | Regularization values |
| `N_VALUES` | `[200, ..., 20 000]` | Training sizes |
| `USE_MULTICLASS` | `True` | `True` = 10-class softmax, `False` = binary |

Output: `results/exp3_figure.pdf`, `results/exp3_summary_final.txt`  
Runtime: ~30 minutes. Requires ~8 GB RAM.

---

## Running All Experiments

From the `experiments/` directory with `mnist_data/` in place:

```bash
python exp1_three_models.py
python exp2_estimator_accuracy.py
python exp3_generalization.py
```

Figures and summary statistics are written to `results/`, which is
created automatically and not tracked in the repository.

---

## Notes

- All scripts use `MASTER_SEED = 42`; individual seeds are `MASTER_SEED + i`
  for `i in range(N_SEEDS)`.
- Model B uses diagonal Fisher because the full $7840 \times 7840$ matrix
  is infeasible. The diagonal approximation has $< 0.1\%$ error vs the
  score upper bound in practice.
- Experiment 3 saves intermediate results to `results/exp3_results.npz`
  for inspection without re-running the full computation.
