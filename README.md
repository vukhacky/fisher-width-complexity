# Fisher Width — final experiment scripts

This folder contains the three experiments intended for the revised TMLR manuscript.

## Final experimental structure

1. `exp1_trained_models.py` — Fisher width in trained models.
2. `exp2_estimator_accuracy.py` — low-rank approximation, fixed-base-point convergence, and structured approximations.
3. `exp3_fisher_alignment.py` — non-spherical subspaces / Fisher alignment / held-out local sensitivity.

The old post-fit generalization experiment is intentionally omitted from the final package.

## Important corrections relative to the previous scripts

- Binary logistic and softmax are fit **without intercepts**, matching the parameter dimensions stated in the paper (`p=784` and `p=7840`).
- Model A uses the **full conditional model Fisher averaged over observed covariates**, not the observed-label score outer-product matrix.
- Model B uses the **diagonal conditional model Fisher**.
- `sqrt(Tr(G))` is called the **trace upper scale**, not a score upper bound.
- Experiment 2.1 compares each matrix only with the rank-k truncation of that same matrix.
- Experiment 2.2 keeps the fitted base point fixed while varying the sample used to estimate the Fisher matrix.
- Experiment 3 uses the official MNIST train/test split and produces one three-panel figure directly with Matplotlib; no PyMuPDF/Pillow dependency is needed.
- Model C uses an analytic Fisher matrix, while its width reference is a high-accuracy Monte Carlo evaluation of the exact spectrum; it is not called an exact analytic width.

## Setup

Recommended: Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\\Scripts\\activate       # Windows PowerShell
pip install -r requirements.txt
```

The scripts automatically download the four standard MNIST IDX gzip files from the Google-hosted MNIST mirror into `mnist_data/` if they are missing.

## Smoke test first

```bash
python run_all.py --quick
```

Quick mode only checks that the pipeline runs. **Do not use quick-mode numbers in the manuscript.**

You can also run the scripts separately:

```bash
python exp1_trained_models.py --quick
python exp2_estimator_accuracy.py --quick
python exp3_fisher_alignment.py --quick
```

## Publication run

Run all three:

```bash
python run_all.py
```

or separately:

```bash
python exp1_trained_models.py --outdir results/exp1
python exp2_estimator_accuracy.py --outdir results/exp2
python exp3_fisher_alignment.py --outdir results/exp3
```

The full run can take substantial time because Experiment 1 fits many logistic/softmax models and Experiment 2 repeatedly diagonalizes 784x784 matrices.

## Outputs to send back for manuscript rewriting

After the publication run, the most useful files are:

```text
results/exp1/exp1_summary.txt
results/exp1/exp1_table.csv
results/exp1/exp1_figure.pdf

results/exp2/exp2_summary.txt
results/exp2/exp2_rankk.csv
results/exp2/exp2_convergence.csv
results/exp2/exp2_structured.csv
results/exp2/exp2_figure.pdf

results/exp3/exp3_run.txt
results/exp3/exp3_summary.csv
results/exp3/exp3_figure.pdf
```

These are enough to update the numerical values, figure captions, and Section 7 prose.

## Expected manuscript interpretation

### Experiment 1

Use it to show how Fisher width changes across model classes and regularization levels. Do not infer isotropy from a tight trace upper scale. Model C should be described as lambda-invariant because its Fisher matrix does not depend on the fitted regression parameter.

### Experiment 2

Panel (a) is the clean numerical check of Proposition 6.1 because every width is compared with the rank-k truncation of the same Fisher matrix.

Panel (b) is a fixed-base-point convergence diagnostic. The fitted parameter is kept fixed, but rank deficiency can still occur, so avoid presenting it as a literal verification of the positive-definite corollary.

Panel (c) compares the full conditional Fisher, its diagonal approximation, and the trace upper scale. The direction of the diagonal approximation error is empirical; Theorem 3.1 controls absolute error only.

### Experiment 3

This is the main set-dependent experiment: equal Euclidean Gaussian width at fixed k, different Fisher alignment, different Fisher widths, and the same qualitative ordering in held-out local loss sensitivity.

## Old Experiment 4

Do not include the old `exp3_generalization*.py` scripts in the anonymous reproducibility package if the corresponding post-fit generalization experiment is removed from the manuscript.
