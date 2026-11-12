# Online Resource 1

**Fisher Width on Statistical Manifolds**  
Vu Khac Ky, Department of Mathematics, FPT University, Hanoi, Vietnam  
Correspondence: kyvk2@fe.edu.vn

This archive contains the Python scripts, full-run numerical outputs, and
figure-generation script for the three MNIST experiments in the manuscript.

## Reproduce the results

Use Python 3.12. The dependency versions in `requirements.txt` match the
verified run. On a POSIX shell:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg \
  python run_all.py --data-dir mnist_data --results-dir rerun
python make_figures.py --results-dir rerun --outdir figures
```

The scripts download the four MNIST IDX gzip files from the
[CVDF mirror](https://github.com/cvdfoundation/mnist) if they are missing.
Their SHA-256 checksums are in `mnist_sha256.txt`. The original MNIST files
are not redistributed in this archive. The optional `--quick` flag to
`run_all.py` is for testing the pipeline with smaller Monte Carlo budgets.
The manuscript uses the full configuration.

## Outputs

- `results/exp1/` contains the trained-model summaries, table, and arrays.
- `results/exp2/` contains the paired spectral-truncation comparisons and
  the diagonal and trace comparisons. The script also saves a fixed-base-point
  diagnostic in `exp2_convergence.csv`. This diagnostic does not enter the
  manuscript's estimates or claims.
- `results/exp3/` contains the held-out subspace comparisons and summaries.
- `figures/` contains the three vector PDFs included in the manuscript.
  `make_figures.py` builds them from the saved arrays and CSV files.

The Fisher matrices are positive semidefinite and may have a numerical
nullspace. When the smallest eigenvalues are tied, an eigensolver may return
different orthonormal bases. The selected bottom subspaces and their test-loss
sensitivity can therefore vary across numerical libraries, even when their
Fisher widths agree at the reported precision.
