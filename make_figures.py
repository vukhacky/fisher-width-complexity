"""Render the manuscript figures from the saved full-run statistics."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, MaxNLocator
import numpy as np

WIDTH = 119 / 25.4
BLUE, ORANGE, GREEN, GRAY = "#165783", "#a94a13", "#397258", "#454545"


def panel(ax, name):
    ax.set_title(name, loc="left", fontweight="bold", pad=3)
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(length=2.8, pad=2)
    ax.minorticks_off()


def series(ax, x, y, spread=None, *, color=BLUE, fmt="o-", **kwargs):
    return ax.errorbar(
        x, y, yerr=spread, fmt=fmt, color=color, capsize=1.6,
        markersize=3.2, linewidth=0.85, **kwargs,
    )


def figure_one(results, output):
    with np.load(results / "exp1/exp1_results.npz") as data:
        x = np.arange(data["lambdas"].size)
        fig, axes = plt.subplots(3, 3, figsize=(WIDTH, 5.65), layout="constrained")
        for row, key in enumerate("ABC"):
            width, ratio, last = axes[row]
            scale = data[f"wE_{key}"].item()
            fisher = data[f"{key}_wF"]
            trace = data[f"{key}_trace_upper"]
            kappa = data[f"{key}_kappa"]
            width.fill_between(x, data[f"{key}_lb"].mean(1), data[f"{key}_ub"].mean(1),
                               color="#dbe6ef", edgecolor="none")
            width.axhline(scale, color=GRAY, linestyle=":")
            series(width, x, fisher.mean(1), fisher.std(1))
            series(width, x, trace.mean(1), trace.std(1), color=ORANGE, fmt="s--")
            if key == "C":
                width.axhline(data["wC_reference"].item(), color=GREEN, linestyle="-.")
            width.set(ylabel=f"Width (model {key})", ylim=(0, None))
            width.yaxis.set_major_locator(MaxNLocator(4))
            series(ratio, x, fisher.mean(1) / scale, fisher.std(1) / scale)
            ratio.axhline(1, color=GRAY, linestyle=":")
            ratio.set(ylabel="Width ratio", yticks=[0, 0.5, 1], ylim=(0, 1.12))
            if key != "C":
                series(last, x, kappa.mean(1), kappa.std(1), fmt="D-")
                last.set(ylabel=r"Spectral ratio", yscale="log")
                last.yaxis.set_major_locator(LogLocator(numticks=4))
            else:
                draws = data["B_values"]
                errors = data["mc_errors"]
                mean = errors.mean(1)
                series(last, draws, mean, errors.std(1))
                last.plot(draws, mean[0] * np.sqrt(draws[0] / draws), "--", color=ORANGE)
                last.set(xlabel="Gaussian draws", ylabel="MC error (%)", xscale="log", yscale="log")
            for col, ax in enumerate(axes[row]):
                panel(ax, f"({chr(97 + col)})" if row == 0 else "")
                if row < 2 or col < 2:
                    ax.set_xticks([0, 2, 4, 6], ["0", r"$10^{-3}$", r"$10^{-1}$", "5"])
                    ax.set_xlabel(r"$\lambda$")

        handles = [
            Line2D([], [], color=BLUE, marker="o", label="Fisher estimate"),
            Line2D([], [], color=ORANGE, marker="s", linestyle="--", label="Trace bound"),
            Line2D([], [], color=GRAY, linestyle=":", label="Euclidean width"),
            Patch(facecolor="#dbe6ef", label="Spectral bounds"),
            Line2D([], [], color=GREEN, linestyle="-.", label="Fixed-spectrum reference"),
        ]
        fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
        fig.savefig(output / "Fig1.pdf")
        plt.close(fig)


def figure_two(results, output):
    ranks = np.genfromtxt(results / "exp2/exp2_rankk.csv", delimiter=",", names=True)
    diag = np.genfromtxt(results / "exp2/exp2_structured.csv", delimiter=",", names=True)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDTH, 2.6), layout="constrained")
    x = ranks["k"]
    series(ax, x, ranks["error_mean"], ranks["error_std"], label="Estimated error")
    series(ax, x, ranks["bound_mean"], ranks["bound_std"],
           color=ORANGE, fmt="s--", label="Upper bound")
    ax.set(xscale="log", yscale="log", xlabel="Rank $k$", ylabel="Error or upper bound")
    ax.set_xticks([1, 10, 100, 392], ["1", "10", "100", "392"])
    ax.legend(frameon=False)
    columns = ["full_conditional_fisher_width", "diagonal_conditional_fisher_width", "trace_upper_scale"]
    stats = np.stack([diag[column] for column in columns])
    bx.bar([0, 1, 2], stats.mean(1), yerr=stats.std(1), color=[BLUE, GREEN, ORANGE], capsize=2)
    bx.set_xticks([0, 1, 2], ["Full", "Diagonal", "Trace"])
    bx.set_ylabel("Width or trace bound")
    panel(ax, "(a)")
    panel(bx, "(b)")
    fig.savefig(output / "Fig2.pdf")
    plt.close(fig)


def figure_three(results, output):
    data = np.genfromtxt(results / "exp3/exp3_summary.csv", delimiter=",", names=True)
    k = data["k"]
    fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 4.5), layout="constrained")
    (ax, bx), (cx, legend_ax) = axs
    specs = [("top", BLUE, "o", "Top"), ("random", ORANGE, "s", "Random"), ("bottom", GREEN, "D", "Bottom")]
    for group, color, marker, label in specs:
        random = group == "random"
        suffix = "_mean" if random else ""
        widths = data[f"{group}_fisher_width{suffix}"]
        traces = data[f"{group}_trace_upper{suffix}"]
        sensitivity = data[f"{group}_heldout_sensitivity{suffix}"]
        width_sd = data["random_fisher_width_std"] if random else None
        trace_sd = data["random_trace_upper_std"] if random else None
        sensitivity_sd = data["random_heldout_sensitivity_std"] if random else None
        series(ax, k, widths, width_sd, color=color, fmt=marker + "-", label=label)
        series(bx, traces, widths, width_sd, color=color, fmt=marker, xerr=trace_sd)
        series(cx, k, sensitivity, sensitivity_sd, color=color, fmt=marker + "-")
    ax.plot(k, data["euclidean_width"], "--", color=GRAY, label="Euclidean width")
    bound = np.max(data["top_trace_upper"])
    bx.plot([0, bound], [0, bound], "--", color=GRAY)
    ax.set_ylabel("Width")
    bx.set(xlabel="Restricted trace bound", ylabel="Fisher width")
    cx.set(ylabel="Held-out sensitivity", yscale="log")
    for chart in (ax, cx):
        chart.set(xscale="log", xlabel="Subspace dimension $k$")
        chart.set_xticks([2, 8, 32, 128, 784], ["2", "8", "32", "128", "784"])
    for i, chart in enumerate((ax, bx, cx)):
        panel(chart, f"({chr(97 + i)})")
    legend_ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", frameon=False)
    fig.savefig(output / "Fig3.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "legend.fontsize": 8, "pdf.fonttype": 42})
    figure_one(args.results_dir, args.outdir)
    figure_two(args.results_dir, args.outdir)
    figure_three(args.results_dir, args.outdir)


if __name__ == "__main__":
    main()
