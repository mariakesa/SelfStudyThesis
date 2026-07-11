#!/usr/bin/env python3
"""
Plot LOO PCA reconstruction performance as a pin/lollipop plot.

This script reads the metadata/results archive created by:

    loo_pca_heldout_reconstruction.py

It plots the mean held-out reconstruction fraction at selected PCA ranks as
vertical pins. The pin head is the mean reconstruction fraction and the
vertical error bar is the stored 95% confidence interval for the mean.

Input
-----
    /home/maria/SelfStudyThesis/results/
        all_neurons_loo_pca_reconstruction/
        loo_pca_reconstruction_results.npz

Outputs
-------
    loo_pca_reconstruction_pins.png
    loo_pca_reconstruction_pins.pdf
    loo_pca_reconstruction_pins.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================

RESULTS_PATH = Path(
    "/home/maria/SelfStudyThesis/results/"
    "all_neurons_loo_pca_reconstruction/"
    "loo_pca_reconstruction_results.npz"
)

OUTDIR = RESULTS_PATH.parent

OUT_PNG = OUTDIR / "loo_pca_reconstruction_pins.png"
OUT_PDF = OUTDIR / "loo_pca_reconstruction_pins.pdf"
OUT_CSV = OUTDIR / "loo_pca_reconstruction_pins.csv"


# =============================================================================
# Settings
# =============================================================================

# These are the ranks shown as pins. Values absent from the results archive are
# ignored safely.
REQUESTED_K = [1, 2, 3, 5, 10, 20, 40, 60, 80, 100, 116]

SHOW_VALUE_LABELS = True
Y_AS_PERCENT = True


# =============================================================================
# Loading
# =============================================================================

def load_reconstruction_results(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find reconstruction results:\n{path}\n\n"
            "Run loo_pca_heldout_reconstruction.py first."
        )

    archive = np.load(path, allow_pickle=True)

    required = {
        "ranks",
        "mean_reconstructed_fraction",
        "ci95_low_mean",
        "ci95_high_mean",
    }
    missing = required.difference(archive.files)

    if missing:
        raise KeyError(
            "The NPZ archive is missing required fields: "
            f"{sorted(missing)}\nAvailable fields: {archive.files}"
        )

    return {
        "ranks": np.asarray(archive["ranks"], dtype=np.int64),
        "mean": np.asarray(
            archive["mean_reconstructed_fraction"],
            dtype=np.float64,
        ),
        "ci_low": np.asarray(
            archive["ci95_low_mean"],
            dtype=np.float64,
        ),
        "ci_high": np.asarray(
            archive["ci95_high_mean"],
            dtype=np.float64,
        ),
    }


def select_requested_ranks(
    ranks: np.ndarray,
    mean: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
) -> pd.DataFrame:
    rank_to_position = {
        int(rank): position
        for position, rank in enumerate(ranks)
    }

    available_k = [
        k for k in REQUESTED_K
        if k in rank_to_position
    ]

    if not available_k:
        raise ValueError(
            "None of the requested PCA ranks are present in the archive."
        )

    rows = []
    for k in available_k:
        position = rank_to_position[k]
        rows.append(
            {
                "k": int(k),
                "mean_reconstructed_fraction": float(mean[position]),
                "ci95_low": float(ci_low[position]),
                "ci95_high": float(ci_high[position]),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================

def plot_pin_chart(table: pd.DataFrame) -> None:
    # Use equally spaced discrete x positions so that ranks 1, 2, and 3 remain
    # readable while preserving the tested-rank ordering.
    x = np.arange(len(table), dtype=float)

    mean = table["mean_reconstructed_fraction"].to_numpy()
    ci_low = table["ci95_low"].to_numpy()
    ci_high = table["ci95_high"].to_numpy()

    lower_error = mean - ci_low
    upper_error = ci_high - mean

    fig, ax = plt.subplots(figsize=(11.0, 6.2))

    # Pin stems.
    ax.vlines(
        x=x,
        ymin=0.0,
        ymax=mean,
        linewidth=2.0,
        alpha=0.8,
        zorder=1,
    )

    # Pin heads plus confidence intervals.
    ax.errorbar(
        x,
        mean,
        yerr=np.vstack([lower_error, upper_error]),
        fmt="o",
        markersize=8,
        linewidth=1.8,
        capsize=4,
        zorder=2,
        label="Mean LOO reconstruction ± 95% CI",
    )

    if SHOW_VALUE_LABELS:
        # Place each number above the *top of the confidence interval*, not
        # above the mean marker. This prevents the text from colliding with
        # the upper error-bar stem and cap.
        label_offset_points = 8

        for x_position, value, upper_bound in zip(x, mean, ci_high):
            label = (
                f"{100.0 * value:.1f}%"
                if Y_AS_PERCENT
                else f"{value:.3f}"
            )
            ax.annotate(
                label,
                xy=(x_position, upper_bound),
                xytext=(0, label_offset_points),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                zorder=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(table["k"].astype(str))

    if Y_AS_PERCENT:
        # Leave enough headroom for labels placed above the upper CI.
        maximum = max(float(ci_high.max()) * 1.28, 0.25)
        ticks = np.linspace(0.0, maximum, 6)
        ax.set_yticks(ticks)
        ax.set_yticklabels(
            [f"{100.0 * tick:.0f}%" for tick in ticks]
        )
        ylabel = "Held-out centered response reconstructed"
    else:
        # Leave enough headroom for labels placed above the upper CI.
        maximum = max(float(ci_high.max()) * 1.28, 0.25)
        ylabel = "Fraction of held-out centered response reconstructed"

    ax.set(
        xlabel="Number of training-fold principal components",
        ylabel=ylabel,
        title="Leave-one-image-out PCA reconstruction",
        ylim=(0.0, maximum),
    )

    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    results = load_reconstruction_results(RESULTS_PATH)

    table = select_requested_ranks(
        ranks=results["ranks"],
        mean=results["mean"],
        ci_low=results["ci_low"],
        ci_high=results["ci_high"],
    )

    table.to_csv(OUT_CSV, index=False)
    plot_pin_chart(table)

    print()
    print("=" * 80)
    print("LOO PCA reconstruction pin plot")
    print("=" * 80)
    print(table.to_string(index=False))
    print()
    print("Saved:")
    print(OUT_PNG)
    print(OUT_PDF)
    print(OUT_CSV)


if __name__ == "__main__":
    main()