#!/usr/bin/env python3
"""
Calculate Wilson confidence intervals for calcium-event probabilities
and plot the distribution of confidence-interval widths.

Each X[neuron, image] value is assumed to be a Bernoulli probability
estimated from 50 repeated presentations.

Outputs
-------
/home/maria/SelfStudyThesis/results/bernoulli_confidence_intervals/

    wilson_ci_width_histogram.png
    wilson_ci_mean_width_per_neuron_histogram.png
    wilson_ci_summary.npz
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_PATH = Path(
    "/home/maria/SelfStudyThesis/data/"
    "allen_natural_scenes_four_class_composite.npy"
)

OUTPUT_DIR = Path(
    "/home/maria/SelfStudyThesis/results/"
    "bernoulli_confidence_intervals"
)

N_TRIALS = 50
CONFIDENCE_LEVEL = 0.95
N_BINS = 80


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_composite(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Load the composite dataset.

    Returns
    -------
    X
        Probability matrix with shape neurons x images.
    labels
        Stimulus labels.
    neuron_metadata
        Neuron metadata dictionary.
    stimulus_metadata
        Stimulus metadata dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(f"Could not find: {path}")

    data = np.load(path, allow_pickle=True)

    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()

    if not isinstance(data, dict):
        raise TypeError(
            "Expected the .npy file to contain a dictionary."
        )

    X = np.asarray(data["X"], dtype=np.float64)

    neuron_metadata = data.get("neuron_metadata", {})
    stimulus_metadata = data.get("stimulus_metadata", {})

    if "labels" in data:
        labels = np.asarray(data["labels"])
    elif "label" in stimulus_metadata:
        labels = np.asarray(stimulus_metadata["label"])
    else:
        raise KeyError(
            "Could not find stimulus labels."
        )

    if X.ndim != 2:
        raise ValueError(
            f"X must be two-dimensional, received {X.shape}."
        )

    # Ensure neurons x images.
    if X.shape[1] != len(labels) and X.shape[0] == len(labels):
        print("Transposing X to neurons x images.")
        X = X.T

    if X.shape[1] != len(labels):
        raise ValueError(
            f"X shape {X.shape} does not agree with "
            f"{len(labels)} stimulus labels."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X contains NaN or infinite values."
        )

    if np.any((X < 0.0) | (X > 1.0)):
        raise ValueError(
            "X contains values outside [0, 1]. "
            "Use the original probabilities, not z-scored data."
        )

    print(f"X shape: {X.shape}")
    print(f"Neurons: {X.shape[0]:,}")
    print(f"Images:  {X.shape[1]:,}")
    print(f"Probability estimates: {X.size:,}")

    return X, labels, neuron_metadata, stimulus_metadata


# ---------------------------------------------------------------------
# Wilson confidence interval
# ---------------------------------------------------------------------

def wilson_interval(
    probabilities: np.ndarray,
    n_trials: int,
    confidence_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Wilson score intervals for Bernoulli probabilities.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must lie between 0 and 1."
        )

    p_hat = np.asarray(probabilities, dtype=np.float64)

    alpha = 1.0 - confidence_level
    z = norm.ppf(1.0 - alpha / 2.0)
    z_squared = z**2

    denominator = 1.0 + z_squared / n_trials

    center = (
        p_hat + z_squared / (2.0 * n_trials)
    ) / denominator

    half_width = (
        z
        / denominator
        * np.sqrt(
            p_hat * (1.0 - p_hat) / n_trials
            + z_squared / (4.0 * n_trials**2)
        )
    )

    lower = np.clip(center - half_width, 0.0, 1.0)
    upper = np.clip(center + half_width, 0.0, 1.0)

    return lower, upper


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_all_ci_widths(
    ci_widths: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot confidence-interval widths across all neuron-image estimates.
    """
    flattened_widths = ci_widths.ravel()

    mean_width = np.mean(flattened_widths)
    median_width = np.median(flattened_widths)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(
        flattened_widths,
        bins=N_BINS,
        edgecolor="black",
        linewidth=0.35,
        alpha=0.85,
    )

    ax.axvline(
        mean_width,
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_width:.3f}",
    )

    ax.axvline(
        median_width,
        linestyle=":",
        linewidth=2,
        label=f"Median = {median_width:.3f}",
    )

    ax.set_xlabel("Full 95% Wilson confidence-interval width")
    ax.set_ylabel("Number of neuron–image estimates")
    ax.set_title(
        "Distribution of uncertainty in calcium-event probabilities"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_mean_ci_width_per_neuron(
    mean_width_per_neuron: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot each neuron's mean confidence-interval width across images.
    """
    grand_mean = np.mean(mean_width_per_neuron)
    median = np.median(mean_width_per_neuron)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(
        mean_width_per_neuron,
        bins=N_BINS,
        edgecolor="black",
        linewidth=0.35,
        alpha=0.85,
    )

    ax.axvline(
        grand_mean,
        linestyle="--",
        linewidth=2,
        label=f"Mean = {grand_mean:.3f}",
    )

    ax.axvline(
        median,
        linestyle=":",
        linewidth=2,
        label=f"Median = {median:.3f}",
    )

    ax.set_xlabel(
        "Mean full 95% Wilson interval width across images"
    )
    ax.set_ylabel("Number of neurons")
    ax.set_title(
        "Mean probability-estimation uncertainty per neuron"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, labels, neuron_metadata, stimulus_metadata = load_composite(
        DATA_PATH
    )

    lower, upper = wilson_interval(
        probabilities=X,
        n_trials=N_TRIALS,
        confidence_level=CONFIDENCE_LEVEL,
    )

    full_width = upper - lower
    one_sided_width = full_width / 2.0

    # Average across images separately for every neuron.
    mean_full_width_per_neuron = np.mean(
        full_width,
        axis=1,
    )

    mean_one_sided_width_per_neuron = np.mean(
        one_sided_width,
        axis=1,
    )

    grand_mean_full_width = np.mean(full_width)
    grand_median_full_width = np.median(full_width)

    percentiles = np.percentile(
        full_width,
        [0, 5, 25, 50, 75, 95, 100],
    )

    print("\n" + "=" * 72)
    print("Wilson confidence-interval widths")
    print("=" * 72)

    print(f"Confidence level: {CONFIDENCE_LEVEL:.1%}")
    print(f"Trials per estimate: {N_TRIALS}")
    print(
        f"Mean full CI width:   "
        f"{grand_mean_full_width:.6f}"
    )
    print(
        f"Median full CI width: "
        f"{grand_median_full_width:.6f}"
    )
    print(
        f"Mean one-sided width: "
        f"{np.mean(one_sided_width):.6f}"
    )

    print("\nFull CI width percentiles")
    print("-" * 72)

    percentile_names = [
        "minimum",
        "5th",
        "25th",
        "median",
        "75th",
        "95th",
        "maximum",
    ]

    for name, value in zip(percentile_names, percentiles):
        print(f"{name:>8}: {value:.6f}")

    all_widths_plot_path = (
        OUTPUT_DIR / "wilson_ci_width_histogram.png"
    )

    neuron_widths_plot_path = (
        OUTPUT_DIR
        / "wilson_ci_mean_width_per_neuron_histogram.png"
    )

    plot_all_ci_widths(
        ci_widths=full_width,
        output_path=all_widths_plot_path,
    )

    plot_mean_ci_width_per_neuron(
        mean_width_per_neuron=mean_full_width_per_neuron,
        output_path=neuron_widths_plot_path,
    )

    results_path = OUTPUT_DIR / "wilson_ci_summary.npz"

    np.savez_compressed(
        results_path,
        confidence_level=CONFIDENCE_LEVEL,
        n_trials=N_TRIALS,
        lower=lower.astype(np.float32),
        upper=upper.astype(np.float32),
        full_width=full_width.astype(np.float32),
        one_sided_width=one_sided_width.astype(np.float32),
        mean_full_width_per_neuron=(
            mean_full_width_per_neuron.astype(np.float32)
        ),
        mean_one_sided_width_per_neuron=(
            mean_one_sided_width_per_neuron.astype(np.float32)
        ),
        grand_mean_full_width=grand_mean_full_width,
        grand_median_full_width=grand_median_full_width,
        percentiles=percentiles,
    )

    print("\nSaved:")
    print(f"  {all_widths_plot_path}")
    print(f"  {neuron_widths_plot_path}")
    print(f"  {results_path}")


if __name__ == "__main__":
    main()