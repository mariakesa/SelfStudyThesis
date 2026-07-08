#!/usr/bin/env python3
"""
Plot side-by-side distributions of:

1. Estimated calcium-event probabilities across all neuron-image pairs.
2. Mean full 95% Wilson confidence-interval width per neuron,
   averaged across all images.

Each X[neuron, image] entry is assumed to be a Bernoulli event
probability estimated from 50 repeated image presentations.

Input
-----
/home/maria/SelfStudyThesis/data/
    allen_natural_scenes_four_class_composite.npy

Output
------
/home/maria/SelfStudyThesis/results/
    probability_and_per_neuron_wilson_ci_histograms.png
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

OUTPUT_PATH = Path(
    "/home/maria/SelfStudyThesis/results/"
    "probability_and_per_neuron_wilson_ci_histograms.png"
)

N_TRIALS = 50
CONFIDENCE_LEVEL = 0.95
N_CI_BINS = 80


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_composite(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Load the composite dataset and ensure X has shape:

        neurons x images
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

    print("\nComposite keys:")
    for key in data:
        print(f"  {key}")

    X = np.asarray(data["X"], dtype=np.float64)

    neuron_metadata = data.get("neuron_metadata", {})
    stimulus_metadata = data.get("stimulus_metadata", {})

    if "labels" in data:
        labels = np.asarray(data["labels"])
    elif "label" in stimulus_metadata:
        labels = np.asarray(stimulus_metadata["label"])
    else:
        raise KeyError(
            "Could not find labels in data['labels'] or "
            "data['stimulus_metadata']['label']."
        )

    if X.ndim != 2:
        raise ValueError(
            f"X must be two-dimensional, received shape {X.shape}."
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
            "Use the original probability matrix, not z-scored data."
        )

    print(f"\nX shape: {X.shape}")
    print(f"Number of neurons: {X.shape[0]:,}")
    print(f"Number of images:  {X.shape[1]:,}")
    print(f"Neuron-image pairs: {X.size:,}")

    return X, labels, neuron_metadata, stimulus_metadata


# ---------------------------------------------------------------------
# Wilson confidence intervals
# ---------------------------------------------------------------------

def wilson_interval(
    probabilities: np.ndarray,
    n_trials: int,
    confidence_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Wilson score confidence intervals.

    Parameters
    ----------
    probabilities
        Estimated Bernoulli probabilities.
    n_trials
        Number of Bernoulli trials underlying each probability.
    confidence_level
        Desired confidence level.

    Returns
    -------
    lower, upper
        Arrays with the same shape as probabilities.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must lie strictly between 0 and 1."
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

    lower = np.clip(
        center - half_width,
        0.0,
        1.0,
    )

    upper = np.clip(
        center + half_width,
        0.0,
        1.0,
    )

    return lower, upper


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def main() -> None:
    X, labels, neuron_metadata, stimulus_metadata = load_composite(
        DATA_PATH
    )

    lower, upper = wilson_interval(
        probabilities=X,
        n_trials=N_TRIALS,
        confidence_level=CONFIDENCE_LEVEL,
    )

    # -------------------------------------------------------------
    # Probability statistics
    # -------------------------------------------------------------

    probabilities = X.ravel()

    mean_probability = np.mean(probabilities)
    median_probability = np.median(probabilities)

    zero_fraction = np.mean(
        np.isclose(
            probabilities,
            0.0,
            atol=1e-8,
        )
    )

    # -------------------------------------------------------------
    # Confidence interval statistics
    # -------------------------------------------------------------

    full_ci_width = upper - lower

    # One value per neuron: average interval width across all images.
    mean_ci_width_per_neuron = np.mean(
        full_ci_width,
        axis=1,
    )

    mean_neuron_ci_width = np.mean(
        mean_ci_width_per_neuron
    )

    median_neuron_ci_width = np.median(
        mean_ci_width_per_neuron
    )

    # -------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------

    print("\nProbability distribution")
    print("-" * 72)
    print(
        f"Mean event probability:               "
        f"{mean_probability:.6f}"
    )
    print(
        f"Median event probability:             "
        f"{median_probability:.6f}"
    )
    print(
        f"Fraction exactly zero:                "
        f"{zero_fraction:.6f}"
    )
    print(
        f"Mean events per {N_TRIALS} trials:           "
        f"{mean_probability * N_TRIALS:.3f}"
    )

    print("\nPer-neuron Wilson interval distribution")
    print("-" * 72)
    print(
        f"Mean full CI width per neuron:        "
        f"{mean_neuron_ci_width:.6f}"
    )
    print(
        f"Median full CI width per neuron:      "
        f"{median_neuron_ci_width:.6f}"
    )

    # -------------------------------------------------------------
    # Histogram binning
    # -------------------------------------------------------------

    # Since probabilities are counts divided by 50, the possible values
    # occur in steps of 1/50 = 0.02. These bins are centered on those
    # discrete values.
    probability_bin_edges = (
        np.arange(-0.5, N_TRIALS + 1.5)
        / N_TRIALS
    )

    # Crop only the displayed range, not the data used in calculations.
    probability_display_max = min(
        1.0,
        np.quantile(probabilities, 0.995)
        + 1.0 / N_TRIALS,
    )

    # -------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(15, 6),
    )

    # =============================================================
    # Left panel: probability values
    # =============================================================

    axes[0].hist(
        probabilities,
        bins=probability_bin_edges,
        edgecolor="black",
        linewidth=0.35,
        alpha=0.85,
    )

    axes[0].axvline(
        mean_probability,
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_probability:.3f}",
    )

    axes[0].axvline(
        median_probability,
        linestyle=":",
        linewidth=2,
        label=f"Median = {median_probability:.3f}",
    )

    axes[0].set_xlabel(
        "Estimated event probability"
    )

    axes[0].set_ylabel(
        "Number of neuron-image pairs"
    )

    axes[0].set_title(
        "Distribution of calcium-event probabilities"
    )

    axes[0].set_xlim(
        -0.01,
        probability_display_max,
    )

    axes[0].grid(
        axis="y",
        alpha=0.25,
    )

    axes[0].legend(
        loc="lower right"
    )

    axes[0].text(
        0.97,
        0.95,
        (
            f"Exactly zero: {zero_fraction:.1%}\n"
            f"n = {probabilities.size:,}"
        ),
        transform=axes[0].transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    # =============================================================
    # Right panel: mean CI width per neuron
    # =============================================================

    axes[1].hist(
        mean_ci_width_per_neuron,
        bins=N_CI_BINS,
        edgecolor="black",
        linewidth=0.35,
        alpha=0.85,
    )

    axes[1].axvline(
        mean_neuron_ci_width,
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_neuron_ci_width:.3f}",
    )

    axes[1].axvline(
        median_neuron_ci_width,
        linestyle=":",
        linewidth=2,
        label=f"Median = {median_neuron_ci_width:.3f}",
    )

    axes[1].set_xlabel(
        "Mean full 95% Wilson interval width across images"
    )

    axes[1].set_ylabel(
        "Number of neurons"
    )

    axes[1].set_title(
        "Mean probability-estimation uncertainty per neuron"
    )

    axes[1].grid(
        axis="y",
        alpha=0.25,
    )

    axes[1].legend(
        loc="upper right"
    )

    # -------------------------------------------------------------
    # Final formatting and save
    # -------------------------------------------------------------

    fig.suptitle(
        "Event probabilities and their sampling uncertainty",
        fontsize=15,
        y=1.02,
    )

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved figure to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()