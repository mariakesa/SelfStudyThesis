#!/usr/bin/env python3
"""
Plot a histogram of the saved animals-vs-rest logistic-regression weights.

By default, this plots the mean standardized LOO coefficient for every
cleaned neuron:

    mean_loo_weight_standardized

Positive weights push the decoder toward "animal"; negative weights push it
toward "non-animal".

Default input:
    /home/maria/SelfStudyThesis/results/all_neurons_animals_vs_rest_adam_loo/
        all_neurons_animals_vs_rest_adam_loo_results.npz

Default output:
    /home/maria/SelfStudyThesis/results/all_neurons_animals_vs_rest_adam_loo/
        mean_loo_weight_histogram.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_NPZ = Path(
    "/home/maria/SelfStudyThesis/results/"
    "all_neurons_animals_vs_rest_adam_loo/"
    "all_neurons_animals_vs_rest_adam_loo_results.npz"
)

DEFAULT_OUTPUT = DEFAULT_NPZ.parent / "mean_loo_weight_histogram.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot and save a histogram of saved decoder weights."
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=DEFAULT_NPZ,
        help="Path to the decoder results .npz file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the saved figure.",
    )
    parser.add_argument(
        "--weight-key",
        default="mean_loo_weight_standardized",
        choices=[
            "mean_loo_weight_standardized",
            "full_weights_standardized",
            "full_weights_raw",
        ],
        help="Which saved one-dimensional weight vector to plot.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(f"Results file not found: {args.npz}")

    with np.load(args.npz, allow_pickle=True) as data:
        if args.weight_key not in data.files:
            available = ", ".join(sorted(data.files))
            raise KeyError(
                f"Weight key {args.weight_key!r} is absent from {args.npz}. "
                f"Available keys: {available}"
            )

        weights = np.asarray(data[args.weight_key], dtype=np.float64).ravel()

    finite_mask = np.isfinite(weights)
    n_invalid = int(np.sum(~finite_mask))
    weights = weights[finite_mask]

    if weights.size == 0:
        raise ValueError("No finite weights were found to plot.")

    n_positive = int(np.sum(weights > 0))
    n_negative = int(np.sum(weights < 0))
    n_zero = int(np.sum(weights == 0))

    mean_weight = float(np.mean(weights))
    median_weight = float(np.median(weights))
    std_weight = float(np.std(weights))

    title_by_key = {
        "mean_loo_weight_standardized": "Distribution of Mean LOO Decoder Weights",
        "full_weights_standardized": "Distribution of Full-Model Standardized Weights",
        "full_weights_raw": "Distribution of Full-Model Raw Weights",
    }

    xlabel_by_key = {
        "mean_loo_weight_standardized": "Mean standardized weight across LOO folds",
        "full_weights_standardized": "Standardized decoder weight",
        "full_weights_raw": "Decoder weight in original activity units",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(weights, bins=args.bins, edgecolor="black", linewidth=0.35)
    ax.axvline(0.0, linestyle="--", linewidth=1.3, label="Zero")
    ax.axvline(
        mean_weight,
        linestyle=":",
        linewidth=1.5,
        label=f"Mean = {mean_weight:.3g}",
    )

    ax.set_title(title_by_key[args.weight_key])
    ax.set_xlabel(xlabel_by_key[args.weight_key])
    ax.set_ylabel("Number of neurons")
    ax.legend()

    summary = (
        f"n = {weights.size:,}\n"
        f"negative = {n_negative:,} | positive = {n_positive:,} | zero = {n_zero:,}\n"
        f"median = {median_weight:.3g} | SD = {std_weight:.3g}"
    )
    ax.text(
        0.98,
        0.96,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("Weight histogram saved successfully.")
    print(f"Input:       {args.npz}")
    print(f"Weight key:  {args.weight_key}")
    print(f"Weights:     {weights.size:,}")
    print(f"Invalid:     {n_invalid:,}")
    print(f"Output:      {args.output}")


if __name__ == "__main__":
    main()
