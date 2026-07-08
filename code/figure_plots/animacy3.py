#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(
    "/home/maria/SelfStudyThesis/data/"
    "allen_natural_scenes_four_class_composite.npy"
)

OUTPUT_DIR = Path(
    "/home/maria/SelfStudyThesis/results/animacy_all_neurons"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANIMAL_LABEL = 0
NEURONS_PER_BIN = 20

MIN_MEAN_RESPONSE = 0.005
MIN_RESPONSE_RANGE = 0.02


def load_data():
    data = np.load(DATA_PATH, allow_pickle=True).item()

    X = np.asarray(data["X"], dtype=np.float32)

    stimulus_metadata = data["stimulus_metadata"]
    labels = np.asarray(stimulus_metadata["label"])

    # Ensure shape is neurons x images.
    if X.shape[0] == len(labels):
        X = X.T

    if X.shape[1] != len(labels):
        raise ValueError(
            f"X shape {X.shape} does not match "
            f"{len(labels)} labels."
        )

    return X, labels


def fit_animacy_scores(X, labels):
    """
    Fit descriptive in-sample animal-vs-rest logistic regression.
    """
    y = (labels == ANIMAL_LABEL).astype(int)

    X_images = X.T

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_images)

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=97,
    )

    model.fit(X_scaled, y)

    decision_scores = model.decision_function(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    print(f"In-sample accuracy: {model.score(X_scaled, y):.4f}")

    return decision_scores, probabilities


def find_valid_neurons(X):
    mean_response = X.mean(axis=1)
    response_range = np.ptp(X, axis=1)

    valid = (
        np.isfinite(X).all(axis=1)
        & (mean_response >= MIN_MEAN_RESPONSE)
        & (response_range >= MIN_RESPONSE_RANGE)
    )

    print(
        f"Valid neurons: {valid.sum()} / {len(valid)}"
    )

    return valid


def compute_modulation(X, labels):
    """
    Mean response to animals minus mean response to non-animals.
    """
    animal_mask = labels == ANIMAL_LABEL
    nonanimal_mask = ~animal_mask

    animal_mean = X[:, animal_mask].mean(axis=1)
    nonanimal_mean = X[:, nonanimal_mask].mean(axis=1)

    return animal_mean - nonanimal_mean


def row_zscore(X):
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)

    return (X - mean) / np.maximum(std, 1e-8)


def bin_neurons(X, bin_size):
    """
    Average neighboring sorted neurons for display.
    Every neuron contributes to exactly one bin.
    """
    n_neurons, n_images = X.shape
    n_bins = int(np.ceil(n_neurons / bin_size))

    output = np.zeros(
        (n_bins, n_images),
        dtype=np.float32,
    )

    for bin_index in range(n_bins):
        start = bin_index * bin_size
        stop = min(start + bin_size, n_neurons)

        output[bin_index] = X[start:stop].mean(axis=0)

    return output


def plot_results(
    X,
    labels,
    decision_scores,
    image_order,
    neuron_order,
    modulation,
):
    X_sorted = X[neuron_order][:, image_order]

    # Z-score each neuron before averaging bins.
    X_z = row_zscore(X_sorted)

    X_display = bin_neurons(
        X_z,
        NEURONS_PER_BIN,
    )

    labels_sorted = labels[image_order]
    scores_sorted = decision_scores[image_order]
    modulation_sorted = modulation[neuron_order]

    zero_crossing_neuron = np.searchsorted(
        modulation_sorted,
        0.0,
    )

    zero_crossing_bin = (
        zero_crossing_neuron / NEURONS_PER_BIN
    )

    animal_positions = np.flatnonzero(
        labels_sorted == ANIMAL_LABEL
    )

    first_animal_position = animal_positions.min()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 12),
        height_ratios=[1.2, 10],
        sharex=True,
        constrained_layout=True,
    )

    x = np.arange(len(image_order))

    axes[0].plot(
        x,
        scores_sorted,
        marker="o",
        markersize=3,
        linewidth=1.5,
    )

    axes[0].axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    axes[0].axvline(
        first_animal_position - 0.5,
        linestyle=":",
        linewidth=1.5,
    )

    axes[0].scatter(
        animal_positions,
        scores_sorted[animal_positions],
        s=18,
        label="ground-truth animal",
    )

    axes[0].set_ylabel(
        "Logistic\ndecision score"
    )

    axes[0].set_title(
        "Images sorted by neural animacy score; "
        "all neurons sorted by animal modulation"
    )

    axes[0].legend(
        loc="upper left"
    )

    image = axes[1].imshow(
        X_display,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-2.5,
        vmax=2.5,
    )

    n_display_rows = X_display.shape[0]

    tick_positions = np.linspace(
        0,
        n_display_rows - 1,
        9,
    )

    tick_neuron_ranks = (
        tick_positions * NEURONS_PER_BIN
    ).astype(int)

    axes[1].set_yticks(tick_positions)
    axes[1].set_yticklabels(tick_neuron_ranks)

    axes[1].set_ylabel(
        "Sorted neuron rank\n"
        f"({NEURONS_PER_BIN} neurons averaged per display row)"
    )

    axes[1].axvline(
        first_animal_position - 0.5,
        linestyle="--",
        linewidth=1.5,
    )

    axes[1].axhline(
        zero_crossing_bin - 0.5,
        linewidth=2,
    )

    axes[1].set_xlabel(
        "Images sorted from non-animal-like to animal-like"
    )

    axes[1].set_ylabel(
        "All valid neurons sorted by modulation\n"
        f"({NEURONS_PER_BIN} neurons averaged per row)"
    )

    colorbar = fig.colorbar(
        image,
        ax=axes[1],
        fraction=0.025,
        pad=0.01,
    )

    colorbar.set_label(
        "Mean within-neuron z-scored response"
    )

    output_path = (
        OUTPUT_DIR
        / "all_neurons_animacy_modulation_heatmap.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved figure: {output_path}")


def main():
    X, labels = load_data()

    print(f"Neural matrix shape: {X.shape}")
    print(f"Labels shape: {labels.shape}")

    decision_scores, probabilities = fit_animacy_scores(
        X,
        labels,
    )

    valid_mask = find_valid_neurons(X)

    modulation = compute_modulation(
        X,
        labels,
    )

    image_order = np.argsort(decision_scores)

    valid_indices = np.flatnonzero(valid_mask)

    neuron_order = valid_indices[
        np.argsort(modulation[valid_mask])
    ]

    plot_results(
        X=X,
        labels=labels,
        decision_scores=decision_scores,
        image_order=image_order,
        neuron_order=neuron_order,
        modulation=modulation,
    )

    results_path = (
        OUTPUT_DIR
        / "all_neurons_animacy_modulation_results.npz"
    )

    np.savez_compressed(
        results_path,
        image_order=image_order,
        neuron_order=neuron_order,
        decision_scores=decision_scores,
        animal_probabilities=probabilities,
        labels=labels,
        modulation=modulation,
        valid_mask=valid_mask,
        neurons_per_bin=NEURONS_PER_BIN,
    )

    print(f"Saved results: {results_path}")


if __name__ == "__main__":
    main()