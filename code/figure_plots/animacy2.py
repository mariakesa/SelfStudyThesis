#!/usr/bin/env python3
"""
Sort Allen natural-scene images by an in-sample neural animacy score,
then sort neurons by their modulation along that axis.

Input
-----
/home/maria/SelfStudyThesis/data/
    allen_natural_scenes_four_class_composite.npy

Outputs
-------
/home/maria/SelfStudyThesis/results/animacy_modulation/

    animacy_modulation_delta_heatmap.png
    animacy_modulation_spearman_heatmap.png
    neuron_modulation_summary.png
    animacy_modulation_results.npz

Interpretation
--------------
Columns:
    Images ordered from non-animal-like to animal-like using an
    in-sample logistic-regression decision score.

Rows in delta heatmap:
    Neurons ordered by

        mean response to animals
        minus
        mean response to non-animals.

Rows in Spearman heatmap:
    Neurons ordered by monotonic association between their response
    probability and the continuous neural animacy score.

This is a descriptive analysis. The same neural population is used to
construct the image ordering and examine neuronal tuning.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_PATH = Path(
    "/home/maria/SelfStudyThesis/data/"
    "allen_natural_scenes_four_class_composite.npy"
)

OUTPUT_DIR = Path(
    "/home/maria/SelfStudyThesis/results/animacy_modulation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

ANIMAL_LABEL = 0
RANDOM_STATE = 97

# Number of strongest negative and positive neurons to display.
N_NEURONS_EACH_SIDE = 500

# Filter neurons that have almost no activity or variation.
MIN_MEAN_RESPONSE = 0.005
MIN_RESPONSE_RANGE = 0.02
MIN_RESPONSE_STD = 1e-4

# Heatmap limits after row-wise z-scoring.
HEATMAP_VMIN = -2.5
HEATMAP_VMAX = 2.5


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_composite(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Load the composite dictionary and return:

        X                neurons x images
        labels           image labels
        neuron_metadata
        stimulus_metadata
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

    X = np.asarray(data["X"], dtype=np.float32)

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
        raise ValueError(f"X must be 2D, received {X.shape}")

    # Ensure neurons x images.
    if X.shape[1] != len(labels) and X.shape[0] == len(labels):
        print("Transposing X to neurons x images.")
        X = X.T

    if X.shape[1] != len(labels):
        raise ValueError(
            f"X shape {X.shape} does not match "
            f"{len(labels)} stimulus labels."
        )

    return X, labels, neuron_metadata, stimulus_metadata


# ---------------------------------------------------------------------
# Animacy axis
# ---------------------------------------------------------------------

def fit_animacy_axis(
    X: np.ndarray,
    labels: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    Fit an in-sample animal-vs-rest logistic regression.

    Parameters
    ----------
    X:
        neurons x images

    Returns
    -------
    decision_scores:
        Raw logistic decision score for each image.

    probabilities:
        Predicted probability of animal for each image.

    weights:
        Logistic-regression weights in original neuron coordinates.

    intercept:
        Logistic-regression intercept.
    """
    X_images = X.T
    y = (labels == ANIMAL_LABEL).astype(np.int64)

    print("\nBinary class counts:")
    print(f"  non-animal: {np.sum(y == 0)}")
    print(f"  animal:     {np.sum(y == 1)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_images)

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )

    model.fit(X_scaled, y)

    decision_scores = model.decision_function(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    accuracy = model.score(X_scaled, y)

    print(f"\nIn-sample accuracy: {accuracy:.4f}")
    print(
        "This accuracy is descriptive and should not be interpreted "
        "as cross-validated performance."
    )

    # Convert weights from standardized coordinates back into the
    # original neural probability coordinates.
    weights_original = (
        model.coef_.squeeze()
        / np.maximum(scaler.scale_, 1e-12)
    )

    intercept_original = float(
        model.intercept_[0]
        - np.sum(
            model.coef_.squeeze()
            * scaler.mean_
            / np.maximum(scaler.scale_, 1e-12)
        )
    )

    return (
        decision_scores,
        probabilities,
        weights_original,
        intercept_original,
    )


# ---------------------------------------------------------------------
# Neuron filtering and modulation
# ---------------------------------------------------------------------

def find_valid_neurons(X: np.ndarray) -> np.ndarray:
    """
    Remove neurons with negligible activity or variation.
    """
    finite = np.isfinite(X).all(axis=1)
    mean_response = np.nanmean(X, axis=1)
    response_range = np.nanmax(X, axis=1) - np.nanmin(X, axis=1)
    response_std = np.nanstd(X, axis=1)

    valid = (
        finite
        & (mean_response >= MIN_MEAN_RESPONSE)
        & (response_range >= MIN_RESPONSE_RANGE)
        & (response_std >= MIN_RESPONSE_STD)
    )

    print(
        f"\nValid neurons after filtering: "
        f"{valid.sum()} / {len(valid)}"
    )

    return valid


def compute_delta_modulation(
    X: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Animal-minus-non-animal mean response probability.
    """
    animal_mask = labels == ANIMAL_LABEL
    nonanimal_mask = ~animal_mask

    animal_mean = X[:, animal_mask].mean(axis=1)
    nonanimal_mean = X[:, nonanimal_mask].mean(axis=1)

    return animal_mean - nonanimal_mean


def compute_spearman_modulation(
    X: np.ndarray,
    decision_scores: np.ndarray,
) -> np.ndarray:
    """
    Compute Spearman correlation between every neuron's response
    profile and the continuous animacy decision score.

    X shape:
        neurons x images
    """
    score_ranks = rankdata(decision_scores).astype(np.float64)
    score_ranks -= score_ranks.mean()

    score_norm = np.sqrt(np.sum(score_ranks ** 2))

    if score_norm <= 0:
        raise ValueError("Animacy scores have zero rank variance.")

    print("\nRanking neuronal response profiles...")

    response_ranks = rankdata(
        X,
        axis=1,
        method="average",
    ).astype(np.float64)

    response_ranks -= response_ranks.mean(
        axis=1,
        keepdims=True,
    )

    response_norms = np.sqrt(
        np.sum(response_ranks ** 2, axis=1)
    )

    correlations = (
        response_ranks @ score_ranks
    ) / np.maximum(
        response_norms * score_norm,
        1e-12,
    )

    return correlations.astype(np.float32)


def choose_extreme_neurons(
    values: np.ndarray,
    valid_mask: np.ndarray,
    n_each_side: int,
) -> np.ndarray:
    """
    Select the strongest negative and strongest positive neurons.
    """
    valid_indices = np.flatnonzero(valid_mask)
    valid_values = values[valid_mask]

    local_order = np.argsort(valid_values)

    n_each_side = min(
        n_each_side,
        len(valid_indices) // 2,
    )

    negative = valid_indices[
        local_order[:n_each_side]
    ]

    positive = valid_indices[
        local_order[-n_each_side:]
    ]

    # Negative neurons first, positive neurons second.
    selected = np.concatenate([negative, positive])

    return selected


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def row_zscore(matrix: np.ndarray) -> np.ndarray:
    """
    Z-score each neuron across the 118 images.
    """
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)

    return (
        matrix - mean
    ) / np.maximum(std, 1e-8)


def find_class_transition(
    labels_sorted: np.ndarray,
) -> float:
    """
    Return the boundary after the last non-animal image.

    This is mainly useful because the in-sample classifier may separate
    the two classes almost perfectly.
    """
    is_animal = labels_sorted == ANIMAL_LABEL

    animal_positions = np.flatnonzero(is_animal)

    if len(animal_positions) == 0:
        return np.nan

    return float(animal_positions.min() - 0.5)


def plot_modulation_heatmap(
    X: np.ndarray,
    labels: np.ndarray,
    decision_scores: np.ndarray,
    image_order: np.ndarray,
    neuron_order: np.ndarray,
    modulation_values: np.ndarray,
    modulation_name: str,
    output_path: Path,
) -> None:
    """
    Plot selected neurons ordered by a specified modulation measure.
    """
    X_selected = X[neuron_order][:, image_order]
    X_z = row_zscore(X_selected)

    labels_sorted = labels[image_order]
    scores_sorted = decision_scores[image_order]

    n_negative = len(neuron_order) // 2
    class_transition = find_class_transition(labels_sorted)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(16, 10),
        height_ratios=[1.2, 8],
        sharex=True,
        constrained_layout=True,
    )

    x_positions = np.arange(len(image_order))

    # --------------------------------------------------------------
    # Top panel: continuous logistic score
    # --------------------------------------------------------------

    axes[0].plot(
        x_positions,
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

    animal_positions = np.flatnonzero(
        labels_sorted == ANIMAL_LABEL
    )

    axes[0].scatter(
        animal_positions,
        scores_sorted[animal_positions],
        s=20,
        label="ground-truth animal",
        zorder=3,
    )

    if np.isfinite(class_transition):
        axes[0].axvline(
            class_transition,
            linestyle=":",
            linewidth=1.5,
        )

    axes[0].set_ylabel(
        "Logistic\ndecision score"
    )

    axes[0].set_title(
        f"Images ordered by neural animacy score; "
        f"neurons ordered by {modulation_name}"
    )

    axes[0].legend(loc="upper left")

    # --------------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------------

    image = axes[1].imshow(
        X_z,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
    )

    # Separates negative-modulation and positive-modulation neurons.
    axes[1].axhline(
        n_negative - 0.5,
        linewidth=2,
    )

    if np.isfinite(class_transition):
        axes[1].axvline(
            class_transition,
            linestyle="--",
            linewidth=1.5,
        )

    axes[1].set_xlabel(
        "Images sorted from non-animal-like to animal-like"
    )

    axes[1].set_ylabel(
        "Selected neurons\n"
        "negative modulation → positive modulation"
    )

    colorbar = figure.colorbar(
        image,
        ax=axes[1],
        fraction=0.025,
        pad=0.01,
    )

    colorbar.set_label(
        "Within-neuron z-scored response probability"
    )

    selected_values = modulation_values[neuron_order]

    figure.text(
        0.5,
        0.005,
        (
            f"Selected modulation range: "
            f"{selected_values.min():.4f} to "
            f"{selected_values.max():.4f}"
        ),
        ha="center",
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_modulation_summary(
    delta: np.ndarray,
    spearman: np.ndarray,
    valid_mask: np.ndarray,
    output_path: Path,
) -> None:
    """
    Compare binary category modulation with continuous-score modulation.
    """
    figure, axis = plt.subplots(
        figsize=(8, 7),
        constrained_layout=True,
    )

    axis.scatter(
        delta[valid_mask],
        spearman[valid_mask],
        s=8,
        alpha=0.35,
    )

    axis.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    axis.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    axis.set_xlabel(
        "Animal minus non-animal mean response probability"
    )

    axis.set_ylabel(
        "Spearman correlation with neural animacy score"
    )

    axis.set_title(
        "Single-neuron modulation along the population animacy axis"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    (
        X,
        labels,
        neuron_metadata,
        stimulus_metadata,
    ) = load_composite(DATA_PATH)

    print("\nLoaded dataset:")
    print(f"  Neural matrix: {X.shape}")
    print(f"  Labels:        {labels.shape}")

    (
        decision_scores,
        animal_probabilities,
        logistic_weights,
        logistic_intercept,
    ) = fit_animacy_axis(X, labels)

    image_order = np.argsort(decision_scores)

    valid_neurons = find_valid_neurons(X)

    delta = compute_delta_modulation(
        X,
        labels,
    )

    spearman = np.zeros(
        X.shape[0],
        dtype=np.float32,
    )

    spearman_valid = compute_spearman_modulation(
        X[valid_neurons],
        decision_scores,
    )

    spearman[valid_neurons] = spearman_valid

    delta_neuron_order = choose_extreme_neurons(
        values=delta,
        valid_mask=valid_neurons,
        n_each_side=N_NEURONS_EACH_SIDE,
    )

    spearman_neuron_order = choose_extreme_neurons(
        values=spearman,
        valid_mask=valid_neurons,
        n_each_side=N_NEURONS_EACH_SIDE,
    )

    delta_heatmap_path = (
        OUTPUT_DIR
        / "animacy_modulation_delta_heatmap.png"
    )

    spearman_heatmap_path = (
        OUTPUT_DIR
        / "animacy_modulation_spearman_heatmap.png"
    )

    summary_path = (
        OUTPUT_DIR
        / "neuron_modulation_summary.png"
    )

    results_path = (
        OUTPUT_DIR
        / "animacy_modulation_results.npz"
    )

    plot_modulation_heatmap(
        X=X,
        labels=labels,
        decision_scores=decision_scores,
        image_order=image_order,
        neuron_order=delta_neuron_order,
        modulation_values=delta,
        modulation_name=(
            "animal-minus-non-animal response modulation"
        ),
        output_path=delta_heatmap_path,
    )

    plot_modulation_heatmap(
        X=X,
        labels=labels,
        decision_scores=decision_scores,
        image_order=image_order,
        neuron_order=spearman_neuron_order,
        modulation_values=spearman,
        modulation_name=(
            "Spearman correlation with animacy score"
        ),
        output_path=spearman_heatmap_path,
    )

    plot_modulation_summary(
        delta=delta,
        spearman=spearman,
        valid_mask=valid_neurons,
        output_path=summary_path,
    )

    np.savez_compressed(
        results_path,
        image_order=image_order,
        labels=labels,
        labels_sorted=labels[image_order],
        decision_scores=decision_scores,
        decision_scores_sorted=decision_scores[image_order],
        animal_probabilities=animal_probabilities,
        animal_probabilities_sorted=(
            animal_probabilities[image_order]
        ),
        logistic_weights=logistic_weights,
        logistic_intercept=logistic_intercept,
        valid_neurons=valid_neurons,
        delta_modulation=delta,
        spearman_modulation=spearman,
        delta_neuron_order=delta_neuron_order,
        spearman_neuron_order=spearman_neuron_order,
    )

    print("\nModulation summaries:")
    print(
        f"  Delta range among valid neurons: "
        f"{delta[valid_neurons].min():.5f} to "
        f"{delta[valid_neurons].max():.5f}"
    )

    print(
        f"  Spearman range among valid neurons: "
        f"{spearman[valid_neurons].min():.5f} to "
        f"{spearman[valid_neurons].max():.5f}"
    )

    print("\nSaved:")
    print(f"  {delta_heatmap_path}")
    print(f"  {spearman_heatmap_path}")
    print(f"  {summary_path}")
    print(f"  {results_path}")

    print(
        "\nThe delta heatmap asks which neurons distinguish animals "
        "from all other classes."
    )

    print(
        "The Spearman heatmap asks which neurons vary monotonically "
        "along the continuous population-defined animacy ordering."
    )


if __name__ == "__main__":
    main()