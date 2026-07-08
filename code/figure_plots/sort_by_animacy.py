#!/usr/bin/env python3
"""
Sort Allen natural-scene images by an in-sample neural animacy score,
then sort neurons with Rastermap.

Input
-----
Composite:
    /home/maria/SelfStudyThesis/data/
    allen_natural_scenes_four_class_composite.npy

Images:
    /home/maria/SelfStudyThesis/data/images

Outputs
-------
results/animacy_rastermap/
    animacy_rastermap_heatmap.png
    sorted_image_strip.png
    sorted_animacy_results.npz

Interpretation
--------------
Columns:
    Images ordered from strongly non-animal to strongly animal according
    to an in-sample logistic-regression decision score.

Rows:
    Neurons ordered by Rastermap according to their response-probability
    profiles across the sorted images.

Pixel:
    Neuron response probability for one image.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from rastermap import Rastermap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_PATH = Path(
    "/home/maria/SelfStudyThesis/data/"
    "allen_natural_scenes_four_class_composite.npy"
)

IMAGE_DIR = Path(
    "/home/maria/SelfStudyThesis/data/images"
)

OUTPUT_DIR = Path(
    "/home/maria/SelfStudyThesis/results/animacy_rastermap"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

ANIMAL_LABEL = 0
RANDOM_STATE = 97

# Rastermap settings.
# n_clusters=None sorts individual neurons.
N_PCS = 64
LOCALITY = 0.1
TIME_LAG_WINDOW = 0

# Display settings.
NEURONS_PER_DISPLAY_BIN = 20
IMAGE_THUMBNAIL_SIZE = 100


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_composite(path: Path) -> tuple[
    np.ndarray,
    np.ndarray,
    dict,
    dict,
]:
    """
    Load the Allen composite dataset.

    Expected structure:
        {
            "X": neurons x images,
            "labels": image labels,
            "neuron_metadata": ...,
            "stimulus_metadata": ...
        }
    """
    if not path.exists():
        raise FileNotFoundError(f"Composite file not found: {path}")

    data = np.load(path, allow_pickle=True)

    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()

    if not isinstance(data, dict):
        raise TypeError(
            "Expected the composite .npy file to contain a dictionary."
        )

    print("\nComposite keys:")
    for key in data:
        print(f"  {key}")

    X = np.asarray(data["X"], dtype=np.float32)

    labels = np.asarray(
        data.get(
            "labels",
            data.get("stimulus_metadata", {}).get("label"),
        )
    )

    neuron_metadata = data.get("neuron_metadata", {})
    stimulus_metadata = data.get("stimulus_metadata", {})

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, received shape {X.shape}")

    # Orient as neurons x images.
    if X.shape[1] != len(labels) and X.shape[0] == len(labels):
        print("Transposing X to neurons x images.")
        X = X.T

    if X.shape[1] != len(labels):
        raise ValueError(
            f"Could not align X shape {X.shape} with "
            f"{len(labels)} stimulus labels."
        )

    return X, labels, neuron_metadata, stimulus_metadata


def natural_sort_key(path: Path) -> list:
    """
    Natural sorting:
        image_2.png before image_10.png
    """
    pieces = re.split(r"(\d+)", path.stem)
    return [
        int(piece) if piece.isdigit() else piece.lower()
        for piece in pieces
    ]


def find_images(
    image_dir: Path,
    stimulus_metadata: dict,
    n_images: int,
) -> list[Path]:
    """
    Resolve image files.

    First attempts to use image paths stored in stimulus metadata.
    Otherwise naturally sorts files found in IMAGE_DIR.
    """
    valid_extensions = {
        ".png", ".jpg", ".jpeg", ".bmp",
        ".tif", ".tiff", ".webp",
    }

    metadata_paths = None

    for possible_key in (
        "image_paths",
        "image_path",
        "paths",
        "filenames",
        "image_filenames",
    ):
        if possible_key in stimulus_metadata:
            metadata_paths = stimulus_metadata[possible_key]
            break

    if metadata_paths is not None:
        resolved = []

        for value in metadata_paths:
            path = Path(str(value))

            if not path.is_absolute():
                path = image_dir / path

            resolved.append(path)

        if len(resolved) == n_images and all(p.exists() for p in resolved):
            print("Using image paths from stimulus metadata.")
            return resolved

        print(
            "Image paths were present in metadata but could not be "
            "fully resolved. Falling back to directory sorting."
        )

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in valid_extensions
        ],
        key=natural_sort_key,
    )

    if len(image_paths) != n_images:
        raise ValueError(
            f"Found {len(image_paths)} image files, but the neural "
            f"matrix contains {n_images} images.\n"
            "Check whether the filenames correspond exactly to the "
            "118 Allen natural-scene images."
        )

    print("Using naturally sorted image files.")
    return image_paths


def fit_animacy_axis(
    X_neurons_by_images: np.ndarray,
    labels: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    LogisticRegression,
    StandardScaler,
]:
    """
    Fit animal-vs-rest logistic regression in sample.

    Returns
    -------
    decision_scores:
        Raw logistic decision values, one per image.

    animal_probabilities:
        Sigmoid/logistic class probabilities, one per image.
    """
    X_images_by_neurons = X_neurons_by_images.T

    y = (labels == ANIMAL_LABEL).astype(np.int64)

    print("\nBinary class counts:")
    print(f"  non-animal: {np.sum(y == 0)}")
    print(f"  animal:     {np.sum(y == 1)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_images_by_neurons)

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
    animal_probabilities = model.predict_proba(X_scaled)[:, 1]

    training_accuracy = model.score(X_scaled, y)

    print(f"\nIn-sample accuracy: {training_accuracy:.4f}")
    print(
        "This score is used descriptively to define an ordering, "
        "not as an unbiased estimate of decoding performance."
    )

    return (
        decision_scores,
        animal_probabilities,
        model,
        scaler,
    )


def run_rastermap(
    X_sorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort individual neurons using Rastermap.

    X_sorted has shape:
        neurons x sorted images
    """
    # Remove neurons with no variance across images.
    neuron_std = np.std(X_sorted, axis=1)
    valid_neurons = np.isfinite(neuron_std) & (neuron_std > 0)

    print(
        f"\nRastermap neurons: {valid_neurons.sum()} / "
        f"{len(valid_neurons)}"
    )

    X_valid = X_sorted[valid_neurons]

    # Row-wise z-scoring lets Rastermap compare response shapes rather
    # than simply grouping neurons by baseline event probability.
    row_mean = X_valid.mean(axis=1, keepdims=True)
    row_std = X_valid.std(axis=1, keepdims=True)

    X_z = (X_valid - row_mean) / np.maximum(row_std, 1e-8)

    model = Rastermap(
    n_clusters=100,
    n_PCs=min(N_PCS, X_z.shape[1] - 1),
    locality=LOCALITY,
    time_lag_window=0,
    grid_upsample=10,
    )

    model.fit(X_z)

    # Current Rastermap exposes a neuron order as isort.
    if hasattr(model, "isort"):
        local_order = np.asarray(model.isort).squeeze()

    # Fallback for versions exposing only embedding.
    elif hasattr(model, "embedding"):
        embedding = np.asarray(model.embedding).squeeze()
        local_order = np.argsort(embedding)

    else:
        raise AttributeError(
            "Rastermap model exposes neither 'isort' nor 'embedding'."
        )

    valid_indices = np.flatnonzero(valid_neurons)
    neuron_order = valid_indices[local_order]

    embedding = np.asarray(
        getattr(model, "embedding", np.arange(len(local_order)))
    ).squeeze()

    return neuron_order, embedding


def bin_neurons(
    matrix: np.ndarray,
    bin_size: int,
) -> np.ndarray:
    """
    Average neighboring Rastermap-sorted neurons for display.

    This reduces ~40,000 rows to a visible number of superneurons
    without changing the underlying saved ordering.
    """
    n_neurons, n_images = matrix.shape
    n_bins = int(np.ceil(n_neurons / bin_size))

    binned = np.empty((n_bins, n_images), dtype=np.float32)

    for bin_index in range(n_bins):
        start = bin_index * bin_size
        stop = min(start + bin_size, n_neurons)
        binned[bin_index] = matrix[start:stop].mean(axis=0)

    return binned


def make_image_strip(
    image_paths_sorted: list[Path],
    decision_scores_sorted: np.ndarray,
    animal_probabilities_sorted: np.ndarray,
    labels_sorted: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save all images in animacy-score order as a labeled contact sheet.
    """
    thumb_size = IMAGE_THUMBNAIL_SIZE
    label_height = 42

    canvas = Image.new(
        "RGB",
        (
            thumb_size * len(image_paths_sorted),
            thumb_size + label_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    for column, image_path in enumerate(image_paths_sorted):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = ImageOps.fit(
                image,
                (thumb_size, thumb_size),
                method=Image.Resampling.LANCZOS,
            )

        x0 = column * thumb_size
        canvas.paste(image, (x0, 0))

        label_text = (
            f"{column}\n"
            f"s={decision_scores_sorted[column]:.2f}\n"
            f"p={animal_probabilities_sorted[column]:.2f}"
        )

        draw.multiline_text(
            (x0 + 3, thumb_size + 1),
            label_text,
            fill="black",
            spacing=0,
        )

        # Add a small mark for the ground-truth animal class.
        if labels_sorted[column] == ANIMAL_LABEL:
            draw.rectangle(
                [
                    x0,
                    0,
                    x0 + thumb_size - 1,
                    thumb_size - 1,
                ],
                outline="red",
                width=3,
            )

    canvas.save(output_path)


def plot_heatmap(
    matrix_sorted: np.ndarray,
    decision_scores_sorted: np.ndarray,
    labels_sorted: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot Rastermap-sorted neural activity.

    Uses per-neuron z-scored probabilities and display binning.
    """
    row_mean = matrix_sorted.mean(axis=1, keepdims=True)
    row_std = matrix_sorted.std(axis=1, keepdims=True)

    matrix_z = (
        matrix_sorted - row_mean
    ) / np.maximum(row_std, 1e-8)

    matrix_display = bin_neurons(
        matrix_z,
        NEURONS_PER_DISPLAY_BIN,
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(15, 10),
        height_ratios=[1, 8],
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        np.arange(len(decision_scores_sorted)),
        decision_scores_sorted,
        linewidth=2,
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
        decision_scores_sorted[animal_positions],
        marker="o",
        s=16,
        label="ground-truth animal",
    )

    axes[0].set_ylabel("Logistic\ndecision score")
    axes[0].set_title(
        "Images sorted from non-animal-like to animal-like"
    )
    axes[0].legend(loc="upper left")

    image = axes[1].imshow(
        matrix_display,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-2.5,
        vmax=2.5,
    )

    axes[1].set_xlabel("Images sorted by neural animacy score")
    axes[1].set_ylabel(
        f"Rastermap-sorted neuron bins\n"
        f"({NEURONS_PER_DISPLAY_BIN} neurons per row)"
    )

    colorbar = figure.colorbar(
        image,
        ax=axes[1],
        fraction=0.02,
        pad=0.01,
    )
    colorbar.set_label("Within-neuron z-scored response probability")

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
    X, labels, neuron_metadata, stimulus_metadata = load_composite(
        DATA_PATH
    )

    print("\nLoaded data:")
    print(f"  X shape:      {X.shape}")
    print(f"  labels shape: {labels.shape}")

    image_paths = find_images(
        IMAGE_DIR,
        stimulus_metadata,
        n_images=X.shape[1],
    )

    (
        decision_scores,
        animal_probabilities,
        logistic_model,
        scaler,
    ) = fit_animacy_axis(X, labels)

    # Sort images from lowest to highest animacy score.
    image_order = np.argsort(decision_scores)

    X_image_sorted = X[:, image_order]
    labels_sorted = labels[image_order]
    scores_sorted = decision_scores[image_order]
    probabilities_sorted = animal_probabilities[image_order]

    image_paths_sorted = [
        image_paths[index]
        for index in image_order
    ]

    # Rastermap sees each neuron as one sample and the 118 sorted
    # image responses as its feature profile.
    neuron_order, rastermap_embedding = run_rastermap(
        X_image_sorted
    )

    X_fully_sorted = X_image_sorted[neuron_order]

    heatmap_path = OUTPUT_DIR / "animacy_rastermap_heatmap.png"
    strip_path = OUTPUT_DIR / "sorted_image_strip.png"
    npz_path = OUTPUT_DIR / "sorted_animacy_results.npz"

    plot_heatmap(
        matrix_sorted=X_fully_sorted,
        decision_scores_sorted=scores_sorted,
        labels_sorted=labels_sorted,
        output_path=heatmap_path,
    )

    make_image_strip(
        image_paths_sorted=image_paths_sorted,
        decision_scores_sorted=scores_sorted,
        animal_probabilities_sorted=probabilities_sorted,
        labels_sorted=labels_sorted,
        output_path=strip_path,
    )

    np.savez_compressed(
        npz_path,
        image_order=image_order,
        neuron_order=neuron_order,
        decision_scores=decision_scores,
        animal_probabilities=animal_probabilities,
        decision_scores_sorted=scores_sorted,
        animal_probabilities_sorted=probabilities_sorted,
        labels=labels,
        labels_sorted=labels_sorted,
        rastermap_embedding=rastermap_embedding,
        logistic_weights=logistic_model.coef_.squeeze(),
        logistic_intercept=logistic_model.intercept_.squeeze(),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        sorted_image_paths=np.asarray(
            [str(path) for path in image_paths_sorted],
            dtype=object,
        ),
    )

    print("\nSaved:")
    print(f"  Heatmap:    {heatmap_path}")
    print(f"  Image strip:{strip_path}")
    print(f"  Data:       {npz_path}")

    print("\nInterpretation:")
    print(
        "  Columns are ordered by the in-sample neural animacy score."
    )
    print(
        "  Rows are ordered by Rastermap based on each neuron's "
        "118-image response profile."
    )
    print(
        "  The heatmap is descriptive: the same population was used "
        "to construct the image axis and visualize tuning to it."
    )


if __name__ == "__main__":
    main()