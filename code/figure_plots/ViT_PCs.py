#!/usr/bin/env python3

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


INPUT_PATH = Path(
    "/home/maria/Documents/HuggingMouseData/MouseViTEmbeddings/"
    "google_vit-base-patch16-224_embeddings_logits.pkl"
)

OUTPUT_DIR = Path(
    "/home/maria/Documents/HuggingMouseData/MouseViTEmbeddings/results"
)


def to_numpy(x):
    """Convert NumPy arrays, lists, or PyTorch tensors to NumPy."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    return np.asarray(x)


def extract_logits(natural_scenes):
    """
    Extract the logits array from the object stored under 'natural_scenes'.

    Handles either:
        data["natural_scenes"] = logits_array

    or:
        data["natural_scenes"] = {
            "logits": logits_array,
            ...
        }
    """
    if isinstance(natural_scenes, dict):
        print("Keys under 'natural_scenes':", list(natural_scenes.keys()))

        likely_keys = [
            "logits",
            "image_logits",
            "vit_logits",
            "predictions",
        ]

        for key in likely_keys:
            if key in natural_scenes:
                print(f"Using natural_scenes['{key}']")
                return to_numpy(natural_scenes[key])

        raise KeyError(
            "The value under 'natural_scenes' is a dictionary, but no "
            "recognized logits key was found. Available keys are: "
            f"{list(natural_scenes.keys())}"
        )

    print("Using data['natural_scenes'] directly as the logits array.")
    return to_numpy(natural_scenes)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading:\n{INPUT_PATH}")

    with INPUT_PATH.open("rb") as file:
        data = pickle.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected the pickle to contain a dictionary, "
            f"but found {type(data).__name__}."
        )

    print("\nTop-level keys:")
    print(list(data.keys()))

    if "natural_scenes" not in data:
        raise KeyError(
            "'natural_scenes' was not found in the pickle dictionary."
        )

    logits = extract_logits(data["natural_scenes"])
    logits = np.asarray(logits, dtype=np.float64)

    print(f"\nOriginal logits shape: {logits.shape}")

    # Remove dimensions of length one, such as (118, 1, 1000).
    logits = np.squeeze(logits)

    if logits.ndim != 2:
        raise ValueError(
            "PCA expects a two-dimensional matrix of shape "
            "(n_images, n_logit_features), but after squeezing the "
            f"logits have shape {logits.shape}."
        )

    # PCA treats rows as observations and columns as features.
    # ViT logits commonly have shape (number of images, 1000 classes).
    #
    # This check catches the common accidental transpose where the data
    # instead have shape (1000 classes, 118 images).
    if logits.shape[0] > logits.shape[1]:
        print(
            "\nWarning: there are more rows than columns. "
            "If rows currently represent logit features and columns represent "
            "images, transpose the array before PCA."
        )

    if not np.isfinite(logits).all():
        n_bad = np.size(logits) - np.isfinite(logits).sum()
        raise ValueError(
            f"The logits contain {n_bad} NaN or infinite values."
        )

    print(f"PCA input shape: {logits.shape}")
    print(f"Number of images: {logits.shape[0]}")
    print(f"Number of logit features: {logits.shape[1]}")

    pca = PCA()
    transformed_logits = pca.fit_transform(logits)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # searchsorted returns the first zero-based position where 0.90 is reached.
    # Add one to convert the index into a number of components.
    n_components_90 = int(
        np.searchsorted(cumulative_variance, 0.90) + 1
    )

    variance_at_threshold = cumulative_variance[n_components_90 - 1]

    print("\n" + "=" * 65)
    print(
        f"PCs required to explain at least 90% of the variance: "
        f"{n_components_90}"
    )
    print(
        f"Cumulative variance explained by those PCs: "
        f"{variance_at_threshold:.6f} "
        f"({100 * variance_at_threshold:.2f}%)"
    )
    print("=" * 65)

    print("\nVariance explained by the first 10 PCs:")
    n_to_show = min(10, len(pca.explained_variance_ratio_))

    for index in range(n_to_show):
        print(
            f"PC{index + 1:02d}: "
            f"individual = "
            f"{100 * pca.explained_variance_ratio_[index]:7.3f}% | "
            f"cumulative = "
            f"{100 * cumulative_variance[index]:7.3f}%"
        )

    # Save the PCA results for later analyses.
    results_path = OUTPUT_DIR / "natural_scenes_logits_pca.npz"

    np.savez_compressed(
        results_path,
        transformed_logits=transformed_logits,
        components=pca.components_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        cumulative_explained_variance=cumulative_variance,
        mean=pca.mean_,
        n_components_90=n_components_90,
    )

    # Plot cumulative explained variance.
    figure_path = (
        OUTPUT_DIR /
        "natural_scenes_logits_pca_cumulative_variance.png"
    )

    component_numbers = np.arange(1, len(cumulative_variance) + 1)

    plt.figure(figsize=(9, 6))
    plt.plot(component_numbers, cumulative_variance, linewidth=2)
    plt.axhline(
        0.90,
        linestyle="--",
        linewidth=1.5,
        label="90% variance",
    )
    plt.axvline(
        n_components_90,
        linestyle="--",
        linewidth=1.5,
        label=f"{n_components_90} PCs",
    )
    plt.scatter(
        n_components_90,
        variance_at_threshold,
        zorder=3,
    )

    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA of ViT logits for natural scenes")
    plt.ylim(0, 1.01)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved PCA results to:\n{results_path}")
    print(f"\nSaved variance plot to:\n{figure_path}")


if __name__ == "__main__":
    main()