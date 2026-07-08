#!/usr/bin/env python3
"""
Plot original and synthetic neural image responses in the SYNTHETIC PCA space.

This is the companion to the plot where synthetic data was projected into the
original/real PCA basis. Here we do the opposite:

    1. Fit PCA on the synthetic image-level neural matrix.
    2. Project both synthetic and original neural matrices into that synthetic PCA basis.
    3. Plot PC1 vs PC2 with original points as circles and synthetic points as crosses.

Expected inputs:
    /home/maria/SelfStudyThesis/data/allen_natural_scenes_four_class_composite.npy
    /home/maria/SelfStudyThesis/data/synthetic_neural_activity_image_probs_loo.npy

Expected shapes after loading/alignment:
    X_real:      images × neurons, usually 118 × total_neurons
    X_synthetic: images × neurons, usually 118 × total_neurons

Outputs:
    /home/maria/SelfStudyThesis/results/original_vs_synthetic_four_class_subspace/
        original_and_synthetic_in_synthetic_pca_space.png
        synthetic_pca_scores.npz
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# =============================================================================
# Paths
# =============================================================================

DATA_DIR = Path("/home/maria/SelfStudyThesis/data")

REAL_COMPOSITE_PATH = DATA_DIR / "allen_natural_scenes_four_class_composite.npy"
SYNTHETIC_PATH = DATA_DIR / "synthetic_neural_activity_image_probs_loo.npy"

OUT_DIR = Path(
    "/home/maria/SelfStudyThesis/results/original_vs_synthetic_four_class_subspace"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG = OUT_DIR / "original_and_synthetic_in_synthetic_pca_space.png"
OUT_NPZ = OUT_DIR / "synthetic_pca_scores.npz"


# =============================================================================
# Constants
# =============================================================================

LABEL_NAMES = {
    -1: "unlabeled",
     0: "animals",
     1: "landscape",
     2: "plant",
     3: "man-made object",
}

# Keep matplotlib default colors unless you explicitly want custom colors.
# This list just creates stable class-to-color assignment from the default cycle.
CLASSES = np.array([0, 1, 2, 3], dtype=int)

N_COMPONENTS = 10


# =============================================================================
# Loading helpers
# =============================================================================

def load_real_composite(path: Path):
    print("=" * 80)
    print("Loading real composite")
    print("=" * 80)
    print(f"Path: {path}")

    data = np.load(path, allow_pickle=True).item()
    X = np.asarray(data["X"], dtype=np.float64)
    stim = data["stimulus_metadata"]
    labels = np.asarray(stim["label"], dtype=np.int64).ravel()

    print(f"Raw X shape:  {X.shape}")
    print(f"Labels shape: {labels.shape}")

    # Old image-level composite is usually neurons × 118.
    # We want images × neurons.
    if X.shape[1] == len(labels):
        print("[INFO] Real X appears to be neurons × images; transposing.")
        X = X.T
    elif X.shape[0] == len(labels):
        print("[INFO] Real X already appears to be images × neurons.")
    else:
        raise ValueError(
            f"Cannot align real X with labels. X={X.shape}, labels={labels.shape}"
        )

    print(f"Aligned real X shape, images × neurons: {X.shape}")
    return X, labels, data


def load_synthetic(path: Path, n_images: int):
    print()
    print("=" * 80)
    print("Loading synthetic image probabilities")
    print("=" * 80)
    print(f"Path: {path}")

    Xs = np.asarray(np.load(path, allow_pickle=True), dtype=np.float64)
    print(f"Raw synthetic shape: {Xs.shape}")

    # Synthetic image-level output may be neurons × 118 or 118 × neurons.
    # We want images × neurons.
    if Xs.shape[1] == n_images:
        print("[INFO] Synthetic X appears to be neurons × images; transposing.")
        Xs = Xs.T
    elif Xs.shape[0] == n_images:
        print("[INFO] Synthetic X already appears to be images × neurons.")
    else:
        raise ValueError(
            f"Cannot align synthetic matrix with n_images={n_images}. "
            f"Synthetic shape={Xs.shape}"
        )

    print(f"Aligned synthetic shape, images × neurons: {Xs.shape}")
    return Xs


def check_shapes(X_real: np.ndarray, X_synthetic: np.ndarray, labels: np.ndarray):
    if X_real.shape != X_synthetic.shape:
        raise ValueError(
            f"Real and synthetic shapes differ after alignment: "
            f"real={X_real.shape}, synthetic={X_synthetic.shape}"
        )

    if X_real.shape[0] != len(labels):
        raise ValueError(
            f"Number of images does not match labels: X={X_real.shape}, labels={labels.shape}"
        )

    if not np.all(np.isfinite(X_real)):
        raise ValueError("Real matrix contains NaN or infinite values.")

    if not np.all(np.isfinite(X_synthetic)):
        raise ValueError("Synthetic matrix contains NaN or infinite values.")


# =============================================================================
# PCA + plotting
# =============================================================================

def fit_synthetic_pca_and_project(X_real: np.ndarray, X_synthetic: np.ndarray):
    """
    Fit PCA on synthetic data and project both matrices into that basis.

    PCA.fit_transform centers synthetic by its own mean.
    PCA.transform(real) centers real using the synthetic mean. This is exactly
    what we want if the question is: where does real data land in the coordinate
    system learned from synthetic data?
    """

    n_components = min(N_COMPONENTS, X_synthetic.shape[0], X_synthetic.shape[1])

    pca = PCA(n_components=n_components)
    synthetic_scores = pca.fit_transform(X_synthetic)
    real_scores = pca.transform(X_real)

    print()
    print("=" * 80)
    print("Synthetic PCA")
    print("=" * 80)
    print(f"n_components: {n_components}")
    print("Explained variance ratio:")
    for k, evr in enumerate(pca.explained_variance_ratio_, start=1):
        print(f"  synthetic PC{k:02d}: {evr:.6f}")

    # Correlation between real and synthetic image scores along each synthetic PC.
    score_corr = []
    for k in range(n_components):
        r = real_scores[:, k]
        s = synthetic_scores[:, k]
        if np.std(r) == 0 or np.std(s) == 0:
            score_corr.append(np.nan)
        else:
            score_corr.append(float(np.corrcoef(r, s)[0, 1]))
    score_corr = np.asarray(score_corr, dtype=float)

    print()
    print("Corr(real score, synthetic score) in synthetic PCA basis:")
    for k, corr in enumerate(score_corr, start=1):
        print(f"  synthetic PC{k:02d}: {corr: .6f} |abs|={abs(corr):.6f}")

    return pca, real_scores, synthetic_scores, score_corr


def plot_pc1_pc2(real_scores: np.ndarray, synthetic_scores: np.ndarray, labels: np.ndarray):
    plt.figure(figsize=(10, 8))

    for cls in CLASSES:
        mask = labels == cls
        if not np.any(mask):
            continue

        name = LABEL_NAMES[int(cls)]

        # Original/measured images: circles.
        plt.scatter(
            real_scores[mask, 0],
            real_scores[mask, 1],
            marker="o",
            alpha=0.65,
            label=f"original {name}",
        )

        # Synthetic images: crosses.
        plt.scatter(
            synthetic_scores[mask, 0],
            synthetic_scores[mask, 1],
            marker="x",
            alpha=0.85,
            label=f"synthetic {name}",
        )

    plt.axhline(0.0, linewidth=1, alpha=0.25)
    plt.axvline(0.0, linewidth=1, alpha=0.25)
    plt.xlabel("Synthetic PC1 score")
    plt.ylabel("Synthetic PC2 score")
    plt.title("Original and synthetic images in synthetic PCA space")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200)
    plt.close()

    print()
    print(f"Saved plot to: {OUT_FIG}")


# =============================================================================
# Main
# =============================================================================

def main():
    X_real, labels, real_data = load_real_composite(REAL_COMPOSITE_PATH)
    X_synthetic = load_synthetic(SYNTHETIC_PATH, n_images=len(labels))
    check_shapes(X_real, X_synthetic, labels)

    # Drop unlabeled images if present. Your current four-class data may have none,
    # but this keeps the script safe and consistent with previous regressions.
    keep_mask = labels >= 0
    if not np.all(keep_mask):
        print()
        print(f"[INFO] Dropping unlabeled images: kept {keep_mask.sum()} / {len(labels)}")
        X_real = X_real[keep_mask]
        X_synthetic = X_synthetic[keep_mask]
        labels_kept = labels[keep_mask]
    else:
        labels_kept = labels

    print()
    print("=" * 80)
    print("Final matrices")
    print("=" * 80)
    print(f"Real:      {X_real.shape}  images × neurons")
    print(f"Synthetic: {X_synthetic.shape}  images × neurons")

    pca, real_scores, synthetic_scores, score_corr = fit_synthetic_pca_and_project(
        X_real=X_real,
        X_synthetic=X_synthetic,
    )

    plot_pc1_pc2(real_scores, synthetic_scores, labels_kept)

    np.savez_compressed(
        OUT_NPZ,
        real_scores_in_synthetic_pca=real_scores,
        synthetic_scores_in_synthetic_pca=synthetic_scores,
        labels=labels,
        labels_kept=labels_kept,
        keep_mask=keep_mask,
        synthetic_pca_components=pca.components_,
        synthetic_pca_mean=pca.mean_,
        synthetic_pca_explained_variance=pca.explained_variance_,
        synthetic_pca_explained_variance_ratio=pca.explained_variance_ratio_,
        score_corr_by_pc=score_corr,
        label_names=np.asarray([LABEL_NAMES[int(c)] for c in CLASSES], dtype=object),
        classes=CLASSES,
    )

    print(f"Saved PCA scores to: {OUT_NPZ}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
