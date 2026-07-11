#!/usr/bin/env python3
"""
LOO low-rank PCA reconstruction of Allen natural-scene population responses.

Question
--------
For each held-out image:

    1. Fit the PCA subspace using only the other images.
    2. Center the held-out neural response using the training-fold mean.
    3. Project the held-out response onto the first k training PCs.
    4. Measure how much of its deviation from the training mean is reconstructed.

This is a leakage-safe estimate of how close unseen image responses lie to the
low-rank subspace learned from the remaining images.

Important interpretation
------------------------
This is reconstruction, not prediction. The held-out response itself is used
to obtain its PC scores. No information from the held-out image is used to fit
the PCA basis or the centering mean.

Input composite file:
    /home/maria/SelfStudyThesis/data/allen_natural_scenes_four_class_composite.npy

Expected composite structure:
    data["X"]                              total_neurons × images
    data["stimulus_metadata"]["label"]     image labels
    data["neuron_metadata"]                row-aligned neuron metadata

Four-class labels:
    -1 = unlabeled
     0 = animals
     1 = landscape
     2 = plant
     3 = man-made object

Outputs
-------
    loo_pca_reconstruction_results.npz
    loo_pca_reconstruction_summary.csv
    loo_pca_reconstruction_curve.png
    loo_pca_reconstruction_curve.pdf
    loo_pca_reconstruction_heatmap.png
    loo_pca_reconstruction_heatmap.pdf
    loo_pca_reconstruction_by_class.png
    loo_pca_reconstruction_by_class.pdf

The main metric is:

    centered variance reconstructed(k)
      = ||projection_k(x_test - train_mean)||²
        / ||x_test - train_mean||²

It is 0 for the training-mean-only baseline and 1 for perfect reconstruction.
Because PCA reconstruction is an orthogonal projection, the centered cosine
similarity is sqrt(centered variance reconstructed).
"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================

COMPOSITE_PATH = Path(
    "/home/maria/SelfStudyThesis/data/"
    "allen_natural_scenes_four_class_composite.npy"
)

OUTDIR = Path(
    "/home/maria/SelfStudyThesis/results/"
    "all_neurons_loo_pca_reconstruction"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUTDIR / "loo_pca_reconstruction_results.npz"
OUT_CSV = OUTDIR / "loo_pca_reconstruction_summary.csv"


# =============================================================================
# Settings
# =============================================================================

RANDOM_SEED = 0
EPS = 1e-12

# Analyze labeled images only. Set False to retain label -1 too.
EXCLUDE_UNLABELED = True

# Neurons that are invalid or constant over the complete retained dataset are
# removed once before LOO. This does not use class labels. A stricter fold-local
# variance filter is also applied inside every fold.
GLOBAL_VARIANCE_TOL = 1e-15
FOLD_VARIANCE_TOL = 1e-15

# Number of bootstrap resamples used for the confidence band across held-out
# images. Set to 0 to use a normal-theory 95% CI instead.
N_BOOTSTRAP = 5000

# Selected ranks highlighted in the class plot. Values above the available
# fold rank are silently omitted.
SELECTED_RANKS = [1, 2, 3, 5, 10, 20, 40, 60, 80, 100]

LABEL_NAMES = {
    -1: "unlabeled",
     0: "animals",
     1: "landscape",
     2: "plant",
     3: "man-made object",
}


# =============================================================================
# Data loading, adapted from the supplied loader
# =============================================================================

def load_composite_data():
    print()
    print("#" * 80)
    print("Loading composite dataset")
    print("#" * 80)
    print(f"Composite path: {COMPOSITE_PATH}")

    data = np.load(COMPOSITE_PATH, allow_pickle=True).item()

    X = np.asarray(data["X"], dtype=np.float64)
    stimulus_metadata = data["stimulus_metadata"]
    neuron_metadata = data["neuron_metadata"]

    labels = np.asarray(stimulus_metadata["label"], dtype=np.int64).ravel()

    print()
    print("=" * 80)
    print("Raw composite contents")
    print("=" * 80)
    print(f"X raw shape:             {X.shape}")
    print(f"labels shape:            {labels.shape}")
    print(f"neuron metadata keys:    {list(neuron_metadata.keys())}")
    print(f"stimulus metadata keys:  {list(stimulus_metadata.keys())}")

    print()
    print("Raw four-class label counts:")
    unique, counts = np.unique(labels, return_counts=True)
    for value, count in zip(unique, counts):
        name = LABEL_NAMES.get(int(value), "UNKNOWN")
        print(f"  {int(value):>2} = {name:<16} n={int(count)}")

    # Composite X is usually neurons × images.
    if X.shape[1] == len(labels):
        print()
        print("[INFO] Transposing X from neurons × images to images × neurons.")
        X = X.T
    elif X.shape[0] == len(labels):
        print()
        print("[INFO] X already appears to be images × neurons.")
    else:
        raise ValueError(
            f"Cannot align X with labels. X={X.shape}, labels={labels.shape}. "
            "Expected one X axis to match the number of image labels."
        )

    n_features = X.shape[1]

    if "brain_area" in neuron_metadata:
        brain_area = np.asarray(
            neuron_metadata["brain_area"], dtype=object
        ).ravel()
    else:
        brain_area = np.full(n_features, "unknown", dtype=object)

    if "cell_specimen_id" in neuron_metadata:
        cell_specimen_id = np.asarray(
            neuron_metadata["cell_specimen_id"], dtype=np.int64
        ).ravel()
    else:
        cell_specimen_id = np.full(n_features, -1, dtype=np.int64)

    if "ophys_experiment_id" in neuron_metadata:
        ophys_experiment_id = np.asarray(
            neuron_metadata["ophys_experiment_id"], dtype=np.int64
        ).ravel()
    else:
        ophys_experiment_id = np.full(n_features, -1, dtype=np.int64)

    for name, values in (
        ("brain_area", brain_area),
        ("cell_specimen_id", cell_specimen_id),
        ("ophys_experiment_id", ophys_experiment_id),
    ):
        if len(values) != n_features:
            raise ValueError(
                f"{name} length does not match feature count. "
                f"{name}={len(values)}, features={n_features}"
            )

    print()
    print("=" * 80)
    print("After orientation alignment")
    print("=" * 80)
    print(f"X shape, images × neurons: {X.shape}")
    print(f"Number of neurons/features: {n_features}")

    return (
        data,
        X,
        labels,
        brain_area,
        cell_specimen_id,
        ophys_experiment_id,
    )


# =============================================================================
# Preprocessing
# =============================================================================

def retain_valid_data(
    X: np.ndarray,
    labels: np.ndarray,
    brain_area: np.ndarray,
    cell_specimen_id: np.ndarray,
    ophys_experiment_id: np.ndarray,
):
    image_mask = np.ones(len(labels), dtype=bool)
    if EXCLUDE_UNLABELED:
        image_mask &= labels != -1

    X = X[image_mask]
    labels = labels[image_mask]
    original_image_indices = np.flatnonzero(image_mask)

    finite_mask = np.all(np.isfinite(X), axis=0)
    variance = np.var(X, axis=0, ddof=0)
    variable_mask = variance > GLOBAL_VARIANCE_TOL
    neuron_mask = finite_mask & variable_mask

    print()
    print("=" * 80)
    print("Retained data")
    print("=" * 80)
    print(f"Retained images:             {X.shape[0]}")
    print(f"Retained finite neurons:     {int(finite_mask.sum())}")
    print(f"Retained variable neurons:   {int(neuron_mask.sum())}")
    print(f"Removed neurons:             {int((~neuron_mask).sum())}")

    X = X[:, neuron_mask]

    return (
        X,
        labels,
        original_image_indices,
        neuron_mask,
        brain_area[neuron_mask],
        cell_specimen_id[neuron_mask],
        ophys_experiment_id[neuron_mask],
    )


# =============================================================================
# LOO PCA reconstruction
# =============================================================================

def loo_pca_reconstruction(X: np.ndarray):
    """
    Compute LOO reconstruction without forming a neurons × PCs matrix.

    For one fold, with centered training matrix A:

        A = U S V^T

    For centered held-out vector x, its PC scores are:

        x V = (x A^T U) / S

    Since the PCs are orthonormal, the squared norm reconstructed by the first
    k PCs is simply the cumulative sum of squared scores.
    """
    n_images, n_neurons = X.shape
    max_rank = n_images - 2  # centered training set has n_train - 1 rank

    ranks = np.arange(max_rank + 1, dtype=np.int64)

    reconstructed_fraction = np.full(
        (n_images, max_rank + 1), np.nan, dtype=np.float64
    )
    normalized_residual = np.full_like(reconstructed_fraction, np.nan)
    centered_cosine = np.full_like(reconstructed_fraction, np.nan)

    fold_rank = np.zeros(n_images, dtype=np.int64)
    fold_n_neurons = np.zeros(n_images, dtype=np.int64)
    heldout_centered_norm2 = np.zeros(n_images, dtype=np.float64)
    singular_values = np.full((n_images, max_rank), np.nan, dtype=np.float64)
    train_explained_variance_ratio = np.full(
        (n_images, max_rank), np.nan, dtype=np.float64
    )

    for test_idx in range(n_images):
        train_mask = np.ones(n_images, dtype=bool)
        train_mask[test_idx] = False

        X_train = X[train_mask]
        x_test = X[test_idx]

        # Fold-local filter. It is learned using training responses only.
        fold_var = np.var(X_train, axis=0, ddof=0)
        fold_feature_mask = fold_var > FOLD_VARIANCE_TOL

        A_raw = X_train[:, fold_feature_mask]
        x_raw = x_test[fold_feature_mask]
        fold_n_neurons[test_idx] = int(fold_feature_mask.sum())

        train_mean = A_raw.mean(axis=0)
        A = A_raw - train_mean
        x = x_raw - train_mean

        norm2 = float(x @ x)
        heldout_centered_norm2[test_idx] = norm2

        # Small sample-space Gram matrix: n_train × n_train.
        gram = A @ A.T
        eigvals, U = np.linalg.eigh(gram)

        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0.0, None)
        U = U[:, order]

        if eigvals.size == 0 or eigvals[0] <= EPS:
            reconstructed_fraction[test_idx, 0] = 0.0
            normalized_residual[test_idx, 0] = 1.0
            centered_cosine[test_idx, 0] = 0.0
            continue

        rank_tol = max(A.shape) * np.finfo(np.float64).eps * eigvals[0]
        positive = eigvals > rank_tol
        r = min(int(positive.sum()), max_rank)

        fold_rank[test_idx] = r

        # Rank 0 is the training-mean-only reconstruction.
        reconstructed_fraction[test_idx, 0] = 0.0
        normalized_residual[test_idx, 0] = 1.0
        centered_cosine[test_idx, 0] = 0.0

        if r == 0 or norm2 <= EPS:
            continue

        eigvals_r = eigvals[:r]
        U_r = U[:, :r]
        s_r = np.sqrt(eigvals_r)

        # x @ A.T gives similarities between the held-out centered response and
        # every centered training image.
        similarities = x @ A.T
        scores = (similarities @ U_r) / s_r

        cumulative_projected_norm2 = np.cumsum(scores ** 2)
        fraction = np.clip(
            cumulative_projected_norm2 / max(norm2, EPS),
            0.0,
            1.0,
        )

        reconstructed_fraction[test_idx, 1 : r + 1] = fraction
        normalized_residual[test_idx, 1 : r + 1] = 1.0 - fraction
        centered_cosine[test_idx, 1 : r + 1] = np.sqrt(fraction)

        # Past the numerical fold rank, reconstruction cannot improve.
        if r < max_rank:
            reconstructed_fraction[test_idx, r + 1 :] = fraction[-1]
            normalized_residual[test_idx, r + 1 :] = 1.0 - fraction[-1]
            centered_cosine[test_idx, r + 1 :] = np.sqrt(fraction[-1])

        singular_values[test_idx, :r] = s_r

        total_train_variance = eigvals_r.sum()
        if total_train_variance > EPS:
            train_explained_variance_ratio[test_idx, :r] = (
                eigvals_r / total_train_variance
            )

        print(
            f"\rLOO fold {test_idx + 1:>3}/{n_images}: "
            f"rank={r:>3}, neurons={fold_n_neurons[test_idx]:>6}",
            end="",
            flush=True,
        )

    print()

    return {
        "ranks": ranks,
        "reconstructed_fraction": reconstructed_fraction,
        "normalized_residual": normalized_residual,
        "centered_cosine": centered_cosine,
        "fold_rank": fold_rank,
        "fold_n_neurons": fold_n_neurons,
        "heldout_centered_norm2": heldout_centered_norm2,
        "singular_values": singular_values,
        "train_explained_variance_ratio": train_explained_variance_ratio,
    }


# =============================================================================
# Summaries and confidence intervals
# =============================================================================

def column_summary(values: np.ndarray, rng: np.random.Generator):
    mean = np.nanmean(values, axis=0)
    median = np.nanmedian(values, axis=0)

    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)

    if N_BOOTSTRAP > 0:
        n = values.shape[0]
        boot_means = np.empty((N_BOOTSTRAP, values.shape[1]), dtype=np.float64)
        for b in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            boot_means[b] = np.nanmean(values[idx], axis=0)
        ci_low = np.nanpercentile(boot_means, 2.5, axis=0)
        ci_high = np.nanpercentile(boot_means, 97.5, axis=0)
    else:
        n_eff = np.sum(np.isfinite(values), axis=0)
        sem = np.nanstd(values, axis=0, ddof=1) / np.sqrt(
            np.maximum(n_eff, 1)
        )
        ci_low = mean - 1.96 * sem
        ci_high = mean + 1.96 * sem

    return {
        "mean": mean,
        "median": median,
        "q25": q25,
        "q75": q75,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def ranks_for_threshold(
    reconstructed_fraction: np.ndarray,
    thresholds=(0.50, 0.75, 0.90, 0.95),
):
    result = {}
    for threshold in thresholds:
        reached = reconstructed_fraction >= threshold
        first = np.argmax(reached, axis=1).astype(np.float64)
        never = ~np.any(reached, axis=1)
        first[never] = np.nan
        result[threshold] = first
    return result


# =============================================================================
# Plotting
# =============================================================================

def plot_reconstruction_curve(
    ranks: np.ndarray,
    summary: dict[str, np.ndarray],
    outdir: Path,
):
    fig, ax = plt.subplots(figsize=(9.0, 5.8))

    ax.fill_between(
        ranks,
        summary["ci_low"],
        summary["ci_high"],
        alpha=0.22,
        label="95% bootstrap CI of mean",
    )
    ax.plot(
        ranks,
        summary["mean"],
        linewidth=2.5,
        label="Mean held-out reconstruction",
    )
    ax.plot(
        ranks,
        summary["median"],
        linewidth=1.8,
        linestyle="--",
        label="Median held-out reconstruction",
    )

    for level in (0.50, 0.75, 0.90, 0.95):
        ax.axhline(level, linewidth=0.8, linestyle=":", alpha=0.55)

    ax.set(
        xlabel="Number of training-fold principal components",
        ylabel="Fraction of held-out centered response reconstructed",
        title="Leave-one-image-out low-rank reconstruction",
        xlim=(0, ranks[-1]),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()

    for suffix in ("png", "pdf"):
        fig.savefig(
            outdir / f"loo_pca_reconstruction_curve.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_reconstruction_heatmap(
    ranks: np.ndarray,
    reconstructed_fraction: np.ndarray,
    labels: np.ndarray,
    outdir: Path,
):
    # Sort by class, then by rank-10 reconstruction where available.
    reference_rank = min(10, ranks[-1])
    order = np.lexsort(
        (
            reconstructed_fraction[:, reference_rank],
            labels,
        )
    )

    matrix = reconstructed_fraction[order]

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Centered response reconstructed")

    ax.set(
        xlabel="Number of principal components",
        ylabel="Held-out image, sorted by class",
        title="LOO reconstruction for every held-out image",
    )

    tick_candidates = [0, 1, 2, 3, 5, 10, 20, 40, 60, 80, 100, ranks[-1]]
    ticks = sorted({x for x in tick_candidates if 0 <= x <= ranks[-1]})
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)

    # Mark class boundaries.
    sorted_labels = labels[order]
    boundaries = np.flatnonzero(np.diff(sorted_labels) != 0) + 0.5
    for boundary in boundaries:
        ax.axhline(boundary, linewidth=1.0)

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            outdir / f"loo_pca_reconstruction_heatmap.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_by_class(
    ranks: np.ndarray,
    reconstructed_fraction: np.ndarray,
    labels: np.ndarray,
    outdir: Path,
):
    selected = [r for r in SELECTED_RANKS if r <= ranks[-1]]
    classes = [int(x) for x in np.unique(labels)]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    width = 0.8 / max(len(classes), 1)
    x = np.arange(len(selected), dtype=float)

    for class_offset, class_value in enumerate(classes):
        class_mask = labels == class_value
        values = reconstructed_fraction[class_mask][:, selected]
        means = np.nanmean(values, axis=0)
        sem = np.nanstd(values, axis=0, ddof=1) / np.sqrt(class_mask.sum())

        positions = x - 0.4 + width / 2 + class_offset * width
        ax.bar(
            positions,
            means,
            width=width,
            yerr=1.96 * sem,
            capsize=2,
            label=LABEL_NAMES.get(class_value, str(class_value)),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(selected)
    ax.set(
        xlabel="Number of principal components",
        ylabel="Mean held-out centered response reconstructed",
        title="LOO low-rank reconstruction by stimulus class",
        ylim=(0, 1.02),
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    for suffix in ("png", "pdf"):
        fig.savefig(
            outdir / f"loo_pca_reconstruction_by_class.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


# =============================================================================
# Saving
# =============================================================================

def save_summary_csv(
    path: Path,
    ranks: np.ndarray,
    summary: dict[str, np.ndarray],
):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "n_components",
                "mean_reconstructed_fraction",
                "median_reconstructed_fraction",
                "q25_reconstructed_fraction",
                "q75_reconstructed_fraction",
                "ci95_low_mean",
                "ci95_high_mean",
                "mean_normalized_residual",
                "mean_centered_cosine",
            ]
        )

        for idx, rank in enumerate(ranks):
            mean_fraction = summary["mean"][idx]
            writer.writerow(
                [
                    int(rank),
                    float(mean_fraction),
                    float(summary["median"][idx]),
                    float(summary["q25"][idx]),
                    float(summary["q75"][idx]),
                    float(summary["ci_low"][idx]),
                    float(summary["ci_high"][idx]),
                    float(1.0 - mean_fraction),
                    float(np.sqrt(max(mean_fraction, 0.0))),
                ]
            )


# =============================================================================
# Main
# =============================================================================

def main():
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    (
        _data,
        X,
        labels,
        brain_area,
        cell_specimen_id,
        ophys_experiment_id,
    ) = load_composite_data()

    (
        X,
        labels,
        original_image_indices,
        neuron_mask,
        brain_area,
        cell_specimen_id,
        ophys_experiment_id,
    ) = retain_valid_data(
        X,
        labels,
        brain_area,
        cell_specimen_id,
        ophys_experiment_id,
    )

    if X.shape[0] < 4:
        raise ValueError("At least four retained images are required.")

    results = loo_pca_reconstruction(X)

    summary = column_summary(results["reconstructed_fraction"], rng)
    threshold_ranks = ranks_for_threshold(results["reconstructed_fraction"])

    print()
    print("=" * 80)
    print("LOO low-rank reconstruction summary")
    print("=" * 80)
    print(f"Images:                  {X.shape[0]}")
    print(f"Globally retained neurons: {X.shape[1]}")
    print(
        f"Fold numerical rank:     "
        f"{int(results['fold_rank'].min())}–"
        f"{int(results['fold_rank'].max())}"
    )

    for threshold, first_ranks in threshold_ranks.items():
        finite = first_ranks[np.isfinite(first_ranks)]
        if finite.size:
            print(
                f"PCs for {threshold:.0%} reconstruction: "
                f"median={np.median(finite):.1f}, "
                f"IQR=[{np.percentile(finite, 25):.1f}, "
                f"{np.percentile(finite, 75):.1f}], "
                f"reached by {finite.size}/{len(first_ranks)} images"
            )
        else:
            print(
                f"PCs for {threshold:.0%} reconstruction: "
                "threshold not reached"
            )

    selected_print_ranks = [
        r for r in (1, 2, 3, 5, 10, 20, 40, 60, 80, 100)
        if r <= results["ranks"][-1]
    ]
    print()
    print("Mean held-out centered response reconstructed:")
    for rank in selected_print_ranks:
        print(
            f"  k={rank:>3}: "
            f"{summary['mean'][rank]:.4f} "
            f"[95% CI {summary['ci_low'][rank]:.4f}, "
            f"{summary['ci_high'][rank]:.4f}]"
        )

    np.savez_compressed(
        OUT_NPZ,
        X_shape=np.asarray(X.shape, dtype=np.int64),
        labels=labels,
        original_image_indices=original_image_indices,
        neuron_mask=neuron_mask,
        brain_area=brain_area,
        cell_specimen_id=cell_specimen_id,
        ophys_experiment_id=ophys_experiment_id,
        ranks=results["ranks"],
        reconstructed_fraction=results["reconstructed_fraction"],
        normalized_residual=results["normalized_residual"],
        centered_cosine=results["centered_cosine"],
        fold_rank=results["fold_rank"],
        fold_n_neurons=results["fold_n_neurons"],
        heldout_centered_norm2=results["heldout_centered_norm2"],
        singular_values=results["singular_values"],
        train_explained_variance_ratio=results[
            "train_explained_variance_ratio"
        ],
        mean_reconstructed_fraction=summary["mean"],
        median_reconstructed_fraction=summary["median"],
        q25_reconstructed_fraction=summary["q25"],
        q75_reconstructed_fraction=summary["q75"],
        ci95_low_mean=summary["ci_low"],
        ci95_high_mean=summary["ci_high"],
        threshold_50_rank=threshold_ranks[0.50],
        threshold_75_rank=threshold_ranks[0.75],
        threshold_90_rank=threshold_ranks[0.90],
        threshold_95_rank=threshold_ranks[0.95],
        random_seed=np.int64(RANDOM_SEED),
        n_bootstrap=np.int64(N_BOOTSTRAP),
    )

    save_summary_csv(OUT_CSV, results["ranks"], summary)

    plot_reconstruction_curve(results["ranks"], summary, OUTDIR)
    plot_reconstruction_heatmap(
        results["ranks"],
        results["reconstructed_fraction"],
        labels,
        OUTDIR,
    )
    plot_by_class(
        results["ranks"],
        results["reconstructed_fraction"],
        labels,
        OUTDIR,
    )

    print()
    print("=" * 80)
    print("Saved")
    print("=" * 80)
    print(OUT_NPZ)
    print(OUT_CSV)
    print(OUTDIR / "loo_pca_reconstruction_curve.png")
    print(OUTDIR / "loo_pca_reconstruction_heatmap.png")
    print(OUTDIR / "loo_pca_reconstruction_by_class.png")


if __name__ == "__main__":
    main()
