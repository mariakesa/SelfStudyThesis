#!/usr/bin/env python3
"""
Compare a supervised 3D encoding subspace with the leading unsupervised PCA
subspace of the real Allen natural-scene responses.

Scientific question
-------------------
Did unsupervised PCA of the real neural responses recover the same structure
that appears in the supervised four-class encoding predictions?

Let:

    E = [e1, e2, e3]

be the first three PCA directions of the supervised synthetic/encoding matrix,
and:

    U_k = [u1, ..., uk]

be the first k PCA directions of the real neural-response matrix.

For each encoding direction e_j, compute:

    r_j^2(k) = || U_k^T e_j ||_2^2

This is the squared fraction of e_j contained in the real-data PCA subspace.

For the full 3D encoding subspace, compute:

    C(k) = (1 / 3) || U_k^T E ||_F^2

and the three principal angles between span(E) and span(U_k).

Two geometries are analyzed:
    1. centered:
       neuron-wise mean centering only

    2. standardized:
       StandardScaler fit on the real matrix, then applied to both real and
       synthetic matrices. This matches the standardized PCA geometry used in
       the PCA animacy-decoding analysis.

Permutation null
----------------
The exact one-hot encoding model is rebuilt after permuting the four-class image
labels. The binary trial-level composite is used so the null follows the same
model family as the original synthetic matrix.

Outputs
-------
    subspace_overlap_summary.csv
    principal_angles.csv
    permutation_summary.csv
    encoding_in_real_pca_overlap.npz

    overlap_curves.png / .pdf
    principal_angles.png / .pdf
    overlap_at_k60.png / .pdf
    observed_vs_null_overlap.png / .pdf

Default paths are taken from the supplied analysis scripts.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Paths
# =============================================================================

REAL_COMPOSITE_PATH = Path(
    "/home/maria/SelfStudyThesis/data/"
    "allen_natural_scenes_four_class_composite.npy"
)

SYNTHETIC_PATH = Path(
    "/home/maria/SelfStudyThesis/results/"
    "four_class_logistic_design_loo_image_probs/"
    "synthetic_neural_activity_image_probs_loo.npy"
)

BINARY_TRIAL_COMPOSITE_PATH = Path(
    "/home/maria/SelfStudyThesis/data/"
    "allen_natural_scenes_four_class_binary_trials_composite.npy"
)

OUTDIR = Path(
    "/home/maria/SelfStudyThesis/results/"
    "encoding_in_unsupervised_pca_subspace"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTDIR / "subspace_overlap_summary.csv"
ANGLES_CSV = OUTDIR / "principal_angles.csv"
PERMUTATION_CSV = OUTDIR / "permutation_summary.csv"
RESULTS_NPZ = OUTDIR / "encoding_in_real_pca_overlap.npz"
INTERPRETATION_TXT = OUTDIR / "interpretation.txt"


# =============================================================================
# Settings
# =============================================================================

RANDOM_SEED = 0
N_ENCODING_PCS = 3
K_VALUES = [1, 2, 3, 5, 10, 20, 40, 60, 80, 100, 116]
MODES = ("centered", "standardized")

# Set to 0 to skip the permutation null.
N_PERMUTATIONS = 1000
N_PERMUTATIONS=0

N_IMAGES = 118
N_TRIALS = 50
EPS = 1e-12

LABEL_NAMES = {
    -1: "unlabeled",
     0: "animals",
     1: "landscape",
     2: "plant",
     3: "man-made object",
}

CLASSES = np.array([0, 1, 2, 3], dtype=np.int64)


# =============================================================================
# Loading and alignment
# =============================================================================

def load_real_and_synthetic():
    print()
    print("#" * 80)
    print("Loading real and supervised synthetic matrices")
    print("#" * 80)
    print(f"Real composite: {REAL_COMPOSITE_PATH}")
    print(f"Synthetic:      {SYNTHETIC_PATH}")

    data = np.load(REAL_COMPOSITE_PATH, allow_pickle=True).item()

    X_real = np.asarray(data["X"], dtype=np.float64)
    labels = np.asarray(
        data["stimulus_metadata"]["label"], dtype=np.int64
    ).ravel()

    if X_real.shape[1] == len(labels):
        X_real = X_real.T
    elif X_real.shape[0] == len(labels):
        pass
    else:
        raise ValueError(
            f"Cannot align real X={X_real.shape} with labels={labels.shape}."
        )

    X_synth = np.asarray(
        np.load(SYNTHETIC_PATH, allow_pickle=True),
        dtype=np.float64,
    )

    if X_synth.shape == X_real.T.shape:
        X_synth = X_synth.T
    elif X_synth.shape == X_real.shape:
        pass
    else:
        raise ValueError(
            f"Real and synthetic shapes do not align: "
            f"real={X_real.shape}, synthetic={X_synth.shape}."
        )

    keep_images = labels >= 0
    X_real = X_real[keep_images]
    X_synth = X_synth[keep_images]
    labels_kept = labels[keep_images]
    original_image_indices = np.flatnonzero(keep_images)

    finite = np.all(np.isfinite(X_real), axis=0)
    finite &= np.all(np.isfinite(X_synth), axis=0)

    real_var = np.var(X_real, axis=0, ddof=0)
    synth_var = np.var(X_synth, axis=0, ddof=0)

    # Keep neurons that can contribute to either geometry, while ensuring
    # StandardScaler and PCA remain numerically valid for the real matrix.
    neuron_mask = finite & (real_var > EPS) & (synth_var > EPS)

    X_real = X_real[:, neuron_mask]
    X_synth = X_synth[:, neuron_mask]

    print()
    print("Aligned data:")
    print(f"  real:               {X_real.shape} images × neurons")
    print(f"  synthetic:          {X_synth.shape} images × neurons")
    print(f"  retained images:    {len(labels_kept)}")
    print(f"  retained neurons:   {int(neuron_mask.sum())}")
    print(f"  removed neurons:    {int((~neuron_mask).sum())}")

    print()
    print("Label counts:")
    values, counts = np.unique(labels_kept, return_counts=True)
    for value, count in zip(values, counts):
        print(
            f"  {int(value):>2} = "
            f"{LABEL_NAMES.get(int(value), 'UNKNOWN'):<16} "
            f"n={int(count)}"
        )

    return (
        X_real,
        X_synth,
        labels_kept,
        original_image_indices,
        neuron_mask,
    )


def load_binary_trial_summaries(
    expected_neuron_mask: np.ndarray,
    original_image_indices: np.ndarray,
):
    """
    Load binary trials and return image sums/counts for the exact neuron subset
    used in the real-vs-synthetic geometry.
    """
    if not BINARY_TRIAL_COMPOSITE_PATH.exists():
        raise FileNotFoundError(
            "Permutation null requested, but binary trial composite is missing:\n"
            f"{BINARY_TRIAL_COMPOSITE_PATH}"
        )

    data = np.load(
        BINARY_TRIAL_COMPOSITE_PATH,
        allow_pickle=True,
    ).item()

    X = np.asarray(data["X"])
    metadata = data["stimulus_metadata"]

    image_index_by_column = np.asarray(
        metadata["image_index"], dtype=np.int64
    ).ravel()

    if X.ndim != 2:
        raise ValueError(f"Binary trial X must be 2D, got {X.shape}.")

    # Expected orientation is neurons × trial columns.
    if X.shape[1] != len(image_index_by_column):
        if X.shape[0] == len(image_index_by_column):
            X = X.T
        else:
            raise ValueError(
                "Cannot align binary trial matrix with image_index metadata."
            )

    if X.shape[0] != len(expected_neuron_mask):
        raise ValueError(
            "Binary and image-level composites do not share the same neuron "
            f"count: binary={X.shape[0]}, image-level={len(expected_neuron_mask)}."
        )

    X = X[expected_neuron_mask]
    X = X.astype(np.float32, copy=False)

    n_images_total = int(image_index_by_column.max()) + 1
    if n_images_total != N_IMAGES:
        raise ValueError(
            f"Expected {N_IMAGES} images in binary composite, got {n_images_total}."
        )

    image_sums = np.zeros((X.shape[0], N_IMAGES), dtype=np.float64)
    image_counts = np.zeros((X.shape[0], N_IMAGES), dtype=np.float64)

    for image_idx in range(N_IMAGES):
        cols = image_index_by_column == image_idx
        Xi = X[:, cols]
        finite = np.isfinite(Xi)
        image_sums[:, image_idx] = np.where(finite, Xi, 0.0).sum(axis=1)
        image_counts[:, image_idx] = finite.sum(axis=1)

    image_sums = image_sums[:, original_image_indices]
    image_counts = image_counts[:, original_image_indices]

    return image_sums, image_counts


# =============================================================================
# Geometry
# =============================================================================

def preprocess_pair(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    mode: str,
):
    if mode == "centered":
        real_mean = X_real.mean(axis=0, keepdims=True)
        synth_mean = X_synth.mean(axis=0, keepdims=True)
        return X_real - real_mean, X_synth - synth_mean, None

    if mode == "standardized":
        scaler = StandardScaler(with_mean=True, with_std=True)
        real_processed = scaler.fit_transform(X_real)
        synth_processed = scaler.transform(X_synth)
        return real_processed, synth_processed, scaler

    raise ValueError(f"Unknown mode: {mode}")


def fit_pca_bases(
    X_real_processed: np.ndarray,
    X_synth_processed: np.ndarray,
):
    max_k = max(K_VALUES)

    pca_real = PCA(
        n_components=max_k,
        svd_solver="full",
    )
    pca_synth = PCA(
        n_components=N_ENCODING_PCS,
        svd_solver="full",
    )

    pca_real.fit(X_real_processed)
    pca_synth.fit(X_synth_processed)

    # sklearn components_: components × neurons
    U = pca_real.components_.T
    E = pca_synth.components_.T

    return pca_real, pca_synth, U, E


def compute_overlap_metrics(
    U: np.ndarray,
    E: np.ndarray,
    mode: str,
):
    overlap_rows = []
    angle_rows = []

    for k in K_VALUES:
        U_k = U[:, :k]

        projection_coordinates = U_k.T @ E
        retained = np.sum(projection_coordinates ** 2, axis=0)
        retained = np.clip(retained, 0.0, 1.0)

        total_capture = float(
            np.sum(projection_coordinates ** 2) / E.shape[1]
        )

        angles = np.degrees(subspace_angles(U_k, E))
        # scipy returns angles in descending order.
        angles_ascending = np.sort(angles)

        row = {
            "mode": mode,
            "k": int(k),
            "encoding_pc1_retained_fraction": float(retained[0]),
            "encoding_pc2_retained_fraction": float(retained[1]),
            "encoding_pc3_retained_fraction": float(retained[2]),
            "encoding_pc1_residual_fraction": float(1.0 - retained[0]),
            "encoding_pc2_residual_fraction": float(1.0 - retained[1]),
            "encoding_pc3_residual_fraction": float(1.0 - retained[2]),
            "mean_3d_subspace_capture": total_capture,
        }
        overlap_rows.append(row)

        angle_row = {
            "mode": mode,
            "k": int(k),
            "n_defined_angles": int(len(angles_ascending)),
            "smallest_principal_angle_deg": float(angles_ascending[0]),
            "mean_principal_angle_deg": float(np.mean(angles_ascending)),
            "largest_principal_angle_deg": float(angles_ascending[-1]),
        }

        for j in range(N_ENCODING_PCS):
            angle_row[f"principal_angle_{j + 1}_deg"] = (
                float(angles_ascending[j])
                if j < len(angles_ascending)
                else np.nan
            )

        angle_rows.append(angle_row)

    return pd.DataFrame(overlap_rows), pd.DataFrame(angle_rows)


# =============================================================================
# Permutation null
# =============================================================================

def synthesize_loo_class_probabilities(
    image_sums: np.ndarray,
    image_counts: np.ndarray,
    permuted_labels: np.ndarray,
):
    """
    Rebuild the exact no-intercept one-hot logistic prediction probabilities.

    For each held-out image, the predicted probability for each neuron is the
    mean event rate among all other images assigned to the held-out image's
    permuted class.
    """
    n_neurons, n_images = image_sums.shape

    class_sums = np.zeros((len(CLASSES), n_neurons), dtype=np.float64)
    class_counts = np.zeros((len(CLASSES), n_neurons), dtype=np.float64)

    for class_idx, cls in enumerate(CLASSES):
        mask = permuted_labels == cls
        class_sums[class_idx] = image_sums[:, mask].sum(axis=1)
        class_counts[class_idx] = image_counts[:, mask].sum(axis=1)

    synthetic = np.full((n_images, n_neurons), np.nan, dtype=np.float64)

    for image_idx in range(n_images):
        cls = int(permuted_labels[image_idx])
        class_idx = int(np.where(CLASSES == cls)[0][0])

        numerator = class_sums[class_idx] - image_sums[:, image_idx]
        denominator = class_counts[class_idx] - image_counts[:, image_idx]

        if np.any(denominator <= 0):
            raise ValueError(
                f"Permutation produced invalid class counts for image {image_idx}."
            )

        synthetic[image_idx] = numerator / denominator

    return synthetic


def run_permutation_null(
    X_real: np.ndarray,
    labels: np.ndarray,
    neuron_mask: np.ndarray,
    original_image_indices: np.ndarray,
    observed_overlap: pd.DataFrame,
    rng: np.random.Generator,
):
    if N_PERMUTATIONS <= 0:
        return pd.DataFrame(), {}

    print()
    print("#" * 80)
    print(f"Running permutation null: {N_PERMUTATIONS} permutations")
    print("#" * 80)

    image_sums, image_counts = load_binary_trial_summaries(
        expected_neuron_mask=neuron_mask,
        original_image_indices=original_image_indices,
    )

    null_values = {
        mode: {
            "pc1": np.zeros((N_PERMUTATIONS, len(K_VALUES))),
            "pc2": np.zeros((N_PERMUTATIONS, len(K_VALUES))),
            "pc3": np.zeros((N_PERMUTATIONS, len(K_VALUES))),
            "mean_capture": np.zeros((N_PERMUTATIONS, len(K_VALUES))),
        }
        for mode in MODES
    }

    # Real PCA bases are constant across permutations.
    real_geometry = {}
    for mode in MODES:
        if mode == "centered":
            Xr = X_real - X_real.mean(axis=0, keepdims=True)
            scaler = None
        else:
            scaler = StandardScaler(with_mean=True, with_std=True)
            Xr = scaler.fit_transform(X_real)

        pca_real = PCA(
            n_components=max(K_VALUES),
            svd_solver="full",
        )
        pca_real.fit(Xr)
        real_geometry[mode] = {
            "U": pca_real.components_.T,
            "scaler": scaler,
        }

    for permutation_idx in range(N_PERMUTATIONS):
        labels_perm = rng.permutation(labels)

        X_synth_perm = synthesize_loo_class_probabilities(
            image_sums=image_sums,
            image_counts=image_counts,
            permuted_labels=labels_perm,
        )

        for mode in MODES:
            if mode == "centered":
                Xs = X_synth_perm - X_synth_perm.mean(
                    axis=0, keepdims=True
                )
            else:
                Xs = real_geometry[mode]["scaler"].transform(X_synth_perm)

            pca_synth = PCA(
                n_components=N_ENCODING_PCS,
                svd_solver="full",
            )
            pca_synth.fit(Xs)
            E_perm = pca_synth.components_.T
            U = real_geometry[mode]["U"]

            for k_idx, k in enumerate(K_VALUES):
                coordinates = U[:, :k].T @ E_perm
                retained = np.sum(coordinates ** 2, axis=0)
                retained = np.clip(retained, 0.0, 1.0)

                null_values[mode]["pc1"][permutation_idx, k_idx] = retained[0]
                null_values[mode]["pc2"][permutation_idx, k_idx] = retained[1]
                null_values[mode]["pc3"][permutation_idx, k_idx] = retained[2]
                null_values[mode]["mean_capture"][
                    permutation_idx, k_idx
                ] = np.sum(coordinates ** 2) / N_ENCODING_PCS

        if (
            permutation_idx == 0
            or (permutation_idx + 1) % 25 == 0
            or permutation_idx == N_PERMUTATIONS - 1
        ):
            print(
                f"\rPermutation {permutation_idx + 1:>4}/"
                f"{N_PERMUTATIONS}",
                end="",
                flush=True,
            )

    print()

    rows = []

    metric_map = {
        "encoding_pc1_retained_fraction": "pc1",
        "encoding_pc2_retained_fraction": "pc2",
        "encoding_pc3_retained_fraction": "pc3",
        "mean_3d_subspace_capture": "mean_capture",
    }

    for mode in MODES:
        observed_mode = observed_overlap[
            observed_overlap["mode"] == mode
        ].set_index("k")

        for k_idx, k in enumerate(K_VALUES):
            for observed_name, null_name in metric_map.items():
                null = null_values[mode][null_name][:, k_idx]
                observed = float(observed_mode.loc[k, observed_name])

                # Add-one empirical upper-tail p-value.
                p_value = (
                    1.0 + np.sum(null >= observed)
                ) / (N_PERMUTATIONS + 1.0)

                rows.append(
                    {
                        "mode": mode,
                        "k": int(k),
                        "metric": observed_name,
                        "observed": observed,
                        "null_mean": float(np.mean(null)),
                        "null_std": float(np.std(null, ddof=1)),
                        "null_ci95_low": float(np.percentile(null, 2.5)),
                        "null_ci95_high": float(np.percentile(null, 97.5)),
                        "empirical_p_upper": float(p_value),
                        "z_vs_null": float(
                            (observed - np.mean(null))
                            / max(np.std(null, ddof=1), EPS)
                        ),
                    }
                )

    return pd.DataFrame(rows), null_values


# =============================================================================
# Plotting
# =============================================================================

def plot_overlap_curves(overlap: pd.DataFrame):
    for mode in MODES:
        table = overlap[overlap["mode"] == mode].sort_values("k")

        fig, ax = plt.subplots(figsize=(9.2, 5.8))

        ax.plot(
            table["k"],
            table["encoding_pc1_retained_fraction"],
            marker="o",
            linewidth=2,
            label="Encoding PC1",
        )
        ax.plot(
            table["k"],
            table["encoding_pc2_retained_fraction"],
            marker="o",
            linewidth=2,
            label="Encoding PC2",
        )
        ax.plot(
            table["k"],
            table["encoding_pc3_retained_fraction"],
            marker="o",
            linewidth=2,
            label="Encoding PC3",
        )
        ax.plot(
            table["k"],
            table["mean_3d_subspace_capture"],
            marker="s",
            linewidth=2.5,
            linestyle="--",
            label="Mean 3D capture",
        )

        ax.axvline(60, linestyle=":", linewidth=1.3)
        ax.set(
            xlabel="Number of unsupervised real-data PCs, k",
            ylabel="Fraction of encoding direction retained",
            title=f"Encoding subspace inside real PCA space: {mode}",
            xlim=(0, max(K_VALUES) + 2),
            ylim=(0, 1.02),
        )
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()

        for suffix in ("png", "pdf"):
            fig.savefig(
                OUTDIR / f"overlap_curves_{mode}.{suffix}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)


def plot_principal_angles(angles: pd.DataFrame):
    for mode in MODES:
        table = angles[angles["mode"] == mode].sort_values("k")

        fig, ax = plt.subplots(figsize=(9.2, 5.8))

        for j in range(1, N_ENCODING_PCS + 1):
            ax.plot(
                table["k"],
                table[f"principal_angle_{j}_deg"],
                marker="o",
                linewidth=2,
                label=f"Principal angle {j}",
            )

        ax.axvline(60, linestyle=":", linewidth=1.3)
        ax.set(
            xlabel="Number of unsupervised real-data PCs, k",
            ylabel="Principal angle (degrees)",
            title=f"Encoding vs real-data PCA subspace angles: {mode}",
            xlim=(0, max(K_VALUES) + 2),
            ylim=(0, 92),
        )
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()

        for suffix in ("png", "pdf"):
            fig.savefig(
                OUTDIR / f"principal_angles_{mode}.{suffix}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)


def plot_overlap_at_k60(overlap: pd.DataFrame):
    k = 60
    table = overlap[overlap["k"] == k].copy()

    labels = ["Encoding PC1", "Encoding PC2", "Encoding PC3", "Mean 3D"]
    columns = [
        "encoding_pc1_retained_fraction",
        "encoding_pc2_retained_fraction",
        "encoding_pc3_retained_fraction",
        "mean_3d_subspace_capture",
    ]

    x = np.arange(len(labels), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.0, 5.8))

    for offset, mode in enumerate(MODES):
        row = table[table["mode"] == mode].iloc[0]
        values = [float(row[column]) for column in columns]
        positions = x + (offset - 0.5) * width

        bars = ax.bar(
            positions,
            values,
            width=width,
            label=mode,
        )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set(
        ylabel="Fraction retained inside first 60 real PCs",
        title="Supervised encoding structure captured by unsupervised PCA",
        ylim=(0, 1.12),
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()

    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTDIR / f"overlap_at_k60.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_observed_vs_null(
    permutation_summary: pd.DataFrame,
):
    if permutation_summary.empty:
        return

    metric = "mean_3d_subspace_capture"

    for mode in MODES:
        table = permutation_summary[
            (permutation_summary["mode"] == mode)
            & (permutation_summary["metric"] == metric)
        ].sort_values("k")

        fig, ax = plt.subplots(figsize=(9.2, 5.8))

        ax.fill_between(
            table["k"].to_numpy(),
            table["null_ci95_low"].to_numpy(),
            table["null_ci95_high"].to_numpy(),
            alpha=0.25,
            label="Permutation null 95% interval",
        )
        ax.plot(
            table["k"],
            table["null_mean"],
            linestyle="--",
            linewidth=2,
            label="Permutation null mean",
        )
        ax.plot(
            table["k"],
            table["observed"],
            marker="o",
            linewidth=2.5,
            label="Observed encoding subspace",
        )

        ax.axvline(60, linestyle=":", linewidth=1.3)
        ax.set(
            xlabel="Number of unsupervised real-data PCs, k",
            ylabel="Mean 3D encoding-subspace capture",
            title=f"Observed overlap versus permuted-label null: {mode}",
            xlim=(0, max(K_VALUES) + 2),
            ylim=(0, 1.02),
        )
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()

        for suffix in ("png", "pdf"):
            fig.savefig(
                OUTDIR / f"observed_vs_null_overlap_{mode}.{suffix}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)


# =============================================================================
# Interpretation
# =============================================================================

def build_interpretation(
    overlap: pd.DataFrame,
    angles: pd.DataFrame,
    permutation_summary: pd.DataFrame,
):
    lines = []
    lines.append("ENCODING SUBSPACE VS UNSUPERVISED REAL-DATA PCA")
    lines.append("=" * 80)

    for mode in MODES:
        row = overlap[
            (overlap["mode"] == mode)
            & (overlap["k"] == 60)
        ].iloc[0]

        angle_row = angles[
            (angles["mode"] == mode)
            & (angles["k"] == 60)
        ].iloc[0]

        lines.append("")
        lines.append(f"Mode: {mode}")
        lines.append(
            "  Fraction retained at k=60: "
            f"PC1={row['encoding_pc1_retained_fraction']:.4f}, "
            f"PC2={row['encoding_pc2_retained_fraction']:.4f}, "
            f"PC3={row['encoding_pc3_retained_fraction']:.4f}."
        )
        lines.append(
            "  Mean 3D subspace capture at k=60: "
            f"{row['mean_3d_subspace_capture']:.4f}."
        )
        lines.append(
            "  Principal angles at k=60: "
            f"{angle_row['principal_angle_1_deg']:.2f}, "
            f"{angle_row['principal_angle_2_deg']:.2f}, "
            f"{angle_row['principal_angle_3_deg']:.2f} degrees."
        )

        if not permutation_summary.empty:
            null_row = permutation_summary[
                (permutation_summary["mode"] == mode)
                & (permutation_summary["k"] == 60)
                & (
                    permutation_summary["metric"]
                    == "mean_3d_subspace_capture"
                )
            ].iloc[0]

            lines.append(
                "  Permutation comparison for mean 3D capture at k=60: "
                f"null mean={null_row['null_mean']:.4f}, "
                f"95% null interval=[{null_row['null_ci95_low']:.4f}, "
                f"{null_row['null_ci95_high']:.4f}], "
                f"p={null_row['empirical_p_upper']:.6f}."
            )

    lines.append("")
    lines.append(
        "Interpretation rule: high retained fractions, small principal angles, "
        "and overlap above the permuted-label null support the claim that the "
        "leading unsupervised real-data PCs contain the same neural structure "
        "revealed by the supervised encoding model."
    )

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    (
        X_real,
        X_synth,
        labels,
        original_image_indices,
        neuron_mask,
    ) = load_real_and_synthetic()

    overlap_tables = []
    angle_tables = []

    saved_geometry = {}

    for mode in MODES:
        print()
        print("#" * 80)
        print(f"Observed subspace geometry: {mode}")
        print("#" * 80)

        Xr, Xs, scaler = preprocess_pair(
            X_real=X_real,
            X_synth=X_synth,
            mode=mode,
        )

        pca_real, pca_synth, U, E = fit_pca_bases(Xr, Xs)

        overlap, angles = compute_overlap_metrics(
            U=U,
            E=E,
            mode=mode,
        )

        overlap_tables.append(overlap)
        angle_tables.append(angles)

        saved_geometry[mode] = {
            "real_components": pca_real.components_,
            "encoding_components": pca_synth.components_,
            "real_evr": pca_real.explained_variance_ratio_,
            "encoding_evr": pca_synth.explained_variance_ratio_,
        }

        k60 = overlap[overlap["k"] == 60].iloc[0]
        print(
            "k=60 retained fractions: "
            f"PC1={k60['encoding_pc1_retained_fraction']:.4f}, "
            f"PC2={k60['encoding_pc2_retained_fraction']:.4f}, "
            f"PC3={k60['encoding_pc3_retained_fraction']:.4f}, "
            f"mean={k60['mean_3d_subspace_capture']:.4f}"
        )

    overlap_all = pd.concat(overlap_tables, ignore_index=True)
    angles_all = pd.concat(angle_tables, ignore_index=True)

    permutation_summary, null_values = run_permutation_null(
        X_real=X_real,
        labels=labels,
        neuron_mask=neuron_mask,
        original_image_indices=original_image_indices,
        observed_overlap=overlap_all,
        rng=rng,
    )

    overlap_all.to_csv(SUMMARY_CSV, index=False)
    angles_all.to_csv(ANGLES_CSV, index=False)
    permutation_summary.to_csv(PERMUTATION_CSV, index=False)

    plot_overlap_curves(overlap_all)
    plot_principal_angles(angles_all)
    plot_overlap_at_k60(overlap_all)
    plot_observed_vs_null(permutation_summary)

    interpretation = build_interpretation(
        overlap=overlap_all,
        angles=angles_all,
        permutation_summary=permutation_summary,
    )
    INTERPRETATION_TXT.write_text(interpretation + "\n")

    npz_payload = {
        "X_real_shape": np.asarray(X_real.shape, dtype=np.int64),
        "X_synthetic_shape": np.asarray(X_synth.shape, dtype=np.int64),
        "labels": labels,
        "original_image_indices": original_image_indices,
        "neuron_mask": neuron_mask,
        "k_values": np.asarray(K_VALUES, dtype=np.int64),
        "modes": np.asarray(MODES, dtype=object),
        "n_encoding_pcs": np.int64(N_ENCODING_PCS),
        "n_permutations": np.int64(N_PERMUTATIONS),
        "random_seed": np.int64(RANDOM_SEED),
        "overlap_table": overlap_all.to_records(index=False),
        "angles_table": angles_all.to_records(index=False),
        "permutation_table": (
            permutation_summary.to_records(index=False)
            if not permutation_summary.empty
            else np.empty(0, dtype=[("empty", "i1")])
        ),
        "settings_json": json.dumps(
            {
                "real_composite_path": str(REAL_COMPOSITE_PATH),
                "synthetic_path": str(SYNTHETIC_PATH),
                "binary_trial_composite_path": str(
                    BINARY_TRIAL_COMPOSITE_PATH
                ),
                "k_values": K_VALUES,
                "modes": MODES,
                "n_encoding_pcs": N_ENCODING_PCS,
                "n_permutations": N_PERMUTATIONS,
            }
        ),
    }

    for mode in MODES:
        npz_payload[f"{mode}_real_components"] = (
            saved_geometry[mode]["real_components"]
        )
        npz_payload[f"{mode}_encoding_components"] = (
            saved_geometry[mode]["encoding_components"]
        )
        npz_payload[f"{mode}_real_evr"] = saved_geometry[mode]["real_evr"]
        npz_payload[f"{mode}_encoding_evr"] = (
            saved_geometry[mode]["encoding_evr"]
        )

        if null_values:
            for metric_name, values in null_values[mode].items():
                npz_payload[f"{mode}_null_{metric_name}"] = values

    np.savez_compressed(RESULTS_NPZ, **npz_payload)

    print()
    print(interpretation)

    print()
    print("=" * 80)
    print("Saved")
    print("=" * 80)
    for path in (
        SUMMARY_CSV,
        ANGLES_CSV,
        PERMUTATION_CSV,
        RESULTS_NPZ,
        INTERPRETATION_TXT,
        OUTDIR / "overlap_at_k60.png",
        OUTDIR / "overlap_curves_standardized.png",
        OUTDIR / "principal_angles_standardized.png",
        OUTDIR / "observed_vs_null_overlap_standardized.png",
    ):
        print(path)


if __name__ == "__main__":
    main()
