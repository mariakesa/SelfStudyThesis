#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Paths
# =============================================================================

DATA_PATH = Path(
    "/home/maria/SelfStudyThesis/data/"
    "synthetic_neural_activity_image_probs_loo.npy"
)

# Contains image_order created by the animal-vs-rest logistic-regression script.
ORDER_RESULTS_PATH = Path(
    "/home/maria/SelfStudyThesis/results/"
    "animacy_all_neurons/"
    "all_neurons_animacy_modulation_results.npz"
)

OUTDIR = Path(
    "/home/maria/SelfStudyThesis/results/"
    "synthetic_neural_activity_figures/pca"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "synthetic_activity_top3_pcs_animacy_sorted.png"
OUT_PDF = OUTDIR / "synthetic_activity_top3_pcs_animacy_sorted.pdf"
OUT_NPZ = OUTDIR / "synthetic_activity_pca_top3_animacy_sorted.npz"


# =============================================================================
# Load synthetic population activity
# =============================================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Synthetic activity file not found:\n{DATA_PATH}"
    )

dat = np.load(DATA_PATH, allow_pickle=True)
dat = np.asarray(dat, dtype=np.float32)

if dat.ndim != 2:
    raise ValueError(
        f"Expected a 2D synthetic activity matrix, got shape {dat.shape}."
    )

print("=" * 80)
print("Loaded synthetic activity matrix")
print("=" * 80)
print(f"dat shape: {dat.shape}")
print(f"min:       {np.nanmin(dat):.6f}")
print(f"max:       {np.nanmax(dat):.6f}")
print(f"mean:      {np.nanmean(dat):.6f}")

dat = np.nan_to_num(
    dat,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)


# =============================================================================
# Load the image ordering
# =============================================================================

if not ORDER_RESULTS_PATH.exists():
    raise FileNotFoundError(
        f"Image-order results file not found:\n{ORDER_RESULTS_PATH}"
    )

order_results = np.load(
    ORDER_RESULTS_PATH,
    allow_pickle=True,
)

required_keys = {
    "image_order",
    "decision_scores",
    "labels",
}

missing_keys = required_keys.difference(order_results.files)

if missing_keys:
    raise KeyError(
        f"Ordering file is missing keys: {sorted(missing_keys)}\n"
        f"Available keys: {order_results.files}"
    )

image_order = np.asarray(
    order_results["image_order"],
    dtype=int,
)

decision_scores = np.asarray(
    order_results["decision_scores"],
    dtype=float,
)

labels = np.asarray(
    order_results["labels"],
)

print()
print("=" * 80)
print("Loaded neural animacy ordering")
print("=" * 80)
print(f"image_order shape:      {image_order.shape}")
print(f"decision_scores shape:  {decision_scores.shape}")
print(f"labels shape:           {labels.shape}")


# =============================================================================
# Orient synthetic data as images × synthetic neurons
# =============================================================================

n_ordered_images = len(image_order)

# The expected input orientation is synthetic neurons × images.
if dat.shape[1] == n_ordered_images:
    X = dat.T

# Allow the opposite orientation as a safeguard.
elif dat.shape[0] == n_ordered_images:
    print("Input already appears to be images × synthetic neurons.")
    X = dat

else:
    raise ValueError(
        "Could not align synthetic data with the saved image ordering.\n"
        f"Synthetic matrix shape: {dat.shape}\n"
        f"Number of ordered images: {n_ordered_images}"
    )

if sorted(image_order.tolist()) != list(range(X.shape[0])):
    raise ValueError(
        "image_order is not a complete permutation of the image indices."
    )

print()
print("=" * 80)
print("PCA input")
print("=" * 80)
print(f"X shape, images × synthetic neurons: {X.shape}")


# =============================================================================
# Standardize synthetic neuron dimensions across images
# =============================================================================

scaler = StandardScaler(
    with_mean=True,
    with_std=True,
)

Xz = scaler.fit_transform(X)


# =============================================================================
# Fit PCA
# =============================================================================

pca = PCA(
    n_components=3,
    random_state=0,
)

# Z is in the original image order at this point.
Z = pca.fit_transform(Xz)

explained = pca.explained_variance_ratio_

print()
print("=" * 80)
print("PCA results")
print("=" * 80)

for component_index, evr in enumerate(explained, start=1):
    print(
        f"PC{component_index}: "
        f"explained variance ratio = {evr:.6f}"
    )


# =============================================================================
# Apply the neural logistic-regression image ordering
# =============================================================================

Z_sorted = Z[image_order]
labels_sorted = labels[image_order]
decision_scores_sorted = decision_scores[image_order]

sorted_original_image_indices = image_order.copy()

ANIMAL_LABEL = 0

animal_positions = np.flatnonzero(
    labels_sorted == ANIMAL_LABEL
)

nonanimal_positions = np.flatnonzero(
    labels_sorted != ANIMAL_LABEL
)


# =============================================================================
# Save PCA arrays and ordering
# =============================================================================

np.savez_compressed(
    OUT_NPZ,

    # PCA scores in original image order.
    scores=Z,

    # PCA scores reordered along the neural animacy axis.
    scores_sorted=Z_sorted,

    explained_variance_ratio=explained,
    components=pca.components_,

    scaler_mean=scaler.mean_,
    scaler_scale=scaler.scale_,

    image_order=image_order,
    original_image_indices_sorted=sorted_original_image_indices,

    decision_scores=decision_scores,
    decision_scores_sorted=decision_scores_sorted,

    labels=labels,
    labels_sorted=labels_sorted,
)

print()
print(f"Saved PCA data to: {OUT_NPZ}")


# =============================================================================
# Plot top 3 PCs over animacy-sorted image rank
# =============================================================================

sorted_rank = np.arange(Z_sorted.shape[0])

fig, axes = plt.subplots(
    2,
    1,
    figsize=(15, 8),
    height_ratios=[1, 3],
    sharex=True,
    constrained_layout=True,
)


# -----------------------------------------------------------------------------
# Top panel: neural logistic-regression decision score
# -----------------------------------------------------------------------------

axes[0].plot(
    sorted_rank,
    decision_scores_sorted,
    linewidth=1.8,
)

axes[0].axhline(
    0,
    linestyle="--",
    linewidth=1,
)

axes[0].scatter(
    animal_positions,
    decision_scores_sorted[animal_positions],
    s=20,
    label="Ground-truth animal",
)

axes[0].set_ylabel(
    "Neural logistic\nscore"
)

axes[0].set_title(
    "Synthetic population PCs in neural animacy-score order"
)

axes[0].legend(
    frameon=False,
    loc="upper left",
)


# -----------------------------------------------------------------------------
# Bottom panel: PCA scores
# -----------------------------------------------------------------------------

axes[1].plot(
    sorted_rank,
    Z_sorted[:, 0],
    label=f"PC1 ({explained[0] * 100:.1f}%)",
    linewidth=1.8,
)

axes[1].plot(
    sorted_rank,
    Z_sorted[:, 1],
    label=f"PC2 ({explained[1] * 100:.1f}%)",
    linewidth=1.8,
)

axes[1].plot(
    sorted_rank,
    Z_sorted[:, 2],
    label=f"PC3 ({explained[2] * 100:.1f}%)",
    linewidth=1.8,
)

axes[1].axhline(
    0,
    linewidth=1,
)

# Mark animal images along the bottom of the plot.
ymin, ymax = axes[1].get_ylim()
marker_y = ymin + 0.03 * (ymax - ymin)

axes[1].scatter(
    animal_positions,
    np.full_like(
        animal_positions,
        marker_y,
        dtype=float,
    ),
    marker="|",
    s=80,
    label="Animal image position",
)

axes[1].set_xlabel(
    "Image rank from non-animal-like to animal-like"
)

axes[1].set_ylabel(
    "Synthetic population PC score"
)

axes[1].legend(
    frameon=False,
    ncol=4,
    loc="upper center",
)

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Saved PNG to: {OUT_PNG}")
print(f"Saved PDF to: {OUT_PDF}")