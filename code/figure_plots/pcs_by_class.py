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

# Contains the neural animacy decision scores and four-class labels.
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

OUT_PNG = OUTDIR / "synthetic_activity_top3_pcs_class_sorted.png"
OUT_PDF = OUTDIR / "synthetic_activity_top3_pcs_class_sorted.pdf"
OUT_NPZ = OUTDIR / "synthetic_activity_pca_top3_class_sorted.npz"


# =============================================================================
# Settings
# =============================================================================

LABEL_NAMES = {
    -1: "unlabeled",
     0: "animal",
     1: "landscape",
     2: "plant",
     3: "man-made object",
}

# Desired left-to-right order.
#
# The three non-animal classes come first, followed by animals.
# Unlabeled images, if present, are placed last.
CLASS_ORDER = [1, 2, 3, 0, -1]

N_COMPONENTS = 3
RANDOM_STATE = 0


# =============================================================================
# Load synthetic population activity
# =============================================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Synthetic activity file not found:\n{DATA_PATH}"
    )

dat = np.load(
    DATA_PATH,
    allow_pickle=True,
)

dat = np.asarray(
    dat,
    dtype=np.float32,
)

if dat.ndim != 2:
    raise ValueError(
        f"Expected a 2D synthetic activity matrix, got shape {dat.shape}."
    )

print("=" * 80)
print("Loaded synthetic activity matrix")
print("=" * 80)
print(f"Path:  {DATA_PATH}")
print(f"Shape: {dat.shape}")
print(f"Min:   {np.nanmin(dat):.6f}")
print(f"Max:   {np.nanmax(dat):.6f}")
print(f"Mean:  {np.nanmean(dat):.6f}")

dat = np.nan_to_num(
    dat,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)


# =============================================================================
# Load labels and neural animacy scores
# =============================================================================

if not ORDER_RESULTS_PATH.exists():
    raise FileNotFoundError(
        f"Ordering results file not found:\n{ORDER_RESULTS_PATH}"
    )

order_results = np.load(
    ORDER_RESULTS_PATH,
    allow_pickle=True,
)

required_keys = {
    "decision_scores",
    "labels",
}

missing_keys = required_keys.difference(order_results.files)

if missing_keys:
    raise KeyError(
        f"Ordering file is missing keys: {sorted(missing_keys)}\n"
        f"Available keys: {order_results.files}"
    )

decision_scores = np.asarray(
    order_results["decision_scores"],
    dtype=float,
).ravel()

labels = np.asarray(
    order_results["labels"],
    dtype=int,
).ravel()

print()
print("=" * 80)
print("Loaded image labels and neural animacy scores")
print("=" * 80)
print(f"Path:                  {ORDER_RESULTS_PATH}")
print(f"Decision scores shape: {decision_scores.shape}")
print(f"Labels shape:          {labels.shape}")

if len(decision_scores) != len(labels):
    raise ValueError(
        "Decision scores and labels have different lengths.\n"
        f"decision_scores={decision_scores.shape}\n"
        f"labels={labels.shape}"
    )

print()
print("Four-class label counts:")

unique_labels, label_counts = np.unique(
    labels,
    return_counts=True,
)

for label, count in zip(unique_labels, label_counts):
    print(
        f"  {int(label):>2} = "
        f"{LABEL_NAMES.get(int(label), 'unknown'):<18} "
        f"n={int(count)}"
    )


# =============================================================================
# Orient synthetic data as images × synthetic neurons
# =============================================================================

n_images = len(labels)

# Expected file orientation:
#     synthetic neurons × images
if dat.shape[1] == n_images:
    X = dat.T
    print()
    print("[INFO] Transposed synthetic activity to images × neurons.")

# Safeguard for opposite orientation:
#     images × synthetic neurons
elif dat.shape[0] == n_images:
    X = dat
    print()
    print("[INFO] Synthetic activity already appears to be images × neurons.")

else:
    raise ValueError(
        "Could not align synthetic activity with the image labels.\n"
        f"Synthetic matrix shape: {dat.shape}\n"
        f"Number of image labels:  {n_images}"
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

# StandardScaler can produce NaNs if an input column is pathological.
Xz = np.nan_to_num(
    Xz,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

print()
print("=" * 80)
print("Standardized PCA matrix")
print("=" * 80)
print(f"Shape: {Xz.shape}")
print(f"Mean:  {np.mean(Xz):.6f}")
print(f"Std:   {np.std(Xz):.6f}")


# =============================================================================
# Fit PCA
# =============================================================================

n_components = min(
    N_COMPONENTS,
    Xz.shape[0],
    Xz.shape[1],
)

if n_components < 1:
    raise ValueError(
        f"Cannot run PCA with input shape {Xz.shape}."
    )

pca = PCA(
    n_components=n_components,
    random_state=RANDOM_STATE,
)

# Scores remain in original image order here.
Z = pca.fit_transform(Xz)

explained = pca.explained_variance_ratio_

print()
print("=" * 80)
print("PCA results")
print("=" * 80)

for component_index, evr in enumerate(explained, start=1):
    print(
        f"PC{component_index}: "
        f"explained variance ratio = {evr:.6f} "
        f"({100 * evr:.2f}%)"
    )

print(
    f"Cumulative variance explained by top {n_components}: "
    f"{100 * np.sum(explained):.2f}%"
)


# =============================================================================
# Sort first by four-class label, then by animacy score within class
# =============================================================================

class_rank_lookup = {
    class_label: rank
    for rank, class_label in enumerate(CLASS_ORDER)
}

default_rank = len(CLASS_ORDER)

class_rank = np.asarray(
    [
        class_rank_lookup.get(
            int(label),
            default_rank,
        )
        for label in labels
    ],
    dtype=int,
)

# np.lexsort uses the LAST key as the primary key.
#
# Primary key:
#     class_rank
#
# Secondary key:
#     decision score within each class
class_sorted_order = np.lexsort(
    (
        decision_scores,
        class_rank,
    )
)

Z_sorted = Z[class_sorted_order]
labels_sorted = labels[class_sorted_order]
decision_scores_sorted = decision_scores[class_sorted_order]

sorted_original_image_indices = class_sorted_order.copy()

print()
print("=" * 80)
print("Applied class-based sorting")
print("=" * 80)
print("Primary ordering: four-class label")
print("Secondary ordering: neural animacy score within class")


# =============================================================================
# Build class blocks
# =============================================================================

class_blocks = []

for class_label in CLASS_ORDER:
    positions = np.flatnonzero(
        labels_sorted == class_label
    )

    if len(positions) == 0:
        continue

    class_blocks.append(
        {
            "label": int(class_label),
            "name": LABEL_NAMES.get(
                int(class_label),
                str(class_label),
            ),
            "start": int(positions[0]),
            "end": int(positions[-1]),
            "center": float(
                (positions[0] + positions[-1]) / 2
            ),
            "count": int(len(positions)),
        }
    )

# Also include unexpected labels not explicitly listed in CLASS_ORDER.
known_class_labels = set(CLASS_ORDER)

for class_label in np.unique(labels_sorted):
    if int(class_label) in known_class_labels:
        continue

    positions = np.flatnonzero(
        labels_sorted == class_label
    )

    class_blocks.append(
        {
            "label": int(class_label),
            "name": LABEL_NAMES.get(
                int(class_label),
                f"class {int(class_label)}",
            ),
            "start": int(positions[0]),
            "end": int(positions[-1]),
            "center": float(
                (positions[0] + positions[-1]) / 2
            ),
            "count": int(len(positions)),
        }
    )

class_blocks = sorted(
    class_blocks,
    key=lambda block: block["start"],
)

print()
print("Sorted class blocks:")

for block in class_blocks:
    print(
        f"  {block['name']:<18} "
        f"n={block['count']:>3} "
        f"positions={block['start']:>3}-{block['end']:>3}"
    )


# =============================================================================
# Save PCA arrays and ordering
# =============================================================================

np.savez_compressed(
    OUT_NPZ,

    # Standardized synthetic data and PCA scores in original image order.
    standardized_activity=Xz,
    scores=Z,

    # PCA scores and metadata in class-sorted order.
    scores_sorted=Z_sorted,
    labels_sorted=labels_sorted,
    decision_scores_sorted=decision_scores_sorted,
    original_image_indices_sorted=sorted_original_image_indices,

    # PCA model.
    explained_variance_ratio=explained,
    components=pca.components_,
    singular_values=pca.singular_values_,
    pca_mean=pca.mean_,

    # Standardization model.
    scaler_mean=scaler.mean_,
    scaler_scale=scaler.scale_,
    scaler_var=scaler.var_,

    # Original metadata.
    labels=labels,
    decision_scores=decision_scores,

    # Sorting information.
    class_sorted_order=class_sorted_order,
    class_order=np.asarray(
        CLASS_ORDER,
        dtype=int,
    ),

    class_block_labels=np.asarray(
        [block["label"] for block in class_blocks],
        dtype=int,
    ),

    class_block_names=np.asarray(
        [block["name"] for block in class_blocks],
        dtype=object,
    ),

    class_block_starts=np.asarray(
        [block["start"] for block in class_blocks],
        dtype=int,
    ),

    class_block_ends=np.asarray(
        [block["end"] for block in class_blocks],
        dtype=int,
    ),

    class_block_centers=np.asarray(
        [block["center"] for block in class_blocks],
        dtype=float,
    ),

    class_block_counts=np.asarray(
        [block["count"] for block in class_blocks],
        dtype=int,
    ),
)

print()
print(f"Saved PCA data to: {OUT_NPZ}")


# =============================================================================
# Plot helpers
# =============================================================================

def add_class_annotations(ax, blocks):
    """
    Add vertical boundaries and class labels to an axis.
    """

    for block_index, block in enumerate(blocks):

        # Light shading on alternating class blocks.
        if block_index % 2 == 0:
            ax.axvspan(
                block["start"] - 0.5,
                block["end"] + 0.5,
                alpha=0.06,
            )

        # Boundary between adjacent classes.
        if block_index > 0:
            boundary = block["start"] - 0.5

            ax.axvline(
                boundary,
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )

        # Class label near the top of the panel.
        ax.text(
            block["center"],
            0.97,
            f"{block['name']}\n(n={block['count']})",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )


# =============================================================================
# Plot top 3 PCs over class-sorted image rank
# =============================================================================

sorted_rank = np.arange(
    Z_sorted.shape[0]
)

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

# Mark animals explicitly.
animal_positions = np.flatnonzero(
    labels_sorted == 0
)

if len(animal_positions) > 0:
    axes[0].scatter(
        animal_positions,
        decision_scores_sorted[animal_positions],
        s=20,
        label="Animal image",
    )

axes[0].set_ylabel(
    "Neural logistic\nscore"
)

axes[0].set_title(
    "Synthetic population PCs grouped by four-class encoding label"
)

if len(animal_positions) > 0:
    axes[0].legend(
        frameon=False,
        loc="lower right",
    )

add_class_annotations(
    axes[0],
    class_blocks,
)


# -----------------------------------------------------------------------------
# Bottom panel: PCA scores
# -----------------------------------------------------------------------------

for component_index in range(n_components):
    axes[1].plot(
        sorted_rank,
        Z_sorted[:, component_index],
        label=(
            f"PC{component_index + 1} "
            f"({explained[component_index] * 100:.1f}%)"
        ),
        linewidth=1.8,
    )

axes[1].axhline(
    0,
    linewidth=1,
)

axes[1].set_xlabel(
    "Images grouped by four-class label and sorted by neural animacy score within class"
)

axes[1].set_ylabel(
    "Synthetic population PC score"
)

axes[1].legend(
    frameon=False,
    ncol=n_components,
    loc="lower center",
)

add_class_annotations(
    axes[1],
    class_blocks,
)

axes[1].set_xlim(
    -0.5,
    len(sorted_rank) - 0.5,
)


# =============================================================================
# Save figure
# =============================================================================

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

print()
print("=" * 80)
print("Done")
print("=" * 80)
print(f"Saved PNG to: {OUT_PNG}")
print(f"Saved PDF to: {OUT_PDF}")
print(f"Saved NPZ to: {OUT_NPZ}")