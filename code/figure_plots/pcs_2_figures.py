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

# Expected keys:
#     image_order
#     decision_scores
#     labels
#
# labels should be the four-class labels:
#     0 = animal
#     1 = landscape
#     2 = plant
#     3 = man-made object
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

OUT_PNG = OUTDIR / "synthetic_activity_pca_two_orderings.png"
OUT_PDF = OUTDIR / "synthetic_activity_pca_two_orderings.pdf"
OUT_NPZ = OUTDIR / "synthetic_activity_pca_two_orderings.npz"


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

# Left-to-right ordering in the class-grouped panel.
CLASS_ORDER = [1, 2, 3, 0, -1]

ANIMAL_LABEL = 0
N_COMPONENTS = 3
RANDOM_STATE = 0


# =============================================================================
# Load synthetic activity
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
print("Loaded synthetic activity")
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
# Load labels, neural animacy scores, and saved animacy ordering
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

image_order_animacy = np.asarray(
    order_results["image_order"],
    dtype=int,
).ravel()

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
print("Loaded image metadata")
print("=" * 80)
print(f"image_order shape:      {image_order_animacy.shape}")
print(f"decision_scores shape:  {decision_scores.shape}")
print(f"labels shape:           {labels.shape}")

if len(decision_scores) != len(labels):
    raise ValueError(
        "decision_scores and labels have different lengths.\n"
        f"decision_scores={decision_scores.shape}\n"
        f"labels={labels.shape}"
    )

n_images = len(labels)

if len(image_order_animacy) != n_images:
    raise ValueError(
        "image_order length does not match labels length.\n"
        f"image_order={len(image_order_animacy)}\n"
        f"labels={n_images}"
    )

if sorted(image_order_animacy.tolist()) != list(range(n_images)):
    raise ValueError(
        "image_order is not a complete permutation of image indices."
    )

print()
print("Label counts:")

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
# Orient synthetic data as images × neurons
# =============================================================================

if dat.shape[1] == n_images:
    X = dat.T
    print()
    print("[INFO] Transposed synthetic matrix to images × neurons.")

elif dat.shape[0] == n_images:
    X = dat
    print()
    print("[INFO] Synthetic matrix already appears to be images × neurons.")

else:
    raise ValueError(
        "Could not align synthetic activity with image metadata.\n"
        f"Synthetic activity shape: {dat.shape}\n"
        f"Number of images:          {n_images}"
    )

print()
print("=" * 80)
print("PCA input")
print("=" * 80)
print(f"X shape, images × neurons: {X.shape}")


# =============================================================================
# Standardize neuron dimensions across images
# =============================================================================

scaler = StandardScaler(
    with_mean=True,
    with_std=True,
)

Xz = scaler.fit_transform(X)

Xz = np.nan_to_num(
    Xz,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)


# =============================================================================
# Fit PCA once
# =============================================================================

n_components = min(
    N_COMPONENTS,
    Xz.shape[0],
    Xz.shape[1],
)

pca = PCA(
    n_components=n_components,
    random_state=RANDOM_STATE,
)

# Z remains in original image order.
Z = pca.fit_transform(Xz)

explained = pca.explained_variance_ratio_

print()
print("=" * 80)
print("PCA results")
print("=" * 80)

for component_index, evr in enumerate(explained, start=1):
    print(
        f"PC{component_index}: "
        f"{evr:.6f} "
        f"({100 * evr:.2f}%)"
    )

print(
    f"Cumulative variance explained by top {n_components}: "
    f"{100 * np.sum(explained):.2f}%"
)


# =============================================================================
# Ordering 1: neural animacy score
# =============================================================================

# Use the ordering saved by the animacy logistic-regression script.
animacy_order = image_order_animacy.copy()

Z_animacy = Z[animacy_order]
labels_animacy = labels[animacy_order]
decision_scores_animacy = decision_scores[animacy_order]

animal_positions_animacy = np.flatnonzero(
    labels_animacy == ANIMAL_LABEL
)


# =============================================================================
# Ordering 2: four-class group, then neural animacy score within class
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

# np.lexsort uses the last key as the primary key.
#
# Primary:
#     class rank
#
# Secondary:
#     neural animacy score within class
class_order = np.lexsort(
    (
        decision_scores,
        class_rank,
    )
)

Z_class = Z[class_order]
labels_class = labels[class_order]
decision_scores_class = decision_scores[class_order]

animal_positions_class = np.flatnonzero(
    labels_class == ANIMAL_LABEL
)


# =============================================================================
# Build class blocks for class-grouped panel
# =============================================================================

class_blocks = []

for class_label in CLASS_ORDER:

    positions = np.flatnonzero(
        labels_class == class_label
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

# Add any unexpected labels after the known ones.
known_labels = set(CLASS_ORDER)

for class_label in np.unique(labels_class):

    if int(class_label) in known_labels:
        continue

    positions = np.flatnonzero(
        labels_class == class_label
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
print("=" * 80)
print("Class-grouped blocks")
print("=" * 80)

for block in class_blocks:
    print(
        f"{block['name']:<18} "
        f"n={block['count']:>3} "
        f"positions={block['start']:>3}-{block['end']:>3}"
    )


# =============================================================================
# Helpers
# =============================================================================

def add_class_block_annotations(ax, blocks):
    """
    Add class boundaries, alternating block shading, and class labels.
    """

    for block_index, block in enumerate(blocks):

        if block_index % 2 == 0:
            ax.axvspan(
                block["start"] - 0.5,
                block["end"] + 0.5,
                alpha=0.06,
            )

        if block_index > 0:
            ax.axvline(
                block["start"] - 0.5,
                linestyle="--",
                linewidth=1,
                alpha=0.65,
            )

        ax.text(
            block["center"],
            0.97,
            f"{block['name']}\n(n={block['count']})",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )


def add_class_strip(ax, ordered_labels):
    """
    Add one small class-colored marker per image along the bottom.

    This is especially useful in the animacy-sorted panel, where the three
    non-animal classes may be interleaved.
    """

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    marker_y = ymin + 0.025 * yrange

    marker_map = {
        -1: "x",
         0: "|",
         1: "s",
         2: "^",
         3: "o",
    }

    for class_label in np.unique(ordered_labels):

        positions = np.flatnonzero(
            ordered_labels == class_label
        )

        if len(positions) == 0:
            continue

        ax.scatter(
            positions,
            np.full(
                len(positions),
                marker_y,
                dtype=float,
            ),
            marker=marker_map.get(int(class_label), "."),
            s=22,
            label=LABEL_NAMES.get(
                int(class_label),
                str(class_label),
            ),
            alpha=0.85,
        )


# =============================================================================
# Shared axis limits
# =============================================================================

all_decision_scores = np.concatenate(
    [
        decision_scores_animacy,
        decision_scores_class,
    ]
)

decision_min = float(np.min(all_decision_scores))
decision_max = float(np.max(all_decision_scores))

decision_padding = 0.05 * (
    decision_max - decision_min
    if decision_max > decision_min
    else 1.0
)

decision_ylim = (
    decision_min - decision_padding,
    decision_max + decision_padding,
)

all_pc_scores = np.vstack(
    [
        Z_animacy,
        Z_class,
    ]
)

pc_min = float(np.min(all_pc_scores))
pc_max = float(np.max(all_pc_scores))

pc_padding = 0.05 * (
    pc_max - pc_min
    if pc_max > pc_min
    else 1.0
)

pc_ylim = (
    pc_min - pc_padding,
    pc_max + pc_padding,
)


# =============================================================================
# Plot comparison figure
# =============================================================================

rank = np.arange(n_images)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(18, 9),
    height_ratios=[1, 3],
    sharex="col",
    sharey="row",
    constrained_layout=True,
)

ax_top_animacy = axes[0, 0]
ax_bottom_animacy = axes[1, 0]

ax_top_class = axes[0, 1]
ax_bottom_class = axes[1, 1]


# =============================================================================
# Left column: animacy-score order
# =============================================================================

ax_top_animacy.plot(
    rank,
    decision_scores_animacy,
    linewidth=1.8,
)

ax_top_animacy.axhline(
    0,
    linestyle="--",
    linewidth=1,
)

ax_top_animacy.scatter(
    animal_positions_animacy,
    decision_scores_animacy[animal_positions_animacy],
    s=18,
    label="Animal image",
)

ax_top_animacy.set_title(
    "A. Images sorted by neural animacy score"
)

ax_top_animacy.set_ylabel(
    "Neural logistic\nscore"
)

ax_top_animacy.set_ylim(
    decision_ylim
)

ax_top_animacy.legend(
    frameon=False,
    loc="upper left",
)


for component_index in range(n_components):

    ax_bottom_animacy.plot(
        rank,
        Z_animacy[:, component_index],
        linewidth=1.8,
        label=(
            f"PC{component_index + 1} "
            f"({100 * explained[component_index]:.1f}%)"
        ),
    )

ax_bottom_animacy.axhline(
    0,
    linewidth=1,
)

ax_bottom_animacy.set_xlabel(
    "Image rank from non-animal-like to animal-like"
)

ax_bottom_animacy.set_ylabel(
    "Synthetic population PC score"
)

ax_bottom_animacy.set_ylim(
    pc_ylim
)

# Add markers revealing the four-class identity of each image.
#add_class_strip(
    #ax_bottom_animacy,
    #labels_animacy,
#)

ax_bottom_animacy.legend(
    frameon=False,
    ncol=4,
    loc="lower center",
    fontsize=8,
)


# =============================================================================
# Right column: four-class order
# =============================================================================

ax_top_class.plot(
    rank,
    decision_scores_class,
    linewidth=1.8,
)

ax_top_class.axhline(
    0,
    linestyle="--",
    linewidth=1,
)

ax_top_class.scatter(
    animal_positions_class,
    decision_scores_class[animal_positions_class],
    s=18,
    label="Animal image",
)

ax_top_class.set_title(
    "B. Images grouped by encoding class"
)

ax_top_class.set_ylim(
    decision_ylim
)

ax_top_class.legend(
    frameon=False,
    loc="lower right",
)

add_class_block_annotations(
    ax_top_class,
    class_blocks,
)


for component_index in range(n_components):

    ax_bottom_class.plot(
        rank,
        Z_class[:, component_index],
        linewidth=1.8,
        label=(
            f"PC{component_index + 1} "
            f"({100 * explained[component_index]:.1f}%)"
        ),
    )

ax_bottom_class.axhline(
    0,
    linewidth=1,
)

ax_bottom_class.set_xlabel(
    "Images grouped by four-class label and sorted by animacy score within class"
)

ax_bottom_class.set_ylim(
    pc_ylim
)

add_class_block_annotations(
    ax_bottom_class,
    class_blocks,
)

ax_bottom_class.legend(
    frameon=False,
    ncol=n_components,
    loc="lower center",
)


# =============================================================================
# Final formatting
# =============================================================================

for ax in axes.ravel():
    ax.set_xlim(
        -0.5,
        n_images - 0.5,
    )

fig.suptitle(
    "The same synthetic PCA representation under two image orderings",
    fontsize=16,
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


# =============================================================================
# Save data
# =============================================================================

np.savez_compressed(
    OUT_NPZ,

    # Input data and preprocessing.
    standardized_activity=Xz,
    scaler_mean=scaler.mean_,
    scaler_scale=scaler.scale_,
    scaler_var=scaler.var_,

    # PCA model.
    scores=Z,
    components=pca.components_,
    explained_variance_ratio=explained,
    singular_values=pca.singular_values_,
    pca_mean=pca.mean_,

    # Original metadata.
    labels=labels,
    decision_scores=decision_scores,

    # Animacy ordering.
    animacy_order=animacy_order,
    scores_animacy_order=Z_animacy,
    labels_animacy_order=labels_animacy,
    decision_scores_animacy_order=decision_scores_animacy,

    # Four-class ordering.
    class_order=class_order,
    scores_class_order=Z_class,
    labels_class_order=labels_class,
    decision_scores_class_order=decision_scores_class,

    # Class block metadata.
    class_display_order=np.asarray(
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


# =============================================================================
# Finish
# =============================================================================

print()
print("=" * 80)
print("Done")
print("=" * 80)
print(f"Saved PNG: {OUT_PNG}")
print(f"Saved PDF: {OUT_PDF}")
print(f"Saved NPZ: {OUT_NPZ}")