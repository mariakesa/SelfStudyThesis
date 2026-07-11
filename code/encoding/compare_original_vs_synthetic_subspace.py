#!/usr/bin/env python3
"""
Compare original Allen image-level neural response probabilities against
LOO synthetic probabilities generated from the four-class design matrix.

Inputs expected in /home/maria/SelfStudyThesis/data:
    allen_natural_scenes_four_class_composite.npy
        Original image-level composite: X is neurons × 118, probabilities.

    synthetic_neural_activity_image_probs_loo.npy
        Synthetic LOO image-level probabilities: neurons × 118.

This script asks:
    1. How much variance in the original image-level neural data is explained by
       the synthetic four-class design-matrix predictions?
    2. Do original and synthetic data occupy similar PCA subspaces?
    3. Which PCA components are shared, and which are residual real-data-only?
    4. What do contrastive PCA directions show for:
           real-enriched variance:      C_real - alpha C_synth
           synthetic-enriched variance: C_synth - alpha C_real

Outputs:
    /home/maria/SelfStudyThesis/results/original_vs_synthetic_four_class_subspace/
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy.linalg import subspace_angles


# =============================================================================
# Paths
# =============================================================================

DATA_DIR = Path("/home/maria/SelfStudyThesis/data")

ORIGINAL_COMPOSITE_PATH = DATA_DIR / "allen_natural_scenes_four_class_composite.npy"
SYNTHETIC_PATH = DATA_DIR / "synthetic_neural_activity_image_probs_loo.npy"

OUT_DIR = Path(
    "/home/maria/SelfStudyThesis/results/original_vs_synthetic_four_class_subspace"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUT_DIR / "original_vs_synthetic_subspace_results.npz"
OUT_SUMMARY_CSV = OUT_DIR / "summary_metrics.csv"
OUT_IMAGE_CSV = OUT_DIR / "per_image_metrics.csv"
OUT_NEURON_CSV = OUT_DIR / "per_neuron_metrics.csv"


# =============================================================================
# Settings
# =============================================================================

N_IMAGES = 118
N_COMPONENTS = 30
RANDOM_STATE = 123
EPS = 1e-12

LABEL_NAMES = {
    -1: "unlabeled",
     0: "animals",
     1: "landscape",
     2: "plant",
     3: "man-made object",
}

# cPCA alpha values. alpha=0 is ordinary PCA of the first matrix.
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]


# =============================================================================
# Helpers
# =============================================================================

def safe_corr(a: np.ndarray, b: np.ndarray, axis: int) -> np.ndarray:
    """Pearson correlation along axis, returning NaN for zero-variance cases."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a0 = a - np.mean(a, axis=axis, keepdims=True)
    b0 = b - np.mean(b, axis=axis, keepdims=True)

    num = np.sum(a0 * b0, axis=axis)
    den = np.sqrt(np.sum(a0 ** 2, axis=axis) * np.sum(b0 ** 2, axis=axis))

    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den

    r[den <= EPS] = np.nan
    return r


def global_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Frobenius/global R^2 over all entries."""
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    return float(1.0 - ss_res / ss_tot)


def centered_frobenius_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Frobenius R^2 after removing each neuron's image-mean.

    This asks whether synthetic data explains image-to-image modulation,
    not baseline firing/event probability differences between neurons.
    """
    yt = y_true - np.mean(y_true, axis=0, keepdims=True)
    yp = y_pred - np.mean(y_pred, axis=0, keepdims=True)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum(yt ** 2)
    return float(1.0 - ss_res / ss_tot)


def load_data():
    print("#" * 80)
    print("Loading data")
    print("#" * 80)
    print(f"Original composite: {ORIGINAL_COMPOSITE_PATH}")
    print(f"Synthetic probs:    {SYNTHETIC_PATH}")

    composite = np.load(ORIGINAL_COMPOSITE_PATH, allow_pickle=True).item()
    X_original = np.asarray(composite["X"], dtype=np.float64)
    X_synth = np.asarray(np.load(SYNTHETIC_PATH, allow_pickle=True), dtype=np.float64)

    stimulus_metadata = composite["stimulus_metadata"]
    neuron_metadata = composite["neuron_metadata"]
    labels = np.asarray(stimulus_metadata["label"], dtype=np.int64).ravel()

    # Original composite should be neurons × images, but tolerate transposed input.
    if X_original.shape[1] == len(labels):
        pass
    elif X_original.shape[0] == len(labels):
        print("[INFO] Transposing original X from images × neurons to neurons × images")
        X_original = X_original.T
    else:
        raise ValueError(
            f"Cannot align original X with labels. X={X_original.shape}, labels={labels.shape}"
        )

    # Synthetic should be neurons × images, but tolerate transposed input.
    if X_synth.shape == X_original.shape:
        pass
    elif X_synth.T.shape == X_original.shape:
        print("[INFO] Transposing synthetic matrix to neurons × images")
        X_synth = X_synth.T
    else:
        raise ValueError(
            f"Original and synthetic shapes do not align. "
            f"original={X_original.shape}, synthetic={X_synth.shape}"
        )

    if X_original.shape[1] != N_IMAGES:
        print(f"[WARNING] Expected {N_IMAGES} images, got {X_original.shape[1]}")

    print()
    print("Shapes:")
    print(f"  original X: {X_original.shape}  neurons × images")
    print(f"  synthetic:  {X_synth.shape}  neurons × images")
    print(f"  labels:     {labels.shape}")

    print()
    print("Label counts:")
    unique, counts = np.unique(labels, return_counts=True)
    for value, count in zip(unique, counts):
        print(f"  {int(value):>2} = {LABEL_NAMES.get(int(value), 'UNKNOWN'):<16} n={int(count)}")

    return composite, X_original, X_synth, labels, neuron_metadata


def basic_entrywise_metrics(X_original_ni, X_synth_ni, labels, neuron_metadata):
    """
    Compute direct prediction metrics.

    Input matrices are neurons × images.
    For sklearn-style metrics, transpose to images × neurons.
    """
    X = X_original_ni.T   # images × neurons
    Y = X_synth_ni.T      # images × neurons

    print()
    print("#" * 80)
    print("Direct original-vs-synthetic metrics")
    print("#" * 80)

    r2_by_neuron = r2_score(X, Y, multioutput="raw_values")
    r2_by_image = np.array([
        r2_score(X[i, :], Y[i, :]) if np.var(X[i, :]) > EPS else np.nan
        for i in range(X.shape[0])
    ])

    corr_by_neuron = safe_corr(X, Y, axis=0)
    corr_by_image = safe_corr(X, Y, axis=1)

    mse_by_neuron = np.mean((X - Y) ** 2, axis=0)
    mse_by_image = np.mean((X - Y) ** 2, axis=1)

    mae_by_neuron = np.mean(np.abs(X - Y), axis=0)
    mae_by_image = np.mean(np.abs(X - Y), axis=1)

    g_r2 = global_r2(X, Y)
    g_r2_centered = centered_frobenius_r2(X, Y)
    global_corr = float(np.corrcoef(X.ravel(), Y.ravel())[0, 1])
    global_mse = float(np.mean((X - Y) ** 2))
    global_mae = float(np.mean(np.abs(X - Y)))

    print(f"Global R^2, raw entries:              {g_r2:.6f}")
    print(f"Global R^2, neuron-image centered:    {g_r2_centered:.6f}")
    print(f"Global entrywise correlation:         {global_corr:.6f}")
    print(f"Global MSE:                           {global_mse:.8f}")
    print(f"Global MAE:                           {global_mae:.8f}")

    finite_r2 = np.isfinite(r2_by_neuron)
    finite_corr = np.isfinite(corr_by_neuron)

    print()
    print("Per-neuron prediction quality:")
    print(f"  median R^2:   {np.nanmedian(r2_by_neuron):.6f}")
    print(f"  mean R^2:     {np.nanmean(r2_by_neuron):.6f}")
    print(f"  max R^2:      {np.nanmax(r2_by_neuron):.6f}")
    print(f"  n R^2 > 0:    {int(np.sum(r2_by_neuron[finite_r2] > 0))}")
    print(f"  n R^2 > .01:  {int(np.sum(r2_by_neuron[finite_r2] > 0.01))}")
    print(f"  n R^2 > .05:  {int(np.sum(r2_by_neuron[finite_r2] > 0.05))}")
    print(f"  median corr:  {np.nanmedian(corr_by_neuron):.6f}")
    print(f"  mean corr:    {np.nanmean(corr_by_neuron):.6f}")

    if "brain_area" in neuron_metadata:
        brain_area = np.asarray(neuron_metadata["brain_area"], dtype=object).astype(str)
    else:
        brain_area = np.full(X.shape[1], "unknown", dtype=object)

    per_neuron = pd.DataFrame({
        "neuron_index": np.arange(X.shape[1]),
        "brain_area": brain_area,
        "r2": r2_by_neuron,
        "corr": corr_by_neuron,
        "mse": mse_by_neuron,
        "mae": mae_by_neuron,
    })

    for key in ["cell_specimen_id", "ophys_experiment_id", "local_neuron_index"]:
        if key in neuron_metadata:
            arr = np.asarray(neuron_metadata[key]).ravel()
            if len(arr) == len(per_neuron):
                per_neuron[key] = arr

    per_image = pd.DataFrame({
        "image_index": np.arange(X.shape[0]),
        "label": labels,
        "label_name": [LABEL_NAMES.get(int(x), "UNKNOWN") for x in labels],
        "r2_across_neurons": r2_by_image,
        "corr_across_neurons": corr_by_image,
        "mse_across_neurons": mse_by_image,
        "mae_across_neurons": mae_by_image,
    })

    print()
    print("Per-image prediction quality by class:")
    print(
        per_image.groupby("label_name")[["corr_across_neurons", "mse_across_neurons", "mae_across_neurons"]]
        .agg(["mean", "median", "std"])
    )

    summary = {
        "global_r2_raw": g_r2,
        "global_r2_neuron_centered": g_r2_centered,
        "global_corr": global_corr,
        "global_mse": global_mse,
        "global_mae": global_mae,
        "mean_r2_by_neuron": float(np.nanmean(r2_by_neuron)),
        "median_r2_by_neuron": float(np.nanmedian(r2_by_neuron)),
        "max_r2_by_neuron": float(np.nanmax(r2_by_neuron)),
        "mean_corr_by_neuron": float(np.nanmean(corr_by_neuron)),
        "median_corr_by_neuron": float(np.nanmedian(corr_by_neuron)),
        "mean_corr_by_image": float(np.nanmean(corr_by_image)),
        "median_corr_by_image": float(np.nanmedian(corr_by_image)),
    }

    per_neuron.to_csv(OUT_NEURON_CSV, index=False)
    per_image.to_csv(OUT_IMAGE_CSV, index=False)

    print()
    print(f"Saved per-neuron metrics: {OUT_NEURON_CSV}")
    print(f"Saved per-image metrics:  {OUT_IMAGE_CSV}")

    return summary, per_neuron, per_image


def run_pca_comparison(X_original_ni, X_synth_ni, labels):
    """Compare real and synthetic#!/usr/bin/env python3
"""
Cross-validated four-class encoding regression.

Question:
    Can image class labels predict held-out neural responses?

For each held-out image:
    1. Fit regression on all other images.
    2. Predict the held-out image's response for every neuron.
    3. Store prediction.

Then compute cross-validated R^2 per neuron.

This is a proper test-set metric across images.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, KFold


# =============================================================================
# Paths
# =============================================================================

COMPOSITE_PATH = Path(
    "/home/maria/SelfStudyThesis/data/allen_natural_scenes_four_class_composite.npy"
)

OUT_DIR = Path("/home/maria/SelfStudyThesis/results/encoding_label_regression_cv")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUT_DIR / "four_class_label_encoding_cv_r2_by_neuron.npz"
OUT_FIG = OUT_DIR / "four_class_label_encoding_cv_r2_histogram.png"
OUT_FIG_ZOOM = OUT_DIR / "four_class_label_encoding_cv_r2_histogram_zoom.png"


# =============================================================================
# Settings
# =============================================================================

CV_MODE = "loo"
N_SPLITS = 10
RANDOM_STATE = 123

R2_THRESHOLD = 0.10


LABEL_NAMES = {
    -1: "unlabeled",
     0: "animals",
     1: "landscape",
     2: "plant",
     3: "man-made object",
}


# =============================================================================
# Loading
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
    print(f"X raw shape:       {X.shape}")
    print(f"labels shape:      {labels.shape}")
    print(f"neuron metadata keys:   {list(neuron_metadata.keys())}")
    print(f"stimulus metadata keys: {list(stimulus_metadata.keys())}")

    print()
    print("Raw four-class label counts:")
    unique, counts = np.unique(labels, return_counts=True)
    for value, count in zip(unique, counts):
        name = LABEL_NAMES.get(int(value), "UNKNOWN")
        print(f"  {int(value):>2} = {name:<16} n={int(count)}")

    # Composite X may be neurons × images.
    # Encoding model expects images × neurons.
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
            "Expected one X axis to match number of image labels."
        )

    n_features = X.shape[1]

    if "brain_area" in neuron_metadata:
        brain_area = np.asarray(neuron_metadata["brain_area"], dtype=object).ravel()
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

    if len(brain_area) != n_features:
        raise ValueError(
            f"brain_area length does not match feature count. "
            f"brain_area={len(brain_area)}, features={n_features}"
        )

    if len(cell_specimen_id) != n_features:
        raise ValueError(
            f"cell_specimen_id length does not match feature count. "
            f"cell_specimen_id={len(cell_specimen_id)}, features={n_features}"
        )

    if len(ophys_experiment_id) != n_features:
        raise ValueError(
            f"ophys_experiment_id length does not match feature count. "
            f"ophys_experiment_id={len(ophys_experiment_id)}, features={n_features}"
        )

    print()
    print("=" * 80)
    print("After orientation alignment")
    print("=" * 80)
    print(f"X shape, images × neurons: {X.shape}")
    print(f"Number of neurons/features: {n_features}")

    return data, X, labels, brain_area, cell_specimen_id, ophys_experiment_id


# =============================================================================
# Design matrix
# =============================================================================

def make_one_hot_design(labels):
    """
    Make one-hot design from four-class labels.

    Drops unlabeled images, label == -1.

    Design has 4 columns and no explicit intercept:
        animals, landscape, plant, man-made object

    With no intercept, each coefficient is just the class mean response.
    This is especially clean for cross-validation.
    """

    keep_mask = labels >= 0
    labels_kept = labels[keep_mask]

    classes = np.array([0, 1, 2, 3], dtype=np.int64)

    design = np.zeros((len(labels_kept), len(classes)), dtype=np.float64)

    for j, cls in enumerate(classes):
        design[:, j] = labels_kept == cls

    column_names = [LABEL_NAMES[int(c)] for c in classes]

    return design, labels_kept, keep_mask, classes, column_names


# =============================================================================
# CV encoding
# =============================================================================

def get_cv_splitter(n_samples):
    if CV_MODE == "loo":
        print()
        print("[INFO] Using Leave-One-Out CV")
        return LeaveOneOut()

    if CV_MODE == "kfold":
        print()
        print(f"[INFO] Using KFold CV, n_splits={N_SPLITS}")
        return KFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    raise ValueError(f"Unknown CV_MODE: {CV_MODE}")


def cross_validated_encoding(X, design, labels_kept):
    """
    Fit encoding model on train images and predict held-out images.

    X:
        images × neurons

    design:
        images × classes

    Returns:
        y_pred_cv:
            images × neurons

        fold_id:
            fold assignment per image
    """

    n_images, n_neurons = X.shape

    y_pred_cv = np.full_like(X, np.nan, dtype=np.float64)
    fold_id = np.full(n_images, -1, dtype=np.int64)

    splitter = get_cv_splitter(n_images)

    print()
    print("#" * 80)
    print("Running cross-validated encoding")
    print("#" * 80)
    print(f"Images:  {n_images}")
    print(f"Neurons: {n_neurons}")
    print(f"Design:  {design.shape}")

    for fold, (train_idx, test_idx) in enumerate(splitter.split(design), start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]

        D_train = design[train_idx]
        D_test = design[test_idx]

        # Sanity check for LOO:
        # each train split should contain at least one example from every class.
        train_labels = labels_kept[train_idx]
        missing_classes = sorted(set([0, 1, 2, 3]) - set(train_labels.tolist()))
        if missing_classes:
            raise ValueError(
                f"Fold {fold} has missing classes in training data: {missing_classes}. "
                "Use KFold or check labels."
            )

        model = LinearRegression(fit_intercept=False)
        model.fit(D_train, X_train)

        y_pred_cv[test_idx] = model.predict(D_test)
        fold_id[test_idx] = fold

        if fold == 1 or fold % 10 == 0 or fold == n_images:
            print(
                f"[fold {fold:>3}] "
                f"train={len(train_idx):>3} "
                f"test={len(test_idx):>3}"
            )

    if np.isnan(y_pred_cv).any():
        raise RuntimeError("Some CV predictions are still NaN. Something went wrong.")

    return y_pred_cv, fold_id


def compute_cv_r2(X, y_pred_cv):
    """
    Compute cross-validated R^2 per neuron.

    R^2 can be negative.

    Negative CV R^2 means:
        class-label prediction is worse than predicting the neuron's global mean.
    """

    r2_cv = r2_score(X, y_pred_cv, multioutput="raw_values")

    # Also compute manually, to make the meaning explicit.
    y_mean = np.mean(X, axis=0, keepdims=True)

    ss_res = np.sum((X - y_pred_cv) ** 2, axis=0)
    ss_tot = np.sum((X - y_mean) ** 2, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        r2_manual = 1.0 - ss_res / ss_tot

    if not np.allclose(r2_cv, r2_manual, equal_nan=True):
        print("[WARNING] sklearn and manual R^2 do not exactly match.")

    return r2_cv


# =============================================================================
# Summaries and plots
# =============================================================================

def summarize_r2(r2, brain_area, title):
    finite = np.isfinite(r2)
    vals = r2[finite]

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(f"Finite neurons:        {finite.sum()} / {len(r2)}")
    print(f"Mean CV R^2:           {np.mean(vals):.6f}")
    print(f"Median CV R^2:         {np.median(vals):.6f}")
    print(f"Std CV R^2:            {np.std(vals):.6f}")
    print(f"Min CV R^2:            {np.min(vals):.6f}")
    print(f"Max CV R^2:            {np.max(vals):.6f}")
    print(f"Neurons CV R^2 > 0:    {np.sum(vals > 0)}")
    print(f"Neurons CV R^2 > 0.01: {np.sum(vals > 0.01)}")
    print(f"Neurons CV R^2 > 0.05: {np.sum(vals > 0.05)}")
    print(f"Neurons CV R^2 > 0.10: {np.sum(vals > 0.10)}")
    print(f"Percent CV R^2 > 0.10: {100 * np.sum(vals > 0.10) / len(vals):.3f}%")

    print()
    print("By brain area:")
    area_str = brain_area.astype(str)

    for area in sorted(np.unique(area_str)):
        mask = (area_str == area) & finite
        area_vals = r2[mask]

        if len(area_vals) == 0:
            continue

        print(
            f"  {area:<8} "
            f"n={len(area_vals):>6} "
            f"mean={np.mean(area_vals):>9.6f} "
            f"median={np.median(area_vals):>9.6f} "
            f"max={np.max(area_vals):>9.6f} "
            f"n>0.10={np.sum(area_vals > 0.10):>5}"
        )


def plot_histogram(r2_cv):
    finite = np.isfinite(r2_cv)
    vals = r2_cv[finite]

    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=100)
    plt.axvline(0.0, linestyle="-", label="zero")
    plt.axvline(np.mean(vals), linestyle="--", label=f"mean = {np.mean(vals):.4f}")
    plt.axvline(np.median(vals), linestyle=":", label=f"median = {np.median(vals):.4f}")
    plt.axvline(R2_THRESHOLD, linestyle="-.", label=f"threshold = {R2_THRESHOLD}")
    plt.xlabel("Cross-validated variance explained per neuron, $R^2$")
    plt.ylabel("Number of neurons")
    plt.title("CV encoding model: four-class image-label design matrix")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200)
    plt.close()

    print()
    print(f"Saved histogram to: {OUT_FIG}")


def plot_histogram_zoom(r2_cv):
    finite = np.isfinite(r2_cv)
    vals = r2_cv[finite]

    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=100, range=(-0.2, 0.2))
    plt.axvline(0.0, linestyle="-", label="zero")
    plt.axvline(np.mean(vals), linestyle="--", label=f"mean = {np.mean(vals):.4f}")
    plt.axvline(np.median(vals), linestyle=":", label=f"median = {np.median(vals):.4f}")
    plt.axvline(R2_THRESHOLD, linestyle="-.", label=f"threshold = {R2_THRESHOLD}")
    plt.xlabel("Cross-validated variance explained per neuron, $R^2$")
    plt.ylabel("Number of neurons")
    plt.title("CV encoding model histogram, zoomed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG_ZOOM, dpi=200)
    plt.close()

    print(f"Saved zoomed histogram to: {OUT_FIG_ZOOM}")


def print_top_neurons(r2_cv, brain_area, cell_specimen_id, ophys_experiment_id, n_top=30):
    finite = np.isfinite(r2_cv)
    finite_indices = np.where(finite)[0]

    order = finite_indices[np.argsort(r2_cv[finite_indices])[::-1]]

    print()
    print("=" * 80)
    print(f"Top {n_top} neurons by cross-validated R^2")
    print("=" * 80)

    for rank, idx in enumerate(order[:n_top], start=1):
        print(
            f"{rank:>2}. "
            f"neuron_idx={idx:>6} "
            f"CV_R2={r2_cv[idx]:>9.5f} "
            f"area={str(brain_area[idx]):<8} "
            f"cell_specimen_id={int(cell_specimen_id[idx])} "
            f"ophys_experiment_id={int(ophys_experiment_id[idx])}"
        )


# =============================================================================
# Main
# =============================================================================

def main():
    (
        data,
        X,
        labels,
        brain_area,
        cell_specimen_id,
        ophys_experiment_id,
    ) = load_composite_data()

    design, labels_kept, keep_mask, classes, column_names = make_one_hot_design(labels)

    X_kept = X[keep_mask]

    print()
    print("=" * 80)
    print("CV design matrix")
    print("=" * 80)
    print(f"Kept images:    {X_kept.shape[0]} / {X.shape[0]}")
    print(f"Design shape:   {design.shape}")
    print(f"Design columns: {column_names}")

    print()
    print("Kept label counts:")
    unique, counts = np.unique(labels_kept, return_counts=True)
    for value, count in zip(unique, counts):
        print(f"  {int(value):>2} = {LABEL_NAMES[int(value)]:<16} n={int(count)}")

    y_pred_cv, fold_id = cross_validated_encoding(X_kept, design, labels_kept)
    r2_cv = compute_cv_r2(X_kept, y_pred_cv)

    summarize_r2(
        r2_cv,
        brain_area,
        title="Cross-validated variance explained summary",
    )

    print_top_neurons(
        r2_cv,
        brain_area,
        cell_specimen_id,
        ophys_experiment_id,
        n_top=30,
    )

    plot_histogram(r2_cv)
    plot_histogram_zoom(r2_cv)

    np.savez_compressed(
        OUT_NPZ,
        r2_cv_by_neuron=r2_cv,
        y_pred_cv=y_pred_cv,
        X_kept=X_kept,
        design=design,
        design_column_names=np.asarray(column_names, dtype=object),
        labels=labels,
        labels_kept=labels_kept,
        keep_mask=keep_mask,
        classes=classes,
        class_names=np.asarray([LABEL_NAMES[int(c)] for c in classes], dtype=object),
        fold_id=fold_id,
        brain_area=brain_area,
        cell_specimen_id=cell_specimen_id,
        ophys_experiment_id=ophys_experiment_id,
        cv_mode=CV_MODE,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        r2_threshold=R2_THRESHOLD,
    )

    print()
    print(f"Saved CV encoding results to: {OUT_NPZ}")

    print()
    print("#" * 80)
    print("Done")
    print("#" * 80)


if __name__ == "__main__":
    main() PCA geometry."""
    X_real = X_original_ni.T   # images × neurons
    X_syn = X_synth_ni.T       # images × neurons

    n_components = min(N_COMPONENTS, X_real.shape[0] - 1, X_real.shape[1])

    # Center both by their own feature means for PCA.
    Xr = X_real - X_real.mean(axis=0, keepdims=True)
    Xs = X_syn - X_syn.mean(axis=0, keepdims=True)

    print()
    print("#" * 80)
    print("PCA / subspace comparison")
    print("#" * 80)
    print(f"n_components = {n_components}")

    pca_real = PCA(n_components=n_components, svd_solver="randomized", random_state=RANDOM_STATE)
    pca_syn = PCA(n_components=n_components, svd_solver="randomized", random_state=RANDOM_STATE)

    real_scores = pca_real.fit_transform(Xr)
    syn_scores = pca_syn.fit_transform(Xs)

    # Project synthetic data into the real PCA basis.
    syn_in_real_scores = pca_real.transform(Xs)

    # Component-wise score correlations in real PCA space.
    score_corr_real_basis = np.array([
        np.corrcoef(real_scores[:, k], syn_in_real_scores[:, k])[0, 1]
        if np.std(real_scores[:, k]) > EPS and np.std(syn_in_real_scores[:, k]) > EPS
        else np.nan
        for k in range(n_components)
    ])

    # How much original variance is captured by synthetic projection onto real PCs?
    # For each real PC k, compare the original PC scores to synthetic PC scores.
    pc_score_r2_real_basis = np.array([
        r2_score(real_scores[:, k], syn_in_real_scores[:, k])
        if np.var(real_scores[:, k]) > EPS else np.nan
        for k in range(n_components)
    ])

    # Principal angles between top-k subspaces.
    ks = np.arange(1, n_components + 1)
    max_angle_deg = np.full(n_components, np.nan)
    mean_angle_deg = np.full(n_components, np.nan)

    for k in ks:
        A = pca_real.components_[:k].T  # neurons × k
        B = pca_syn.components_[:k].T
        angles = subspace_angles(A, B)
        max_angle_deg[k - 1] = np.degrees(np.max(angles))
        mean_angle_deg[k - 1] = np.degrees(np.mean(angles))

    # Variance of original data explained by reconstructing original through real PCs,
    # versus reconstructing synthetic through real PCs and comparing to original.
    cumulative_real_pca_reconstruction_r2 = []
    cumulative_synth_in_real_basis_r2 = []

    real_mean = X_real.mean(axis=0, keepdims=True)
    synth_mean = X_syn.mean(axis=0, keepdims=True)

    for k in ks:
        W = pca_real.components_[:k]

        real_recon = real_scores[:, :k] @ W + real_mean
        synth_recon_in_real_basis = syn_in_real_scores[:, :k] @ W + synth_mean

        cumulative_real_pca_reconstruction_r2.append(global_r2(X_real, real_recon))
        cumulative_synth_in_real_basis_r2.append(global_r2(X_real, synth_recon_in_real_basis))

    cumulative_real_pca_reconstruction_r2 = np.asarray(cumulative_real_pca_reconstruction_r2)
    cumulative_synth_in_real_basis_r2 = np.asarray(cumulative_synth_in_real_basis_r2)

    print()
    print("Top real PCA explained variance ratios:")
    for k in range(min(10, n_components)):
        print(
            f"  PC{k+1:02d}: EVR={pca_real.explained_variance_ratio_[k]:.5f} "
            f"score_corr(real, synth-in-real-basis)={score_corr_real_basis[k]:.5f} "
            f"score_R2={pc_score_r2_real_basis[k]:.5f}"
        )

    pca_table = pd.DataFrame({
        "component": np.arange(1, n_components + 1),
        "real_evr": pca_real.explained_variance_ratio_,
        "real_cumulative_evr": np.cumsum(pca_real.explained_variance_ratio_),
        "synthetic_evr": pca_syn.explained_variance_ratio_,
        "synthetic_cumulative_evr": np.cumsum(pca_syn.explained_variance_ratio_),
        "score_corr_real_basis": score_corr_real_basis,
        "score_r2_real_basis": pc_score_r2_real_basis,
        "max_subspace_angle_deg_top_k": max_angle_deg,
        "mean_subspace_angle_deg_top_k": mean_angle_deg,
        "real_pca_reconstruction_r2_top_k": cumulative_real_pca_reconstruction_r2,
        "synth_in_real_basis_r2_top_k": cumulative_synth_in_real_basis_r2,
    })

    pca_csv = OUT_DIR / "pca_subspace_metrics.csv"
    pca_table.to_csv(pca_csv, index=False)
    print(f"Saved PCA/subspace metrics: {pca_csv}")

    # Plots
    plt.figure(figsize=(8, 5))
    plt.plot(ks, np.cumsum(pca_real.explained_variance_ratio_), marker="o", label="original")
    plt.plot(ks, np.cumsum(pca_syn.explained_variance_ratio_), marker="o", label="synthetic")
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative explained variance ratio")
    plt.title("PCA variance spectra")
    plt.legend()
    plt.tight_layout()
    out_fig = OUT_DIR / "pca_cumulative_variance.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(ks, score_corr_real_basis, marker="o")
    plt.axhline(0.0, linestyle="-")
    plt.xlabel("Real PCA component")
    plt.ylabel("Corr(original score, synthetic score)")
    plt.title("Synthetic data projected into original PCA basis")
    plt.tight_layout()
    out_fig2 = OUT_DIR / "real_pc_score_correlations.png"
    plt.savefig(out_fig2, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(ks, mean_angle_deg, marker="o", label="mean angle")
    plt.plot(ks, max_angle_deg, marker="o", label="max angle")
    plt.xlabel("Top-k subspace")
    plt.ylabel("Principal angle, degrees")
    plt.title("Original vs synthetic PCA subspace angles")
    plt.legend()
    plt.tight_layout()
    out_fig3 = OUT_DIR / "pca_subspace_angles.png"
    plt.savefig(out_fig3, dpi=200)
    plt.close()

    # PC1 / PC2 scatter, original and synthetic in original basis.
    if n_components >= 2:
        plt.figure(figsize=(7, 6))
        for lab in sorted(np.unique(labels)):
            mask = labels == lab
            plt.scatter(real_scores[mask, 0], real_scores[mask, 1], label=f"original {LABEL_NAMES.get(int(lab), lab)}", alpha=0.75)
            plt.scatter(syn_in_real_scores[mask, 0], syn_in_real_scores[mask, 1], marker="x", label=f"synthetic {LABEL_NAMES.get(int(lab), lab)}", alpha=0.75)
        plt.xlabel("Real PC1 score")
        plt.ylabel("Real PC2 score")
        plt.title("Original and synthetic images in original PCA space")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        out_fig4 = OUT_DIR / "original_synthetic_real_pca_pc1_pc2.png"
        plt.savefig(out_fig4, dpi=200)
        plt.close()

    print(f"Saved PCA figures in: {OUT_DIR}")

    return {
        "pca_table": pca_table,
        "real_scores": real_scores,
        "synthetic_scores": syn_scores,
        "synthetic_in_real_scores": syn_in_real_scores,
        "real_components": pca_real.components_,
        "synthetic_components": pca_syn.components_,
        "real_evr": pca_real.explained_variance_ratio_,
        "synthetic_evr": pca_syn.explained_variance_ratio_,
    }


def covariance_in_sample_space(X_images_by_neurons: np.ndarray) -> np.ndarray:
    """
    Return image × image covariance/Gram matrix.

    Since neurons >> images, doing cPCA in sample space is tiny and fast.
    This finds stimulus contrast directions rather than neuron loading vectors first.
    """
    Xc = X_images_by_neurons - X_images_by_neurons.mean(axis=0, keepdims=True)
    return (Xc @ Xc.T) / max(Xc.shape[1] - 1, 1)


def run_contrastive_pca(X_original_ni, X_synth_ni, labels):
    """
    Contrastive PCA in image/sample space.

    real-enriched:      G_real - alpha G_synthetic
    synthetic-enriched: G_synthetic - alpha G_real

    Because there are only 118 images and ~40k neurons, sample-space cPCA is the
    cleanest diagnostic: it gives image contrast directions enriched in one matrix
    relative to the other.
    """
    X_real = X_original_ni.T
    X_syn = X_synth_ni.T

    G_real = covariance_in_sample_space(X_real)
    G_syn = covariance_in_sample_space(X_syn)

    print()
    print("#" * 80)
    print("Contrastive PCA in image/sample space")
    print("#" * 80)

    rows = []
    for alpha in ALPHAS:
        for mode, G_target, G_background in [
            ("real_minus_alpha_synthetic", G_real, G_syn),
            ("synthetic_minus_alpha_real", G_syn, G_real),
        ]:
            C = G_target - alpha * G_background
            eigvals, eigvecs = np.linalg.eigh(C)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            top_score = eigvecs[:, 0]

            # Orient score for readability: animals positive when possible.
            if 0 in labels:
                if np.nanmean(top_score[labels == 0]) < np.nanmean(top_score[labels != 0]):
                    top_score = -top_score

            class_means = {
                f"mean_score_{LABEL_NAMES.get(int(lab), lab).replace(' ', '_')}": float(np.mean(top_score[labels == lab]))
                for lab in np.unique(labels)
            }

            rows.append({
                "mode": mode,
                "alpha": alpha,
                "top_eigenvalue": float(eigvals[0]),
                "second_eigenvalue": float(eigvals[1]) if len(eigvals) > 1 else np.nan,
                "n_positive_eigenvalues": int(np.sum(eigvals > 0)),
                **class_means,
            })

            score_csv = OUT_DIR / f"cpca_scores_{mode}_alpha_{str(alpha).replace('.', 'p')}.csv"
            pd.DataFrame({
                "image_index": np.arange(len(labels)),
                "label": labels,
                "label_name": [LABEL_NAMES.get(int(x), "UNKNOWN") for x in labels],
                "cpca1_score": top_score,
            }).to_csv(score_csv, index=False)

            plt.figure(figsize=(8, 4.5))
            for lab in sorted(np.unique(labels)):
                mask = labels == lab
                plt.scatter(np.where(mask)[0], top_score[mask], label=LABEL_NAMES.get(int(lab), str(lab)))
            plt.axhline(0.0, linestyle="-")
            plt.xlabel("Image index")
            plt.ylabel("cPC1 score")
            plt.title(f"cPCA score: {mode}, alpha={alpha}")
            plt.legend(fontsize=8)
            plt.tight_layout()
            fig_path = OUT_DIR / f"cpca1_{mode}_alpha_{str(alpha).replace('.', 'p')}.png"
            plt.savefig(fig_path, dpi=200)
            plt.close()

    cpca_summary = pd.DataFrame(rows)
    cpca_csv = OUT_DIR / "contrastive_pca_summary.csv"
    cpca_summary.to_csv(cpca_csv, index=False)
    print(f"Saved cPCA summary: {cpca_csv}")
    print(f"Saved cPCA score files and figures in: {OUT_DIR}")

    return cpca_summary


def main():
    composite, X_original, X_synth, labels, neuron_metadata = load_data()

    summary, per_neuron, per_image = basic_entrywise_metrics(
        X_original, X_synth, labels, neuron_metadata
    )

    pca_results = run_pca_comparison(X_original, X_synth, labels)
    cpca_summary = run_contrastive_pca(X_original, X_synth, labels)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

    np.savez_compressed(
        OUT_NPZ,
        X_original=X_original,
        X_synthetic=X_synth,
        labels=labels,
        summary_keys=np.asarray(list(summary.keys()), dtype=object),
        summary_values=np.asarray(list(summary.values()), dtype=np.float64),
        real_scores=pca_results["real_scores"],
        synthetic_scores=pca_results["synthetic_scores"],
        synthetic_in_real_scores=pca_results["synthetic_in_real_scores"],
        real_components=pca_results["real_components"],
        synthetic_components=pca_results["synthetic_components"],
        real_evr=pca_results["real_evr"],
        synthetic_evr=pca_results["synthetic_evr"],
        pca_table=pca_results["pca_table"].to_records(index=False),
        cpca_summary=cpca_summary.to_records(index=False),
    )

    print()
    print("#" * 80)
    print("DONE")
    print("#" * 80)
    print(f"Saved summary CSV: {OUT_SUMMARY_CSV}")
    print(f"Saved result NPZ:   {OUT_NPZ}")
    print(f"Output dir:         {OUT_DIR}")


if __name__ == "__main__":
    main()
