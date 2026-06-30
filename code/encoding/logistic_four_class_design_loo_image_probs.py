#!/usr/bin/env python3
"""
LOO four-class logistic encoding model on binary trial-level neural activity.

Purpose:
    Train a four-class one-hot logistic encoding model on binary trial responses,
    leaving out one stimulus/image at a time, and produce ONE synthetic neural
    activity probability per held-out image and neuron.

Input composite:
    /home/maria/SelfStudyThesis/data/allen_natural_scenes_four_class_binary_trials_composite.npy

Expected input X:
    total_neurons × 5900

where:
    5900 = 118 natural scene images × 50 trials/image
    X[i, k] is a binary event indicator for neuron i on trial column k.

Output synthetic probabilities:
    total_neurons × 118

where:
    synthetic_image_probability[i, j] is the LOO-predicted probability that
    neuron i is active for held-out image j, using only the image's four-class
    label and training on all other images/trials.

Design matrix:
    Four mutually exclusive one-hot columns:
        animals, landscape, plant, man-made object

Cross-validation:
    Leave-one-stimulus/image-out.

For each held-out image:
    1. Remove all 50 trials of that image from the training base.
    2. Fit a no-intercept one-hot logistic encoding model from class design to
       each neuron's binary event probability.
    3. Predict a single synthetic probability for the held-out image.
    4. Store that probability in a neurons × 118 array.

Important note:
    With a one-hot design matrix and no intercept, the logistic-regression MLE
    has a closed form per class and neuron:

        logit(p_neuron,class) = logit(mean binary event rate for that class)

    Therefore this script computes the exact one-hot logistic solution instead
    of fitting tens of thousands of separate sklearn LogisticRegression models.
    It is the same model family as no-intercept logistic regression with four
    mutually exclusive class columns.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np


# =============================================================================
# Paths
# =============================================================================

COMPOSITE_PATH = Path(
    "/home/maria/SelfStudyThesis/data/allen_natural_scenes_four_class_binary_trials_composite.npy"
)

OUT_DIR = Path(
    "/home/maria/SelfStudyThesis/results/four_class_logistic_design_loo_image_probs"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SYNTHETIC_NPY = OUT_DIR / "synthetic_neural_activity_image_probs_loo.npy"
OUT_OBSERVED_NPY = OUT_DIR / "observed_neural_activity_image_probs.npy"
OUT_RESULTS_NPZ = OUT_DIR / "four_class_logistic_design_loo_image_probs_results.npz"
OUT_SUMMARY_JSON = OUT_DIR / "four_class_logistic_design_loo_image_probs_summary.json"


# =============================================================================
# Constants / settings
# =============================================================================

N_IMAGES = 118
N_TRIALS = 50
EXPECTED_COLUMNS = N_IMAGES * N_TRIALS

EPS = 1e-6

LABEL_NAMES = {
    -1: "unlabeled",
     0: "animals",
     1: "landscape",
     2: "plant",
     3: "man-made object",
}

CLASSES = np.array([0, 1, 2, 3], dtype=np.int64)
CLASS_NAMES = np.asarray([LABEL_NAMES[int(c)] for c in CLASSES], dtype=object)


# =============================================================================
# Loading / validation
# =============================================================================

def load_composite():
    print()
    print("#" * 80)
    print("Loading binary trial composite")
    print("#" * 80)
    print(f"Composite path: {COMPOSITE_PATH}")

    data = np.load(COMPOSITE_PATH, allow_pickle=True).item()

    X = np.asarray(data["X"])
    neuron_metadata = data["neuron_metadata"]
    stimulus_metadata = data["stimulus_metadata"]

    print()
    print("=" * 80)
    print("Raw composite contents")
    print("=" * 80)
    print(f"X shape:                {X.shape}")
    print(f"X dtype:                {X.dtype}")
    print(f"neuron metadata keys:   {list(neuron_metadata.keys())}")
    print(f"stimulus metadata keys: {list(stimulus_metadata.keys())}")

    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D neurons × trials, got {X.shape}")

    if X.shape[1] != EXPECTED_COLUMNS:
        raise ValueError(
            f"Expected X to have {EXPECTED_COLUMNS} columns "
            f"({N_IMAGES} images × {N_TRIALS} trials), got {X.shape[1]}."
        )

    n_neurons = X.shape[0]

    labels_by_column = np.asarray(stimulus_metadata["label"], dtype=np.int64).ravel()
    image_index_by_column = np.asarray(
        stimulus_metadata["image_index"], dtype=np.int64
    ).ravel()
    trial_index_by_column = np.asarray(
        stimulus_metadata["trial_index"], dtype=np.int64
    ).ravel()

    for name, arr in [
        ("label", labels_by_column),
        ("image_index", image_index_by_column),
        ("trial_index", trial_index_by_column),
    ]:
        if arr.shape[0] != EXPECTED_COLUMNS:
            raise ValueError(
                f"stimulus_metadata['{name}'] should have length {EXPECTED_COLUMNS}, "
                f"got {arr.shape}."
            )

    # Recover one label per image from the 5900 trial-level columns.
    image_labels = np.full(N_IMAGES, -999, dtype=np.int64)
    for image_idx in range(N_IMAGES):
        cols = image_index_by_column == image_idx
        vals = np.unique(labels_by_column[cols])
        if len(vals) != 1:
            raise ValueError(
                f"Image {image_idx} has inconsistent labels across columns: {vals}"
            )
        image_labels[image_idx] = int(vals[0])

    # Check image-major layout: image 0 trials 0..49, image 1 trials 0..49, etc.
    expected_image_index = np.repeat(np.arange(N_IMAGES, dtype=np.int64), N_TRIALS)
    expected_trial_index = np.tile(np.arange(N_TRIALS, dtype=np.int64), N_IMAGES)

    if not np.array_equal(image_index_by_column, expected_image_index):
        raise ValueError(
            "Column image_index is not image-major 0..117 repeated 50 times. "
            "This script assumes columns are grouped by image."
        )

    if not np.array_equal(trial_index_by_column, expected_trial_index):
        raise ValueError(
            "Column trial_index is not 0..49 tiled across images. "
            "This script assumes columns are grouped by image."
        )

    print()
    print("Image-level label counts:")
    unique, counts = np.unique(image_labels, return_counts=True)
    for value, count in zip(unique, counts):
        name = LABEL_NAMES.get(int(value), "UNKNOWN")
        print(f"  {int(value):>2} = {name:<16} n={int(count)}")

    if "brain_area" in neuron_metadata:
        brain_area = np.asarray(neuron_metadata["brain_area"], dtype=object).ravel()
    else:
        brain_area = np.full(n_neurons, "unknown", dtype=object)

    if "cell_specimen_id" in neuron_metadata:
        cell_specimen_id = np.asarray(
            neuron_metadata["cell_specimen_id"], dtype=np.int64
        ).ravel()
    else:
        cell_specimen_id = np.full(n_neurons, -1, dtype=np.int64)

    if "ophys_experiment_id" in neuron_metadata:
        ophys_experiment_id = np.asarray(
            neuron_metadata["ophys_experiment_id"], dtype=np.int64
        ).ravel()
    else:
        ophys_experiment_id = np.full(n_neurons, -1, dtype=np.int64)

    for name, arr in [
        ("brain_area", brain_area),
        ("cell_specimen_id", cell_specimen_id),
        ("ophys_experiment_id", ophys_experiment_id),
    ]:
        if len(arr) != n_neurons:
            raise ValueError(
                f"neuron_metadata['{name}'] length {len(arr)} does not match "
                f"number of neurons {n_neurons}."
            )

    return (
        data,
        X,
        image_labels,
        labels_by_column,
        image_index_by_column,
        trial_index_by_column,
        brain_area,
        cell_specimen_id,
        ophys_experiment_id,
    )


# =============================================================================
# Design matrix
# =============================================================================

def make_image_design(image_labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Create four-class binary design matrix aligned to images.

    Output shape:
        118 × 4
    """
    design = np.zeros((len(image_labels), len(CLASSES)), dtype=np.float32)

    for j, cls in enumerate(CLASSES):
        design[:, j] = image_labels == cls

    return design, CLASS_NAMES.astype(str).tolist()


# =============================================================================
# LOO logistic encoding
# =============================================================================

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def compute_image_sums_and_counts(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Summarize binary trial activity per image.

    Returns:
        image_sums:
            neurons × images, number of active trials per neuron/image

        image_valid_counts:
            neurons × images, number of finite trials per neuron/image

        observed_image_probability:
            neurons × images, observed event probability per image
    """
    print()
    print("#" * 80)
    print("Precomputing image-level event probabilities")
    print("#" * 80)

    X_float = X.astype(np.float32, copy=False)
    X_img = X_float.reshape(X.shape[0], N_IMAGES, N_TRIALS)

    finite = np.isfinite(X_img)
    image_valid_counts = finite.sum(axis=2).astype(np.float32)
    image_sums = np.where(finite, X_img, 0.0).sum(axis=2).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        observed_image_probability = image_sums / image_valid_counts

    print(f"image_sums shape:                 {image_sums.shape}")
    print(f"image_valid_counts shape:         {image_valid_counts.shape}")
    print(f"observed_image_probability shape: {observed_image_probability.shape}")

    return image_sums, image_valid_counts, observed_image_probability.astype(np.float32)


def precompute_class_totals(
    image_sums: np.ndarray,
    image_valid_counts: np.ndarray,
    image_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Total events/counts per class before held-out-image subtraction.

    Returns:
        class_sums:
            classes × neurons

        class_counts:
            classes × neurons
    """
    n_neurons = image_sums.shape[0]
    class_sums = np.zeros((len(CLASSES), n_neurons), dtype=np.float32)
    class_counts = np.zeros((len(CLASSES), n_neurons), dtype=np.float32)

    for j, cls in enumerate(CLASSES):
        image_mask = image_labels == cls
        if not np.any(image_mask):
            raise ValueError(f"No images found for class {cls} = {LABEL_NAMES[int(cls)]}")

        class_sums[j] = image_sums[:, image_mask].sum(axis=1)
        class_counts[j] = image_valid_counts[:, image_mask].sum(axis=1)

    return class_sums, class_counts


def run_loo_logistic_image_probabilities(
    image_labels: np.ndarray,
    image_sums: np.ndarray,
    image_valid_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run image-level LOO logistic encoding.

    Returns:
        synthetic_image_probability:
            neurons × images predicted probabilities

        synthetic_image_logit:
            neurons × images predicted logits

        fold_id_by_image:
            length-118 fold/image ids
    """
    n_neurons = image_sums.shape[0]

    class_sums, class_counts = precompute_class_totals(
        image_sums=image_sums,
        image_valid_counts=image_valid_counts,
        image_labels=image_labels,
    )

    synthetic_image_probability = np.full(
        (n_neurons, N_IMAGES), np.nan, dtype=np.float32
    )
    synthetic_image_logit = np.full(
        (n_neurons, N_IMAGES), np.nan, dtype=np.float32
    )
    fold_id_by_image = np.full(N_IMAGES, -1, dtype=np.int64)

    print()
    print("#" * 80)
    print("Running leave-one-image-out one-hot logistic encoding")
    print("#" * 80)
    print(f"Neurons: {n_neurons}")
    print(f"Images:  {N_IMAGES}")
    print(f"Output synthetic probabilities: neurons × images = {synthetic_image_probability.shape}")

    for image_idx in range(N_IMAGES):
        cls = int(image_labels[image_idx])

        if cls not in CLASSES:
            print(
                f"[fold {image_idx + 1:>3}/{N_IMAGES}] "
                f"image={image_idx:>3} label={cls:>2} skipped: unlabeled/not in classes"
            )
            continue

        class_j = int(np.where(CLASSES == cls)[0][0])

        # Leave out this image from the training base.
        train_sums_for_class = class_sums[class_j] - image_sums[:, image_idx]
        train_counts_for_class = class_counts[class_j] - image_valid_counts[:, image_idx]

        if np.any(train_counts_for_class <= 0):
            bad = int(np.sum(train_counts_for_class <= 0))
            raise ValueError(
                f"Fold image {image_idx}, class {cls}, has {bad} neurons with no "
                "valid training trials for the held-out class."
            )

        p_hat = train_sums_for_class / train_counts_for_class
        p_hat = np.clip(p_hat, EPS, 1.0 - EPS).astype(np.float32)
        z_hat = logit(p_hat).astype(np.float32)

        # One probability per held-out image and neuron.
        synthetic_image_probability[:, image_idx] = p_hat
        synthetic_image_logit[:, image_idx] = z_hat
        fold_id_by_image[image_idx] = image_idx

        if image_idx == 0 or (image_idx + 1) % 10 == 0 or image_idx == N_IMAGES - 1:
            print(
                f"[fold {image_idx + 1:>3}/{N_IMAGES}] "
                f"held-out image={image_idx:>3} "
                f"class={cls} ({LABEL_NAMES[cls]:<16}) "
                f"p_mean={float(np.mean(p_hat)):.5f}"
            )

    return synthetic_image_probability, synthetic_image_logit, fold_id_by_image


# =============================================================================
# Metrics / summaries
# =============================================================================

def compute_image_probability_metrics(
    observed_image_probability: np.ndarray,
    synthetic_image_probability: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    """
    Compare observed image-level probabilities to synthetic image-level probabilities.

    This is not binary log-loss anymore; it is image-level probability prediction.
    """
    pred_images = np.isfinite(synthetic_image_probability[0])

    Y = observed_image_probability[:, pred_images].astype(np.float32, copy=False)
    P = synthetic_image_probability[:, pred_images].astype(np.float32, copy=False)

    finite = np.isfinite(Y) & np.isfinite(P)
    valid_counts = finite.sum(axis=1).astype(np.float32)

    diff2 = np.where(finite, (Y - P) ** 2, 0.0)
    mse_by_neuron = diff2.sum(axis=1) / valid_counts

    y_mean = np.where(finite, Y, 0.0).sum(axis=1) / valid_counts
    ss_res = diff2.sum(axis=1)
    ss_tot = np.where(finite, (Y - y_mean[:, None]) ** 2, 0.0).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        r2_by_neuron = 1.0 - ss_res / ss_tot

    return {
        "predicted_images": pred_images,
        "valid_image_counts_by_neuron": valid_counts,
        "mse_image_probability_by_neuron": mse_by_neuron.astype(np.float32),
        "r2_image_probability_by_neuron": r2_by_neuron.astype(np.float32),
        "mean_mse_image_probability": float(np.nanmean(mse_by_neuron)),
        "mean_r2_image_probability": float(np.nanmean(r2_by_neuron)),
        "n_predicted_images": int(pred_images.sum()),
    }


def compute_trial_log_loss_metrics(
    X: np.ndarray,
    synthetic_image_probability: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    """
    Evaluate the 118 image probabilities against all binary trials.

    This repeats each image probability across its 50 trials only for evaluation,
    but the saved synthetic output remains neurons × 118.
    """
    pred_images = np.isfinite(synthetic_image_probability[0])
    pred_cols = np.repeat(pred_images, N_TRIALS)

    Y = X[:, pred_cols].astype(np.float32, copy=False)
    P_image = synthetic_image_probability[:, pred_images].astype(np.float32, copy=False)
    P = np.repeat(P_image, N_TRIALS, axis=1)
    P = np.clip(P, EPS, 1.0 - EPS)

    finite = np.isfinite(Y)
    Y0 = np.where(finite, Y, 0.0)
    valid_counts = finite.sum(axis=1).astype(np.float32)

    event_rate = Y0.sum(axis=1) / valid_counts

    log_loss_terms = -(Y0 * np.log(P) + (1.0 - Y0) * np.log(1.0 - P))
    log_loss_sum = np.where(finite, log_loss_terms, 0.0).sum(axis=1)
    log_loss_by_neuron = log_loss_sum / valid_counts

    p_null = np.clip(event_rate[:, None], EPS, 1.0 - EPS)
    null_log_loss_terms = -(Y0 * np.log(p_null) + (1.0 - Y0) * np.log(1.0 - p_null))
    null_log_loss_sum = np.where(finite, null_log_loss_terms, 0.0).sum(axis=1)
    null_log_loss_by_neuron = null_log_loss_sum / valid_counts

    pseudo_r2_log_loss = 1.0 - (log_loss_by_neuron / null_log_loss_by_neuron)

    return {
        "predicted_trial_columns": pred_cols,
        "event_rate_by_neuron": event_rate.astype(np.float32),
        "log_loss_by_neuron": log_loss_by_neuron.astype(np.float32),
        "null_log_loss_by_neuron": null_log_loss_by_neuron.astype(np.float32),
        "pseudo_r2_log_loss_by_neuron": pseudo_r2_log_loss.astype(np.float32),
        "mean_log_loss": float(np.nanmean(log_loss_by_neuron)),
        "mean_null_log_loss": float(np.nanmean(null_log_loss_by_neuron)),
        "mean_pseudo_r2_log_loss": float(np.nanmean(pseudo_r2_log_loss)),
        "n_predicted_trial_columns": int(pred_cols.sum()),
    }


def summarize_by_area(values: np.ndarray, brain_area: np.ndarray, metric_name: str) -> list[dict]:
    area_str = brain_area.astype(str)
    rows = []

    print()
    print(f"By brain area: {metric_name}")
    for area in sorted(np.unique(area_str)):
        mask = area_str == area
        vals = values[mask]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue

        row = {
            "brain_area": area,
            "n_neurons": int(len(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n_gt_0": int(np.sum(vals > 0.0)),
            "n_gt_0p01": int(np.sum(vals > 0.01)),
        }
        rows.append(row)

        print(
            f"  {area:<8} "
            f"n={row['n_neurons']:>6} "
            f"mean={row['mean']:>10.6f} "
            f"median={row['median']:>10.6f} "
            f"max={row['max']:>10.6f} "
            f"n>0={row['n_gt_0']:>6}"
        )

    return rows


def print_top_neurons(
    values: np.ndarray,
    brain_area: np.ndarray,
    cell_specimen_id: np.ndarray,
    ophys_experiment_id: np.ndarray,
    metric_name: str,
    n_top: int = 30,
) -> None:
    finite_idx = np.where(np.isfinite(values))[0]
    order = finite_idx[np.argsort(values[finite_idx])[::-1]]

    print()
    print("=" * 80)
    print(f"Top {n_top} neurons by {metric_name}")
    print("=" * 80)

    for rank, idx in enumerate(order[:n_top], start=1):
        print(
            f"{rank:>2}. "
            f"neuron_idx={idx:>6} "
            f"{metric_name}={float(values[idx]):>10.6f} "
            f"area={str(brain_area[idx]):<8} "
            f"cell_specimen_id={int(cell_specimen_id[idx])} "
            f"ophys_experiment_id={int(ophys_experiment_id[idx])}"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    (
        data,
        X,
        image_labels,
        labels_by_column,
        image_index_by_column,
        trial_index_by_column,
        brain_area,
        cell_specimen_id,
        ophys_experiment_id,
    ) = load_composite()

    image_design, design_column_names = make_image_design(image_labels)

    print()
    print("=" * 80)
    print("Image-level design matrix")
    print("=" * 80)
    print(f"Design shape:   {image_design.shape}")
    print(f"Design columns: {design_column_names}")

    image_sums, image_valid_counts, observed_image_probability = compute_image_sums_and_counts(X)

    synthetic_image_probability, synthetic_image_logit, fold_id_by_image = (
        run_loo_logistic_image_probabilities(
            image_labels=image_labels,
            image_sums=image_sums,
            image_valid_counts=image_valid_counts,
        )
    )

    np.save(OUT_SYNTHETIC_NPY, synthetic_image_probability, allow_pickle=False)
    np.save(OUT_OBSERVED_NPY, observed_image_probability, allow_pickle=False)

    image_metrics = compute_image_probability_metrics(
        observed_image_probability=observed_image_probability,
        synthetic_image_probability=synthetic_image_probability,
    )

    trial_metrics = compute_trial_log_loss_metrics(
        X=X,
        synthetic_image_probability=synthetic_image_probability,
    )

    print()
    print("=" * 80)
    print("Cross-validated synthetic image probability metrics")
    print("=" * 80)
    print(f"Predicted images:              {image_metrics['n_predicted_images']} / {N_IMAGES}")
    print(f"Synthetic output shape:        {synthetic_image_probability.shape}")
    print(f"Mean image-probability MSE:    {image_metrics['mean_mse_image_probability']:.6f}")
    print(f"Mean image-probability R2:     {image_metrics['mean_r2_image_probability']:.6f}")
    print(f"Trial-level mean log loss:     {trial_metrics['mean_log_loss']:.6f}")
    print(f"Trial-level null log loss:     {trial_metrics['mean_null_log_loss']:.6f}")
    print(f"Trial-level pseudo-R2:         {trial_metrics['mean_pseudo_r2_log_loss']:.6f}")

    area_summary_image_r2 = summarize_by_area(
        values=image_metrics["r2_image_probability_by_neuron"],
        brain_area=brain_area,
        metric_name="r2_image_probability",
    )

    area_summary_trial_pseudo_r2 = summarize_by_area(
        values=trial_metrics["pseudo_r2_log_loss_by_neuron"],
        brain_area=brain_area,
        metric_name="pseudo_r2_log_loss",
    )

    print_top_neurons(
        values=image_metrics["r2_image_probability_by_neuron"],
        brain_area=brain_area,
        cell_specimen_id=cell_specimen_id,
        ophys_experiment_id=ophys_experiment_id,
        metric_name="r2_image_probability",
        n_top=30,
    )

    np.savez_compressed(
        OUT_RESULTS_NPZ,
        synthetic_image_probability=synthetic_image_probability,
        observed_image_probability=observed_image_probability,
        synthetic_image_logit=synthetic_image_logit,
        synthetic_image_probability_path=np.asarray(str(OUT_SYNTHETIC_NPY), dtype=object),
        observed_image_probability_path=np.asarray(str(OUT_OBSERVED_NPY), dtype=object),
        composite_path=np.asarray(str(COMPOSITE_PATH), dtype=object),
        image_design=image_design,
        design_column_names=np.asarray(design_column_names, dtype=object),
        classes=CLASSES,
        class_names=CLASS_NAMES,
        image_labels=image_labels,
        labels_by_column=labels_by_column,
        image_index_by_column=image_index_by_column,
        trial_index_by_column=trial_index_by_column,
        fold_id_by_image=fold_id_by_image,
        predicted_images=image_metrics["predicted_images"],
        predicted_trial_columns=trial_metrics["predicted_trial_columns"],
        image_valid_counts=image_valid_counts,
        valid_image_counts_by_neuron=image_metrics["valid_image_counts_by_neuron"],
        mse_image_probability_by_neuron=image_metrics["mse_image_probability_by_neuron"],
        r2_image_probability_by_neuron=image_metrics["r2_image_probability_by_neuron"],
        event_rate_by_neuron=trial_metrics["event_rate_by_neuron"],
        log_loss_by_neuron=trial_metrics["log_loss_by_neuron"],
        null_log_loss_by_neuron=trial_metrics["null_log_loss_by_neuron"],
        pseudo_r2_log_loss_by_neuron=trial_metrics["pseudo_r2_log_loss_by_neuron"],
        brain_area=brain_area,
        cell_specimen_id=cell_specimen_id,
        ophys_experiment_id=ophys_experiment_id,
        n_images=N_IMAGES,
        n_trials=N_TRIALS,
        eps=EPS,
        cv_mode=np.asarray("leave_one_image_out", dtype=object),
        model=np.asarray("no_intercept_four_class_one_hot_logistic_regression", dtype=object),
    )

    summary = {
        "composite_path": str(COMPOSITE_PATH),
        "synthetic_image_probability_path": str(OUT_SYNTHETIC_NPY),
        "observed_image_probability_path": str(OUT_OBSERVED_NPY),
        "results_npz": str(OUT_RESULTS_NPZ),
        "n_neurons": int(X.shape[0]),
        "n_trial_columns": int(X.shape[1]),
        "n_images": int(N_IMAGES),
        "n_trials_per_image": int(N_TRIALS),
        "synthetic_image_probability_shape": list(synthetic_image_probability.shape),
        "observed_image_probability_shape": list(observed_image_probability.shape),
        "n_predicted_images": int(image_metrics["n_predicted_images"]),
        "n_predicted_trial_columns_for_eval": int(trial_metrics["n_predicted_trial_columns"]),
        "mean_mse_image_probability": float(image_metrics["mean_mse_image_probability"]),
        "mean_r2_image_probability": float(image_metrics["mean_r2_image_probability"]),
        "mean_log_loss": float(trial_metrics["mean_log_loss"]),
        "mean_null_log_loss": float(trial_metrics["mean_null_log_loss"]),
        "mean_pseudo_r2_log_loss": float(trial_metrics["mean_pseudo_r2_log_loss"]),
        "design_column_names": design_column_names,
        "area_summary_image_r2": area_summary_image_r2,
        "area_summary_trial_pseudo_r2": area_summary_trial_pseudo_r2,
    }

    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("#" * 80)
    print("DONE")
    print("#" * 80)
    print("Saved synthetic image-level neural activity probabilities:")
    print(f"  {OUT_SYNTHETIC_NPY}")
    print("Saved observed image-level neural activity probabilities:")
    print(f"  {OUT_OBSERVED_NPY}")
    print("Saved result metadata/metrics:")
    print(f"  {OUT_RESULTS_NPZ}")
    print("Saved summary JSON:")
    print(f"  {OUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
