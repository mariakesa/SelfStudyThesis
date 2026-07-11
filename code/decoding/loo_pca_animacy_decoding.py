#!/usr/bin/env python3
"""
Leakage-safe leave-one-out animacy decoding as a function of PCA dimensionality.

This extends the existing LOO PCA reconstruction analysis while preserving its
data-loading logic and output-directory family.

Two PCA geometries are evaluated:

A. centered
   Training-fold mean centering only. This is the natural continuation of the
   centered-norm reconstruction analysis.

B. standardized
   StandardScaler fitted on each training fold, followed by PCA. This gives
   low-variance and high-variance neurons equal weight.

For every held-out image and every requested k:
    1. Hold out exactly one image.
    2. Fit all preprocessing on the other images only.
    3. Fit PCA on the preprocessed training fold only.
    4. Project train and held-out image into the training-derived PCA space.
    5. Fit logistic regression on the first k training PC scores.
    6. Save held-out class, animate probability, decision score, and true label.

The script also:
    * computes accuracy, balanced accuracy, ROC AUC, confusion matrices,
      Wilson intervals, and bootstrap AUC intervals;
    * saves fold-level predictions;
    * compares every PCA model with the exact same-image full-feature
      predictions using McNemar's paired test;
    * reports incremental gains and the smallest predictive dimensionality;
    * connects predictive dimensionality to the previously measured
      held-out reconstruction fraction.

Input:
    /home/maria/SelfStudyThesis/data/
        allen_natural_scenes_four_class_composite.npy

Outputs:
    /home/maria/SelfStudyThesis/results/
        all_neurons_loo_pca_reconstruction/animacy_decoding/

Notes on the full-feature reference
-----------------------------------
The script first attempts to load the original full-feature LOO predictions
from FULL_FEATURE_RESULTS_PATH. Several common NPZ key names are supported.
If exact predictions cannot be loaded, it computes a leakage-safe fallback
full-feature LOO classifier using the same sklearn logistic configuration and
fold-wise StandardScaler. The output explicitly records which reference was
used.

Binary labels:
    1 = animals (four-class label 0)
    0 = landscape, plant, or man-made object (labels 1, 2, 3)
"""

from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Paths
# =============================================================================

COMPOSITE_PATH = Path(
    "/home/maria/SelfStudyThesis/data/"
    "allen_natural_scenes_four_class_composite.npy"
)

# Preserve the reconstruction analysis output-directory family.
RECONSTRUCTION_OUTDIR = Path(
    "/home/maria/SelfStudyThesis/results/"
    "all_neurons_loo_pca_reconstruction"
)

OUTDIR = RECONSTRUCTION_OUTDIR / "animacy_decoding"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Original full-neuron Adam LOO output, if available.
FULL_FEATURE_RESULTS_PATH = Path(
    "/home/maria/SelfStudyThesis/results/"
    "all_neurons_animals_vs_rest_adam_loo/"
    "all_neurons_animals_vs_rest_adam_loo_results.npz"
)

RAW_PREDICTIONS_CSV = OUTDIR / "loo_pca_animacy_fold_predictions.csv"
SUMMARY_CSV = OUTDIR / "loo_pca_animacy_summary.csv"
INCREMENTAL_CSV = OUTDIR / "loo_pca_animacy_incremental_gain.csv"
PAIRED_CSV = OUTDIR / "loo_pca_animacy_paired_mcnemar.csv"
CONFUSIONS_CSV = OUTDIR / "loo_pca_animacy_confusion_matrices.csv"
RESULTS_NPZ = OUTDIR / "loo_pca_animacy_results.npz"
INTERPRETATION_TXT = OUTDIR / "loo_pca_animacy_interpretation.txt"


# =============================================================================
# Settings
# =============================================================================

RANDOM_SEED = 0
EPS = 1e-12

EXCLUDE_UNLABELED = True

# Keep the same complete-dataset nonfinite/constant-neuron cleanup convention
# as the reconstruction script, then apply a fold-local variance check too.
GLOBAL_VARIANCE_TOL = 1e-15
FOLD_VARIANCE_TOL = 1e-15

K_VALUES = [1, 2, 3, 5, 10, 20, 40, 60, 80, 100, 116]
PCA_MODES = ("centered", "standardized")

LOGISTIC_KWARGS = dict(
    penalty="l2",
    C=1.0,
    solver="liblinear",
    max_iter=10000,
    class_weight=None,
    random_state=RANDOM_SEED,
)

# Bootstrap samples for the ROC-AUC confidence interval.
N_AUC_BOOTSTRAP = 5000

# Paired significance threshold for McNemar's exact test.
ALPHA = 0.05

# Existing held-out reconstruction values supplied by the user.
RECONSTRUCTION_FRACTION = {
    1: 0.0330,
    2: 0.0480,
    3: 0.0684,
    5: 0.0923,
    10: 0.1197,
    20: 0.1517,
    40: 0.1761,
    60: 0.1901,
    80: 0.2004,
    100: 0.2073,
}

LABEL_NAMES = {
    -1: "unlabeled",
     0: "animals",
     1: "landscape",
     2: "plant",
     3: "man-made object",
}


# =============================================================================
# Data loading, preserved from the reconstruction script
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
# Data retention and binary labels
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

    # Animal is positive class.
    y = (labels == 0).astype(np.int64)

    unique_y, counts_y = np.unique(y, return_counts=True)
    print("Binary animacy counts:")
    for value, count in zip(unique_y, counts_y):
        print(f"  y={value}: n={count}")

    if set(unique_y.tolist()) != {0, 1}:
        raise ValueError("Binary labels must contain both classes 0 and 1.")

    return (
        X,
        labels,
        y,
        original_image_indices,
        neuron_mask,
        brain_area[neuron_mask],
        cell_specimen_id[neuron_mask],
        ophys_experiment_id[neuron_mask],
    )


# =============================================================================
# PCA in sample space
# =============================================================================

def fit_fold_pca_scores(
    X_train: np.ndarray,
    x_test: np.ndarray,
    mode: str,
    max_k: int,
):
    """
    Fit fold-local preprocessing and PCA, then return train/test PC scores.

    The PCA is computed through the small training-sample Gram matrix rather
    than a 40,064 × 116 feature-space decomposition.

    For centered training matrix A = U S V^T:
        training scores = U S
        held-out scores = (x A^T U) / S
    """
    if mode not in PCA_MODES:
        raise ValueError(f"Unknown PCA mode: {mode}")

    # Fold-local feature filter: derived from training responses only.
    fold_var = np.var(X_train, axis=0, ddof=0)
    feature_mask = np.isfinite(fold_var) & (fold_var > FOLD_VARIANCE_TOL)

    X_train_f = X_train[:, feature_mask]
    x_test_f = x_test[feature_mask].reshape(1, -1)

    if X_train_f.shape[1] == 0:
        raise ValueError("No fold-valid neurons remain.")

    if mode == "standardized":
        scaler = StandardScaler(with_mean=True, with_std=True)
        A = scaler.fit_transform(X_train_f)
        x = scaler.transform(x_test_f).ravel()
    else:
        train_mean = X_train_f.mean(axis=0)
        A = X_train_f - train_mean
        x = (x_test_f.ravel() - train_mean)

    # Numerical cleanup.
    A = np.asarray(A, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    gram = A @ A.T
    eigvals, U = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    U = U[:, order]

    if eigvals.size == 0 or eigvals[0] <= EPS:
        raise ValueError("Training fold has no nonzero PCA directions.")

    rank_tol = max(A.shape) * np.finfo(np.float64).eps * eigvals[0]
    positive = eigvals > rank_tol
    numerical_rank = int(positive.sum())

    requested_rank = min(max_k, A.shape[0] - 1)
    usable_rank = min(numerical_rank, requested_rank)

    if usable_rank < requested_rank:
        warnings.warn(
            f"Fold numerical rank {usable_rank} is below requested "
            f"rank {requested_rank}; unavailable dimensions will be omitted."
        )

    eigvals_r = eigvals[:usable_rank]
    U_r = U[:, :usable_rank]
    singular_values = np.sqrt(eigvals_r)

    train_scores = U_r * singular_values[np.newaxis, :]

    similarities = x @ A.T
    test_scores = (similarities @ U_r) / singular_values

    return (
        train_scores,
        test_scores,
        numerical_rank,
        int(feature_mask.sum()),
    )


# =============================================================================
# LOO PCA decoding
# =============================================================================

def run_loo_pca_decoding(
    X: np.ndarray,
    y: np.ndarray,
    original_image_indices: np.ndarray,
):
    n_samples = X.shape[0]
    max_k = max(K_VALUES)

    rows: list[dict] = []
    fold_diagnostics: list[dict] = []

    for fold_idx in range(n_samples):
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[fold_idx] = False

        X_train = X[train_mask]
        y_train = y[train_mask]
        x_test = X[fold_idx]
        y_test = int(y[fold_idx])

        if len(np.unique(y_train)) != 2:
            raise ValueError(f"Fold {fold_idx} training labels contain one class.")

        for mode in PCA_MODES:
            (
                train_scores,
                test_scores,
                numerical_rank,
                n_fold_features,
            ) = fit_fold_pca_scores(
                X_train=X_train,
                x_test=x_test,
                mode=mode,
                max_k=max_k,
            )

            fold_diagnostics.append(
                {
                    "fold_index": fold_idx,
                    "original_image_index": int(original_image_indices[fold_idx]),
                    "mode": mode,
                    "numerical_rank": numerical_rank,
                    "n_fold_features": n_fold_features,
                }
            )

            for k in K_VALUES:
                if k > train_scores.shape[1]:
                    raise ValueError(
                        f"Requested k={k}, but fold {fold_idx}, mode={mode} "
                        f"has only {train_scores.shape[1]} usable PCs."
                    )

                model = LogisticRegression(**LOGISTIC_KWARGS)
                model.fit(train_scores[:, :k], y_train)

                test_2d = test_scores[:k].reshape(1, -1)
                predicted_class = int(model.predict(test_2d)[0])
                animate_probability = float(
                    model.predict_proba(test_2d)[0, 1]
                )
                decision_score = float(
                    np.asarray(model.decision_function(test_2d)).ravel()[0]
                )

                rows.append(
                    {
                        "mode": mode,
                        "k": int(k),
                        "fold_index": int(fold_idx),
                        "original_image_index": int(
                            original_image_indices[fold_idx]
                        ),
                        "true_label": y_test,
                        "predicted_class": predicted_class,
                        "animate_probability": animate_probability,
                        "decision_score": decision_score,
                        "correct": int(predicted_class == y_test),
                    }
                )

        print(
            f"\rPCA decoding fold {fold_idx + 1:>3}/{n_samples}",
            end="",
            flush=True,
        )

    print()
    return pd.DataFrame(rows), pd.DataFrame(fold_diagnostics)


# =============================================================================
# Full-feature reference
# =============================================================================

def _first_matching_key(npz, candidates):
    for key in candidates:
        if key in npz.files:
            return key
    return None


def load_original_full_feature_predictions(
    path: Path,
    y: np.ndarray,
    original_image_indices: np.ndarray,
):
    """
    Load exact-image predictions from the original LOO result if possible.

    Returns:
        dataframe or None
    """
    if not path.exists():
        return None

    npz = np.load(path, allow_pickle=True)

    pred_key = _first_matching_key(
        npz,
        [
            "y_pred",
            "predicted_class",
            "predictions",
            "loo_predictions",
            "pred_class",
        ],
    )
    prob_key = _first_matching_key(
        npz,
        [
            "y_prob",
            "y_probability",
            "predicted_probability",
            "probabilities",
            "animate_probability",
            "probs",
        ],
    )
    score_key = _first_matching_key(
        npz,
        [
            "decision_scores",
            "decision_score",
            "logits",
            "scores",
            "loo_scores",
        ],
    )
    true_key = _first_matching_key(
        npz,
        ["y_true", "true_label", "labels_binary", "binary_labels", "y"],
    )
    index_key = _first_matching_key(
        npz,
        ["original_image_indices", "image_indices", "stimulus_indices"],
    )

    if pred_key is None and prob_key is None and score_key is None:
        print(
            "[WARN] Full-feature NPZ exists but no recognized prediction, "
            "probability, or score key was found."
        )
        print(f"       Available keys: {npz.files}")
        return None

    n = len(y)

    pred = None
    prob = None
    score = None

    if pred_key is not None:
        pred = np.asarray(npz[pred_key]).ravel()
    if prob_key is not None:
        prob = np.asarray(npz[prob_key]).ravel()
    if score_key is not None:
        score = np.asarray(npz[score_key]).ravel()

    lengths = [
        len(arr) for arr in (pred, prob, score) if arr is not None
    ]
    if not lengths or any(length != n for length in lengths):
        print(
            "[WARN] Full-feature prediction arrays do not match the retained "
            f"sample count n={n}. Lengths={lengths}"
        )
        return None

    # If probabilities are Nx2, recover positive-class probability.
    if prob_key is not None:
        raw_prob = np.asarray(npz[prob_key])
        if raw_prob.ndim == 2 and raw_prob.shape == (n, 2):
            prob = raw_prob[:, 1]
        elif raw_prob.ndim > 1:
            prob = raw_prob.reshape(n, -1)[:, -1]

    if pred is None:
        if prob is not None:
            pred = (prob >= 0.5).astype(np.int64)
        else:
            pred = (score >= 0.0).astype(np.int64)

    pred = pred.astype(np.int64)

    if score is None:
        if prob is not None:
            clipped = np.clip(prob, EPS, 1.0 - EPS)
            score = np.log(clipped / (1.0 - clipped))
        else:
            score = pred.astype(np.float64)

    if prob is None:
        prob = 1.0 / (1.0 + np.exp(-np.clip(score, -40, 40)))

    if true_key is not None:
        loaded_true = np.asarray(npz[true_key]).ravel().astype(np.int64)
        if len(loaded_true) == n and not np.array_equal(loaded_true, y):
            print(
                "[WARN] Loaded full-feature y_true differs from the current "
                "sample ordering; refusing to use it."
            )
            return None

    if index_key is not None:
        loaded_indices = np.asarray(npz[index_key]).ravel().astype(np.int64)
        if (
            len(loaded_indices) == n
            and not np.array_equal(loaded_indices, original_image_indices)
        ):
            print(
                "[WARN] Loaded full-feature image indices differ from the "
                "current ordering; refusing to use them."
            )
            return None

    return pd.DataFrame(
        {
            "fold_index": np.arange(n, dtype=np.int64),
            "original_image_index": original_image_indices.astype(np.int64),
            "true_label": y.astype(np.int64),
            "predicted_class": pred,
            "animate_probability": prob.astype(np.float64),
            "decision_score": score.astype(np.float64),
            "correct": (pred == y).astype(np.int64),
            "reference_source": "original_full_feature_npz",
        }
    )


def compute_fallback_full_feature_predictions(
    X: np.ndarray,
    y: np.ndarray,
    original_image_indices: np.ndarray,
):
    """
    Leakage-safe fallback full-feature reference.

    This is not claimed to reproduce the Adam classifier. It uses the requested
    sklearn L2/liblinear convention and a training-fold StandardScaler.
    """
    rows = []
    n = len(y)

    print()
    print(
        "[INFO] Computing fallback full-feature LOO reference with fold-wise "
        "StandardScaler + sklearn logistic regression."
    )

    for fold_idx in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[fold_idx] = False

        X_train = X[train_mask]
        y_train = y[train_mask]
        x_test = X[fold_idx].reshape(1, -1)

        fold_var = np.var(X_train, axis=0, ddof=0)
        feature_mask = np.isfinite(fold_var) & (
            fold_var > FOLD_VARIANCE_TOL
        )

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_scaled = scaler.fit_transform(X_train[:, feature_mask])
        x_test_scaled = scaler.transform(x_test[:, feature_mask])

        model = LogisticRegression(**LOGISTIC_KWARGS)
        model.fit(X_train_scaled, y_train)

        pred = int(model.predict(x_test_scaled)[0])
        prob = float(model.predict_proba(x_test_scaled)[0, 1])
        score = float(
            np.asarray(model.decision_function(x_test_scaled)).ravel()[0]
        )

        rows.append(
            {
                "fold_index": fold_idx,
                "original_image_index": int(original_image_indices[fold_idx]),
                "true_label": int(y[fold_idx]),
                "predicted_class": pred,
                "animate_probability": prob,
                "decision_score": score,
                "correct": int(pred == y[fold_idx]),
                "reference_source": "fallback_scaled_full_feature_sklearn",
            }
        )

        print(
            f"\rFull-feature fallback fold {fold_idx + 1:>3}/{n}",
            end="",
            flush=True,
        )

    print()
    return pd.DataFrame(rows)


# =============================================================================
# Statistics
# =============================================================================

def wilson_interval(
    n_correct: int,
    n_total: int,
    confidence: float = 0.95,
):
    if n_total <= 0:
        return np.nan, np.nan

    # z=1.959963984540054 for 95%.
    if confidence != 0.95:
        raise ValueError("This implementation currently supports 95% only.")
    z = 1.959963984540054

    p = n_correct / n_total
    denominator = 1.0 + z**2 / n_total
    center = (p + z**2 / (2.0 * n_total)) / denominator
    half_width = (
        z
        * np.sqrt(
            p * (1.0 - p) / n_total
            + z**2 / (4.0 * n_total**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def bootstrap_auc_interval(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = N_AUC_BOOTSTRAP,
):
    aucs = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        y_boot = y_true[indices]

        if np.unique(y_boot).size < 2:
            continue

        aucs.append(roc_auc_score(y_boot, probabilities[indices]))

    if not aucs:
        return np.nan, np.nan

    return (
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
    )


def summarize_predictions(
    predictions: pd.DataFrame,
    rng: np.random.Generator,
):
    summary_rows = []
    confusion_rows = []

    group_columns = ["mode", "k"]

    for (mode, k), group in predictions.groupby(
        group_columns, sort=False
    ):
        group = group.sort_values("fold_index")
        y_true = group["true_label"].to_numpy(dtype=np.int64)
        y_pred = group["predicted_class"].to_numpy(dtype=np.int64)
        probability = group["animate_probability"].to_numpy(dtype=float)

        n_total = len(group)
        n_correct = int((y_true == y_pred).sum())
        accuracy = accuracy_score(y_true, y_pred)
        balanced = balanced_accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, probability)
        ci_low, ci_high = wilson_interval(n_correct, n_total)
        auc_low, auc_high = bootstrap_auc_interval(
            y_true, probability, rng
        )

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        summary_rows.append(
            {
                "mode": mode,
                "k": int(k),
                "correct": n_correct,
                "n": n_total,
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "roc_auc": auc,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "auc_ci_low": auc_low,
                "auc_ci_high": auc_high,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

        confusion_rows.append(
            {
                "mode": mode,
                "k": int(k),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "matrix": f"[[{tn}, {fp}], [{fn}, {tp}]]",
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(confusion_rows)


def summarize_full_reference(
    full_predictions: pd.DataFrame,
    rng: np.random.Generator,
):
    y_true = full_predictions["true_label"].to_numpy(dtype=np.int64)
    y_pred = full_predictions["predicted_class"].to_numpy(dtype=np.int64)
    probability = full_predictions[
        "animate_probability"
    ].to_numpy(dtype=float)

    n = len(y_true)
    correct = int((y_true == y_pred).sum())
    accuracy = accuracy_score(y_true, y_pred)
    balanced = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probability)
    ci_low, ci_high = wilson_interval(correct, n)
    auc_low, auc_high = bootstrap_auc_interval(y_true, probability, rng)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "mode": "full",
        "k": "full",
        "correct": correct,
        "n": n,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "roc_auc": auc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "auc_ci_low": auc_low,
        "auc_ci_high": auc_high,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "reference_source": full_predictions[
            "reference_source"
        ].iloc[0],
    }


def exact_mcnemar(
    model_correct: np.ndarray,
    full_correct: np.ndarray,
):
    """
    Exact McNemar test using discordant fold-wise correctness pairs.

    b = PCA correct, full wrong
    c = PCA wrong, full correct
    Under H0, b ~ Binomial(b+c, 0.5).
    """
    model_correct = np.asarray(model_correct, dtype=bool)
    full_correct = np.asarray(full_correct, dtype=bool)

    b = int(np.sum(model_correct & ~full_correct))
    c = int(np.sum(~model_correct & full_correct))
    discordant = b + c

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(
            binomtest(
                k=min(b, c),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    return b, c, discordant, p_value


def paired_comparisons(
    pca_predictions: pd.DataFrame,
    full_predictions: pd.DataFrame,
):
    full = full_predictions.sort_values("fold_index")
    full_correct = full["correct"].to_numpy(dtype=bool)
    full_indices = full["original_image_index"].to_numpy(dtype=np.int64)

    rows = []

    for (mode, k), group in pca_predictions.groupby(
        ["mode", "k"], sort=False
    ):
        group = group.sort_values("fold_index")

        indices = group["original_image_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(indices, full_indices):
            raise ValueError(
                f"Sample ordering mismatch for mode={mode}, k={k}."
            )

        model_correct = group["correct"].to_numpy(dtype=bool)
        b, c, discordant, p_value = exact_mcnemar(
            model_correct, full_correct
        )

        rows.append(
            {
                "mode": mode,
                "k": int(k),
                "pca_correct_full_wrong": b,
                "pca_wrong_full_correct": c,
                "discordant_pairs": discordant,
                "mcnemar_exact_p": p_value,
                "statistically_indistinguishable_at_0.05": (
                    p_value >= ALPHA
                ),
            }
        )

    return pd.DataFrame(rows)


def incremental_gain_table(
    summary: pd.DataFrame,
    full_accuracy: float,
):
    rows = []

    for mode in PCA_MODES:
        mode_table = (
            summary[summary["mode"] == mode]
            .sort_values("k")
            .reset_index(drop=True)
        )

        previous = np.nan
        for _, row in mode_table.iterrows():
            accuracy = float(row["accuracy"])
            change = (
                np.nan if np.isnan(previous) else accuracy - previous
            )
            rows.append(
                {
                    "mode": mode,
                    "k": int(row["k"]),
                    "accuracy": accuracy,
                    "change_from_previous_k": change,
                    "difference_from_full_model": (
                        accuracy - full_accuracy
                    ),
                }
            )
            previous = accuracy

    return pd.DataFrame(rows)


# =============================================================================
# Interpretation
# =============================================================================

def reconstruction_at_k(k: int):
    if k in RECONSTRUCTION_FRACTION:
        return RECONSTRUCTION_FRACTION[k], "exact supplied value"

    lower = sorted(x for x in RECONSTRUCTION_FRACTION if x < k)
    upper = sorted(x for x in RECONSTRUCTION_FRACTION if x > k)

    if lower and upper:
        k0 = lower[-1]
        k1 = upper[0]
        y0 = RECONSTRUCTION_FRACTION[k0]
        y1 = RECONSTRUCTION_FRACTION[k1]
        fraction = y0 + (k - k0) * (y1 - y0) / (k1 - k0)
        return fraction, f"linear interpolation between k={k0} and k={k1}"

    return np.nan, "not available from supplied reconstruction values"


def build_interpretation(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    full_summary: dict,
):
    lines = []
    full_accuracy = float(full_summary["accuracy"])

    lines.append("CONCISE INTERPRETATION")
    lines.append("=" * 80)
    lines.append(
        f"Full-feature reference accuracy: {full_accuracy:.4f} "
        f"({full_summary['correct']}/{full_summary['n']}); "
        f"source={full_summary['reference_source']}."
    )

    for mode in PCA_MODES:
        mode_summary = summary[summary["mode"] == mode].sort_values("k")
        mode_paired = paired[paired["mode"] == mode].sort_values("k")

        within = mode_summary[
            mode_summary["accuracy"] >= full_accuracy - 0.01
        ]
        smallest_within = (
            int(within.iloc[0]["k"]) if not within.empty else None
        )

        indistinguishable = mode_paired[
            mode_paired["mcnemar_exact_p"] >= ALPHA
        ]
        smallest_indist = (
            int(indistinguishable.iloc[0]["k"])
            if not indistinguishable.empty
            else None
        )

        lines.append("")
        lines.append(f"Mode: {mode}")

        if smallest_within is None:
            lines.append(
                "  No tested k reaches within 1 percentage point of the "
                "full-feature accuracy."
            )
        else:
            acc = float(
                mode_summary.loc[
                    mode_summary["k"] == smallest_within, "accuracy"
                ].iloc[0]
            )
            lines.append(
                f"  Smallest k within 1 percentage point of full: "
                f"k={smallest_within}, accuracy={acc:.4f}."
            )

        if smallest_indist is None:
            lines.append(
                "  Every tested k differs from full-feature correctness at "
                "p<0.05 under exact McNemar testing."
            )
        else:
            p = float(
                mode_paired.loc[
                    mode_paired["k"] == smallest_indist,
                    "mcnemar_exact_p",
                ].iloc[0]
            )
            lines.append(
                f"  Smallest k not significantly different from full under "
                f"paired exact McNemar testing: k={smallest_indist}, "
                f"p={p:.4g}. This means failure to detect a paired "
                f"difference, not proof of equivalence."
            )

        predictive_k = smallest_within
        if predictive_k is None:
            predictive_k = smallest_indist

        if predictive_k is not None:
            fraction, source = reconstruction_at_k(predictive_k)
            if np.isfinite(fraction):
                lines.append(
                    f"  Held-out response energy reconstructed at that "
                    f"predictive k: {fraction:.4f} ({fraction:.1%}; {source})."
                )
                if fraction < 0.25:
                    lines.append(
                        "  Decoding therefore plateaus far earlier than "
                        "held-out response reconstruction: strong prediction "
                        "is possible while most response energy remains "
                        "outside the low-rank predictive subspace."
                    )
                else:
                    lines.append(
                        "  Compare this reconstruction fraction with the "
                        "accuracy curve before claiming an early plateau."
                    )
            else:
                lines.append(
                    f"  Reconstruction energy at k={predictive_k} is not "
                    "available from the supplied values."
                )
        else:
            lines.append(
                "  No predictive k was identified, so reconstruction energy "
                "at a predictive plateau cannot be reported."
            )

    return "\n".join(lines)


# =============================================================================
# Plotting
# =============================================================================

def plot_accuracy_and_auc(
    summary: pd.DataFrame,
    full_summary: dict,
):
    full_accuracy = float(full_summary["accuracy"])
    full_auc = float(full_summary["roc_auc"])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    for mode in PCA_MODES:
        table = summary[summary["mode"] == mode].sort_values("k")
        x = np.arange(len(table))
        accuracy = table["accuracy"].to_numpy()
        lower = accuracy - table["ci_low"].to_numpy()
        upper = table["ci_high"].to_numpy() - accuracy

        axes[0].errorbar(
            x,
            accuracy,
            yerr=np.vstack([lower, upper]),
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=mode,
        )

        auc = table["roc_auc"].to_numpy()
        auc_lower = auc - table["auc_ci_low"].to_numpy()
        auc_upper = table["auc_ci_high"].to_numpy() - auc

        axes[1].errorbar(
            x,
            auc,
            yerr=np.vstack([auc_lower, auc_upper]),
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=mode,
        )

    tick_positions = np.arange(len(K_VALUES))
    tick_labels = [str(k) for k in K_VALUES]

    axes[0].axhline(
        full_accuracy,
        linestyle="--",
        linewidth=1.5,
        label=f"full feature ({full_accuracy:.3f})",
    )
    axes[0].axhline(
        0.5,
        linestyle=":",
        linewidth=1.2,
        label="chance",
    )
    axes[0].set(
        xlabel="Number of principal components",
        ylabel="LOO accuracy",
        title="Animacy decoding accuracy",
        ylim=(0.35, 1.0),
    )
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(tick_labels, rotation=45)
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    axes[1].axhline(
        full_auc,
        linestyle="--",
        linewidth=1.5,
        label=f"full feature ({full_auc:.3f})",
    )
    axes[1].axhline(
        0.5,
        linestyle=":",
        linewidth=1.2,
        label="chance",
    )
    axes[1].set(
        xlabel="Number of principal components",
        ylabel="LOO ROC AUC",
        title="Animacy decoding ROC AUC",
        ylim=(0.35, 1.0),
    )
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels, rotation=45)
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)

    fig.tight_layout()

    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTDIR / f"loo_pca_animacy_accuracy_auc.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_accuracy_vs_reconstruction(
    summary: pd.DataFrame,
):
    available_k = [
        k for k in K_VALUES if k in RECONSTRUCTION_FRACTION
    ]
    reconstruction = np.asarray(
        [RECONSTRUCTION_FRACTION[k] for k in available_k],
        dtype=float,
    )

    fig, ax1 = plt.subplots(figsize=(9.5, 5.8))
    x = np.arange(len(available_k))

    for mode in PCA_MODES:
        table = (
            summary[
                (summary["mode"] == mode)
                & (summary["k"].isin(available_k))
            ]
            .sort_values("k")
        )
        ax1.plot(
            x,
            table["accuracy"],
            marker="o",
            linewidth=2,
            label=f"{mode} decoding accuracy",
        )

    ax1.set(
        xlabel="Number of principal components",
        ylabel="LOO decoding accuracy",
        title="Predictive performance versus held-out reconstruction",
        ylim=(0.35, 1.0),
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(available_k)
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        reconstruction,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="held-out response reconstructed",
    )
    ax2.set(
        ylabel="Held-out centered response reconstructed",
        ylim=(0.0, max(0.30, reconstruction.max() + 0.03)),
    )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        frameon=False,
        loc="best",
    )

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTDIR / f"loo_pca_accuracy_vs_reconstruction.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


# =============================================================================
# Reporting
# =============================================================================

def print_summary(
    summary: pd.DataFrame,
    full_summary: dict,
    paired: pd.DataFrame,
):
    for mode in PCA_MODES:
        print()
        print("=" * 80)
        print(f"PCA decoding summary: {mode}")
        print("=" * 80)

        table = summary[summary["mode"] == mode].sort_values("k")
        for _, row in table.iterrows():
            print(
                f"k={int(row['k']):>3}  "
                f"correct={int(row['correct']):>3}/{int(row['n'])}  "
                f"acc={row['accuracy']:.4f}  "
                f"bal_acc={row['balanced_accuracy']:.4f}  "
                f"AUC={row['roc_auc']:.4f}  "
                f"Wilson=[{row['ci_low']:.4f}, {row['ci_high']:.4f}]  "
                f"AUC bootstrap=[{row['auc_ci_low']:.4f}, "
                f"{row['auc_ci_high']:.4f}]  "
                f"CM=[[{int(row['tn'])}, {int(row['fp'])}], "
                f"[{int(row['fn'])}, {int(row['tp'])}]]"
            )

    print()
    print("=" * 80)
    print("Full-feature reference")
    print("=" * 80)
    print(
        f"correct={full_summary['correct']}/{full_summary['n']}  "
        f"acc={full_summary['accuracy']:.4f}  "
        f"bal_acc={full_summary['balanced_accuracy']:.4f}  "
        f"AUC={full_summary['roc_auc']:.4f}  "
        f"Wilson=[{full_summary['ci_low']:.4f}, "
        f"{full_summary['ci_high']:.4f}]  "
        f"source={full_summary['reference_source']}"
    )

    print()
    print("=" * 80)
    print("Paired McNemar comparisons with full feature")
    print("=" * 80)
    for _, row in paired.sort_values(["mode", "k"]).iterrows():
        print(
            f"{row['mode']:<12} k={int(row['k']):>3}: "
            f"PCA+/full-={int(row['pca_correct_full_wrong'])}, "
            f"PCA-/full+={int(row['pca_wrong_full_correct'])}, "
            f"p={row['mcnemar_exact_p']:.4g}"
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
        labels_four_class,
        brain_area,
        cell_specimen_id,
        ophys_experiment_id,
    ) = load_composite_data()

    (
        X,
        labels_four_class,
        y,
        original_image_indices,
        neuron_mask,
        brain_area,
        cell_specimen_id,
        ophys_experiment_id,
    ) = retain_valid_data(
        X,
        labels_four_class,
        brain_area,
        cell_specimen_id,
        ophys_experiment_id,
    )

    if X.shape[0] != 118:
        print(
            f"[WARN] Expected 118 retained images from the prompt, found "
            f"{X.shape[0]}. The analysis will continue with retained ordering."
        )

    if max(K_VALUES) > X.shape[0] - 2:
        raise ValueError(
            f"Maximum k={max(K_VALUES)} exceeds centered LOO training rank "
            f"limit n_train-1={X.shape[0]-2}."
        )

    pca_predictions, fold_diagnostics = run_loo_pca_decoding(
        X=X,
        y=y,
        original_image_indices=original_image_indices,
    )

    full_predictions = load_original_full_feature_predictions(
        FULL_FEATURE_RESULTS_PATH,
        y=y,
        original_image_indices=original_image_indices,
    )

    if full_predictions is None:
        full_predictions = compute_fallback_full_feature_predictions(
            X=X,
            y=y,
            original_image_indices=original_image_indices,
        )

    # Preserve exact fold identity in one raw file.
    pca_raw = pca_predictions.copy()
    pca_raw["model"] = (
        pca_raw["mode"] + "_pca_k" + pca_raw["k"].astype(str)
    )

    full_raw = full_predictions.copy()
    full_raw["mode"] = "full"
    full_raw["k"] = "full"
    full_raw["model"] = "full_feature"

    raw_columns = [
        "model",
        "mode",
        "k",
        "fold_index",
        "original_image_index",
        "true_label",
        "predicted_class",
        "animate_probability",
        "decision_score",
        "correct",
    ]
    raw_combined = pd.concat(
        [
            pca_raw[raw_columns],
            full_raw[raw_columns],
        ],
        ignore_index=True,
    )
    raw_combined.to_csv(RAW_PREDICTIONS_CSV, index=False)

    summary, confusion_table = summarize_predictions(
        pca_predictions,
        rng,
    )
    full_summary = summarize_full_reference(full_predictions, rng)

    # Results table with one "full" row, as requested. There are two PCA modes,
    # so mode identifies which geometry each k belongs to.
    full_summary_row = {
        key: full_summary.get(key, np.nan)
        for key in summary.columns
    }
    full_summary_row["mode"] = "full"
    full_summary_row["k"] = "full"

    summary_with_full = pd.concat(
        [summary, pd.DataFrame([full_summary_row])],
        ignore_index=True,
    )
    summary_with_full.to_csv(SUMMARY_CSV, index=False)

    confusion_table.to_csv(CONFUSIONS_CSV, index=False)

    paired = paired_comparisons(
        pca_predictions=pca_predictions,
        full_predictions=full_predictions,
    )
    paired.to_csv(PAIRED_CSV, index=False)

    incremental = incremental_gain_table(
        summary=summary,
        full_accuracy=float(full_summary["accuracy"]),
    )
    incremental.to_csv(INCREMENTAL_CSV, index=False)

    plot_accuracy_and_auc(summary, full_summary)
    plot_accuracy_vs_reconstruction(summary)

    interpretation = build_interpretation(
        summary=summary,
        paired=paired,
        full_summary=full_summary,
    )
    INTERPRETATION_TXT.write_text(interpretation + "\n")

    # Compact binary archive in addition to the CSVs.
    np.savez_compressed(
        RESULTS_NPZ,
        X_shape=np.asarray(X.shape, dtype=np.int64),
        labels_four_class=labels_four_class,
        y=y,
        original_image_indices=original_image_indices,
        neuron_mask=neuron_mask,
        k_values=np.asarray(K_VALUES, dtype=np.int64),
        pca_modes=np.asarray(PCA_MODES, dtype=object),
        fold_predictions=pca_predictions.to_records(index=False),
        full_predictions=full_predictions.to_records(index=False),
        summary=summary_with_full.to_records(index=False),
        paired_mcnemar=paired.to_records(index=False),
        incremental_gain=incremental.to_records(index=False),
        fold_diagnostics=fold_diagnostics.to_records(index=False),
        reconstruction_k=np.asarray(
            sorted(RECONSTRUCTION_FRACTION), dtype=np.int64
        ),
        reconstruction_fraction=np.asarray(
            [
                RECONSTRUCTION_FRACTION[k]
                for k in sorted(RECONSTRUCTION_FRACTION)
            ],
            dtype=np.float64,
        ),
        logistic_kwargs_json=json.dumps(LOGISTIC_KWARGS),
        random_seed=np.int64(RANDOM_SEED),
        n_auc_bootstrap=np.int64(N_AUC_BOOTSTRAP),
        full_reference_source=np.asarray(
            full_summary["reference_source"], dtype=object
        ),
    )

    print_summary(summary, full_summary, paired)

    print()
    print(interpretation)

    print()
    print("=" * 80)
    print("Saved")
    print("=" * 80)
    for path in (
        RAW_PREDICTIONS_CSV,
        SUMMARY_CSV,
        INCREMENTAL_CSV,
        PAIRED_CSV,
        CONFUSIONS_CSV,
        RESULTS_NPZ,
        INTERPRETATION_TXT,
        OUTDIR / "loo_pca_animacy_accuracy_auc.png",
        OUTDIR / "loo_pca_accuracy_vs_reconstruction.png",
    ):
        print(path)


if __name__ == "__main__":
    main()
