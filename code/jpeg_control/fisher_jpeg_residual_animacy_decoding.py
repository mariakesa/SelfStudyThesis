#!/usr/bin/env python3
"""
Leakage-safe comparison of raw vs JPEG-residual neural animacy decoding,
with and without Fisher-subspace selection.

For every held-out image, this script does ALL learned operations on the
training images only.

RAW branch
----------
1. Fit StandardScaler on X_train.
2. Raw baseline: fit logistic regression directly on standardized X_train.
3. Fisher branch:
   a. Fit logistic regression on standardized X_train.
   b. Form Fisher geometry implicitly with
          F = X.T W X / n,   W_ii = p_i (1-p_i)
      via the thin SVD of
          B = sqrt(W/n) X.
   c. Keep top-k Fisher eigenvectors.
   d. Project train and held-out image into that fold-specific Fisher basis.
   e. Fit a new logistic decoder in the reduced space.

JPEG-RESIDUAL branch
--------------------
1. Fit JPEG StandardScaler on jpeg_train only.
2. Fit JPEG PCA on jpeg_train only.
3. Regress neural activity on [intercept + JPEG PCs] using X_train only.
4. Residualize both X_train and X_test with that training-fitted nuisance model.
5. Fit a neural StandardScaler on residual_train only.
6. Residual baseline: decode directly from standardized residual activity.
7. Residual Fisher branch:
   a. Learn Fisher basis from standardized residual_train only.
   b. Project residual_train and residual_test into that basis.
   c. Fit a new reduced-space logistic decoder.
   d. Predict the held-out image.

Thus the held-out image NEVER contributes to:
- JPEG scaling
- JPEG PCA
- nuisance regression coefficients
- neural scaling
- Fisher weights/eigenvectors
- decoder fitting

The key comparison is:

    raw baseline
    raw Fisher(k)
    JPEG-residual baseline
    JPEG-residual Fisher(k)

If JPEG-residual activity still decodes animacy above chance, semantic
information remains after removing the JPEG-complexity component.

Expected neural file:
    allen_natural_scenes_four_class_composite.npy

Expected dict fields:
    data["X"]
    data["stimulus_metadata"]["label"]

Four-class label convention:
    0 = animals
    1 = landscape
    2 = plant
    3 = man-made object
   -1 = unlabeled

Binary target:
    animal = 1
    everything else = 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_NEURAL_PATH = Path(
    "/home/maria/SelfStudyThesis/data/"
    "allen_natural_scenes_four_class_composite.npy"
)

DEFAULT_JPEG_PATH = Path(
    "/home/maria/SelfStudyThesis/results/"
    "loo_jpeg_residual_pca_animacy_decoding/"
    "sizes_bytes.npy"
)

DEFAULT_OUTDIR = Path(
    "/home/maria/SelfStudyThesis/results/"
    "fisher_raw_vs_jpeg_residual"
)

DEFAULT_K_VALUES = [
    1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 80, 100
]

RANDOM_STATE = 123
MAX_ITER = 5000
C_FISHER_MODEL = 1.0
C_REDUCED_DECODER = 1.0
VAR_EPS = 1e-12


# =============================================================================
# Utilities
# =============================================================================

def parse_k_values(text: str) -> list[int]:
    values = sorted({int(v.strip()) for v in text.split(",") if v.strip()})
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError(
            "k values must be comma-separated positive integers"
        )
    return values


def load_numeric_array(path: Path) -> np.ndarray:
    """Load a numeric JPEG feature array from .npy or .npz."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npy":
        obj = np.load(path, allow_pickle=True)

        if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
            obj = obj.item()
            if isinstance(obj, dict):
                for key in ("sizes_bytes", "sizes_kb", "jpeg", "features", "X"):
                    if key in obj:
                        return np.asarray(obj[key])
                raise ValueError(
                    f"{path} is a dictionary but contains no recognized array key"
                )

        return np.asarray(obj)

    if path.suffix == ".npz":
        archive = np.load(path, allow_pickle=True)

        for key in ("sizes_bytes", "sizes_kb", "jpeg", "features", "X"):
            if key in archive.files:
                return np.asarray(archive[key])

        if len(archive.files) == 1:
            return np.asarray(archive[archive.files[0]])

        raise ValueError(
            f"{path} contains multiple arrays and no recognized JPEG key"
        )

    raise ValueError("JPEG features must be stored in .npy or .npz")


def orient_rows(array: np.ndarray, n_rows: int, name: str) -> np.ndarray:
    """Ensure samples/images are rows."""
    array = np.asarray(array)

    if array.ndim == 1:
        if len(array) != n_rows:
            raise ValueError(
                f"{name} length {len(array)} does not match {n_rows} rows"
            )
        return array[:, None]

    if array.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D; got {array.shape}")

    if array.shape[0] == n_rows:
        return array

    if array.shape[1] == n_rows:
        return array.T

    raise ValueError(
        f"Cannot orient {name} with shape {array.shape} to {n_rows} rows"
    )


def make_classifier(C: float) -> LogisticRegression:
    return LogisticRegression(
        penalty="l2",
        C=C,
        solver="liblinear",
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


# =============================================================================
# Data
# =============================================================================

def load_data(neural_path: Path, jpeg_path: Path):
    data = np.load(neural_path, allow_pickle=True).item()

    X = np.asarray(data["X"], dtype=np.float64)
    labels4 = np.asarray(
        data["stimulus_metadata"]["label"],
        dtype=np.int64,
    ).ravel()

    # Orient X as images x neurons.
    if X.shape[0] == len(labels4):
        pass
    elif X.shape[1] == len(labels4):
        X = X.T
    else:
        raise ValueError(
            f"Cannot align X={X.shape} with labels={labels4.shape}"
        )

    n_original_images = len(labels4)

    # JPEG features must initially correspond to the original image ordering.
    jpeg_all = orient_rows(
        load_numeric_array(jpeg_path),
        n_rows=n_original_images,
        name="JPEG feature matrix",
    ).astype(np.float64)

    # Remove unlabeled images identically from both modalities.
    labeled = labels4 != -1
    original_indices = np.flatnonzero(labeled)

    X = X[labeled]
    labels4 = labels4[labeled]
    jpeg = jpeg_all[labeled]

    y = (labels4 == 0).astype(np.int64)

    # Keep only globally finite/nontrivial neural features.
    # Standardization itself remains fold-local.
    finite = np.all(np.isfinite(X), axis=0)
    nonconstant = np.var(X, axis=0) > VAR_EPS
    neuron_mask = finite & nonconstant

    if not np.any(neuron_mask):
        raise ValueError("No finite, nonconstant neurons remain")

    X = X[:, neuron_mask]

    if not np.all(np.isfinite(jpeg)):
        raise ValueError("JPEG features contain NaN or infinite values")

    return X, jpeg, y, labels4, original_indices, neuron_mask


# =============================================================================
# JPEG nuisance model
# =============================================================================

def fit_jpeg_coordinates(
    jpeg_train: np.ndarray,
    jpeg_test: np.ndarray,
    n_components: int,
):
    """
    Learn nuisance coordinates from TRAINING JPEG features only.
    """
    scaler = StandardScaler()
    J_train_z = scaler.fit_transform(jpeg_train)
    J_test_z = scaler.transform(jpeg_test)

    max_components = min(
        int(n_components),
        J_train_z.shape[1],
        J_train_z.shape[0] - 1,
    )
    if max_components < 1:
        raise ValueError("Cannot fit any JPEG nuisance component")

    # If there is only one JPEG feature, PCA adds nothing useful.
    if J_train_z.shape[1] == 1:
        return J_train_z[:, :1], J_test_z[:, :1], 1.0

    pca = PCA(
        n_components=max_components,
        svd_solver="full",
    )
    J_train_pc = pca.fit_transform(J_train_z)
    J_test_pc = pca.transform(J_test_z)

    return (
        J_train_pc,
        J_test_pc,
        float(pca.explained_variance_ratio_.sum()),
    )


def residualize_neural_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    nuisance_train: np.ndarray,
    nuisance_test: np.ndarray,
):
    """
    Fit X ~ intercept + nuisance on TRAINING images only,
    then residualize train and held-out image with the same coefficients.
    """
    D_train = np.column_stack(
        [np.ones(nuisance_train.shape[0]), nuisance_train]
    )
    D_test = np.column_stack(
        [np.ones(nuisance_test.shape[0]), nuisance_test]
    )

    beta, *_ = np.linalg.lstsq(
        D_train,
        X_train,
        rcond=None,
    )

    X_train_hat = D_train @ beta
    X_test_hat = D_test @ beta

    R_train = X_train - X_train_hat
    R_test = X_test - X_test_hat

    return R_train, R_test, X_test_hat


# =============================================================================
# Fisher geometry
# =============================================================================

def compute_fisher_basis(
    X_train_z: np.ndarray,
    y_train: np.ndarray,
):
    """
    Compute eigenvectors of the logistic Fisher matrix in feature space.

        F = X.T W X / n

    without ever forming the huge feature x feature matrix.

    If
        B = sqrt(W/n) X = U S V.T
    then
        F = V S^2 V.T.
    """
    fisher_model = make_classifier(C_FISHER_MODEL)
    fisher_model.fit(X_train_z, y_train)

    p = fisher_model.predict_proba(X_train_z)[:, 1]
    w = p * (1.0 - p)

    n = X_train_z.shape[0]
    B = np.sqrt(w / n)[:, None] * X_train_z

    _, s, Vt = np.linalg.svd(
        B,
        full_matrices=False,
    )

    eigenvalues = s ** 2
    V = Vt.T

    return V, eigenvalues


def direct_decode(
    X_train_z: np.ndarray,
    X_test_z: np.ndarray,
    y_train: np.ndarray,
):
    clf = make_classifier(C_REDUCED_DECODER)
    clf.fit(X_train_z, y_train)

    score = float(clf.predict_proba(X_test_z)[0, 1])
    pred = int(score >= 0.5)

    return pred, score


def fisher_decode_all_k(
    X_train_z: np.ndarray,
    X_test_z: np.ndarray,
    y_train: np.ndarray,
    k_values: list[int],
):
    V, eigenvalues = compute_fisher_basis(
        X_train_z,
        y_train,
    )

    results = {}
    available_rank = V.shape[1]

    for k in k_values:
        k_eff = min(k, available_rank)

        Vk = V[:, :k_eff]

        Z_train = X_train_z @ Vk
        Z_test = X_test_z @ Vk

        clf = make_classifier(C_REDUCED_DECODER)
        clf.fit(Z_train, y_train)

        score = float(clf.predict_proba(Z_test)[0, 1])
        pred = int(score >= 0.5)

        results[k] = (pred, score)

    return results, eigenvalues


# =============================================================================
# LOO experiment
# =============================================================================

def initialize_prediction_store(
    branches: list[str],
    k_values: list[int],
    n_images: int,
):
    store = {}

    for branch in branches:
        store[branch] = {}

        if branch.endswith("_baseline"):
            store[branch]["pred"] = np.full(n_images, -1, dtype=np.int64)
            store[branch]["score"] = np.full(n_images, np.nan, dtype=np.float64)
        else:
            for k in k_values:
                store[branch][k] = {
                    "pred": np.full(n_images, -1, dtype=np.int64),
                    "score": np.full(n_images, np.nan, dtype=np.float64),
                }

    return store


def run_loo(
    X: np.ndarray,
    jpeg: np.ndarray,
    y: np.ndarray,
    k_values: list[int],
    n_jpeg_components: int,
):
    n_images = X.shape[0]
    indices = np.arange(n_images)

    branches = [
        "raw_baseline",
        "raw_fisher",
        "residual_baseline",
        "residual_fisher",
    ]
    store = initialize_prediction_store(
        branches=branches,
        k_values=k_values,
        n_images=n_images,
    )

    raw_fisher_spectra = []
    residual_fisher_spectra = []

    # Every row here is generated from a nuisance model that did NOT train on it.
    cross_fitted_residuals = np.full_like(X, np.nan, dtype=np.float64)
    cross_fitted_jpeg_prediction = np.full_like(X, np.nan, dtype=np.float64)
    jpeg_evr = np.full(n_images, np.nan, dtype=np.float64)

    print()
    print("=" * 88)
    print("LOO: RAW vs JPEG-RESIDUAL, with Fisher subspaces")
    print("=" * 88)

    for test_i in indices:
        train_mask = indices != test_i

        X_train = X[train_mask]
        X_test = X[[test_i]]
        J_train = jpeg[train_mask]
        J_test = jpeg[[test_i]]
        y_train = y[train_mask]

        # ---------------------------------------------------------------------
        # RAW branch
        # ---------------------------------------------------------------------
        raw_scaler = StandardScaler()
        X_train_z = raw_scaler.fit_transform(X_train)
        X_test_z = raw_scaler.transform(X_test)

        pred, score = direct_decode(
            X_train_z,
            X_test_z,
            y_train,
        )
        store["raw_baseline"]["pred"][test_i] = pred
        store["raw_baseline"]["score"][test_i] = score

        raw_fisher_results, raw_eigs = fisher_decode_all_k(
            X_train_z,
            X_test_z,
            y_train,
            k_values,
        )
        raw_fisher_spectra.append(raw_eigs)

        for k, (pred, score) in raw_fisher_results.items():
            store["raw_fisher"][k]["pred"][test_i] = pred
            store["raw_fisher"][k]["score"][test_i] = score

        # ---------------------------------------------------------------------
        # JPEG residualization -- ALL FITTING INSIDE THIS FOLD
        # ---------------------------------------------------------------------
        J_train_pc, J_test_pc, evr = fit_jpeg_coordinates(
            J_train,
            J_test,
            n_components=n_jpeg_components,
        )
        jpeg_evr[test_i] = evr

        R_train, R_test, X_test_hat = residualize_neural_fold(
            X_train,
            X_test,
            J_train_pc,
            J_test_pc,
        )

        cross_fitted_residuals[test_i] = R_test[0]
        cross_fitted_jpeg_prediction[test_i] = X_test_hat[0]

        # ---------------------------------------------------------------------
        # RESIDUAL branch
        # ---------------------------------------------------------------------
        residual_scaler = StandardScaler()
        R_train_z = residual_scaler.fit_transform(R_train)
        R_test_z = residual_scaler.transform(R_test)

        pred, score = direct_decode(
            R_train_z,
            R_test_z,
            y_train,
        )
        store["residual_baseline"]["pred"][test_i] = pred
        store["residual_baseline"]["score"][test_i] = score

        residual_fisher_results, residual_eigs = fisher_decode_all_k(
            R_train_z,
            R_test_z,
            y_train,
            k_values,
        )
        residual_fisher_spectra.append(residual_eigs)

        for k, (pred, score) in residual_fisher_results.items():
            store["residual_fisher"][k]["pred"][test_i] = pred
            store["residual_fisher"][k]["score"][test_i] = score

        if (
            test_i == 0
            or (test_i + 1) % 5 == 0
            or test_i == n_images - 1
        ):
            print(
                f"\rFold {test_i + 1:>3}/{n_images}",
                end="",
                flush=True,
            )

    print()

    return {
        "store": store,
        "raw_fisher_spectra": raw_fisher_spectra,
        "residual_fisher_spectra": residual_fisher_spectra,
        "cross_fitted_residuals": cross_fitted_residuals,
        "cross_fitted_jpeg_prediction": cross_fitted_jpeg_prediction,
        "jpeg_evr": jpeg_evr,
    }


# =============================================================================
# Metrics
# =============================================================================

def metric_row(
    representation: str,
    y: np.ndarray,
    pred: np.ndarray,
    score: np.ndarray,
    k: int | None = None,
):
    if np.any(pred < 0) or np.any(~np.isfinite(score)):
        raise RuntimeError(f"Missing predictions in {representation}, k={k}")

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    return {
        "representation": representation,
        "k": np.nan if k is None else int(k),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": safe_auc(y, score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_correct": int(np.sum(pred == y)),
        "n_images": int(len(y)),
    }


def build_metrics(
    y: np.ndarray,
    store: dict,
    k_values: list[int],
):
    rows = []

    for branch in ("raw_baseline", "residual_baseline"):
        rows.append(
            metric_row(
                representation=branch,
                y=y,
                pred=store[branch]["pred"],
                score=store[branch]["score"],
                k=None,
            )
        )

    for branch in ("raw_fisher", "residual_fisher"):
        for k in k_values:
            rows.append(
                metric_row(
                    representation=branch,
                    y=y,
                    pred=store[branch][k]["pred"],
                    score=store[branch][k]["score"],
                    k=k,
                )
            )

    return pd.DataFrame(rows)


def best_fisher_row(metrics: pd.DataFrame, branch: str) -> pd.Series:
    sub = metrics[metrics["representation"] == branch].copy()

    return sub.sort_values(
        ["balanced_accuracy", "roc_auc", "k"],
        ascending=[False, False, True],
    ).iloc[0]


# =============================================================================
# Reporting
# =============================================================================

def print_summary(metrics: pd.DataFrame):
    raw_base = metrics[
        metrics["representation"] == "raw_baseline"
    ].iloc[0]

    residual_base = metrics[
        metrics["representation"] == "residual_baseline"
    ].iloc[0]

    raw_best = best_fisher_row(metrics, "raw_fisher")
    residual_best = best_fisher_row(metrics, "residual_fisher")

    print()
    print("=" * 88)
    print("HEADLINE RESULTS")
    print("=" * 88)
    print(
        f"{'Representation':<30}"
        f"{'k':>6}"
        f"{'Accuracy':>12}"
        f"{'Bal.Acc':>12}"
        f"{'AUC':>12}"
    )
    print("-" * 72)

    def show(label, row):
        k_text = "-" if np.isnan(row["k"]) else str(int(row["k"]))
        print(
            f"{label:<30}"
            f"{k_text:>6}"
            f"{row['accuracy']:>12.4f}"
            f"{row['balanced_accuracy']:>12.4f}"
            f"{row['roc_auc']:>12.4f}"
        )

    show("Raw baseline", raw_base)
    show("Best raw Fisher", raw_best)
    show("JPEG-residual baseline", residual_base)
    show("Best residual Fisher", residual_best)

    print()
    print("Primary scientific question:")
    print(
        "  Do the JPEG-residual neural responses still decode animacy?"
    )
    print(
        f"  Residual baseline balanced accuracy = "
        f"{residual_base['balanced_accuracy']:.4f}, "
        f"AUC = {residual_base['roc_auc']:.4f}"
    )
    print(
        f"  Best residual Fisher balanced accuracy = "
        f"{residual_best['balanced_accuracy']:.4f}, "
        f"AUC = {residual_best['roc_auc']:.4f} "
        f"(k={int(residual_best['k'])})"
    )
    print()
    print(
        "Interpretation note: above-chance observed LOO performance suggests "
        "decodable animacy remains after the fold-local JPEG nuisance model. "
        "Use a permutation test / confidence interval before making a formal "
        "statistical claim."
    )


def pad_spectra(spectra_list: list[np.ndarray]) -> np.ndarray:
    max_len = max(len(x) for x in spectra_list)
    out = np.full((len(spectra_list), max_len), np.nan)

    for i, values in enumerate(spectra_list):
        out[i, :len(values)] = values

    return out


def plot_comparison(
    metrics: pd.DataFrame,
    metric: str,
    ylabel: str,
    outpath: Path,
):
    fig, ax = plt.subplots(figsize=(9.5, 6.0))

    raw = metrics[metrics["representation"] == "raw_fisher"].sort_values("k")
    residual = metrics[
        metrics["representation"] == "residual_fisher"
    ].sort_values("k")

    raw_base = metrics[
        metrics["representation"] == "raw_baseline"
    ].iloc[0][metric]

    residual_base = metrics[
        metrics["representation"] == "residual_baseline"
    ].iloc[0][metric]

    ax.plot(
        raw["k"],
        raw[metric],
        marker="o",
        label="Raw Fisher",
    )
    ax.plot(
        residual["k"],
        residual[metric],
        marker="o",
        label="JPEG-residual Fisher",
    )

    ax.axhline(
        raw_base,
        linestyle="--",
        linewidth=1.2,
        label="Raw baseline",
    )
    ax.axhline(
        residual_base,
        linestyle=":",
        linewidth=1.5,
        label="JPEG-residual baseline",
    )
    ax.axhline(
        0.5,
        linestyle="-.",
        linewidth=1.0,
        label="Chance reference",
    )

    ax.set_xlabel("Number of retained Fisher directions")
    ax.set_ylabel(ylabel)
    ax.set_title("Raw vs JPEG-residual Fisher-subspace animacy decoding")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    outdir: Path,
    metrics: pd.DataFrame,
    results: dict,
    y: np.ndarray,
    labels4: np.ndarray,
    original_indices: np.ndarray,
    neuron_mask: np.ndarray,
    k_values: list[int],
    n_jpeg_components: int,
    neural_path: Path,
    jpeg_path: Path,
):
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_path = outdir / "metrics.csv"
    predictions_path = outdir / "predictions.npz"
    residual_path = outdir / "cross_fitted_jpeg_residual_matrix.npy"
    jpeg_prediction_path = outdir / "cross_fitted_jpeg_predicted_neural_matrix.npy"
    summary_path = outdir / "summary.json"

    metrics.to_csv(metrics_path, index=False)

    np.save(
        residual_path,
        results["cross_fitted_residuals"],
    )
    np.save(
        jpeg_prediction_path,
        results["cross_fitted_jpeg_prediction"],
    )

    store = results["store"]

    payload = {
        "y_animacy": y,
        "labels_four_class": labels4,
        "original_image_indices": original_indices,
        "neuron_mask": neuron_mask,
        "k_values": np.asarray(k_values, dtype=np.int64),
        "jpeg_pca_evr_by_fold": results["jpeg_evr"],
        "raw_fisher_eigenvalues_by_fold": pad_spectra(
            results["raw_fisher_spectra"]
        ),
        "residual_fisher_eigenvalues_by_fold": pad_spectra(
            results["residual_fisher_spectra"]
        ),
        "raw_baseline_pred": store["raw_baseline"]["pred"],
        "raw_baseline_score": store["raw_baseline"]["score"],
        "residual_baseline_pred": store["residual_baseline"]["pred"],
        "residual_baseline_score": store["residual_baseline"]["score"],
    }

    for k in k_values:
        payload[f"raw_fisher_pred_k{k}"] = store["raw_fisher"][k]["pred"]
        payload[f"raw_fisher_score_k{k}"] = store["raw_fisher"][k]["score"]
        payload[f"residual_fisher_pred_k{k}"] = store["residual_fisher"][k]["pred"]
        payload[f"residual_fisher_score_k{k}"] = store["residual_fisher"][k]["score"]

    np.savez_compressed(predictions_path, **payload)

    raw_base = metrics[
        metrics["representation"] == "raw_baseline"
    ].iloc[0]
    residual_base = metrics[
        metrics["representation"] == "residual_baseline"
    ].iloc[0]
    raw_best = best_fisher_row(metrics, "raw_fisher")
    residual_best = best_fisher_row(metrics, "residual_fisher")

    def serializable_row(row):
        out = {}
        for key, value in row.to_dict().items():
            if isinstance(value, (np.integer,)):
                out[key] = int(value)
            elif isinstance(value, (np.floating,)):
                out[key] = None if np.isnan(value) else float(value)
            else:
                out[key] = value
        return out

    summary = {
        "method": (
            "LOO raw decoding and LOO JPEG-residual decoding, "
            "each with fold-local Fisher-subspace selection"
        ),
        "neural_path": str(neural_path),
        "jpeg_path": str(jpeg_path),
        "n_images": int(len(y)),
        "n_retained_neurons": int(np.sum(neuron_mask)),
        "n_jpeg_components_removed": int(n_jpeg_components),
        "k_values": [int(k) for k in k_values],
        "raw_baseline": serializable_row(raw_base),
        "best_raw_fisher_observed": serializable_row(raw_best),
        "residual_baseline": serializable_row(residual_base),
        "best_residual_fisher_observed": serializable_row(residual_best),
        "mean_jpeg_pc_evr": float(np.mean(results["jpeg_evr"])),
        "selection_warning": (
            "The best k is selected after inspection of this dataset. "
            "All individual-k LOO predictions are leakage-safe, but the maximum "
            "over k is not an independently validated hyperparameter estimate."
        ),
        "artifacts": {
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
            "cross_fitted_residuals": str(residual_path),
            "cross_fitted_jpeg_prediction": str(jpeg_prediction_path),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    plot_comparison(
        metrics,
        metric="balanced_accuracy",
        ylabel="LOO balanced accuracy",
        outpath=outdir / "balanced_accuracy_comparison.png",
    )
    plot_comparison(
        metrics,
        metric="roc_auc",
        ylabel="LOO ROC AUC",
        outpath=outdir / "auc_comparison.png",
    )

    return [
        metrics_path,
        predictions_path,
        residual_path,
        jpeg_prediction_path,
        summary_path,
        outdir / "balanced_accuracy_comparison.png",
        outdir / "auc_comparison.png",
    ]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw and JPEG-residual animacy decoding with "
            "fold-local Fisher subspaces."
        )
    )

    parser.add_argument(
        "--neural-path",
        type=Path,
        default=DEFAULT_NEURAL_PATH,
    )
    parser.add_argument(
        "--jpeg-path",
        type=Path,
        default=DEFAULT_JPEG_PATH,
        help=(
            "JPEG feature matrix in original image order. "
            "Can be 118x1 or 118xn_features."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
    )
    parser.add_argument(
        "--k-values",
        type=parse_k_values,
        default=DEFAULT_K_VALUES,
        help="Comma-separated Fisher subspace dimensions",
    )
    parser.add_argument(
        "--n-jpeg-components",
        type=int,
        default=1,
        help="Number of fold-fitted JPEG PCs to regress out",
    )

    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 88)
    print("Loading data")
    print("=" * 88)

    X, jpeg, y, labels4, original_indices, neuron_mask = load_data(
        args.neural_path,
        args.jpeg_path,
    )

    # In LOO, training-set size is n-1. After centering, maximal rank <= n-2.
    max_valid_k = min(
        X.shape[0] - 2,
        X.shape[1],
    )
    k_values = [
        k for k in args.k_values
        if k <= max_valid_k
    ]

    if not k_values:
        raise ValueError(
            f"No requested Fisher k is valid; LOO maximum is {max_valid_k}"
        )

    n_jpeg_components = max(
        1,
        min(
            int(args.n_jpeg_components),
            jpeg.shape[1],
            X.shape[0] - 2,
        ),
    )

    print(f"Neural matrix:             {X.shape}")
    print(f"JPEG matrix:               {jpeg.shape}")
    print(f"Animals:                   {int(np.sum(y == 1))}")
    print(f"Non-animals:               {int(np.sum(y == 0))}")
    print(f"JPEG PCs removed:          {n_jpeg_components}")
    print(f"Fisher k values:           {k_values}")

    results = run_loo(
        X=X,
        jpeg=jpeg,
        y=y,
        k_values=k_values,
        n_jpeg_components=n_jpeg_components,
    )

    metrics = build_metrics(
        y=y,
        store=results["store"],
        k_values=k_values,
    )

    print_summary(metrics)

    saved = save_outputs(
        outdir=args.outdir,
        metrics=metrics,
        results=results,
        y=y,
        labels4=labels4,
        original_indices=original_indices,
        neuron_mask=neuron_mask,
        k_values=k_values,
        n_jpeg_components=n_jpeg_components,
        neural_path=args.neural_path,
        jpeg_path=args.jpeg_path,
    )

    print()
    print("=" * 88)
    print("Saved")
    print("=" * 88)
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
