#!/usr/bin/env python3

"""
Cross-validated Fisher-subspace decoding.

For every held-out image:

    1. Fit preprocessing ONLY on the training images.
    2. Fit logistic regression on the training images.
    3. Compute the training-set Fisher matrix implicitly:

           F = X.T W X / n

       where W_ii = p_i(1-p_i).

    4. Recover its eigenvectors using the thin SVD of

           B = sqrt(W/n) X.

       If

           B = U S V.T

       then

           F = V S^2 V.T.

       Therefore:
           Fisher eigenvectors = columns of V
           Fisher eigenvalues  = S^2

    5. Keep the top-k Fisher directions.
    6. Project train and held-out data into that Fisher subspace.
    7. Fit a new logistic decoder in the reduced space.
    8. Evaluate the held-out image.

This prevents test leakage because the Fisher basis is recomputed
inside each training fold.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)


# =============================================================================
# Paths
# =============================================================================

COMPOSITE_PATH = Path(
    "/home/maria/SelfStudyThesis/data/"
    "allen_natural_scenes_four_class_composite.npy"
)

OUTDIR = Path(
    "/home/maria/SelfStudyThesis/results/"
    "fisher_subspace_selection_animals_vs_rest"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUTDIR / "fisher_subspace_cv_results.npz"
OUT_TXT = OUTDIR / "fisher_subspace_cv_results.txt"
OUT_FIG = OUTDIR / "fisher_subspace_cv_performance.png"


# =============================================================================
# Settings
# =============================================================================

RANDOM_STATE = 123

# Number of leading Fisher eigenvectors to retain.
K_VALUES = [
    1,
    2,
    3,
    5,
    10,
    20,
    30,
    40,
    50,
    60,
    80,
    100,
]

VAR_EPS = 1e-12

# Logistic regularization.
# sklearn minimizes loss + regularization.
C_FISHER_MODEL = 1.0
C_REDUCED_DECODER = 1.0


LABEL_NAMES = {
    -1: "unlabeled",
     0: "animals",
     1: "landscape",
     2: "plant",
     3: "man-made object",
}


# =============================================================================
# Data loading
# =============================================================================

def load_data():
    print("=" * 80)
    print("Loading Allen composite data")
    print("=" * 80)

    data = np.load(COMPOSITE_PATH, allow_pickle=True).item()

    X = np.asarray(data["X"], dtype=np.float64)
    metadata = data["stimulus_metadata"]

    labels = np.asarray(
        metadata["label"],
        dtype=np.int64,
    ).ravel()

    # Orient as images x neurons.
    if X.shape[1] == len(labels):
        X = X.T
    elif X.shape[0] != len(labels):
        raise ValueError(
            f"Cannot align X={X.shape} with labels={labels.shape}"
        )

    # Remove unlabeled images.
    labeled = labels != -1

    X = X[labeled]
    labels = labels[labeled]

    # animals = 1
    # everything else = 0
    y = (labels == 0).astype(np.int64)

    # Remove neurons that are globally invalid.
    #
    # IMPORTANT:
    # We only use this for finite/obviously constant features.
    # Standardization itself happens inside CV.
    finite = np.all(np.isfinite(X), axis=0)
    variances = np.var(X, axis=0)
    nonconstant = variances > VAR_EPS

    keep = finite & nonconstant
    X = X[:, keep]

    print(f"Images:             {X.shape[0]}")
    print(f"Retained neurons:   {X.shape[1]}")
    print(f"Animals:            {np.sum(y == 1)}")
    print(f"Non-animals:        {np.sum(y == 0)}")

    return X, y, keep


# =============================================================================
# Fisher basis
# =============================================================================

def compute_fisher_basis(
    X_train,
    y_train,
):
    """
    Fit logistic regression and compute the Fisher eigenbasis.

    X_train must already be standardized.

    Returns
    -------
    V : array, shape (n_features, rank)
        Fisher eigenvectors in neuron space.

    eigenvalues : array, shape (rank,)
        Fisher eigenvalues, descending.

    probs : array
        Training probabilities used to construct W.
    """

    # -------------------------------------------------------------
    # Fit model defining the local Fisher geometry
    # -------------------------------------------------------------

    model = LogisticRegression(
        penalty="l2",
        C=C_FISHER_MODEL,
        solver="liblinear",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_train)[:, 1]

    # -------------------------------------------------------------
    # Fisher weights
    # -------------------------------------------------------------

    w = probs * (1.0 - probs)

    n = X_train.shape[0]

    # B.T B = Fisher matrix
    #
    # No intercept here:
    # we want Fisher directions purely in NEURON FEATURE SPACE.
    B = np.sqrt(w / n)[:, None] * X_train

    # -------------------------------------------------------------
    # Thin SVD
    #
    # B = U S V.T
    #
    # F = B.T B
    #   = V S^2 V.T
    # -------------------------------------------------------------

    U, s, Vt = np.linalg.svd(
        B,
        full_matrices=False,
    )

    eigenvalues = s ** 2
    V = Vt.T

    return V, eigenvalues, probs


# =============================================================================
# Baseline decoder
# =============================================================================

def evaluate_raw_baseline(X, y):
    loo = LeaveOneOut()

    y_true = []
    y_pred = []
    y_score = []

    for train_idx, test_idx in loo.split(X):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        scaler = StandardScaler()

        X_train_z = scaler.fit_transform(X_train)
        X_test_z = scaler.transform(X_test)

        clf = LogisticRegression(
            penalty="l2",
            C=C_REDUCED_DECODER,
            solver="liblinear",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )

        clf.fit(X_train_z, y_train)

        prob = clf.predict_proba(X_test_z)[0, 1]
        pred = int(prob >= 0.5)

        y_true.append(int(y_test[0]))
        y_pred.append(pred)
        y_score.append(prob)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "auc": roc_auc_score(
            y_true,
            y_score,
        ),
    }


# =============================================================================
# Fisher-subspace CV
# =============================================================================

def evaluate_fisher_subspaces(
    X,
    y,
    k_values,
):
    loo = LeaveOneOut()

    predictions = {
        k: {
            "true": [],
            "pred": [],
            "score": [],
        }
        for k in k_values
    }

    # Store spectra so we can inspect stability later.
    fold_eigenvalues = []

    n_folds = len(y)

    for fold, (train_idx, test_idx) in enumerate(loo.split(X), start=1):

        print(
            f"\rLOO fold {fold:3d}/{n_folds}",
            end="",
            flush=True,
        )

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # ---------------------------------------------------------
        # Standardization INSIDE training fold
        # ---------------------------------------------------------

        scaler = StandardScaler()

        X_train_z = scaler.fit_transform(X_train)
        X_test_z = scaler.transform(X_test)

        # ---------------------------------------------------------
        # Learn Fisher geometry from training data only
        # ---------------------------------------------------------

        V, eigenvalues, probs = compute_fisher_basis(
            X_train_z,
            y_train,
        )

        fold_eigenvalues.append(eigenvalues)

        available_rank = V.shape[1]

        # ---------------------------------------------------------
        # Try different numbers of Fisher directions
        # ---------------------------------------------------------

        for k in k_values:

            k_eff = min(k, available_rank)

            V_k = V[:, :k_eff]

            # Project neural activity into Fisher coordinates.
            Z_train = X_train_z @ V_k
            Z_test = X_test_z @ V_k

            # -----------------------------------------------------
            # New decoder in Fisher subspace
            # -----------------------------------------------------

            clf = LogisticRegression(
                penalty="l2",
                C=C_REDUCED_DECODER,
                solver="liblinear",
                max_iter=5000,
                random_state=RANDOM_STATE,
            )

            clf.fit(Z_train, y_train)

            score = clf.predict_proba(Z_test)[0, 1]
            pred = int(score >= 0.5)

            predictions[k]["true"].append(
                int(y_test[0])
            )

            predictions[k]["pred"].append(
                pred
            )

            predictions[k]["score"].append(
                score
            )

    print()

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

    results = {}

    for k in k_values:

        true = np.asarray(
            predictions[k]["true"]
        )

        pred = np.asarray(
            predictions[k]["pred"]
        )

        score = np.asarray(
            predictions[k]["score"]
        )

        results[k] = {
            "accuracy":
                accuracy_score(true, pred),

            "balanced_accuracy":
                balanced_accuracy_score(
                    true,
                    pred,
                ),

            "auc":
                roc_auc_score(
                    true,
                    score,
                ),
        }

    return results, fold_eigenvalues


# =============================================================================
# Reporting
# =============================================================================

def print_results(
    baseline,
    results,
):
    print("\n" + "=" * 80)
    print("LOO decoding results")
    print("=" * 80)

    print(
        f"{'Representation':<20}"
        f"{'Accuracy':>12}"
        f"{'Bal.Acc':>12}"
        f"{'AUC':>12}"
    )

    print("-" * 56)

    print(
        f"{'raw-zscore':<20}"
        f"{baseline['accuracy']:>12.4f}"
        f"{baseline['balanced_accuracy']:>12.4f}"
        f"{baseline['auc']:>12.4f}"
    )

    for k, m in results.items():

        print(
            f"{'Fisher k=' + str(k):<20}"
            f"{m['accuracy']:>12.4f}"
            f"{m['balanced_accuracy']:>12.4f}"
            f"{m['auc']:>12.4f}"
        )


def save_results(
    baseline,
    results,
    fold_eigenvalues,
):

    # Pad fold spectra to rectangular matrix.
    max_len = max(
        len(x)
        for x in fold_eigenvalues
    )

    spectra = np.full(
        (len(fold_eigenvalues), max_len),
        np.nan,
    )

    for i, vals in enumerate(fold_eigenvalues):
        spectra[i, :len(vals)] = vals

    ks = np.asarray(list(results.keys()))

    acc = np.asarray([
        results[k]["accuracy"]
        for k in ks
    ])

    bal = np.asarray([
        results[k]["balanced_accuracy"]
        for k in ks
    ])

    auc = np.asarray([
        results[k]["auc"]
        for k in ks
    ])

    np.savez_compressed(
        OUT_NPZ,

        k_values=ks,

        accuracy=acc,
        balanced_accuracy=bal,
        auc=auc,

        raw_accuracy=
            baseline["accuracy"],

        raw_balanced_accuracy=
            baseline["balanced_accuracy"],

        raw_auc=
            baseline["auc"],

        fold_fisher_eigenvalues=spectra,
    )

    # -------------------------------------------------------------
    # Text
    # -------------------------------------------------------------

    lines = [
        "FISHER SUBSPACE LOO DECODING",
        "=" * 80,
        "",
        "Representation        Accuracy     Bal.Acc         AUC",
        "-" * 56,
        (
            f"raw-zscore"
            f"{baseline['accuracy']:>17.4f}"
            f"{baseline['balanced_accuracy']:>12.4f}"
            f"{baseline['auc']:>12.4f}"
        ),
    ]

    for k in ks:

        m = results[int(k)]

        lines.append(
            f"Fisher k={k:<10}"
            f"{m['accuracy']:>12.4f}"
            f"{m['balanced_accuracy']:>12.4f}"
            f"{m['auc']:>12.4f}"
        )

    OUT_TXT.write_text(
        "\n".join(lines) + "\n"
    )

    # -------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        ks,
        acc,
        marker="o",
        label="Accuracy",
    )

    ax.plot(
        ks,
        bal,
        marker="o",
        label="Balanced accuracy",
    )

    ax.plot(
        ks,
        auc,
        marker="o",
        label="AUC",
    )

    ax.axhline(
        baseline["accuracy"],
        linestyle="--",
        label="Raw accuracy",
    )

    ax.set_xlabel(
        "Number of retained Fisher directions"
    )

    ax.set_ylabel(
        "Cross-validated score"
    )

    ax.set_title(
        "Animacy decoding after Fisher-subspace selection"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        OUT_FIG,
        dpi=300,
    )

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():

    X, y, feature_mask = load_data()

    print("\nComputing raw baseline...")

    baseline = evaluate_raw_baseline(
        X,
        y,
    )

    print(
        "\nRaw z-scored LOO:"
        f"\n  accuracy = {baseline['accuracy']:.4f}"
        f"\n  bal acc  = {baseline['balanced_accuracy']:.4f}"
        f"\n  AUC      = {baseline['auc']:.4f}"
    )

    print("\nComputing Fisher subspaces...")

    results, fold_eigenvalues = (
        evaluate_fisher_subspaces(
            X,
            y,
            K_VALUES,
        )
    )

    print_results(
        baseline,
        results,
    )

    save_results(
        baseline,
        results,
        fold_eigenvalues,
    )

    print("\nSaved:")
    print(f"  {OUT_NPZ}")
    print(f"  {OUT_TXT}")
    print(f"  {OUT_FIG}")


if __name__ == "__main__":
    main()