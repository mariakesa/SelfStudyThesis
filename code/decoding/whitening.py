#!/usr/bin/env python3
"""
Compare four animals-vs-rest LOO logistic decoders:

    1. Raw neurons
    2. Z-scored neurons
    3. PCA-60
    4. PCA-60 + whitening

Important:
    All preprocessing is fit ONLY on the training set inside each LOO fold.

Input:
    /home/maria/SelfStudyThesis/data/allen_natural_scenes_four_class_composite.npy

Expected:
    data["X"]                             neurons × images
    data["stimulus_metadata"]["label"]   image labels

Labels:
    -1 = unlabeled
     0 = animals
     1 = landscape
     2 = plant
     3 = man-made object

Binary decoding target:
     1 = animal
     0 = all other labeled images
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
)


# =============================================================================
# Paths
# =============================================================================

COMPOSITE_PATH = Path(
    "/home/maria/SelfStudyThesis/data/allen_natural_scenes_four_class_composite.npy"
)

OUTDIR = Path(
    "/home/maria/SelfStudyThesis/results/whitening_animals_vs_rest_loo"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUTDIR / "whitening_animals_vs_rest_loo_results.npz"


# =============================================================================
# Settings
# =============================================================================

LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 3000

PCA_COMPONENTS = 60

RANDOM_SEED = 0
EPS = 1e-12


LABEL_NAMES = {
    -1: "unlabeled",
     0: "animals",
     1: "landscape",
     2: "plant",
     3: "man-made object",
}


DECODER_MODES = [
    "raw",
    "zscore",
    "pca60",
    "pca60_whiten",
]


# =============================================================================
# Utilities
# =============================================================================

def sigmoid_np(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def load_and_prepare_data():
    print()
    print("#" * 100)
    print("Loading Allen natural-scenes composite")
    print("#" * 100)
    print(f"Input: {COMPOSITE_PATH}")

    data = np.load(COMPOSITE_PATH, allow_pickle=True).item()

    X = np.asarray(data["X"], dtype=np.float64)
    stimulus_metadata = data["stimulus_metadata"]
    neuron_metadata = data["neuron_metadata"]

    labels = np.asarray(
        stimulus_metadata["label"],
        dtype=np.int64,
    ).ravel()

    print()
    print(f"Raw X shape: {X.shape}")
    print(f"Labels shape: {labels.shape}")

    # -------------------------------------------------------------------------
    # Orient X as images × neurons
    # -------------------------------------------------------------------------

    if X.shape[1] == len(labels):
        print("[INFO] Transposing X: neurons × images -> images × neurons")
        X = X.T

    elif X.shape[0] == len(labels):
        print("[INFO] X already has shape images × neurons")

    else:
        raise ValueError(
            f"Cannot align X {X.shape} with labels {labels.shape}"
        )

    # -------------------------------------------------------------------------
    # Remove unlabeled images
    # -------------------------------------------------------------------------

    labeled_mask = labels != -1

    X = X[labeled_mask]
    labels_labeled = labels[labeled_mask]

    original_indices = np.where(labeled_mask)[0]

    # 1 = animal
    # 0 = everything else
    y = (labels_labeled == 0).astype(np.int64)

    print()
    print("Four-class counts:")
    for value, count in zip(
        *np.unique(labels_labeled, return_counts=True)
    ):
        print(
            f"  {value:>2} "
            f"{LABEL_NAMES.get(int(value), 'UNKNOWN'):<18} "
            f"n={count}"
        )

    print()
    print(
        "Binary counts [non-animal, animal]:",
        np.bincount(y, minlength=2),
    )

    # -------------------------------------------------------------------------
    # Remove globally invalid / constant neurons
    #
    # This is okay here because this step does not use class labels.
    # -------------------------------------------------------------------------

    finite_cols = np.all(np.isfinite(X), axis=0)

    finite_indices = np.where(finite_cols)[0]

    variable_finite_cols = (
        np.std(X[:, finite_cols], axis=0) > EPS
    )

    good_cols = np.zeros(X.shape[1], dtype=bool)
    good_cols[
        finite_indices[variable_finite_cols]
    ] = True

    X = X[:, good_cols]

    print()
    print("=" * 100)
    print("Prepared data")
    print("=" * 100)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Removed neurons: {np.sum(~good_cols)}")

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    if "brain_area" in neuron_metadata:
        brain_area = np.asarray(
            neuron_metadata["brain_area"],
            dtype=object,
        ).ravel()[good_cols]
    else:
        brain_area = np.full(
            X.shape[1],
            "unknown",
            dtype=object,
        )

    if "cell_specimen_id" in neuron_metadata:
        cell_specimen_id = np.asarray(
            neuron_metadata["cell_specimen_id"],
            dtype=np.int64,
        ).ravel()[good_cols]
    else:
        cell_specimen_id = np.full(
            X.shape[1],
            -1,
            dtype=np.int64,
        )

    if "ophys_experiment_id" in neuron_metadata:
        ophys_experiment_id = np.asarray(
            neuron_metadata["ophys_experiment_id"],
            dtype=np.int64,
        ).ravel()[good_cols]
    else:
        ophys_experiment_id = np.full(
            X.shape[1],
            -1,
            dtype=np.int64,
        )

    return {
        "X": X,
        "y": y,
        "labels_labeled": labels_labeled,
        "labeled_mask": labeled_mask,
        "original_indices": original_indices,
        "good_cols": good_cols,
        "brain_area": brain_area,
        "cell_specimen_id": cell_specimen_id,
        "ophys_experiment_id": ophys_experiment_id,
    }


# =============================================================================
# Logistic regression
# =============================================================================

class TorchLogisticRegression(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.linear = torch.nn.Linear(
            n_features,
            1,
        )

    def forward(self, x):
        return self.linear(x)


def fit_adam_logistic(
    X_train,
    y_train,
    seed,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_t = torch.tensor(
        X_train.astype(np.float32)
    )

    y_t = torch.tensor(
        y_train.astype(np.float32)
    ).view(-1, 1)

    model = TorchLogisticRegression(
        X_train.shape[1]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()

    for _ in range(EPOCHS):

        optimizer.zero_grad()

        logits = model(X_t)

        loss = loss_fn(
            logits,
            y_t,
        )

        loss.backward()

        optimizer.step()

    model.eval()

    with torch.no_grad():

        w = (
            model.linear.weight
            .detach()
            .cpu()
            .numpy()
            .ravel()
            .astype(np.float64)
        )

        b = float(
            model.linear.bias
            .detach()
            .cpu()
            .numpy()[0]
        )

    return w, b


# =============================================================================
# Fold preprocessing
# =============================================================================

def preprocess_fold(
    X_train_raw,
    X_test_raw,
    mode,
):

    # =========================================================================
    # RAW
    # =========================================================================

    if mode == "raw":

        return (
            X_train_raw.copy(),
            X_test_raw.copy(),
            None,
        )

    # =========================================================================
    # Z-SCORE
    # =========================================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_raw
    )

    X_test_scaled = scaler.transform(
        X_test_raw
    )

    if mode == "zscore":

        return (
            X_train_scaled,
            X_test_scaled,
            {
                "scaler": scaler,
            },
        )

    # =========================================================================
    # PCA
    # =========================================================================

    max_components = min(
        PCA_COMPONENTS,
        X_train_scaled.shape[0] - 1,
        X_train_scaled.shape[1],
    )

    if max_components < 1:
        raise ValueError(
            "Not enough training samples for PCA."
        )

    whiten = (
        mode == "pca60_whiten"
    )

    pca = PCA(
        n_components=max_components,
        whiten=whiten,
        svd_solver="full",
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    )

    X_test_pca = pca.transform(
        X_test_scaled
    )

    return (
        X_train_pca,
        X_test_pca,
        {
            "scaler": scaler,
            "pca": pca,
        },
    )


# =============================================================================
# LOO
# =============================================================================

def run_loo_decoder(
    X,
    y,
    mode,
):
    n_samples = X.shape[0]

    logits = np.zeros(
        n_samples,
        dtype=np.float64,
    )

    probs = np.zeros(
        n_samples,
        dtype=np.float64,
    )

    preds = np.zeros(
        n_samples,
        dtype=np.int64,
    )

    print()
    print("#" * 100)
    print(f"Decoder: {mode}")
    print("#" * 100)

    for test_idx in range(n_samples):

        train_mask = (
            np.arange(n_samples) != test_idx
        )

        X_train_raw = X[train_mask]
        X_test_raw = X[~train_mask]

        y_train = y[train_mask]

        # ---------------------------------------------------------------------
        # Fit preprocessing ONLY on training data
        # ---------------------------------------------------------------------

        X_train, X_test, preprocessing = preprocess_fold(
            X_train_raw,
            X_test_raw,
            mode,
        )

        # ---------------------------------------------------------------------
        # Fit decoder
        # ---------------------------------------------------------------------

        w, b = fit_adam_logistic(
            X_train,
            y_train,
            seed=10_000 + test_idx,
        )

        logit = float(
            X_test[0] @ w + b
        )

        prob = float(
            sigmoid_np(logit)
        )

        pred = int(
            prob >= 0.5
        )

        logits[test_idx] = logit
        probs[test_idx] = prob
        preds[test_idx] = pred

        running_acc = accuracy_score(
            y[: test_idx + 1],
            preds[: test_idx + 1],
        )

        print(
            f"[{mode:<14} "
            f"{test_idx + 1:03d}/{n_samples}] "
            f"true={y[test_idx]} "
            f"logit={logit:+.5f} "
            f"p={prob:.4f} "
            f"pred={pred} "
            f"acc={running_acc:.4f}"
        )

    # =========================================================================
    # Metrics
    # =========================================================================

    accuracy = accuracy_score(
        y,
        preds,
    )

    balanced_accuracy = balanced_accuracy_score(
        y,
        preds,
    )

    auc = roc_auc_score(
        y,
        probs,
    )

    cm = confusion_matrix(
        y,
        preds,
        labels=[0, 1],
    )

    print()
    print("=" * 100)
    print(f"{mode} results")
    print("=" * 100)
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"AUC:               {auc:.4f}")
    print()
    print("Confusion matrix:")
    print(cm)

    return {
        "mode": mode,
        "logits": logits,
        "probs": probs,
        "preds": preds,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "auc": auc,
        "confusion_matrix": cm,
    }


# =============================================================================
# Main
# =============================================================================

def main():

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print()
    print("#" * 100)
    print("ALLEN ANIMALS VS REST")
    print("RAW vs Z-SCORE vs PCA vs PCA-WHITENING")
    print("#" * 100)

    prepared = load_and_prepare_data()

    X = prepared["X"]
    y = prepared["y"]

    results = {}

    for mode in DECODER_MODES:

        result = run_loo_decoder(
            X,
            y,
            mode,
        )

        results[mode] = result

    # =========================================================================
    # Summary
    # =========================================================================

    print()
    print()
    print("#" * 100)
    print("FINAL COMPARISON")
    print("#" * 100)

    print(
        f"{'Decoder':<18}"
        f"{'Accuracy':>12}"
        f"{'Bal.Acc':>12}"
        f"{'AUC':>12}"
    )

    print("-" * 54)

    for mode in DECODER_MODES:

        r = results[mode]

        print(
            f"{mode:<18}"
            f"{r['accuracy']:>12.4f}"
            f"{r['balanced_accuracy']:>12.4f}"
            f"{r['auc']:>12.4f}"
        )

    # =========================================================================
    # Save
    # =========================================================================

    save_dict = {
        # -------------------------------------------------------------
        # Data alignment
        # -------------------------------------------------------------

        "y": y,

        "four_class_labels_labeled":
            prepared["labels_labeled"],

        "labeled_mask":
            prepared["labeled_mask"],

        "original_indices":
            prepared["original_indices"],

        "good_cols":
            prepared["good_cols"],

        # -------------------------------------------------------------
        # Metadata
        # -------------------------------------------------------------

        "brain_area":
            prepared["brain_area"],

        "cell_specimen_id":
            prepared["cell_specimen_id"],

        "ophys_experiment_id":
            prepared["ophys_experiment_id"],

        # -------------------------------------------------------------
        # Settings
        # -------------------------------------------------------------

        "lr": LR,

        "weight_decay":
            WEIGHT_DECAY,

        "epochs":
            EPOCHS,

        "pca_components":
            PCA_COMPONENTS,

        "random_seed":
            RANDOM_SEED,
    }

    # Add decoder-specific results
    for mode, r in results.items():

        prefix = mode

        save_dict[
            f"{prefix}_logits"
        ] = r["logits"]

        save_dict[
            f"{prefix}_probs"
        ] = r["probs"]

        save_dict[
            f"{prefix}_preds"
        ] = r["preds"]

        save_dict[
            f"{prefix}_accuracy"
        ] = r["accuracy"]

        save_dict[
            f"{prefix}_balanced_accuracy"
        ] = r["balanced_accuracy"]

        save_dict[
            f"{prefix}_auc"
        ] = r["auc"]

        save_dict[
            f"{prefix}_confusion_matrix"
        ] = r["confusion_matrix"]

    np.savez_compressed(
        OUT_NPZ,
        **save_dict,
    )

    print()
    print("=" * 100)
    print("Saved")
    print("=" * 100)
    print(OUT_NPZ)

    print()
    print("Done.")


if __name__ == "__main__":
    main()