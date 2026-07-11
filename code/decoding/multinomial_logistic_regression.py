#!/usr/bin/env python3
"""
Four-class softmax decoder using Torch + Adam.

Task:
    Decode image label among:
        0 = animals
        1 = landscape
        2 = plant
        3 = man-made object

Model:
    Multiclass linear softmax regression:
        logits = X @ W.T + b

Loss:
    CrossEntropyLoss

Evaluation:
    Leave-one-image-out cross-validation.

Uses the same Adam hyperparameters as the previous binary decoder:
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 3000
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    log_loss,
)


# =============================================================================
# Paths
# =============================================================================

COMPOSITE_PATH = Path(
    "/home/maria/SelfStudyThesis/data/allen_natural_scenes_four_class_composite.npy"
)

OUTDIR = Path(
    "/home/maria/SelfStudyThesis/results/all_neurons_four_class_softmax_adam_loo"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUTDIR / "all_neurons_four_class_softmax_adam_loo_results.npz"


# =============================================================================
# Settings copied from Adam decoder script
# =============================================================================

LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 3000

RANDOM_SEED = 0
EPS = 1e-12

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


LABEL_NAMES = {
    -1: "unlabeled",
     0: "animals",
     1: "landscape",
     2: "plant",
     3: "man-made object",
}

CLASSES = np.array([0, 1, 2, 3], dtype=np.int64)
CLASS_NAMES = [LABEL_NAMES[int(c)] for c in CLASSES]


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

    # Composite X is usually total_neurons × images.
    # Decoder expects images × neurons/features.
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


def prepare_four_class_dataset():
    (
        data,
        X,
        labels,
        brain_area,
        cell_specimen_id,
        ophys_experiment_id,
    ) = load_composite_data()

    # Exclude unlabeled images.
    labeled_mask = labels != -1
    original_indices = np.where(labeled_mask)[0]

    X_labeled = X[labeled_mask]
    y = labels[labeled_mask].astype(np.int64)

    print()
    print("=" * 80)
    print("Four-class decoding target")
    print("=" * 80)
    print(f"X labeled shape: {X_labeled.shape}")
    print(f"y shape:         {y.shape}")
    print(f"Original image indices used: {original_indices.shape}")

    print()
    print("Four-class counts after excluding unlabeled:")
    unique, counts = np.unique(y, return_counts=True)
    for value, count in zip(unique, counts):
        name = LABEL_NAMES.get(int(value), "UNKNOWN")
        print(f"  {int(value):>2} = {name:<16} n={int(count)}")

    # Check labels are exactly 0,1,2,3 after removing unlabeled.
    if not np.array_equal(np.sort(np.unique(y)), CLASSES):
        raise ValueError(
            f"Expected labels {CLASSES.tolist()}, got {np.sort(np.unique(y)).tolist()}"
        )

    # Globally remove invalid / nonconstant features while preserving metadata alignment.
    finite_cols = np.all(np.isfinite(X_labeled), axis=0)

    finite_indices = np.where(finite_cols)[0]
    std_cols_among_finite = np.std(X_labeled[:, finite_cols], axis=0) > EPS

    good_cols = np.zeros(X_labeled.shape[1], dtype=bool)
    good_cols[finite_indices[std_cols_among_finite]] = True

    X_clean = X_labeled[:, good_cols]

    brain_area_clean = brain_area[good_cols]
    cell_specimen_id_clean = cell_specimen_id[good_cols]
    ophys_experiment_id_clean = ophys_experiment_id[good_cols]

    print()
    print("=" * 80)
    print("After feature cleaning")
    print("=" * 80)
    print(f"X clean shape: {X_clean.shape}")
    print(f"Removed bad/nonconstant neurons: {int(np.sum(~good_cols))}")

    print()
    print("Feature counts by brain area after cleaning:")
    areas, area_counts = np.unique(brain_area_clean.astype(str), return_counts=True)
    for area, count in zip(areas, area_counts):
        print(f"  {area:<8} {int(count)}")

    return {
        "data": data,
        "X": X_clean,
        "y": y,
        "labeled_mask": labeled_mask,
        "original_indices": original_indices,
        "good_cols": good_cols,
        "brain_area_clean": brain_area_clean,
        "cell_specimen_id_clean": cell_specimen_id_clean,
        "ophys_experiment_id_clean": ophys_experiment_id_clean,
    }


# =============================================================================
# Torch softmax regression
# =============================================================================

class TorchSoftmaxRegression(torch.nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.linear(x)


def fit_adam_softmax(
    X_train,
    y_train,
    n_classes: int,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    epochs=EPOCHS,
    seed=0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

    X_t = torch.tensor(X_train.astype(np.float32), device=DEVICE)
    y_t = torch.tensor(y_train.astype(np.int64), device=DEVICE)

    model = TorchSoftmaxRegression(X_train.shape[1], n_classes).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        W = model.linear.weight.detach().cpu().numpy().astype(np.float64)
        b = model.linear.bias.detach().cpu().numpy().astype(np.float64)

    return W, b


def softmax_np(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


# =============================================================================
# LOO decoder
# =============================================================================

def run_loo_softmax_decoder_all_neurons(X, y, decoder_name: str):
    n, d = X.shape
    n_classes = len(CLASSES)

    if d == 0:
        raise ValueError("No features found after cleaning.")

    logits = np.zeros((n, n_classes), dtype=np.float64)
    probs = np.zeros((n, n_classes), dtype=np.float64)
    preds = np.zeros(n, dtype=np.int64)

    # Optional: store fold weights.
    # Shape: heldout_image × class × neuron
    # float32 keeps it smaller.
    weights_by_fold = np.zeros((n, n_classes, d), dtype=np.float32)
    bias_by_fold = np.zeros((n, n_classes), dtype=np.float32)

    print()
    print("#" * 80)
    print(f"Running LOO Adam softmax decoder: {decoder_name}")
    print("#" * 80)
    print(f"X shape: {X.shape}")
    print(f"Label counts: {np.bincount(y, minlength=n_classes)}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"LR={LR}, weight_decay={WEIGHT_DECAY}, epochs={EPOCHS}")
    print(f"Device: {DEVICE}")
    print()

    for test_idx in range(n):
        train_mask = np.arange(n) != test_idx

        X_train_raw = X[train_mask]
        y_train = y[train_mask]
        X_test_raw = X[~train_mask]

        missing_classes = sorted(set(CLASSES.tolist()) - set(y_train.tolist()))
        if missing_classes:
            raise ValueError(
                f"LOO fold {test_idx + 1} has missing classes in training data: "
                f"{missing_classes}"
            )

        # Train-fold-only standardization.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        W, b = fit_adam_softmax(
            X_train,
            y_train,
            n_classes=n_classes,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            epochs=EPOCHS,
            seed=10_000 + test_idx,
        )

        test_logits = X_test @ W.T + b
        test_probs = softmax_np(test_logits)
        pred = int(np.argmax(test_probs[0]))

        logits[test_idx] = test_logits[0]
        probs[test_idx] = test_probs[0]
        preds[test_idx] = pred

        weights_by_fold[test_idx] = W.astype(np.float32)
        bias_by_fold[test_idx] = b.astype(np.float32)

        running_acc = accuracy_score(y[: test_idx + 1], preds[: test_idx + 1])
        running_bal_acc = balanced_accuracy_score(
            y[: test_idx + 1],
            preds[: test_idx + 1],
        )

        true_prob = float(test_probs[0, y[test_idx]])

        print(
            f"[{decoder_name} LOO {test_idx + 1:03d}/{n}] "
            f"true={y[test_idx]} "
            f"pred={pred} "
            f"p_true={true_prob:.4f} "
            f"p={np.round(test_probs[0], 3).tolist()} "
            f"running_acc={running_acc:.4f} "
            f"running_bal_acc={running_bal_acc:.4f}"
        )

    acc = accuracy_score(y, preds)
    bal_acc = balanced_accuracy_score(y, preds)
    cm = confusion_matrix(y, preds, labels=CLASSES)

    # Multiclass negative log loss using held-out probabilities.
    nll = log_loss(y, probs, labels=CLASSES)

    print()
    print("=" * 80)
    print(f"{decoder_name} LOO summary")
    print("=" * 80)
    print(f"Features / neurons: {d}")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Balanced accuracy:  {bal_acc:.4f}")
    print(f"Log loss:           {nll:.4f}")

    print()
    print("Confusion matrix rows=true, cols=pred:")
    print("Labels:", CLASS_NAMES)
    print(cm)

    print()
    print("Classification report:")
    print(
        classification_report(
            y,
            preds,
            labels=CLASSES,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print()
    print("Mean held-out probability assigned to true class:")
    true_probs = probs[np.arange(n), y]
    print(f"Mean:   {np.mean(true_probs):.6f}")
    print(f"Median: {np.median(true_probs):.6f}")
    print(f"Min:    {np.min(true_probs):.6f}")
    print(f"Max:    {np.max(true_probs):.6f}")

    return {
        "decoder_name": decoder_name,
        "n_features": d,
        "logits": logits,
        "probs": probs,
        "preds": preds,
        "weights_by_fold": weights_by_fold,
        "bias_by_fold": bias_by_fold,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "log_loss": nll,
        "confusion_matrix": cm,
        "true_probs": true_probs,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print()
    print("#" * 80)
    print("Four-class Adam softmax LOO decoder — all neurons")
    print("#" * 80)

    prepared = prepare_four_class_dataset()

    X = prepared["X"]
    y = prepared["y"]

    result = run_loo_softmax_decoder_all_neurons(
        X,
        y,
        decoder_name="four_class_softmax_all_neurons",
    )

    np.savez_compressed(
        OUT_NPZ,

        # Data alignment
        y=y,
        labeled_mask=prepared["labeled_mask"],
        original_indices=prepared["original_indices"],
        good_cols=prepared["good_cols"],

        # Feature metadata after cleaning
        brain_area_clean=prepared["brain_area_clean"],
        cell_specimen_id_clean=prepared["cell_specimen_id_clean"],
        ophys_experiment_id_clean=prepared["ophys_experiment_id_clean"],

        # Decoder outputs
        n_features=result["n_features"],
        logits=result["logits"],
        probs=result["probs"],
        preds=result["preds"],
        weights_by_fold=result["weights_by_fold"],
        bias_by_fold=result["bias_by_fold"],
        accuracy=result["accuracy"],
        balanced_accuracy=result["balanced_accuracy"],
        log_loss=result["log_loss"],
        confusion_matrix=result["confusion_matrix"],
        true_probs=result["true_probs"],

        # Settings
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        random_seed=RANDOM_SEED,
        eps=EPS,
        device=DEVICE,

        # Labels
        classes=CLASSES,
        class_names=np.asarray(CLASS_NAMES, dtype=object),
    )

    print()
    print("=" * 80)
    print("Saved results")
    print("=" * 80)
    print(OUT_NPZ)

    print()
    print("Done.")


if __name__ == "__main__":
    main()