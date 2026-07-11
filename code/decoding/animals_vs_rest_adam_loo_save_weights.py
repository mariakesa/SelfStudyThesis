#!/usr/bin/env python3
"""
Decode animals vs all other labeled image classes using all Allen neurons.

Input composite file:
    /home/maria/SelfStudyThesis/data/allen_natural_scenes_four_class_composite.npy

Expected composite structure:
    data["X"]                       total_neurons × 118
    data["stimulus_metadata"]["label"]    118 labels
    data["neuron_metadata"]              row-aligned neuron metadata

Four-class labels:
    -1 = unlabeled
     0 = animals
     1 = landscape
     2 = plant
     3 = man-made object

This script:
    1. Loads the composite file.
    2. Transposes X to images × neurons.
    3. Excludes unlabeled images.
    4. Converts labels to binary:
           1 = animals
           0 = all other labeled classes
    5. Removes invalid / nonconstant neurons.
    6. Runs leave-one-out Adam logistic regression using all neurons.
    7. Saves results to .npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

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
    "/home/maria/SelfStudyThesis/results/all_neurons_animals_vs_rest_adam_loo"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUTDIR / "all_neurons_animals_vs_rest_adam_loo_results.npz"


# =============================================================================
# Settings copied from Adam decoder script
# =============================================================================

LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 3000

RANDOM_SEED = 0
EPS = 1e-12


LABEL_NAMES = {
    -1: "unlabeled",
     0: "animals",
     1: "landscape",
     2: "plant",
     3: "man-made object",
}


# =============================================================================
# Utility
# =============================================================================

def sigmoid_np(z: float | np.ndarray) -> np.ndarray:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


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

    # Composite X is total_neurons × images.
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


def prepare_binary_animals_vs_rest():
    data, X, labels, brain_area, cell_specimen_id, ophys_experiment_id = load_composite_data()

    # Exclude unlabeled images.
    labeled_mask = labels != -1
    original_indices = np.where(labeled_mask)[0]

    X_labeled = X[labeled_mask]
    labels_labeled = labels[labeled_mask]

    # Binary target:
    #   1 = animals
    #   0 = all other labeled classes
    y = (labels_labeled == 0).astype(np.int64)

    print()
    print("=" * 80)
    print("Binary decoding target")
    print("=" * 80)
    print("Positive class: 1 = animals")
    print("Negative class: 0 = landscape + plant + man-made object")
    print()
    print(f"X labeled shape: {X_labeled.shape}")
    print(f"y shape:         {y.shape}")
    print(f"Original image indices used: {original_indices.shape}")

    print()
    print("Four-class counts after excluding unlabeled:")
    unique, counts = np.unique(labels_labeled, return_counts=True)
    for value, count in zip(unique, counts):
        name = LABEL_NAMES.get(int(value), "UNKNOWN")
        print(f"  {int(value):>2} = {name:<16} n={int(count)}")

    print()
    print("Binary counts [non-animal, animal]:")
    print(np.bincount(y, minlength=2))

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
        "four_class_labels_labeled": labels_labeled,
        "labeled_mask": labeled_mask,
        "original_indices": original_indices,
        "good_cols": good_cols,
        "brain_area_clean": brain_area_clean,
        "cell_specimen_id_clean": cell_specimen_id_clean,
        "ophys_experiment_id_clean": ophys_experiment_id_clean,
    }


# =============================================================================
# Adam logistic regression
# =============================================================================

class TorchLogisticRegression(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


def fit_adam_logistic(
    X_train,
    y_train,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    epochs=EPOCHS,
    seed=0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_t = torch.tensor(X_train.astype(np.float32))
    y_t = torch.tensor(y_train.astype(np.float32)).view(-1, 1)

    model = TorchLogisticRegression(X_train.shape[1])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        w = model.linear.weight.detach().cpu().numpy().ravel().astype(np.float64)
        b = float(model.linear.bias.detach().cpu().numpy()[0])

    return w, b


# =============================================================================
# LOO decoder
# =============================================================================

def run_loo_decoder_all_neurons(X, y, decoder_name: str = "all_neurons"):
    n, d = X.shape

    if d == 0:
        raise ValueError("No features found after cleaning.")

    logits = np.zeros(n, dtype=np.float64)
    probs = np.zeros(n, dtype=np.float64)
    preds = np.zeros(n, dtype=np.int64)

    # One model is fitted per LOO fold, so there is not just one weight vector.
    # Rows correspond to held-out images; columns correspond to cleaned neurons.
    #
    # standardized weights:
    #     coefficients acting on StandardScaler-transformed activity
    #
    # raw weights:
    #     equivalent coefficients acting directly on the original activity
    #
    # heldout_evidence:
    #     per-neuron contribution to the held-out image's logit:
    #         evidence[test_idx, neuron] = z_scored_activity * standardized_weight
    #     Therefore:
    #         heldout_evidence[test_idx].sum() + loo_bias_standardized[test_idx]
    #         == logits[test_idx]
    loo_weights_standardized = np.zeros((n, d), dtype=np.float32)
    loo_weights_raw = np.zeros((n, d), dtype=np.float32)
    loo_bias_standardized = np.zeros(n, dtype=np.float64)
    loo_bias_raw = np.zeros(n, dtype=np.float64)
    heldout_evidence = np.zeros((n, d), dtype=np.float32)

    print()
    print("#" * 80)
    print(f"Running LOO Adam decoder: {decoder_name}")
    print("#" * 80)
    print(f"X shape: {X.shape}")
    print(f"Label counts [non-animal, animal]: {np.bincount(y, minlength=2)}")
    print(f"LR={LR}, weight_decay={WEIGHT_DECAY}, epochs={EPOCHS}")
    print()

    for test_idx in range(n):
        train_mask = np.arange(n) != test_idx

        X_train_raw = X[train_mask]
        y_train = y[train_mask]
        X_test_raw = X[~train_mask]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        w, b = fit_adam_logistic(
            X_train,
            y_train,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            epochs=EPOCHS,
            seed=10_000 + test_idx,
        )

        # Save the fold-specific decoder axis.
        loo_weights_standardized[test_idx] = w.astype(np.float32)
        loo_bias_standardized[test_idx] = b

        # Convert the standardized-space model back to the original activity
        # units. For z = (x - mean) / scale:
        #
        #     z @ w + b = x @ (w / scale) + [b - mean @ (w / scale)]
        #
        # StandardScaler protects zero-variance columns by using scale=1, but
        # such columns were already removed above.
        w_raw = w / scaler.scale_
        b_raw = float(b - scaler.mean_ @ w_raw)

        loo_weights_raw[test_idx] = w_raw.astype(np.float32)
        loo_bias_raw[test_idx] = b_raw

        # Signed evidence contributed by every neuron for this held-out image.
        # Positive values push toward "animal"; negative values push toward
        # "non-animal".
        evidence = X_test[0] * w
        heldout_evidence[test_idx] = evidence.astype(np.float32)

        logit = float(evidence.sum() + b)
        prob = float(sigmoid_np(logit))
        pred = int(prob >= 0.5)

        logits[test_idx] = logit
        probs[test_idx] = prob
        preds[test_idx] = pred

        running_acc = accuracy_score(y[: test_idx + 1], preds[: test_idx + 1])
        running_bal_acc = balanced_accuracy_score(
            y[: test_idx + 1],
            preds[: test_idx + 1],
        )

        print(
            f"[{decoder_name} LOO {test_idx + 1:03d}/{n}] "
            f"true={y[test_idx]} "
            f"logit={logit:+.6f} "
            f"prob={prob:.4f} "
            f"pred={pred} "
            f"running_acc={running_acc:.4f} "
            f"running_bal_acc={running_bal_acc:.4f}"
        )

    acc = accuracy_score(y, preds)
    bal_acc = balanced_accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    cm = confusion_matrix(y, preds, labels=[0, 1])

    print()
    print("=" * 80)
    print(f"{decoder_name} LOO summary")
    print("=" * 80)
    print(f"Features / neurons: {d}")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Balanced accuracy:  {bal_acc:.4f}")
    print(f"AUC:                {auc:.4f}")
    print()
    print("Confusion matrix rows=true [non-animal, animal], cols=pred [non-animal, animal]:")
    print(cm)

    # Also fit one descriptive model on the complete labeled dataset. This is
    # useful as a single population axis for plotting and interpretation.
    # It is NOT used to compute the cross-validated performance above.
    full_scaler = StandardScaler()
    X_full_standardized = full_scaler.fit_transform(X)

    full_w_standardized, full_b_standardized = fit_adam_logistic(
        X_full_standardized,
        y,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        seed=RANDOM_SEED,
    )

    full_w_raw = full_w_standardized / full_scaler.scale_
    full_b_raw = float(
        full_b_standardized - full_scaler.mean_ @ full_w_raw
    )

    print()
    print("Saved weight interpretation:")
    print("  positive coefficient/evidence -> pushes toward animal")
    print("  negative coefficient/evidence -> pushes toward non-animal")
    print("  LOO weight matrix shape:      ", loo_weights_standardized.shape)
    print("  held-out evidence shape:      ", heldout_evidence.shape)

    return {
        "decoder_name": decoder_name,
        "n_features": d,
        "logits": logits,
        "probs": probs,
        "preds": preds,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "auc": auc,
        "confusion_matrix": cm,

        # Fold-specific coefficients and per-image evidence
        "loo_weights_standardized": loo_weights_standardized,
        "loo_weights_raw": loo_weights_raw,
        "loo_bias_standardized": loo_bias_standardized,
        "loo_bias_raw": loo_bias_raw,
        "heldout_evidence": heldout_evidence,
        "mean_loo_weight_standardized": np.mean(
            loo_weights_standardized, axis=0
        ).astype(np.float32),
        "mean_abs_loo_weight_standardized": np.mean(
            np.abs(loo_weights_standardized), axis=0
        ).astype(np.float32),

        # One descriptive model fitted to all labeled images
        "full_weights_standardized": full_w_standardized.astype(np.float32),
        "full_bias_standardized": full_b_standardized,
        "full_weights_raw": full_w_raw.astype(np.float32),
        "full_bias_raw": full_b_raw,
        "full_scaler_mean": full_scaler.mean_.astype(np.float32),
        "full_scaler_scale": full_scaler.scale_.astype(np.float32),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print()
    print("#" * 80)
    print("Animals vs rest Adam logistic LOO decoder — all neurons")
    print("#" * 80)

    prepared = prepare_binary_animals_vs_rest()

    X = prepared["X"]
    y = prepared["y"]

    result = run_loo_decoder_all_neurons(
        X,
        y,
        decoder_name="animals_vs_rest_all_neurons",
    )

    np.savez_compressed(
        OUT_NPZ,

        # Data alignment
        y=y,
        four_class_labels_labeled=prepared["four_class_labels_labeled"],
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
        accuracy=result["accuracy"],
        balanced_accuracy=result["balanced_accuracy"],
        auc=result["auc"],
        confusion_matrix=result["confusion_matrix"],

        # Fold-specific decoder weights.
        # Axis 0 = held-out image/fold; axis 1 = cleaned neuron.
        loo_weights_standardized=result["loo_weights_standardized"],
        loo_weights_raw=result["loo_weights_raw"],
        loo_bias_standardized=result["loo_bias_standardized"],
        loo_bias_raw=result["loo_bias_raw"],

        # Per-neuron signed contribution to each held-out image's logit.
        # Positive evidence favors animal; negative evidence favors non-animal.
        heldout_evidence=result["heldout_evidence"],

        # Across-fold summaries for ranking neurons.
        mean_loo_weight_standardized=result["mean_loo_weight_standardized"],
        mean_abs_loo_weight_standardized=result[
            "mean_abs_loo_weight_standardized"
        ],

        # One descriptive model fitted to all labeled images.
        # This model is convenient for a single interpretable population axis,
        # but it is not used for the LOO performance estimates.
        full_weights_standardized=result["full_weights_standardized"],
        full_bias_standardized=result["full_bias_standardized"],
        full_weights_raw=result["full_weights_raw"],
        full_bias_raw=result["full_bias_raw"],
        full_scaler_mean=result["full_scaler_mean"],
        full_scaler_scale=result["full_scaler_scale"],

        # Settings
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        random_seed=RANDOM_SEED,
        eps=EPS,

        # Labels
        positive_class_label=0,
        positive_class_name="animals",
        negative_class_name="landscape + plant + man-made object",
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