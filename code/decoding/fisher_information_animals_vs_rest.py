#!/usr/bin/env python3
"""
Fisher-information analysis for animals-vs-rest logistic regression.

This script:
    1. Loads the Allen four-class composite dataset.
    2. Converts the task to animals (1) vs all other labeled classes (0).
    3. Removes invalid and constant neurons.
    4. Standardizes all retained neural features.
    5. Fits one full-data logistic-regression model with Adam.
    6. Computes the empirical Fisher information / logistic Hessian

           F = (1 / n) X_aug.T @ W @ X_aug

       where W_ii = p_i(1 - p_i).

       Because p >> n, the nonzero eigenvalues are computed through the
       much smaller n x n matrix

           K = B @ B.T,
           B = sqrt(W / n)[:, None] * X_aug.

       K and F have the same nonzero eigenvalues.
    7. Quantifies:
         - numerical rank
         - nullity
         - largest and smallest positive eigenvalues
         - condition number on the identifiable subspace
         - effective rank
         - participation ratio
         - curvature along the learned weight direction
         - curvature along random directions
    8. Saves arrays, a text summary, and separate publication-style plots.

Important:
    This is a descriptive full-data geometry analysis, not a cross-validated
    estimate of decoding performance.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Paths
# =============================================================================

COMPOSITE_PATH = Path(
    "/home/maria/SelfStudyThesis/data/allen_natural_scenes_four_class_composite.npy"
)

OUTDIR = Path(
    "/home/maria/SelfStudyThesis/results/"
    "fisher_information_animals_vs_rest"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUTDIR / "fisher_information_animals_vs_rest.npz"
OUT_TXT = OUTDIR / "fisher_information_summary.txt"
OUT_JSON = OUTDIR / "fisher_information_summary.json"


# =============================================================================
# Optimization settings
# =============================================================================

LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 3000
RANDOM_SEED = 0

# Numerical settings
VAR_EPS = 1e-12
EIGENVALUE_REL_TOL = 1e-10
N_RANDOM_DIRECTIONS = 5000


LABEL_NAMES = {
    -1: "unlabeled",
     0: "animals",
     1: "landscape",
     2: "plant",
     3: "man-made object",
}


# =============================================================================
# Reproducibility
# =============================================================================

def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# =============================================================================
# Data loading
# =============================================================================

def load_composite_data():
    print("\n" + "#" * 80)
    print("Loading composite dataset")
    print("#" * 80)
    print(f"Composite path: {COMPOSITE_PATH}")

    data = np.load(COMPOSITE_PATH, allow_pickle=True).item()

    X = np.asarray(data["X"], dtype=np.float64)
    stimulus_metadata = data["stimulus_metadata"]
    neuron_metadata = data["neuron_metadata"]

    labels = np.asarray(stimulus_metadata["label"], dtype=np.int64).ravel()

    print("\nRaw four-class label counts:")
    unique, counts = np.unique(labels, return_counts=True)
    for value, count in zip(unique, counts):
        name = LABEL_NAMES.get(int(value), "UNKNOWN")
        print(f"  {int(value):>2} = {name:<16} n={int(count)}")

    if X.shape[1] == len(labels):
        print("\n[INFO] Transposing X from neurons x images to images x neurons.")
        X = X.T
    elif X.shape[0] == len(labels):
        print("\n[INFO] X already appears to be images x neurons.")
    else:
        raise ValueError(
            f"Cannot align X with labels. X={X.shape}, labels={labels.shape}."
        )

    return data, X, labels, neuron_metadata


def prepare_binary_dataset(
    X: np.ndarray,
    labels: np.ndarray,
):
    labeled_mask = labels != -1
    X = X[labeled_mask]
    labels_labeled = labels[labeled_mask]

    y = (labels_labeled == 0).astype(np.float64)

    finite_mask = np.all(np.isfinite(X), axis=0)
    variances = np.var(X, axis=0)
    nonconstant_mask = variances > VAR_EPS
    feature_mask = finite_mask & nonconstant_mask

    X = X[:, feature_mask]

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xz = scaler.fit_transform(X)

    print("\n" + "=" * 80)
    print("Prepared binary dataset")
    print("=" * 80)
    print(f"Samples:                    {Xz.shape[0]}")
    print(f"Original neurons:           {len(feature_mask)}")
    print(f"Retained neurons:           {Xz.shape[1]}")
    print(f"Removed invalid neurons:    {np.sum(~finite_mask)}")
    print(f"Removed constant neurons:   {np.sum(finite_mask & ~nonconstant_mask)}")
    print(f"Animals:                    {int(y.sum())}")
    print(f"Other labeled classes:      {int(len(y) - y.sum())}")

    return (
        Xz.astype(np.float64),
        y.astype(np.float64),
        labeled_mask,
        feature_mask,
        scaler,
    )


# =============================================================================
# Logistic regression
# =============================================================================

class LogisticRegressionTorch(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


def fit_logistic_adam(
    X: np.ndarray,
    y: np.ndarray,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    X_t = torch.as_tensor(X, dtype=torch.float64, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float64, device=device)

    model = LogisticRegressionTorch(X.shape[1]).to(device=device, dtype=torch.float64)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

    losses = np.empty(EPOCHS, dtype=np.float64)

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

        losses[epoch] = float(loss.detach().cpu())

        if epoch == 0 or (epoch + 1) % 250 == 0:
            print(f"Epoch {epoch + 1:4d}/{EPOCHS}: loss={losses[epoch]:.8f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_t).cpu().numpy()
        probs = torch.sigmoid(model(X_t)).cpu().numpy()

    weight = model.linear.weight.detach().cpu().numpy().ravel()
    intercept = float(model.linear.bias.detach().cpu().numpy().ravel()[0])

    predictions = (probs >= 0.5).astype(np.int64)
    training_accuracy = float(np.mean(predictions == y))

    print(f"\nFull-data training accuracy: {training_accuracy:.6f}")
    print(f"Final BCE loss:             {losses[-1]:.8f}")
    print(f"Weight norm:                {np.linalg.norm(weight):.8f}")
    print(f"Intercept:                  {intercept:.8f}")

    return weight, intercept, logits, probs, losses, training_accuracy


# =============================================================================
# Fisher analysis
# =============================================================================

def entropy_effective_rank(eigenvalues: np.ndarray) -> float:
    positive = eigenvalues[eigenvalues > 0]
    if len(positive) == 0:
        return 0.0
    q = positive / positive.sum()
    entropy = -np.sum(q * np.log(q))
    return float(np.exp(entropy))


def participation_ratio(eigenvalues: np.ndarray) -> float:
    positive = eigenvalues[eigenvalues > 0]
    if len(positive) == 0:
        return 0.0
    return float((positive.sum() ** 2) / np.sum(positive ** 2))


def fisher_analysis(
    X: np.ndarray,
    probs: np.ndarray,
    weight: np.ndarray,
    intercept: float,
):
    n_samples, n_features = X.shape

    # Add intercept column. The resulting Fisher has dimension (p + 1) x (p + 1).
    X_aug = np.column_stack([X, np.ones(n_samples, dtype=np.float64)])

    variance_weights = probs * (1.0 - probs)
    B = np.sqrt(variance_weights / n_samples)[:, None] * X_aug

    # Nonzero eigenvalues of F = B.T B equal those of K = B B.T.
    K = B @ B.T
    eigvals, eigvecs_small = np.linalg.eigh(K)

    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs_small = eigvecs_small[:, order]

    lambda_max = float(eigvals[0]) if len(eigvals) else 0.0
    absolute_tol = EIGENVALUE_REL_TOL * lambda_max if lambda_max > 0 else 0.0
    positive_mask = eigvals > absolute_tol
    positive_eigvals = eigvals[positive_mask]

    numerical_rank = int(np.sum(positive_mask))
    fisher_dimension = n_features + 1
    nullity = int(fisher_dimension - numerical_rank)

    smallest_positive = (
        float(positive_eigvals[-1]) if len(positive_eigvals) else np.nan
    )
    condition_number = (
        float(lambda_max / smallest_positive)
        if len(positive_eigvals) and smallest_positive > 0
        else np.inf
    )

    total_curvature = float(np.sum(eigvals))
    cumulative_fraction = (
        np.cumsum(eigvals) / total_curvature
        if total_curvature > 0
        else np.zeros_like(eigvals)
    )

    effective_rank_value = entropy_effective_rank(positive_eigvals)
    participation_ratio_value = participation_ratio(positive_eigvals)

    # Curvature along the fitted parameter vector, including intercept.
    theta = np.concatenate([weight, np.array([intercept])])
    theta_norm = np.linalg.norm(theta)

    if theta_norm > 0:
        theta_unit = theta / theta_norm
        fisher_theta_curvature = float(np.sum((B @ theta_unit) ** 2))
    else:
        fisher_theta_curvature = np.nan

    # Random directions in the full p+1 dimensional space.
    # We do not form F. For a unit vector v, v.T F v = ||B v||^2.
    rng = np.random.default_rng(RANDOM_SEED)
    random_curvatures = np.empty(N_RANDOM_DIRECTIONS, dtype=np.float64)

    batch_size = 250
    for start in range(0, N_RANDOM_DIRECTIONS, batch_size):
        stop = min(start + batch_size, N_RANDOM_DIRECTIONS)
        V = rng.normal(size=(fisher_dimension, stop - start))
        V /= np.linalg.norm(V, axis=0, keepdims=True)
        BV = B @ V
        random_curvatures[start:stop] = np.sum(BV ** 2, axis=0)

    random_mean = float(np.mean(random_curvatures))
    random_std = float(np.std(random_curvatures, ddof=1))
    random_median = float(np.median(random_curvatures))
    random_p95 = float(np.quantile(random_curvatures, 0.95))

    if random_std > 0 and np.isfinite(fisher_theta_curvature):
        theta_z = float((fisher_theta_curvature - random_mean) / random_std)
    else:
        theta_z = np.nan

    if random_mean > 0 and np.isfinite(fisher_theta_curvature):
        theta_ratio = float(fisher_theta_curvature / random_mean)
    else:
        theta_ratio = np.nan

    # Recover leading feature-space eigenvectors only for interpretability.
    # v_j = B.T u_j / sqrt(lambda_j)
    top_k = min(10, numerical_rank)
    top_eigvecs = np.empty((fisher_dimension, top_k), dtype=np.float64)
    for j in range(top_k):
        lam = eigvals[j]
        top_eigvecs[:, j] = (B.T @ eigvecs_small[:, j]) / np.sqrt(lam)

    summary = {
        "n_samples": int(n_samples),
        "n_features_without_intercept": int(n_features),
        "fisher_dimension_with_intercept": int(fisher_dimension),
        "maximum_possible_rank": int(min(n_samples, fisher_dimension)),
        "numerical_rank": numerical_rank,
        "nullity": nullity,
        "null_fraction": float(nullity / fisher_dimension),
        "eigenvalue_tolerance": float(absolute_tol),
        "largest_eigenvalue": lambda_max,
        "smallest_positive_eigenvalue": smallest_positive,
        "condition_number_positive_subspace": condition_number,
        "trace_total_curvature": total_curvature,
        "effective_rank_entropy": effective_rank_value,
        "participation_ratio": participation_ratio_value,
        "learned_direction_curvature": fisher_theta_curvature,
        "random_direction_curvature_mean": random_mean,
        "random_direction_curvature_std": random_std,
        "random_direction_curvature_median": random_median,
        "random_direction_curvature_95th_percentile": random_p95,
        "learned_to_random_mean_ratio": theta_ratio,
        "learned_direction_random_z_score": theta_z,
        "mean_p_times_one_minus_p": float(np.mean(variance_weights)),
        "min_p_times_one_minus_p": float(np.min(variance_weights)),
        "max_p_times_one_minus_p": float(np.max(variance_weights)),
    }

    return {
        "summary": summary,
        "eigenvalues": eigvals,
        "positive_eigenvalues": positive_eigvals,
        "cumulative_fraction": cumulative_fraction,
        "random_curvatures": random_curvatures,
        "top_eigenvectors_with_intercept": top_eigvecs,
        "variance_weights": variance_weights,
        "small_gram_matrix": K,
    }


# =============================================================================
# Plotting
# =============================================================================

def save_plots(
    eigenvalues: np.ndarray,
    positive_eigenvalues: np.ndarray,
    cumulative_fraction: np.ndarray,
    random_curvatures: np.ndarray,
    learned_curvature: float,
    losses: np.ndarray,
):
    # 1. Full nonzero spectrum
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ranks = np.arange(1, len(positive_eigenvalues) + 1)
    ax.plot(ranks, positive_eigenvalues, marker="o", markersize=3, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Fisher eigenvalue")
    ax.set_title("Empirical Fisher-information spectrum")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fisher_eigenvalue_spectrum.png", dpi=300)
    fig.savefig(OUTDIR / "fisher_eigenvalue_spectrum.pdf")
    plt.close(fig)

    # 2. Cumulative curvature
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(1, len(cumulative_fraction) + 1)
    ax.plot(x, cumulative_fraction, linewidth=2)
    ax.axhline(0.90, linestyle="--", linewidth=1)
    ax.axhline(0.95, linestyle="--", linewidth=1)
    ax.axhline(0.99, linestyle="--", linewidth=1)
    ax.set_xlabel("Number of Fisher directions")
    ax.set_ylabel("Cumulative fraction of total curvature")
    ax.set_ylim(0, 1.02)
    ax.set_title("Concentration of Fisher curvature")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fisher_cumulative_curvature.png", dpi=300)
    fig.savefig(OUTDIR / "fisher_cumulative_curvature.pdf")
    plt.close(fig)

    # 3. Learned direction versus random directions
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist(random_curvatures, bins=50, alpha=0.8)
    ax.axvline(
        learned_curvature,
        linestyle="--",
        linewidth=2,
        label="Learned parameter direction",
    )
    ax.set_xlabel(r"Directional curvature $v^\top Fv$")
    ax.set_ylabel("Count")
    ax.set_title("Fisher curvature: learned direction vs random directions")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "learned_vs_random_fisher_curvature.png", dpi=300)
    fig.savefig(OUTDIR / "learned_vs_random_fisher_curvature.pdf")
    plt.close(fig)

    # 4. Adam loss
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(np.arange(1, len(losses) + 1), losses, linewidth=1.5)
    ax.set_xlabel("Adam epoch")
    ax.set_ylabel("Binary cross-entropy loss")
    ax.set_title("Full-data logistic-regression optimization")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "adam_training_loss.png", dpi=300)
    fig.savefig(OUTDIR / "adam_training_loss.pdf")
    plt.close(fig)

    # 5. Spectrum normalized by the largest eigenvalue
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    if len(positive_eigenvalues) > 0:
        normalized = positive_eigenvalues / positive_eigenvalues[0]
        ax.plot(
            np.arange(1, len(normalized) + 1),
            normalized,
            marker="o",
            markersize=3,
            linewidth=1.2,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel(r"$\lambda_j / \lambda_{\max}$")
    ax.set_title("Relative Fisher-information spectrum")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fisher_relative_spectrum.png", dpi=300)
    fig.savefig(OUTDIR / "fisher_relative_spectrum.pdf")
    plt.close(fig)


# =============================================================================
# Reporting
# =============================================================================

def number_directions_for_fraction(
    cumulative_fraction: np.ndarray,
    target: float,
) -> int:
    idx = np.searchsorted(cumulative_fraction, target, side="left")
    return int(min(idx + 1, len(cumulative_fraction)))


def write_summary(
    summary: dict,
    cumulative_fraction: np.ndarray,
    training_accuracy: float,
    final_loss: float,
):
    n90 = number_directions_for_fraction(cumulative_fraction, 0.90)
    n95 = number_directions_for_fraction(cumulative_fraction, 0.95)
    n99 = number_directions_for_fraction(cumulative_fraction, 0.99)

    summary = dict(summary)
    summary.update(
        {
            "directions_for_90_percent_curvature": n90,
            "directions_for_95_percent_curvature": n95,
            "directions_for_99_percent_curvature": n99,
            "full_data_training_accuracy": float(training_accuracy),
            "final_bce_loss": float(final_loss),
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "random_seed": RANDOM_SEED,
        }
    )

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "FISHER-INFORMATION ANALYSIS",
        "=" * 80,
        "",
        f"Samples:                              {summary['n_samples']}",
        f"Features, excluding intercept:        {summary['n_features_without_intercept']}",
        f"Parameter dimension, with intercept:  {summary['fisher_dimension_with_intercept']}",
        f"Maximum possible Fisher rank:         {summary['maximum_possible_rank']}",
        f"Numerical Fisher rank:                {summary['numerical_rank']}",
        f"Nullity:                              {summary['nullity']}",
        f"Null fraction:                        {summary['null_fraction']:.8f}",
        "",
        f"Largest eigenvalue:                   {summary['largest_eigenvalue']:.12g}",
        f"Smallest positive eigenvalue:         {summary['smallest_positive_eigenvalue']:.12g}",
        f"Condition number, positive subspace:  {summary['condition_number_positive_subspace']:.12g}",
        f"Trace / total curvature:              {summary['trace_total_curvature']:.12g}",
        f"Entropy effective rank:               {summary['effective_rank_entropy']:.6f}",
        f"Participation ratio:                  {summary['participation_ratio']:.6f}",
        "",
        f"Directions for 90% curvature:         {n90}",
        f"Directions for 95% curvature:         {n95}",
        f"Directions for 99% curvature:         {n99}",
        "",
        f"Learned-direction curvature:          {summary['learned_direction_curvature']:.12g}",
        f"Random-direction mean curvature:      {summary['random_direction_curvature_mean']:.12g}",
        f"Random-direction SD:                  {summary['random_direction_curvature_std']:.12g}",
        f"Learned/random curvature ratio:       {summary['learned_to_random_mean_ratio']:.6f}",
        f"Learned-direction random z-score:     {summary['learned_direction_random_z_score']:.6f}",
        "",
        f"Mean p(1-p):                          {summary['mean_p_times_one_minus_p']:.12g}",
        f"Min p(1-p):                           {summary['min_p_times_one_minus_p']:.12g}",
        f"Max p(1-p):                           {summary['max_p_times_one_minus_p']:.12g}",
        "",
        f"Full-data training accuracy:          {training_accuracy:.6f}",
        f"Final BCE loss:                       {final_loss:.12g}",
        "",
        "Interpretation:",
        "The empirical Fisher can have rank no larger than the number of samples.",
        "The remaining parameter-space directions have exactly zero empirical",
        "curvature before regularization and are therefore not identified by the data.",
        "The reported condition number concerns only the numerically positive-curvature",
        "subspace. L2 regularization adds curvature in all weight directions.",
    ]

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


# =============================================================================
# Main
# =============================================================================

def main():
    set_seeds(RANDOM_SEED)

    _, X_raw, labels, _ = load_composite_data()

    (
        X,
        y,
        labeled_mask,
        feature_mask,
        scaler,
    ) = prepare_binary_dataset(X_raw, labels)

    (
        weight,
        intercept,
        logits,
        probs,
        losses,
        training_accuracy,
    ) = fit_logistic_adam(X, y)

    fisher = fisher_analysis(
        X=X,
        probs=probs,
        weight=weight,
        intercept=intercept,
    )

    write_summary(
        summary=fisher["summary"],
        cumulative_fraction=fisher["cumulative_fraction"],
        training_accuracy=training_accuracy,
        final_loss=float(losses[-1]),
    )

    save_plots(
        eigenvalues=fisher["eigenvalues"],
        positive_eigenvalues=fisher["positive_eigenvalues"],
        cumulative_fraction=fisher["cumulative_fraction"],
        random_curvatures=fisher["random_curvatures"],
        learned_curvature=fisher["summary"]["learned_direction_curvature"],
        losses=losses,
    )

    np.savez_compressed(
        OUT_NPZ,
        weight=weight,
        intercept=np.array(intercept),
        logits=logits,
        probabilities=probs,
        labels_binary=y,
        labeled_mask=labeled_mask,
        retained_feature_mask=feature_mask,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        losses=losses,
        eigenvalues=fisher["eigenvalues"],
        positive_eigenvalues=fisher["positive_eigenvalues"],
        cumulative_fraction=fisher["cumulative_fraction"],
        random_curvatures=fisher["random_curvatures"],
        top_eigenvectors_with_intercept=fisher[
            "top_eigenvectors_with_intercept"
        ],
        variance_weights=fisher["variance_weights"],
        small_gram_matrix=fisher["small_gram_matrix"],
    )

    print("\nSaved outputs:")
    print(f"  {OUT_NPZ}")
    print(f"  {OUT_TXT}")
    print(f"  {OUT_JSON}")
    print(f"  {OUTDIR / 'fisher_eigenvalue_spectrum.png'}")
    print(f"  {OUTDIR / 'fisher_cumulative_curvature.png'}")
    print(f"  {OUTDIR / 'learned_vs_random_fisher_curvature.png'}")
    print(f"  {OUTDIR / 'adam_training_loss.png'}")
    print(f"  {OUTDIR / 'fisher_relative_spectrum.png'}")


if __name__ == "__main__":
    main()
