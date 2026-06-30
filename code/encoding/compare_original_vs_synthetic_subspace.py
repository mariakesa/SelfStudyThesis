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
    """Compare real and synthetic PCA geometry."""
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
