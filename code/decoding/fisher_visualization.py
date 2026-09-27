#!/usr/bin/env python3

"""
Visualize Fisher eigenvectors for the Allen animacy analysis.

This fits the same model used in the cross-validated Fisher analysis,
but on the FULL dataset for visualization purposes only.

IMPORTANT:
    This full-data Fisher basis should NOT be used for reporting
    cross-validated decoding performance. It is purely for inspecting
    what the learned Fisher directions look like.

Outputs:
    1. Fisher eigenvalue spectrum
    2. Heatmap of leading Fisher eigenvectors
    3. Sorted neuron loadings for each leading direction
    4. Histogram of loadings
    5. Cumulative loading-energy curves
    6. NPZ containing the Fisher directions
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# =============================================================================
# Paths
# =============================================================================

COMPOSITE_PATH = Path(
    "/home/maria/SelfStudyThesis/data/"
    "allen_natural_scenes_four_class_composite.npy"
)

OUTDIR = Path(
    "/home/maria/SelfStudyThesis/results/"
    "fisher_subspace_selection_animals_vs_rest/"
    "fisher_vector_visualization"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Settings
# =============================================================================

RANDOM_STATE = 123
C_FISHER_MODEL = 1.0
VAR_EPS = 1e-12

# How many Fisher vectors to inspect.
N_DIRECTIONS = 10


# =============================================================================
# Load data
# =============================================================================

def load_data():

    data = np.load(
        COMPOSITE_PATH,
        allow_pickle=True,
    ).item()

    X = np.asarray(
        data["X"],
        dtype=np.float64,
    )

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
            f"Cannot align X={X.shape} "
            f"with labels={labels.shape}"
        )

    # Remove unlabeled images.
    labeled = labels != -1

    X = X[labeled]
    labels = labels[labeled]

    # animals = 1
    # everything else = 0
    y = (labels == 0).astype(np.int64)

    # Same global feature filtering as original script.
    finite = np.all(
        np.isfinite(X),
        axis=0,
    )

    variances = np.var(
        X,
        axis=0,
    )

    nonconstant = variances > VAR_EPS

    keep = finite & nonconstant

    X = X[:, keep]

    print(f"Images:   {X.shape[0]}")
    print(f"Neurons:  {X.shape[1]}")
    print(f"Animals:  {np.sum(y == 1)}")
    print(f"Other:    {np.sum(y == 0)}")

    return X, y, keep


# =============================================================================
# Fisher basis
# =============================================================================

def compute_fisher_basis(X_z, y):

    model = LogisticRegression(
        penalty="l2",
        C=C_FISHER_MODEL,
        solver="liblinear",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_z,
        y,
    )

    probs = model.predict_proba(
        X_z
    )[:, 1]

    # Logistic Fisher weights.
    w = probs * (1.0 - probs)

    n = X_z.shape[0]

    # B.T @ B = Fisher matrix.
    B = (
        np.sqrt(w / n)[:, None]
        * X_z
    )

    # B = U S V.T
    U, s, Vt = np.linalg.svd(
        B,
        full_matrices=False,
    )

    eigenvalues = s ** 2

    # Columns = Fisher eigenvectors.
    V = Vt.T

    return (
        V,
        eigenvalues,
        probs,
        model,
    )


# =============================================================================
# Plot eigenvalue spectrum
# =============================================================================

def plot_spectrum(eigenvalues):

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.semilogy(
        np.arange(1, len(eigenvalues) + 1),
        eigenvalues,
        marker=".",
    )

    ax.set_xlabel("Fisher direction")
    ax.set_ylabel("Fisher eigenvalue")
    ax.set_title("Fisher information spectrum")

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        OUTDIR / "01_fisher_spectrum.png",
        dpi=300,
    )

    plt.close(fig)


# =============================================================================
# Heatmap
# =============================================================================

def plot_vector_heatmap(V, n_directions):

    V_show = V[:, :n_directions]

    # ------------------------------------------------------------
    # A raw neuron-order heatmap is usually not very informative
    # because neuron index itself has no geometric meaning.
    #
    # Instead sort neurons according to Fisher direction 1.
    # Then ask how all other Fisher vectors look in that ordering.
    # ------------------------------------------------------------

    order = np.argsort(V_show[:, 0])

    sorted_V = V_show[order]

    # Robust color scale so a few huge coefficients do not dominate.
    vmax = np.percentile(
        np.abs(sorted_V),
        99.5,
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    im = ax.imshow(
        sorted_V.T,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xlabel(
        "Neurons, sorted by Fisher direction 1"
    )

    ax.set_ylabel(
        "Fisher direction"
    )

    ax.set_yticks(
        np.arange(n_directions)
    )

    ax.set_yticklabels(
        np.arange(1, n_directions + 1)
    )

    ax.set_title(
        "Leading Fisher population directions"
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        "Eigenvector loading"
    )

    fig.tight_layout()

    fig.savefig(
        OUTDIR / "02_fisher_vectors_heatmap.png",
        dpi=300,
    )

    plt.close(fig)


# =============================================================================
# Sorted loading curves
# =============================================================================

def plot_sorted_loadings(V, eigenvalues, n_directions):

    for j in range(n_directions):

        v = V[:, j]

        ordered = np.sort(v)

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.plot(
            ordered,
            linewidth=1,
        )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.set_xlabel(
            "Neuron rank"
        )

        ax.set_ylabel(
            "Fisher eigenvector loading"
        )

        ax.set_title(
            f"Fisher direction {j + 1} "
            f"(eigenvalue={eigenvalues[j]:.3e})"
        )

        ax.grid(
            True,
            alpha=0.2,
        )

        fig.tight_layout()

        fig.savefig(
            OUTDIR
            / f"03_direction_{j+1:02d}_sorted_loadings.png",
            dpi=300,
        )

        plt.close(fig)


# =============================================================================
# Loading histograms
# =============================================================================

def plot_loading_histograms(V, n_directions):

    for j in range(n_directions):

        v = V[:, j]

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.hist(
            v,
            bins=100,
        )

        ax.set_xlabel(
            "Neuron loading"
        )

        ax.set_ylabel(
            "Number of neurons"
        )

        ax.set_title(
            f"Fisher direction {j + 1}: loading distribution"
        )

        fig.tight_layout()

        fig.savefig(
            OUTDIR
            / f"04_direction_{j+1:02d}_histogram.png",
            dpi=300,
        )

        plt.close(fig)


# =============================================================================
# Loading concentration
# =============================================================================

def plot_loading_energy(V, n_directions):

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    n_neurons = V.shape[0]

    x = (
        np.arange(1, n_neurons + 1)
        / n_neurons
        * 100
    )

    for j in range(n_directions):

        v = V[:, j]

        # Eigenvectors have ||v||_2 = 1.
        # Therefore v_i^2 is the fraction of squared loading
        # associated with neuron i.

        energy = v ** 2

        order = np.argsort(
            energy
        )[::-1]

        cumulative = np.cumsum(
            energy[order]
        )

        ax.plot(
            x,
            cumulative,
            label=f"F{j + 1}",
        )

    ax.set_xlabel(
        "Top neurons retained (%)"
    )

    ax.set_ylabel(
        "Cumulative squared loading"
    )

    ax.set_title(
        "How distributed are the Fisher directions?"
    )

    ax.axhline(
        0.5,
        linestyle="--",
        alpha=0.5,
    )

    ax.axhline(
        0.9,
        linestyle="--",
        alpha=0.5,
    )

    ax.set_ylim(
        0,
        1.01,
    )

    ax.legend(
        ncol=2,
    )

    ax.grid(
        True,
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        OUTDIR / "05_fisher_loading_concentration.png",
        dpi=300,
    )

    plt.close(fig)


# =============================================================================
# Top-neuron report
# =============================================================================

def report_top_neurons(
    V,
    eigenvalues,
    n_directions,
    top_n=20,
):

    lines = []

    for j in range(n_directions):

        v = V[:, j]

        order = np.argsort(
            np.abs(v)
        )[::-1]

        lines.append(
            "=" * 80
        )

        lines.append(
            f"FISHER DIRECTION {j + 1}"
        )

        lines.append(
            f"Eigenvalue: {eigenvalues[j]:.8e}"
        )

        lines.append("")
        lines.append(
            "Rank   retained_neuron_index      loading       abs_loading"
        )
        lines.append(
            "-" * 65
        )

        for rank, idx in enumerate(
            order[:top_n],
            start=1,
        ):

            lines.append(
                f"{rank:4d}"
                f"{idx:24d}"
                f"{v[idx]:13.6e}"
                f"{abs(v[idx]):13.6e}"
            )

        lines.append("")

    path = (
        OUTDIR
        / "06_top_neurons_per_fisher_direction.txt"
    )

    path.write_text(
        "\n".join(lines)
    )


# =============================================================================
# Main
# =============================================================================

def main():

    X, y, feature_mask = load_data()

    # -------------------------------------------------------------
    # Standardize exactly as in Fisher CV analysis
    # -------------------------------------------------------------

    scaler = StandardScaler()

    X_z = scaler.fit_transform(X)

    # -------------------------------------------------------------
    # Full-data Fisher basis
    # -------------------------------------------------------------

    (
        V,
        eigenvalues,
        probs,
        model,
    ) = compute_fisher_basis(
        X_z,
        y,
    )

    print()
    print("=" * 80)
    print("Fisher basis")
    print("=" * 80)

    print(
        f"V shape:          {V.shape}"
    )

    print(
        f"Fisher rank:      {len(eigenvalues)}"
    )

    print(
        f"Top eigenvalue:   {eigenvalues[0]:.6e}"
    )

    print(
        f"Bottom eigenvalue:{eigenvalues[-1]:.6e}"
    )

    n_directions = min(
        N_DIRECTIONS,
        V.shape[1],
    )

    # -------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------

    plot_spectrum(
        eigenvalues
    )

    plot_vector_heatmap(
        V,
        n_directions,
    )

    plot_sorted_loadings(
        V,
        eigenvalues,
        n_directions,
    )

    plot_loading_histograms(
        V,
        n_directions,
    )

    plot_loading_energy(
        V,
        n_directions,
    )

    report_top_neurons(
        V,
        eigenvalues,
        n_directions,
    )

    # -------------------------------------------------------------
    # Save actual vectors
    # -------------------------------------------------------------

    np.savez_compressed(
        OUTDIR / "fisher_vectors_full_data.npz",

        fisher_vectors=
            V[:, :n_directions],

        fisher_eigenvalues=
            eigenvalues,

        logistic_weights=
            model.coef_.ravel(),

        logistic_intercept=
            model.intercept_,

        probabilities=
            probs,

        scaler_mean=
            scaler.mean_,

        scaler_scale=
            scaler.scale_,

        retained_feature_mask=
            feature_mask,
    )

    print()
    print("Saved visualizations to:")
    print(f"  {OUTDIR}")


if __name__ == "__main__":
    main()