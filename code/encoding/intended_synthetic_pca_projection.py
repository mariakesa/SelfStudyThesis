#!/usr/bin/env python3
"""
Intended analysis: project real neural activity into a PCA basis learned from
synthetic four-class encoding predictions.

Pipeline:
    four-class labels
        -> cross-validated encoding/regression predictions X_synthetic
        -> PCA(X_synthetic)
        -> project X_real into the synthetic-derived PCA basis

In symbols, with matrices oriented as images × neurons for sklearn:
    pca_syn.fit(X_synthetic_img)
    synthetic_scores = pca_syn.transform(X_synthetic_img)
    real_scores      = pca_syn.transform(X_real_img)

This script deliberately does NOT fit PCA on X_real.

Expected input files:
    /home/maria/SelfStudyThesis/data/allen_natural_scenes_four_class_composite.npy
    /home/maria/SelfStudyThesis/data/synthetic_neural_activity_image_probs_loo.npy

Outputs:
    /home/maria/SelfStudyThesis/results/synthetic_pca_projection_intended/
        original_and_synthetic_in_synthetic_pca_space_2d.png
        original_and_synthetic_in_synthetic_pca_space_3d_plotly.html
        synthetic_pca_projection_scores.csv
        synthetic_pca_projection_results.npz

Run:
    python intended_synthetic_pca_projection.py

Optional:
    python intended_synthetic_pca_projection.py --show
    python intended_synthetic_pca_projection.py --open-html
"""

from __future__ import annotations

import argparse
from pathlib import Path
import webbrowser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


# =============================================================================
# Default paths
# =============================================================================

DATA_DIR = Path("/home/maria/SelfStudyThesis/data")

DEFAULT_REAL_COMPOSITE_PATH = DATA_DIR / "allen_natural_scenes_four_class_composite.npy"
DEFAULT_SYNTHETIC_PATH = DATA_DIR / "synthetic_neural_activity_image_probs_loo.npy"

DEFAULT_OUT_DIR = Path(
    "/home/maria/SelfStudyThesis/results/synthetic_pca_projection_intended"
)


# =============================================================================
# Settings
# =============================================================================

LABEL_NAMES = {
    -1: "unlabeled",
     0: "animals",
     1: "landscape",
     2: "plant",
     3: "man-made object",
}

PLOT_LABELS = [0, 1, 2, 3]

# The synthetic four-class prototype model has at most about 3 meaningful
# centered class-contrast dimensions, but we save more components for diagnostics.
DEFAULT_N_COMPONENTS = 10


# =============================================================================
# Loading and orientation
# =============================================================================

def load_real_composite(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Load real Allen composite.

    Returns:
        composite: original dictionary
        X_real_img: images × neurons
        labels: image labels, length n_images
    """
    composite = np.load(path, allow_pickle=True).item()

    X_real = np.asarray(composite["X"], dtype=np.float64)
    stimulus_metadata = composite["stimulus_metadata"]
    labels = np.asarray(stimulus_metadata["label"], dtype=np.int64).ravel()

    n_images = len(labels)

    # Real composite is commonly neurons × images.
    # sklearn PCA needs samples × features = images × neurons.
    if X_real.shape[1] == n_images:
        X_real_img = X_real.T
        orientation = "neurons × images -> transposed to images × neurons"
    elif X_real.shape[0] == n_images:
        X_real_img = X_real
        orientation = "already images × neurons"
    else:
        raise ValueError(
            "Cannot align real X with labels. Expected one axis to equal n_images.\n"
            f"  X_real shape: {X_real.shape}\n"
            f"  labels shape: {labels.shape}"
        )

    print(f"[real] loaded {path}")
    print(f"[real] raw shape: {X_real.shape}")
    print(f"[real] orientation: {orientation}")
    print(f"[real] aligned shape: {X_real_img.shape} images × neurons")

    return composite, X_real_img, labels


def load_synthetic_predictions(path: Path, n_images: int) -> np.ndarray:
    """
    Load synthetic LOO image probabilities.

    Returns:
        X_synth_img: images × neurons
    """
    X_synth = np.asarray(np.load(path, allow_pickle=True), dtype=np.float64)

    # Synthetic file is expected as neurons × images, but support either orientation.
    if X_synth.shape[1] == n_images:
        X_synth_img = X_synth.T
        orientation = "neurons × images -> transposed to images × neurons"
    elif X_synth.shape[0] == n_images:
        X_synth_img = X_synth
        orientation = "already images × neurons"
    else:
        raise ValueError(
            "Cannot align synthetic X with labels. Expected one axis to equal n_images.\n"
            f"  X_synth shape: {X_synth.shape}\n"
            f"  n_images: {n_images}"
        )

    print(f"[synthetic] loaded {path}")
    print(f"[synthetic] raw shape: {X_synth.shape}")
    print(f"[synthetic] orientation: {orientation}")
    print(f"[synthetic] aligned shape: {X_synth_img.shape} images × neurons")

    return X_synth_img


def validate_same_neural_space(X_real_img: np.ndarray, X_synth_img: np.ndarray) -> None:
    if X_real_img.shape != X_synth_img.shape:
        raise ValueError(
            "Real and synthetic matrices must be in the same image × neuron space.\n"
            f"  X_real_img:  {X_real_img.shape}\n"
            f"  X_synth_img: {X_synth_img.shape}"
        )


# =============================================================================
# Intended PCA analysis
# =============================================================================

def fit_pca_on_synthetic_project_real(
    X_real_img: np.ndarray,
    X_synth_img: np.ndarray,
    n_components: int,
) -> tuple[PCA, np.ndarray, np.ndarray]:
    """
    This is the intended analysis.

    Critical data flow:
        pca_syn.fit(X_synth_img)
        synthetic_scores = pca_syn.transform(X_synth_img)
        real_scores      = pca_syn.transform(X_real_img)

    PCA is NOT fit on X_real_img.
    """
    max_components = min(X_synth_img.shape[0], X_synth_img.shape[1])
    n_components = min(n_components, max_components)

    pca_syn = PCA(n_components=n_components)

    # -------------------------------------------------------------------------
    # CRITICAL LINE 1:
    # Learn the PCA basis from synthetic encoding predictions only.
    # -------------------------------------------------------------------------
    synthetic_scores = pca_syn.fit_transform(X_synth_img)

    # -------------------------------------------------------------------------
    # CRITICAL LINE 2:
    # Project real measured neural activity into that synthetic-derived basis.
    # -------------------------------------------------------------------------
    real_scores = pca_syn.transform(X_real_img)

    return pca_syn, real_scores, synthetic_scores


# =============================================================================
# Plotting
# =============================================================================

def plot_2d(
    real_scores: np.ndarray,
    synthetic_scores: np.ndarray,
    labels: np.ndarray,
    explained_variance_ratio: np.ndarray,
    out_png: Path,
    show: bool = False,
) -> None:
    pc1 = 100 * explained_variance_ratio[0]
    pc2 = 100 * explained_variance_ratio[1]

    plt.figure(figsize=(11, 8.5))

    for label_value in PLOT_LABELS:
        mask = labels == label_value
        label_name = LABEL_NAMES[label_value]

        plt.scatter(
            real_scores[mask, 0],
            real_scores[mask, 1],
            marker="o",
            alpha=0.70,
            label=f"original {label_name}",
        )

        plt.scatter(
            synthetic_scores[mask, 0],
            synthetic_scores[mask, 1],
            marker="x",
            alpha=0.95,
            label=f"synthetic {label_name}",
        )

    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel(f"Synthetic PC1 score ({pc1:.1f}% EVR)")
    plt.ylabel(f"Synthetic PC2 score ({pc2:.1f}% EVR)")
    plt.title("Original neural responses projected into synthetic PCA space")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)

    if show:
        plt.show()
    else:
        plt.close()


def make_plotly_3d(
    real_scores: np.ndarray,
    synthetic_scores: np.ndarray,
    labels: np.ndarray,
    image_indices: np.ndarray,
    explained_variance_ratio: np.ndarray,
    out_html: Path,
) -> None:
    if go is None:
        print("[plotly] plotly is not installed; skipping 3D HTML plot.")
        return

    pc1 = 100 * explained_variance_ratio[0]
    pc2 = 100 * explained_variance_ratio[1]
    pc3 = 100 * explained_variance_ratio[2]

    fig = go.Figure()

    def hover_text(source: str, idxs: np.ndarray, labs: np.ndarray) -> list[str]:
        rows = []
        for image_index, lab in zip(idxs, labs):
            rows.append(
                f"{source}<br>"
                f"image_index: {int(image_index)}<br>"
                f"label: {int(lab)}<br>"
                f"label_name: {LABEL_NAMES[int(lab)]}"
            )
        return rows

    for label_value in PLOT_LABELS:
        mask = labels == label_value
        label_name = LABEL_NAMES[label_value]

        fig.add_trace(
            go.Scatter3d(
                x=real_scores[mask, 0],
                y=real_scores[mask, 1],
                z=real_scores[mask, 2],
                mode="markers",
                name=f"original {label_name}",
                text=np.asarray(hover_text("original", image_indices, labels))[mask],
                hovertemplate=(
                    "%{text}<br>"
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}<br>"
                    "PC3: %{z:.3f}<extra></extra>"
                ),
                marker=dict(size=5, opacity=0.70, symbol="circle"),
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=synthetic_scores[mask, 0],
                y=synthetic_scores[mask, 1],
                z=synthetic_scores[mask, 2],
                mode="markers",
                name=f"synthetic {label_name}",
                text=np.asarray(hover_text("synthetic", image_indices, labels))[mask],
                hovertemplate=(
                    "%{text}<br>"
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}<br>"
                    "PC3: %{z:.3f}<extra></extra>"
                ),
                marker=dict(size=7, opacity=0.95, symbol="x"),
            )
        )

    fig.update_layout(
        title=(
            "Original neural responses projected into synthetic PCA space, 3D<br>"
            "<sup>PCA basis fit on synthetic four-class encoding predictions only</sup>"
        ),
        scene=dict(
            xaxis_title=f"Synthetic PC1 ({pc1:.1f}% EVR)",
            yaxis_title=f"Synthetic PC2 ({pc2:.1f}% EVR)",
            zaxis_title=f"Synthetic PC3 ({pc3:.1f}% EVR)",
        ),
        legend=dict(title="Source and class"),
        width=1100,
        height=850,
        margin=dict(l=0, r=0, b=0, t=80),
    )

    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)


# =============================================================================
# Saving
# =============================================================================

def save_scores_csv(
    real_scores: np.ndarray,
    synthetic_scores: np.ndarray,
    labels: np.ndarray,
    image_indices: np.ndarray,
    out_csv: Path,
) -> None:
    rows = []
    n_components = real_scores.shape[1]

    for row_idx, image_index in enumerate(image_indices):
        lab = int(labels[row_idx])
        label_name = LABEL_NAMES[lab]

        real_row = {
            "source": "original",
            "image_index": int(image_index),
            "label": lab,
            "label_name": label_name,
        }
        synth_row = {
            "source": "synthetic",
            "image_index": int(image_index),
            "label": lab,
            "label_name": label_name,
        }

        for k in range(n_components):
            real_row[f"pc{k + 1}"] = float(real_scores[row_idx, k])
            synth_row[f"pc{k + 1}"] = float(synthetic_scores[row_idx, k])

        rows.append(real_row)
        rows.append(synth_row)

    pd.DataFrame(rows).to_csv(out_csv, index=False)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit PCA on synthetic encoding predictions and project real neural activity into that basis."
    )
    parser.add_argument("--real-composite", type=Path, default=DEFAULT_REAL_COMPOSITE_PATH)
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-components", type=int, default=DEFAULT_N_COMPONENTS)
    parser.add_argument("--show", action="store_true", help="Call plt.show() after saving the 2D PNG.")
    parser.add_argument("--open-html", action="store_true", help="Open the 3D Plotly HTML in a browser.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    out_png = args.out_dir / "original_and_synthetic_in_synthetic_pca_space_2d.png"
    out_html = args.out_dir / "original_and_synthetic_in_synthetic_pca_space_3d_plotly.html"
    out_csv = args.out_dir / "synthetic_pca_projection_scores.csv"
    out_npz = args.out_dir / "synthetic_pca_projection_results.npz"

    print("=" * 88)
    print("INTENDED SYNTHETIC PCA PROJECTION ANALYSIS")
    print("=" * 88)
    print("Pipeline:")
    print("  pca_syn.fit(X_synthetic)")
    print("  real_scores = pca_syn.transform(X_real)")
    print("  PCA is never fit on X_real in this script.")
    print()

    _, X_real_img, labels = load_real_composite(args.real_composite)
    X_synth_img = load_synthetic_predictions(args.synthetic, n_images=len(labels))
    validate_same_neural_space(X_real_img, X_synth_img)

    keep_mask = labels >= 0
    image_indices = np.arange(len(labels))[keep_mask]
    labels_kept = labels[keep_mask]
    X_real_kept = X_real_img[keep_mask]
    X_synth_kept = X_synth_img[keep_mask]

    print()
    print("After removing unlabeled images:")
    print(f"  X_real_kept:  {X_real_kept.shape} images × neurons")
    print(f"  X_synth_kept: {X_synth_kept.shape} images × neurons")
    print(f"  labels_kept:  {labels_kept.shape}")
    print()

    pca_syn, real_scores, synthetic_scores = fit_pca_on_synthetic_project_real(
        X_real_img=X_real_kept,
        X_synth_img=X_synth_kept,
        n_components=args.n_components,
    )

    evr = pca_syn.explained_variance_ratio_

    print("Synthetic PCA explained variance ratio:")
    for i, value in enumerate(evr, start=1):
        print(f"  synthetic PC{i:02d}: {value:.6f} ({100 * value:.2f}%)")

    plot_2d(
        real_scores=real_scores,
        synthetic_scores=synthetic_scores,
        labels=labels_kept,
        explained_variance_ratio=evr,
        out_png=out_png,
        show=args.show,
    )

    if real_scores.shape[1] >= 3:
        make_plotly_3d(
            real_scores=real_scores,
            synthetic_scores=synthetic_scores,
            labels=labels_kept,
            image_indices=image_indices,
            explained_variance_ratio=evr,
            out_html=out_html,
        )
    else:
        print("[plotly] fewer than 3 components available; skipping 3D plot.")

    save_scores_csv(
        real_scores=real_scores,
        synthetic_scores=synthetic_scores,
        labels=labels_kept,
        image_indices=image_indices,
        out_csv=out_csv,
    )

    np.savez_compressed(
        out_npz,
        real_scores_in_synthetic_pca=real_scores,
        synthetic_scores_in_synthetic_pca=synthetic_scores,
        labels=labels_kept,
        image_indices=image_indices,
        explained_variance_ratio=evr,
        synthetic_pca_components=pca_syn.components_,
        synthetic_pca_mean=pca_syn.mean_,
        real_composite_path=str(args.real_composite),
        synthetic_path=str(args.synthetic),
        analysis_description=(
            "PCA was fit on X_synthetic only; X_real was projected into that synthetic PCA basis."
        ),
    )

    print()
    print("Saved outputs:")
    print(f"  2D PNG: {out_png}")
    if go is not None and real_scores.shape[1] >= 3:
        print(f"  3D HTML: {out_html}")
    print(f"  CSV:    {out_csv}")
    print(f"  NPZ:    {out_npz}")

    if args.open_html and go is not None and out_html.exists():
        webbrowser.open(out_html.as_uri())


if __name__ == "__main__":
    main()
