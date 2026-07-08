#!/usr/bin/env python3
"""
Interactive 3D Plotly plot:
    Original and synthetic images in synthetic PCA space.

PCA is fit on the synthetic neural activity matrix, then both the synthetic
matrix and the original measured matrix are projected into that same synthetic
PC basis.

Expected inputs:
    /home/maria/SelfStudyThesis/data/allen_natural_scenes_four_class_composite.npy
    /home/maria/SelfStudyThesis/data/synthetic_neural_activity_image_probs_loo.npy

Outputs:
    /home/maria/SelfStudyThesis/results/original_vs_synthetic_four_class_subspace/
        original_and_synthetic_in_synthetic_pca_space_3d_plotly.html
        synthetic_pca_3d_scores.npz
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.graph_objects as go


# =============================================================================
# Paths
# =============================================================================

DEFAULT_REAL_COMPOSITE_PATH = Path(
    "/home/maria/SelfStudyThesis/data/allen_natural_scenes_four_class_composite.npy"
)

DEFAULT_SYNTHETIC_PATH = Path(
    "/home/maria/SelfStudyThesis/data/synthetic_neural_activity_image_probs_loo.npy"
)

DEFAULT_OUT_DIR = Path(
    "/home/maria/SelfStudyThesis/results/original_vs_synthetic_four_class_subspace"
)

LABEL_NAMES = {
    -1: "unlabeled",
     0: "animals",
     1: "landscape",
     2: "plant",
     3: "man-made object",
}

# Fixed for consistency with earlier 2D plots.
# Feel free to change these if you want a different palette.
LABEL_COLORS = {
    0: "#1f77b4",  # animals
    1: "#2ca02c",  # landscape
    2: "#9467bd",  # plant
    3: "#e377c2",  # man-made object
}

SYNTHETIC_COLORS = {
    0: "#ff7f0e",  # synthetic animals
    1: "#d62728",  # synthetic landscape
    2: "#8c564b",  # synthetic plant
    3: "#7f7f7f",  # synthetic man-made object
}


# =============================================================================
# Loading helpers
# =============================================================================

def load_real_composite(path: Path):
    data = np.load(path, allow_pickle=True).item()

    X_real = np.asarray(data["X"], dtype=np.float64)
    stimulus_metadata = data["stimulus_metadata"]
    labels = np.asarray(stimulus_metadata["label"], dtype=np.int64).ravel()

    # Original composite is usually neurons × images.
    if X_real.shape[1] == len(labels):
        pass
    elif X_real.shape[0] == len(labels):
        X_real = X_real.T
    else:
        raise ValueError(
            f"Could not align real matrix with labels. X_real={X_real.shape}, "
            f"labels={labels.shape}"
        )

    return data, X_real, labels


def load_synthetic(path: Path, n_images: int):
    X_synth = np.asarray(np.load(path, allow_pickle=True), dtype=np.float64)

    # Desired orientation is neurons × images.
    if X_synth.shape[1] == n_images:
        pass
    elif X_synth.shape[0] == n_images:
        X_synth = X_synth.T
    else:
        raise ValueError(
            f"Could not align synthetic matrix with images. X_synth={X_synth.shape}, "
            f"n_images={n_images}"
        )

    return X_synth


def assert_same_space(X_real: np.ndarray, X_synth: np.ndarray):
    if X_real.shape != X_synth.shape:
        raise ValueError(
            "Real and synthetic matrices must have the same shape after alignment.\n"
            f"  X_real:  {X_real.shape}\n"
            f"  X_synth: {X_synth.shape}"
        )


# =============================================================================
# PCA and plotting
# =============================================================================

def fit_synthetic_pca_and_project(X_real: np.ndarray, X_synth: np.ndarray):
    """
    X_real, X_synth:
        neurons × images

    PCA expects samples × features, so transpose to:
        images × neurons
    """
    X_real_img = X_real.T
    X_synth_img = X_synth.T

    pca = PCA(n_components=3)
    synth_scores = pca.fit_transform(X_synth_img)
    real_scores = pca.transform(X_real_img)

    return pca, real_scores, synth_scores


def make_hover_text(kind: str, labels: np.ndarray):
    text = []
    for i, lab in enumerate(labels):
        label_name = LABEL_NAMES.get(int(lab), "UNKNOWN")
        text.append(
            f"{kind}<br>"
            f"image_index: {i}<br>"
            f"label: {int(lab)}<br>"
            f"label_name: {label_name}"
        )
    return text


def add_class_trace(
    fig: go.Figure,
    scores: np.ndarray,
    labels: np.ndarray,
    label_value: int,
    kind: str,
    color: str,
    symbol: str,
    size: int,
    opacity: float,
):
    mask = labels == label_value
    label_name = LABEL_NAMES[int(label_value)]

    fig.add_trace(
        go.Scatter3d(
            x=scores[mask, 0],
            y=scores[mask, 1],
            z=scores[mask, 2],
            mode="markers",
            name=f"{kind} {label_name}",
            text=np.asarray(make_hover_text(kind, labels))[mask],
            hovertemplate=(
                "%{text}<br>"
                "PC1: %{x:.3f}<br>"
                "PC2: %{y:.3f}<br>"
                "PC3: %{z:.3f}"
                "<extra></extra>"
            ),
            marker=dict(
                size=size,
                color=color,
                symbol=symbol,
                opacity=opacity,
                line=dict(width=0.5, color="black"),
            ),
        )
    )


def make_plotly_figure(
    real_scores: np.ndarray,
    synth_scores: np.ndarray,
    labels: np.ndarray,
    explained: np.ndarray,
):
    fig = go.Figure()

    classes = [0, 1, 2, 3]

    # Original measured data: circles.
    for cls in classes:
        add_class_trace(
            fig=fig,
            scores=real_scores,
            labels=labels,
            label_value=cls,
            kind="original",
            color=LABEL_COLORS[cls],
            symbol="circle",
            size=5,
            opacity=0.70,
        )

    # Synthetic data: crosses.
    for cls in classes:
        add_class_trace(
            fig=fig,
            scores=synth_scores,
            labels=labels,
            label_value=cls,
            kind="synthetic",
            color=SYNTHETIC_COLORS[cls],
            symbol="x",
            size=6,
            opacity=0.95,
        )

    pc1 = 100 * explained[0]
    pc2 = 100 * explained[1]
    pc3 = 100 * explained[2]

    fig.update_layout(
        title=(
            "Original and synthetic images in synthetic PCA space, 3D<br>"
            f"<sup>PCA fit on synthetic data only. EVR: "
            f"PC1={pc1:.1f}%, PC2={pc2:.1f}%, PC3={pc3:.1f}%</sup>"
        ),
        scene=dict(
            xaxis_title=f"Synthetic PC1 ({pc1:.1f}% EVR)",
            yaxis_title=f"Synthetic PC2 ({pc2:.1f}% EVR)",
            zaxis_title=f"Synthetic PC3 ({pc3:.1f}% EVR)",
            xaxis=dict(showbackground=True, zeroline=True),
            yaxis=dict(showbackground=True, zeroline=True),
            zaxis=dict(showbackground=True, zeroline=True),
        ),
        legend=dict(
            title="Data source and class",
            itemsizing="constant",
        ),
        width=1100,
        height=850,
        margin=dict(l=0, r=0, b=0, t=80),
    )

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-composite", type=Path, default=DEFAULT_REAL_COMPOSITE_PATH)
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--open", action="store_true", help="Open the HTML plot in browser if possible.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    out_html = args.out_dir / "original_and_synthetic_in_synthetic_pca_space_3d_plotly.html"
    out_npz = args.out_dir / "synthetic_pca_3d_scores.npz"
    out_csv = args.out_dir / "synthetic_pca_3d_scores.csv"

    print("=" * 80)
    print("3D Plotly synthetic PCA space")
    print("=" * 80)
    print(f"Real composite: {args.real_composite}")
    print(f"Synthetic:      {args.synthetic}")
    print(f"Output HTML:    {out_html}")
    print()

    _, X_real, labels = load_real_composite(args.real_composite)
    X_synth = load_synthetic(args.synthetic, n_images=len(labels))
    assert_same_space(X_real, X_synth)

    keep_mask = labels >= 0
    X_real = X_real[:, keep_mask]
    X_synth = X_synth[:, keep_mask]
    labels_kept = labels[keep_mask]
    image_indices = np.arange(len(labels))[keep_mask]

    print(f"X_real shape, neurons × kept images:  {X_real.shape}")
    print(f"X_synth shape, neurons × kept images: {X_synth.shape}")
    print()

    pca, real_scores, synth_scores = fit_synthetic_pca_and_project(X_real, X_synth)
    explained = pca.explained_variance_ratio_

    fig = make_plotly_figure(
        real_scores=real_scores,
        synth_scores=synth_scores,
        labels=labels_kept,
        explained=explained,
    )

    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)

    rows = []
    for row_idx, image_index in enumerate(image_indices):
        lab = int(labels_kept[row_idx])
        label_name = LABEL_NAMES[lab]
        rows.append({
            "source": "original",
            "image_index": int(image_index),
            "label": lab,
            "label_name": label_name,
            "pc1": float(real_scores[row_idx, 0]),
            "pc2": float(real_scores[row_idx, 1]),
            "pc3": float(real_scores[row_idx, 2]),
        })
        rows.append({
            "source": "synthetic",
            "image_index": int(image_index),
            "label": lab,
            "label_name": label_name,
            "pc1": float(synth_scores[row_idx, 0]),
            "pc2": float(synth_scores[row_idx, 1]),
            "pc3": float(synth_scores[row_idx, 2]),
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    np.savez_compressed(
        out_npz,
        real_scores=real_scores,
        synth_scores=synth_scores,
        labels=labels_kept,
        image_indices=image_indices,
        explained_variance_ratio=explained,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
    )

    print("Explained variance ratio of synthetic PCA basis:")
    for i, evr in enumerate(explained, start=1):
        print(f"  PC{i}: {evr:.6f} ({100 * evr:.2f}%)")

    print()
    print("Saved:")
    print(f"  {out_html}")
    print(f"  {out_npz}")
    print(f"  {out_csv}")

    if args.open:
        fig.show()


if __name__ == "__main__":
    main()
