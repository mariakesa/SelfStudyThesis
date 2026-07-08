#!/usr/bin/env python3
"""
Interactive 3D Plotly plot inherited from the original unmorphed figure.

The PCA endpoint uses the exact same PCA fit, coordinates, traces, marker sizes,
colors, legend, scene styling, dimensions, and margins as the original script.
The only additions are:
    1. a fixed-seed random projection of the same data,
    2. Plotly animation frames interpolating between random and PCA coordinates,
    3. Random -> PCA, PCA -> Random, and Pause buttons.

Expected inputs:
    /home/maria/SelfStudyThesis/data/allen_natural_scenes_four_class_composite.npy
    /home/maria/SelfStudyThesis/data/synthetic_neural_activity_image_probs_loo.npy
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

LABEL_COLORS = {
    0: "#1f77b4",
    1: "#2ca02c",
    2: "#9467bd",
    3: "#e377c2",
}

SYNTHETIC_COLORS = {
    0: "#ff7f0e",
    1: "#d62728",
    2: "#8c564b",
    3: "#7f7f7f",
}

CLASSES = [0, 1, 2, 3]


# =============================================================================
# Loading helpers -- unchanged from the original script
# =============================================================================

def load_real_composite(path: Path):
    data = np.load(path, allow_pickle=True).item()

    X_real = np.asarray(data["X"], dtype=np.float64)
    stimulus_metadata = data["stimulus_metadata"]
    labels = np.asarray(stimulus_metadata["label"], dtype=np.int64).ravel()

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
# PCA and random projection
# =============================================================================

def fit_synthetic_pca_and_project(X_real: np.ndarray, X_synth: np.ndarray):
    """Exactly the original PCA computation."""
    X_real_img = X_real.T
    X_synth_img = X_synth.T

    pca = PCA(n_components=3)
    synth_scores = pca.fit_transform(X_synth_img)
    real_scores = pca.transform(X_real_img)

    return pca, real_scores, synth_scores


def make_random_projection(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    seed: int,
):
    """Project both matrices through one shared random orthonormal basis."""
    X_real_img = X_real.T
    X_synth_img = X_synth.T

    rng = np.random.default_rng(seed)
    gaussian_basis = rng.normal(size=(X_synth_img.shape[1], 3))
    random_basis, _ = np.linalg.qr(gaussian_basis)

    synthetic_mean = X_synth_img.mean(axis=0)
    random_real = (X_real_img - synthetic_mean) @ random_basis
    random_synth = (X_synth_img - synthetic_mean) @ random_basis

    return random_basis, random_real, random_synth


def align_random_to_pca_display(
    random_real: np.ndarray,
    random_synth: np.ndarray,
    pca_real: np.ndarray,
    pca_synth: np.ndarray,
):
    """
    Move only the random projection into the PCA display coordinate system.

    The PCA arrays are never modified. The random cloud receives one pooled
    similarity transform: translation, orthogonal rotation/reflection, and one
    isotropic scale factor. This keeps the random geometry intact while avoiding
    a dramatic jump in plot scale during the morph.
    """
    source = np.vstack([random_real, random_synth])
    target = np.vstack([pca_real, pca_synth])

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source0 = source - source_center
    target0 = target - target_center

    u, singular_values, vt = np.linalg.svd(source0.T @ target0)
    rotation = u @ vt
    scale = float(np.sum(singular_values) / np.sum(source0 ** 2))

    def transform(scores: np.ndarray) -> np.ndarray:
        return (scores - source_center) @ rotation * scale + target_center

    return transform(random_real), transform(random_synth)


# =============================================================================
# Original plotting helpers, with only frame support added
# =============================================================================

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
    """Original trace construction, unchanged."""
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


def add_all_traces(
    fig: go.Figure,
    real_scores: np.ndarray,
    synth_scores: np.ndarray,
    labels: np.ndarray,
):
    for cls in CLASSES:
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

    for cls in CLASSES:
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


def frame_trace_updates(
    real_scores: np.ndarray,
    synth_scores: np.ndarray,
    labels: np.ndarray,
):
    updates = []
    for cls in CLASSES:
        mask = labels == cls
        updates.append(go.Scatter3d(
            x=real_scores[mask, 0],
            y=real_scores[mask, 1],
            z=real_scores[mask, 2],
        ))
    for cls in CLASSES:
        mask = labels == cls
        updates.append(go.Scatter3d(
            x=synth_scores[mask, 0],
            y=synth_scores[mask, 1],
            z=synth_scores[mask, 2],
        ))
    return updates


def smooth_interpolate(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    s = t * t * (3.0 - 2.0 * t)
    return (1.0 - s) * a + s * b


def make_frames(
    random_real: np.ndarray,
    random_synth: np.ndarray,
    pca_real: np.ndarray,
    pca_synth: np.ndarray,
    labels: np.ndarray,
    n_steps: int,
):
    if n_steps < 2:
        raise ValueError("--animation-steps must be at least 2")

    frames = []
    forward_names = []
    reverse_names = []
    trace_indices = list(range(8))

    for i, t in enumerate(np.linspace(0.0, 1.0, n_steps)):
        name = f"to_pca_{i:03d}"
        forward_names.append(name)
        frames.append(go.Frame(
            name=name,
            data=frame_trace_updates(
                smooth_interpolate(random_real, pca_real, float(t)),
                smooth_interpolate(random_synth, pca_synth, float(t)),
                labels,
            ),
            traces=trace_indices,
        ))

    for i, t in enumerate(np.linspace(1.0, 0.0, n_steps)):
        name = f"to_random_{i:03d}"
        reverse_names.append(name)
        frames.append(go.Frame(
            name=name,
            data=frame_trace_updates(
                smooth_interpolate(random_real, pca_real, float(t)),
                smooth_interpolate(random_synth, pca_synth, float(t)),
                labels,
            ),
            traces=trace_indices,
        ))

    return frames, forward_names, reverse_names


def make_plotly_figure(
    random_real: np.ndarray,
    random_synth: np.ndarray,
    pca_real: np.ndarray,
    pca_synth: np.ndarray,
    labels: np.ndarray,
    explained: np.ndarray,
    animation_steps: int,
    frame_duration_ms: int,
):
    """
    Build the original figure first, then add animation.

    Crucially, the layout block below is the original layout block. We do not
    set camera, aspect mode, scene domain, axis ranges, or custom margins.
    """
    fig = go.Figure()

    # Start at the random endpoint, then smoothly autoplay to PCA once on load.
    add_all_traces(fig, random_real, random_synth, labels)

    pc1 = 100 * explained[0]
    pc2 = 100 * explained[1]
    pc3 = 100 * explained[2]

    # This is intentionally inherited verbatim from the original figure.
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
        margin=dict(l=0, r=0, b=0, t=120),
    )

    frames, forward_names, reverse_names = make_frames(
        random_real=random_real,
        random_synth=random_synth,
        pca_real=pca_real,
        pca_synth=pca_synth,
        labels=labels,
        n_steps=animation_steps,
    )
    fig.frames = frames

    animation_options = dict(
        # For 3D traces, many short redraws look smoother than long transitions.
        frame=dict(duration=frame_duration_ms, redraw=True),
        transition=dict(duration=0),
        mode="immediate",
        fromcurrent=True,
    )

    # The buttons are the only visible layout addition.
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.02,
                y=1.01,
                xanchor="left",
                yanchor="bottom",
                showactive=False,
                buttons=[
                    dict(
                        label="Random → PCA",
                        method="animate",
                        args=[forward_names, animation_options],
                    ),
                    dict(
                        label="PCA → Random",
                        method="animate",
                        args=[reverse_names, animation_options],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
            )
        ]
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
    parser.add_argument("--random-seed", type=int, default=20260708)
    parser.add_argument("--animation-steps", type=int, default=100)
    parser.add_argument("--frame-duration-ms", type=int, default=35)
    parser.add_argument("--open", action="store_true", help="Open the HTML plot in browser if possible.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    out_html = args.out_dir / "random_vs_synthetic_pca_3d_morph.html"
    out_npz = args.out_dir / "random_vs_synthetic_pca_3d_scores.npz"
    out_csv = args.out_dir / "random_vs_synthetic_pca_3d_scores.csv"

    print("=" * 80)
    print("Original Plotly figure with random-projection morph")
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

    pca, pca_real, pca_synth = fit_synthetic_pca_and_project(X_real, X_synth)
    random_basis, random_real_raw, random_synth_raw = make_random_projection(
        X_real, X_synth, seed=args.random_seed
    )
    random_real, random_synth = align_random_to_pca_display(
        random_real_raw,
        random_synth_raw,
        pca_real,
        pca_synth,
    )

    fig = make_plotly_figure(
        random_real=random_real,
        random_synth=random_synth,
        pca_real=pca_real,
        pca_synth=pca_synth,
        labels=labels_kept,
        explained=pca.explained_variance_ratio_,
        animation_steps=args.animation_steps,
        frame_duration_ms=args.frame_duration_ms,
    )

    # Autoplay only the forward random-to-PCA sequence after Plotly has mounted.
    # The short delay avoids the browser trying to animate before the 3D scene exists.
    forward_names = [f"to_pca_{i:03d}" for i in range(args.animation_steps)]
    autoplay_js = f"""
    setTimeout(function() {{
        const gd = document.getElementById('{{plot_id}}');
        Plotly.animate(
            gd,
            {forward_names!r},
            {{
                frame: {{duration: {args.frame_duration_ms}, redraw: true}},
                transition: {{duration: 0}},
                mode: 'immediate',
                fromcurrent: true
            }}
        );
    }}, 450);
    """

    fig.write_html(
        out_html,
        include_plotlyjs="cdn",
        full_html=True,
        auto_play=False,
        post_script=autoplay_js,
    )

    rows = []
    for row_idx, image_index in enumerate(image_indices):
        lab = int(labels_kept[row_idx])
        for source, random_scores, pca_scores in (
            ("original", random_real_raw, pca_real),
            ("synthetic", random_synth_raw, pca_synth),
        ):
            rows.append({
                "source": source,
                "image_index": int(image_index),
                "label": lab,
                "label_name": LABEL_NAMES[lab],
                "random_1": float(random_scores[row_idx, 0]),
                "random_2": float(random_scores[row_idx, 1]),
                "random_3": float(random_scores[row_idx, 2]),
                "synthetic_pc1": float(pca_scores[row_idx, 0]),
                "synthetic_pc2": float(pca_scores[row_idx, 1]),
                "synthetic_pc3": float(pca_scores[row_idx, 2]),
            })

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    np.savez_compressed(
        out_npz,
        random_real_scores=random_real_raw,
        random_synth_scores=random_synth_raw,
        pca_real_scores=pca_real,
        pca_synth_scores=pca_synth,
        labels=labels_kept,
        image_indices=image_indices,
        explained_variance_ratio=pca.explained_variance_ratio_,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        random_basis=random_basis,
    )

    print("Saved:")
    print(f"  {out_html}")
    print(f"  {out_npz}")
    print(f"  {out_csv}")

    if args.open:
        fig.show()


if __name__ == "__main__":
    main()
