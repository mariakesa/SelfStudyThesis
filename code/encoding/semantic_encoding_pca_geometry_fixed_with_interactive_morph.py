#!/usr/bin/env python3
"""
Semantic encoding geometry:
LOO four-class encoding -> PCA on synthetic predictions -> project REAL neural
activity into the encoding-derived PCA basis.

This is the intended pipeline:

    semantic labels
        -> leave-one-out encoding regression
        -> synthetic class-predictable neural activity
        -> PCA(synthetic activity)
        -> project real neural activity onto those synthetic PCs

The important point is that the PCA basis is learned ONLY from the
class-predictable synthetic activity, not from the real activity.

Input
-----
/home/maria/SelfStudyThesis/data/allen_natural_scenes_four_class_composite.npy

Expected composite fields:
    data["X"]                         neurons x images OR images x neurons
    data["stimulus_metadata"]["label"]
    data["neuron_metadata"]

Labels:
    -1 unlabeled
     0 animals
     1 landscape
     2 plant
     3 man-made object

Outputs
-------
/home/maria/SelfStudyThesis/results/semantic_encoding_pca_geometry/

Main outputs:
    synthetic_neural_activity_image_probs_loo.npy
    semantic_encoding_pca_geometry_results.npz
    synthetic_pca_explained_variance.csv
    pairwise_distance_retention.csv
    real_projected_into_encoding_pcs_2d.png
    real_projected_into_encoding_pcs_3d.png
    synthetic_and_real_in_encoding_pcs_2d.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut


# =============================================================================
# Paths / settings
# =============================================================================

COMPOSITE_PATH = Path(
    "/home/maria/SelfStudyThesis/data/allen_natural_scenes_four_class_composite.npy"
)

OUT_DIR = Path(
    "/home/maria/SelfStudyThesis/results/semantic_encoding_pca_geometry"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC_OUT = OUT_DIR / "synthetic_neural_activity_image_probs_loo.npy"
RESULTS_OUT = OUT_DIR / "semantic_encoding_pca_geometry_results.npz"
EVR_CSV = OUT_DIR / "synthetic_pca_explained_variance.csv"
DIST_CSV = OUT_DIR / "pairwise_distance_retention.csv"

N_PCS_TO_FIT = 30
N_PCS_FOR_GEOMETRY = 3

LABEL_NAMES = {
    -1: "unlabeled",
     0: "animals",
     1: "landscape",
     2: "plant",
     3: "man-made object",
}

MARKERS = {
    0: "o",
    1: "s",
    2: "^",
    3: "D",
}


# =============================================================================
# Loading
# =============================================================================

def load_data():
    print("=" * 88)
    print("Loading Allen four-class composite")
    print("=" * 88)
    print(f"Path: {COMPOSITE_PATH}")

    data = np.load(COMPOSITE_PATH, allow_pickle=True).item()

    X = np.asarray(data["X"], dtype=np.float64)
    labels = np.asarray(
        data["stimulus_metadata"]["label"], dtype=np.int64
    ).ravel()

    # Standardize to images x neurons.
    if X.shape[0] == len(labels):
        pass
    elif X.shape[1] == len(labels):
        X = X.T
        print("[INFO] Transposed X to images x neurons.")
    else:
        raise ValueError(
            f"Cannot align X={X.shape} with labels={labels.shape}."
        )

    keep = labels >= 0
    X = X[keep]
    labels = labels[keep]

    print(f"Kept X shape: {X.shape} (images x neurons)")
    print("Class counts:")
    for lab, count in zip(*np.unique(labels, return_counts=True)):
        print(f"  {lab}: {LABEL_NAMES[int(lab)]:<16} n={count}")

    return data, X, labels, keep


# =============================================================================
# Four-class LOO encoding
# =============================================================================

def make_design(labels):
    classes = np.array([0, 1, 2, 3], dtype=np.int64)
    D = np.column_stack([(labels == c).astype(np.float64) for c in classes])
    return D, classes


def loo_encoding_predictions(X, labels):
    """
    For every held-out image:
      - fit four-class one-hot regression on all other images
      - predict the held-out neural population vector

    With fit_intercept=False, each fitted class coefficient is the training-set
    mean neural response for that class.
    """
    D, classes = make_design(labels)

    pred = np.full_like(X, np.nan, dtype=np.float64)
    loo = LeaveOneOut()

    print()
    print("=" * 88)
    print("Leave-one-out semantic encoding")
    print("=" * 88)

    n = len(labels)
    for fold, (train_idx, test_idx) in enumerate(loo.split(D), start=1):
        train_labels = labels[train_idx]

        missing = sorted(set(classes.tolist()) - set(train_labels.tolist()))
        if missing:
            raise RuntimeError(
                f"Fold {fold}: training split is missing classes {missing}"
            )

        model = LinearRegression(fit_intercept=False)
        model.fit(D[train_idx], X[train_idx])
        pred[test_idx] = model.predict(D[test_idx])

        if fold == 1 or fold % 10 == 0 or fold == n:
            print(f"Fold {fold:>3}/{n}")

    if np.isnan(pred).any():
        raise RuntimeError("LOO prediction matrix contains NaNs.")

    np.save(SYNTHETIC_OUT, pred)
    print(f"Saved LOO synthetic activity: {SYNTHETIC_OUT}")

    return pred, D, classes


# =============================================================================
# Encoding quality
# =============================================================================

def summarize_encoding_quality(X_real, X_syn):
    r2 = r2_score(X_real, X_syn, multioutput="raw_values")

    finite = np.isfinite(r2)
    vals = r2[finite]

    ss_res = np.sum((X_real - X_syn) ** 2)
    ss_tot = np.sum((X_real - X_real.mean(axis=0, keepdims=True)) ** 2)
    global_r2 = 1.0 - ss_res / ss_tot

    print()
    print("=" * 88)
    print("Encoding quality")
    print("=" * 88)
    print(f"Global image-by-neuron R^2: {global_r2:.6f}")
    print(f"Per-neuron mean R^2:        {np.mean(vals):.6f}")
    print(f"Per-neuron median R^2:      {np.median(vals):.6f}")
    print(f"Per-neuron max R^2:         {np.max(vals):.6f}")
    print(f"Neurons with R^2 > 0:       {np.sum(vals > 0)}")

    return global_r2, r2


# =============================================================================
# PCA learned FROM SYNTHETIC ENCODING ACTIVITY
# =============================================================================

def fit_encoding_pca(X_syn):
    """
    Fit PCA ONLY on synthetic class-predictable activity.

    sklearn stores the synthetic training mean internally in pca.mean_.
    Any later pca.transform(X_real) subtracts THIS SAME synthetic mean,
    which is what we want for a common coordinate system.
    """
    n_components = min(
        N_PCS_TO_FIT,
        X_syn.shape[0] - 1,
        X_syn.shape[1],
    )

    pca = PCA(n_components=n_components, svd_solver="full")
    syn_scores = pca.fit_transform(X_syn)

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    table = pd.DataFrame({
        "pc": np.arange(1, len(evr) + 1),
        "explained_variance_ratio": evr,
        "cumulative_explained_variance_ratio": cum,
    })
    table.to_csv(EVR_CSV, index=False)

    print()
    print("=" * 88)
    print("PCA of synthetic encoding activity")
    print("=" * 88)
    for i in range(min(10, len(evr))):
        print(
            f"PC{i+1:02d}: EVR={evr[i]:.6f}  cumulative={cum[i]:.6f}"
        )
    print(f"Saved PCA spectrum: {EVR_CSV}")

    return pca, syn_scores, table


def project_real_into_encoding_pcs(pca, X_real):
    """
    THIS is the key intended operation.

    The real data are projected onto the PCs learned from synthetic activity.
    Do NOT center X_real by its own mean first. pca.transform uses pca.mean_,
    i.e. the synthetic training mean, preserving the common coordinate system.
    """
    return pca.transform(X_real)


# =============================================================================
# Geometry diagnostics
# =============================================================================

def pairwise_distance_retention(X_real, real_scores, labels, k=3):
    """
    For each pair of real images:
        retention = projected squared distance / full squared distance

    Because the encoding PCs are orthonormal, projection cannot increase
    Euclidean distance, so retention lies in [0, 1] up to numerical noise.
    """
    k = min(k, real_scores.shape[1])

    rows = []
    n = X_real.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            full_sq = float(np.sum((X_real[i] - X_real[j]) ** 2))
            proj_sq = float(np.sum((real_scores[i, :k] - real_scores[j, :k]) ** 2))

            if full_sq > 0:
                retention = proj_sq / full_sq
            else:
                retention = np.nan

            same_class = labels[i] == labels[j]

            rows.append({
                "i": i,
                "j": j,
                "label_i": int(labels[i]),
                "label_j": int(labels[j]),
                "same_class": bool(same_class),
                "full_squared_distance": full_sq,
                f"encoding_pc1_to_pc{k}_squared_distance": proj_sq,
                "distance_retention": retention,
            })

    df = pd.DataFrame(rows)
    df.to_csv(DIST_CSV, index=False)

    print()
    print("=" * 88)
    print(f"Real-data pairwise distance retention in first {k} encoding PCs")
    print("=" * 88)
    print(f"All pairs median:          {df['distance_retention'].median():.6f}")
    print(f"All pairs mean:            {df['distance_retention'].mean():.6f}")
    print(
        "Same-class median:       "
        f"{df.loc[df.same_class, 'distance_retention'].median():.6f}"
    )
    print(
        "Different-class median:  "
        f"{df.loc[~df.same_class, 'distance_retention'].median():.6f}"
    )
    print(f"Saved pairwise diagnostics: {DIST_CSV}")

    return df


def projected_energy_fraction(pca, X_real, k=3):
    """
    Fraction of real-data centered energy captured by first k ENCODING PCs.

    Center real data using the SYNTHETIC PCA training mean, because that is the
    origin of the encoding coordinate system.
    """
    k = min(k, len(pca.components_))

    Xc = X_real - pca.mean_
    W = pca.components_[:k]

    projected = Xc @ W.T
    captured = np.sum(projected ** 2)
    total = np.sum(Xc ** 2)

    return float(captured / total) if total > 0 else np.nan


# =============================================================================
# Plotting
# =============================================================================

def plot_real_2d(real_scores, labels):
    path = OUT_DIR / "real_projected_into_encoding_pcs_2d.png"

    fig, ax = plt.subplots(figsize=(8, 7))
    for lab in sorted(np.unique(labels)):
        mask = labels == lab
        ax.scatter(
            real_scores[mask, 0],
            real_scores[mask, 1],
            marker=MARKERS.get(int(lab), "o"),
            label=LABEL_NAMES[int(lab)],
            alpha=0.8,
        )

    ax.set_xlabel("Encoding-derived PC1")
    ax.set_ylabel("Encoding-derived PC2")
    ax.set_title("REAL neural activity projected into semantic encoding PCA space")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_real_3d(real_scores, labels):
    if real_scores.shape[1] < 3:
        return

    path = OUT_DIR / "real_projected_into_encoding_pcs_3d.png"

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    for lab in sorted(np.unique(labels)):
        mask = labels == lab
        ax.scatter(
            real_scores[mask, 0],
            real_scores[mask, 1],
            real_scores[mask, 2],
            marker=MARKERS.get(int(lab), "o"),
            label=LABEL_NAMES[int(lab)],
            alpha=0.8,
        )

    ax.set_xlabel("Encoding-derived PC1")
    ax.set_ylabel("Encoding-derived PC2")
    ax.set_zlabel("Encoding-derived PC3")
    ax.set_title("REAL neural activity in semantic encoding PCA space")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_synthetic_and_real_2d(syn_scores, real_scores, labels):
    path = OUT_DIR / "synthetic_and_real_in_encoding_pcs_2d.png"

    fig, ax = plt.subplots(figsize=(9, 7))

    for lab in sorted(np.unique(labels)):
        mask = labels == lab

        # Synthetic states: x markers.
        ax.scatter(
            syn_scores[mask, 0],
            syn_scores[mask, 1],
            marker="x",
            alpha=0.9,
            label=f"synthetic {LABEL_NAMES[int(lab)]}",
        )

        # Real states: class-specific marker.
        ax.scatter(
            real_scores[mask, 0],
            real_scores[mask, 1],
            marker=MARKERS.get(int(lab), "o"),
            alpha=0.55,
            label=f"real {LABEL_NAMES[int(lab)]}",
        )

    ax.set_xlabel("Encoding-derived PC1")
    ax.set_ylabel("Encoding-derived PC2")
    ax.set_title("Synthetic encoding states and REAL states in the SAME PCA basis")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")




# =============================================================================
# Interactive Plotly morph: random neural directions -> encoding-derived PCs
# =============================================================================

def make_interactive_morph_plot(
    X_real,
    X_syn,
    labels,
    pca_encoding,
    real_in_encoding_scores,
    syn_scores,
    original_image_indices=None,
):
    """
    Recreate the earlier interactive 3-D Plotly visualization, but with the
    CORRECT coordinate system:

        PCA is fit on LOO synthetic semantic-encoding activity,
        then REAL activity is projected into those encoding-derived PCs.

    The animation morphs from a random orthonormal 3-D neural subspace to the
    first three encoding-derived PC loading vectors.

    Synthetic and real points are both projected through the SAME interpolated
    basis at every animation frame.

    Output:
        semantic_encoding_random_to_pc_morph.html
    """
    import json
    import plotly.graph_objects as go

    if pca_encoding.components_.shape[0] < 3:
        raise ValueError("Need at least 3 PCA components for the 3-D morph.")

    n_images, n_neurons = X_real.shape

    # Preserve original Allen image indices if unlabeled images were dropped.
    if original_image_indices is None:
        original_image_indices = np.arange(n_images)
    original_image_indices = np.asarray(original_image_indices)

    # IMPORTANT: use the synthetic PCA mean as the common origin.
    # This matches pca_encoding.transform(X_real).
    Xr = X_real - pca_encoding.mean_
    Xs = X_syn - pca_encoding.mean_

    # Target basis: first 3 PCA loading vectors learned from synthetic activity.
    W_pc = pca_encoding.components_[:3].T  # neurons x 3

    # Random orthonormal neural directions, deterministic for reproducibility.
    rng = np.random.default_rng(123)
    Q, _ = np.linalg.qr(rng.normal(size=(n_neurons, 3)))
    W_random = Q[:, :3]

    def orthonormalize(W):
        # QR keeps a smooth enough path for visualization while ensuring that
        # each frame remains a legitimate orthonormal projection basis.
        Q, _ = np.linalg.qr(W)
        return Q[:, :3]

    def project(Xc, W):
        return Xc @ W

    # Coordinates at the two endpoints.
    real_random = project(Xr, W_random)
    syn_random = project(Xs, W_random)
    real_pc = real_in_encoding_scores[:, :3]
    syn_pc = syn_scores[:, :3]

    label_names = LABEL_NAMES

    # Match the class palette / symbols from the original HTML.
    real_colors = {
        0: "#1f77b4",
        1: "#2ca02c",
        2: "#9467bd",
        3: "#e377c2",
    }
    syn_colors = {
        0: "#ff7f0e",
        1: "#d62728",
        2: "#8c564b",
        3: "#7f7f7f",
    }

    # Point-image paths are intentionally relative, just like the old plot.
    # Change this constant if your image folder lives elsewhere relative to HTML.
    image_rel_dir = "../pc_visualization/data_dev"

    def make_customdata(mask, source):
        rows = []
        for local_idx in np.where(mask)[0]:
            original_idx = int(original_image_indices[local_idx])
            rows.append([
                original_idx,
                label_names[int(labels[local_idx])],
                source,
                f"{image_rel_dir}/scene_{original_idx:03d}.png",
            ])
        return rows

    def make_text(mask, source):
        out = []
        for local_idx in np.where(mask)[0]:
            original_idx = int(original_image_indices[local_idx])
            lab = int(labels[local_idx])
            out.append(
                f"{source}<br>"
                f"image_index: {original_idx}<br>"
                f"label: {lab}<br>"
                f"label_name: {label_names[lab]}"
            )
        return out

    fig = go.Figure()

    trace_meta = []
    # Real traces first, then synthetic traces, as in the earlier plot.
    for source, coords, colors, symbol, opacity, size in [
        ("original", real_random, real_colors, "circle", 0.70, 5),
        ("synthetic", syn_random, syn_colors, "x", 0.95, 6),
    ]:
        for lab in [0, 1, 2, 3]:
            mask = labels == lab
            fig.add_trace(
                go.Scatter3d(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    z=coords[mask, 2],
                    mode="markers",
                    name=f"{source} {label_names[lab]}",
                    marker=dict(
                        color=colors[lab],
                        size=size,
                        opacity=opacity,
                        symbol=symbol,
                        line=dict(color="black", width=0.5),
                    ),
                    text=make_text(mask, source),
                    customdata=make_customdata(mask, source),
                    hovertemplate=(
                        "%{text}<br>"
                        "PC1: %{x:.3f}<br>"
                        "PC2: %{y:.3f}<br>"
                        "PC3: %{z:.3f}"
                        "<extra></extra>"
                    ),
                )
            )
            trace_meta.append((source, lab, mask))

    # Build smooth frames in both directions.
    n_frames = 100

    def coords_for_t(t):
        # Linear interpolation in loading-vector space, then orthonormalize.
        W = orthonormalize((1.0 - t) * W_random + t * W_pc)
        return project(Xr, W), project(Xs, W)

    frames_to_pc = []
    frames_to_random = []

    for i in range(n_frames):
        t = (i + 1) / n_frames
        rr, ss = coords_for_t(t)

        frame_data = []
        for source, lab, mask in trace_meta:
            C = rr if source == "original" else ss
            frame_data.append(
                go.Scatter3d(
                    x=C[mask, 0],
                    y=C[mask, 1],
                    z=C[mask, 2],
                )
            )
        frames_to_pc.append(
            go.Frame(
                name=f"to_pca_{i:03d}",
                data=frame_data,
                traces=list(range(len(trace_meta))),
            )
        )

    for i in range(n_frames):
        t = 1.0 - (i + 1) / n_frames
        rr, ss = coords_for_t(t)

        frame_data = []
        for source, lab, mask in trace_meta:
            C = rr if source == "original" else ss
            frame_data.append(
                go.Scatter3d(
                    x=C[mask, 0],
                    y=C[mask, 1],
                    z=C[mask, 2],
                )
            )
        frames_to_random.append(
            go.Frame(
                name=f"to_random_{i:03d}",
                data=frame_data,
                traces=list(range(len(trace_meta))),
            )
        )

    fig.frames = frames_to_pc + frames_to_random

    fig.update_layout(
        title=(
            "Projections of neural data onto random vectors versus derived PC's "
            "of synthetic regression predictions"
            "<br><sup>"
            "Decomposition was carried out on LOO synthetic semantic-encoding "
            "activity only; real data are projected into that basis."
            "</sup>"
        ),
        height=850,
        autosize=True,
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=120),
        scene=dict(
            xaxis=dict(title="", showbackground=True, zeroline=True),
            yaxis=dict(title="", showbackground=True, zeroline=True),
            zaxis=dict(title="", showbackground=True, zeroline=True),
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                showactive=False,
                x=0.02,
                xanchor="left",
                y=1.01,
                yanchor="bottom",
                buttons=[
                    dict(
                        label="Random → Synthetic PC's",
                        method="animate",
                        args=[
                            [f"to_pca_{i:03d}" for i in range(n_frames)],
                            dict(
                                frame=dict(duration=35, redraw=True),
                                transition=dict(duration=0),
                                mode="immediate",
                                fromcurrent=True,
                            ),
                        ],
                    ),
                    dict(
                        label="Synthetic PC's → Random",
                        method="animate",
                        args=[
                            [f"to_random_{i:03d}" for i in range(n_frames)],
                            dict(
                                frame=dict(duration=35, redraw=True),
                                transition=dict(duration=0),
                                mode="immediate",
                                fromcurrent=True,
                            ),
                        ],
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
        ],
    )

    # Start at RANDOM, so the button visibly morphs to the encoding PCs.
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        div_id="synthetic-pca-morph",
        config={"responsive": True},
    )

    # Add the same image-inspector interaction used by the earlier HTML:
    # hover previews, click pins, click again/unpin via background click.
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synthetic PCA morph with image inspector</title>
<style>
  html, body {{ margin: 0; background: white; font-family: Arial, sans-serif; color: #243b63; }}
  .figure-shell {{ display: flex; align-items: flex-start; gap: 16px; width: min(1480px, 100%); margin: 0 auto; }}
  .plot-wrap {{ flex: 1 1 auto; min-width: 0; }}
  .inspector {{
    flex: 0 0 290px; margin-top: 92px; margin-right: 18px; padding: 16px;
    border: 1px solid #d6deea; border-radius: 12px; background: #fbfcfe;
    box-shadow: 0 4px 18px rgba(36,59,99,0.08); position: sticky; top: 16px;
  }}
  .inspector h3 {{ margin: 0 0 10px; font-size: 18px; }}
  .thumb-frame {{ width: 100%; aspect-ratio: 4 / 3; background: #eef2f7; border-radius: 8px; overflow: hidden; display: grid; place-items: center; }}
  .thumb-frame img {{ width: 100%; height: 100%; object-fit: contain; display: none; }}
  .placeholder {{ color: #6d7d99; font-size: 14px; text-align: center; padding: 16px; }}
  .meta {{ margin-top: 12px; font-size: 14px; line-height: 1.5; }}
  .meta strong {{ color: #1d3154; }}
  .hint {{ margin-top: 10px; font-size: 12px; color: #6d7d99; }}
  .pin-state {{ display: inline-block; margin-top: 8px; padding: 3px 8px; border-radius: 999px; background: #e7edf7; font-size: 12px; }}
  @media (max-width: 1050px) {{
    .figure-shell {{ display: block; }}
    .inspector {{ margin: 0 16px 18px; position: static; }}
  }}
</style>
</head>
<body>
<div class="figure-shell">
  <div class="plot-wrap">{plot_html}</div>
  <aside class="inspector">
    <h3>Image inspector</h3>
    <div class="thumb-frame">
      <img id="scene-thumb" alt="scene preview">
      <div id="scene-placeholder" class="placeholder">Hover over a point to inspect its stimulus.</div>
    </div>
    <div class="meta">
      <div><strong>Image:</strong> <span id="meta-index">—</span></div>
      <div><strong>Class:</strong> <span id="meta-class">—</span></div>
      <div><strong>Source:</strong> <span id="meta-source">—</span></div>
      <span id="pin-state" class="pin-state">hover mode</span>
    </div>
    <div class="hint">Hover previews. Click a point to pin it. Double-click the plot background to unpin.</div>
  </aside>
</div>

<script>
(function() {{
  const gd = document.getElementById('synthetic-pca-morph');
  const img = document.getElementById('scene-thumb');
  const placeholder = document.getElementById('scene-placeholder');
  const metaIndex = document.getElementById('meta-index');
  const metaClass = document.getElementById('meta-class');
  const metaSource = document.getElementById('meta-source');
  const pinState = document.getElementById('pin-state');
  let pinned = false;

  function showPoint(pt) {{
    if (!pt || !pt.customdata) return;
    const d = pt.customdata;
    metaIndex.textContent = d[0];
    metaClass.textContent = d[1];
    metaSource.textContent = d[2];
    img.src = d[3];
    img.style.display = 'block';
    placeholder.style.display = 'none';
  }}

  gd.on('plotly_hover', function(ev) {{
    if (!pinned && ev.points && ev.points.length) showPoint(ev.points[0]);
  }});

  gd.on('plotly_click', function(ev) {{
    if (ev.points && ev.points.length) {{
      showPoint(ev.points[0]);
      pinned = true;
      pinState.textContent = 'pinned';
    }}
  }});

  gd.on('plotly_doubleclick', function() {{
    pinned = false;
    pinState.textContent = 'hover mode';
  }});
}})();
</script>
</body>
</html>
"""

    out_html = OUT_DIR / "semantic_encoding_random_to_pc_morph.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"Saved interactive morph: {out_html}")
    return out_html


# =============================================================================
# Main
# =============================================================================

def main():
    _, X_real, labels, keep_mask = load_data()

    # Map filtered rows back to original Allen image indices for the inspector.
    original_image_indices = np.where(keep_mask)[0]

    # 1) Generate semantic-class-predictable neural activity with proper LOO.
    X_syn, design, classes = loo_encoding_predictions(X_real, labels)

    # 2) Quantify how much real neural modulation the encoding model predicts.
    global_r2, r2_by_neuron = summarize_encoding_quality(X_real, X_syn)

    # 3) Learn PCA axes ONLY from synthetic encoding predictions.
    pca_encoding, syn_scores, evr_table = fit_encoding_pca(X_syn)

    # 4) Project REAL activity onto those encoding-derived axes.
    real_in_encoding_scores = project_real_into_encoding_pcs(
        pca_encoding, X_real
    )

    # 5) Geometry diagnostics.
    k_geom = min(N_PCS_FOR_GEOMETRY, real_in_encoding_scores.shape[1])

    distance_df = pairwise_distance_retention(
        X_real,
        real_in_encoding_scores,
        labels,
        k=k_geom,
    )

    energy_fraction = projected_energy_fraction(
        pca_encoding,
        X_real,
        k=k_geom,
    )

    print()
    print(
        f"Fraction of real-data energy captured by first {k_geom} "
        f"encoding PCs: {energy_fraction:.6f}"
    )

    # 6) Static plots.
    plot_real_2d(real_in_encoding_scores, labels)
    plot_real_3d(real_in_encoding_scores, labels)
    plot_synthetic_and_real_2d(
        syn_scores,
        real_in_encoding_scores,
        labels,
    )

    # 7) Recreate the old interactive Random -> Synthetic-PC morph, but now
    #    using the CORRECT encoding-derived PCA basis.
    morph_html = make_interactive_morph_plot(
        X_real=X_real,
        X_syn=X_syn,
        labels=labels,
        pca_encoding=pca_encoding,
        real_in_encoding_scores=real_in_encoding_scores,
        syn_scores=syn_scores,
        original_image_indices=original_image_indices,
    )

    # 8) Save everything necessary for later analyses.
    np.savez_compressed(
        RESULTS_OUT,
        X_real=X_real,
        X_synthetic_loo=X_syn,
        labels=labels,
        keep_mask=keep_mask,
        original_image_indices=original_image_indices,
        design=design,
        classes=classes,
        pca_mean=pca_encoding.mean_,
        encoding_pc_components=pca_encoding.components_,
        encoding_pc_explained_variance_ratio=pca_encoding.explained_variance_ratio_,
        synthetic_scores=syn_scores,
        real_in_encoding_scores=real_in_encoding_scores,
        global_encoding_r2=global_r2,
        r2_by_neuron=r2_by_neuron,
        real_energy_fraction_first_k_encoding_pcs=energy_fraction,
        k_geometry=k_geom,
    )

    print()
    print("=" * 88)
    print("DONE")
    print("=" * 88)
    print(f"Results:          {RESULTS_OUT}")
    print(f"Interactive plot: {morph_html}")
    print()
    print("Interpretation:")
    print("  PCA axes were learned from LOO class-predictable synthetic activity.")
    print("  Real neural responses were projected into exactly those axes.")
    print("  The interactive morph compares that semantic basis against a random")
    print("  orthonormal 3-D neural subspace.")

if __name__ == "__main__":
    main()
