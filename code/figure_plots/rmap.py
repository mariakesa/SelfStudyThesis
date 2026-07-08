#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# Rastermap import
from rastermap import Rastermap


# =============================================================================
# Paths
# =============================================================================

DATA_PATH = Path(
    "/home/maria/SelfStudyThesis/data/synthetic_neural_activity_image_probs_loo.npy"
)

OUTDIR = Path(
    "/home/maria/SelfStudyThesis/results/synthetic_neural_activity_figures/rastermap"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "synthetic_activity_probs_rastermap_heatmap.png"
OUT_PDF = OUTDIR / "synthetic_activity_probs_rastermap_heatmap.pdf"
OUT_NPY = OUTDIR / "synthetic_activity_probs_rastermap_order.npy"


# =============================================================================
# Load
# =============================================================================

dat = np.load(DATA_PATH, allow_pickle=True)

print("=" * 80)
print("Loaded synthetic activity matrix")
print("=" * 80)
print(f"shape: {dat.shape}")
print(f"min:   {np.nanmin(dat):.6f}")
print(f"max:   {np.nanmax(dat):.6f}")
print(f"mean:  {np.nanmean(dat):.6f}")

# Expected shape:
#   rows    = synthetic neurons
#   columns = images
#
# dat[n, i] = predicted probability of synthetic neuron n for image i


# =============================================================================
# Clean / normalize for Rastermap
# =============================================================================

X = np.asarray(dat, dtype=np.float32)

# Remove any possible NaN/inf goblins
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Rastermap works better if rows are normalized.
# Here each synthetic neuron is z-scored across images.
Xz = StandardScaler(with_mean=True, with_std=True).fit_transform(X.T).T

print()
print("=" * 80)
print("Fitting Rastermap")
print("=" * 80)
print(f"Rastermap input shape: {Xz.shape}")


# =============================================================================
# Fit Rastermap across synthetic neurons
# =============================================================================

model = Rastermap(
    n_clusters=100,
    n_PCs=64,
    locality=0.75,
    time_lag_window=0,
    grid_upsample=10,
)

model.fit(Xz)

# Rastermap returns an ordering of rows / neurons
isort = model.isort

np.save(OUT_NPY, isort)

print()
print("=" * 80)
print("Rastermap finished")
print("=" * 80)
print(f"Saved neuron order to: {OUT_NPY}")


# =============================================================================
# Apply ordering
# =============================================================================

X_sorted = X[isort, :]


# =============================================================================
# Plot raw probability heatmap after Rastermap sorting
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 9))

im = ax.imshow(
    X_sorted,
    cmap="hot",
    interpolation="nearest",
    aspect="auto",
)

ax.set_title("Synthetic predicted activity probabilities, Rastermap-sorted neurons")
ax.set_xlabel("Image index")
ax.set_ylabel("Synthetic neuron index, Rastermap order")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Predicted activity probability")

plt.tight_layout()

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")

print(f"Saved PNG to: {OUT_PNG}")
print(f"Saved PDF to: {OUT_PDF}")

plt.show()