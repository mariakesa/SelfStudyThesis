#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Paths
# =============================================================================

DATA_PATH = Path(
    "/home/maria/SelfStudyThesis/data/synthetic_neural_activity_image_probs_loo.npy"
)

OUTDIR = Path(
    "/home/maria/SelfStudyThesis/results/synthetic_neural_activity_figures/pca"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "synthetic_activity_top3_pcs_lineplot.png"
OUT_PDF = OUTDIR / "synthetic_activity_top3_pcs_lineplot.pdf"
OUT_NPZ = OUTDIR / "synthetic_activity_pca_top3.npz"


# =============================================================================
# Load
# =============================================================================

dat = np.load(DATA_PATH, allow_pickle=True)
dat = np.asarray(dat, dtype=np.float32)

print("=" * 80)
print("Loaded synthetic activity matrix")
print("=" * 80)
print(f"dat shape: {dat.shape}")
print(f"min:      {np.nanmin(dat):.6f}")
print(f"max:      {np.nanmax(dat):.6f}")
print(f"mean:     {np.nanmean(dat):.6f}")

dat = np.nan_to_num(dat, nan=0.0, posinf=0.0, neginf=0.0)


# =============================================================================
# PCA over images
# =============================================================================
# dat is assumed to be:
#   synthetic neurons × images
#
# We want:
#   images × synthetic neurons
#
# Each image becomes one high-dimensional population vector.

X = dat.T

print()
print("=" * 80)
print("PCA input")
print("=" * 80)
print(f"X shape, images × synthetic neurons: {X.shape}")


# Z-score features/neuron dimensions across images
Xz = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

pca = PCA(n_components=3, random_state=0)
Z = pca.fit_transform(Xz)

explained = pca.explained_variance_ratio_

print()
print("=" * 80)
print("PCA results")
print("=" * 80)
for k, evr in enumerate(explained, start=1):
    print(f"PC{k}: explained variance ratio = {evr:.6f}")


# =============================================================================
# Save PCA arrays
# =============================================================================

np.savez(
    OUT_NPZ,
    scores=Z,
    explained_variance_ratio=explained,
    components=pca.components_,
)

print()
print(f"Saved PCA data to: {OUT_NPZ}")


# =============================================================================
# Plot top 3 PCs simply as line plots over image index
# =============================================================================

image_index = np.arange(Z.shape[0])

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(image_index, Z[:, 0], label=f"PC1 ({explained[0] * 100:.1f}%)")
ax.plot(image_index, Z[:, 1], label=f"PC2 ({explained[1] * 100:.1f}%)")
ax.plot(image_index, Z[:, 2], label=f"PC3 ({explained[2] * 100:.1f}%)")

ax.axhline(0, linewidth=1)

ax.set_title("Top 3 PCs of synthetic population responses")
ax.set_xlabel("Image index")
ax.set_ylabel("PC score")
ax.legend(frameon=False)

plt.tight_layout()

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")

print(f"Saved PNG to: {OUT_PNG}")
print(f"Saved PDF to: {OUT_PDF}")

plt.show()