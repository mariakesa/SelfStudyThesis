#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = Path(
    "/home/maria/SelfStudyThesis/data/synthetic_neural_activity_image_probs_loo.npy"
)

OUTDIR = Path(
    "/home/maria/SelfStudyThesis/results/synthetic_neural_activity_figures"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "synthetic_activity_probs_neurons_by_images.png"
OUT_PDF = OUTDIR / "synthetic_activity_probs_neurons_by_images.pdf"

dat = np.load(DATA_PATH, allow_pickle=True)

print(f"Loaded data shape: {dat.shape}")
print(f"min={dat.min():.6f}, max={dat.max():.6f}, mean={dat.mean():.6f}")

fig, ax = plt.subplots(figsize=(14, 8))

im = ax.imshow(
    dat,
    cmap="hot",
    interpolation="nearest",
    aspect="auto",
)

ax.set_title("Synthetic predicted neural activity probabilities")
ax.set_xlabel("Image index")
ax.set_ylabel("Synthetic neuron index")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Predicted activity probability")

plt.tight_layout()

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")

print(f"Saved PNG to: {OUT_PNG}")
print(f"Saved PDF to: {OUT_PDF}")

plt.show()