#!/usr/bin/env python3
"""
Sort Allen natural-scene images by an in-sample neural animacy score,
then sort neurons by their modulation along that axis.

Input
-----
/home/maria/SelfStudyThesis/data/
    allen_natural_scenes_four_class_composite.npy

Outputs
-------
/home/maria/SelfStudyThesis/results/animacy_modulation/

    animacy_modulation_delta_heatmap.png
    animacy_modulation_spearman_heatmap.png
    neuron_modulation_summary.png
    animacy_modulation_results.npz

Interpretation
--------------
Columns:
    Images ordered from non-animal-like to animal-like using an
    in-sample logistic-regression decision score.

Rows in delta heatmap:
    Neurons ordered by

        mean response to animals
        minus
        mean response to non-animals.

Rows in Spearman heatmap:
    Neurons ordered by monotonic association between their response
    probability and the continuous neural animacy score.

This is a descriptive analysis. The same neural population is used to
construct the image ordering and examine neuronal tuning.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_PATH = Path(
    "/home/maria/SelfStudyThesis/data/"
    "allen_natural_scenes_four_class_composite.npy"
)

# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_composite(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Load the composite dictionary and return:

        X                neurons x images
        labels           image labels
        neuron_metadata
        stimulus_metadata
    """
    if not path.exists():
        raise FileNotFoundError(f"Could not find: {path}")

    data = np.load(path, allow_pickle=True)

    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()

    if not isinstance(data, dict):
        raise TypeError(
            "Expected the .npy file to contain a dictionary."
        )

    print("\nComposite keys:")
    for key in data:
        print(f"  {key}")

    X = np.asarray(data["X"], dtype=np.float32)

    neuron_metadata = data.get("neuron_metadata", {})
    stimulus_metadata = data.get("stimulus_metadata", {})

    if "labels" in data:
        labels = np.asarray(data["labels"])
    elif "label" in stimulus_metadata:
        labels = np.asarray(stimulus_metadata["label"])
    else:
        raise KeyError(
            "Could not find labels in data['labels'] or "
            "data['stimulus_metadata']['label']."
        )

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, received {X.shape}")

    # Ensure neurons x images.
    if X.shape[1] != len(labels) and X.shape[0] == len(labels):
        print("Transposing X to neurons x images.")
        X = X.T

    if X.shape[1] != len(labels):
        raise ValueError(
            f"X shape {X.shape} does not match "
            f"{len(labels)} stimulus labels."
        )
    
    print(X.shape)

    return X, labels, neuron_metadata, stimulus_metadata

if __name__ == "__main__":
    #DATA_PATH=Path("/home/maria/SelfStudyThesis/data/allen_natural_scenes_four_class_binary_trials_composite.npy")
    X, labels, neuron_metadata, stimulus_metadata = load_composite(DATA_PATH)
    print(X.mean())

    zero_probability_fraction = np.mean(X == 0.0)
    print(f"Fraction of zero probabilities: {zero_probability_fraction:.6f}")
    '''
    print(np.mean(X))

    print("shape:", X.shape)
    print("unique values:", np.unique(X))
    print("mean:", X.mean())
    print("zero fraction:", np.mean(X == 0))
    print("one fraction:", np.mean(X == 1))
    print("NaNs:", np.isnan(X).sum())

    events_per_neuron = X.sum(axis=1)
    events_per_presentation = X.sum(axis=0)

    print("\nEvents per neuron:")
    print(np.percentile(events_per_neuron, [0, 1, 5, 25, 50, 75, 95, 99, 100]))

    print("\nActive neurons per presentation:")
    print(np.percentile(
        events_per_presentation,
        [0, 1, 5, 25, 50, 75, 95, 99, 100]
    ))

        # Assumes columns are ordered as 118 images × 50 repeats.
    X_by_image = X.reshape(X.shape[0], 118, 50)

    mean_rate_per_image = X_by_image.mean(axis=(0, 2))
    mean_rate_per_repeat = X_by_image.mean(axis=(0, 1))

    print("Image-rate percentiles:")
    print(np.percentile(
        mean_rate_per_image,
        [0, 5, 25, 50, 75, 95, 100]
    ))

    print("Repeat-rate percentiles:")
    print(np.percentile(
        mean_rate_per_repeat,
        [0, 5, 25, 50, 75, 95, 100]
    ))'''