#!/usr/bin/env python3
"""
Build composite Allen natural-scenes binary response probability dataset.

Input:
    Per-session binary matrices:
        /media/maria/notsudata/AllenOptical/neural_activity_matrices_/*_neural_responses.npy

    Each matrix is expected to be:
        neurons × 5900

    where:
        5900 = 118 images × 50 trials

    Four-class image labels:
        /home/maria/Science/data/four_class_image_labels.npy

Output:
    One composite .npy dictionary saved to:
        /home/maria/SelfStudyThesis/data/allen_natural_scenes_four_class_composite.npy

The final aggregate matrix X has shape:

    total_neurons × 118

where each value is:

    P(neuron active | image)

computed by averaging the 50 binary trial responses.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from allensdk.core.brain_observatory_cache import BrainObservatoryCache


# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────

ALLEN_CACHE_PATH = Path("/media/maria/notsudata/AllenOptical")

INPUT_DIR = ALLEN_CACHE_PATH / "neural_activity_matrices_"

LABEL_PATH = Path("/home/maria/Science/data/four_class_image_labels.npy")

OUTPUT_DIR = Path("/home/maria/SelfStudyThesis/data")
OUTPUT_PATH = OUTPUT_DIR / "allen_natural_scenes_four_class_composite.npy"


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

N_IMAGES = 118
N_TRIALS = 50
EXPECTED_COLUMNS = N_IMAGES * N_TRIALS

LABEL_NAMES = {
    -1: "unlabeled",
     0: "animals",
     1: "landscape",
     2: "plant",
     3: "man-made object",
}


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_four_class_labels(label_path: Path) -> np.ndarray:
    """
    Load four-class stimulus labels.

    Handles either:
        1. a plain NumPy label array of shape (118,)
        2. a saved dictionary with key "labels"
    """
    obj = np.load(label_path, allow_pickle=True)

    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()

    if isinstance(obj, dict):
        if "labels" not in obj:
            raise KeyError(
                f"{label_path} is a dict but does not contain key 'labels'. "
                f"Available keys: {list(obj.keys())}"
            )
        labels = np.asarray(obj["labels"])
    else:
        labels = np.asarray(obj)

    if labels.shape[0] != N_IMAGES:
        raise ValueError(
            f"Expected {N_IMAGES} image labels, got shape {labels.shape}"
        )

    labels = labels.astype(int)

    valid_values = set(LABEL_NAMES.keys())
    observed_values = set(np.unique(labels).tolist())

    unknown_values = observed_values - valid_values
    if unknown_values:
        raise ValueError(
            f"Image labels contain unknown label values: {unknown_values}. "
            f"Expected only {valid_values}"
        )

    return labels


def get_experiment_metadata_table(boc: BrainObservatoryCache) -> pd.DataFrame:
    """
    Get Allen experiment metadata indexed by ophys experiment id.
    """
    experiments = pd.DataFrame(boc.get_ophys_experiments())

    if "id" not in experiments.columns:
        raise KeyError("Allen experiment table does not contain column 'id'.")

    experiments = experiments.set_index("id", drop=False)

    return experiments


def get_area_for_session(
    sid: int,
    experiment_table: pd.DataFrame,
    dataset,
) -> str:
    """
    Try to get the brain area / targeted structure for an ophys experiment.
    """
    if sid in experiment_table.index:
        row = experiment_table.loc[sid]

        for col in ["targeted_structure", "structure", "imaging_area"]:
            if col in row and pd.notna(row[col]):
                return str(row[col])

    # Fallback through NWB metadata if available.
    try:
        metadata = dataset.get_metadata()
        for key in ["targeted_structure", "structure", "imaging_area"]:
            if key in metadata and metadata[key] is not None:
                return str(metadata[key])
    except Exception:
        pass

    return "unknown"


def get_session_metadata(
    sid: int,
    experiment_table: pd.DataFrame,
    dataset,
) -> dict:
    """
    Collect useful session-level Allen metadata.
    """
    meta = {
        "ophys_experiment_id": int(sid),
        "brain_area": get_area_for_session(sid, experiment_table, dataset),
    }

    if sid in experiment_table.index:
        row = experiment_table.loc[sid]

        useful_cols = [
            "experiment_container_id",
            "session_type",
            "targeted_structure",
            "imaging_depth",
            "cre_line",
            "reporter_line",
            "specimen_name",
            "donor_name",
        ]

        for col in useful_cols:
            if col in experiment_table.columns:
                value = row[col]
                if pd.isna(value):
                    value = None
                if isinstance(value, np.generic):
                    value = value.item()
                meta[col] = value

    try:
        nwb_meta = dataset.get_metadata()
        for key, value in nwb_meta.items():
            if key not in meta:
                if isinstance(value, np.generic):
                    value = value.item()
                meta[key] = value
    except Exception:
        pass

    return meta


def average_trials_to_probabilities(matrix: np.ndarray) -> np.ndarray:
    """
    Convert neurons × 5900 binary trial matrix into neurons × 118 probability matrix.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")

    n_neurons, n_cols = matrix.shape

    if n_cols != EXPECTED_COLUMNS:
        raise ValueError(
            f"Expected {EXPECTED_COLUMNS} columns "
            f"({N_IMAGES} images × {N_TRIALS} trials), got {n_cols}"
        )

    reshaped = matrix.reshape(n_neurons, N_IMAGES, N_TRIALS)

    # Mean of binary responses across repeats:
    # 0/1 values -> probability of event activity.
    probabilities = np.nanmean(reshaped, axis=2)

    return probabilities


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Building Allen natural-scenes four-class composite dataset")
    print("=" * 80)

    print(f"Allen cache: {ALLEN_CACHE_PATH}")
    print(f"Input dir:   {INPUT_DIR}")
    print(f"Label path:  {LABEL_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print()

    labels = load_four_class_labels(LABEL_PATH)

    print("Loaded image labels:")
    for label_value, label_name in LABEL_NAMES.items():
        count = int(np.sum(labels == label_value))
        print(f"  {label_value:>2} = {label_name:<16} n={count}")
    print()

    boc = BrainObservatoryCache(
        manifest_file=str(ALLEN_CACHE_PATH / "brain_observatory_manifest.json")
    )

    experiment_table = get_experiment_metadata_table(boc)

    files = sorted(INPUT_DIR.glob("*_neural_responses.npy"))

    if len(files) == 0:
        raise FileNotFoundError(f"No *_neural_responses.npy files found in {INPUT_DIR}")

    print(f"Found {len(files)} saved neural response matrix files.")
    print()

    probability_matrices = []

    neuron_cell_specimen_ids = []
    neuron_brain_areas = []
    neuron_session_ids = []
    neuron_local_indices = []

    session_metadata = {}
    session_neuron_counts = {}
    session_valid_trial_counts = {}
    session_nan_counts_before = {}
    session_nan_counts_after = {}

    used_files = []
    skipped_files = []

    for path in files:
        sid_str = path.stem.replace("_neural_responses", "")

        try:
            sid = int(sid_str)
        except ValueError:
            print(f"Skipping {path.name}: could not parse session id from filename.")
            skipped_files.append(str(path))
            continue

        print(f"Processing session {sid}")

        try:
            matrix = np.load(path)

            probabilities = average_trials_to_probabilities(matrix)

            n_neurons = probabilities.shape[0]

            dataset = boc.get_ophys_experiment_data(sid)

            try:
                cell_specimen_ids = np.asarray(dataset.get_cell_specimen_ids())
            except Exception as e:
                print(f"  Warning: could not load cell specimen ids: {e}")
                cell_specimen_ids = np.full(n_neurons, -1, dtype=int)

            if len(cell_specimen_ids) != n_neurons:
                print(
                    f"  Warning: cell specimen id count mismatch. "
                    f"Matrix has {n_neurons} neurons, AllenSDK returned "
                    f"{len(cell_specimen_ids)} cell ids."
                )
                min_len = min(len(cell_specimen_ids), n_neurons)

                probabilities = probabilities[:min_len, :]
                n_neurons = min_len
                cell_specimen_ids = cell_specimen_ids[:min_len]

            brain_area = get_area_for_session(sid, experiment_table, dataset)

            probability_matrices.append(probabilities)

            neuron_cell_specimen_ids.extend(cell_specimen_ids.astype(int).tolist())
            neuron_brain_areas.extend([brain_area] * n_neurons)
            neuron_session_ids.extend([sid] * n_neurons)
            neuron_local_indices.extend(list(range(n_neurons)))

            session_metadata[sid] = get_session_metadata(sid, experiment_table, dataset)
            session_neuron_counts[sid] = int(n_neurons)

            # For each neuron × image, this says how many non-NaN trials contributed.
            valid_trial_counts = np.sum(
                np.isfinite(matrix.reshape(matrix.shape[0], N_IMAGES, N_TRIALS)),
                axis=2,
            )

            if valid_trial_counts.shape[0] != n_neurons:
                valid_trial_counts = valid_trial_counts[:n_neurons, :]

            session_valid_trial_counts[sid] = valid_trial_counts

            session_nan_counts_before[sid] = int(np.isnan(matrix).sum())
            session_nan_counts_after[sid] = int(np.isnan(probabilities).sum())

            used_files.append(str(path))

            print(f"  brain area: {brain_area}")
            print(f"  neurons:    {n_neurons}")
            print(f"  output:     {probabilities.shape}")
            print(f"  NaNs before trial averaging: {session_nan_counts_before[sid]}")
            print(f"  NaNs after trial averaging:  {session_nan_counts_after[sid]}")
            print()

        except Exception as e:
            print(f"  Failed: {e}")
            print()
            skipped_files.append(str(path))
            continue

    if len(probability_matrices) == 0:
        raise RuntimeError("No valid sessions were processed. Nothing to save.")

    X = np.vstack(probability_matrices)

    neuron_metadata = {
        "cell_specimen_id": np.asarray(neuron_cell_specimen_ids, dtype=int),
        "brain_area": np.asarray(neuron_brain_areas, dtype=object),
        "ophys_experiment_id": np.asarray(neuron_session_ids, dtype=int),
        "local_neuron_index": np.asarray(neuron_local_indices, dtype=int),
        "description": (
            "Neuron-level metadata aligned row-by-row with X. "
            "For row i in X, use neuron_metadata['cell_specimen_id'][i], "
            "neuron_metadata['brain_area'][i], etc."
        ),
    }

    stimulus_metadata = {
        "image_index": np.arange(N_IMAGES, dtype=int),
        "label": labels,
        "label_name": np.asarray([LABEL_NAMES[int(x)] for x in labels], dtype=object),
        "label_names": LABEL_NAMES,
        "description": (
            "Stimulus metadata aligned column-by-column with X. "
            "For column j in X, use stimulus_metadata['label'][j] "
            "and stimulus_metadata['label_name'][j]."
        ),
    }

    composite = {
        "X": X,
        "neuron_metadata": neuron_metadata,
        "stimulus_metadata": stimulus_metadata,
        "label_names": LABEL_NAMES,
        "session_metadata": session_metadata,
        "session_neuron_counts": session_neuron_counts,
        "session_valid_trial_counts": session_valid_trial_counts,
        "session_nan_counts_before_trial_average": session_nan_counts_before,
        "session_nan_counts_after_trial_average": session_nan_counts_after,
        "used_files": used_files,
        "skipped_files": skipped_files,
        "constants": {
            "n_images": N_IMAGES,
            "n_trials": N_TRIALS,
            "expected_columns": EXPECTED_COLUMNS,
        },
        "description": (
            "Composite Allen natural-scenes dataset. X is total_neurons × 118. "
            "Rows are neurons concatenated across ophys sessions. Columns are "
            "natural scene images. X[i, j] is the average binary event response "
            "across up to 50 trials, interpreted as P(neuron active | image). "
            "Neuron metadata is row-aligned with X. Stimulus metadata is "
            "column-aligned with X."
        ),
    }

    np.save(OUTPUT_PATH, composite, allow_pickle=True)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved composite file:")
    print(f"  {OUTPUT_PATH}")
    print()
    print(f"X shape:")
    print(f"  {X.shape}")
    print()
    print(f"Total sessions used:   {len(session_neuron_counts)}")
    print(f"Total neurons:         {X.shape[0]}")
    print(f"Total stimuli/images:  {X.shape[1]}")
    print(f"Skipped files:         {len(skipped_files)}")
    print()

    print("Neuron counts by brain area:")
    areas, counts = np.unique(neuron_metadata["brain_area"], return_counts=True)
    for area, count in zip(areas, counts):
        print(f"  {area:<8} {count}")

    print()
    print("Stimulus label counts:")
    for label_value, label_name in LABEL_NAMES.items():
        count = int(np.sum(labels == label_value))
        print(f"  {label_value:>2} = {label_name:<16} n={count}")


if __name__ == "__main__":
    main()