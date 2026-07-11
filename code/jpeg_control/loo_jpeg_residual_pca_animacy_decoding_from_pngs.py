#!/usr/bin/env python3
"""
Leakage-safe JPEG residualization followed by LOO PCA-logistic animacy decoding.

For each held-out image:
  1. Fit JPEG StandardScaler + PCA on the training images only.
  2. Fit neural activity ~ intercept + JPEG PCs on training images only.
  3. Residualize both training and held-out neural responses with that model.
  4. Fit neural StandardScaler + PCA on residualized training data only.
  5. Fit logistic regression for each requested PCA dimensionality.
  6. Predict the held-out image.

The script reports:
  - accuracy
  - balanced accuracy
  - ROC AUC
  - confusion-matrix counts
  - best observed PCA dimensionality
  - results at k=60, when requested

It also saves a cross-fitted residual matrix in which every row was produced
without using that row to fit the JPEG residualization model.

Default neural data path:
  /home/maria/SelfStudyThesis/data/
  allen_natural_scenes_four_class_composite.npy

JPEG input:
  By default, the script creates JPEG byte-size features itself.

  You may provide either:
    --images-dir /path/to/folder/with/scene_000.png ... scene_117.png
    --frames-path /path/to/natural_scene_frames.npy
    --jpeg-path /path/to/existing/sizes_bytes.npy

  When --images-dir is used, files are sorted by the integer in the filename,
  so scene_000.png maps to image 0, scene_001.png to image 1, and so on.

Examples:
  python loo_jpeg_residual_pca_animacy_decoding.py \
      --images-dir /home/maria/mariakesa.github.io/thesis/pc_visualization/data_dev

  python loo_jpeg_residual_pca_animacy_decoding.py \
      --frames-path /path/to/natural_scene_frames.npy

  python loo_jpeg_residual_pca_animacy_decoding.py \
      --jpeg-path /path/to/sizes_bytes.npy
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


DEFAULT_NEURAL_PATH = Path(
    "/home/maria/SelfStudyThesis/data/"
    "allen_natural_scenes_four_class_composite.npy"
)

DEFAULT_OUTDIR = Path(
    "/home/maria/SelfStudyThesis/results/"
    "loo_jpeg_residual_pca_animacy_decoding"
)

DEFAULT_K_VALUES = [1, 2, 3, 5, 10, 20, 40, 60, 80, 100, 116]

RANDOM_STATE = 0
MAX_ITER = 10000
LOGISTIC_C = 1.0
EPS = 1e-12



DEFAULT_JPEG_QUALITIES = [10, 25, 50, 75, 90]

COMMON_FRAME_FILENAMES = (
    "natural_scenes.npy",
    "natural_scene_images.npy",
    "natural_scene_frames.npy",
    "natural_scenes_templates.npy",
    "stimulus_templates.npy",
    "images.npy",
    "frames.npy",
)


def parse_qualities(text: str) -> list[int]:
    qualities = sorted({int(v.strip()) for v in text.split(",") if v.strip()})
    if not qualities:
        raise argparse.ArgumentTypeError(
            "qualities must be comma-separated integers"
        )
    invalid = [q for q in qualities if q < 1 or q > 95]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"JPEG qualities must lie in [1, 95], got {invalid}"
        )
    return qualities


def extract_frame_stack(obj, source_name: str) -> np.ndarray:
    """
    Extract an image stack from a loaded .npy/.npz object.

    Accepted final shapes:
      (118, H, W)
      (H, W, 118)
    """
    candidates = []

    if isinstance(obj, dict):
        preferred_keys = (
            "frames",
            "images",
            "natural_scenes",
            "natural_scene_images",
            "stimulus_templates",
            "templates",
            "data",
            "X",
        )
        for key in preferred_keys:
            if key in obj:
                candidates.append((key, np.asarray(obj[key])))
    else:
        candidates.append(("array", np.asarray(obj)))

    for key, array in candidates:
        if array.ndim != 3:
            continue

        if array.shape[0] == 118:
            return np.asarray(array)
        if array.shape[-1] == 118:
            return np.moveaxis(array, -1, 0)

    shapes = [(key, arr.shape) for key, arr in candidates]
    raise ValueError(
        f"Could not find a 118-image frame stack in {source_name}. "
        f"Candidate shapes: {shapes}"
    )


def load_frame_stack(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npy":
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
            obj = obj.item()
        return extract_frame_stack(obj, str(path))

    if path.suffix == ".npz":
        archive = np.load(path, allow_pickle=True)
        obj = {key: archive[key] for key in archive.files}
        return extract_frame_stack(obj, str(path))

    raise ValueError("Frame stack must be stored in .npy or .npz format")



def load_images_from_directory(images_dir: Path) -> np.ndarray:
    """
    Load scene images from a directory.

    Expected filenames include an integer image index, for example:
      scene_000.png
      scene_001.png
      ...
      scene_117.png

    Files are sorted by that integer, never alphabetically alone.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to read PNG images. Install it with:\n"
            "  pip install pillow"
        ) from exc

    images_dir = Path(images_dir)
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory does not exist: {images_dir}"
        )

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    indexed_files = []

    for path in images_dir.iterdir():
        if path.suffix.lower() not in extensions:
            continue

        matches = re.findall(r"(\d+)", path.stem)
        if not matches:
            continue

        image_index = int(matches[-1])
        indexed_files.append((image_index, path))

    if not indexed_files:
        raise FileNotFoundError(
            f"No indexed image files found in {images_dir}"
        )

    indexed_files.sort(key=lambda item: item[0])
    indices = [idx for idx, _ in indexed_files]

    if len(indexed_files) != 118:
        raise ValueError(
            f"Expected 118 indexed images, found {len(indexed_files)} "
            f"in {images_dir}"
        )

    expected = list(range(118))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        duplicates = sorted(
            idx for idx in set(indices) if indices.count(idx) > 1
        )
        raise ValueError(
            "Image indices must be exactly 0..117. "
            f"Missing={missing}, duplicates={duplicates}"
        )

    frames = []
    expected_shape = None

    for image_index, path in indexed_files:
        with Image.open(path) as image:
            frame = np.asarray(image.convert("L"))

        if expected_shape is None:
            expected_shape = frame.shape
        elif frame.shape != expected_shape:
            raise ValueError(
                f"Image {path} has shape {frame.shape}, "
                f"expected {expected_shape}"
            )

        frames.append(frame)

    return np.stack(frames, axis=0)


def discover_frame_stack(project_root: Path) -> Path:
    """
    Search common project data locations for a valid 118-image stack.
    """
    roots = [
        project_root / "data",
        project_root / "results",
    ]

    candidate_paths = []
    for root in roots:
        if not root.exists():
            continue
        for filename in COMMON_FRAME_FILENAMES:
            candidate_paths.extend(root.rglob(filename))

    valid = []
    for path in sorted(set(candidate_paths)):
        try:
            frames = load_frame_stack(path)
        except Exception:
            continue
        if frames.ndim == 3 and frames.shape[0] == 118:
            valid.append(path)

    if len(valid) == 1:
        print(f"[INFO] Auto-discovered frame stack: {valid[0]}")
        return valid[0]

    if len(valid) == 0:
        raise FileNotFoundError(
            "No 118-image natural-scene frame stack was found automatically. "
            "Run with --frames-path /absolute/path/to/frames.npy"
        )

    choices = "\n".join(f"  - {p}" for p in valid)
    raise RuntimeError(
        "Multiple plausible frame stacks were found. "
        "Choose one explicitly with --frames-path:\n"
        f"{choices}"
    )


def to_uint8_global(frames: np.ndarray) -> np.ndarray:
    """
    Normalize the complete frame stack with one global min/max, matching the
    supplied JPEG extraction code.
    """
    frames = np.asarray(frames, dtype=np.float32)
    frames = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)

    lo = float(np.min(frames))
    hi = float(np.max(frames))

    if hi == lo:
        return np.zeros_like(frames, dtype=np.uint8)

    scaled = (frames - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def make_jpeg_sizes(
    frames: np.ndarray,
    qualities: list[int],
) -> np.ndarray:
    """
    Encode every frame as grayscale JPEG at each quality and return byte sizes.

    Output shape:
      n_images x n_qualities
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to create JPEG features. Install it with:\n"
            "  pip install pillow"
        ) from exc

    frames_u8 = to_uint8_global(frames)
    sizes = np.zeros(
        (frames_u8.shape[0], len(qualities)),
        dtype=np.int64,
    )

    print()
    print("#" * 80)
    print("Creating JPEG byte-size features")
    print("#" * 80)

    for i, frame in enumerate(frames_u8):
        for j, quality in enumerate(qualities):
            buffer = io.BytesIO()
            Image.fromarray(frame).convert("L").save(
                buffer,
                format="JPEG",
                quality=int(quality),
            )
            sizes[i, j] = buffer.tell()

        if i == 0 or (i + 1) % 10 == 0 or i == len(frames_u8) - 1:
            print(
                f"\rEncoded image {i + 1:>3}/{len(frames_u8)}",
                end="",
                flush=True,
            )

    print()
    return sizes


def parse_k_values(text: str) -> list[int]:
    values = sorted({int(v.strip()) for v in text.split(",") if v.strip()})
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError(
            "k values must be comma-separated positive integers"
        )
    return values


def load_numeric_array(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npy":
        obj = np.load(path, allow_pickle=True)

        if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
            obj = obj.item()
            if not isinstance(obj, dict):
                return np.asarray(obj)

            for key in ("sizes_bytes", "sizes_kb", "jpeg", "features", "X"):
                if key in obj:
                    return np.asarray(obj[key])

            raise ValueError(
                f"{path} contains a dictionary but no recognized array key"
            )

        return np.asarray(obj)

    if path.suffix == ".npz":
        archive = np.load(path, allow_pickle=True)

        for key in ("sizes_bytes", "sizes_kb", "jpeg", "features", "X"):
            if key in archive.files:
                return np.asarray(archive[key])

        if len(archive.files) == 1:
            return np.asarray(archive[archive.files[0]])

        raise ValueError(
            f"{path} contains multiple arrays; specify one of "
            "sizes_bytes, sizes_kb, jpeg, features, or X"
        )

    raise ValueError("JPEG file must be .npy or .npz")


def orient_rows(array: np.ndarray, n_rows: int, name: str) -> np.ndarray:
    array = np.asarray(array)

    if array.ndim == 1:
        if len(array) != n_rows:
            raise ValueError(
                f"{name} length {len(array)} does not match {n_rows}"
            )
        return array[:, None]

    if array.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got {array.shape}")

    if array.shape[0] == n_rows:
        return array

    if array.shape[1] == n_rows:
        return array.T

    raise ValueError(
        f"Cannot orient {name} with shape {array.shape} to {n_rows} rows"
    )


def load_neural_data(path: Path):
    data = np.load(path, allow_pickle=True).item()

    X = np.asarray(data["X"], dtype=np.float64)
    labels4 = np.asarray(
        data["stimulus_metadata"]["label"],
        dtype=np.int64,
    ).ravel()

    if X.shape[0] == len(labels4):
        pass
    elif X.shape[1] == len(labels4):
        X = X.T
    else:
        raise ValueError(
            f"Cannot align X {X.shape} with labels {labels4.shape}"
        )

    keep_images = labels4 >= 0
    original_image_indices = np.flatnonzero(keep_images)

    X = X[keep_images]
    labels4 = labels4[keep_images]

    # 1 = animate, 0 = inanimate
    y = (labels4 == 0).astype(np.int64)

    finite = np.all(np.isfinite(X), axis=0)
    nonconstant = np.var(X, axis=0) > EPS
    neuron_mask = finite & nonconstant

    if not neuron_mask.any():
        raise ValueError("No finite, nonconstant neurons remain")

    X = X[:, neuron_mask]

    return X, y, labels4, original_image_indices, neuron_mask


def fit_jpeg_nuisance_coordinates(
    jpeg_train: np.ndarray,
    jpeg_test: np.ndarray,
    n_components: int,
):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(jpeg_train)
    test_scaled = scaler.transform(jpeg_test)

    max_components = min(
        int(n_components),
        train_scaled.shape[1],
        train_scaled.shape[0] - 1,
    )
    if max_components < 1:
        raise ValueError("Cannot fit any JPEG nuisance component")

    if train_scaled.shape[1] == 1:
        return train_scaled[:, :1], test_scaled[:, :1], 1.0

    pca = PCA(
        n_components=max_components,
        svd_solver="full",
    )
    train_pc = pca.fit_transform(train_scaled)
    test_pc = pca.transform(test_scaled)

    return (
        train_pc,
        test_pc,
        float(pca.explained_variance_ratio_.sum()),
    )


def residualize_fold(
    neural_train: np.ndarray,
    neural_test: np.ndarray,
    nuisance_train: np.ndarray,
    nuisance_test: np.ndarray,
):
    d_train = np.column_stack(
        [np.ones(nuisance_train.shape[0]), nuisance_train]
    )
    d_test = np.column_stack(
        [np.ones(nuisance_test.shape[0]), nuisance_test]
    )

    beta, *_ = np.linalg.lstsq(
        d_train,
        neural_train,
        rcond=None,
    )

    residual_train = neural_train - d_train @ beta
    residual_test = neural_test - d_test @ beta

    return residual_train, residual_test


def make_classifier() -> LogisticRegression:
    return LogisticRegression(
        penalty="l2",
        C=LOGISTIC_C,
        solver="liblinear",
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )


def decode_all_k(
    residual_train: np.ndarray,
    residual_test: np.ndarray,
    y_train: np.ndarray,
    k_values: list[int],
):
    scaler = StandardScaler()
    train_std = scaler.fit_transform(residual_train)
    test_std = scaler.transform(residual_test)

    max_k = min(
        max(k_values),
        train_std.shape[0] - 1,
        train_std.shape[1],
    )
    valid_k = [k for k in k_values if k <= max_k]

    pca = PCA(
        n_components=max(valid_k),
        svd_solver="full",
    )
    train_scores = pca.fit_transform(train_std)
    test_scores = pca.transform(test_std)

    results = {}

    for k in valid_k:
        clf = make_classifier()
        clf.fit(train_scores[:, :k], y_train)

        prediction = int(
            clf.predict(test_scores[:, :k])[0]
        )
        probability = float(
            clf.predict_proba(test_scores[:, :k])[0, 1]
        )

        results[k] = (prediction, probability)

    return results


def run_loo(
    X: np.ndarray,
    jpeg: np.ndarray,
    y: np.ndarray,
    k_values: list[int],
    n_jpeg_components: int,
):
    n_images = X.shape[0]

    predictions = {
        k: np.full(n_images, -1, dtype=np.int64)
        for k in k_values
    }
    probabilities = {
        k: np.full(n_images, np.nan, dtype=np.float64)
        for k in k_values
    }

    cross_fitted_residuals = np.full_like(
        X,
        np.nan,
        dtype=np.float64,
    )

    jpeg_evr = np.zeros(n_images, dtype=np.float64)

    all_indices = np.arange(n_images)

    print()
    print("#" * 80)
    print("Running leakage-safe LOO JPEG residualization + PCA decoding")
    print("#" * 80)

    for test_index in all_indices:
        train_mask = all_indices != test_index

        X_train = X[train_mask]
        X_test = X[[test_index]]
        jpeg_train = jpeg[train_mask]
        jpeg_test = jpeg[[test_index]]
        y_train = y[train_mask]

        nuisance_train, nuisance_test, evr = (
            fit_jpeg_nuisance_coordinates(
                jpeg_train=jpeg_train,
                jpeg_test=jpeg_test,
                n_components=n_jpeg_components,
            )
        )
        jpeg_evr[test_index] = evr

        residual_train, residual_test = residualize_fold(
            neural_train=X_train,
            neural_test=X_test,
            nuisance_train=nuisance_train,
            nuisance_test=nuisance_test,
        )

        cross_fitted_residuals[test_index] = residual_test[0]

        fold_results = decode_all_k(
            residual_train=residual_train,
            residual_test=residual_test,
            y_train=y_train,
            k_values=k_values,
        )

        for k, (pred, prob) in fold_results.items():
            predictions[k][test_index] = pred
            probabilities[k][test_index] = prob

        if (
            test_index == 0
            or (test_index + 1) % 5 == 0
            or test_index == n_images - 1
        ):
            print(
                f"\rFold {test_index + 1:>3}/{n_images}",
                end="",
                flush=True,
            )

    print()

    return (
        predictions,
        probabilities,
        cross_fitted_residuals,
        jpeg_evr,
    )


def safe_auc(y: np.ndarray, probability: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, probability))
    except ValueError:
        return float("nan")


def build_metrics(
    y: np.ndarray,
    predictions: dict[int, np.ndarray],
    probabilities: dict[int, np.ndarray],
    k_values: list[int],
) -> pd.DataFrame:
    rows = []

    for k in k_values:
        pred = predictions[k]
        prob = probabilities[k]

        if np.any(pred < 0) or np.any(~np.isfinite(prob)):
            raise RuntimeError(f"Missing predictions for k={k}")

        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        rows.append(
            {
                "k": int(k),
                "accuracy": float(accuracy_score(y, pred)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y, pred)
                ),
                "roc_auc": safe_auc(y, prob),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "n_correct": int(np.sum(pred == y)),
                "n_images": int(len(y)),
            }
        )

    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def choose_best(metrics: pd.DataFrame) -> pd.Series:
    # Headline criterion: balanced accuracy.
    # Tie-break: ROC AUC, then fewer PCs.
    return metrics.sort_values(
        ["balanced_accuracy", "roc_auc", "k"],
        ascending=[False, False, True],
    ).iloc[0]


def plot_metric(
    metrics: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    outdir: Path,
):
    fig, ax = plt.subplots(figsize=(8.8, 5.5))

    ax.plot(
        metrics["k"],
        metrics[column],
        marker="o",
        linewidth=2,
    )
    ax.axhline(0.5, linestyle=":", linewidth=1.2)
    if 60 in metrics["k"].values:
        ax.axvline(60, linestyle="--", linewidth=1.1)

    ax.set(
        xlabel="Number of residual neural PCA dimensions, k",
        ylabel=ylabel,
        title=title,
        xlim=(0, metrics["k"].max() + 2),
        ylim=(0.35, 1.02),
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()

    for suffix in ("png", "pdf"):
        fig.savefig(
            outdir / f"{column}_by_dimension.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "LOO JPEG residualization followed by LOO PCA-logistic "
            "animacy decoding"
        )
    )

    parser.add_argument(
        "--neural-path",
        type=Path,
        default=DEFAULT_NEURAL_PATH,
    )
    parser.add_argument(
        "--jpeg-path",
        type=Path,
        default=None,
        help=(
            "Optional existing JPEG feature matrix. When omitted, the script "
            "creates sizes_bytes from the image stack."
        ),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing scene_000.png through scene_117.png. "
            "Used when --jpeg-path is omitted."
        ),
    )
    parser.add_argument(
        "--frames-path",
        type=Path,
        default=None,
        help=(
            "Optional natural-scene frame stack with shape (118, H, W). "
            "Used when --jpeg-path and --images-dir are omitted."
        ),
    )
    parser.add_argument(
        "--jpeg-qualities",
        type=parse_qualities,
        default=DEFAULT_JPEG_QUALITIES,
        help="Comma-separated JPEG quality levels; default 10,25,50,75,90",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
    )
    parser.add_argument(
        "--k-values",
        type=parse_k_values,
        default=DEFAULT_K_VALUES,
        help="Comma-separated PCA dimensions",
    )
    parser.add_argument(
        "--n-jpeg-components",
        type=int,
        default=1,
        help="Number of fold-fitted JPEG PCs to remove",
    )

    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print()
    print("#" * 80)
    print("Loading data")
    print("#" * 80)

    X, y, labels4, original_indices, neuron_mask = load_neural_data(
        args.neural_path
    )

    n_original_images = int(np.max(original_indices) + 1)

    if args.jpeg_path is not None:
        jpeg_source = str(args.jpeg_path)
        jpeg_all = load_numeric_array(args.jpeg_path)
        jpeg_all = orient_rows(
            jpeg_all,
            n_rows=n_original_images,
            name="JPEG feature matrix",
        )
    else:
        project_root = args.neural_path.parent.parent

        if args.images_dir is not None:
            frames = load_images_from_directory(args.images_dir)
            frame_source = str(args.images_dir)
            source_kind = "image_directory"
        else:
            frames_path = (
                args.frames_path
                if args.frames_path is not None
                else discover_frame_stack(project_root)
            )
            frames = load_frame_stack(frames_path)
            frame_source = str(frames_path)
            source_kind = "frame_stack"

        if frames.shape[0] != n_original_images:
            raise ValueError(
                f"Image source has {frames.shape[0]} images, "
                f"expected {n_original_images}"
            )

        jpeg_all = make_jpeg_sizes(
            frames=frames,
            qualities=args.jpeg_qualities,
        )
        jpeg_source = f"generated from {frame_source}"

        generated_path = args.outdir / "sizes_bytes.npy"
        np.save(generated_path, jpeg_all)

        metadata_path = args.outdir / "jpeg_feature_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "source_kind": source_kind,
                    "source_path": frame_source,
                    "qualities": [int(q) for q in args.jpeg_qualities],
                    "shape": [int(v) for v in jpeg_all.shape],
                    "units": "bytes",
                    "normalization": "global_min_max_to_uint8",
                    "ordering": "integer index extracted from filename"
                    if source_kind == "image_directory"
                    else "stored array order",
                },
                indent=2,
            )
            + "\n"
        )

    jpeg = np.asarray(
        jpeg_all[original_indices],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(jpeg)):
        raise ValueError("JPEG features contain NaN or infinite values")

    max_loo_k = min(X.shape[0] - 2, X.shape[1])
    k_values = [k for k in args.k_values if k <= max_loo_k]

    if not k_values:
        raise ValueError(
            f"No requested k is valid; LOO maximum is {max_loo_k}"
        )

    n_jpeg_components = max(
        1,
        min(
            int(args.n_jpeg_components),
            jpeg.shape[1],
            X.shape[0] - 2,
        ),
    )

    print(f"Neural matrix:          {X.shape}")
    print(f"JPEG feature matrix:    {jpeg.shape}")
    print(f"JPEG feature source:    {jpeg_source}")
    print(f"Animate images:         {int(np.sum(y == 1))}")
    print(f"Inanimate images:       {int(np.sum(y == 0))}")
    print(f"JPEG PCs removed:       {n_jpeg_components}")
    print(f"Neural PCA dimensions:  {k_values}")

    (
        predictions,
        probabilities,
        residual_matrix,
        jpeg_evr,
    ) = run_loo(
        X=X,
        jpeg=jpeg,
        y=y,
        k_values=k_values,
        n_jpeg_components=n_jpeg_components,
    )

    metrics = build_metrics(
        y=y,
        predictions=predictions,
        probabilities=probabilities,
        k_values=k_values,
    )

    best = choose_best(metrics)

    print()
    print("#" * 80)
    print("Results")
    print("#" * 80)
    print(
        f"Best observed k={int(best['k'])}: "
        f"accuracy={best['accuracy']:.4f}, "
        f"balanced_accuracy={best['balanced_accuracy']:.4f}, "
        f"AUC={best['roc_auc']:.4f}"
    )

    row60 = metrics[metrics["k"] == 60]
    if not row60.empty:
        row60 = row60.iloc[0]
        print(
            f"k=60: accuracy={row60['accuracy']:.4f}, "
            f"balanced_accuracy={row60['balanced_accuracy']:.4f}, "
            f"AUC={row60['roc_auc']:.4f}"
        )

    metrics_path = args.outdir / "metrics_by_k.csv"
    summary_path = args.outdir / "summary.json"
    predictions_path = args.outdir / "predictions_by_k.npz"
    residual_path = (
        args.outdir / "cross_fitted_jpeg_residual_matrix.npy"
    )

    metrics.to_csv(metrics_path, index=False)
    np.save(residual_path, residual_matrix)

    payload = {
        "y_animacy": y,
        "labels_four_class": labels4,
        "original_image_indices": original_indices,
        "neuron_mask": neuron_mask,
        "k_values": np.asarray(k_values, dtype=np.int64),
        "jpeg_features": jpeg,
        "jpeg_pca_evr_by_fold": jpeg_evr,
    }

    for k in k_values:
        payload[f"predictions_k{k}"] = predictions[k]
        payload[f"probabilities_k{k}"] = probabilities[k]

    np.savez_compressed(predictions_path, **payload)

    best_dict = {
        key: (
            int(value)
            if isinstance(value, (np.integer,))
            else float(value)
            if isinstance(value, (np.floating,))
            else value
        )
        for key, value in best.to_dict().items()
    }

    summary = {
        "method": (
            "LOO JPEG residualization + fold-local StandardScaler + "
            "PCA + logistic regression"
        ),
        "neural_path": str(args.neural_path),
        "jpeg_source": jpeg_source,
        "n_images": int(X.shape[0]),
        "n_neurons": int(X.shape[1]),
        "n_jpeg_features": int(jpeg.shape[1]),
        "n_jpeg_components_removed": int(n_jpeg_components),
        "k_values": [int(k) for k in k_values],
        "best_observed_setting": best_dict,
        "mean_jpeg_pc_explained_variance_ratio": float(
            np.mean(jpeg_evr)
        ),
        "selection_note": (
            "Predictions for every k are leakage-safe. The best observed k "
            "is selected after inspecting this dataset and should be described "
            "as the best observed dimensionality, not as an independently "
            "validated hyperparameter."
        ),
        "artifacts": {
            "metrics_by_k": str(metrics_path),
            "predictions_by_k": str(predictions_path),
            "cross_fitted_residual_matrix": str(residual_path),
            "generated_sizes_bytes": (
                str(args.outdir / "sizes_bytes.npy")
                if args.jpeg_path is None
                else None
            ),
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    plot_metric(
        metrics,
        column="accuracy",
        ylabel="LOO accuracy",
        title="JPEG-residual animacy decoding accuracy",
        outdir=args.outdir,
    )
    plot_metric(
        metrics,
        column="balanced_accuracy",
        ylabel="LOO balanced accuracy",
        title="JPEG-residual animacy decoding balanced accuracy",
        outdir=args.outdir,
    )
    plot_metric(
        metrics,
        column="roc_auc",
        ylabel="LOO ROC AUC",
        title="JPEG-residual animacy decoding ROC AUC",
        outdir=args.outdir,
    )

    print()
    print("#" * 80)
    print("Saved")
    print("#" * 80)
    for path in (
        metrics_path,
        summary_path,
        predictions_path,
        residual_path,
        args.outdir / "accuracy_by_dimension.png",
        args.outdir / "balanced_accuracy_by_dimension.png",
        args.outdir / "roc_auc_by_dimension.png",
    ):
        print(path)


if __name__ == "__main__":
    main()
