"""
run_assessment.py — Project Kestrel
Evaluate the trained ONNX model on the held-out test split and run a
Kestrel CLI integration test against 5 sampled species images.

Usage:
    python training/run_assessment.py

Pass criteria:
    Part 1 — Test set accuracy: top-1 >= 0.95 AND top-3 >= 0.99
    Part 2 — Integration test : at least 1 image identified as a named species
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE = 300
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = Path(__file__).resolve().parent.parent / "analyzer" / "models"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _preprocess(image_path: str) -> np.ndarray:
    """Load an image and return a normalised [1, 3, 300, 300] float32 array.

    Replicates the preprocessing used during training (and by
    BirdSpeciesClassifier._preprocess): resize to 300x300, normalise with
    ImageNet mean/std, transpose HWC -> CHW, add batch dimension.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)         # (300, 300, 3)
    arr = arr / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD     # normalise per-channel
    arr = arr.transpose(2, 0, 1)                   # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)              # -> (1, 3, 300, 300)
    return arr


# ---------------------------------------------------------------------------
# Part 1 — Test set accuracy
# ---------------------------------------------------------------------------


def assess_test_set() -> bool:
    """Evaluate the ONNX model on the held-out test split.

    Returns True if top-1 >= 0.95 AND top-3 >= 0.99.
    """
    onnx_path = MODELS_DIR / "model.onnx"
    labels_path = MODELS_DIR / "labels.txt"
    label_mapping_path = CHECKPOINT_DIR / "label_mapping.json"
    test_csv_path = SPLITS_DIR / "test.csv"

    # --- Load artefacts -------------------------------------------------------
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not label_mapping_path.exists():
        raise FileNotFoundError(f"Label mapping not found: {label_mapping_path}")
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test split not found: {test_csv_path}")

    log.info("Loading ONNX model from %s", onnx_path)
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    with labels_path.open(encoding="utf-8") as fh:
        onnx_labels: list[str] = [line.strip() for line in fh if line.strip()]
    log.info("Loaded %d ONNX output labels.", len(onnx_labels))

    with label_mapping_path.open(encoding="utf-8") as fh:
        mapping: dict = json.load(fh)
    label_to_idx: dict[str, int] = mapping["label_to_idx"]
    log.info("Loaded label mapping with %d entries.", len(label_to_idx))

    test_df = pd.read_csv(test_csv_path)
    log.info("Test split contains %d images across %d classes.",
             len(test_df), test_df["label"].nunique())

    # --- Evaluation loop ------------------------------------------------------
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    # Per-class tracking: label -> [correct_count, total_count]
    per_class: dict[str, list[int]] = {}
    # Confusion pairs: (true_common_name, predicted_common_name) -> count
    confusion: Counter = Counter()

    idx_to_label: dict[int, str] = {v: k for k, v in label_to_idx.items()}

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        image_path: str = row["path"]
        true_label: str = row["label"]
        true_common: str = row["common_name"]

        if true_label not in label_to_idx:
            log.warning("Label %r not in mapping — skipping.", true_label)
            continue

        true_idx: int = label_to_idx[true_label]

        try:
            input_arr = _preprocess(image_path)
        except Exception as exc:
            log.warning("Could not preprocess %s: %s — skipping.", image_path, exc)
            continue

        outputs = session.run(None, {"data": input_arr})
        logits: np.ndarray = outputs[0][0]  # shape (N,)

        pred_top1_idx: int = int(np.argmax(logits))
        top3_indices: list[int] = np.argsort(logits)[::-1][:3].tolist()

        is_top1_correct = pred_top1_idx == true_idx
        is_top3_correct = true_idx in top3_indices

        # Accumulate overall metrics
        correct_top1 += int(is_top1_correct)
        correct_top3 += int(is_top3_correct)
        total += 1

        # Per-class tracking
        if true_label not in per_class:
            per_class[true_label] = [0, 0]
        per_class[true_label][1] += 1
        if is_top1_correct:
            per_class[true_label][0] += 1

        # Confusion tracking (only for top-1 misses)
        if not is_top1_correct:
            pred_label = idx_to_label.get(pred_top1_idx, "")
            pred_common = test_df.loc[
                test_df["label"] == pred_label, "common_name"
            ].iloc[0] if pred_label and pred_label in test_df["label"].values else pred_label
            confusion[(true_common, pred_common)] += 1

    # --- Compute metrics ------------------------------------------------------
    if total == 0:
        log.error("No images were successfully evaluated.")
        return False

    top1_acc = correct_top1 / total
    top3_acc = correct_top3 / total

    # --- Report ---------------------------------------------------------------
    log.info("=" * 60)
    log.info("Test set assessment results (%d images)", total)
    log.info("=" * 60)
    log.info("Top-1 accuracy : %.4f  (%.2f%%)", top1_acc, top1_acc * 100)
    log.info("Top-3 accuracy : %.4f  (%.2f%%)", top3_acc, top3_acc * 100)

    # Worst 10 classes by accuracy
    class_accs: list[tuple[str, float]] = []
    for label, (correct_count, class_total) in per_class.items():
        acc = correct_count / class_total if class_total > 0 else 0.0
        class_accs.append((label, acc))
    class_accs.sort(key=lambda x: x[1])

    log.info("-" * 60)
    log.info("Worst 10 classes by accuracy:")
    for label, acc in class_accs[:10]:
        # Resolve common name from test split
        common_rows = test_df.loc[test_df["label"] == label, "common_name"]
        common = common_rows.iloc[0] if not common_rows.empty else label
        class_total = per_class[label][1]
        log.info("  %-40s  %.4f  (%d images)", common, acc, class_total)

    # Top 10 confusion pairs
    log.info("-" * 60)
    log.info("Top 10 confusion pairs (true -> predicted : count):")
    for (true_name, pred_name), count in confusion.most_common(10):
        log.info("  %-30s -> %-30s : %d", true_name, pred_name, count)

    # --- Pass / fail ----------------------------------------------------------
    TOP1_THRESHOLD = 0.95
    TOP3_THRESHOLD = 0.99
    passed = top1_acc >= TOP1_THRESHOLD and top3_acc >= TOP3_THRESHOLD

    log.info("=" * 60)
    log.info(
        "Part 1 result: %s  (top-1=%.4f >= %.2f, top-3=%.4f >= %.2f)",
        "PASS" if passed else "FAIL",
        top1_acc, TOP1_THRESHOLD,
        top3_acc, TOP3_THRESHOLD,
    )
    log.info("=" * 60)

    return passed


# ---------------------------------------------------------------------------
# Part 2 — Integration test
# ---------------------------------------------------------------------------


def integration_test() -> bool:
    """Run the Kestrel CLI on 5 sampled test images and verify identification.

    Returns True if at least 1 image is identified as a named species (not
    "Unknown" or "Error").
    """
    test_csv_path = SPLITS_DIR / "test.csv"

    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test split not found: {test_csv_path}")

    test_df = pd.read_csv(test_csv_path)

    # Sample 1 image per species, then pick 5 with a fixed seed
    per_species = (
        test_df.groupby("label", group_keys=False)
        .first()
        .reset_index(drop=True)
    )
    sample_df = per_species.sample(n=min(5, len(per_species)), random_state=42)

    cli_path = Path(__file__).resolve().parent.parent / "analyzer" / "cli.py"
    if not cli_path.exists():
        raise FileNotFoundError(f"Kestrel CLI not found: {cli_path}")

    with tempfile.TemporaryDirectory(prefix="kestrel_integration_") as tmpdir:
        # Copy sampled images into the temp dir with .jpg extension
        for _, row in sample_df.iterrows():
            src = Path(row["path"])
            dest_name = src.stem + ".jpg"
            dest = Path(tmpdir) / dest_name
            shutil.copy2(str(src), str(dest))

        log.info("Running Kestrel CLI on %d images in %s …", len(sample_df), tmpdir)

        result = subprocess.run(
            [sys.executable, str(cli_path), tmpdir, "--no-gpu"],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            log.error("Kestrel CLI exited with code %d.", result.returncode)
            log.error("stdout:\n%s", result.stdout)
            log.error("stderr:\n%s", result.stderr)
            return False

        db_path = Path(tmpdir) / ".kestrel" / "kestrel_database.csv"
        if not db_path.exists():
            log.error(
                "Expected output database not found: %s", db_path
            )
            log.error("CLI stdout:\n%s", result.stdout)
            log.error("CLI stderr:\n%s", result.stderr)
            return False

        db_df = pd.read_csv(db_path)

        # Report each row
        log.info("-" * 60)
        log.info("Integration test results:")
        for _, db_row in db_df.iterrows():
            log.info(
                "  file=%-35s  species=%-30s  confidence=%-6s  family=%-20s  quality=%s",
                db_row.get("filename", ""),
                db_row.get("species", ""),
                db_row.get("species_confidence", ""),
                db_row.get("family", ""),
                db_row.get("quality", ""),
            )

        # Pass if at least one row has a real species
        identified = db_df[
            ~db_df["species"].isin(["Unknown", "Error"])
            & db_df["species"].notna()
            & (db_df["species"].str.strip() != "")
        ]
        passed = len(identified) >= 1

        log.info("-" * 60)
        log.info(
            "Part 2 result: %s  (%d/%d images identified as a named species)",
            "PASS" if passed else "FAIL",
            len(identified),
            len(db_df),
        )

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the trained ONNX model on the test split and run a "
            "Kestrel CLI integration test."
        )
    )
    parser.add_argument(
        "--skip-test-set",
        action="store_true",
        help="Skip Part 1 (test set accuracy assessment).",
    )
    parser.add_argument(
        "--skip-integration",
        action="store_true",
        help="Skip Part 2 (CLI integration test).",
    )
    parser.parse_args()

    results: dict[str, bool | None] = {
        "test_set": None,
        "integration": None,
    }

    args = parser.parse_args()

    if not args.skip_test_set:
        log.info("=" * 60)
        log.info("Part 1: Test set accuracy assessment")
        log.info("=" * 60)
        results["test_set"] = assess_test_set()
    else:
        log.info("Part 1: skipped.")

    if not args.skip_integration:
        log.info("=" * 60)
        log.info("Part 2: Kestrel CLI integration test")
        log.info("=" * 60)
        results["integration"] = integration_test()
    else:
        log.info("Part 2: skipped.")

    # --- Summary --------------------------------------------------------------
    log.info("=" * 60)
    log.info("Assessment summary")
    log.info("=" * 60)

    any_failure = False
    for name, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
            any_failure = True
        log.info("  %-20s : %s", name, status)

    log.info("=" * 60)

    if any_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
