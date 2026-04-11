"""
Validate downloaded images and build stratified train/val/test splits.

Reads data/uk_bird_taxa_full.json for species metadata, iterates over every
image in data/uk_birds/, validates each file (corrupt or undersized images are
deleted), then writes stratified 80/10/10 CSV splits to data/splits/.

Usage:
    python training/validate_data.py

Re-running is safe: only valid images are included in the splits.
"""

import json
import logging
import statistics
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_SIZE = 100          # Minimum width AND height in pixels
MIN_PER_CLASS = 10      # Minimum valid images required to include a species

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PHOTOS_DIR = DATA_DIR / "uk_birds"
SPECIES_PATH = DATA_DIR / "uk_bird_taxa_full.json"
SPLITS_DIR = DATA_DIR / "splits"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Species metadata
# ---------------------------------------------------------------------------


def load_species_metadata(species_path: Path) -> dict[str, str]:
    """Return a mapping of scientific_name -> common_name from the JSON file."""
    if not species_path.exists():
        raise FileNotFoundError(f"Species metadata not found: {species_path}")

    with species_path.open(encoding="utf-8") as fh:
        species_list: list[dict] = json.load(fh)

    return {entry["scientific_name"]: entry["common_name"] for entry in species_list}


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------


def _validate_image(path: Path) -> tuple[bool, str]:
    """Validate a single image file.

    Returns (is_valid, reason).  reason is empty on success.

    PIL.Image.verify() closes the file handle after running, so the image must
    be reopened to inspect dimensions.
    """
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as exc:
        return False, f"corrupt: {exc}"

    # Reopen after verify() — it leaves the file closed / in an unusable state.
    try:
        with Image.open(path) as img:
            width, height = img.size
    except Exception as exc:
        return False, f"cannot read dimensions: {exc}"

    if width < MIN_SIZE or height < MIN_SIZE:
        return False, f"too small ({width}x{height}px, minimum {MIN_SIZE}px)"

    return True, ""


def validate_species_directory(
    species_dir: Path,
) -> tuple[list[Path], int]:
    """Validate all images in a species directory.

    Deletes corrupt or undersized images in-place.

    Returns (valid_paths, removed_count).
    """
    image_files = [
        p for p in species_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    valid_paths: list[Path] = []
    removed = 0

    for image_path in image_files:
        ok, reason = _validate_image(image_path)
        if ok:
            valid_paths.append(image_path)
        else:
            log.debug("Removing %s — %s", image_path.name, reason)
            image_path.unlink()
            removed += 1

    return valid_paths, removed


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def build_dataframe(
    photos_dir: Path,
    name_map: dict[str, str],
) -> tuple[pd.DataFrame, int, list[str]]:
    """Iterate over species directories, validate images, build a DataFrame.

    Returns:
        df            — DataFrame with columns: path, label, common_name
        total_removed — count of images deleted across all species
        excluded      — list of scientific names excluded (< MIN_PER_CLASS)
    """
    if not photos_dir.exists():
        raise FileNotFoundError(f"Photos directory not found: {photos_dir}")

    species_dirs = sorted(
        d for d in photos_dir.iterdir() if d.is_dir()
    )

    rows: list[dict] = []
    total_removed = 0
    excluded: list[str] = []

    for species_dir in tqdm(species_dirs, desc="Validating species", unit="species"):
        label = species_dir.name  # e.g. "Turdus_merula"
        scientific_name = label.replace("_", " ")
        common_name = name_map.get(scientific_name, scientific_name)

        valid_paths, removed = validate_species_directory(species_dir)
        total_removed += removed

        if len(valid_paths) < MIN_PER_CLASS:
            log.warning(
                "Excluding %s (%s): only %d valid images (minimum %d)",
                scientific_name,
                common_name,
                len(valid_paths),
                MIN_PER_CLASS,
            )
            excluded.append(scientific_name)
            continue

        for image_path in valid_paths:
            rows.append(
                {
                    "path": str(image_path),
                    "label": label,
                    "common_name": common_name,
                }
            )

    df = pd.DataFrame(rows, columns=["path", "label", "common_name"])
    return df, total_removed, excluded


# ---------------------------------------------------------------------------
# Stratified splits
# ---------------------------------------------------------------------------


def make_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified 80/10/10 train/val/test splits.

    Uses random_state=42 for reproducibility.
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=42,
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Statistics reporting
# ---------------------------------------------------------------------------


def report_statistics(
    df: pd.DataFrame,
    total_removed: int,
    excluded: list[str],
    name_map: dict[str, str],
) -> None:
    """Log a summary of the validated dataset."""
    total_valid = len(df)
    included_species = df["label"].nunique()
    excluded_count = len(excluded)

    log.info("=" * 60)
    log.info("Dataset statistics")
    log.info("=" * 60)
    log.info("Total valid images  : %d", total_valid)
    log.info("Images removed      : %d", total_removed)
    log.info("Species included    : %d", included_species)
    log.info("Species excluded    : %d", excluded_count)

    if excluded:
        log.info("Excluded species:")
        for sci_name in excluded:
            log.info("  %s (%s)", sci_name, name_map.get(sci_name, sci_name))

    if included_species == 0:
        log.warning("No species passed the minimum threshold — splits will be empty.")
        return

    counts_per_class: list[int] = (
        df.groupby("label").size().tolist()
    )
    log.info("Per-class image counts:")
    log.info("  min    : %d", min(counts_per_class))
    log.info("  max    : %d", max(counts_per_class))
    log.info(
        "  median : %.1f",
        statistics.median(counts_per_class),
    )
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    name_map = load_species_metadata(SPECIES_PATH)
    log.info("Loaded metadata for %d species", len(name_map))

    df, total_removed, excluded = build_dataframe(PHOTOS_DIR, name_map)

    report_statistics(df, total_removed, excluded, name_map)

    if df.empty:
        log.error("No valid images found — cannot create splits.")
        raise SystemExit(1)

    train_df, val_df, test_df = make_splits(df)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    log.info(
        "Splits written to %s — train: %d, val: %d, test: %d",
        SPLITS_DIR,
        len(train_df),
        len(val_df),
        len(test_df),
    )


if __name__ == "__main__":
    main()
