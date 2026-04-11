#!/usr/bin/env python3
"""
Generate the label and family mapping files that Kestrel's BirdSpeciesClassifier expects.

This script reads:
  - data/uk_bird_taxa_full.json       — per-species taxonomy data
  - training/checkpoints/label_mapping.json — model label index mapping

And writes:
  - analyzer/models/labels.txt             — common names ordered by model index
  - analyzer/models/labels_scispecies.csv  — Species -> Scientific Family
  - analyzer/models/scispecies_dispname.csv — Scientific Family -> Display Name

Usage
-----
Run from the repository root::

    python training/generate_family_maps.py

Optional arguments
------------------
  --dry-run   Validate inputs and print a summary without writing any files.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
MODELS_DIR = Path(__file__).resolve().parent.parent / "analyzer" / "models"

SPECIES_PATH = DATA_DIR / "uk_bird_taxa_full.json"
LABEL_MAPPING_PATH = CHECKPOINT_DIR / "label_mapping.json"

LABELS_TXT_PATH = MODELS_DIR / "labels.txt"
LABELS_SCISPECIES_PATH = MODELS_DIR / "labels_scispecies.csv"
SCISPECIES_DISPNAME_PATH = MODELS_DIR / "scispecies_dispname.csv"

# Sentinel used when a species cannot be matched to a family.
_UNKNOWN_FAMILY = "Unknown"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_taxa(path: Path) -> list[dict]:
    """Load and return the list of species records from the taxa JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Taxa file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array in {path}, got {type(data).__name__}"
        )
    return data


def load_label_mapping(path: Path) -> dict[int, str]:
    """Return a dict mapping integer index -> label string (e.g. 'Erithacus_rubecula')."""
    if not path.exists():
        raise FileNotFoundError(f"Label mapping not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    idx_to_label_raw: dict = raw.get("idx_to_label")
    if idx_to_label_raw is None:
        raise KeyError(
            f"'idx_to_label' key missing from {path}"
        )

    try:
        return {int(k): v for k, v in idx_to_label_raw.items()}
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"idx_to_label keys must be integer strings in {path}"
        ) from exc


# ---------------------------------------------------------------------------
# Build lookup: underscore-label -> species info
# ---------------------------------------------------------------------------

def build_sci_to_info(taxa: list[dict]) -> dict[str, dict]:
    """
    Return a dict keyed by scientific_name with spaces replaced by underscores,
    e.g. 'Cygnus_olor' -> {common_name, scientific_family, display_family, ...}.

    If duplicate keys exist (shouldn't in practice) the last entry wins and a
    warning is emitted.
    """
    lookup: dict[str, dict] = {}
    for entry in taxa:
        sci_name = entry.get("scientific_name", "")
        key = sci_name.replace(" ", "_")
        if not key:
            logger.warning("Taxa entry missing scientific_name, skipping: %s", entry)
            continue
        if key in lookup:
            logger.warning(
                "Duplicate scientific_name key '%s' in taxa; using later entry.", key
            )
        lookup[key] = entry
    return lookup


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_mappings(
    idx_to_label: dict[int, str],
    sci_to_info: dict[str, dict],
) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Build the three output structures from the index-ordered labels.

    Returns
    -------
    ordered_common_names : list[str]
        Common name for each model output index, in ascending index order.
    species_family_rows : list[tuple[str, str]]
        (common_name, scientific_family) in the same index order as above.
    family_display_rows : list[tuple[str, str]]
        (scientific_family, display_name), one row per unique family,
        sorted alphabetically by scientific family name.
    """
    sorted_indices = sorted(idx_to_label.keys())

    ordered_common_names: list[str] = []
    species_family_rows: list[tuple[str, str]] = []
    family_display_map: dict[str, str] = {}  # scientific_family -> display_family

    unknown_labels: list[str] = []

    for idx in sorted_indices:
        label = idx_to_label[idx]
        info = sci_to_info.get(label)

        if info is None:
            logger.warning(
                "Label '%s' (index %d) not found in taxa JSON; "
                "family will be recorded as '%s'.",
                label,
                idx,
                _UNKNOWN_FAMILY,
            )
            unknown_labels.append(label)
            common_name = label.replace("_", " ")
            sci_family = _UNKNOWN_FAMILY
            display_family = _UNKNOWN_FAMILY
        else:
            common_name = info.get("common_name") or label.replace("_", " ")
            sci_family = info.get("scientific_family") or _UNKNOWN_FAMILY
            display_family = info.get("display_family") or sci_family

            if sci_family == _UNKNOWN_FAMILY:
                unknown_labels.append(label)

        ordered_common_names.append(common_name)
        species_family_rows.append((common_name, sci_family))

        # Track the display name for each scientific family.  If we see the same
        # scientific family more than once we expect identical display_family
        # values; warn if they differ.
        if sci_family not in family_display_map:
            family_display_map[sci_family] = display_family
        elif family_display_map[sci_family] != display_family:
            logger.warning(
                "Conflicting display names for family '%s': previously '%s', "
                "now '%s'. Keeping the first value seen.",
                sci_family,
                family_display_map[sci_family],
                display_family,
            )

    family_display_rows = sorted(family_display_map.items(), key=lambda t: t[0])

    return ordered_common_names, species_family_rows, family_display_rows, unknown_labels


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_labels_txt(path: Path, common_names: list[str]) -> None:
    """Write one common name per line, preserving model-index order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for name in common_names:
            fh.write(name + "\n")


def write_labels_scispecies_csv(
    path: Path, rows: list[tuple[str, str]]
) -> None:
    """Write Species, Scientific Family CSV in model-index order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["Species", "Scientific Family"])
    df.to_csv(path, index=False)


def write_scispecies_dispname_csv(
    path: Path, rows: list[tuple[str, str]]
) -> None:
    """Write Scientific Family, Display Name CSV sorted alphabetically by family."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["Scientific Family", "Display Name"])
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> None:
    logger.info("Loading taxa from %s", SPECIES_PATH)
    taxa = load_taxa(SPECIES_PATH)
    logger.info("Loaded %d taxa entries.", len(taxa))

    logger.info("Loading label mapping from %s", LABEL_MAPPING_PATH)
    idx_to_label = load_label_mapping(LABEL_MAPPING_PATH)
    logger.info("Loaded %d label indices.", len(idx_to_label))

    sci_to_info = build_sci_to_info(taxa)

    ordered_names, species_family_rows, family_display_rows, unknown_labels = build_mappings(
        idx_to_label, sci_to_info
    )

    # --- Summary ---
    n_species = len(ordered_names)
    n_families = len(family_display_rows)

    logger.info("Species written to labels.txt:         %d", n_species)
    logger.info("Rows in labels_scispecies.csv:         %d", len(species_family_rows))
    logger.info("Families in scispecies_dispname.csv:   %d", n_families)

    if unknown_labels:
        logger.warning(
            "%d species have '%s' family: %s",
            len(unknown_labels),
            _UNKNOWN_FAMILY,
            ", ".join(unknown_labels),
        )

    if dry_run:
        logger.info("Dry-run mode — no files written.")
        return

    logger.info("Writing %s", LABELS_TXT_PATH)
    write_labels_txt(LABELS_TXT_PATH, ordered_names)

    logger.info("Writing %s", LABELS_SCISPECIES_PATH)
    write_labels_scispecies_csv(LABELS_SCISPECIES_PATH, species_family_rows)

    logger.info("Writing %s", SCISPECIES_DISPNAME_PATH)
    write_scispecies_dispname_csv(SCISPECIES_DISPNAME_PATH, family_display_rows)

    logger.info("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print a summary without writing any files.",
    )
    args = parser.parse_args()

    try:
        main(dry_run=args.dry_run)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
