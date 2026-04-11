"""
Download research-grade observation photos from iNaturalist for UK bird species.

Reads data/uk_bird_taxa_full.json, queries the iNaturalist API for photo IDs,
caches results to data/photo_ids/, then downloads medium-resolution images
(500px) to data/uk_birds/{Scientific_name_with_underscores}/.

Usage:
    python training/download_photos.py

Re-running is safe: cached photo IDs are reused and already-downloaded images
are skipped.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
SPECIES_PATH = REPO_ROOT / "data" / "uk_bird_taxa_full.json"
PHOTO_IDS_DIR = REPO_ROOT / "data" / "photo_ids"
PHOTOS_DIR = REPO_ROOT / "data" / "uk_birds"

MAX_PER_SPECIES = 1000
API_PAGE_SIZE = 200  # iNat API maximum
API_DELAY = 1.1  # seconds between API calls (stay under 60/min)
S3_WORKERS = 8  # parallel download threads

INAT_API_URL = "https://api.inaturalist.org/v1/observations"
S3_BASE_URL = "https://inaturalist-open-data.s3.amazonaws.com/photos"

# Minimum content-length to consider a download valid (guards against stub
# error pages being saved as image files).
MIN_PHOTO_BYTES = 1000


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def fetch_photo_ids(
    taxon_id: str,
    max_photos: int = MAX_PER_SPECIES,
) -> list[tuple[int, str]]:
    """Query iNaturalist for research-grade photo IDs for a taxon.

    Returns a list of (photo_id, extension) tuples, ordered by votes
    (highest quality first).  At most *max_photos* entries are returned.
    """
    results: list[tuple[int, str]] = []
    page = 1

    while len(results) < max_photos:
        want = min(API_PAGE_SIZE, max_photos - len(results))
        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "photos": "true",
            "per_page": want,
            "page": page,
            "order": "desc",
            "order_by": "votes",
        }

        try:
            response = requests.get(INAT_API_URL, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            log.error("API request failed (taxon_id=%s, page=%d): %s", taxon_id, page, exc)
            break

        data = response.json()
        observations = data.get("results", [])

        if not observations:
            break  # No more pages

        for obs in observations:
            for photo in obs.get("photos", []):
                photo_id = photo.get("id")
                url = photo.get("url", "")
                if photo_id is None:
                    continue
                ext = _extension_from_url(url)
                results.append((photo_id, ext))
                if len(results) >= max_photos:
                    return results

        # iNat pages are 0-indexed results; if we got fewer than requested we
        # have exhausted the API for this taxon.
        if len(observations) < want:
            break

        page += 1
        time.sleep(API_DELAY)

    return results


def _extension_from_url(url: str) -> str:
    """Extract the file extension from a photo URL, defaulting to 'jpg'."""
    if not url:
        return "jpg"
    path = urlparse(url).path
    suffix = Path(path).suffix
    return suffix.lstrip(".") if suffix else "jpg"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path(scientific_name: str) -> Path:
    safe_name = scientific_name.replace(" ", "_")
    return PHOTO_IDS_DIR / f"{safe_name}.json"


def load_cached_photo_ids(scientific_name: str) -> list[tuple[int, str]] | None:
    """Return cached photo IDs, or None if no cache file exists."""
    path = _cache_path(scientific_name)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    # Stored as [[photo_id, ext], ...] for JSON serialisability
    return [(entry[0], entry[1]) for entry in raw]


def save_cached_photo_ids(
    scientific_name: str,
    photo_ids: list[tuple[int, str]],
) -> None:
    """Persist photo IDs to the cache directory."""
    PHOTO_IDS_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(scientific_name)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(photo_ids, fh)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _dest_dir(scientific_name: str) -> Path:
    safe_name = scientific_name.replace(" ", "_")
    return PHOTOS_DIR / safe_name


def download_photo(photo_id: int, ext: str, dest_dir: Path) -> bool:
    """Download one medium-resolution photo from the iNat S3 bucket.

    Returns True on success, False on any failure (network error, bad response,
    suspiciously small file).
    """
    dest_path = dest_dir / f"{photo_id}.{ext}"
    if dest_path.exists():
        return True  # Already downloaded

    url = f"{S3_BASE_URL}/{photo_id}/medium.{ext}"
    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        log.debug("Download failed for photo %d: %s", photo_id, exc)
        return False

    if response.status_code != 200:
        log.debug("HTTP %d for photo %d", response.status_code, photo_id)
        return False

    if len(response.content) < MIN_PHOTO_BYTES:
        log.debug(
            "Photo %d too small (%d bytes), skipping",
            photo_id,
            len(response.content),
        )
        return False

    dest_path.write_bytes(response.content)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not SPECIES_PATH.exists():
        log.error("Species file not found: %s", SPECIES_PATH)
        raise FileNotFoundError(SPECIES_PATH)

    with SPECIES_PATH.open(encoding="utf-8") as fh:
        species_list: list[dict] = json.load(fh)

    total_species = len(species_list)
    total_downloaded = 0

    for idx, species in enumerate(species_list, start=1):
        common_name: str = species["common_name"]
        scientific_name: str = species["scientific_name"]
        taxon_id: str = species["taxon_id"]

        dest_dir = _dest_dir(scientific_name)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Count already-downloaded photos
        existing = list(dest_dir.glob("*.*"))
        if len(existing) >= MAX_PER_SPECIES:
            log.info(
                "[%d/%d] %s: already has %d photos, skipping",
                idx,
                total_species,
                common_name,
                len(existing),
            )
            total_downloaded += len(existing)
            continue

        # Load or fetch photo IDs
        photo_ids = load_cached_photo_ids(scientific_name)
        if photo_ids is None:
            log.info(
                "[%d/%d] %s: fetching photo IDs from API ...",
                idx,
                total_species,
                common_name,
            )
            photo_ids = fetch_photo_ids(taxon_id)
            save_cached_photo_ids(scientific_name, photo_ids)
            time.sleep(API_DELAY)

        # Determine which photos still need downloading
        existing_ids = {int(p.stem) for p in existing if p.stem.isdigit()}
        to_download = [
            (pid, ext) for pid, ext in photo_ids if pid not in existing_ids
        ]

        if not to_download:
            log.info(
                "[%d/%d] %s: no new photos to download",
                idx,
                total_species,
                common_name,
            )
            total_downloaded += len(existing)
            continue

        print(
            f"[{idx}/{total_species}] {common_name}: downloading {len(to_download)} photos..."
        )

        succeeded = 0
        with ThreadPoolExecutor(max_workers=S3_WORKERS) as executor:
            futures = {
                executor.submit(download_photo, pid, ext, dest_dir): (pid, ext)
                for pid, ext in to_download
            }
            with tqdm(total=len(to_download), unit="photo", leave=False) as pbar:
                for future in as_completed(futures):
                    try:
                        ok = future.result()
                    except Exception as exc:
                        pid, ext = futures[future]
                        log.warning("Unexpected error for photo %d: %s", pid, exc)
                        ok = False
                    if ok:
                        succeeded += 1
                    pbar.update(1)

        total_downloaded += len(existing) + succeeded
        log.info(
            "[%d/%d] %s: downloaded %d/%d photos",
            idx,
            total_species,
            common_name,
            succeeded,
            len(to_download),
        )

    print(f"\nDone. Total photos downloaded this run: {total_downloaded}")


if __name__ == "__main__":
    main()
