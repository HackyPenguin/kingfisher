"""Lightweight folder inspection utilities used by the visualizer.

This module deliberately avoids importing the heavy ML pipeline so the
visualizer can show folder progress without loading models.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Set

try:
    import pandas as pd  # pandas is fast for reading CSVs
except Exception:
    pd = None

try:
    from kestrel_analyzer.config import RAW_EXTENSIONS, JPEG_EXTENSIONS, KESTREL_DIR_NAME, DATABASE_NAME
    from kestrel_analyzer.database import load_database
except Exception:
    # If kestrel_analyzer isn't available, fall back to reasonable defaults.
    RAW_EXTENSIONS = ['.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.raf', '.rw2', '.pef', '.sr2', '.x3f']
    JPEG_EXTENSIONS = ['.jpg', '.jpeg', '.png']
    KESTREL_DIR_NAME = '.kingfisher'
    DATABASE_NAME = 'kingfisher_database.csv'


def _scan_images_in_folder(folder: str) -> tuple[list[str], int, bool]:
    """Scan once and prefer RAW files when present, otherwise JPEG/PNG."""
    try:
        raw_files: list[str] = []
        jpeg_files: list[str] = []
        raw_exts = set(RAW_EXTENSIONS)
        jpeg_exts = set(JPEG_EXTENSIONS)
        with os.scandir(folder) as entries:
            for entry in entries:
                try:
                    if not entry.is_file():
                        continue
                except Exception:
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in raw_exts:
                    raw_files.append(entry.name)
                elif ext in jpeg_exts:
                    jpeg_files.append(entry.name)
        raw_files.sort()
        jpeg_files.sort()
        return (raw_files if raw_files else jpeg_files), len(raw_files), bool(raw_files)
    except Exception:
        return [], 0, False


def _list_images_in_folder(folder: str) -> list:
    files, _, _ = _scan_images_in_folder(folder)
    return files


def _read_processed_filenames(kestrel_dir: str, db_path: str) -> Set[str]:
    if pd is not None:
        try:
            df = pd.read_csv(db_path, usecols=['filename'])
            return set(df['filename'].astype(str).values)
        except Exception:
            pass
    try:
        db, _ = load_database(kestrel_dir, analyzer_name='visualizer-inspector')
        if not db.empty and 'filename' in db.columns:
            return {str(v) for v in db['filename'].values}
    except Exception:
        pass
    return set()


def inspect_folder(path: str) -> Dict[str, int | str | bool]:
    """Return a small summary about a folder.

    Returns keys: 'root' (abs path), 'has_kestrel' (bool), 'total' (int), 'processed' (int), 'db_path' (str)
    """
    result = {'root': '', 'has_kestrel': False, 'total': 0, 'processed': 0, 'db_path': ''}
    if not path:
        return result
    p = path.strip()
    while p and p[-1] in ('/', '\\'):
        p = p[:-1]
    if not p:
        return result
    # If caller passed the .kingfisher folder itself, use the parent as root
    base_name = os.path.basename(p)
    if base_name == KESTREL_DIR_NAME:
        root = os.path.dirname(p)
    else:
        root = p

    result['root'] = root
    files, raw_total, has_raw = _scan_images_in_folder(root)
    total = raw_total if has_raw else len(files)
    result['total'] = total

    kestrel_dir = os.path.join(root, KESTREL_DIR_NAME)
    db_path = os.path.join(kestrel_dir, DATABASE_NAME)
    result['db_path'] = db_path
    if os.path.isfile(db_path):
        result['has_kestrel'] = True
        try:
            processed_set = _read_processed_filenames(kestrel_dir, db_path)
            result['processed'] = int(sum(1 for f in files if f in processed_set))
        except Exception:
            # Fail silently; the visualizer should still work without DB details
            result['processed'] = 0
    return result


def inspect_folders(paths: List[str]) -> Dict[str, Dict]:
    """Batch-inspect many folders quickly.

    Returns a mapping: {path: info_dict}
    The inspection is ordered by path depth (shallow first) to surface
    high-level folders quickly.
    """
    out: Dict[str, Dict] = {}
    if not paths:
        return out
    # Deduplicate and normalize
    uniq = []
    seen = set()
    for p in paths:
        if not p:
            continue
        pp = p.strip()
        while pp and pp[-1] in ('/', '\\'):
            pp = pp[:-1]
        if not pp:
            continue
        if pp in seen:
            continue
        seen.add(pp)
        uniq.append(pp)

    # Sort by path depth ascending (shallow folders first)
    uniq.sort(key=lambda x: (x.count(os.sep), len(x)))

    for p in uniq:
        try:
            info = inspect_folder(p)
            out[p] = info
        except Exception:
            out[p] = {'root': p, 'has_kestrel': False, 'total': 0, 'processed': 0, 'db_path': ''}
    return out
