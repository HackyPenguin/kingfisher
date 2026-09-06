"""Read-only, deterministic discovery for historical photo libraries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Callable, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from .historical_store import HistoricalStore


RAW_EXTENSIONS = frozenset(
    {
        ".3fr",
        ".arw",
        ".bay",
        ".cap",
        ".cr2",
        ".cr3",
        ".dcr",
        ".dcs",
        ".dng",
        ".drf",
        ".eip",
        ".erf",
        ".fff",
        ".iiq",
        ".k25",
        ".kdc",
        ".mef",
        ".mos",
        ".mrw",
        ".nef",
        ".nrw",
        ".obm",
        ".orf",
        ".pef",
        ".ptx",
        ".pxn",
        ".r3d",
        ".raf",
        ".raw",
        ".rw2",
        ".rwl",
        ".rwz",
        ".sr2",
        ".srw",
        ".x3f",
    }
)
RENDERED_EXTENSIONS = frozenset(
    {".avif", ".heic", ".heif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | RENDERED_EXTENSIONS
EXCLUDED_DIRECTORY_NAMES = frozenset({".kingfisher"})
RECONCILIATION_BLOCKING_DIAGNOSTICS = frozenset(
    {"directory_unreadable", "entry_unreadable"}
)
_ASSET_NAMESPACE = uuid.UUID("8239620a-c80b-45cb-b15f-87d912bec081")


class SourceChangedDuringHash(RuntimeError):
    """Raised when a source is not stable for the duration of a full hash."""


@dataclass(frozen=True)
class DiscoveredAsset:
    asset_id: str
    library_id: str
    relative_path: str
    display_name: str
    extension: str
    kind: str
    byte_size: int
    mtime_ns: int
    device: int | None
    inode: int | None


@dataclass(frozen=True)
class DiscoveryDiagnostic:
    relative_path: str
    code: str


@dataclass(frozen=True)
class DiscoveryReport:
    assets: tuple[DiscoveredAsset, ...]
    diagnostics: tuple[DiscoveryDiagnostic, ...]


@dataclass(frozen=True)
class SourceFingerprint:
    algorithm: str
    content_digest: str
    byte_size: int
    mtime_ns: int
    device: int | None
    inode: int | None


@dataclass(frozen=True)
class ScanSummary:
    scan_id: str
    status: str
    observed_count: int
    diagnostic_count: int


def _normalise_identifier(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalised = value.strip()
    if not normalised:
        raise ValueError(f"{name} must not be blank")
    return normalised


def _path_sort_key(value: str) -> tuple[str, str]:
    return value.casefold(), value


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_size, value.st_mtime_ns, value.st_dev, value.st_ino


def _asset_id(library_id: str, relative_path: str) -> str:
    return str(uuid.uuid5(_ASSET_NAMESPACE, f"{library_id}\0{relative_path}"))


def discover_library(
    root: Path | str,
    library_id: str,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> DiscoveryReport:
    """Discover supported images recursively without following symlinks.

    Paths and identifiers are root-relative so a remount does not invalidate the
    library. Unsupported files are ignored; unsafe or unreadable entries become
    diagnostics and never become work items.
    """

    library_id = _normalise_identifier(library_id, "library_id")
    root_path = Path(root).resolve(strict=True)
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)

    assets: list[DiscoveredAsset] = []
    diagnostics: list[DiscoveryDiagnostic] = []

    def walk(directory: Path) -> None:
        if should_stop is not None and should_stop():
            return
        try:
            entries = sorted(
                os.scandir(directory),
                key=lambda entry: _path_sort_key(entry.name),
            )
        except OSError:
            relative = directory.relative_to(root_path).as_posix() or "."
            diagnostics.append(DiscoveryDiagnostic(relative, "directory_unreadable"))
            return

        for entry in entries:
            if should_stop is not None and should_stop():
                return
            path = Path(entry.path)
            relative_path = path.relative_to(root_path).as_posix()
            try:
                if entry.is_symlink():
                    diagnostics.append(DiscoveryDiagnostic(relative_path, "symlink_skipped"))
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name in EXCLUDED_DIRECTORY_NAMES:
                        continue
                    walk(path)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                extension = path.suffix.lower()
                if extension not in SUPPORTED_EXTENSIONS:
                    continue
                source_stat = entry.stat(follow_symlinks=False)
            except OSError:
                diagnostics.append(DiscoveryDiagnostic(relative_path, "entry_unreadable"))
                continue

            assets.append(
                DiscoveredAsset(
                    asset_id=_asset_id(library_id, relative_path),
                    library_id=library_id,
                    relative_path=relative_path,
                    display_name=entry.name,
                    extension=extension,
                    kind="raw" if extension in RAW_EXTENSIONS else "rendered",
                    byte_size=source_stat.st_size,
                    mtime_ns=source_stat.st_mtime_ns,
                    device=getattr(source_stat, "st_dev", None),
                    inode=getattr(source_stat, "st_ino", None),
                )
            )

    walk(root_path)
    assets.sort(key=lambda item: _path_sort_key(item.relative_path))
    diagnostics.sort(key=lambda item: (_path_sort_key(item.relative_path), item.code))
    return DiscoveryReport(tuple(assets), tuple(diagnostics))


def hash_file_stably(
    path: Path | str,
    *,
    buffer_size: int = 1024 * 1024,
    stat_function: Callable[..., os.stat_result] = os.stat,
) -> SourceFingerprint:
    """Return a full SHA-256 only when the source is unchanged while reading."""

    if buffer_size <= 0:
        raise ValueError("buffer_size must be positive")
    source_path = Path(path)
    before = stat_function(source_path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("source must be a regular file")

    digest = hashlib.sha256()
    open_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source_path, open_flags)
    with os.fdopen(descriptor, "rb") as source:
        opened_before = os.fstat(source.fileno())
        if _stat_signature(opened_before) != _stat_signature(before):
            raise SourceChangedDuringHash(f"source changed before hashing: {source_path.name}")
        while True:
            chunk = source.read(buffer_size)
            if not chunk:
                break
            digest.update(chunk)
        opened_after = os.fstat(source.fileno())

    after = stat_function(source_path, follow_symlinks=False)
    if not (
        _stat_signature(before)
        == _stat_signature(opened_before)
        == _stat_signature(opened_after)
        == _stat_signature(after)
    ):
        raise SourceChangedDuringHash(f"source changed while hashing: {source_path.name}")

    return SourceFingerprint(
        algorithm="sha256",
        content_digest=digest.hexdigest(),
        byte_size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        device=getattr(after, "st_dev", None),
        inode=getattr(after, "st_ino", None),
    )


class HistoricalIndexer:
    """Coordinate repeatable scanner passes with the transactional store."""

    def __init__(
        self,
        store: "HistoricalStore",
        source_root: Path | str,
        library_id: str,
        *,
        root_config_digest: str = "default",
        mutate_review_proposals: bool = True,
    ) -> None:
        self.store = store
        self.source_root = Path(source_root).resolve(strict=True)
        if self.source_root != store.source_root:
            raise ValueError("indexer source_root must match the store source_root")
        self.library_id = _normalise_identifier(library_id, "library_id")
        self.root_config_digest = _normalise_identifier(root_config_digest, "root_config_digest")
        if not isinstance(mutate_review_proposals, bool):
            raise TypeError("mutate_review_proposals must be a bool")
        self.mutate_review_proposals = mutate_review_proposals

    def run(
        self,
        *,
        scan_id: str | None = None,
        max_items: int | None = None,
        full_hash_audit: bool = False,
        should_stop: Callable[[], bool] | None = None,
    ) -> ScanSummary:
        if max_items is not None and max_items < 0:
            raise ValueError("max_items must be non-negative or None")
        if scan_id is None:
            scan_id = self.store.begin_scan(
                self.library_id,
                self.source_root,
                self.root_config_digest,
            )
        else:
            self.store.restart_scan(scan_id, self.library_id)

        if max_items == 0 or (should_stop is not None and should_stop()):
            return ScanSummary(scan_id, "running", 0, 0)

        report = discover_library(
            self.source_root,
            self.library_id,
            should_stop=should_stop,
        )
        for diagnostic in report.diagnostics:
            self.store.record_scan_error(
                scan_id,
                diagnostic.relative_path,
                diagnostic.code,
            )

        pending_assets = tuple(
            asset
            for asset in report.assets
            if not self.store.scan_observation_matches(scan_id, asset)
        )
        observed_count = 0
        diagnostic_count = len(report.diagnostics)
        source_instability_detected = False
        for asset in pending_assets:
            if should_stop is not None and should_stop():
                break
            if max_items is not None and observed_count >= max_items:
                break
            try:
                self.store.observe_asset(
                    scan_id,
                    asset,
                    self.source_root / asset.relative_path,
                    force_hash=full_hash_audit,
                    supersede_proposals=self.mutate_review_proposals,
                )
            except SourceChangedDuringHash:
                self.store.record_scan_error(
                    scan_id,
                    asset.relative_path,
                    "source_changed_during_hash",
                )
                self.store.mark_seen_without_version(
                    scan_id,
                    asset,
                    supersede_proposals=self.mutate_review_proposals,
                )
                diagnostic_count += 1
                source_instability_detected = True
            observed_count += 1

        traversal_complete = not any(
            diagnostic.code in RECONCILIATION_BLOCKING_DIAGNOSTICS
            for diagnostic in report.diagnostics
        )
        interrupted = should_stop is not None and should_stop()
        if (
            all(
                self.store.scan_observation_matches(scan_id, asset)
                for asset in report.assets
            )
            and traversal_complete
            and not source_instability_detected
            and not interrupted
        ):
            self.store.complete_scan(
                scan_id,
                current_asset_ids=tuple(asset.asset_id for asset in report.assets),
            )
            status = "completed"
        else:
            status = "running"
        return ScanSummary(scan_id, status, observed_count, diagnostic_count)
