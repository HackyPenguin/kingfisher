"""Transactional SQLite state for the non-destructive historical workflow."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
from typing import Any, Iterator, Mapping
import uuid

from .historical_index import DiscoveredAsset, hash_file_stably
from .review_policy import Decision, ReviewProposal


SCHEMA_VERSION = 3
MANIFEST_SCHEMA_VERSION = 1
_VERSION_NAMESPACE = uuid.UUID("40f11d91-8f27-4058-ae61-e519dd5e85a1")
_ACTIONABLE_DECISIONS = frozenset(
    {
        Decision.CLEAR_AI_REVIEW.value,
        Decision.MANUAL_REVIEW_FOCUS.value,
        Decision.MANUAL_REVIEW_UNCERTAIN.value,
    }
)
_ANALYSIS_FAILURE_CODES = frozenset(
    {
        "decoder_failed",
        "invalid_prediction",
        "provider_failed",
        "source_version_mismatch",
    }
)
_ANALYSIS_SKIP_CODES = frozenset({"source_too_large"})


@dataclass(frozen=True)
class AnalysisAssetVersion:
    """A current, stable source version resolved for analysis."""

    asset_id: str
    asset_version_id: str
    relative_path: str
    source_path: Path
    fingerprint_algorithm: str
    content_digest: str
    byte_size: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
        return True
    except ValueError:
        return False


def _require_identifier(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalised = value.strip()
    if not normalised:
        raise ValueError(f"{name} must not be blank")
    return normalised


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _validate_private_directory(path: Path) -> None:
    details = os.lstat(path)
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("state_root must be a real directory")
    if hasattr(os, "geteuid") and details.st_uid != os.geteuid():
        raise ValueError("state_root must be owned by the runtime user")


def _validate_private_file(
    path: Path,
    *,
    expected_device: int | None = None,
    expected_inode: int | None = None,
) -> os.stat_result:
    details = os.lstat(path)
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ValueError("SQLite state files must be regular files")
    if details.st_nlink != 1:
        raise ValueError("SQLite state files must not have hard links")
    if hasattr(os, "geteuid") and details.st_uid != os.geteuid():
        raise ValueError("SQLite state files must be owned by the runtime user")
    if expected_device is not None and details.st_dev != expected_device:
        raise ValueError("SQLite database identity changed during open")
    if expected_inode is not None and details.st_ino != expected_inode:
        raise ValueError("SQLite database identity changed during open")
    return details


def _securely_open_database(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    if _lexists(path):
        _validate_private_file(path)
    try:
        descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise ValueError("SQLite database must not be a symlink") from error
            raise
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            raise ValueError("SQLite database must be a single-link regular file")
        _validate_private_file(
            path,
            expected_device=details.st_dev,
            expected_inode=details.st_ino,
        )
        return descriptor, details
    except Exception:
        os.close(descriptor)
        raise


class HistoricalStore:
    """Single-writer durable state, always outside the source photo root."""

    def __init__(self, state_root: Path | str, source_root: Path | str) -> None:
        self.source_root = Path(source_root).resolve(strict=True)
        state_path = Path(state_root).expanduser().absolute()
        if _lexists(state_path) and state_path.is_symlink():
            raise ValueError("state_root must not be a symlink")
        resolved_candidate = state_path.resolve(strict=False)
        if _is_within(resolved_candidate, self.source_root):
            raise ValueError("state_root must be outside source_root")
        state_path.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.state_root = state_path.resolve(strict=True)
        if _is_within(self.state_root, self.source_root):
            raise ValueError("state_root must be outside source_root")
        _validate_private_directory(self.state_root)
        os.chmod(self.state_root, 0o700)
        self.database_path = self.state_root / "historical.sqlite3"
        self._connection: sqlite3.Connection | None = None
        self._database_guard_fd: int | None = None
        companions = tuple(
            Path(f"{self.database_path}{suffix}")
            for suffix in ("-journal", "-shm", "-wal")
        )
        for companion in companions:
            if _lexists(companion):
                _validate_private_file(companion)
        descriptor, opened = _securely_open_database(self.database_path)
        self._database_guard_fd = descriptor
        try:
            self._connection = sqlite3.connect(
                self.database_path,
                isolation_level=None,
                timeout=5.0,
            )
            _validate_private_file(
                self.database_path,
                expected_device=opened.st_dev,
                expected_inode=opened.st_ino,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA busy_timeout=5000")
            os.chmod(self.database_path, 0o600)
            _validate_private_file(
                self.database_path,
                expected_device=opened.st_dev,
                expected_inode=opened.st_ino,
            )
            for companion in companions:
                if _lexists(companion):
                    _validate_private_file(companion)
                    os.chmod(companion, 0o600)
            self._migrate()
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "HistoricalStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._database_guard_fd is not None:
            os.close(self._database_guard_fd)
            self._database_guard_fd = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("store is closed")
        return self._connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()

    def _migrate(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS libraries (
                library_id TEXT PRIMARY KEY,
                root_identity TEXT NOT NULL,
                created_at TEXT NOT NULL,
                retired_at TEXT
            );

            CREATE TABLE IF NOT EXISTS scan_runs (
                scan_id TEXT PRIMARY KEY,
                library_id TEXT NOT NULL REFERENCES libraries(library_id),
                root_config_digest TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
                started_at TEXT NOT NULL,
                completed_at TEXT,
                last_checkpoint_path TEXT
            );

            CREATE TABLE IF NOT EXISTS assets (
                asset_id TEXT PRIMARY KEY,
                library_id TEXT NOT NULL REFERENCES libraries(library_id),
                canonical_relative_path TEXT NOT NULL,
                display_name TEXT NOT NULL,
                extension TEXT NOT NULL,
                kind TEXT NOT NULL CHECK (kind IN ('raw', 'rendered')),
                first_seen_at TEXT NOT NULL,
                last_seen_scan_id TEXT REFERENCES scan_runs(scan_id),
                state TEXT NOT NULL CHECK (state IN ('active', 'missing')),
                current_asset_version_id TEXT REFERENCES asset_versions(asset_version_id),
                current_byte_size INTEGER,
                current_mtime_ns INTEGER,
                UNIQUE(library_id, canonical_relative_path)
            );

            CREATE TABLE IF NOT EXISTS asset_versions (
                asset_version_id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL REFERENCES assets(asset_id),
                fingerprint_algorithm TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                byte_size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                device INTEGER,
                inode INTEGER,
                observed_at TEXT NOT NULL,
                source_stable INTEGER NOT NULL CHECK (source_stable = 1),
                UNIQUE(asset_id, fingerprint_algorithm, content_digest)
            );

            CREATE TABLE IF NOT EXISTS scan_observations (
                scan_id TEXT NOT NULL REFERENCES scan_runs(scan_id) ON DELETE CASCADE,
                asset_id TEXT NOT NULL REFERENCES assets(asset_id),
                asset_version_id TEXT REFERENCES asset_versions(asset_version_id),
                relative_path TEXT NOT NULL,
                byte_size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                PRIMARY KEY(scan_id, asset_id)
            );

            CREATE TABLE IF NOT EXISTS scan_errors (
                error_id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id TEXT NOT NULL REFERENCES scan_runs(scan_id) ON DELETE CASCADE,
                relative_path TEXT NOT NULL,
                code TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analysis_runs (
                analysis_run_id TEXT PRIMARY KEY,
                analyzer_version TEXT NOT NULL,
                model_digest TEXT NOT NULL,
                policy_digest TEXT NOT NULL,
                config_digest TEXT NOT NULL,
                canonical_config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(analyzer_version, model_digest, policy_digest, config_digest)
            );

            CREATE TABLE IF NOT EXISTS analysis_results (
                result_id TEXT PRIMARY KEY,
                asset_version_id TEXT NOT NULL REFERENCES asset_versions(asset_version_id),
                analysis_run_id TEXT NOT NULL REFERENCES analysis_runs(analysis_run_id),
                output_schema_version INTEGER NOT NULL,
                canonical_output_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('analysed', 'superseded')),
                created_at TEXT NOT NULL,
                UNIQUE(asset_version_id, analysis_run_id)
            );

            CREATE TABLE IF NOT EXISTS analysis_attempt_failures (
                attempt_id TEXT PRIMARY KEY,
                asset_version_id TEXT NOT NULL REFERENCES asset_versions(asset_version_id),
                analysis_run_id TEXT NOT NULL REFERENCES analysis_runs(analysis_run_id),
                error_code TEXT NOT NULL CHECK (error_code IN (
                    'decoder_failed', 'invalid_prediction', 'provider_failed',
                    'source_version_mismatch'
                )),
                retryable INTEGER NOT NULL CHECK (retryable = 1),
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analysis_terminal_skips (
                skip_id TEXT PRIMARY KEY,
                asset_version_id TEXT NOT NULL REFERENCES asset_versions(asset_version_id),
                analysis_run_id TEXT NOT NULL REFERENCES analysis_runs(analysis_run_id),
                reason_code TEXT NOT NULL CHECK (reason_code IN ('source_too_large')),
                created_at TEXT NOT NULL,
                UNIQUE(asset_version_id, analysis_run_id)
            );

            CREATE TRIGGER IF NOT EXISTS analysis_terminal_skips_no_update
            BEFORE UPDATE ON analysis_terminal_skips
            BEGIN
                SELECT RAISE(ABORT, 'analysis terminal skips are immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS analysis_terminal_skips_no_delete
            BEFORE DELETE ON analysis_terminal_skips
            BEGIN
                SELECT RAISE(ABORT, 'analysis terminal skips are immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS analysis_attempt_failures_no_update
            BEFORE UPDATE ON analysis_attempt_failures
            BEGIN
                SELECT RAISE(ABORT, 'analysis attempt failures are immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS analysis_attempt_failures_no_delete
            BEFORE DELETE ON analysis_attempt_failures
            BEGIN
                SELECT RAISE(ABORT, 'analysis attempt failures are immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS analysis_runs_no_update
            BEFORE UPDATE ON analysis_runs
            BEGIN
                SELECT RAISE(ABORT, 'analysis runs are immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS analysis_runs_no_delete
            BEFORE DELETE ON analysis_runs
            BEGIN
                SELECT RAISE(ABORT, 'analysis runs are immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS analysis_results_no_update
            BEFORE UPDATE ON analysis_results
            BEGIN
                SELECT RAISE(ABORT, 'analysis results are immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS analysis_results_no_delete
            BEFORE DELETE ON analysis_results
            BEGIN
                SELECT RAISE(ABORT, 'analysis results are immutable');
            END;

            CREATE TABLE IF NOT EXISTS review_proposals (
                proposal_id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL REFERENCES assets(asset_id),
                analysis_result_id TEXT NOT NULL REFERENCES analysis_results(result_id),
                decision TEXT NOT NULL CHECK (decision IN (
                    'none', 'protected_keep', 'clear_ai_review',
                    'manual_review_focus', 'manual_review_uncertain'
                )),
                canonical_delta_json TEXT NOT NULL,
                lifecycle TEXT NOT NULL CHECK (lifecycle IN ('proposed', 'superseded', 'applied')),
                created_at TEXT NOT NULL,
                supersedes_proposal_id TEXT REFERENCES review_proposals(proposal_id)
            );

            CREATE UNIQUE INDEX IF NOT EXISTS one_current_proposal_per_asset
                ON review_proposals(asset_id) WHERE lifecycle = 'proposed';

            CREATE TABLE IF NOT EXISTS application_operations (
                operation_id TEXT PRIMARY KEY,
                proposal_id TEXT NOT NULL UNIQUE REFERENCES review_proposals(proposal_id),
                asset_id TEXT NOT NULL REFERENCES assets(asset_id),
                expected_receipt_json TEXT,
                status TEXT NOT NULL CHECK (status IN (
                    'prepared', 'applied', 'superseded', 'needs_manual_recovery'
                )),
                prepared_at TEXT NOT NULL,
                applied_at TEXT,
                post_apply_metadata_revision TEXT,
                exact_applied_keyword TEXT,
                exact_applied_colour TEXT,
                failure_reason TEXT
            );
            """
        )
        for version in range(2, SCHEMA_VERSION + 1):
            self.connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (version, _utc_now()),
            )

    def pragma_value(self, name: str) -> Any:
        if name not in {"journal_mode", "foreign_keys", "synchronous", "busy_timeout"}:
            raise ValueError("unsupported pragma")
        row = self.connection.execute(f"PRAGMA {name}").fetchone()
        return row[0]

    def integrity_check(self) -> str:
        return str(self.connection.execute("PRAGMA integrity_check").fetchone()[0])

    def begin_scan(self, library_id: str, source_root: Path, root_config_digest: str) -> str:
        library_id = _require_identifier(library_id, "library_id")
        root_config_digest = _require_identifier(root_config_digest, "root_config_digest")
        resolved_root = Path(source_root).resolve(strict=True)
        if resolved_root != self.source_root:
            raise ValueError("scan source_root must match the store source_root")
        # Identity is explicitly configured and independent of the current
        # mount path, so remounting a NAS library does not invalidate assets.
        root_identity = library_id
        scan_id = str(uuid.uuid4())
        now = _utc_now()
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT root_identity FROM libraries WHERE library_id = ?",
                (library_id,),
            ).fetchone()
            if existing is not None and existing[0] != root_identity:
                raise ValueError("library identity is inconsistent with the existing store")
            connection.execute(
                "INSERT OR IGNORE INTO libraries(library_id, root_identity, created_at) VALUES (?, ?, ?)",
                (library_id, root_identity, now),
            )
            connection.execute(
                """
                INSERT INTO scan_runs(
                    scan_id, library_id, root_config_digest, status, started_at
                ) VALUES (?, ?, ?, 'running', ?)
                """,
                (scan_id, library_id, root_config_digest, now),
            )
        return scan_id

    def restart_scan(self, scan_id: str, library_id: str) -> None:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT library_id, status FROM scan_runs WHERE scan_id = ?",
                (scan_id,),
            ).fetchone()
            if row is None:
                raise ValueError("unknown scan_id")
            if row["library_id"] != library_id:
                raise ValueError("scan belongs to another library")
            if row["status"] != "running":
                raise ValueError("only an interrupted running scan can be resumed")

    def record_scan_error(self, scan_id: str, relative_path: str, code: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO scan_errors(scan_id, relative_path, code, created_at)
                SELECT ?, ?, ?, ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM scan_errors
                    WHERE scan_id = ? AND relative_path = ? AND code = ?
                )
                """,
                (
                    scan_id,
                    relative_path,
                    code,
                    _utc_now(),
                    scan_id,
                    relative_path,
                    code,
                ),
            )

    def scan_observation_matches(
        self,
        scan_id: str,
        asset: DiscoveredAsset,
    ) -> bool:
        row = self.connection.execute(
            """
            SELECT asset_version_id, byte_size, mtime_ns
            FROM scan_observations
            WHERE scan_id = ? AND asset_id = ?
            """,
            (scan_id, asset.asset_id),
        ).fetchone()
        return bool(
            row is not None
            and row["asset_version_id"] is not None
            and row["byte_size"] == asset.byte_size
            and row["mtime_ns"] == asset.mtime_ns
        )

    def _upsert_asset(self, connection: sqlite3.Connection, scan_id: str, asset: DiscoveredAsset) -> None:
        now = _utc_now()
        existing = connection.execute(
            "SELECT asset_id FROM assets WHERE library_id = ? AND canonical_relative_path = ?",
            (asset.library_id, asset.relative_path),
        ).fetchone()
        if existing is not None and existing["asset_id"] != asset.asset_id:
            raise ValueError("canonical path resolved to an inconsistent asset identity")
        connection.execute(
            """
            INSERT INTO assets(
                asset_id, library_id, canonical_relative_path, display_name,
                extension, kind, first_seen_at, last_seen_scan_id, state,
                current_byte_size, current_mtime_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
            ON CONFLICT(asset_id) DO UPDATE SET
                display_name = excluded.display_name,
                extension = excluded.extension,
                kind = excluded.kind,
                last_seen_scan_id = excluded.last_seen_scan_id,
                state = 'active'
            """,
            (
                asset.asset_id,
                asset.library_id,
                asset.relative_path,
                asset.display_name,
                asset.extension,
                asset.kind,
                now,
                scan_id,
                asset.byte_size,
                asset.mtime_ns,
            ),
        )

    def observe_asset(
        self,
        scan_id: str,
        asset: DiscoveredAsset,
        source_path: Path,
        *,
        force_hash: bool = False,
        supersede_proposals: bool = True,
    ) -> str:
        resolved_source = Path(source_path).resolve(strict=True)
        expected_source = self.source_root / asset.relative_path
        if resolved_source != expected_source or not _is_within(resolved_source, self.source_root):
            raise ValueError("source path does not match the indexed root-relative path")
        current = self.connection.execute(
            """
            SELECT current_asset_version_id, current_byte_size, current_mtime_ns
            FROM assets WHERE asset_id = ?
            """,
            (asset.asset_id,),
        ).fetchone()
        unchanged = (
            current is not None
            and current["current_asset_version_id"] is not None
            and current["current_byte_size"] == asset.byte_size
            and current["current_mtime_ns"] == asset.mtime_ns
        )
        prior_version_id = (
            None if current is None else current["current_asset_version_id"]
        )
        fingerprint = None if unchanged and not force_hash else hash_file_stably(resolved_source)
        if fingerprint is None:
            version_id = str(current["current_asset_version_id"])
            byte_size = asset.byte_size
            mtime_ns = asset.mtime_ns
        else:
            version_id = str(
                uuid.uuid5(
                    _VERSION_NAMESPACE,
                    f"{asset.asset_id}\0{fingerprint.algorithm}\0{fingerprint.content_digest}",
                )
            )
            byte_size = fingerprint.byte_size
            mtime_ns = fingerprint.mtime_ns

        with self.transaction() as connection:
            self._upsert_asset(connection, scan_id, asset)
            if fingerprint is not None:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO asset_versions(
                        asset_version_id, asset_id, fingerprint_algorithm,
                        content_digest, byte_size, mtime_ns, device, inode,
                        observed_at, source_stable
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        version_id,
                        asset.asset_id,
                        fingerprint.algorithm,
                        fingerprint.content_digest,
                        fingerprint.byte_size,
                        fingerprint.mtime_ns,
                        fingerprint.device,
                        fingerprint.inode,
                        _utc_now(),
                    ),
                )
            if (
                supersede_proposals
                and prior_version_id is not None
                and prior_version_id != version_id
            ):
                connection.execute(
                    """
                    UPDATE review_proposals SET lifecycle = 'superseded'
                    WHERE asset_id = ? AND lifecycle = 'proposed'
                    """,
                    (asset.asset_id,),
                )
            connection.execute(
                """
                UPDATE assets SET current_asset_version_id = ?, current_byte_size = ?,
                    current_mtime_ns = ?, last_seen_scan_id = ?, state = 'active'
                WHERE asset_id = ?
                """,
                (version_id, byte_size, mtime_ns, scan_id, asset.asset_id),
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO scan_observations(
                    scan_id, asset_id, asset_version_id, relative_path, byte_size, mtime_ns
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (scan_id, asset.asset_id, version_id, asset.relative_path, byte_size, mtime_ns),
            )
            connection.execute(
                "UPDATE scan_runs SET last_checkpoint_path = ? WHERE scan_id = ?",
                (asset.relative_path, scan_id),
            )
        return version_id

    def mark_seen_without_version(
        self,
        scan_id: str,
        asset: DiscoveredAsset,
        *,
        supersede_proposals: bool = True,
    ) -> None:
        with self.transaction() as connection:
            self._upsert_asset(connection, scan_id, asset)
            if supersede_proposals:
                connection.execute(
                    """
                    UPDATE review_proposals SET lifecycle = 'superseded'
                    WHERE asset_id = ? AND lifecycle = 'proposed'
                    """,
                    (asset.asset_id,),
                )
            connection.execute(
                """
                UPDATE assets SET current_asset_version_id = NULL,
                    current_byte_size = ?, current_mtime_ns = ?,
                    last_seen_scan_id = ?, state = 'active'
                WHERE asset_id = ?
                """,
                (
                    asset.byte_size,
                    asset.mtime_ns,
                    scan_id,
                    asset.asset_id,
                ),
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO scan_observations(
                    scan_id, asset_id, asset_version_id, relative_path, byte_size, mtime_ns
                ) VALUES (?, ?, NULL, ?, ?, ?)
                """,
                (scan_id, asset.asset_id, asset.relative_path, asset.byte_size, asset.mtime_ns),
            )
            connection.execute(
                "UPDATE scan_runs SET last_checkpoint_path = ? WHERE scan_id = ?",
                (asset.relative_path, scan_id),
            )

    def complete_scan(
        self,
        scan_id: str,
        *,
        current_asset_ids: tuple[str, ...] | None = None,
    ) -> None:
        now = _utc_now()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT library_id, status FROM scan_runs WHERE scan_id = ?",
                (scan_id,),
            ).fetchone()
            if row is None or row["status"] != "running":
                raise ValueError("scan is not running")
            if current_asset_ids is not None:
                connection.execute(
                    "CREATE TEMP TABLE current_scan_assets(asset_id TEXT PRIMARY KEY)"
                )
                connection.executemany(
                    "INSERT INTO current_scan_assets(asset_id) VALUES (?)",
                    ((asset_id,) for asset_id in current_asset_ids),
                )
                connection.execute(
                    """
                    DELETE FROM scan_observations
                    WHERE scan_id = ?
                      AND asset_id NOT IN (SELECT asset_id FROM current_scan_assets)
                    """,
                    (scan_id,),
                )
                connection.execute("DROP TABLE current_scan_assets")
            connection.execute(
                """
                UPDATE assets SET state = 'missing'
                WHERE library_id = ?
                  AND asset_id NOT IN (
                      SELECT asset_id FROM scan_observations WHERE scan_id = ?
                  )
                """,
                (row["library_id"], scan_id),
            )
            connection.execute(
                "UPDATE scan_runs SET status = 'completed', completed_at = ? WHERE scan_id = ?",
                (now, scan_id),
            )

    def asset_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0])

    def asset_version_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM asset_versions").fetchone()[0])

    def list_asset_paths(self) -> tuple[str, ...]:
        rows = self.connection.execute(
            """
            SELECT canonical_relative_path FROM assets
            WHERE state = 'active'
            ORDER BY canonical_relative_path COLLATE NOCASE, canonical_relative_path
            """
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def asset_id(self, library_id: str, relative_path: str) -> str:
        row = self.connection.execute(
            "SELECT asset_id FROM assets WHERE library_id = ? AND canonical_relative_path = ?",
            (library_id, relative_path),
        ).fetchone()
        if row is None:
            raise ValueError("unknown asset")
        return str(row[0])

    def asset_state(self, library_id: str, relative_path: str) -> str:
        row = self.connection.execute(
            "SELECT state FROM assets WHERE library_id = ? AND canonical_relative_path = ?",
            (library_id, relative_path),
        ).fetchone()
        if row is None:
            raise ValueError("unknown asset")
        return str(row[0])

    def current_version_id(self, library_id: str, relative_path: str) -> str:
        row = self.connection.execute(
            """
            SELECT current_asset_version_id FROM assets
            WHERE library_id = ? AND canonical_relative_path = ? AND state = 'active'
            """,
            (library_id, relative_path),
        ).fetchone()
        if row is None or row[0] is None:
            raise ValueError("asset has no stable current version")
        return str(row[0])

    def resolve_analysis_asset(
        self,
        library_id: str,
        relative_path: str,
    ) -> AnalysisAssetVersion:
        """Resolve only an active current version beneath ``source_root``."""

        library_id = _require_identifier(library_id, "library_id")
        relative_path = _require_identifier(relative_path, "relative_path")
        row = self.connection.execute(
            """
            SELECT a.asset_id, a.current_asset_version_id,
                   a.canonical_relative_path, v.fingerprint_algorithm,
                   v.content_digest, v.byte_size
            FROM assets AS a
            JOIN asset_versions AS v
              ON v.asset_version_id = a.current_asset_version_id
            WHERE a.library_id = ?
              AND a.canonical_relative_path = ?
              AND a.state = 'active'
              AND v.source_stable = 1
            """,
            (library_id, relative_path),
        ).fetchone()
        if row is None:
            raise ValueError("asset has no stable current version")

        lexical_path = self.source_root / str(row["canonical_relative_path"])
        resolved_path = lexical_path.resolve(strict=True)
        if not _is_within(resolved_path, self.source_root):
            raise ValueError("analysis source must be beneath source_root")
        if resolved_path != lexical_path:
            raise ValueError("analysis source path must not traverse symlinks")
        if not resolved_path.is_file():
            raise ValueError("analysis source must be a regular file")
        return AnalysisAssetVersion(
            asset_id=str(row["asset_id"]),
            asset_version_id=str(row["current_asset_version_id"]),
            relative_path=str(row["canonical_relative_path"]),
            source_path=resolved_path,
            fingerprint_algorithm=str(row["fingerprint_algorithm"]),
            content_digest=str(row["content_digest"]),
            byte_size=int(row["byte_size"]),
        )

    def assert_current_analysis_asset(self, asset: AnalysisAssetVersion) -> None:
        if not isinstance(asset, AnalysisAssetVersion):
            raise TypeError("asset must be an AnalysisAssetVersion")
        row = self.connection.execute(
            """
            SELECT current_asset_version_id, state
            FROM assets WHERE asset_id = ?
            """,
            (asset.asset_id,),
        ).fetchone()
        if (
            row is None
            or row["state"] != "active"
            or row["current_asset_version_id"] != asset.asset_version_id
        ):
            raise ValueError("indexed asset version is no longer current")

    def scan_status(self, scan_id: str) -> str:
        row = self.connection.execute(
            "SELECT status FROM scan_runs WHERE scan_id = ?",
            (scan_id,),
        ).fetchone()
        if row is None:
            raise ValueError("unknown scan")
        return str(row[0])

    def scan_checkpoint(self, scan_id: str) -> str | None:
        row = self.connection.execute(
            "SELECT last_checkpoint_path FROM scan_runs WHERE scan_id = ?",
            (scan_id,),
        ).fetchone()
        if row is None:
            raise ValueError("unknown scan")
        return None if row[0] is None else str(row[0])

    def proposal_lifecycle(self, proposal_id: str) -> str:
        row = self.connection.execute(
            "SELECT lifecycle FROM review_proposals WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        if row is None:
            raise ValueError("unknown proposal")
        return str(row[0])

    def ensure_analysis_run(
        self,
        analyzer_version: str,
        model_digest: str,
        policy_digest: str,
        config: Mapping[str, Any],
    ) -> str:
        analyzer_version = _require_identifier(analyzer_version, "analyzer_version")
        model_digest = _require_identifier(model_digest, "model_digest")
        policy_digest = _require_identifier(policy_digest, "policy_digest")
        canonical_config = _canonical_json(config)
        config_digest = hashlib.sha256(canonical_config.encode("utf-8")).hexdigest()
        identity = _canonical_json(
            {
                "analyzer_version": analyzer_version,
                "config_digest": config_digest,
                "model_digest": model_digest,
                "policy_digest": policy_digest,
            }
        )
        run_id = "analysis-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO analysis_runs(
                    analysis_run_id, analyzer_version, model_digest, policy_digest,
                    config_digest, canonical_config_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    analyzer_version,
                    model_digest,
                    policy_digest,
                    config_digest,
                    canonical_config,
                    _utc_now(),
                ),
            )
        return run_id

    def stale_asset_paths(
        self,
        analysis_run_id: str,
        *,
        library_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[str, ...]:
        if limit is not None and (not isinstance(limit, int) or limit < 0):
            raise ValueError("limit must be a non-negative integer or None")
        parameters: tuple[str | int, ...] = (analysis_run_id, analysis_run_id)
        library_filter = ""
        if library_id is not None:
            library_filter = "AND a.library_id = ?"
            parameters = (
                analysis_run_id,
                analysis_run_id,
                _require_identifier(library_id, "library_id"),
            )
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            parameters = (*parameters, limit)
        rows = self.connection.execute(
            f"""
            SELECT a.canonical_relative_path
            FROM assets AS a
            WHERE a.state = 'active'
              AND a.current_asset_version_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM analysis_results AS r
                  WHERE r.asset_version_id = a.current_asset_version_id
                    AND r.analysis_run_id = ?
                    AND r.status = 'analysed'
            )
              AND NOT EXISTS (
                  SELECT 1 FROM analysis_terminal_skips AS s
                  WHERE s.asset_version_id = a.current_asset_version_id
                    AND s.analysis_run_id = ?
              )
              {library_filter}
            ORDER BY a.canonical_relative_path COLLATE NOCASE, a.canonical_relative_path
            {limit_clause}
            """,
            parameters,
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def stale_asset_count(
        self,
        analysis_run_id: str,
        *,
        library_id: str | None = None,
    ) -> int:
        parameters: tuple[str, ...] = (analysis_run_id, analysis_run_id)
        library_filter = ""
        if library_id is not None:
            library_filter = "AND a.library_id = ?"
            parameters = (
                analysis_run_id,
                analysis_run_id,
                _require_identifier(library_id, "library_id"),
            )
        row = self.connection.execute(
            f"""
            SELECT COUNT(*)
            FROM assets AS a
            WHERE a.state = 'active'
              AND a.current_asset_version_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM analysis_results AS r
                  WHERE r.asset_version_id = a.current_asset_version_id
                    AND r.analysis_run_id = ?
                    AND r.status = 'analysed'
              )
              AND NOT EXISTS (
                  SELECT 1 FROM analysis_terminal_skips AS s
                  WHERE s.asset_version_id = a.current_asset_version_id
                    AND s.analysis_run_id = ?
              )
              {library_filter}
            """,
            parameters,
        ).fetchone()
        return int(row[0])

    def record_analysis_result(
        self,
        result_id: str,
        asset_version_id: str,
        analysis_run_id: str,
        output: Mapping[str, Any],
        *,
        output_schema_version: int = 1,
    ) -> None:
        result_id = _require_identifier(result_id, "result_id")
        canonical_output = _canonical_json(output)
        expected = (
            result_id,
            asset_version_id,
            analysis_run_id,
            output_schema_version,
            canonical_output,
        )
        with self.transaction() as connection:
            existing = connection.execute(
                """
                SELECT result_id, asset_version_id, analysis_run_id,
                       output_schema_version, canonical_output_json
                FROM analysis_results
                WHERE result_id = ? OR (asset_version_id = ? AND analysis_run_id = ?)
                """,
                (result_id, asset_version_id, analysis_run_id),
            ).fetchone()
            if existing is not None:
                actual = tuple(existing[key] for key in existing.keys())
                if actual != expected:
                    raise ValueError("analysis results are immutable")
                return
            connection.execute(
                """
                INSERT INTO analysis_results(
                    result_id, asset_version_id, analysis_run_id,
                    output_schema_version, canonical_output_json, status, created_at
                ) VALUES (?, ?, ?, ?, ?, 'analysed', ?)
                """,
                (*expected, _utc_now()),
            )

    def analysis_result(
        self,
        asset_version_id: str,
        analysis_run_id: str,
    ) -> tuple[str, Mapping[str, Any]] | None:
        row = self.connection.execute(
            """
            SELECT result_id, canonical_output_json
            FROM analysis_results
            WHERE asset_version_id = ? AND analysis_run_id = ?
              AND status = 'analysed'
            """,
            (asset_version_id, analysis_run_id),
        ).fetchone()
        if row is None:
            return None
        return str(row["result_id"]), json.loads(row["canonical_output_json"])

    def record_analysis_failure(
        self,
        asset_version_id: str,
        analysis_run_id: str,
        error_code: str,
    ) -> str:
        """Append one retryable failure without retaining exception details."""

        if error_code not in _ANALYSIS_FAILURE_CODES:
            raise ValueError("unsupported analysis failure code")
        attempt_id = str(uuid.uuid4())
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO analysis_attempt_failures(
                    attempt_id, asset_version_id, analysis_run_id,
                    error_code, retryable, created_at
                ) VALUES (?, ?, ?, ?, 1, ?)
                """,
                (
                    attempt_id,
                    asset_version_id,
                    analysis_run_id,
                    error_code,
                    _utc_now(),
                ),
            )
        return attempt_id

    def record_analysis_skip(
        self,
        asset_version_id: str,
        analysis_run_id: str,
        reason_code: str,
    ) -> str:
        """Persist one immutable terminal exclusion for a version/run pair."""

        if reason_code not in _ANALYSIS_SKIP_CODES:
            raise ValueError("unsupported analysis skip code")
        skip_id = str(
            uuid.uuid5(
                _VERSION_NAMESPACE,
                f"skip\0{asset_version_id}\0{analysis_run_id}\0{reason_code}",
            )
        )
        with self.transaction() as connection:
            existing = connection.execute(
                """
                SELECT skip_id, reason_code
                FROM analysis_terminal_skips
                WHERE asset_version_id = ? AND analysis_run_id = ?
                """,
                (asset_version_id, analysis_run_id),
            ).fetchone()
            if existing is not None:
                if existing["skip_id"] != skip_id or existing["reason_code"] != reason_code:
                    raise ValueError("analysis terminal skips are immutable")
                return skip_id
            connection.execute(
                """
                INSERT INTO analysis_terminal_skips(
                    skip_id, asset_version_id, analysis_run_id,
                    reason_code, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    skip_id,
                    asset_version_id,
                    analysis_run_id,
                    reason_code,
                    _utc_now(),
                ),
            )
        return skip_id

    def analysis_skips(self) -> tuple[Mapping[str, Any], ...]:
        rows = self.connection.execute(
            """
            SELECT skip_id, asset_version_id, analysis_run_id,
                   reason_code, created_at
            FROM analysis_terminal_skips
            ORDER BY created_at, skip_id
            """
        ).fetchall()
        return tuple(dict(row) for row in rows)

    def analysis_failures(self) -> tuple[Mapping[str, Any], ...]:
        rows = self.connection.execute(
            """
            SELECT attempt_id, asset_version_id, analysis_run_id,
                   error_code, retryable, created_at
            FROM analysis_attempt_failures
            ORDER BY created_at, attempt_id
            """
        ).fetchall()
        return tuple(dict(row) for row in rows)

    def analysis_result_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM analysis_results").fetchone()[0])

    def review_proposal_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM review_proposals").fetchone()[0])

    def record_review_proposal(
        self,
        proposal_id: str,
        asset_id: str,
        result_id: str,
        proposal: ReviewProposal,
    ) -> None:
        if not isinstance(proposal, ReviewProposal):
            raise TypeError("proposal must be a ReviewProposal")
        proposal_id = _require_identifier(proposal_id, "proposal_id")
        if proposal.result_id != result_id:
            raise ValueError("proposal result does not match result_id")
        canonical_delta = _canonical_json(proposal.to_dict())
        existing = self.connection.execute(
            """
            SELECT asset_id, analysis_result_id, decision, canonical_delta_json
            FROM review_proposals WHERE proposal_id = ?
            """,
            (proposal_id,),
        ).fetchone()
        expected = (asset_id, result_id, proposal.decision.value, canonical_delta)
        if existing is not None:
            actual = tuple(existing[key] for key in existing.keys())
            if actual != expected:
                raise ValueError("review proposals are immutable")
            return

        result_asset = self.connection.execute(
            """
            SELECT v.asset_id
            FROM analysis_results AS r
            JOIN asset_versions AS v ON v.asset_version_id = r.asset_version_id
            WHERE r.result_id = ?
            """,
            (result_id,),
        ).fetchone()
        if result_asset is None or result_asset[0] != asset_id:
            raise ValueError("result does not belong to asset")

        with self.transaction() as connection:
            previous = connection.execute(
                "SELECT proposal_id FROM review_proposals WHERE asset_id = ? AND lifecycle = 'proposed'",
                (asset_id,),
            ).fetchone()
            if previous is not None:
                connection.execute(
                    "UPDATE review_proposals SET lifecycle = 'superseded' WHERE proposal_id = ?",
                    (previous[0],),
                )
            connection.execute(
                """
                INSERT INTO review_proposals(
                    proposal_id, asset_id, analysis_result_id, decision,
                    canonical_delta_json, lifecycle, created_at, supersedes_proposal_id
                ) VALUES (?, ?, ?, ?, ?, 'proposed', ?, ?)
                """,
                (
                    proposal_id,
                    asset_id,
                    result_id,
                    proposal.decision.value,
                    canonical_delta,
                    _utc_now(),
                    None if previous is None else previous[0],
                ),
            )

    def export_dry_run_manifest(self, destination: Path | str) -> bytes:
        destination_path = Path(destination).resolve(strict=False)
        if not _is_within(destination_path, self.state_root):
            raise ValueError("manifest destination must be within state_root")
        rows = self.connection.execute(
            """
            SELECT p.proposal_id, p.asset_id, a.library_id,
                   a.canonical_relative_path, p.analysis_result_id,
                   p.canonical_delta_json
            FROM review_proposals AS p
            JOIN assets AS a ON a.asset_id = p.asset_id
            JOIN analysis_results AS r ON r.result_id = p.analysis_result_id
            WHERE p.lifecycle = 'proposed'
              AND r.asset_version_id = a.current_asset_version_id
              AND p.decision IN (
                  'clear_ai_review', 'manual_review_focus', 'manual_review_uncertain'
              )
            ORDER BY a.library_id COLLATE NOCASE, a.library_id,
                     a.canonical_relative_path COLLATE NOCASE, a.canonical_relative_path,
                     p.analysis_result_id
            """
        ).fetchall()
        proposals = []
        for row in rows:
            delta = json.loads(row["canonical_delta_json"])
            if delta["decision"] not in _ACTIONABLE_DECISIONS:
                continue
            proposals.append(
                {
                    "asset_id": row["asset_id"],
                    "decision": delta["decision"],
                    "keyword": delta["keyword"],
                    "library_id": row["library_id"],
                    "proposal_id": row["proposal_id"],
                    "relative_path": row["canonical_relative_path"],
                    "result_id": row["analysis_result_id"],
                    "review_reason": delta["review_reason"],
                    "suggested_color": delta["suggested_color"],
                    "supersedes": delta["supersedes"],
                }
            )
        payload = (
            json.dumps(
                {"proposals": proposals, "schema_version": MANIFEST_SCHEMA_VERSION},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")

        destination_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary_path = destination_path.with_name(
            f".{destination_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary_path.open("wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, destination_path)
            directory_fd = os.open(destination_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return payload
