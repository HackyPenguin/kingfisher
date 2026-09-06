"""Bounded, machine-readable NAS runtime for historical analysis.

This entry point is intentionally separate from Kingfisher's legacy desktop
CLI. It can only index source files and write immutable analysis state under a
separate state root; it has no metadata, rating, proposal, sidecar, or source
mutation path.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import signal
import sys
import threading
from typing import Any, Callable, Iterator, Mapping, Sequence, TextIO

from .historical_analysis import (
    AnalysisConfig,
    HistoricalAnalysisError,
    HistoricalAnalysisRunner,
    HistoricalAnalysisSkipped,
    ModelSpec,
)
from .historical_artifacts import (
    ArtifactInterrupted,
    ArtifactProvisionError,
    ArtifactVerification,
    provision_artifacts,
    verify_artifacts,
)
from .historical_index import HistoricalIndexer, ScanSummary
from .historical_store import HistoricalStore
from .pybioclip_adapter import LocalBioClipAssets, PyBioClipProvider


CLI_SCHEMA_VERSION = 1
MAX_WORK_ITEMS = 1_000_000
MAX_RETRIES = 10
DEFAULT_MAX_SOURCE_BYTES = 512 * 1024 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024 * 1024
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_SIGTERM = 143


class CliUsageError(ValueError):
    pass


class MachineArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliUsageError("invalid_arguments")


def _bounded_integer(name: str, maximum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            parsed = int(value, 10)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from error
        if not 0 <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be between 0 and {maximum}"
            )
        return parsed

    return parse


def _positive_bounded_integer(name: str, maximum: int) -> Callable[[str], int]:
    parse_bounded = _bounded_integer(name, maximum)

    def parse(value: str) -> int:
        parsed = parse_bounded(value)
        if parsed == 0:
            raise argparse.ArgumentTypeError(f"{name} must be positive")
        return parsed

    return parse


def _common_library_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--library-id", required=True)


def _artifact_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--artifact-root",
        required=True,
        help="Pre-provisioned immutable model tree; never fetched during inference",
    )


def _source_size_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-source-bytes",
        default=DEFAULT_MAX_SOURCE_BYTES,
        type=_positive_bounded_integer("max-source-bytes", MAX_SOURCE_BYTES),
    )


def build_parser() -> MachineArgumentParser:
    parser = MachineArgumentParser(prog="kingfisher-historical")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index")
    _common_library_arguments(index)
    index.add_argument(
        "--max-items",
        required=True,
        type=_bounded_integer("max-items", MAX_WORK_ITEMS),
    )
    index.add_argument("--scan-id")
    index.add_argument("--full-hash-audit", action="store_true")

    analyze = commands.add_parser("analyze")
    _common_library_arguments(analyze)
    _artifact_argument(analyze)
    _source_size_argument(analyze)
    analyze.add_argument(
        "--limit",
        required=True,
        type=_bounded_integer("limit", MAX_WORK_ITEMS),
    )
    analyze.add_argument(
        "--max-retries",
        default=0,
        type=_bounded_integer("max-retries", MAX_RETRIES),
    )

    run = commands.add_parser("run")
    _common_library_arguments(run)
    _artifact_argument(run)
    _source_size_argument(run)
    run.add_argument(
        "--max-items",
        required=True,
        type=_bounded_integer("max-items", MAX_WORK_ITEMS),
    )
    run.add_argument(
        "--limit",
        required=True,
        type=_bounded_integer("limit", MAX_WORK_ITEMS),
    )
    run.add_argument(
        "--max-retries",
        default=0,
        type=_bounded_integer("max-retries", MAX_RETRIES),
    )
    run.add_argument("--scan-id")
    run.add_argument("--full-hash-audit", action="store_true")

    status = commands.add_parser("status")
    _common_library_arguments(status)

    smoke = commands.add_parser("smoke")
    _common_library_arguments(smoke)
    _artifact_argument(smoke)
    _source_size_argument(smoke)
    smoke.add_argument("--relative-path", required=True)

    artifacts = commands.add_parser("artifacts")
    artifact_commands = artifacts.add_subparsers(dest="artifact_command", required=True)
    verify = artifact_commands.add_parser("verify")
    _artifact_argument(verify)
    provision = artifact_commands.add_parser("provision")
    _artifact_argument(provision)
    return parser


def _emit(output: TextIO, value: Mapping[str, Any]) -> None:
    output.write(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    output.flush()


@contextmanager
def termination_handlers(stop_event: threading.Event) -> Iterator[None]:
    """Convert SIGTERM/SIGINT into cooperative stops and restore handlers."""

    previous: dict[int, Any] = {}

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, request_stop)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _scan_document(summary: ScanSummary) -> dict[str, Any]:
    return {
        "diagnostic_count": summary.diagnostic_count,
        "observed_count": summary.observed_count,
        "scan_id": summary.scan_id,
        "scan_status": summary.status,
    }


def _default_provider_factory(model: ModelSpec, artifact_root: Path) -> PyBioClipProvider:
    assets = LocalBioClipAssets(
        model_directory=artifact_root / "model",
        model_config=artifact_root / "model" / "open_clip_config.json",
        model_weights=artifact_root / "model" / "open_clip_model.safetensors",
        taxonomy_embeddings=(
            artifact_root / "taxonomy" / "embeddings" / "txt_emb_species.npy"
        ),
        taxonomy_labels=(
            artifact_root / "taxonomy" / "embeddings" / "txt_emb_species.json"
        ),
    )
    return PyBioClipProvider(model, local_assets_resolver=lambda _model: assets)


def _verify_for_inference(
    artifact_root: Path,
    model: ModelSpec,
    should_stop: Callable[[], bool],
) -> ArtifactVerification:
    report = verify_artifacts(artifact_root, model, should_stop=should_stop)
    if report.status != "verified":
        raise ArtifactProvisionError("artifact_verification_failed")
    return report


def _analysis_document(
    store: HistoricalStore,
    *,
    library_id: str,
    artifact_root: Path,
    limit: int,
    max_retries: int,
    max_source_bytes: int,
    stop_event: threading.Event,
    provider_factory: Callable[[ModelSpec, Path], Any],
    decoder: Any,
    relative_path: str | None = None,
) -> tuple[dict[str, Any], ArtifactVerification]:
    model = ModelSpec()
    verification = _verify_for_inference(artifact_root, model, stop_event.is_set)
    provider = provider_factory(model, artifact_root)
    runner = HistoricalAnalysisRunner(
        store,
        config=AnalysisConfig(model=model, max_source_bytes=max_source_bytes),
        provider=provider,
        decoder=decoder,
    )
    analysis_run_id = runner.ensure_analysis_run()
    if relative_path is None:
        available_count = store.stale_asset_count(
            analysis_run_id,
            library_id=library_id,
        )
        selected_paths = store.stale_asset_paths(
            analysis_run_id,
            library_id=library_id,
            limit=limit,
        )
        mode = "stale"
    else:
        available_count = 1
        selected_paths = (relative_path,)
        mode = "real_model"

    attempts = 0
    retry_count = 0
    failure_attempt_count = 0
    failed_count = 0
    skipped_count = 0
    processed_count = 0
    cached_count = 0
    errors: Counter[str] = Counter()
    skips: Counter[str] = Counter()
    results: list[dict[str, Any]] = []

    for path in selected_paths:
        if stop_event.is_set():
            break
        retries_used = 0
        while True:
            attempts += 1
            try:
                outcome = runner.run(
                    library_id,
                    path,
                    verify_cached=relative_path is not None,
                )
            except HistoricalAnalysisSkipped as error:
                skipped_count += 1
                processed_count += 1
                skips[error.error_code] += 1
                break
            except HistoricalAnalysisError as error:
                failure_attempt_count += 1
                errors[error.error_code] += 1
                if stop_event.is_set():
                    break
                if retries_used >= max_retries:
                    failed_count += 1
                    processed_count += 1
                    break
                retries_used += 1
                retry_count += 1
                continue
            results.append(
                {
                    "cached": outcome.cached,
                    "relative_path": path,
                    "result_id": outcome.result_id,
                }
            )
            cached_count += int(outcome.cached)
            processed_count += 1
            break
        if stop_event.is_set():
            break

    results.sort(key=lambda item: (item["relative_path"].casefold(), item["relative_path"]))
    return (
        {
            "analysis_run_id": analysis_run_id,
            "attempt_count": attempts,
            "available_count": available_count,
            "cached_count": cached_count,
            "deferred_count": available_count - len(selected_paths),
            "errors": [
                {"count": errors[error_code], "error_code": error_code}
                for error_code in sorted(errors)
            ],
            "failed_count": failed_count,
            "failure_attempt_count": failure_attempt_count,
            "limit": limit,
            "max_retries": max_retries,
            "mode": mode,
            "remaining_count": len(selected_paths) - processed_count,
            "results": results,
            "retry_count": retry_count,
            "selected_count": len(selected_paths),
            "skipped_count": skipped_count,
            "skips": [
                {"count": skips[error_code], "error_code": error_code}
                for error_code in sorted(skips)
            ],
            "success_count": len(results),
        },
        verification,
    )


def _library_status(store: HistoricalStore, library_id: str) -> dict[str, int]:
    row = store.connection.execute(
        """
        SELECT
          COUNT(DISTINCT a.asset_id) AS asset_count,
          COUNT(DISTINCT a.current_asset_version_id) AS stable_asset_count,
          COUNT(DISTINCT r.result_id) AS result_count,
          COUNT(DISTINCT f.attempt_id) AS failure_attempt_count,
          COUNT(DISTINCT s.skip_id) AS skipped_count
        FROM assets AS a
        LEFT JOIN asset_versions AS v
          ON v.asset_version_id = a.current_asset_version_id
        LEFT JOIN analysis_results AS r
          ON r.asset_version_id = v.asset_version_id
        LEFT JOIN analysis_attempt_failures AS f
          ON f.asset_version_id = v.asset_version_id
        LEFT JOIN analysis_terminal_skips AS s
          ON s.asset_version_id = v.asset_version_id
        WHERE a.library_id = ? AND a.state = 'active'
        """,
        (library_id,),
    ).fetchone()
    return {key: int(row[key]) for key in row.keys()}


def _overall_status(
    *,
    interrupted: bool,
    index: Mapping[str, Any] | None,
    analysis: Mapping[str, Any] | None,
) -> str:
    if interrupted:
        return "interrupted"
    if analysis is not None and analysis["failed_count"]:
        return "completed_with_failures"
    if index is not None and index["scan_status"] != "completed":
        return "bounded"
    if analysis is not None and analysis["deferred_count"]:
        return "bounded"
    return "completed"


def _run_library_command(
    arguments: argparse.Namespace,
    *,
    stop_event: threading.Event,
    provider_factory: Callable[[ModelSpec, Path], Any],
    decoder: Any,
) -> tuple[int, dict[str, Any]]:
    source_root = Path(arguments.source_root).expanduser().resolve(strict=True)
    state_root = Path(arguments.state_root).expanduser().resolve(strict=False)
    index_document: dict[str, Any] | None = None
    analysis_document: dict[str, Any] | None = None
    artifact_document: dict[str, Any] | None = None

    with HistoricalStore(state_root, source_root) as store:
        if arguments.command in {"index", "run"}:
            scan = HistoricalIndexer(
                store,
                source_root,
                arguments.library_id,
                mutate_review_proposals=False,
            ).run(
                scan_id=arguments.scan_id,
                max_items=arguments.max_items,
                full_hash_audit=arguments.full_hash_audit,
                should_stop=stop_event.is_set,
            )
            index_document = _scan_document(scan)

        if arguments.command == "status":
            status = "interrupted" if stop_event.is_set() else "completed"
            return (
                EXIT_SIGTERM if stop_event.is_set() else 0,
                {
                    "command": "status",
                    "library": _library_status(store, arguments.library_id),
                    "library_id": arguments.library_id,
                    "schema_version": CLI_SCHEMA_VERSION,
                    "status": status,
                },
            )

        if arguments.command in {"analyze", "run", "smoke"} and not stop_event.is_set():
            relative_path = (
                arguments.relative_path if arguments.command == "smoke" else None
            )
            limit = 1 if arguments.command == "smoke" else arguments.limit
            max_retries = 0 if arguments.command == "smoke" else arguments.max_retries
            try:
                analysis_document, verification = _analysis_document(
                    store,
                    library_id=arguments.library_id,
                    artifact_root=(
                        Path(arguments.artifact_root).expanduser().resolve(strict=True)
                    ),
                    limit=limit,
                    max_retries=max_retries,
                    max_source_bytes=arguments.max_source_bytes,
                    stop_event=stop_event,
                    provider_factory=provider_factory,
                    decoder=decoder,
                    relative_path=relative_path,
                )
            except ArtifactInterrupted:
                stop_event.set()
            else:
                artifact_document = verification.to_dict()

    overall = _overall_status(
        interrupted=stop_event.is_set(),
        index=index_document,
        analysis=analysis_document,
    )
    document: dict[str, Any] = {
        "command": arguments.command,
        "library_id": arguments.library_id,
        "schema_version": CLI_SCHEMA_VERSION,
        "status": overall,
    }
    if index_document is not None:
        document["index"] = index_document
    if analysis_document is not None:
        document["analysis"] = analysis_document
    if artifact_document is not None:
        document["artifacts"] = artifact_document
    if overall == "interrupted":
        return EXIT_SIGTERM, document
    if overall == "completed_with_failures":
        return EXIT_FAILED, document
    return 0, document


def _run_artifact_command(
    arguments: argparse.Namespace,
    stop_event: threading.Event,
) -> tuple[int, dict[str, Any]]:
    root = Path(arguments.artifact_root).expanduser().absolute()
    if arguments.artifact_command == "verify":
        verification = verify_artifacts(
            root,
            ModelSpec(),
            should_stop=stop_event.is_set,
        )
        return (
            0 if verification.status == "verified" else EXIT_FAILED,
            {
                "artifacts": verification.to_dict(),
                "command": "artifacts.verify",
                "schema_version": CLI_SCHEMA_VERSION,
                "status": verification.status,
            },
        )
    result = provision_artifacts(
        root,
        model=ModelSpec(),
        should_stop=stop_event.is_set,
    )
    return (
        0,
        {
            "artifacts": result.verification.to_dict(),
            "command": "artifacts.provision",
            "provisioned": result.provisioned,
            "schema_version": CLI_SCHEMA_VERSION,
            "status": "completed",
        },
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    output: TextIO | None = None,
    provider_factory: Callable[[ModelSpec, Path], Any] = _default_provider_factory,
    decoder: Any = None,
    stop_event: threading.Event | None = None,
    install_signal_handlers: bool = True,
) -> int:
    output = output or sys.stdout
    try:
        arguments = build_parser().parse_args(argv)
    except CliUsageError:
        _emit(
            output,
            {
                "error_code": "invalid_arguments",
                "schema_version": CLI_SCHEMA_VERSION,
                "status": "error",
            },
        )
        return EXIT_USAGE

    event = stop_event or threading.Event()
    handlers = termination_handlers(event) if install_signal_handlers else nullcontext()
    try:
        with handlers:
            if arguments.command == "artifacts":
                exit_code, document = _run_artifact_command(arguments, event)
            else:
                exit_code, document = _run_library_command(
                    arguments,
                    stop_event=event,
                    provider_factory=provider_factory,
                    decoder=decoder,
                )
    except ArtifactInterrupted:
        exit_code = EXIT_SIGTERM
        document = {
            "command": (
                f"artifacts.{arguments.artifact_command}"
                if arguments.command == "artifacts"
                else arguments.command
            ),
            "schema_version": CLI_SCHEMA_VERSION,
            "status": "interrupted",
        }
    except ArtifactProvisionError as error:
        exit_code = EXIT_FAILED
        document = {
            "command": (
                f"artifacts.{arguments.artifact_command}"
                if arguments.command == "artifacts"
                else arguments.command
            ),
            "error_code": error.error_code,
            "schema_version": CLI_SCHEMA_VERSION,
            "status": "error",
        }
    except (FileNotFoundError, NotADirectoryError, PermissionError, ValueError):
        exit_code = EXIT_USAGE
        document = {
            "command": arguments.command,
            "error_code": "invalid_configuration",
            "schema_version": CLI_SCHEMA_VERSION,
            "status": "error",
        }
    except Exception:
        exit_code = EXIT_FAILED
        document = {
            "command": arguments.command,
            "error_code": "internal_error",
            "schema_version": CLI_SCHEMA_VERSION,
            "status": "error",
        }
    _emit(output, document)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
