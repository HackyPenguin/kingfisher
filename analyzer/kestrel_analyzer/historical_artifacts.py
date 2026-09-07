"""Provision and verify the exact offline artifacts used by historical BioCLIP.

Normal inference never calls this module's network fetcher. Provisioning is an
explicit operator action which downloads immutable revisions into a staging
tree, verifies every SHA-256, and atomically installs the complete tree.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, BinaryIO, Callable, Literal, Mapping
from urllib.parse import quote
from urllib.request import Request, urlopen

from .historical_analysis import ModelSpec


ARTIFACT_SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_BUFFER_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ArtifactDescriptor:
    relative_path: str
    sha256: str
    url: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.relative_path,
            "sha256": self.sha256,
            "url": self.url,
        }


@dataclass(frozen=True)
class ArtifactVerification:
    status: str
    artifacts: tuple[Mapping[str, str], ...]
    manifest_status: str = "verified"
    model_revision: str | None = None
    taxonomy_revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "artifacts": [dict(item) for item in self.artifacts],
            "manifest_status": self.manifest_status,
            "status": self.status,
        }
        if self.model_revision is not None:
            document["model_revision"] = self.model_revision
        if self.taxonomy_revision is not None:
            document["taxonomy_revision"] = self.taxonomy_revision
        return document


@dataclass(frozen=True)
class ArtifactProvisionResult:
    provisioned: bool
    verification: ArtifactVerification


class ArtifactProvisionError(RuntimeError):
    """Closed provisioning error which does not retain response details."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class ArtifactInterrupted(ArtifactProvisionError):
    """Raised when cooperative termination stops artifact work."""

    def __init__(self) -> None:
        super().__init__("interrupted")


def _raise_if_stopped(should_stop: Callable[[], bool] | None) -> None:
    if should_stop is not None and should_stop():
        raise ArtifactInterrupted()


def _resolved_url(
    repo_id: str,
    revision: str,
    filename: str,
    *,
    repository_type: Literal["model", "dataset"],
) -> str:
    repository_prefix = "" if repository_type == "model" else "datasets/"
    return (
        "https://huggingface.co/"
        f"{repository_prefix}{quote(repo_id, safe='/')}/resolve/{revision}/"
        f"{quote(filename, safe='/')}"
        "?download=true"
    )


def artifact_descriptors(model: ModelSpec | None = None) -> tuple[ArtifactDescriptor, ...]:
    model = model or ModelSpec()
    if not isinstance(model, ModelSpec):
        raise TypeError("model must be a ModelSpec")
    descriptors = (
        ArtifactDescriptor(
            "model/open_clip_config.json",
            model.model_config_sha256,
            _resolved_url(
                model.model_str.removeprefix("hf-hub:"),
                model.model_revision,
                "open_clip_config.json",
                repository_type="model",
            ),
        ),
        ArtifactDescriptor(
            "model/open_clip_model.safetensors",
            model.weights_sha256,
            _resolved_url(
                model.model_str.removeprefix("hf-hub:"),
                model.model_revision,
                "open_clip_model.safetensors",
                repository_type="model",
            ),
        ),
        ArtifactDescriptor(
            "taxonomy/embeddings/txt_emb_species.npy",
            model.taxonomy_embeddings_sha256,
            _resolved_url(
                model.taxonomy_repo_id,
                model.taxonomy_repo_revision,
                "embeddings/txt_emb_species.npy",
                repository_type="dataset",
            ),
        ),
        ArtifactDescriptor(
            "taxonomy/embeddings/txt_emb_species.json",
            model.taxonomy_labels_sha256,
            _resolved_url(
                model.taxonomy_repo_id,
                model.taxonomy_repo_revision,
                "embeddings/txt_emb_species.json",
                repository_type="dataset",
            ),
        ),
    )
    return tuple(sorted(descriptors, key=lambda item: item.url))


def artifact_manifest(model: ModelSpec | None = None) -> dict[str, Any]:
    model = model or ModelSpec()
    return {
        "artifacts": [
            descriptor.to_dict()
            for descriptor in sorted(
                artifact_descriptors(model), key=lambda item: item.relative_path
            )
        ],
        "model": {
            "repository": model.model_str.removeprefix("hf-hub:"),
            "revision": model.model_revision,
        },
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "taxonomy": {
            "repository": model.taxonomy_repo_id,
            "revision": model.taxonomy_repo_revision,
        },
    }


def _canonical_manifest_bytes(model: ModelSpec) -> bytes:
    return (
        json.dumps(
            artifact_manifest(model),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(
    path: Path,
    should_stop: Callable[[], bool] | None = None,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            _raise_if_stopped(should_stop)
            chunk = source.read(_BUFFER_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(root: Path, relative_path: str) -> Path:
    candidate = root.joinpath(*relative_path.split("/"))
    current = candidate
    while True:
        if current.is_symlink():
            raise OSError("artifact path traverses a symlink")
        if current == root:
            break
        if root not in current.parents:
            raise OSError("artifact path escapes artifact root")
        current = current.parent
    return candidate


def verify_artifacts(
    artifact_root: Path | str,
    model: ModelSpec | None = None,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> ArtifactVerification:
    """Return a stable closed report for the expected immutable artifact tree."""

    model = model or ModelSpec()
    root = Path(artifact_root).expanduser().absolute()
    artifact_results: list[dict[str, str]] = []
    for descriptor in sorted(
        artifact_descriptors(model), key=lambda item: item.relative_path
    ):
        _raise_if_stopped(should_stop)
        status = "missing"
        try:
            path = _artifact_path(root, descriptor.relative_path)
            if path.is_file() and not path.is_symlink():
                status = (
                    "verified"
                    if _sha256_file(path, should_stop) == descriptor.sha256
                    else "digest_mismatch"
                )
            elif path.exists() or path.is_symlink():
                status = "unsafe_type"
        except OSError:
            status = "unreadable"
        artifact_results.append(
            {
                "path": descriptor.relative_path,
                "sha256": descriptor.sha256,
                "status": status,
            }
        )

    manifest_status = "missing"
    _raise_if_stopped(should_stop)
    manifest_path = root / _MANIFEST_NAME
    try:
        if root.is_symlink():
            manifest_status = "unsafe_root"
        elif manifest_path.is_file() and not manifest_path.is_symlink():
            manifest_status = (
                "verified"
                if manifest_path.read_bytes() == _canonical_manifest_bytes(model)
                else "mismatch"
            )
        elif manifest_path.exists() or manifest_path.is_symlink():
            manifest_status = "unsafe_type"
    except OSError:
        manifest_status = "unreadable"

    verified = manifest_status == "verified" and all(
        item["status"] == "verified" for item in artifact_results
    )
    return ArtifactVerification(
        status="verified" if verified else "invalid",
        artifacts=tuple(artifact_results),
        manifest_status=manifest_status,
        model_revision=model.model_revision,
        taxonomy_revision=model.taxonomy_repo_revision,
    )


def _network_fetch(url: str) -> BinaryIO:
    request = Request(url, headers={"User-Agent": "Kingfisher-artifact-provisioner/1"})
    return urlopen(request, timeout=120)


def _write_download(
    destination: Path,
    source: BinaryIO,
    expected_digest: str,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    digest = hashlib.sha256()
    with destination.open("xb") as output:
        while True:
            _raise_if_stopped(should_stop)
            chunk = source.read(_BUFFER_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            output.write(chunk)
        _raise_if_stopped(should_stop)
        output.flush()
        os.fsync(output.fileno())
    if digest.hexdigest() != expected_digest:
        raise ArtifactProvisionError("digest_mismatch")
    destination.chmod(0o444)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_staging_tree(path: Path) -> None:
    if not path.exists():
        return
    for directory, _, _ in os.walk(path, topdown=False):
        Path(directory).chmod(0o700)
    shutil.rmtree(path)


def _rename_directory_without_replace(source: Path, destination: Path) -> None:
    """Atomically publish a tree while refusing every pre-existing target."""

    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        result = library.renameat2(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            1,
        )
    elif sys.platform == "darwin" and hasattr(library, "renamex_np"):
        result = library.renamex_np(source_bytes, destination_bytes, 0x00000004)
    else:
        if os.path.lexists(destination):
            raise FileExistsError(errno.EEXIST, "destination exists", destination)
        os.rename(source, destination)
        return
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)


def provision_artifacts(
    artifact_root: Path | str,
    *,
    model: ModelSpec | None = None,
    fetcher: Callable[[str], BinaryIO] = _network_fetch,
    should_stop: Callable[[], bool] | None = None,
) -> ArtifactProvisionResult:
    """Explicitly fetch, verify, and atomically install a complete artifact tree."""

    model = model or ModelSpec()
    destination = Path(artifact_root).expanduser().absolute()
    _raise_if_stopped(should_stop)
    if destination.exists() or destination.is_symlink():
        existing = verify_artifacts(destination, model, should_stop=should_stop)
        if existing.status == "verified":
            return ArtifactProvisionResult(False, existing)
        raise ArtifactProvisionError("destination_not_pristine")

    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for descriptor in artifact_descriptors(model):
            _raise_if_stopped(should_stop)
            target = staging.joinpath(*descriptor.relative_path.split("/"))
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                with fetcher(descriptor.url) as source:
                    _write_download(
                        target,
                        source,
                        descriptor.sha256,
                        should_stop,
                    )
            except ArtifactProvisionError:
                raise
            except Exception as error:
                raise ArtifactProvisionError("fetch_failed") from error

        manifest_path = staging / _MANIFEST_NAME
        with manifest_path.open("xb") as output:
            output.write(_canonical_manifest_bytes(model))
            output.flush()
            os.fsync(output.fileno())
        manifest_path.chmod(0o444)

        _raise_if_stopped(should_stop)
        staged_report = verify_artifacts(staging, model, should_stop=should_stop)
        if staged_report.status != "verified":
            raise ArtifactProvisionError("staged_verification_failed")
        for directory, _, _ in os.walk(staging, topdown=False):
            _raise_if_stopped(should_stop)
            path = Path(directory)
            _fsync_directory(path)
            path.chmod(0o555)
        _raise_if_stopped(should_stop)
        try:
            _rename_directory_without_replace(staging, destination)
        except OSError as error:
            if (
                destination.exists()
                and verify_artifacts(
                    destination,
                    model,
                    should_stop=should_stop,
                ).status
                == "verified"
            ):
                return ArtifactProvisionResult(
                    False,
                    verify_artifacts(
                        destination,
                        model,
                        should_stop=should_stop,
                    ),
                )
            raise ArtifactProvisionError("atomic_install_failed") from error
        _fsync_directory(destination.parent)
        installed = verify_artifacts(destination, model, should_stop=should_stop)
        if installed.status != "verified":
            raise ArtifactProvisionError("installed_verification_failed")
        return ArtifactProvisionResult(True, installed)
    finally:
        _remove_staging_tree(staging)
