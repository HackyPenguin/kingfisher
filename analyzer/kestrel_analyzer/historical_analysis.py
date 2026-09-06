"""Analysis-only BioCLIP runner for indexed historical assets.

This module is deliberately independent of the legacy folder pipeline. It
reads a verified current asset into memory and can persist only immutable
analysis results or append-only retry diagnostics in ``HistoricalStore``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import stat
from typing import Any, Mapping, Protocol, Sequence

from .historical_index import RAW_EXTENSIONS
from .historical_store import AnalysisAssetVersion, HistoricalStore


ANALYZER_VERSION = "historical-bioclip-1"
OUTPUT_SCHEMA_VERSION = 1
BROAD_LABELS = ("landscape", "architecture", "human", "animal")
_TAXONOMY_FIELDS = (
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species_epithet",
    "species",
    "common_name",
)
_POLICY_SPEC = {
    "broad_labels": list(BROAD_LABELS),
    "broad_mode": "multi_label",
    "broad_scoring": "independent_positive_vs_scene_without_label",
    "interpretation": "suggestions_not_ground_truth",
    "output_schema_version": OUTPUT_SCHEMA_VERSION,
    "taxonomy_gate": "animal_score_greater_than_or_equal_to_threshold",
    "taxonomy_rank": "species",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be blank")
    return normalized


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return score


def _hex_digest(value: str, length: int, name: str) -> str:
    normalized = _required_text(value, name).lower()
    if len(normalized) != length or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a {length}-character hexadecimal digest")
    return normalized


@dataclass(frozen=True)
class ModelSpec:
    package: str = "pybioclip"
    package_version: str = "2.1.6"
    open_clip_package: str = "open-clip-torch"
    open_clip_package_version: str = "3.3.0"
    model_str: str = "hf-hub:imageomics/bioclip-2"
    model_revision: str = "2957b322090f9cb17ae72c71981c7218a28d81e0"
    model_config_sha256: str = (
        "1bf947e96e943fe50efd5c3e26c37f843a2fa3c358967719a68c8a6d17ce68c8"
    )
    weights_sha256: str = "b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef"
    taxonomy_repo_id: str = "imageomics/TreeOfLife-200M"
    taxonomy_repo_revision: str = "5f2dc493b3dc0e544438a04038ab15faa646b749"
    taxonomy_embeddings_sha256: str = (
        "c72442de7b0cb7fcb55ab7ca08099d0f42fbd6769efe16ca64c1daa7a8b87db2"
    )
    taxonomy_labels_sha256: str = (
        "4648928b006f85d83d28e5a27074ca9363465d82e778d708b369c5eaf54b8ef5"
    )
    pretrained_str: str | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in (
            "package",
            "package_version",
            "open_clip_package",
            "open_clip_package_version",
            "model_str",
            "taxonomy_repo_id",
            "device",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if self.pretrained_str is not None:
            object.__setattr__(
                self,
                "pretrained_str",
                _required_text(self.pretrained_str, "pretrained_str"),
            )
        object.__setattr__(
            self,
            "model_revision",
            _hex_digest(self.model_revision, 40, "model_revision"),
        )
        object.__setattr__(
            self,
            "model_config_sha256",
            _hex_digest(self.model_config_sha256, 64, "model_config_sha256"),
        )
        object.__setattr__(
            self,
            "weights_sha256",
            _hex_digest(self.weights_sha256, 64, "weights_sha256"),
        )
        object.__setattr__(
            self,
            "taxonomy_repo_revision",
            _hex_digest(self.taxonomy_repo_revision, 40, "taxonomy_repo_revision"),
        )
        object.__setattr__(
            self,
            "taxonomy_embeddings_sha256",
            _hex_digest(
                self.taxonomy_embeddings_sha256,
                64,
                "taxonomy_embeddings_sha256",
            ),
        )
        object.__setattr__(
            self,
            "taxonomy_labels_sha256",
            _hex_digest(self.taxonomy_labels_sha256, 64, "taxonomy_labels_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "expected_model_config_sha256": self.model_config_sha256,
            "expected_model_revision": self.model_revision,
            "expected_taxonomy_embeddings_sha256": self.taxonomy_embeddings_sha256,
            "expected_taxonomy_labels_sha256": self.taxonomy_labels_sha256,
            "expected_taxonomy_repo_revision": self.taxonomy_repo_revision,
            "expected_weights_sha256": self.weights_sha256,
            "model_str": self.model_str,
            "open_clip_package": self.open_clip_package,
            "open_clip_package_version": self.open_clip_package_version,
            "package": self.package,
            "package_version": self.package_version,
            "pretrained_str": self.pretrained_str,
            "taxonomy_embeddings_filename": "embeddings/txt_emb_species.npy",
            "taxonomy_labels_filename": "embeddings/txt_emb_species.json",
            "taxonomy_repo_id": self.taxonomy_repo_id,
            "verification_status": "local-artifact-verification-required",
        }


@dataclass(frozen=True)
class PreprocessingSpec:
    decoder: str = "kingfisher-pillow-rawpy"
    decoder_version: str = "1"
    colour_space: str = "RGB (sRGB)"
    orientation: str = "EXIF transpose for rendered; rawpy for RAW"
    raw_conversion: str = "camera white balance, sRGB, 8-bit"

    def __post_init__(self) -> None:
        for name in (
            "decoder",
            "decoder_version",
            "colour_space",
            "orientation",
            "raw_conversion",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))

    def to_dict(self) -> dict[str, str]:
        return {
            "colour_space": self.colour_space,
            "decoder": self.decoder,
            "decoder_version": self.decoder_version,
            "orientation": self.orientation,
            "raw_conversion": self.raw_conversion,
        }


@dataclass(frozen=True)
class AnalysisConfig:
    animal_threshold: float = 0.5
    taxonomy_top_k: int = 5
    candidate_species: tuple[str, ...] = ()
    model: ModelSpec = field(default_factory=ModelSpec)
    preprocessing: PreprocessingSpec = field(default_factory=PreprocessingSpec)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "animal_threshold",
            _probability(self.animal_threshold, "animal_threshold"),
        )
        if isinstance(self.taxonomy_top_k, bool) or not isinstance(self.taxonomy_top_k, int):
            raise TypeError("taxonomy_top_k must be an integer")
        if self.taxonomy_top_k <= 0:
            raise ValueError("taxonomy_top_k must be positive")
        if not isinstance(self.model, ModelSpec):
            raise TypeError("model must be a ModelSpec")
        if not isinstance(self.preprocessing, PreprocessingSpec):
            raise TypeError("preprocessing must be a PreprocessingSpec")
        if isinstance(self.candidate_species, str):
            raise TypeError("candidate_species must be a sequence of strings")
        normalized = tuple(
            sorted(
                (_required_text(value, "candidate_species item") for value in self.candidate_species),
                key=lambda value: (value.casefold(), value),
            )
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError("candidate_species must not contain duplicates")
        object.__setattr__(self, "candidate_species", normalized)

    def run_config(self) -> dict[str, Any]:
        return {
            "animal_threshold": self.animal_threshold,
            "broad_labels": list(BROAD_LABELS),
            "candidate_species": list(self.candidate_species),
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "preprocessing": self.preprocessing.to_dict(),
            "taxonomy_top_k": self.taxonomy_top_k,
        }


class ImageDecoder(Protocol):
    def decode(self, source_bytes: bytes, suffix: str) -> Any: ...


class BioClipProvider(Protocol):
    def predict_broad(
        self,
        image: Any,
        labels: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]: ...

    def predict_taxonomy(
        self,
        image: Any,
        candidate_species: Sequence[str],
        top_k: int,
    ) -> Sequence[Mapping[str, Any]]: ...


class HistoricalImageDecoder:
    """Decode rendered or RAW sources into an in-memory RGB PIL image."""

    def decode(self, source_bytes: bytes, suffix: str) -> Any:
        source_buffer = BytesIO(source_bytes)
        if suffix in RAW_EXTENSIONS:
            import rawpy
            from PIL import Image

            with rawpy.imread(source_buffer) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_color=rawpy.ColorSpace.sRGB,
                    output_bps=8,
                )
            return Image.fromarray(rgb, mode="RGB")

        from PIL import Image, ImageOps

        with Image.open(source_buffer) as source:
            rendered = ImageOps.exif_transpose(source).convert("RGB")
            rendered.load()
            return rendered.copy()


class HistoricalAnalysisError(RuntimeError):
    def __init__(self, error_code: str, message: str | None = None) -> None:
        self.error_code = error_code
        super().__init__(message or error_code)


class SourceVersionMismatch(HistoricalAnalysisError):
    def __init__(self) -> None:
        super().__init__(
            "source_version_mismatch",
            "source_version_mismatch: source does not match indexed asset version",
        )


@dataclass(frozen=True)
class AnalysisOutcome:
    analysis_run_id: str
    result_id: str
    output: Mapping[str, Any]
    cached: bool


class HistoricalAnalysisRunner:
    """Run suggestion-only analysis against one indexed current asset."""

    def __init__(
        self,
        store: HistoricalStore,
        *,
        config: AnalysisConfig | None = None,
        provider: BioClipProvider | None = None,
        decoder: ImageDecoder | None = None,
    ) -> None:
        if not isinstance(store, HistoricalStore):
            raise TypeError("store must be a HistoricalStore")
        self.store = store
        self.config = config or AnalysisConfig()
        if not isinstance(self.config, AnalysisConfig):
            raise TypeError("config must be an AnalysisConfig")
        if provider is None:
            from .pybioclip_adapter import PyBioClipProvider

            provider = PyBioClipProvider(self.config.model)
        self.provider = provider
        self.decoder = decoder or HistoricalImageDecoder()

    @property
    def model_digest(self) -> str:
        return _digest(self.config.model.to_dict())

    @property
    def policy_digest(self) -> str:
        return _digest(_POLICY_SPEC)

    def ensure_analysis_run(self) -> str:
        return self.store.ensure_analysis_run(
            analyzer_version=ANALYZER_VERSION,
            model_digest=self.model_digest,
            policy_digest=self.policy_digest,
            config=self.config.run_config(),
        )

    def run(
        self,
        library_id: str,
        relative_path: str,
        *,
        verify_cached: bool = False,
    ) -> AnalysisOutcome:
        """Analyze an indexed asset, optionally exercising a cached model path.

        ``verify_cached`` is reserved for an explicit real-model smoke check. It
        reruns decode and prediction but returns the existing immutable result
        without replacing or comparing it.
        """

        if not isinstance(verify_cached, bool):
            raise TypeError("verify_cached must be a boolean")
        asset = self.store.resolve_analysis_asset(library_id, relative_path)
        run_id = self.ensure_analysis_run()
        source_bytes = self._read_verified_source_or_record(asset, run_id)

        existing = self.store.analysis_result(asset.asset_version_id, run_id)
        if existing is not None and not verify_cached:
            result_id, output = existing
            return AnalysisOutcome(run_id, result_id, output, True)

        try:
            image = self.decoder.decode(
                source_bytes,
                PurePosixPath(asset.relative_path).suffix.lower(),
            )
        except Exception as error:
            self.store.record_analysis_failure(asset.asset_version_id, run_id, "decoder_failed")
            raise HistoricalAnalysisError("decoder_failed") from error

        try:
            broad_raw = self.provider.predict_broad(image, BROAD_LABELS)
            broad = self._normalize_broad(broad_raw)
            animal_score = next(item["score"] for item in broad if item["label"] == "animal")
            taxonomy_raw: Sequence[Mapping[str, Any]] = ()
            taxonomy_status = "not_run"
            if animal_score >= self.config.animal_threshold:
                taxonomy_raw = self.provider.predict_taxonomy(
                    image,
                    self.config.candidate_species,
                    self.config.taxonomy_top_k,
                )
                taxonomy_status = "suggested"
        except HistoricalAnalysisError as error:
            self.store.record_analysis_failure(asset.asset_version_id, run_id, "invalid_prediction")
            raise error
        except Exception as error:
            self.store.record_analysis_failure(asset.asset_version_id, run_id, "provider_failed")
            raise HistoricalAnalysisError("provider_failed") from error

        try:
            taxonomy = self._normalize_taxonomy(taxonomy_raw)
        except HistoricalAnalysisError as error:
            self.store.record_analysis_failure(asset.asset_version_id, run_id, "invalid_prediction")
            raise error

        try:
            self.store.assert_current_analysis_asset(asset)
        except ValueError as error:
            self.store.record_analysis_failure(
                asset.asset_version_id,
                run_id,
                "source_version_mismatch",
            )
            raise SourceVersionMismatch() from error
        output = self._build_output(
            asset,
            run_id,
            broad,
            animal_score,
            taxonomy_status,
            taxonomy,
        )
        if existing is not None:
            result_id, existing_output = existing
            return AnalysisOutcome(run_id, result_id, existing_output, True)
        result_id = "result-" + hashlib.sha256(
            f"{asset.asset_version_id}\0{run_id}".encode("utf-8")
        ).hexdigest()
        self.store.record_analysis_result(
            result_id,
            asset.asset_version_id,
            run_id,
            output,
            output_schema_version=OUTPUT_SCHEMA_VERSION,
        )
        return AnalysisOutcome(run_id, result_id, output, False)

    def _read_verified_source_or_record(
        self,
        asset: AnalysisAssetVersion,
        run_id: str,
    ) -> bytes:
        try:
            source_bytes, content_digest = self._read_source_bytes(asset.relative_path)
        except (OSError, ValueError) as error:
            self.store.record_analysis_failure(
                asset.asset_version_id,
                run_id,
                "source_version_mismatch",
            )
            raise SourceVersionMismatch() from error
        if (
            asset.fingerprint_algorithm != "sha256"
            or content_digest != asset.content_digest
            or len(source_bytes) != asset.byte_size
        ):
            self.store.record_analysis_failure(
                asset.asset_version_id,
                run_id,
                "source_version_mismatch",
            )
            raise SourceVersionMismatch()
        return source_bytes

    def _read_source_bytes(self, relative_path: str) -> tuple[bytes, str]:
        """Read through no-follow descriptors anchored at ``source_root``."""

        pure_path = PurePosixPath(relative_path)
        parts = pure_path.parts
        if pure_path.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("analysis source must be a normalized relative path")
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        directory_flag = getattr(os, "O_DIRECTORY", 0)
        if not no_follow or os.open not in os.supports_dir_fd:
            raise OSError("secure descriptor traversal is unavailable")

        read_flag = os.O_RDONLY | getattr(os, "O_BINARY", 0) | no_follow
        directory_read_flag = read_flag | directory_flag
        directory_descriptors: list[int] = []
        file_descriptor: int | None = None
        try:
            root_descriptor = os.open(self.store.source_root, directory_read_flag)
            directory_descriptors.append(root_descriptor)
            current_descriptor = root_descriptor
            for part in parts[:-1]:
                current_descriptor = os.open(
                    part,
                    directory_read_flag,
                    dir_fd=current_descriptor,
                )
                directory_descriptors.append(current_descriptor)
            file_descriptor = os.open(
                parts[-1],
                read_flag,
                dir_fd=current_descriptor,
            )
            before = os.fstat(file_descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("analysis source must be a regular file")
            digest = hashlib.sha256()
            source_bytes = bytearray()
            with os.fdopen(file_descriptor, "rb") as source:
                file_descriptor = None
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    source_bytes.extend(chunk)
                after = os.fstat(source.fileno())
            before_signature = (
                before.st_size,
                before.st_mtime_ns,
                before.st_dev,
                before.st_ino,
            )
            after_signature = (
                after.st_size,
                after.st_mtime_ns,
                after.st_dev,
                after.st_ino,
            )
            if before_signature != after_signature or len(source_bytes) != after.st_size:
                raise ValueError("analysis source changed while reading")
            return bytes(source_bytes), digest.hexdigest()
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)
            for descriptor in reversed(directory_descriptors):
                os.close(descriptor)

    def _normalize_broad(
        self,
        predictions: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        by_label: dict[str, float] = {}
        try:
            for prediction in predictions:
                label = prediction["classification"]
                if label not in BROAD_LABELS or label in by_label:
                    raise ValueError("broad labels must exactly match the closed label set")
                by_label[label] = _probability(prediction["score"], f"{label} score")
        except (KeyError, TypeError, ValueError) as error:
            raise HistoricalAnalysisError("invalid_prediction") from error
        if set(by_label) != set(BROAD_LABELS):
            raise HistoricalAnalysisError("invalid_prediction")
        return [{"label": label, "score": by_label[label]} for label in BROAD_LABELS]

    def _normalize_taxonomy(
        self,
        predictions: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized = []
        try:
            for prediction in predictions:
                item: dict[str, Any] = {
                    field: self._optional_taxonomy_text(prediction.get(field), field)
                    for field in _TAXONOMY_FIELDS
                }
                item["score"] = _probability(prediction["score"], "taxonomy score")
                if item["species"] is None:
                    raise ValueError("taxonomy species is required")
                if (
                    self.config.candidate_species
                    and item["species"] not in self.config.candidate_species
                ):
                    raise ValueError("taxonomy species is outside configured candidates")
                normalized.append(item)
        except (KeyError, TypeError, ValueError) as error:
            raise HistoricalAnalysisError("invalid_prediction") from error
        normalized.sort(
            key=lambda item: (
                -item["score"],
                item["species"].casefold(),
                item["species"],
                tuple("" if item[field] is None else item[field] for field in _TAXONOMY_FIELDS),
            )
        )
        return normalized[: self.config.taxonomy_top_k]

    @staticmethod
    def _optional_taxonomy_text(value: Any, name: str) -> str | None:
        if value is None:
            return None
        return _required_text(value, name)

    def _build_output(
        self,
        asset: AnalysisAssetVersion,
        run_id: str,
        broad: list[dict[str, Any]],
        animal_score: float,
        taxonomy_status: str,
        taxonomy: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "broad_categories": {
                "mode": "multi_label",
                "scores": broad,
            },
            "input": {
                "asset_version_id": asset.asset_version_id,
                "fingerprint_algorithm": asset.fingerprint_algorithm,
                "source_digest": asset.content_digest,
            },
            "interpretation": "suggestions_not_ground_truth",
            "provenance": {
                "analysis_run_id": run_id,
                "analyzer_version": ANALYZER_VERSION,
                "configuration": {
                    "animal_threshold": self.config.animal_threshold,
                    "broad_labels": list(BROAD_LABELS),
                    "broad_scoring": "independent_positive_vs_scene_without_label",
                    "candidate_species": list(self.config.candidate_species),
                    "taxonomy_top_k": self.config.taxonomy_top_k,
                },
                "model": self.config.model.to_dict(),
                "preprocessing": self.config.preprocessing.to_dict(),
            },
            "result_type": "bioclip_suggestions",
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "taxonomy": {
                "gate": {
                    "animal_score": animal_score,
                    "threshold": self.config.animal_threshold,
                },
                "rank": "species",
                "status": taxonomy_status,
                "suggestions": taxonomy,
            },
        }
