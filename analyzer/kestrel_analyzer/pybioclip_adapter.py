"""Lazy adapter for the official pybioclip 2.1.6 prediction APIs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module
from importlib.metadata import version as package_version
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .historical_analysis import ModelSpec


def _installed_package_version(package: str) -> str:
    return package_version(package)


@dataclass(frozen=True)
class LocalBioClipAssets:
    """Exact local files used by pybioclip; no Hub identifiers are loaded."""

    model_directory: Path
    model_config: Path
    model_weights: Path
    taxonomy_embeddings: Path
    taxonomy_labels: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _local_assets_from_environment(model: ModelSpec) -> LocalBioClipAssets:
    model_directory_value = os.environ.get("KINGFISHER_BIOCLIP_MODEL_DIR")
    taxonomy_directory_value = os.environ.get("KINGFISHER_BIOCLIP_TAXONOMY_DIR")
    if not model_directory_value or not taxonomy_directory_value:
        raise RuntimeError(
            "local BioCLIP model and taxonomy directories must be configured"
        )
    model_directory = Path(model_directory_value).resolve(strict=True)
    taxonomy_directory = Path(taxonomy_directory_value).resolve(strict=True)
    return LocalBioClipAssets(
        model_directory=model_directory,
        model_config=model_directory / "open_clip_config.json",
        model_weights=model_directory / "open_clip_model.safetensors",
        taxonomy_embeddings=taxonomy_directory / "embeddings" / "txt_emb_species.npy",
        taxonomy_labels=taxonomy_directory / "embeddings" / "txt_emb_species.json",
    )


class PyBioClipProvider:
    """Load pybioclip classifiers only when their prediction path is used."""

    def __init__(
        self,
        model: ModelSpec,
        *,
        module_loader: Callable[[str], Any] = import_module,
        package_version_resolver: Callable[[str], str] = _installed_package_version,
        local_assets_resolver: Callable[[ModelSpec], LocalBioClipAssets] = (
            _local_assets_from_environment
        ),
    ) -> None:
        if not isinstance(model, ModelSpec):
            raise TypeError("model must be a ModelSpec")
        if (
            model.package != "pybioclip"
            or model.package_version != "2.1.6"
            or model.open_clip_package != "open-clip-torch"
            or model.open_clip_package_version != "3.3.0"
        ):
            raise ValueError(
                "PyBioClipProvider requires pybioclip 2.1.6 and open-clip-torch 3.3.0"
            )
        self.model = model
        self._module_loader = module_loader
        self._package_version_resolver = package_version_resolver
        self._local_assets_resolver = local_assets_resolver
        self._local_assets: LocalBioClipAssets | None = None
        self._broad_classifier = None
        self._broad_labels: tuple[str, ...] | None = None
        self._taxonomy_classifier = None
        self._taxonomy_candidates: tuple[str, ...] | None = None
        self._species_rank = None

    def _assert_runtime_packages(self) -> None:
        expected_versions = (
            (self.model.package, self.model.package_version),
            (self.model.open_clip_package, self.model.open_clip_package_version),
        )
        for package, expected_version in expected_versions:
            actual_version = self._package_version_resolver(package)
            if actual_version != expected_version:
                raise RuntimeError(
                    f"installed {package} version does not match model provenance"
                )

    def _verified_local_assets(self) -> LocalBioClipAssets:
        if self._local_assets is None:
            self._local_assets = self._local_assets_resolver(self.model)
        assets = self._local_assets
        expected = {
            assets.model_config: self.model.model_config_sha256,
            assets.model_weights: self.model.weights_sha256,
            assets.taxonomy_embeddings: self.model.taxonomy_embeddings_sha256,
            assets.taxonomy_labels: self.model.taxonomy_labels_sha256,
        }
        if not assets.model_directory.is_dir():
            raise RuntimeError("local BioCLIP model directory is unavailable")
        for path, expected_digest in expected.items():
            resolved = path.resolve(strict=True)
            if not resolved.is_file() or _sha256_file(resolved) != expected_digest:
                raise RuntimeError("local BioCLIP artifact does not match configured provenance")
        return assets

    def _classifier_kwargs(self, assets: LocalBioClipAssets) -> dict[str, Any]:
        return {
            "device": self.model.device,
            "model_str": f"local-dir:{assets.model_directory}",
            "pretrained_str": None,
        }

    def predict_broad(
        self,
        image: Any,
        labels: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        labels_tuple = tuple(labels)
        if self._broad_classifier is None:
            self._assert_runtime_packages()
            assets = self._verified_local_assets()
            predict_module = self._module_loader("bioclip.predict")
            prompts = tuple(
                prompt
                for label in labels_tuple
                for prompt in (label, f"scene without {label}")
            )
            self._broad_classifier = predict_module.CustomLabelsClassifier(
                cls_ary=list(prompts),
                **self._classifier_kwargs(assets),
            )
            self._verified_local_assets()
            self._broad_labels = labels_tuple
        elif labels_tuple != self._broad_labels:
            raise ValueError("broad classifier labels cannot change after initialization")
        prompts = tuple(
            prompt
            for label in labels_tuple
            for prompt in (label, f"scene without {label}")
        )
        raw_predictions = self._broad_classifier.predict([image], k=len(prompts))
        scores: dict[str, float] = {}
        for prediction in raw_predictions:
            classification = prediction["classification"]
            if classification not in prompts or classification in scores:
                raise ValueError("broad prompt predictions must match the configured pairs")
            score = float(prediction["score"])
            if not math.isfinite(score) or score < 0.0:
                raise ValueError("broad prompt scores must be finite and non-negative")
            scores[classification] = score
        if set(scores) != set(prompts):
            raise ValueError("broad prompt predictions are incomplete")
        result = []
        for label in labels_tuple:
            positive = scores[label]
            negative = scores[f"scene without {label}"]
            denominator = positive + negative
            if denominator <= 0.0:
                raise ValueError("broad prompt score pair must have positive mass")
            result.append(
                {
                    "classification": label,
                    "score": positive / denominator,
                }
            )
        return tuple(result)

    def predict_taxonomy(
        self,
        image: Any,
        candidate_species: Sequence[str],
        top_k: int,
    ) -> Sequence[Mapping[str, Any]]:
        candidates_tuple = tuple(candidate_species)
        if self._taxonomy_classifier is None:
            self._assert_runtime_packages()
            assets = self._verified_local_assets()
            predict_module = self._module_loader("bioclip.predict")
            bioclip_module = self._module_loader("bioclip")
            self._species_rank = bioclip_module.Rank.SPECIES
            taxonomy_files = {
                "embeddings/txt_emb_species.npy": str(assets.taxonomy_embeddings),
                "embeddings/txt_emb_species.json": str(assets.taxonomy_labels),
            }
            base_classifier = predict_module.TreeOfLifeClassifier

            class LocalTreeOfLifeClassifier(base_classifier):
                def get_cached_datafile(self, filename: str) -> str:
                    try:
                        return taxonomy_files[filename]
                    except KeyError as error:
                        raise RuntimeError("unexpected taxonomy artifact request") from error

            self._taxonomy_classifier = LocalTreeOfLifeClassifier(
                **self._classifier_kwargs(assets)
            )
            self._verified_local_assets()
            if candidates_tuple:
                taxa_filter = self._taxonomy_classifier.create_taxa_filter(
                    self._species_rank,
                    list(candidates_tuple),
                )
                self._taxonomy_classifier.apply_filter(taxa_filter)
            self._taxonomy_candidates = candidates_tuple
        elif candidates_tuple != self._taxonomy_candidates:
            raise ValueError("taxonomy candidates cannot change after initialization")
        return self._taxonomy_classifier.predict(
            [image],
            self._species_rank,
            k=top_k,
        )
