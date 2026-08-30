"""Lazy adapter for the official pybioclip 2.1.6 prediction APIs."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version as package_version
import math
from typing import Any, Callable, Mapping, Sequence

from .historical_analysis import ModelSpec


def _installed_package_version(package: str) -> str:
    return package_version(package)


class PyBioClipProvider:
    """Load pybioclip classifiers only when their prediction path is used."""

    def __init__(
        self,
        model: ModelSpec,
        *,
        module_loader: Callable[[str], Any] = import_module,
        package_version_resolver: Callable[[str], str] = _installed_package_version,
    ) -> None:
        if not isinstance(model, ModelSpec):
            raise TypeError("model must be a ModelSpec")
        if model.package != "pybioclip" or model.package_version != "2.1.6":
            raise ValueError("PyBioClipProvider requires pybioclip 2.1.6")
        self.model = model
        self._module_loader = module_loader
        self._package_version_resolver = package_version_resolver
        self._broad_classifier = None
        self._broad_labels: tuple[str, ...] | None = None
        self._taxonomy_classifier = None
        self._taxonomy_candidates: tuple[str, ...] | None = None
        self._species_rank = None

    def _assert_runtime_package(self) -> None:
        actual_package_version = self._package_version_resolver(self.model.package)
        if actual_package_version != self.model.package_version:
            raise RuntimeError("installed pybioclip version does not match model provenance")

    def _classifier_kwargs(self) -> dict[str, Any]:
        return {
            "device": self.model.device,
            "model_str": self.model.model_str,
            "pretrained_str": self.model.pretrained_str,
        }

    def predict_broad(
        self,
        image: Any,
        labels: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        labels_tuple = tuple(labels)
        if self._broad_classifier is None:
            self._assert_runtime_package()
            predict_module = self._module_loader("bioclip.predict")
            prompts = tuple(
                prompt
                for label in labels_tuple
                for prompt in (label, f"scene without {label}")
            )
            self._broad_classifier = predict_module.CustomLabelsClassifier(
                cls_ary=list(prompts),
                **self._classifier_kwargs(),
            )
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
            self._assert_runtime_package()
            predict_module = self._module_loader("bioclip.predict")
            bioclip_module = self._module_loader("bioclip")
            self._species_rank = bioclip_module.Rank.SPECIES
            self._taxonomy_classifier = predict_module.TreeOfLifeClassifier(
                **self._classifier_kwargs()
            )
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
