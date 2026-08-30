import hashlib
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from analyzer.kestrel_analyzer.historical_index import HistoricalIndexer
from analyzer.kestrel_analyzer.historical_store import HistoricalStore


BROAD_SCORES = (
    {"classification": "animal", "score": 0.8, "ignored": "closed schema"},
    {"classification": "human", "score": 0.1},
    {"classification": "landscape", "score": 0.6},
    {"classification": "architecture", "score": 0.2},
)

TAXONOMY_SCORES = (
    {
        "score": 0.4,
        "kingdom": "Animalia",
        "phylum": "Chordata",
        "class": "Aves",
        "order": "Passeriformes",
        "family": "Alcedinidae",
        "genus": "Alcedo",
        "species_epithet": "atthis",
        "species": "Alcedo atthis",
        "common_name": "Common kingfisher",
        "unsafe_extra": "not persisted",
    },
    {
        "score": 0.7,
        "kingdom": "Animalia",
        "phylum": "Chordata",
        "class": "Aves",
        "order": "Passeriformes",
        "family": "Alcedinidae",
        "genus": "Megaceryle",
        "species_epithet": "alcyon",
        "species": "Megaceryle alcyon",
        "common_name": "Belted kingfisher",
    },
)


class FakeDecoder:
    def __init__(self, value="decoded-image"):
        self.value = value
        self.inputs = []

    def decode(self, source_bytes, suffix):
        self.inputs.append((source_bytes, suffix))
        return self.value


class FailingDecoder:
    def decode(self, source_bytes, suffix):
        raise OSError("decoder details are not durable state")


class FakeProvider:
    def __init__(self, broad=BROAD_SCORES, taxonomy=TAXONOMY_SCORES, failures=0):
        self.broad = broad
        self.taxonomy = taxonomy
        self.failures = failures
        self.broad_calls = []
        self.taxonomy_calls = []

    def predict_broad(self, image, labels):
        self.broad_calls.append((image, tuple(labels)))
        if self.failures:
            self.failures -= 1
            raise RuntimeError("transient model failure with sensitive details")
        return self.broad

    def predict_taxonomy(self, image, candidate_species, top_k):
        self.taxonomy_calls.append((image, tuple(candidate_species), top_k))
        return self.taxonomy


class HistoricalAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.source_root = root / "photos"
        self.state_root = root / "state"
        self.source_root.mkdir()
        self.photo = self.source_root / "nested" / "bird.jpg"
        self.photo.parent.mkdir()
        self.photo.write_bytes(b"indexed photo bytes")
        self.sidecar = self.photo.with_suffix(".xmp")
        self.sidecar.write_bytes(b"lightroom metadata")
        self.store = HistoricalStore(self.state_root, self.source_root)
        HistoricalIndexer(self.store, self.source_root, "private-library").run()

    def tearDown(self):
        self.store.close()
        self.temporary_directory.cleanup()

    @staticmethod
    def imports():
        from analyzer.kestrel_analyzer.historical_analysis import (
            AnalysisConfig,
            HistoricalAnalysisRunner,
            ModelSpec,
            PreprocessingSpec,
        )

        return AnalysisConfig, HistoricalAnalysisRunner, ModelSpec, PreprocessingSpec

    def config(self, **overrides):
        AnalysisConfig, _, ModelSpec, PreprocessingSpec = self.imports()
        values = {
            "animal_threshold": 0.5,
            "taxonomy_top_k": 2,
            "candidate_species": ("Alcedo atthis", "Megaceryle alcyon"),
            "model": ModelSpec(),
            "preprocessing": PreprocessingSpec(
                decoder="test-decoder",
                decoder_version="1",
                colour_space="RGB",
                orientation="preserved",
                raw_conversion="not-applicable",
            ),
        }
        values.update(overrides)
        return AnalysisConfig(**values)

    def runner(self, provider=None, decoder=None, config=None):
        _, HistoricalAnalysisRunner, _, _ = self.imports()
        return HistoricalAnalysisRunner(
            self.store,
            config=config or self.config(),
            provider=provider or FakeProvider(),
            decoder=decoder or FakeDecoder(),
        )

    def source_snapshot(self):
        return {
            path.relative_to(self.source_root).as_posix(): (
                path.read_bytes(),
                path.stat().st_mtime_ns,
            )
            for path in sorted(self.source_root.rglob("*"))
            if path.is_file()
        }

    def test_broad_output_is_closed_multilabel_finite_and_deterministic(self):
        provider = FakeProvider()
        decoder = FakeDecoder()

        outcome = self.runner(provider, decoder).run("private-library", "nested/bird.jpg")

        self.assertFalse(outcome.cached)
        self.assertEqual([(b"indexed photo bytes", ".jpg")], decoder.inputs)
        self.assertEqual(
            (
                "landscape",
                "architecture",
                "human",
                "animal",
            ),
            tuple(item["label"] for item in outcome.output["broad_categories"]["scores"]),
        )
        self.assertEqual(
            (0.6, 0.2, 0.1, 0.8),
            tuple(item["score"] for item in outcome.output["broad_categories"]["scores"]),
        )
        self.assertEqual("multi_label", outcome.output["broad_categories"]["mode"])
        self.assertEqual("suggestions_not_ground_truth", outcome.output["interpretation"])
        self.assertEqual("pybioclip", outcome.output["provenance"]["model"]["package"])
        self.assertEqual("2.1.6", outcome.output["provenance"]["model"]["package_version"])
        self.assertEqual(
            "open-clip-torch",
            outcome.output["provenance"]["model"]["open_clip_package"],
        )
        self.assertEqual(
            "3.3.0",
            outcome.output["provenance"]["model"]["open_clip_package_version"],
        )
        self.assertEqual(
            "2957b322090f9cb17ae72c71981c7218a28d81e0",
            outcome.output["provenance"]["model"]["expected_model_revision"],
        )
        self.assertEqual(
            "b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef",
            outcome.output["provenance"]["model"]["expected_weights_sha256"],
        )
        self.assertEqual(
            "1bf947e96e943fe50efd5c3e26c37f843a2fa3c358967719a68c8a6d17ce68c8",
            outcome.output["provenance"]["model"]["expected_model_config_sha256"],
        )
        self.assertEqual(
            "local-artifact-verification-required",
            outcome.output["provenance"]["model"]["verification_status"],
        )
        self.assertNotIn("model_revision", outcome.output["provenance"]["model"])
        self.assertNotIn("weights_sha256", outcome.output["provenance"]["model"])
        self.assertEqual(
            "imageomics/TreeOfLife-200M",
            outcome.output["provenance"]["model"]["taxonomy_repo_id"],
        )
        self.assertEqual(
            "5f2dc493b3dc0e544438a04038ab15faa646b749",
            outcome.output["provenance"]["model"]["expected_taxonomy_repo_revision"],
        )
        self.assertEqual(
            "c72442de7b0cb7fcb55ab7ca08099d0f42fbd6769efe16ca64c1daa7a8b87db2",
            outcome.output["provenance"]["model"]["expected_taxonomy_embeddings_sha256"],
        )
        self.assertEqual(
            "4648928b006f85d83d28e5a27074ca9363465d82e778d708b369c5eaf54b8ef5",
            outcome.output["provenance"]["model"]["expected_taxonomy_labels_sha256"],
        )
        self.assertEqual(
            "test-decoder",
            outcome.output["provenance"]["preprocessing"]["decoder"],
        )
        self.assertEqual(
            ["Alcedo atthis", "Megaceryle alcyon"],
            outcome.output["provenance"]["configuration"]["candidate_species"],
        )
        self.assertEqual(
            {
                "broad_categories",
                "input",
                "interpretation",
                "provenance",
                "result_type",
                "schema_version",
                "taxonomy",
            },
            set(outcome.output),
        )
        self.assertTrue(
            all(set(item) == {"label", "score"} for item in outcome.output["broad_categories"]["scores"])
        )
        self.assertTrue(
            all(math.isfinite(item["score"]) for item in outcome.output["broad_categories"]["scores"])
        )

    def test_taxonomy_is_gated_and_species_suggestions_are_stably_sorted(self):
        below = FakeProvider(
            broad=tuple(
                {**item, "score": 0.49} if item["classification"] == "animal" else item
                for item in BROAD_SCORES
            )
        )
        below_outcome = self.runner(below).run("private-library", "nested/bird.jpg")

        self.assertEqual([], below.taxonomy_calls)
        self.assertEqual("not_run", below_outcome.output["taxonomy"]["status"])
        self.assertEqual([], below_outcome.output["taxonomy"]["suggestions"])

        at_gate = FakeProvider()
        changed_config = self.config(animal_threshold=0.8)
        at_gate_outcome = self.runner(at_gate, config=changed_config).run(
            "private-library", "nested/bird.jpg"
        )

        self.assertEqual(1, len(at_gate.taxonomy_calls))
        self.assertEqual(
            ("Alcedo atthis", "Megaceryle alcyon"),
            at_gate.taxonomy_calls[0][1],
        )
        self.assertEqual("suggested", at_gate_outcome.output["taxonomy"]["status"])
        self.assertEqual(
            ("Megaceryle alcyon", "Alcedo atthis"),
            tuple(item["species"] for item in at_gate_outcome.output["taxonomy"]["suggestions"]),
        )
        self.assertTrue(
            all(
                set(item)
                == {
                    "class",
                    "common_name",
                    "family",
                    "genus",
                    "kingdom",
                    "order",
                    "phylum",
                    "score",
                    "species",
                    "species_epithet",
                }
                for item in at_gate_outcome.output["taxonomy"]["suggestions"]
            )
        )

    def test_success_is_idempotent_without_redecoding_or_reinference(self):
        provider = FakeProvider()
        decoder = FakeDecoder()
        runner = self.runner(provider, decoder)

        first = runner.run("private-library", "nested/bird.jpg")
        second = runner.run("private-library", "nested/bird.jpg")

        self.assertEqual(first.result_id, second.result_id)
        self.assertEqual(first.output, second.output)
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(1, len(decoder.inputs))
        self.assertEqual(1, len(provider.broad_calls))
        self.assertEqual(1, self.store.analysis_result_count())

    def test_model_config_candidates_and_preprocessing_change_run_identity(self):
        AnalysisConfig, _, ModelSpec, PreprocessingSpec = self.imports()
        base = self.config()
        variants = (
            AnalysisConfig(
                animal_threshold=base.animal_threshold,
                taxonomy_top_k=base.taxonomy_top_k,
                candidate_species=base.candidate_species,
                model=ModelSpec(model_str="hf-hub:imageomics/another-model"),
                preprocessing=base.preprocessing,
            ),
            self.config(animal_threshold=0.6),
            self.config(candidate_species=("Alcedo atthis",)),
            self.config(
                preprocessing=PreprocessingSpec(
                    decoder="test-decoder",
                    decoder_version="2",
                    colour_space="RGB",
                    orientation="preserved",
                    raw_conversion="not-applicable",
                )
            ),
            self.config(
                model=ModelSpec(
                    model_config_sha256="a" * 64,
                )
            ),
            self.config(
                model=ModelSpec(
                    weights_sha256="a" * 64,
                )
            ),
            self.config(
                model=ModelSpec(
                    taxonomy_embeddings_sha256="a" * 64,
                )
            ),
            self.config(
                model=ModelSpec(
                    open_clip_package_version="3.3.1",
                )
            ),
        )

        identities = {self.runner(config=base).ensure_analysis_run()}
        identities.update(self.runner(config=item).ensure_analysis_run() for item in variants)

        self.assertEqual(9, len(identities))

    def test_unindexed_or_changed_source_never_reaches_the_provider(self):
        provider = FakeProvider()
        runner = self.runner(provider)

        with self.assertRaises(ValueError):
            runner.run("private-library", "../outside.jpg")
        self.photo.write_bytes(b"changed without reindexing")
        with self.assertRaisesRegex(Exception, "indexed asset version"):
            runner.run("private-library", "nested/bird.jpg")

        self.assertEqual([], provider.broad_calls)
        failures = self.store.analysis_failures()
        self.assertEqual(1, len(failures))
        self.assertEqual("source_version_mismatch", failures[0]["error_code"])
        self.assertEqual(1, failures[0]["retryable"])

    def test_failures_are_append_only_and_a_retry_can_succeed(self):
        provider = FakeProvider(failures=1)
        runner = self.runner(provider)

        with self.assertRaisesRegex(Exception, "provider_failed"):
            runner.run("private-library", "nested/bird.jpg")
        first_failures = self.store.analysis_failures()
        self.assertEqual(1, len(first_failures))
        self.assertEqual("provider_failed", first_failures[0]["error_code"])
        self.assertNotIn("sensitive details", repr(first_failures))

        outcome = runner.run("private-library", "nested/bird.jpg")

        self.assertFalse(outcome.cached)
        self.assertEqual(first_failures, self.store.analysis_failures())
        self.assertEqual(2, len(provider.broad_calls))
        self.assertEqual(1, self.store.analysis_result_count())

    def test_indexed_version_change_during_inference_is_retryable(self):
        class ReindexingProvider(FakeProvider):
            def predict_broad(inner_self, image, labels):
                self.photo.write_bytes(b"replacement during inference")
                HistoricalIndexer(
                    self.store,
                    self.source_root,
                    "private-library",
                ).run()
                return super().predict_broad(image, labels)

        with self.assertRaisesRegex(Exception, "source_version_mismatch"):
            self.runner(ReindexingProvider()).run(
                "private-library",
                "nested/bird.jpg",
            )

        self.assertEqual(0, self.store.analysis_result_count())
        failures = self.store.analysis_failures()
        self.assertEqual(1, len(failures))
        self.assertEqual("source_version_mismatch", failures[0]["error_code"])
        self.assertEqual(1, failures[0]["retryable"])

    def test_concurrent_identical_result_writers_are_idempotent(self):
        run_id = self.runner().ensure_analysis_run()
        version_id = self.store.current_version_id("private-library", "nested/bird.jpg")
        result_id = "result-concurrent-idempotence"
        output = {"schema_version": 1, "status": "analysed"}
        barrier = threading.Barrier(2)
        errors = []

        def write_result():
            try:
                with HistoricalStore(self.state_root, self.source_root) as store:
                    barrier.wait(timeout=5)
                    store.record_analysis_result(
                        result_id,
                        version_id,
                        run_id,
                        output,
                    )
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=write_result) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual([], errors)
        self.assertEqual(1, self.store.analysis_result_count())

    def test_failure_attempts_are_database_append_only(self):
        runner = self.runner(FakeProvider(failures=1))
        with self.assertRaises(Exception):
            runner.run("private-library", "nested/bird.jpg")
        attempt_id = self.store.analysis_failures()[0]["attempt_id"]

        with self.assertRaises(sqlite3.IntegrityError):
            self.store.connection.execute(
                "UPDATE analysis_attempt_failures SET error_code = 'decoder_failed' "
                "WHERE attempt_id = ?",
                (attempt_id,),
            )
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.connection.execute(
                "DELETE FROM analysis_attempt_failures WHERE attempt_id = ?",
                (attempt_id,),
            )

    def test_schema_extension_upgrades_a_version_one_store_append_only(self):
        upgrade_source = Path(self.temporary_directory.name) / "upgrade-photos"
        upgrade_state = Path(self.temporary_directory.name) / "upgrade-state"
        upgrade_source.mkdir()
        upgrade_state.mkdir()
        database_path = upgrade_state / "historical.sqlite3"
        connection = sqlite3.connect(database_path)
        connection.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (1, 'prior-version')"
        )
        connection.commit()
        connection.close()

        with HistoricalStore(upgrade_state, upgrade_source) as upgraded:
            versions = tuple(
                row[0]
                for row in upgraded.connection.execute(
                    "SELECT version FROM schema_migrations ORDER BY version"
                )
            )
            table = upgraded.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name = 'analysis_attempt_failures'"
            ).fetchone()
            triggers = upgraded.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                "AND (name LIKE 'analysis_attempt_failures_no_%' "
                "OR name LIKE 'analysis_runs_no_%' "
                "OR name LIKE 'analysis_results_no_%') ORDER BY name"
            ).fetchall()

        self.assertEqual((1, 2), versions)
        self.assertEqual("analysis_attempt_failures", table[0])
        self.assertEqual(6, len(triggers))

    def test_analysis_runs_and_results_are_database_immutable(self):
        outcome = self.runner().run("private-library", "nested/bird.jpg")

        statements = (
            (
                "UPDATE analysis_runs SET analyzer_version = 'changed' "
                "WHERE analysis_run_id = ?",
                outcome.analysis_run_id,
            ),
            (
                "DELETE FROM analysis_runs WHERE analysis_run_id = ?",
                outcome.analysis_run_id,
            ),
            (
                "UPDATE analysis_results SET canonical_output_json = '{}' "
                "WHERE result_id = ?",
                outcome.result_id,
            ),
            (
                "DELETE FROM analysis_results WHERE result_id = ?",
                outcome.result_id,
            ),
        )
        for statement, identifier in statements:
            with self.subTest(statement=statement):
                with self.assertRaises(sqlite3.IntegrityError):
                    self.store.connection.execute(statement, (identifier,))

    def test_no_follow_descriptor_read_rejects_a_last_moment_symlink_swap(self):
        outside = Path(self.temporary_directory.name) / "outside.jpg"
        outside.write_bytes(b"outside bytes must never be read")
        provider = FakeProvider()
        runner = self.runner(provider)
        original_open = __import__("os").open

        def swap_before_open(path, flags, *args, **kwargs):
            if path == "bird.jpg":
                self.photo.unlink()
                self.photo.symlink_to(outside)
            return original_open(path, flags, *args, **kwargs)

        with patch(
            "analyzer.kestrel_analyzer.historical_analysis.os.open",
            side_effect=swap_before_open,
        ):
            with self.assertRaisesRegex(Exception, "indexed asset version"):
                runner.run("private-library", "nested/bird.jpg")

        self.assertEqual([], provider.broad_calls)
        self.assertEqual("source_version_mismatch", self.store.analysis_failures()[0]["error_code"])

    def test_invalid_or_non_finite_predictions_are_not_persisted(self):
        invalid_sets = (
            ({"classification": "animal", "score": float("nan")},),
            tuple(item for item in BROAD_SCORES if item["classification"] != "human"),
            BROAD_SCORES + ({"classification": "vehicle", "score": 0.1},),
        )

        for index, broad in enumerate(invalid_sets):
            config = self.config(preprocessing=self.config().preprocessing)
            runner = self.runner(FakeProvider(broad=broad), config=config)
            with self.subTest(index=index):
                with self.assertRaisesRegex(Exception, "invalid_prediction"):
                    runner.run("private-library", "nested/bird.jpg")

        self.assertEqual(0, self.store.analysis_result_count())
        self.assertEqual(3, len(self.store.analysis_failures()))

    def test_decoder_failure_is_retryable_without_calling_the_provider(self):
        provider = FakeProvider()
        runner = self.runner(provider, FailingDecoder())

        with self.assertRaisesRegex(Exception, "decoder_failed"):
            runner.run("private-library", "nested/bird.jpg")

        self.assertEqual([], provider.broad_calls)
        self.assertEqual("decoder_failed", self.store.analysis_failures()[0]["error_code"])
        self.assertEqual(0, self.store.analysis_result_count())

    def test_configuration_rejects_ambiguous_or_non_finite_identity_inputs(self):
        AnalysisConfig, _, ModelSpec, PreprocessingSpec = self.imports()

        with self.assertRaises(ValueError):
            AnalysisConfig(animal_threshold=float("nan"))
        with self.assertRaises(ValueError):
            AnalysisConfig(taxonomy_top_k=0)
        with self.assertRaises(TypeError):
            AnalysisConfig(taxonomy_top_k=True)
        with self.assertRaises(TypeError):
            AnalysisConfig(candidate_species="Alcedo atthis")
        with self.assertRaises(ValueError):
            AnalysisConfig(candidate_species=("Alcedo atthis", "Alcedo atthis"))
        with self.assertRaises(ValueError):
            PreprocessingSpec(decoder="")
        self.assertEqual("checkpoint", ModelSpec(pretrained_str=" checkpoint ").pretrained_str)
        with self.assertRaises(ValueError):
            ModelSpec(model_revision="not-a-revision")
        with self.assertRaises(ValueError):
            ModelSpec(model_config_sha256="not-a-digest")
        with self.assertRaises(ValueError):
            ModelSpec(weights_sha256="not-a-digest")
        with self.assertRaises(ValueError):
            ModelSpec(taxonomy_repo_revision="not-a-revision")
        with self.assertRaises(ValueError):
            ModelSpec(taxonomy_embeddings_sha256="not-a-digest")
        with self.assertRaises(ValueError):
            ModelSpec(taxonomy_labels_sha256="not-a-digest")

    def test_runtime_dependency_pins_match_model_provenance(self):
        _, _, ModelSpec, _ = self.imports()
        requirements = (
            Path(__file__).resolve().parents[1] / "requirements.txt"
        ).read_text(encoding="utf-8").splitlines()
        model = ModelSpec()

        self.assertIn(
            f"{model.package}=={model.package_version}",
            requirements,
        )
        self.assertIn(
            f"{model.open_clip_package}=={model.open_clip_package_version}",
            requirements,
        )

    def test_runner_does_not_change_sources_sidecars_or_review_proposals(self):
        before = self.source_snapshot()
        before_proposals = self.store.review_proposal_count()

        outcome = self.runner().run("private-library", "nested/bird.jpg")

        self.assertEqual(before, self.source_snapshot())
        self.assertEqual(before_proposals, self.store.review_proposal_count())
        serialized = repr(outcome.output).lower()
        for forbidden in ("rating", "pick", "reject", "delete", "move", "proposal", "xmp"):
            self.assertNotIn(forbidden, serialized)


class PyBioClipAdapterTests(unittest.TestCase):
    def local_assets(self):
        from analyzer.kestrel_analyzer.historical_analysis import ModelSpec
        from analyzer.kestrel_analyzer.pybioclip_adapter import LocalBioClipAssets

        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        model_directory = root / "model"
        taxonomy_directory = root / "taxonomy" / "embeddings"
        model_directory.mkdir()
        taxonomy_directory.mkdir(parents=True)
        payloads = {
            model_directory / "open_clip_config.json": b"pinned local model config",
            model_directory / "open_clip_model.safetensors": b"pinned local model weights",
            taxonomy_directory / "txt_emb_species.npy": b"pinned taxonomy embeddings",
            taxonomy_directory / "txt_emb_species.json": b"pinned taxonomy labels",
        }
        for path, payload in payloads.items():
            path.write_bytes(payload)
        digest = lambda path: hashlib.sha256(payloads[path]).hexdigest()
        model = ModelSpec(
            model_config_sha256=digest(model_directory / "open_clip_config.json"),
            weights_sha256=digest(model_directory / "open_clip_model.safetensors"),
            taxonomy_embeddings_sha256=digest(
                taxonomy_directory / "txt_emb_species.npy"
            ),
            taxonomy_labels_sha256=digest(
                taxonomy_directory / "txt_emb_species.json"
            ),
        )
        assets = LocalBioClipAssets(
            model_directory=model_directory,
            model_config=model_directory / "open_clip_config.json",
            model_weights=model_directory / "open_clip_model.safetensors",
            taxonomy_embeddings=taxonomy_directory / "txt_emb_species.npy",
            taxonomy_labels=taxonomy_directory / "txt_emb_species.json",
        )
        return model, lambda configured_model: assets

    def test_adapter_is_lazy_and_uses_pybioclip_216_classifier_apis(self):
        from analyzer.kestrel_analyzer.historical_analysis import ModelSpec
        from analyzer.kestrel_analyzer.pybioclip_adapter import PyBioClipProvider

        model, local_assets_resolver = self.local_assets()
        calls = []
        instances = []
        network_resolver_calls = []

        class CustomLabelsClassifier:
            def __init__(self, cls_ary, **kwargs):
                instances.append(("broad", tuple(cls_ary), kwargs, self))

            def predict(self, image, **kwargs):
                calls.append(("broad-predict", image, kwargs))
                return tuple(
                    {
                        "classification": label,
                        "score": 0.025 if label.startswith("scene without ") else 0.225,
                    }
                    for label in instances[0][1]
                )

        class TreeOfLifeClassifier:
            def __init__(self, **kwargs):
                instances.append(("taxonomy", kwargs, self))
                calls.append(
                    (
                        "taxonomy-artifacts",
                        self.get_cached_datafile("embeddings/txt_emb_species.npy"),
                        self.get_cached_datafile("embeddings/txt_emb_species.json"),
                    )
                )

            def create_taxa_filter(self, rank, values):
                calls.append(("create-filter", rank, tuple(values)))
                return "filter"

            def apply_filter(self, value):
                calls.append(("apply-filter", value))

            def predict(self, image, rank, **kwargs):
                calls.append(("taxonomy-predict", image, rank, kwargs))
                return TAXONOMY_SCORES

        rank = types.SimpleNamespace(SPECIES="species-rank")
        modules = {
            "bioclip.predict": types.SimpleNamespace(
                CustomLabelsClassifier=CustomLabelsClassifier,
                TreeOfLifeClassifier=TreeOfLifeClassifier,
            ),
            "bioclip": types.SimpleNamespace(Rank=rank),
        }

        def loader(name):
            calls.append(("import", name))
            return modules[name]

        def forbidden_network_resolver(*args, **kwargs):
            network_resolver_calls.append((args, kwargs))
            raise AssertionError("remote Hugging Face metadata must not be queried")

        hugging_face_module = types.ModuleType("huggingface_hub")
        hugging_face_module.model_info = forbidden_network_resolver
        hugging_face_module.dataset_info = forbidden_network_resolver
        hugging_face_module.get_hf_file_metadata = forbidden_network_resolver
        hugging_face_module.hf_hub_url = forbidden_network_resolver
        hugging_face_module.hf_hub_download = forbidden_network_resolver

        self.assertNotIn("bioclip", sys.modules)
        with patch.dict(sys.modules, {"huggingface_hub": hugging_face_module}):
            provider = PyBioClipProvider(
                model,
                local_assets_resolver=local_assets_resolver,
                module_loader=loader,
                package_version_resolver=lambda package: {
                    "pybioclip": "2.1.6",
                    "open-clip-torch": "3.3.0",
                }[package],
            )
            self.assertEqual([], calls)
            self.assertEqual([], instances)

            broad = provider.predict_broad(
                "image", ("landscape", "architecture", "human", "animal")
            )
            taxonomy = provider.predict_taxonomy("image", ("Alcedo atthis",), 3)

        self.assertEqual(
            tuple(
                {"classification": label, "score": 0.9}
                for label in ("landscape", "architecture", "human", "animal")
            ),
            broad,
        )
        self.assertEqual(TAXONOMY_SCORES, taxonomy)
        self.assertEqual(
            (
                "landscape",
                "scene without landscape",
                "architecture",
                "scene without architecture",
                "human",
                "scene without human",
                "animal",
                "scene without animal",
            ),
            instances[0][1],
        )
        self.assertEqual(
            f"local-dir:{local_assets_resolver(model).model_directory}",
            instances[0][2]["model_str"],
        )
        self.assertEqual("cpu", instances[0][2]["device"])
        self.assertIn(("broad-predict", ["image"], {"k": 8}), calls)
        self.assertIn(("create-filter", "species-rank", ("Alcedo atthis",)), calls)
        self.assertIn(("apply-filter", "filter"), calls)
        self.assertIn(("taxonomy-predict", ["image"], "species-rank", {"k": 3}), calls)
        self.assertIn(
            (
                "taxonomy-artifacts",
                str(local_assets_resolver(model).taxonomy_embeddings),
                str(local_assets_resolver(model).taxonomy_labels),
            ),
            calls,
        )

        provider.predict_broad("second", ("landscape", "architecture", "human", "animal"))
        provider.predict_taxonomy("second", ("Alcedo atthis",), 1)
        with self.assertRaises(ValueError):
            provider.predict_broad("image", ("landscape", "animal"))
        with self.assertRaises(ValueError):
            provider.predict_taxonomy("image", (), 1)
        with self.assertRaises(ValueError):
            PyBioClipProvider(ModelSpec(package_version="2.1.5"), module_loader=loader)
        self.assertEqual([], network_resolver_calls)
        self.assertNotIn("bioclip", sys.modules)

    def test_adapter_rejects_package_drift_and_invalid_pairwise_scores(self):
        from analyzer.kestrel_analyzer.pybioclip_adapter import PyBioClipProvider

        model, local_assets_resolver = self.local_assets()
        labels = ("landscape", "architecture", "human", "animal")
        loader_calls = []

        class Classifier:
            predictions = ()

            def __init__(self, cls_ary, **kwargs):
                self.prompts = tuple(cls_ary)

            def predict(self, image, **kwargs):
                return self.predictions or tuple(
                    {"classification": prompt, "score": 0.0}
                    for prompt in self.prompts
                )

        def loader(name):
            loader_calls.append(name)
            return types.SimpleNamespace(CustomLabelsClassifier=Classifier)

        def provider(**overrides):
            values = {
                "local_assets_resolver": local_assets_resolver,
                "module_loader": loader,
                "package_version_resolver": lambda package: {
                    model.package: model.package_version,
                    model.open_clip_package: model.open_clip_package_version,
                }[package],
            }
            values.update(overrides)
            return PyBioClipProvider(model, **values)

        with self.assertRaisesRegex(RuntimeError, "installed pybioclip version"):
            provider(
                package_version_resolver=lambda package: {
                    "pybioclip": "2.1.5",
                    "open-clip-torch": "3.3.0",
                }[package]
            ).predict_broad("image", labels)
        with self.assertRaisesRegex(RuntimeError, "installed open-clip-torch version"):
            provider(
                package_version_resolver=lambda package: {
                    "pybioclip": "2.1.6",
                    "open-clip-torch": "3.2.0",
                }[package]
            ).predict_broad("image", labels)
        self.assertEqual([], loader_calls)

        with self.assertRaisesRegex(ValueError, "positive mass"):
            provider().predict_broad("image", labels)
        Classifier.predictions = (
            {"classification": "landscape", "score": 1.0},
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            provider().predict_broad("image", labels)

        local_assets_resolver(model).model_weights.write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "local BioCLIP artifact"):
            provider().predict_broad("image", labels)

    def test_adapter_normalizes_each_prompt_pair_independently(self):
        from analyzer.kestrel_analyzer.pybioclip_adapter import PyBioClipProvider

        model, local_assets_resolver = self.local_assets()
        labels = ("landscape", "architecture", "human", "animal")
        predictions = (
            {"classification": "landscape", "score": 0.20},
            {"classification": "scene without landscape", "score": 0.10},
            {"classification": "architecture", "score": 0.01},
            {"classification": "scene without architecture", "score": 0.09},
            {"classification": "human", "score": 0.30},
            {"classification": "scene without human", "score": 0.30},
            {"classification": "animal", "score": 0.39},
            {"classification": "scene without animal", "score": 0.61},
        )

        class Classifier:
            def __init__(self, cls_ary, **kwargs):
                self.prompts = tuple(cls_ary)

            def predict(self, image, **kwargs):
                return predictions

        provider = PyBioClipProvider(
            model,
            local_assets_resolver=local_assets_resolver,
            module_loader=lambda name: types.SimpleNamespace(
                CustomLabelsClassifier=Classifier
            ),
            package_version_resolver=lambda package: {
                "pybioclip": "2.1.6",
                "open-clip-torch": "3.3.0",
            }[package],
        )

        result = provider.predict_broad("image", labels)

        self.assertAlmostEqual(2 / 3, result[0]["score"])
        self.assertAlmostEqual(0.1, result[1]["score"])
        self.assertAlmostEqual(0.5, result[2]["score"])
        self.assertAlmostEqual(0.39, result[3]["score"])


if __name__ == "__main__":
    unittest.main()
