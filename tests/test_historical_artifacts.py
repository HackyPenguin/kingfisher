import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
import unittest

from analyzer.kestrel_analyzer.historical_analysis import ModelSpec


class HistoricalArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.payloads = {
            "model/open_clip_config.json": b"model config\n",
            "model/open_clip_model.safetensors": b"model weights\n",
            "taxonomy/embeddings/txt_emb_species.npy": b"taxonomy embeddings\n",
            "taxonomy/embeddings/txt_emb_species.json": b"taxonomy labels\n",
        }
        digest = lambda path: hashlib.sha256(self.payloads[path]).hexdigest()
        self.model = ModelSpec(
            model_config_sha256=digest("model/open_clip_config.json"),
            weights_sha256=digest("model/open_clip_model.safetensors"),
            taxonomy_embeddings_sha256=digest(
                "taxonomy/embeddings/txt_emb_species.npy"
            ),
            taxonomy_labels_sha256=digest(
                "taxonomy/embeddings/txt_emb_species.json"
            ),
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_manifest_uses_exact_immutable_revisions_and_digests(self):
        from analyzer.kestrel_analyzer.historical_artifacts import artifact_manifest

        manifest = artifact_manifest(ModelSpec())

        self.assertEqual(1, manifest["schema_version"])
        self.assertEqual(
            "2957b322090f9cb17ae72c71981c7218a28d81e0",
            manifest["model"]["revision"],
        )
        self.assertEqual(
            "5f2dc493b3dc0e544438a04038ab15faa646b749",
            manifest["taxonomy"]["revision"],
        )
        self.assertEqual(
            {
                "model/open_clip_config.json": "1bf947e96e943fe50efd5c3e26c37f843a2fa3c358967719a68c8a6d17ce68c8",
                "model/open_clip_model.safetensors": "b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef",
                "taxonomy/embeddings/txt_emb_species.json": "4648928b006f85d83d28e5a27074ca9363465d82e778d708b369c5eaf54b8ef5",
                "taxonomy/embeddings/txt_emb_species.npy": "c72442de7b0cb7fcb55ab7ca08099d0f42fbd6769efe16ca64c1daa7a8b87db2",
            },
            {
                item["path"]: item["sha256"]
                for item in manifest["artifacts"]
            },
        )
        self.assertTrue(
            all(
                manifest["model"]["revision"] in item["url"]
                or manifest["taxonomy"]["revision"] in item["url"]
                for item in manifest["artifacts"]
            )
        )

    def test_provision_stages_verifies_and_atomically_installs_exact_tree(self):
        from analyzer.kestrel_analyzer.historical_artifacts import (
            artifact_descriptors,
            provision_artifacts,
            verify_artifacts,
        )

        by_url = {
            descriptor.url: self.payloads[descriptor.relative_path]
            for descriptor in artifact_descriptors(self.model)
        }
        requested = []

        def fetch(url):
            requested.append(url)
            return BytesIO(by_url[url])

        destination = self.root / "artifacts"
        result = provision_artifacts(destination, model=self.model, fetcher=fetch)

        self.assertTrue(result.provisioned)
        self.assertEqual("verified", result.verification.status)
        self.assertEqual(sorted(by_url), requested)
        self.assertEqual(
            self.payloads,
            {
                path: (destination / path).read_bytes()
                for path in sorted(self.payloads)
            },
        )
        manifest = json.loads((destination / "manifest.json").read_text("utf-8"))
        self.assertEqual(self.model.model_revision, manifest["model"]["revision"])
        self.assertEqual("verified", verify_artifacts(destination, self.model).status)
        verification = verify_artifacts(destination, self.model).to_dict()
        self.assertEqual(self.model.model_revision, verification["model_revision"])
        self.assertEqual(
            self.model.taxonomy_repo_revision,
            verification["taxonomy_revision"],
        )
        self.assertEqual(
            sorted(
                descriptor.sha256
                for descriptor in artifact_descriptors(self.model)
            ),
            sorted(item["sha256"] for item in verification["artifacts"]),
        )
        self.assertEqual([], list(self.root.glob(".artifacts.*")))

        second = provision_artifacts(
            destination,
            model=self.model,
            fetcher=lambda _url: self.fail("verified artifacts must not be fetched again"),
        )
        self.assertFalse(second.provisioned)

    def test_digest_failure_leaves_no_partial_destination_or_staging_tree(self):
        from analyzer.kestrel_analyzer.historical_artifacts import (
            ArtifactProvisionError,
            provision_artifacts,
        )

        destination = self.root / "artifacts"
        with self.assertRaisesRegex(ArtifactProvisionError, "digest_mismatch"):
            provision_artifacts(
                destination,
                model=self.model,
                fetcher=lambda _url: BytesIO(b"incorrect"),
            )

        self.assertFalse(destination.exists())
        self.assertEqual([], list(self.root.glob(".artifacts.*")))

    def test_existing_mismatch_is_never_overwritten(self):
        from analyzer.kestrel_analyzer.historical_artifacts import (
            ArtifactProvisionError,
            provision_artifacts,
            verify_artifacts,
        )

        destination = self.root / "artifacts"
        destination.mkdir()
        sentinel = destination / "do-not-overwrite"
        sentinel.write_bytes(b"operator data")

        report = verify_artifacts(destination, self.model)
        self.assertEqual("invalid", report.status)
        with self.assertRaisesRegex(ArtifactProvisionError, "destination_not_pristine"):
            provision_artifacts(
                destination,
                model=self.model,
                fetcher=lambda _url: self.fail("must fail before network use"),
            )
        self.assertEqual(b"operator data", sentinel.read_bytes())

    def test_cancellation_removes_staging_and_never_publishes_partial_tree(self):
        from analyzer.kestrel_analyzer.historical_artifacts import (
            ArtifactInterrupted,
            artifact_descriptors,
            provision_artifacts,
        )

        event = threading.Event()
        by_url = {
            descriptor.url: self.payloads[descriptor.relative_path]
            for descriptor in artifact_descriptors(self.model)
        }

        class CancellingStream(BytesIO):
            def read(self, size=-1):
                chunk = super().read(size)
                event.set()
                return chunk

        destination = self.root / "artifacts"
        with self.assertRaises(ArtifactInterrupted):
            provision_artifacts(
                destination,
                model=self.model,
                fetcher=lambda url: CancellingStream(by_url[url]),
                should_stop=event.is_set,
            )

        self.assertFalse(destination.exists())
        self.assertEqual([], list(self.root.glob(".artifacts.*")))

    def test_atomic_publish_refuses_even_an_empty_existing_directory(self):
        from analyzer.kestrel_analyzer.historical_artifacts import (
            _rename_directory_without_replace,
        )

        staged = self.root / "staged"
        destination = self.root / "artifacts"
        staged.mkdir()
        destination.mkdir()

        with self.assertRaises(OSError):
            _rename_directory_without_replace(staged, destination)

        self.assertTrue(staged.is_dir())
        self.assertTrue(destination.is_dir())


if __name__ == "__main__":
    unittest.main()
