import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analyzer.kestrel_analyzer.historical_index import (
    HistoricalIndexer,
    SourceChangedDuringHash,
)
from analyzer.kestrel_analyzer.historical_store import HistoricalStore
from analyzer.kestrel_analyzer.review_policy import Decision, ReviewProposal


class HistoricalStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.source_root = root / "photos"
        self.state_root = root / "state"
        self.source_root.mkdir()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write(self, relative_path: str, content: bytes = b"photo") -> Path:
        path = self.source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def open_store(self):
        return HistoricalStore(self.state_root, self.source_root)

    def test_state_root_must_be_separate_from_photo_root(self):
        with self.assertRaises(ValueError):
            HistoricalStore(self.source_root / ".state", self.source_root)

    def test_indexer_cannot_read_from_a_different_source_root(self):
        other_root = Path(self.temporary_directory.name) / "other-photos"
        other_root.mkdir()
        with self.open_store() as store:
            with self.assertRaises(ValueError):
                HistoricalIndexer(store, other_root, "private-library")

    def test_store_enables_durable_sqlite_guards(self):
        with self.open_store() as store:
            self.assertEqual("wal", store.pragma_value("journal_mode").lower())
            self.assertEqual(1, store.pragma_value("foreign_keys"))
            self.assertEqual(2, store.pragma_value("synchronous"))
            self.assertEqual("ok", store.integrity_check())

    def test_two_identical_scans_are_idempotent(self):
        self.write("nested/bird.CR3", b"same bytes")

        with self.open_store() as store:
            indexer = HistoricalIndexer(store, self.source_root, "private-library")
            first = indexer.run()
            second = indexer.run()

            self.assertEqual("completed", first.status)
            self.assertEqual("completed", second.status)
            self.assertEqual(1, store.asset_count())
            self.assertEqual(1, store.asset_version_count())
            self.assertEqual(("nested/bird.CR3",), store.list_asset_paths())

    def test_scan_does_not_write_photos_or_sidecars(self):
        photo = self.write("nested/bird.CR3", b"source bytes")
        sidecar = self.write("nested/bird.xmp", b"lightroom metadata")
        before = {
            path.relative_to(self.source_root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in (photo, sidecar)
        }

        with self.open_store() as store:
            HistoricalIndexer(store, self.source_root, "private-library").run()

        after = {
            path.relative_to(self.source_root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in (photo, sidecar)
        }
        self.assertEqual(before, after)

    def test_changed_content_creates_a_new_immutable_version(self):
        path = self.write("bird.NEF", b"version one")

        with self.open_store() as store:
            indexer = HistoricalIndexer(store, self.source_root, "private-library")
            indexer.run()
            first_version = store.current_version_id("private-library", "bird.NEF")
            path.write_bytes(b"version two is different")
            indexer.run()
            second_version = store.current_version_id("private-library", "bird.NEF")

            self.assertNotEqual(first_version, second_version)
            self.assertEqual(2, store.asset_version_count())

    def test_full_hash_audit_catches_timestamp_preserving_replacement(self):
        path = self.write("bird.NEF", b"AAAA")
        initial_stat = path.stat()

        with self.open_store() as store:
            indexer = HistoricalIndexer(store, self.source_root, "private-library")
            indexer.run()
            path.write_bytes(b"BBBB")
            os.utime(path, ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns))

            indexer.run()
            self.assertEqual(1, store.asset_version_count())
            indexer.run(full_hash_audit=True)
            self.assertEqual(2, store.asset_version_count())

    def test_interrupted_scan_restarts_safely_and_only_complete_scan_marks_missing(self):
        self.write("a.CR3")
        removable = self.write("b.CR3")

        with self.open_store() as store:
            indexer = HistoricalIndexer(store, self.source_root, "private-library")
            initial = indexer.run()
            removable.unlink()
            interrupted = indexer.run(max_items=0)

            self.assertEqual("running", interrupted.status)
            self.assertEqual(("a.CR3", "b.CR3"), store.list_asset_paths())

            resumed = indexer.run(scan_id=interrupted.scan_id)
            self.assertEqual("completed", resumed.status)
            self.assertEqual(("a.CR3",), store.list_asset_paths())
            self.assertEqual("missing", store.asset_state("private-library", "b.CR3"))
            self.assertEqual("completed", store.scan_status(resumed.scan_id))
            self.assertEqual("a.CR3", store.scan_checkpoint(resumed.scan_id))

    def test_unreadable_directory_never_marks_previously_seen_assets_missing(self):
        self.write("blocked/bird.CR3")

        with self.open_store() as store:
            indexer = HistoricalIndexer(store, self.source_root, "private-library")
            indexer.run()
            original_scandir = os.scandir

            def fail_for_blocked(path):
                if Path(path).name == "blocked":
                    raise PermissionError("simulated unreadable directory")
                return original_scandir(path)

            with patch(
                "analyzer.kestrel_analyzer.historical_index.os.scandir",
                side_effect=fail_for_blocked,
            ):
                incomplete = indexer.run()

            self.assertEqual("running", incomplete.status)
            self.assertEqual("active", store.asset_state("private-library", "blocked/bird.CR3"))
            self.assertEqual(("blocked/bird.CR3",), store.list_asset_paths())

    def test_model_or_policy_spec_change_makes_current_asset_stale(self):
        self.write("bird.CR3")

        with self.open_store() as store:
            HistoricalIndexer(store, self.source_root, "private-library").run()
            version_id = store.current_version_id("private-library", "bird.CR3")
            first_run = store.ensure_analysis_run(
                analyzer_version="2.0",
                model_digest="model-a",
                policy_digest="policy-a",
                config={"threshold": 0.45},
            )
            changed_run = store.ensure_analysis_run(
                analyzer_version="2.0",
                model_digest="model-a",
                policy_digest="policy-b",
                config={"threshold": 0.40},
            )

            self.assertEqual(("bird.CR3",), store.stale_asset_paths(first_run))
            store.record_analysis_result(
                result_id="result-one",
                asset_version_id=version_id,
                analysis_run_id=first_run,
                output={"focus_score": 0.2},
            )
            self.assertEqual((), store.stale_asset_paths(first_run))
            self.assertEqual(("bird.CR3",), store.stale_asset_paths(changed_run))

    def test_result_recording_is_idempotent_but_immutable(self):
        self.write("bird.CR3")

        with self.open_store() as store:
            HistoricalIndexer(store, self.source_root, "private-library").run()
            version_id = store.current_version_id("private-library", "bird.CR3")
            run_id = store.ensure_analysis_run("2.0", "model", "policy", {})
            store.record_analysis_result("result-one", version_id, run_id, {"score": 0.2})
            store.record_analysis_result("result-one", version_id, run_id, {"score": 0.2})

            with self.assertRaises(ValueError):
                store.record_analysis_result("result-one", version_id, run_id, {"score": 0.9})

    def test_source_version_change_supersedes_and_hides_old_proposal(self):
        path = self.write("bird.CR3", b"version one")
        manifest_path = self.state_root / "exports" / "review.json"

        with self.open_store() as store:
            indexer = HistoricalIndexer(store, self.source_root, "private-library")
            indexer.run()
            version_id = store.current_version_id("private-library", "bird.CR3")
            run_id = store.ensure_analysis_run("2.0", "model", "policy", {})
            store.record_analysis_result("result-one", version_id, run_id, {"score": 0.1})
            store.record_review_proposal(
                "proposal-one",
                store.asset_id("private-library", "bird.CR3"),
                "result-one",
                ReviewProposal(
                    decision=Decision.MANUAL_REVIEW_FOCUS,
                    result_id="result-one",
                    review_reason="subject_focus_below_threshold",
                    keyword="AI Review|Focus",
                    suggested_color="Red",
                ),
            )

            path.write_bytes(b"version two")
            indexer.run()

            self.assertEqual("superseded", store.proposal_lifecycle("proposal-one"))
            manifest = json.loads(store.export_dry_run_manifest(manifest_path))
            self.assertEqual([], manifest["proposals"])

    def test_unstable_source_withdraws_old_proposal_until_stably_rehashed(self):
        path = self.write("bird.CR3", b"version one")
        manifest_path = self.state_root / "exports" / "review.json"

        with self.open_store() as store:
            indexer = HistoricalIndexer(store, self.source_root, "private-library")
            indexer.run()
            version_id = store.current_version_id("private-library", "bird.CR3")
            run_id = store.ensure_analysis_run("2.0", "model", "policy", {})
            store.record_analysis_result("result-one", version_id, run_id, {"score": 0.1})
            store.record_review_proposal(
                "proposal-one",
                store.asset_id("private-library", "bird.CR3"),
                "result-one",
                ReviewProposal(
                    decision=Decision.MANUAL_REVIEW_FOCUS,
                    result_id="result-one",
                    review_reason="subject_focus_below_threshold",
                    keyword="AI Review|Focus",
                    suggested_color="Red",
                ),
            )
            path.write_bytes(b"changing version")

            with patch(
                "analyzer.kestrel_analyzer.historical_store.hash_file_stably",
                side_effect=SourceChangedDuringHash("simulated concurrent import"),
            ):
                incomplete = indexer.run()

            self.assertEqual("running", incomplete.status)
            self.assertEqual("superseded", store.proposal_lifecycle("proposal-one"))
            with self.assertRaises(ValueError):
                store.current_version_id("private-library", "bird.CR3")
            manifest = json.loads(store.export_dry_run_manifest(manifest_path))
            self.assertEqual([], manifest["proposals"])

    def test_dry_run_manifest_is_stable_closed_and_actionable_only(self):
        self.write("z/bird.CR3")
        self.write("a/bird.CR3")
        manifest_path = self.state_root / "exports" / "review.json"

        with self.open_store() as store:
            HistoricalIndexer(store, self.source_root, "private-library").run()
            run_id = store.ensure_analysis_run("2.0", "model", "policy", {})
            paths = store.list_asset_paths()
            for index, relative_path in enumerate(reversed(paths)):
                version_id = store.current_version_id("private-library", relative_path)
                result_id = f"result-{index}"
                store.record_analysis_result(result_id, version_id, run_id, {"focus_score": 0.1})
                store.record_review_proposal(
                    proposal_id=f"proposal-{index}",
                    asset_id=store.asset_id("private-library", relative_path),
                    result_id=result_id,
                    proposal=ReviewProposal(
                        decision=Decision.MANUAL_REVIEW_FOCUS,
                        result_id=result_id,
                        review_reason="subject_focus_below_threshold",
                        keyword="AI Review|Focus",
                        suggested_color="Red",
                    ),
                )

            payload_one = store.export_dry_run_manifest(manifest_path)
            payload_two = store.export_dry_run_manifest(manifest_path)

        self.assertEqual(payload_one, payload_two)
        document = json.loads(payload_one)
        self.assertEqual(1, document["schema_version"])
        self.assertEqual(["a/bird.CR3", "z/bird.CR3"], [item["relative_path"] for item in document["proposals"]])
        self.assertEqual(
            {
                "asset_id",
                "decision",
                "keyword",
                "library_id",
                "proposal_id",
                "relative_path",
                "result_id",
                "review_reason",
                "suggested_color",
                "supersedes",
            },
            set(document["proposals"][0]),
        )


if __name__ == "__main__":
    unittest.main()
