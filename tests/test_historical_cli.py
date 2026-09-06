import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch


BROAD = (
    {"classification": "landscape", "score": 0.1},
    {"classification": "architecture", "score": 0.2},
    {"classification": "human", "score": 0.3},
    {"classification": "animal", "score": 0.9},
)
TAXONOMY = (
    {
        "score": 0.8,
        "kingdom": "Animalia",
        "phylum": "Chordata",
        "class": "Aves",
        "order": "Coraciiformes",
        "family": "Alcedinidae",
        "genus": "Alcedo",
        "species_epithet": "atthis",
        "species": "Alcedo atthis",
        "common_name": "Common kingfisher",
    },
)


class FakeDecoder:
    def decode(self, source_bytes, suffix):
        return source_bytes, suffix


class FakeProvider:
    def __init__(self, *, failures=0, stop_event=None):
        self.failures = failures
        self.stop_event = stop_event
        self.calls = 0

    def predict_broad(self, image, labels):
        self.calls += 1
        if self.failures:
            self.failures -= 1
            raise RuntimeError("sensitive provider detail")
        if self.stop_event is not None:
            self.stop_event.set()
        return BROAD

    def predict_taxonomy(self, image, candidate_species, top_k):
        return TAXONOMY


class HistoricalCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.source_root = root / "photos"
        self.state_root = root / "state"
        self.artifact_root = root / "artifacts"
        self.source_root.mkdir()
        self.artifact_root.mkdir()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write(self, relative_path, content=b"photo"):
        path = self.source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    @staticmethod
    def verified_report():
        from analyzer.kestrel_analyzer.historical_artifacts import ArtifactVerification

        return ArtifactVerification(
            status="verified",
            artifacts=(
                {"path": "model/open_clip_config.json", "status": "verified"},
            ),
        )

    def run_cli(self, arguments, provider, *, stop_event=None):
        from analyzer.kestrel_analyzer import historical_cli

        output = io.StringIO()
        with patch.object(
            historical_cli,
            "verify_artifacts",
            return_value=self.verified_report(),
        ):
            exit_code = historical_cli.main(
                arguments,
                output=output,
                provider_factory=lambda _model, _root: provider,
                decoder=FakeDecoder(),
                stop_event=stop_event,
                install_signal_handlers=False,
            )
        lines = output.getvalue().splitlines()
        self.assertEqual(1, len(lines))
        self.assertEqual(
            json.dumps(json.loads(lines[0]), separators=(",", ":"), sort_keys=True),
            lines[0],
        )
        return exit_code, json.loads(lines[0])

    def common(self):
        return [
            "--source-root",
            str(self.source_root),
            "--state-root",
            str(self.state_root),
            "--library-id",
            "private-library",
        ]

    def test_run_indexes_and_analyzes_with_hard_bounds_and_retry_accounting(self):
        photo = self.write("b.jpg", b"b")
        first = self.write("A.jpg", b"a")
        sidecar = self.write("A.xmp", b"lightroom")
        before = {
            path.name: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in (photo, first, sidecar)
        }
        provider = FakeProvider(failures=1)

        code, summary = self.run_cli(
            [
                "run",
                *self.common(),
                "--artifact-root",
                str(self.artifact_root),
                "--max-items",
                "2",
                "--limit",
                "1",
                "--max-retries",
                "1",
            ],
            provider,
        )

        self.assertEqual(0, code)
        self.assertEqual("bounded", summary["status"])
        self.assertEqual(2, summary["index"]["observed_count"])
        self.assertEqual(1, summary["analysis"]["selected_count"])
        self.assertEqual(2, summary["analysis"]["attempt_count"])
        self.assertEqual(1, summary["analysis"]["retry_count"])
        self.assertEqual(1, summary["analysis"]["failure_attempt_count"])
        self.assertEqual(0, summary["analysis"]["failed_count"])
        self.assertEqual(
            [{"count": 1, "error_code": "provider_failed"}],
            summary["analysis"]["errors"],
        )
        self.assertEqual("A.jpg", summary["analysis"]["results"][0]["relative_path"])
        after = {
            path.name: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in (photo, first, sidecar)
        }
        self.assertEqual(before, after)
        from analyzer.kestrel_analyzer.historical_store import HistoricalStore

        with HistoricalStore(self.state_root, self.source_root) as store:
            self.assertEqual(0, store.review_proposal_count())

    def test_index_bound_leaves_a_resumable_running_scan(self):
        self.write("a.jpg")
        self.write("b.jpg")

        code, summary = self.run_cli(
            ["index", *self.common(), "--max-items", "1"],
            FakeProvider(),
        )

        self.assertEqual(0, code)
        self.assertEqual("bounded", summary["status"])
        self.assertEqual("running", summary["index"]["scan_status"])
        self.assertEqual(1, summary["index"]["observed_count"])
        self.assertIsInstance(summary["index"]["scan_id"], str)

    def test_headless_index_never_changes_proposal_or_application_rows(self):
        from analyzer.kestrel_analyzer.historical_store import HistoricalStore
        from analyzer.kestrel_analyzer.review_policy import Decision, ReviewProposal

        photo = self.write("bird.jpg", b"first version")
        self.run_cli(
            ["index", *self.common(), "--max-items", "1"],
            FakeProvider(),
        )
        with HistoricalStore(self.state_root, self.source_root) as store:
            version_id = store.current_version_id("private-library", "bird.jpg")
            run_id = store.ensure_analysis_run("2.0", "model", "policy", {})
            store.record_analysis_result("result-one", version_id, run_id, {"score": 0.1})
            asset_id = store.asset_id("private-library", "bird.jpg")
            store.record_review_proposal(
                "proposal-one",
                asset_id,
                "result-one",
                ReviewProposal(
                    decision=Decision.MANUAL_REVIEW_FOCUS,
                    result_id="result-one",
                    review_reason="subject_focus_below_threshold",
                    keyword="AI Review|Focus",
                    suggested_color="Red",
                ),
            )
            store.connection.execute(
                """
                INSERT INTO application_operations(
                    operation_id, proposal_id, asset_id, status, prepared_at
                ) VALUES ('operation-one', 'proposal-one', ?, 'prepared', 'fixed')
                """,
                (asset_id,),
            )
            proposal_before = tuple(
                tuple(row)
                for row in store.connection.execute(
                    "SELECT * FROM review_proposals ORDER BY proposal_id"
                )
            )
            application_before = tuple(
                tuple(row)
                for row in store.connection.execute(
                    "SELECT * FROM application_operations ORDER BY operation_id"
                )
            )

        photo.write_bytes(b"a different second version")
        self.run_cli(
            ["index", *self.common(), "--max-items", "1"],
            FakeProvider(),
        )

        with HistoricalStore(self.state_root, self.source_root) as store:
            proposal_after = tuple(
                tuple(row)
                for row in store.connection.execute(
                    "SELECT * FROM review_proposals ORDER BY proposal_id"
                )
            )
            application_after = tuple(
                tuple(row)
                for row in store.connection.execute(
                    "SELECT * FROM application_operations ORDER BY operation_id"
                )
            )
            self.assertEqual("proposed", store.proposal_lifecycle("proposal-one"))

        self.assertEqual(proposal_before, proposal_after)
        self.assertEqual(application_before, application_after)

    def test_exhausted_retries_are_counted_per_attempt_without_exception_text(self):
        self.write("bird.jpg")
        self.run_cli(
            ["index", *self.common(), "--max-items", "1"],
            FakeProvider(),
        )

        code, summary = self.run_cli(
            [
                "analyze",
                *self.common(),
                "--artifact-root",
                str(self.artifact_root),
                "--limit",
                "1",
                "--max-retries",
                "1",
            ],
            FakeProvider(failures=2),
        )

        self.assertEqual(1, code)
        self.assertEqual("completed_with_failures", summary["status"])
        self.assertEqual(2, summary["analysis"]["attempt_count"])
        self.assertEqual(1, summary["analysis"]["retry_count"])
        self.assertEqual(2, summary["analysis"]["failure_attempt_count"])
        self.assertEqual(1, summary["analysis"]["failed_count"])
        self.assertNotIn("sensitive provider detail", json.dumps(summary))

    def test_stop_flag_finishes_current_asset_then_emits_interrupted_summary(self):
        self.write("a.jpg", b"a")
        self.write("b.jpg", b"b")
        self.run_cli(
            ["index", *self.common(), "--max-items", "2"],
            FakeProvider(),
        )
        stop_event = threading.Event()
        provider = FakeProvider(stop_event=stop_event)

        code, summary = self.run_cli(
            [
                "analyze",
                *self.common(),
                "--artifact-root",
                str(self.artifact_root),
                "--limit",
                "2",
                "--max-retries",
                "0",
            ],
            provider,
            stop_event=stop_event,
        )

        self.assertEqual(143, code)
        self.assertEqual("interrupted", summary["status"])
        self.assertEqual(1, summary["analysis"]["success_count"])
        self.assertEqual(1, summary["analysis"]["remaining_count"])
        self.assertEqual(0, summary["analysis"]["failure_attempt_count"])

    def test_run_interrupted_during_artifact_verification_keeps_scan_summary(self):
        from analyzer.kestrel_analyzer import historical_cli
        from analyzer.kestrel_analyzer.historical_artifacts import ArtifactInterrupted

        self.write("bird.jpg")
        output = io.StringIO()
        event = threading.Event()

        def interrupt_verification(*_args, **_kwargs):
            event.set()
            raise ArtifactInterrupted()

        with patch.object(
            historical_cli,
            "verify_artifacts",
            side_effect=interrupt_verification,
        ):
            code = historical_cli.main(
                [
                    "run",
                    *self.common(),
                    "--artifact-root",
                    str(self.artifact_root),
                    "--max-items",
                    "1",
                    "--limit",
                    "1",
                ],
                output=output,
                provider_factory=lambda _model, _root: FakeProvider(),
                decoder=FakeDecoder(),
                stop_event=event,
                install_signal_handlers=False,
            )

        summary = json.loads(output.getvalue())
        self.assertEqual(143, code)
        self.assertEqual("interrupted", summary["status"])
        self.assertEqual("private-library", summary["library_id"])
        self.assertEqual("completed", summary["index"]["scan_status"])
        self.assertIsInstance(summary["index"]["scan_id"], str)

    def test_sigterm_handler_sets_flag_without_raising(self):
        from analyzer.kestrel_analyzer.historical_cli import termination_handlers

        installed = {}

        def fake_signal(signum, handler):
            previous = installed.get(signum, object())
            installed[signum] = handler
            return previous

        event = threading.Event()
        with patch("analyzer.kestrel_analyzer.historical_cli.signal.signal", fake_signal):
            with termination_handlers(event):
                installed[15](15, None)
                self.assertTrue(event.is_set())

    def test_real_model_smoke_path_targets_one_explicit_indexed_asset(self):
        self.write("nested/bird.jpg")
        self.run_cli(
            ["index", *self.common(), "--max-items", "1"],
            FakeProvider(),
        )

        code, summary = self.run_cli(
            [
                "smoke",
                *self.common(),
                "--artifact-root",
                str(self.artifact_root),
                "--relative-path",
                "nested/bird.jpg",
            ],
            FakeProvider(),
        )

        self.assertEqual(0, code)
        self.assertEqual("real_model", summary["analysis"]["mode"])
        self.assertEqual("nested/bird.jpg", summary["analysis"]["results"][0]["relative_path"])
        self.assertEqual("verified", summary["artifacts"]["status"])

        cached_provider = FakeProvider()
        code, cached_summary = self.run_cli(
            [
                "smoke",
                *self.common(),
                "--artifact-root",
                str(self.artifact_root),
                "--relative-path",
                "nested/bird.jpg",
            ],
            cached_provider,
        )
        self.assertEqual(0, code)
        self.assertTrue(cached_summary["analysis"]["results"][0]["cached"])
        self.assertEqual(1, cached_provider.calls)

    def test_parser_rejects_unbounded_or_excessive_work_as_machine_json(self):
        from analyzer.kestrel_analyzer import historical_cli

        output = io.StringIO()
        code = historical_cli.main(
            ["index", *self.common(), "--max-items", "1000001"],
            output=output,
            install_signal_handlers=False,
        )

        self.assertEqual(2, code)
        self.assertEqual("invalid_arguments", json.loads(output.getvalue())["error_code"])

    def test_artifact_commands_honor_preexisting_termination_request(self):
        from analyzer.kestrel_analyzer import historical_cli

        for subcommand in ("verify", "provision"):
            with self.subTest(subcommand=subcommand):
                output = io.StringIO()
                event = threading.Event()
                event.set()
                code = historical_cli.main(
                    [
                        "artifacts",
                        subcommand,
                        "--artifact-root",
                        str(self.artifact_root),
                    ],
                    output=output,
                    stop_event=event,
                    install_signal_handlers=False,
                )

                self.assertEqual(143, code)
                self.assertEqual(
                    {
                        "command": f"artifacts.{subcommand}",
                        "schema_version": 1,
                        "status": "interrupted",
                    },
                    json.loads(output.getvalue()),
                )


if __name__ == "__main__":
    unittest.main()
