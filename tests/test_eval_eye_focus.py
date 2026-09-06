import csv
import importlib.util
import io
import sys
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


def load_evaluator_module():
    path = Path(__file__).parents[1] / "scripts" / "eval_eye_focus.py"
    spec = importlib.util.spec_from_file_location("eval_eye_focus", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EyeFocusEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_evaluator_module()

    def test_json_parsing_and_evaluation_are_deterministic(self):
        detected_provenance = {"advisory": True, "detector": "heuristic_eye_focus", "version": 1, "status": "detected", "reason": "detected"}
        no_eye_provenance = {"advisory": True, "detector": "heuristic_eye_focus", "version": 1, "status": "no_eye", "reason": "candidate_not_found"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "records.json"
            path.write_text(
                json.dumps(
                    [
                        {"scene": "zebra", "crops": [{"quality": "0.5", "combined_score": 0.7, "analysis_status": "detected", "provenance": detected_provenance, "eye_detected": True, "eye_confidence": 0.9, "eye_focus_score": 0.8}]},
                        {"scene": "alpha", "crops": [{"quality": 0.5, "combined_score": 0.5, "analysis_status": "no_eye", "provenance": no_eye_provenance, "eye_detected": False, "eye_confidence": 0.0, "eye_focus_score": 0.0}, {"quality": 0.5, "combined_score": 0.5, "analysis_status": "no_eye", "provenance": no_eye_provenance, "eye_detected": False, "eye_confidence": 0.0, "eye_focus_score": 0.0}]},
                    ]
                ),
                encoding="utf-8",
            )
            records = self.evaluator.load_records(path)

        results = [self.evaluator.evaluate_record(record) for record in records]
        self.assertEqual(0, results[1].baseline_primary)
        self.assertEqual(0, results[1].prototype_primary)

        first_output = io.StringIO()
        second_output = io.StringIO()
        with redirect_stdout(first_output):
            self.evaluator.print_cases(sorted(results, key=lambda item: item.scene), show="all", limit=20)
        with redirect_stdout(second_output):
            self.evaluator.print_cases(sorted(results, key=lambda item: item.scene), show="all", limit=20)
        self.assertEqual(first_output.getvalue(), second_output.getvalue())
        self.assertLess(first_output.getvalue().find("scene=alpha"), first_output.getvalue().find("scene=zebra"))
        self.assertIn("analysis_status='detected'", first_output.getvalue())

    def test_advisory_metadata_validation_and_non_finite_scores(self):
        valid_provenance = {"advisory": True, "detector": "heuristic_eye_focus", "version": 1, "status": "detected", "reason": "detected"}
        record = {"scene": "advisory", "crops": [
            {"quality": float("inf"), "combined_score": float("inf"), "analysis_status": "detected", "provenance": valid_provenance, "eye_detected": True, "eye_confidence": float("nan"), "eye_focus_score": float("inf")},
            {"quality": 0.4, "combined_score": 0.5, "analysis_status": "no_eye", "provenance": {**valid_provenance, "status": "no_eye", "reason": "candidate_not_found"}, "eye_detected": False, "eye_confidence": 0.0, "eye_focus_score": 0.0},
            {"quality": 0.3, "combined_score": 0.3, "eye_detected": False, "eye_confidence": 0.0, "eye_focus_score": 0.0},
            {"quality": 0.2, "combined_score": 0.2, "analysis_status": "error", "provenance": {**valid_provenance, "status": "unknown"}, "eye_detected": False, "eye_confidence": 1.5, "eye_focus_score": 0.0},
        ]}

        result = self.evaluator.evaluate_record(record)

        self.assertEqual(1, result.baseline_primary)
        self.assertEqual(1, result.prototype_primary)
        self.assertIn("crop_0_invalid_quality", result.missing_reasons)
        self.assertIn("crop_0_invalid_combined_score", result.missing_reasons)
        self.assertIn("crop_0_invalid_eye_confidence", result.missing_reasons)
        self.assertIn("crop_0_invalid_eye_focus_score", result.missing_reasons)
        self.assertIn("crop_2_missing_analysis_status", result.missing_reasons)
        self.assertIn("crop_3_invalid_provenance_status", result.missing_reasons)
        self.assertIn("crop_3_invalid_eye_confidence", result.missing_reasons)

    def test_advisory_metadata_requires_boolean_detection_and_reason(self):
        issues = self.evaluator.advisory_metadata_issues(
            {
                "analysis_status": "no_eye",
                "provenance": {"advisory": True, "detector": "heuristic_eye_focus", "version": 1, "status": "no_eye"},
                "eye_detected": "false",
            },
            0,
        )

        self.assertIn("crop_0_invalid_provenance_reason", issues)
        self.assertIn("crop_0_invalid_eye_detected", issues)

    def test_terminal_control_characters_in_scene_are_escaped(self):
        result = self.evaluator.CaseResult("scene\n\x1b[2J", 0, 0, 0.0, 0.0, 0.0, False, (), ())
        output = io.StringIO()

        with redirect_stdout(output):
            self.evaluator.print_case(result)

        self.assertIn("scene=scene\\n\\x1b[2J", output.getvalue())
        self.assertNotIn("scene\n", output.getvalue())

    def test_malformed_advisory_fields_are_all_reported_without_raising(self):
        for status in ([], {}):
            with self.subTest(status=status):
                issues = self.evaluator.advisory_metadata_issues(
                    {"analysis_status": status, "provenance": "not-an-object", "eye_detected": True}, 0
                )

                self.assertIn("crop_0_invalid_analysis_status", issues)
                self.assertIn("crop_0_missing_provenance", issues)

    def test_cli_escapes_control_characters_in_input_errors(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad\x1b]2;spoof\x07.csv"
            path.write_text("filename,crops_json\nphoto,not-json\n", encoding="utf-8")
            stderr = io.StringIO()
            original_argv = sys.argv
            try:
                sys.argv = ["eval_eye_focus.py", "--input", str(path)]
                with redirect_stderr(stderr):
                    self.assertEqual(2, self.evaluator.main())
            finally:
                sys.argv = original_argv

        self.assertIn("\\x1b]2;spoof\\x07", stderr.getvalue())
        self.assertNotIn("\x1b]2;spoof\x07", stderr.getvalue())

    def test_csv_parsing_rejects_invalid_crops_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "kingfisher_database.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("filename", "crops_json"))
                writer.writeheader()
                writer.writerow({"filename": "broken", "crops_json": "not-json"})
            with self.assertRaises(self.evaluator.EvalInputError):
                self.evaluator.load_records(path)


if __name__ == "__main__":
    unittest.main()
