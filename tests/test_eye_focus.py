import unittest
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np


def load_detector_module():
    path = Path(__file__).parents[1] / "analyzer" / "kestrel_analyzer" / "ml" / "eye_focus.py"
    spec = importlib.util.spec_from_file_location("eye_focus", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EyeFocusAnalyzer = load_detector_module().EyeFocusAnalyzer


class EyeFocusAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = EyeFocusAnalyzer(eye_confidence_threshold=0.0)

    @staticmethod
    def image_with_eye() -> tuple[np.ndarray, np.ndarray]:
        image = np.full((128, 128, 3), 180, dtype=np.uint8)
        cv2.circle(image, (64, 20), 4, (0, 0, 0), thickness=-1)
        cv2.circle(image, (63, 19), 1, (255, 255, 255), thickness=-1)
        return image, np.ones((128, 128), dtype=np.uint8)

    def assert_advisory_result(self, result, status):
        self.assertEqual(status, result["analysis_status"])
        self.assertTrue(result["provenance"]["advisory"])
        self.assertEqual("heuristic_eye_focus", result["provenance"]["detector"])
        self.assertEqual(status, result["provenance"]["status"])

    def test_empty_or_invalid_input_is_unknown_and_advisory(self):
        for image in (None, np.array([], dtype=np.uint8), np.zeros((8,), dtype=np.uint8)):
            with self.subTest(image=image):
                result = self.analyzer.analyze(image, np.ones((8, 8), dtype=np.uint8))
                self.assert_advisory_result(result, "unknown")
                self.assertFalse(result["eye_detected"])
                self.assertIsNone(result["eye_bbox"])

    def test_invalid_array_like_image_or_mask_never_raises(self):
        class BrokenArray:
            size = 1

            def __array__(self, *args, **kwargs):
                raise RuntimeError("invalid test array")

        image, mask = self.image_with_eye()
        for invalid_image, invalid_mask in ((BrokenArray(), mask), (image, BrokenArray())):
            with self.subTest(invalid_image=invalid_image, invalid_mask=invalid_mask):
                result = self.analyzer.analyze(invalid_image, invalid_mask)
                self.assert_advisory_result(result, "unknown")
                self.assertEqual("input_normalization_error", result["provenance"]["reason"])

    def test_non_finite_confidence_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            EyeFocusAnalyzer(float("nan"))

    def test_grayscale_and_rgba_inputs_normalize_to_the_same_result(self):
        image, mask = self.image_with_eye()
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rgba = np.dstack((image, np.full(image.shape[:2], 17, dtype=np.uint8)))

        grayscale_result = self.analyzer.analyze(grayscale, mask)
        rgba_result = self.analyzer.analyze(rgba, mask)

        self.assert_advisory_result(grayscale_result, "detected")
        self.assertEqual(grayscale_result, rgba_result)

    def test_unit_interval_float_rgb_normalizes_like_uint8_rgb(self):
        image, mask = self.image_with_eye()
        float_image = image.astype(np.float32) / 255.0

        self.assertEqual(self.analyzer.analyze(image, mask), self.analyzer.analyze(float_image, mask))

    def test_mismatched_mask_is_normalized_and_boxes_are_bounded(self):
        image, _ = self.image_with_eye()
        result = self.analyzer.analyze(image, np.ones((64, 64), dtype=np.uint8))

        self.assert_advisory_result(result, "detected")
        for bbox_name in ("head_bbox", "eye_bbox"):
            bbox = result[bbox_name]
            self.assertIsNotNone(bbox)
            self.assertGreaterEqual(bbox["x_min"], 0)
            self.assertGreaterEqual(bbox["y_min"], 0)
            self.assertLessEqual(bbox["x_max"], image.shape[1])
            self.assertLessEqual(bbox["y_max"], image.shape[0])
            self.assertEqual(bbox["x_max"] - bbox["x_min"], bbox["width"])
            self.assertEqual(bbox["y_max"] - bbox["y_min"], bbox["height"])

    def test_empty_mask_is_unknown_not_a_negative_eye_result(self):
        image, _ = self.image_with_eye()
        result = self.analyzer.analyze(image, np.zeros((128, 128), dtype=np.uint8))

        self.assert_advisory_result(result, "unknown")
        self.assertEqual("mask_empty", result["provenance"]["reason"])

    def test_no_candidate_is_distinct_from_unknown(self):
        image = np.full((128, 128, 3), 180, dtype=np.uint8)
        result = self.analyzer.analyze(image, np.ones((128, 128), dtype=np.uint8))

        self.assert_advisory_result(result, "no_eye")
        self.assertFalse(result["eye_detected"])

    def test_unexpected_detector_failure_is_reported_as_error(self):
        image, mask = self.image_with_eye()
        self.analyzer._detect_eye = lambda *_: (_ for _ in ()).throw(RuntimeError("test failure"))

        result = self.analyzer.analyze(image, mask)

        self.assert_advisory_result(result, "error")
        self.assertEqual("detector_error", result["provenance"]["reason"])
        self.assertNotIn("test failure", repr(result))

    def test_output_is_deterministic(self):
        image, mask = self.image_with_eye()
        self.assertEqual(self.analyzer.analyze(image, mask), self.analyzer.analyze(image, mask))

    def test_sharp_eye_scores_higher_than_blurred_eye(self):
        image, mask = self.image_with_eye()
        blurred = cv2.GaussianBlur(image, (9, 9), 2)

        sharp_result = self.analyzer.analyze(image, mask)
        blurred_result = self.analyzer.analyze(blurred, mask)

        self.assert_advisory_result(sharp_result, "detected")
        self.assert_advisory_result(blurred_result, "detected")
        self.assertGreater(sharp_result["eye_focus_score"], blurred_result["eye_focus_score"])


if __name__ == "__main__":
    unittest.main()
