"""Advisory, heuristic eye-focus detection for already-cropped bird images.

This module deliberately has no pipeline integration.  Its results are
diagnostic metadata only and must not be used to change crop selection,
ratings, metadata, or review decisions without a separately approved change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class _EyeFocusDefaults:
    eye_detected: bool = False
    eye_confidence: float = 0.0
    eye_focus_score: float = 0.0
    eye_bbox: dict[str, int] | None = None
    head_bbox: dict[str, int] | None = None


class EyeFocusAnalyzer:
    """Estimate whether a bird crop contains a sharp, plausible eye.

    The detector uses a supplied bird mask to estimate an upper head region,
    then scores compact dark contours in that region.  It is intentionally
    advisory: ``analysis_status`` and ``provenance`` always state whether a
    result is unknown, an internal error, no detected eye, or a detection.
    """

    DETECTOR_NAME = "heuristic_eye_focus"
    DETECTOR_VERSION = 1

    def __init__(self, eye_confidence_threshold: float = 0.6):
        threshold = float(eye_confidence_threshold)
        if not np.isfinite(threshold):
            raise ValueError("eye_confidence_threshold must be finite")
        self.eye_confidence_threshold = float(np.clip(threshold, 0.0, 1.0))

    def analyze(self, cropped_image: np.ndarray, cropped_mask: np.ndarray | None) -> dict[str, Any]:
        """Return advisory eye metadata without raising for malformed inputs."""
        try:
            gray, image_reason = self._normalize_image(cropped_image)
            if gray is None:
                return self._result("unknown", image_reason or "invalid_image")

            mask, mask_reason = self._normalize_mask(cropped_mask, gray.shape)
            if mask is None:
                return self._result("unknown", mask_reason or "mask_invalid")
        except Exception:
            return self._result("unknown", "input_normalization_error")

        try:
            head_bbox = self._estimate_head_bbox(mask)
            if head_bbox is None:
                return self._result("unknown", "mask_empty")

            head_patch = gray[
                head_bbox["y_min"]:head_bbox["y_max"],
                head_bbox["x_min"]:head_bbox["x_max"],
            ]
            if head_patch.size == 0:
                return self._result("unknown", "head_region_empty", head_bbox=head_bbox)

            eye_bbox, confidence, focus_score = self._detect_eye(head_patch, head_bbox)
            if eye_bbox is None:
                return self._result("no_eye", "candidate_not_found", head_bbox=head_bbox)

            confidence = float(np.clip(confidence, 0.0, 1.0))
            focus_score = float(np.clip(focus_score, 0.0, 1.0))
            if confidence < self.eye_confidence_threshold:
                return self._result(
                    "no_eye",
                    "below_confidence_threshold",
                    eye_confidence=confidence,
                    eye_focus_score=focus_score,
                    head_bbox=head_bbox,
                )
            return self._result(
                "detected",
                "detected",
                eye_detected=True,
                eye_confidence=confidence,
                eye_focus_score=focus_score,
                eye_bbox=eye_bbox,
                head_bbox=head_bbox,
            )
        except Exception:
            return self._result("error", "detector_error")

    @classmethod
    def _result(cls, status: str, reason: str, **updates: Any) -> dict[str, Any]:
        defaults = _EyeFocusDefaults()
        result: dict[str, Any] = {
            "eye_detected": defaults.eye_detected,
            "eye_confidence": defaults.eye_confidence,
            "eye_focus_score": defaults.eye_focus_score,
            "eye_bbox": defaults.eye_bbox,
            "head_bbox": defaults.head_bbox,
            "analysis_status": status,
            "provenance": {
                "advisory": True,
                "detector": cls.DETECTOR_NAME,
                "version": cls.DETECTOR_VERSION,
                "status": status,
                "reason": reason,
            },
        }
        result.update(updates)
        return result

    @staticmethod
    def _normalize_image(image: np.ndarray | None) -> tuple[np.ndarray | None, str | None]:
        if image is None or getattr(image, "size", 0) == 0:
            return None, "image_empty"
        try:
            array = np.asarray(image)
            if not np.isfinite(array).all():
                return None, "image_values_invalid"
            if np.issubdtype(array.dtype, np.floating) and array.size and array.min() >= 0 and array.max() <= 1:
                array = array * 255.0
            if array.ndim == 2:
                gray = array
            elif array.ndim == 3 and array.shape[2] == 1:
                gray = array[:, :, 0]
            elif array.ndim == 3 and array.shape[2] == 3:
                gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
            elif array.ndim == 3 and array.shape[2] == 4:
                gray = cv2.cvtColor(array, cv2.COLOR_RGBA2GRAY)
            else:
                return None, "image_shape_invalid"
            if gray.shape[0] == 0 or gray.shape[1] == 0:
                return None, "image_values_invalid"
            if gray.dtype != np.uint8:
                gray = np.clip(gray, 0, 255).astype(np.uint8)
            return gray, None
        except (TypeError, ValueError, cv2.error):
            return None, "image_invalid"

    @staticmethod
    def _normalize_mask(
        mask: np.ndarray | None, shape: tuple[int, int]
    ) -> tuple[np.ndarray | None, str | None]:
        if mask is None or getattr(mask, "size", 0) == 0:
            return None, "mask_missing"
        try:
            array = np.asarray(mask)
            if array.ndim == 3:
                array = np.any(array > 0, axis=2)
            if array.ndim != 2 or not np.isfinite(array).all():
                return None, "mask_invalid"
            normalized = (array > 0).astype(np.uint8)
            if normalized.shape != shape:
                normalized = cv2.resize(
                    normalized, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
                )
                normalized = (normalized > 0).astype(np.uint8)
            if not np.any(normalized):
                return None, "mask_empty"
            return normalized, None
        except (TypeError, ValueError, cv2.error):
            return None, "mask_invalid"

    @staticmethod
    def _estimate_head_bbox(mask: np.ndarray) -> dict[str, int] | None:
        ys, xs = np.where(mask > 0)
        if not len(xs):
            return None
        x_min, x_max = int(xs.min()), int(xs.max()) + 1
        y_min, y_max = int(ys.min()), int(ys.max()) + 1
        width, height = max(1, x_max - x_min), max(1, y_max - y_min)
        head_height = max(24, int(round(height * 0.35)))
        head_width = max(24, int(round(width * 0.5)))
        image_h, image_w = mask.shape
        center_x = int(round((x_min + x_max) / 2.0))
        head_x_min = max(0, center_x - head_width // 2)
        head_x_max = min(image_w, head_x_min + head_width)
        head_x_min = max(0, head_x_max - head_width)
        head_y_min = max(0, y_min)
        head_y_max = min(image_h, head_y_min + head_height)
        head_y_min = max(0, head_y_max - head_height)
        if head_x_max <= head_x_min or head_y_max <= head_y_min:
            return None
        return {
            "x_min": int(head_x_min), "x_max": int(head_x_max),
            "y_min": int(head_y_min), "y_max": int(head_y_max),
            "width": int(head_x_max - head_x_min), "height": int(head_y_max - head_y_min),
        }

    @staticmethod
    def _detect_eye(
        head_patch: np.ndarray, head_bbox: dict[str, int]
    ) -> tuple[dict[str, int] | None, float, float]:
        blurred = cv2.GaussianBlur(head_patch, (5, 5), 0)
        _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        patch_h, patch_w = head_patch.shape
        patch_area = float(max(1, patch_h * patch_w))
        best_candidate: tuple[dict[str, int], float] | None = None
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = float(width * height)
            if not patch_area * 0.002 <= area <= patch_area * 0.08:
                continue
            aspect_ratio = width / float(max(1, height))
            if not 0.4 <= aspect_ratio <= 2.5:
                continue
            center_y_norm = (y + height / 2.0) / float(max(1, patch_h))
            score = area / patch_area
            score *= max(0.0, 1.0 - abs(center_y_norm - 0.45))
            score *= max(0.0, 1.2 - abs(aspect_ratio - 1.0))
            bbox = {
                "x_min": int(head_bbox["x_min"] + x), "x_max": int(head_bbox["x_min"] + x + width),
                "y_min": int(head_bbox["y_min"] + y), "y_max": int(head_bbox["y_min"] + y + height),
                "width": int(width), "height": int(height),
            }
            if best_candidate is None or score > best_candidate[1]:
                best_candidate = (bbox, score)
        if best_candidate is None:
            return None, 0.0, 0.0
        eye_bbox, candidate_score = best_candidate
        eye_patch = head_patch[
            eye_bbox["y_min"] - head_bbox["y_min"]:eye_bbox["y_max"] - head_bbox["y_min"],
            eye_bbox["x_min"] - head_bbox["x_min"]:eye_bbox["x_max"] - head_bbox["x_min"],
        ]
        if eye_patch.size == 0:
            return None, 0.0, 0.0
        focus_score = float(np.clip(cv2.Laplacian(eye_patch, cv2.CV_32F).var() / 4000.0, 0.0, 1.0))
        confidence = float(np.clip(candidate_score * 18.0 + focus_score * 0.45, 0.0, 1.0))
        return eye_bbox, confidence, focus_score
