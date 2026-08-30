"""Non-destructive review decisions for historical photo analysis.

This module is deliberately independent of the filesystem, ML runtimes, and
Lightroom.  It turns explicit catalogue and analysis signals into a narrow,
serializable proposal.  The proposal schema contains no operation capable of
deleting, moving, rejecting, or rating a source photograph.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Iterable


class Decision(str, Enum):
    """The complete set of outcomes allowed by the review workflow."""

    NONE = "none"
    PROTECTED_KEEP = "protected_keep"
    CLEAR_AI_REVIEW = "clear_ai_review"
    MANUAL_REVIEW_FOCUS = "manual_review_focus"
    MANUAL_REVIEW_UNCERTAIN = "manual_review_uncertain"


_PICK_STATES = frozenset({"unflagged", "picked", "rejected"})
_AI_KEYWORDS = frozenset({"AI Review|Focus", "AI Review|Uncertain"})
_PROTECTION_REASONS = frozenset(
    {
        "develop_edits",
        "manual_rating",
        "manual_pick",
        "manual_reject",
        "user_color_label",
        "user_keywords",
        "external_xmp",
    }
)
_NONE_REASONS = frozenset(
    {
        "catalogue_state_unavailable",
        "not_wildlife",
        "wildlife_subject_not_detected",
        "subject_focus_at_or_above_threshold",
    }
)
_REVIEW_CONTRACT = {
    Decision.MANUAL_REVIEW_FOCUS: (
        frozenset({"subject_focus_below_threshold"}),
        "AI Review|Focus",
    ),
    Decision.MANUAL_REVIEW_UNCERTAIN: (
        frozenset({"focus_evidence_missing", "focus_confidence_below_threshold"}),
        "AI Review|Uncertain",
    ),
}


def _require_bool(name: str, value: object) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")


def _normalise_keywords(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("user_keywords must be an iterable of strings")
    normalised: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("user_keywords must contain only strings")
        keyword = value.strip()
        if keyword:
            normalised.add(keyword)
    return tuple(sorted(normalised, key=lambda item: (item.casefold(), item)))


def _normalise_identifier(name: str, value: object, *, required: bool) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalised = value.strip()
    if required and not normalised:
        raise ValueError(f"{name} must not be blank")
    return normalised


def _validate_unit_interval(name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number or None")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")


@dataclass(frozen=True)
class AppliedMetadataReceipt:
    """Trusted ledger evidence for metadata previously applied by Kingfisher.

    Receipts must be loaded from the application ledger, never inferred from
    Lightroom values.  Matching additionally requires the same asset identity
    and exact current keyword/color state, preventing a stale receipt from
    claiming a user's metadata.
    """

    asset_id: str
    result_id: str
    metadata_revision: str
    keyword: str
    color_label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "asset_id",
            _normalise_identifier("asset_id", self.asset_id, required=True),
        )
        object.__setattr__(
            self,
            "result_id",
            _normalise_identifier("result_id", self.result_id, required=True),
        )
        object.__setattr__(
            self,
            "metadata_revision",
            _normalise_identifier(
                "metadata_revision",
                self.metadata_revision,
                required=True,
            ),
        )
        if self.keyword not in _AI_KEYWORDS:
            raise ValueError(f"keyword must be one of {sorted(_AI_KEYWORDS)}")
        if self.color_label not in (None, "Red"):
            raise ValueError("color_label must be None or Red")

    def matches_current_state(
        self,
        *,
        asset_id: str,
        active_application_result_id: str,
        metadata_revision: str,
        keywords: tuple[str, ...],
        color_label: str,
    ) -> bool:
        if (
            self.asset_id != asset_id
            or self.result_id != active_application_result_id
            or self.metadata_revision != metadata_revision
            or self.keyword not in keywords
        ):
            return False
        if self.color_label is not None and self.color_label != color_label:
            return False
        return True

    def to_supersession(self) -> AppliedMetadataSupersession:
        return AppliedMetadataSupersession(
            asset_id=self.asset_id,
            applied_result_id=self.result_id,
            metadata_revision=self.metadata_revision,
            keyword=self.keyword,
            color_label=self.color_label,
        )


@dataclass(frozen=True)
class AppliedMetadataSupersession:
    """Exact ledger-owned metadata that an atomic update may replace or clear."""

    asset_id: str
    applied_result_id: str
    metadata_revision: str
    keyword: str
    color_label: str | None = None

    def __post_init__(self) -> None:
        for name in ("asset_id", "applied_result_id", "metadata_revision"):
            object.__setattr__(
                self,
                name,
                _normalise_identifier(name, getattr(self, name), required=True),
            )
        if self.keyword not in _AI_KEYWORDS:
            raise ValueError(f"keyword must be one of {sorted(_AI_KEYWORDS)}")
        if self.color_label not in (None, "Red"):
            raise ValueError("color_label must be None or Red")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "asset_id": self.asset_id,
            "applied_result_id": self.applied_result_id,
            "metadata_revision": self.metadata_revision,
            "keyword": self.keyword,
            "color_label": self.color_label,
        }


@dataclass(frozen=True)
class CatalogSignals:
    """Human-owned intent exported from Lightroom or a conservative fallback."""

    catalogue_state_available: bool
    asset_id: str = ""
    active_application_result_id: str = ""
    metadata_revision: str = ""
    has_develop_edits: bool = False
    manual_rating: int = 0
    pick_state: str = "unflagged"
    color_label: str = ""
    keywords: tuple[str, ...] = ()
    external_xmp_present: bool = False
    application_receipt: AppliedMetadataReceipt | None = None

    def __post_init__(self) -> None:
        _require_bool("catalogue_state_available", self.catalogue_state_available)
        _require_bool("has_develop_edits", self.has_develop_edits)
        _require_bool("external_xmp_present", self.external_xmp_present)

        asset_id = _normalise_identifier("asset_id", self.asset_id, required=False)
        object.__setattr__(self, "asset_id", asset_id)
        active_application_result_id = _normalise_identifier(
            "active_application_result_id",
            self.active_application_result_id,
            required=False,
        )
        object.__setattr__(
            self,
            "active_application_result_id",
            active_application_result_id,
        )
        metadata_revision = _normalise_identifier(
            "metadata_revision",
            self.metadata_revision,
            required=False,
        )
        object.__setattr__(self, "metadata_revision", metadata_revision)

        if type(self.manual_rating) is not int:
            raise TypeError("manual_rating must be an integer")
        if not 0 <= self.manual_rating <= 5:
            raise ValueError("manual_rating must be within [0, 5]")

        if not isinstance(self.pick_state, str):
            raise TypeError("pick_state must be a string")
        pick_state = self.pick_state.strip().lower()
        if pick_state not in _PICK_STATES:
            raise ValueError(f"pick_state must be one of {sorted(_PICK_STATES)}")
        object.__setattr__(self, "pick_state", pick_state)

        if not isinstance(self.color_label, str):
            raise TypeError("color_label must be a string")
        color_label = self.color_label.strip()
        object.__setattr__(self, "color_label", color_label)

        object.__setattr__(self, "keywords", _normalise_keywords(self.keywords))

        if self.application_receipt is not None:
            if not isinstance(self.application_receipt, AppliedMetadataReceipt):
                raise TypeError("application_receipt must be AppliedMetadataReceipt or None")
            if not self.catalogue_state_available:
                raise ValueError("application_receipt requires available catalogue state")
            if not asset_id:
                raise ValueError("asset_id is required when application_receipt is present")
            if not active_application_result_id:
                raise ValueError(
                    "active_application_result_id is required when "
                    "application_receipt is present"
                )
            if not metadata_revision:
                raise ValueError(
                    "metadata_revision is required when application_receipt is present"
                )

    def _verified_ai_metadata(self) -> tuple[frozenset[str], str | None]:
        receipt = self.verified_application_receipt()
        if receipt is None:
            return frozenset(), None
        return frozenset({receipt.keyword}), receipt.color_label

    def verified_application_receipt(self) -> AppliedMetadataReceipt | None:
        receipt = self.application_receipt
        if receipt is None or not receipt.matches_current_state(
            asset_id=self.asset_id,
            active_application_result_id=self.active_application_result_id,
            metadata_revision=self.metadata_revision,
            keywords=self.keywords,
            color_label=self.color_label,
        ):
            return None
        return receipt

    def protection_reasons(self) -> tuple[str, ...]:
        """Return deterministic reasons why automated review must be suppressed."""

        reasons: list[str] = []
        ai_keywords, ai_color_label = self._verified_ai_metadata()
        if self.has_develop_edits:
            reasons.append("develop_edits")
        if self.manual_rating > 0:
            reasons.append("manual_rating")
        if self.pick_state == "picked":
            reasons.append("manual_pick")
        elif self.pick_state == "rejected":
            reasons.append("manual_reject")
        if self.color_label and self.color_label != ai_color_label:
            reasons.append("user_color_label")
        if any(keyword not in ai_keywords for keyword in self.keywords):
            reasons.append("user_keywords")
        if self.external_xmp_present and not self.catalogue_state_available:
            reasons.append("external_xmp")
        return tuple(reasons)


@dataclass(frozen=True)
class AnalysisSignals:
    """Analysis evidence relevant to the first wildlife-focus policy."""

    result_id: str
    is_wildlife: bool
    subject_detected: bool
    focus_score: float | None = None
    focus_confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "result_id",
            _normalise_identifier("result_id", self.result_id, required=True),
        )
        _require_bool("is_wildlife", self.is_wildlife)
        _require_bool("subject_detected", self.subject_detected)
        _validate_unit_interval("focus_score", self.focus_score)
        _validate_unit_interval("focus_confidence", self.focus_confidence)


@dataclass(frozen=True)
class ReviewProposal:
    """An additive metadata proposal or an explicit no-action outcome."""

    decision: Decision
    result_id: str
    protected_reasons: tuple[str, ...] = ()
    review_reason: str | None = None
    keyword: str | None = None
    suggested_color: str | None = None
    supersedes: AppliedMetadataSupersession | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, Decision):
            raise TypeError("decision must be a Decision")
        object.__setattr__(
            self,
            "result_id",
            _normalise_identifier("result_id", self.result_id, required=True),
        )
        if self.suggested_color not in (None, "Red"):
            raise ValueError("suggested_color must be None or Red")
        if self.supersedes is not None and not isinstance(
            self.supersedes,
            AppliedMetadataSupersession,
        ):
            raise TypeError("supersedes must be AppliedMetadataSupersession or None")

        if self.decision in _REVIEW_CONTRACT:
            allowed_reasons, required_keyword = _REVIEW_CONTRACT[self.decision]
            if self.review_reason not in allowed_reasons:
                raise ValueError("review_reason is not allowed for this decision")
            if self.keyword != required_keyword:
                raise ValueError("keyword is not allowed for this decision")
            if self.protected_reasons:
                raise ValueError("review outcomes cannot contain protected reasons")
            return

        if self.decision is Decision.CLEAR_AI_REVIEW:
            if self.review_reason != "analysis_no_longer_requires_review":
                raise ValueError("clear outcomes require the canonical review reason")
            if self.keyword is not None or self.suggested_color is not None:
                raise ValueError("clear outcomes cannot recommend replacement metadata")
            if self.protected_reasons:
                raise ValueError("clear outcomes cannot contain protected reasons")
            if self.supersedes is None:
                raise ValueError("clear outcomes require receipt-bound supersession")
            return

        if self.keyword is not None or self.suggested_color is not None:
            raise ValueError("non-review outcomes cannot recommend metadata changes")
        if self.supersedes is not None:
            raise ValueError("this outcome cannot supersede applied metadata")

        if self.decision is Decision.PROTECTED_KEEP:
            if self.review_reason != "human_curation_protected":
                raise ValueError("protected outcomes require the canonical review reason")
            if not self.protected_reasons:
                raise ValueError("protected outcomes require at least one reason")
            if any(reason not in _PROTECTION_REASONS for reason in self.protected_reasons):
                raise ValueError("protected outcome contains an unknown reason")
            if len(set(self.protected_reasons)) != len(self.protected_reasons):
                raise ValueError("protected reasons must be unique")
            return

        if self.protected_reasons:
            raise ValueError("no-action outcomes cannot contain protected reasons")
        if self.review_reason not in _NONE_REASONS:
            raise ValueError("review_reason is not allowed for a no-action outcome")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable, intentionally non-destructive manifest shape."""

        return {
            "decision": self.decision.value,
            "result_id": self.result_id,
            "protected_reasons": list(self.protected_reasons),
            "review_reason": self.review_reason,
            "keyword": self.keyword,
            "suggested_color": self.suggested_color,
            "supersedes": self.supersedes.to_dict() if self.supersedes else None,
        }


class ReviewPolicy:
    """Evaluate catalogue intent before considering wildlife focus evidence."""

    def __init__(self, focus_threshold: float = 0.45, min_focus_confidence: float = 0.65):
        _validate_unit_interval("focus_threshold", focus_threshold)
        _validate_unit_interval("min_focus_confidence", min_focus_confidence)
        self.focus_threshold = float(focus_threshold)
        self.min_focus_confidence = float(min_focus_confidence)

    def evaluate(
        self,
        catalog: CatalogSignals,
        analysis: AnalysisSignals,
    ) -> ReviewProposal:
        if not isinstance(catalog, CatalogSignals):
            raise TypeError("catalog must be CatalogSignals")
        if not isinstance(analysis, AnalysisSignals):
            raise TypeError("analysis must be AnalysisSignals")

        protected_reasons = catalog.protection_reasons()
        if protected_reasons:
            return ReviewProposal(
                decision=Decision.PROTECTED_KEEP,
                result_id=analysis.result_id,
                protected_reasons=protected_reasons,
                review_reason="human_curation_protected",
            )

        if not catalog.catalogue_state_available:
            return ReviewProposal(
                decision=Decision.NONE,
                result_id=analysis.result_id,
                review_reason="catalogue_state_unavailable",
            )

        applied_receipt = catalog.verified_application_receipt()

        if not analysis.is_wildlife:
            return self._no_review_proposal(
                analysis.result_id,
                "not_wildlife",
                applied_receipt,
            )

        if not analysis.subject_detected:
            return self._no_review_proposal(
                analysis.result_id,
                "wildlife_subject_not_detected",
                applied_receipt,
            )

        if analysis.focus_score is None or analysis.focus_confidence is None:
            keyword = "AI Review|Uncertain"
            supersedes = self._supersession_for(
                analysis.result_id,
                keyword,
                applied_receipt,
            )
            return ReviewProposal(
                decision=Decision.MANUAL_REVIEW_UNCERTAIN,
                result_id=analysis.result_id,
                review_reason="focus_evidence_missing",
                keyword=keyword,
                suggested_color=self._suggested_color(
                    catalog.color_label,
                    supersedes,
                    applied_receipt,
                ),
                supersedes=supersedes,
            )

        if analysis.focus_confidence < self.min_focus_confidence:
            keyword = "AI Review|Uncertain"
            supersedes = self._supersession_for(
                analysis.result_id,
                keyword,
                applied_receipt,
            )
            return ReviewProposal(
                decision=Decision.MANUAL_REVIEW_UNCERTAIN,
                result_id=analysis.result_id,
                review_reason="focus_confidence_below_threshold",
                keyword=keyword,
                suggested_color=self._suggested_color(
                    catalog.color_label,
                    supersedes,
                    applied_receipt,
                ),
                supersedes=supersedes,
            )

        if analysis.focus_score < self.focus_threshold:
            keyword = "AI Review|Focus"
            supersedes = self._supersession_for(
                analysis.result_id,
                keyword,
                applied_receipt,
            )
            return ReviewProposal(
                decision=Decision.MANUAL_REVIEW_FOCUS,
                result_id=analysis.result_id,
                review_reason="subject_focus_below_threshold",
                keyword=keyword,
                suggested_color=self._suggested_color(
                    catalog.color_label,
                    supersedes,
                    applied_receipt,
                ),
                supersedes=supersedes,
            )

        return self._no_review_proposal(
            analysis.result_id,
            "subject_focus_at_or_above_threshold",
            applied_receipt,
        )

    @staticmethod
    def _supersession_for(
        result_id: str,
        target_keyword: str | None,
        applied_receipt: AppliedMetadataReceipt | None,
    ) -> AppliedMetadataSupersession | None:
        if applied_receipt is None:
            return None
        if (
            applied_receipt.result_id == result_id
            and applied_receipt.keyword == target_keyword
        ):
            return None
        return applied_receipt.to_supersession()

    @staticmethod
    def _suggested_color(
        current_color: str,
        supersedes: AppliedMetadataSupersession | None,
        applied_receipt: AppliedMetadataReceipt | None,
    ) -> str | None:
        if not current_color:
            return "Red"
        if (
            supersedes is not None
            and applied_receipt is not None
            and applied_receipt.color_label == "Red"
        ):
            return "Red"
        return None

    @classmethod
    def _no_review_proposal(
        cls,
        result_id: str,
        reason: str,
        applied_receipt: AppliedMetadataReceipt | None,
    ) -> ReviewProposal:
        supersedes = cls._supersession_for(result_id, None, applied_receipt)
        if supersedes is not None:
            return ReviewProposal(
                decision=Decision.CLEAR_AI_REVIEW,
                result_id=result_id,
                review_reason="analysis_no_longer_requires_review",
                supersedes=supersedes,
            )
        return ReviewProposal(
            decision=Decision.NONE,
            result_id=result_id,
            review_reason=reason,
        )


__all__ = [
    "AnalysisSignals",
    "AppliedMetadataReceipt",
    "AppliedMetadataSupersession",
    "CatalogSignals",
    "Decision",
    "ReviewPolicy",
    "ReviewProposal",
]
