import math
import unittest

from analyzer.kestrel_analyzer.review_policy import (
    AnalysisSignals,
    AppliedMetadataReceipt,
    AppliedMetadataSupersession,
    CatalogSignals,
    Decision,
    ReviewPolicy,
    ReviewProposal,
)


class ReviewPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = ReviewPolicy(focus_threshold=0.45, min_focus_confidence=0.65)

    @staticmethod
    def catalog(**overrides):
        overrides.setdefault("catalogue_state_available", True)
        return CatalogSignals(**overrides)

    @staticmethod
    def analysis(**overrides):
        values = {
            "result_id": "result-123",
            "is_wildlife": True,
            "subject_detected": True,
            "focus_score": 0.2,
            "focus_confidence": 0.9,
        }
        values.update(overrides)
        return AnalysisSignals(**values)

    def test_develop_edits_are_protected(self):
        proposal = self.policy.evaluate(
            self.catalog(has_develop_edits=True),
            self.analysis(focus_score=0.1, focus_confidence=0.99),
        )

        self.assertEqual(Decision.PROTECTED_KEEP, proposal.decision)
        self.assertEqual(("develop_edits",), proposal.protected_reasons)
        self.assertIsNone(proposal.keyword)
        self.assertIsNone(proposal.suggested_color)

    def test_every_manual_curation_signal_is_protected(self):
        cases = {
            "manual_rating": self.catalog(manual_rating=3),
            "manual_pick": self.catalog(pick_state="picked"),
            "manual_reject": self.catalog(pick_state="rejected"),
            "user_color_label": self.catalog(color_label="Blue"),
            "user_keywords": self.catalog(keywords=("Portfolio",)),
            "external_xmp": self.catalog(
                catalogue_state_available=False,
                external_xmp_present=True,
            ),
        }

        analysis = self.analysis(focus_score=0.1, focus_confidence=0.99)
        for reason, catalog in cases.items():
            with self.subTest(reason=reason):
                proposal = self.policy.evaluate(catalog, analysis)
                self.assertEqual(Decision.PROTECTED_KEEP, proposal.decision)
                self.assertIn(reason, proposal.protected_reasons)

    def test_multiple_protection_reasons_are_stable_and_complete(self):
        proposal = self.policy.evaluate(
            self.catalog(
                has_develop_edits=True,
                manual_rating=5,
                pick_state="picked",
                color_label="Green",
                keywords=("Portfolio", "Birds"),
                catalogue_state_available=False,
                external_xmp_present=True,
            ),
            self.analysis(is_wildlife=False, subject_detected=False),
        )

        self.assertEqual(
            (
                "develop_edits",
                "manual_rating",
                "manual_pick",
                "user_color_label",
                "user_keywords",
                "external_xmp",
            ),
            proposal.protected_reasons,
        )

    def test_confident_soft_wildlife_is_proposed_for_focus_review(self):
        proposal = self.policy.evaluate(
            self.catalog(),
            self.analysis(),
        )

        self.assertEqual(Decision.MANUAL_REVIEW_FOCUS, proposal.decision)
        self.assertEqual("subject_focus_below_threshold", proposal.review_reason)
        self.assertEqual("AI Review|Focus", proposal.keyword)
        self.assertEqual("Red", proposal.suggested_color)

    def test_uncertain_focus_is_proposed_for_uncertain_review(self):
        for score, confidence in ((None, None), (0.2, 0.4)):
            with self.subTest(score=score, confidence=confidence):
                proposal = self.policy.evaluate(
                    self.catalog(),
                    self.analysis(focus_score=score, focus_confidence=confidence),
                )
                self.assertEqual(Decision.MANUAL_REVIEW_UNCERTAIN, proposal.decision)
                self.assertEqual("AI Review|Uncertain", proposal.keyword)
                self.assertEqual("Red", proposal.suggested_color)

    def test_external_xmp_is_only_a_fallback_without_catalogue_state(self):
        analysis = self.analysis()

        available = self.policy.evaluate(
            self.catalog(catalogue_state_available=True, external_xmp_present=True),
            analysis,
        )
        unavailable = self.policy.evaluate(
            self.catalog(catalogue_state_available=False, external_xmp_present=True),
            analysis,
        )

        self.assertEqual(Decision.MANUAL_REVIEW_FOCUS, available.decision)
        self.assertEqual(Decision.PROTECTED_KEEP, unavailable.decision)
        self.assertEqual(("external_xmp",), unavailable.protected_reasons)

    def test_missing_catalogue_state_never_emits_actionable_metadata(self):
        proposal = self.policy.evaluate(
            self.catalog(catalogue_state_available=False, external_xmp_present=False),
            self.analysis(),
        )

        self.assertEqual(Decision.NONE, proposal.decision)
        self.assertEqual("catalogue_state_unavailable", proposal.review_reason)
        self.assertIsNone(proposal.keyword)
        self.assertIsNone(proposal.suggested_color)

    def test_exact_application_receipt_proves_ai_owned_metadata(self):
        receipt = AppliedMetadataReceipt(
            asset_id="library:2020/IMG_001.CR3",
            result_id="result-123",
            metadata_revision="revision-5",
            keyword="AI Review|Focus",
            color_label="Red",
        )
        proposal = self.policy.evaluate(
            self.catalog(
                asset_id="library:2020/IMG_001.CR3",
                active_application_result_id="result-123",
                metadata_revision="revision-5",
                color_label="Red",
                keywords=("AI Review|Focus",),
                application_receipt=receipt,
            ),
            self.analysis(),
        )

        self.assertEqual(Decision.MANUAL_REVIEW_FOCUS, proposal.decision)
        self.assertIsNone(proposal.suggested_color)

    def test_stale_or_mismatched_receipt_cannot_claim_manual_red_label(self):
        receipt = AppliedMetadataReceipt(
            asset_id="library:other.CR3",
            result_id="result-123",
            metadata_revision="revision-5",
            keyword="AI Review|Focus",
            color_label="Red",
        )
        proposal = self.policy.evaluate(
            self.catalog(
                asset_id="library:IMG_001.CR3",
                active_application_result_id="result-123",
                metadata_revision="revision-5",
                color_label="Red",
                application_receipt=receipt,
            ),
            self.analysis(),
        )

        self.assertEqual(Decision.PROTECTED_KEEP, proposal.decision)
        self.assertEqual(("user_color_label",), proposal.protected_reasons)

    def test_post_application_re_evaluation_is_stable_until_user_curates(self):
        asset_id = "library:2020/IMG_001.CR3"
        receipt = AppliedMetadataReceipt(
            asset_id=asset_id,
            result_id="result-123",
            metadata_revision="revision-5",
            keyword="AI Review|Focus",
            color_label="Red",
        )
        analysis = self.analysis()
        applied = self.catalog(
            asset_id=asset_id,
            active_application_result_id="result-123",
            metadata_revision="revision-5",
            color_label="Red",
            keywords=("AI Review|Focus",),
            application_receipt=receipt,
        )
        user_curated = self.catalog(
            asset_id=asset_id,
            active_application_result_id="result-123",
            metadata_revision="revision-5",
            color_label="Red",
            keywords=("AI Review|Focus", "Portfolio"),
            application_receipt=receipt,
        )

        stable = self.policy.evaluate(applied, analysis)
        protected = self.policy.evaluate(user_curated, analysis)

        self.assertEqual(Decision.MANUAL_REVIEW_FOCUS, stable.decision)
        self.assertEqual(Decision.PROTECTED_KEEP, protected.decision)
        self.assertEqual(("user_keywords",), protected.protected_reasons)

    def test_receipt_must_match_active_application_and_metadata_revision(self):
        asset_id = "library:2020/IMG_001.CR3"
        receipt = AppliedMetadataReceipt(
            asset_id=asset_id,
            result_id="result-previous",
            metadata_revision="revision-previous",
            keyword="AI Review|Focus",
            color_label="Red",
        )
        current_values = {
            "asset_id": asset_id,
            "color_label": "Red",
            "keywords": ("AI Review|Focus",),
            "application_receipt": receipt,
        }

        stale_application_receipt = self.policy.evaluate(
            self.catalog(
                active_application_result_id="result-other",
                metadata_revision="revision-previous",
                **current_values,
            ),
            self.analysis(result_id="result-current"),
        )
        changed_metadata = self.policy.evaluate(
            self.catalog(
                active_application_result_id="result-previous",
                metadata_revision="revision-current",
                **current_values,
            ),
            self.analysis(result_id="result-previous"),
        )

        for proposal in (stale_application_receipt, changed_metadata):
            with self.subTest(proposal=proposal):
                self.assertEqual(Decision.PROTECTED_KEEP, proposal.decision)
                self.assertEqual(
                    ("user_color_label", "user_keywords"),
                    proposal.protected_reasons,
                )

    def test_new_analysis_can_supersede_unchanged_ai_owned_metadata(self):
        asset_id = "library:2020/IMG_001.CR3"
        receipt = AppliedMetadataReceipt(
            asset_id=asset_id,
            result_id="result-old",
            metadata_revision="revision-5",
            keyword="AI Review|Focus",
            color_label="Red",
        )
        catalog = self.catalog(
            asset_id=asset_id,
            active_application_result_id="result-old",
            metadata_revision="revision-5",
            color_label="Red",
            keywords=("AI Review|Focus",),
            application_receipt=receipt,
        )

        proposal = self.policy.evaluate(
            catalog,
            self.analysis(result_id="result-new"),
        )

        self.assertEqual(Decision.MANUAL_REVIEW_FOCUS, proposal.decision)
        self.assertEqual("result-new", proposal.result_id)
        self.assertEqual("Red", proposal.suggested_color)
        self.assertEqual("result-old", proposal.supersedes.applied_result_id)

    def test_reanalysis_still_protects_changed_revision_or_user_metadata(self):
        asset_id = "library:2020/IMG_001.CR3"
        receipt = AppliedMetadataReceipt(
            asset_id=asset_id,
            result_id="result-old",
            metadata_revision="revision-old",
            keyword="AI Review|Focus",
            color_label="Red",
        )
        common = {
            "asset_id": asset_id,
            "active_application_result_id": "result-old",
            "color_label": "Red",
            "keywords": ("AI Review|Focus",),
            "application_receipt": receipt,
        }
        changed_revision = self.catalog(metadata_revision="revision-new", **common)
        user_keyword = self.catalog(
            metadata_revision="revision-old",
            **{**common, "keywords": ("AI Review|Focus", "Portfolio")},
        )

        for catalog in (changed_revision, user_keyword):
            with self.subTest(catalog=catalog):
                proposal = self.policy.evaluate(
                    catalog,
                    self.analysis(result_id="result-new"),
                )
                self.assertEqual(Decision.PROTECTED_KEEP, proposal.decision)

    def test_focus_and_uncertain_transitions_replace_ai_metadata_atomically(self):
        asset_id = "library:2020/IMG_001.CR3"
        transitions = (
            (
                "AI Review|Focus",
                "AI Review|Uncertain",
                self.analysis(
                    result_id="result-new",
                    focus_score=None,
                    focus_confidence=None,
                ),
                Decision.MANUAL_REVIEW_UNCERTAIN,
            ),
            (
                "AI Review|Uncertain",
                "AI Review|Focus",
                self.analysis(result_id="result-new"),
                Decision.MANUAL_REVIEW_FOCUS,
            ),
        )

        for old_keyword, new_keyword, new_analysis, expected_decision in transitions:
            with self.subTest(old_keyword=old_keyword, new_keyword=new_keyword):
                old_receipt = AppliedMetadataReceipt(
                    asset_id=asset_id,
                    result_id="result-old",
                    metadata_revision="revision-old",
                    keyword=old_keyword,
                    color_label="Red",
                )
                current = self.catalog(
                    asset_id=asset_id,
                    active_application_result_id="result-old",
                    metadata_revision="revision-old",
                    color_label="Red",
                    keywords=(old_keyword,),
                    application_receipt=old_receipt,
                )

                replacement = self.policy.evaluate(current, new_analysis)

                self.assertEqual(expected_decision, replacement.decision)
                self.assertEqual(new_keyword, replacement.keyword)
                self.assertEqual("Red", replacement.suggested_color)
                self.assertEqual(
                    AppliedMetadataSupersession(
                        asset_id=asset_id,
                        applied_result_id="result-old",
                        metadata_revision="revision-old",
                        keyword=old_keyword,
                        color_label="Red",
                    ),
                    replacement.supersedes,
                )

                new_receipt = AppliedMetadataReceipt(
                    asset_id=asset_id,
                    result_id="result-new",
                    metadata_revision="revision-new",
                    keyword=new_keyword,
                    color_label="Red",
                )
                applied = self.catalog(
                    asset_id=asset_id,
                    active_application_result_id="result-new",
                    metadata_revision="revision-new",
                    color_label="Red",
                    keywords=(new_keyword,),
                    application_receipt=new_receipt,
                )
                stable = self.policy.evaluate(applied, new_analysis)

                self.assertEqual(expected_decision, stable.decision)
                self.assertEqual(new_keyword, stable.keyword)
                self.assertIsNone(stable.supersedes)

    def test_new_no_review_result_clears_only_receipt_owned_metadata(self):
        asset_id = "library:2020/IMG_001.CR3"
        receipt = AppliedMetadataReceipt(
            asset_id=asset_id,
            result_id="result-old",
            metadata_revision="revision-old",
            keyword="AI Review|Focus",
            color_label="Red",
        )
        current = self.catalog(
            asset_id=asset_id,
            active_application_result_id="result-old",
            metadata_revision="revision-old",
            color_label="Red",
            keywords=("AI Review|Focus",),
            application_receipt=receipt,
        )

        proposal = self.policy.evaluate(
            current,
            self.analysis(result_id="result-new", focus_score=0.9),
        )

        self.assertEqual(Decision.CLEAR_AI_REVIEW, proposal.decision)
        self.assertEqual("analysis_no_longer_requires_review", proposal.review_reason)
        self.assertEqual("result-old", proposal.supersedes.applied_result_id)
        self.assertIsNone(proposal.keyword)
        self.assertIsNone(proposal.suggested_color)

    def test_sharp_wildlife_and_non_applicable_photos_receive_no_action(self):
        cases = (
            self.analysis(focus_score=0.8),
            self.analysis(is_wildlife=False, subject_detected=False),
            self.analysis(is_wildlife=True, subject_detected=False),
        )

        for analysis in cases:
            with self.subTest(analysis=analysis):
                proposal = self.policy.evaluate(self.catalog(), analysis)
                self.assertEqual(Decision.NONE, proposal.decision)
                self.assertIsNone(proposal.keyword)
                self.assertIsNone(proposal.suggested_color)

    def test_serialized_proposal_has_only_non_destructive_fields(self):
        proposal = self.policy.evaluate(
            self.catalog(),
            self.analysis(),
        )

        serialized = proposal.to_dict()
        self.assertEqual(
            {
                "decision": "manual_review_focus",
                "result_id": "result-123",
                "protected_reasons": [],
                "review_reason": "subject_focus_below_threshold",
                "keyword": "AI Review|Focus",
                "suggested_color": "Red",
                "supersedes": None,
            },
            serialized,
        )
        forbidden = {"delete", "move", "reject", "rating", "pick_flag", "source_path"}
        self.assertTrue(forbidden.isdisjoint(serialized))

    def test_proposal_schema_rejects_values_outside_the_metadata_allowlist(self):
        invalid_factories = (
            lambda: ReviewProposal(
                Decision.MANUAL_REVIEW_FOCUS,
                result_id="result",
                review_reason="subject_focus_below_threshold",
                keyword="Any keyword",
            ),
            lambda: ReviewProposal(
                Decision.MANUAL_REVIEW_FOCUS,
                result_id="result",
                review_reason="subject_focus_below_threshold",
                keyword="AI Review|Focus",
                suggested_color="Blue",
            ),
            lambda: ReviewProposal(
                Decision.MANUAL_REVIEW_FOCUS,
                result_id="result",
                review_reason="focus_evidence_missing",
                keyword="AI Review|Focus",
            ),
            lambda: ReviewProposal(
                Decision.NONE,
                result_id="result",
                protected_reasons=("develop_edits",),
                review_reason="not_wildlife",
            ),
            lambda: ReviewProposal(
                Decision.PROTECTED_KEEP,
                result_id="result",
                protected_reasons=("develop_edits",),
                review_reason="wrong_reason",
            ),
            lambda: ReviewProposal(
                Decision.CLEAR_AI_REVIEW,
                result_id="result",
                review_reason="analysis_no_longer_requires_review",
            ),
            lambda: ReviewProposal(
                Decision.NONE,
                result_id="result",
                review_reason="not_wildlife",
                supersedes=AppliedMetadataSupersession(
                    asset_id="asset",
                    applied_result_id="old-result",
                    metadata_revision="revision",
                    keyword="AI Review|Focus",
                    color_label="Red",
                ),
            ),
        )

        for factory in invalid_factories:
            with self.subTest(factory=factory):
                with self.assertRaises((TypeError, ValueError)):
                    factory()

    def test_invalid_catalog_and_analysis_inputs_are_rejected(self):
        invalid_factories = (
            lambda: CatalogSignals(),
            lambda: self.catalog(manual_rating=-1),
            lambda: self.catalog(manual_rating=6),
            lambda: self.catalog(manual_rating=True),
            lambda: self.catalog(pick_state="maybe"),
            lambda: self.catalog(keywords="Portfolio"),
            lambda: self.catalog(keywords=(1,)),
            lambda: self.catalog(
                application_receipt=AppliedMetadataReceipt(
                    asset_id="asset",
                    result_id="result",
                    metadata_revision="revision",
                    keyword="AI Review|Focus",
                )
            ),
            lambda: self.catalog(
                asset_id="asset",
                metadata_revision="revision",
                application_receipt=AppliedMetadataReceipt(
                    asset_id="asset",
                    result_id="result",
                    metadata_revision="revision",
                    keyword="AI Review|Focus",
                ),
            ),
            lambda: AppliedMetadataReceipt(
                asset_id="",
                result_id="result",
                metadata_revision="revision",
                keyword="AI Review|Focus",
            ),
            lambda: AppliedMetadataReceipt(
                asset_id="asset",
                result_id="",
                metadata_revision="revision",
                keyword="AI Review|Focus",
            ),
            lambda: AppliedMetadataReceipt(
                asset_id="asset",
                result_id="result",
                metadata_revision="",
                keyword="AI Review|Focus",
            ),
            lambda: AppliedMetadataReceipt(
                asset_id="asset",
                result_id="result",
                metadata_revision="revision",
                keyword="Portfolio",
            ),
            lambda: AppliedMetadataReceipt(
                asset_id="asset",
                result_id="result",
                metadata_revision="revision",
                keyword="AI Review|Focus",
                color_label="Blue",
            ),
            lambda: self.analysis(
                is_wildlife=True,
                subject_detected=True,
                focus_score=-0.1,
                focus_confidence=0.5,
            ),
            lambda: self.analysis(
                is_wildlife=True,
                subject_detected=True,
                focus_score=0.5,
                focus_confidence=1.1,
            ),
            lambda: self.analysis(
                is_wildlife=True,
                subject_detected=True,
                focus_score=math.nan,
                focus_confidence=0.5,
            ),
            lambda: AnalysisSignals(
                result_id="",
                is_wildlife=True,
                subject_detected=True,
            ),
            lambda: ReviewPolicy(focus_threshold=2.0),
            lambda: self.policy.evaluate(self.catalog(), object()),
        )

        for factory in invalid_factories:
            with self.subTest(factory=factory):
                with self.assertRaises((TypeError, ValueError)):
                    factory()


if __name__ == "__main__":
    unittest.main()
