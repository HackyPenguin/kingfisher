# Historical Culling PRD

## Problem

The existing library contains historical photographs that are already managed by
Lightroom. Kingfisher can analyse images, but its current filename-only resume
logic, complete XMP packet writer, and optional reject-file mover are unsafe for
an established catalogue. The historical workflow must help identify images for
human review without changing ownership decisions or source files.

## Goals

- Analyse existing and newly imported photographs with the same background
  pipeline.
- Categorise photographs as landscape, architecture, humans, animals, and more
  specific wildlife taxa when confidence supports it.
- Rank wildlife focus relative to comparable images and propose uncertain or
  poorly focused images for manual review.
- Treat photographs with Lightroom develop edits or other explicit human
  curation as protected.
- Apply only additive, reversible review metadata through Lightroom.
- Resume historical backfills safely when interrupted and reanalyse when a
  source, model, or policy changes.

## Non-goals

- Automatic deletion, rejection, movement, or quarantine of photographs.
- Changing star ratings, pick/reject flags, titles, captions, existing colours,
  or user-owned keywords.
- Allowing the NAS worker to write XMP sidecars beside originals.
- Declaring a photograph technically bad solely from a global sharpness score.

## Safety invariants

1. Photo roots are mounted read-only in the headless worker.
2. A result can propose only `manual_review_focus` or
   `manual_review_uncertain`, plus receipt-bound clearing of obsolete AI review
   metadata; it can never propose photo deletion, movement, or rejection.
3. A protected photograph cannot receive an automated review proposal.
4. Lightroom applies metadata deltas; the NAS never replaces an existing XMP
   packet.
5. The AI-owned keyword is additive. A red colour is optional and may be set
   only when no colour is already present.
6. Repeating the same result is idempotent. AI ownership is recognized only
   from an application-ledger receipt matching the asset, active applied result,
   Lightroom metadata revision, and exact current values. A newly analysed
   result may supersede that applied result; undo removes only metadata proven
   by the old receipt.
7. Changing review category is an atomic supersession: revalidate the old
   receipt/revision, remove only its exact AI keyword, apply the new AI metadata,
   and commit the new receipt together.

## Protection signals

A photograph is protected when Lightroom reports any of the following:

- develop edits;
- a non-zero manual rating;
- a manual pick/reject decision;
- an existing colour label;
- user keywords or another explicit curation marker.

When Lightroom catalogue evidence is unavailable, the policy always returns no
actionable metadata. An existing external XMP sidecar additionally records the
asset as protected. When a verified catalogue snapshot is available, the
snapshot is authoritative and sidecar presence alone does not protect every
asset. Lack of evidence never grants permission to mark, delete, or move a
photograph.

## Review metadata

- Hierarchical keyword: `AI Review|Focus` or `AI Review|Uncertain`.
- Optional colour suggestion: `Red`, only for an otherwise unlabelled photo.
- Audit fields: result ID, analyzer/model/policy versions, reason, confidence,
  source fingerprint, creation time supplied by the run, and application state.
- Supersession fields: exact old applied result, metadata revision, keyword, and
  optional AI-owned Red value copied from the verified receipt.

## Acceptance criteria for the first implementation slice

- Edited or manually curated photos always resolve to `protected_keep`.
- Unedited, confidently soft wildlife photos resolve to
  `manual_review_focus`.
- Low-confidence wildlife assessments resolve to `manual_review_uncertain`.
- Sharp photos and photos without an applicable wildlife subject receive no
  review action.
- Existing colour labels suppress the red-colour suggestion.
- AI-owned labels and keywords bypass protection only with an exact trusted
  ledger receipt matching the active applied result and current metadata
  revision. A new analysis result may supersede unchanged AI-owned metadata;
  stale, mismatched, or missing receipts protect current values as human
  metadata.
- The serialized proposal contains no delete, move, reject, or rating action.
- Focus/uncertain transitions leave exactly one AI review keyword, and a new
  no-review result can clear only receipt-owned AI metadata.
- Unit tests cover every rule and run without loading ML models.
