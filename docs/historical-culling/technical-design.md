# Historical Culling Technical Design

## Review policy API

The first slice introduces a pure-Python policy module with no ML or filesystem
dependencies:

```python
proposal = ReviewPolicy().evaluate(catalog_signals, analysis_signals)
```

`CatalogSignals` requires explicit Lightroom catalogue availability and carries
Lightroom-owned intent plus an opaque metadata revision. An omitted snapshot
cannot silently default to verified. `AppliedMetadataReceipt` carries trusted
ledger evidence for a previous application, bound to asset ID, active result
ID, metadata revision, and exact current metadata. `AnalysisSignals` requires a
result ID and carries wildlife applicability, focus score, and confidence.
`ReviewProposal` has a stable dictionary representation suitable for a later
manifest schema.

The active applied result and new analysis result are deliberately distinct.
The former proves ownership of existing AI metadata; the latter is copied into
the new proposal so a changed model or policy can supersede the old result.
`AppliedMetadataSupersession` copies only the exact old receipt evidence needed
for a guarded Lightroom transaction. It never identifies a source-file
operation.

Allowed decisions are:

- `none`
- `protected_keep`
- `clear_ai_review`
- `manual_review_focus`
- `manual_review_uncertain`

Allowed metadata recommendations are closed, decision-specific keyword/reason
pairs and optional `Red`. Constructors reject arbitrary keywords, colours,
reasons, and protection combinations. There is deliberately no field for
deletion, movement, rejection, pick flag, or rating.

## Validation

- Scores and confidence must be finite values within `[0, 1]`.
- Ratings must be integers within `[0, 5]`.
- Pick state is one of `unflagged`, `picked`, or `rejected`.
- Existing keywords are normalized deterministically for policy evaluation but
  are never returned as replacement values.
- External XMP is a protection fallback only when verified catalogue state is
  unavailable.
- Unavailable catalogue state always produces a non-actionable no-op, regardless
  of focus evidence.
- AI metadata is excluded from human-curation protection only when a trusted
  receipt matches the asset, active applied result, metadata revision, and exact
  current values. A different new analysis result does not invalidate that
  ownership proof.
- A review-category transition must remove the old receipt-owned keyword before
  applying the replacement; a `clear_ai_review` outcome removes only verified
  obsolete AI metadata.

## Next storage slice

Use SQLite with foreign keys and WAL mode in the dedicated state root:

- `assets`: current canonical identity and fingerprint;
- `analysis_runs`: immutable analyzer/model/policy configuration;
- `analysis_results`: model outputs for one asset/run;
- `review_proposals`: current allowed decision and reasons;
- `application_ledger`: proposed/applied/superseded state and plugin receipt.

Writes occur in transactions. The manifest exporter sorts by library ID,
relative path, and result ID for deterministic output.

## Testing strategy

The policy is covered with standard-library unit tests, including edited
protection, manual metadata protection, review decisions, label preservation,
input validation, deterministic serialization, Focus/Uncertain supersession,
receipt-owned clearing, and an assertion that forbidden source mutations cannot
appear. ML accuracy, scanner integration, SQLite resume, and Lightroom
application are separate later test layers.
