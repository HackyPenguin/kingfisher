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

Slice 2 uses SQLite with foreign keys, WAL mode, full synchronous durability,
and a busy timeout in the dedicated state root. The state root must be outside
the photo root and the worker does not write to the photo root:

- `libraries`: persistent configured identities independent of mount paths;
- `scan_runs`, `scan_observations`, and `scan_errors`: resumable discovery and
  diagnostics. Only a completed pass may mark an unseen asset missing;
- `assets`: root-relative path lookup plus current stat-signature fast gate;
- `asset_versions`: immutable full SHA-256 fingerprints. First observations and
  changed stat signatures are hashed after a before/after stability check;
- `analysis_runs`: immutable analyzer/model/policy configuration;
- `analysis_results`: model outputs for one asset/run;
- `review_proposals`: immutable allowed decisions with at most one current
  proposal per asset;
- `application_operations`: prepared/applied/superseded/manual-recovery state
  and exact plugin receipt evidence.

Writes occur in transactions. The manifest exporter sorts by library ID,
relative path, and result ID for deterministic output, writes through a
temporary file with `fsync`, and atomically replaces only a destination beneath
the state root. The closed manifest contains no delete, move, reject, rating, or
pick operation.

Normal reconciliation stats every eligible source and reuses a prior digest
only when size and nanosecond modification time match. A bounded full-hash audit
can catch rare timestamp-preserving replacements. Symlinks are never followed;
legacy per-folder `.kingfisher` state and unsupported sidecars are excluded.

SQLite and Lightroom cannot share a transaction manager. Later plugin work must
therefore use a recoverable prepared-operation protocol: prepare in SQLite,
apply one Lightroom catalogue transaction, then finalize the ledger with the
post-apply catalogue revision and exact metadata. An unresolved prepared
operation may only be finalized from exact proof or moved to manual recovery.

## Slice 3 BioCLIP analysis stream

Historical BioCLIP analysis is a separate runner and never invokes the legacy
folder pipeline. It resolves an active current `asset_version` from
`HistoricalStore`, reads it once through no-follow descriptors anchored at the
configured `source_root`, verifies the indexed full SHA-256, and decodes only
those verified in-memory bytes. The only writes are database-guarded immutable
`analysis_runs` and `analysis_results`, plus append-only retry diagnostics that
retain a closed error code rather than exception text.

The lazy adapter is pinned to `pybioclip==2.1.6` and
`open-clip-torch==3.3.0`, the latter providing the offline `local-dir:` loader,
and fails closed unless both installed packages match those versions. Both
versions participate in analysis identity and result provenance. Runtime model
paths come only from the
`KINGFISHER_BIOCLIP_MODEL_DIR` and `KINGFISHER_BIOCLIP_TAXONOMY_DIR`
configuration. The adapter verifies the local OpenCLIP config, safetensors,
TreeOfLife embeddings, and TreeOfLife labels against their configured SHA-256
values before and after classifier construction. It gives OpenCLIP a
`local-dir:` model and overrides pybioclip's taxonomy datafile lookup with
those verified local paths, so normal inference has no Hub lookup or download
path. Expected revisions and artifact digests remain in run identity and
result provenance; the result states that local verification is required
rather than treating a remote repository identifier as proof. Broad inference
uses one
`CustomLabelsClassifier` with a positive/`scene without ...` prompt pair per
closed category. Each positive score is normalized only against its paired
negative score, cancelling the classifier's shared softmax denominator and
producing independent multi-label evidence for `landscape`, `architecture`,
`human`, and `animal`. Every finite score is retained in that order.
`TreeOfLifeClassifier` with `Rank.SPECIES` runs only when the animal score meets
the configured threshold. Optional candidate species are applied through the
official taxonomy filter API. Model and taxonomy artifact configuration,
candidate set, threshold, top-k, output contract, and preprocessing provenance
all participate in the analysis-run identity.

The closed result document labels both broad and taxonomic predictions as
suggestions rather than ground truth. It has no proposal, rating, pick/reject,
delete, move, sidecar, or XMP operation. A successful version/run pair is
transactionally idempotent across concurrent writers. Source-version changes
discovered after inference are recorded as retryable failures; failures leave
the asset stale and never mutate earlier failure attempts.

## Testing strategy

The policy is covered with standard-library unit tests, including edited
protection, manual metadata protection, review decisions, label preservation,
input validation, deterministic serialization, Focus/Uncertain supersession,
receipt-owned clearing, and an assertion that forbidden source mutations cannot
appear. ML accuracy and Lightroom application are separate later test layers.
Scanner and SQLite integration tests now cover deterministic recursive
discovery, duplicate basenames, symlink exclusion, full source hashing,
idempotent scans, content changes, timestamp-preserving replacement audits,
interrupted restart, version-aware analysis staleness, immutable results, and
stable dry-run export. BioCLIP tests use fake providers, a fake decoder, and
fake import modules so they exercise gating, provenance identities, closed
deterministic outputs, retries, source/version validation, and source/sidecar
safety without importing a real model or downloading weights.
