# Historical Culling Task List

## Slice 1: safety foundation

- [x] Audit destructive, XMP, resume, and focus paths.
- [x] Define PRD, architecture, system, and technical contracts.
- [x] Add policy tests first and observe the missing-module failure.
- [x] Implement the pure review policy.
- [x] Run unit tests and coverage.
- [x] Complete the first adversarial review and resolve all blocking findings.
- [x] Complete the post-fix adversarial review with no blocking findings.

## Slice 2: historical index and result store

- [x] Add recursive mixed-format discovery.
- [x] Add stable source fingerprints and version-aware staleness.
- [x] Add transactional SQLite result and audit stores.
- [x] Add checkpoint/resume and idempotency integration tests.
- [x] Add dry-run manifest export.

## Slice 3: analysis integration

- [x] Import the existing eye-focus prototype and its evaluator deliberately.
- [ ] Benchmark subject, head, and eye focus against labelled historical data.
- [ ] Add burst-relative ranking and uncertainty calibration.
- [x] Add analysis-only broad category and gated BioCLIP taxonomy suggestions.
- [ ] Prevent quality scores from becoming user star ratings.

## Slice 4: Lightroom integration

- [ ] Export protected catalogue signals from a Lightroom Classic plugin.
- [ ] Poll and apply idempotent additive review deltas.
- [ ] Preserve existing colours and all user metadata.
- [ ] Add guarded undo and application receipts.
- [ ] Verify on a copied catalogue and representative sample.

## Slice 5: NAS deployment

- [ ] Build and pin an x86-64 CPU image.
- [ ] Add a k3s/Argo CD service with read-only photo mounts.
- [ ] Add separate state/cache persistence and resource limits.
- [ ] Run a dry-run backfill, benchmark, soak, and recovery test.
- [ ] Enable bounded background reconciliation.
