# Historical Culling System Design

## Processing modes

### Backfill

The scanner walks a configured library root deterministically, enqueues unseen
or stale fingerprints, and checkpoints after bounded batches. Initial rollout
is dry-run only, followed by a representative sample spanning cameras, years,
RAW formats, JPEGs, scans, high ISO, and intentional motion blur.

### Background imports

After backfill, the same queue periodically discovers new or changed assets.
Filesystem notifications may reduce latency but are not the correctness
mechanism; deterministic reconciliation remains authoritative.

## Asset identity

The durable identity record contains:

- library ID and canonical relative path;
- source size and nanosecond modification time;
- a content fingerprint;
- analyzer, model, and policy versions.

A rename, replacement, or model change therefore cannot be silently skipped as
it is in the current bare-filename resume logic.

## Decision sequence

1. Reject malformed or out-of-range analysis inputs.
2. If any catalogue protection signal is present, return `protected_keep` with
   explicit reasons.
3. If the item is outside the wildlife-focus scope, return no action.
4. If focus evidence is absent or below the confidence threshold, propose
   `manual_review_uncertain`.
5. If focus is confidently below the configured threshold, propose
   `manual_review_focus`.
6. Otherwise return no action.

The first policy uses absolute thresholds only as a testable foundation.
Burst-relative ranking will replace the final focus decision after it is
calibrated against representative photographs.

## Lightroom delta

Each proposal carries an AI-owned keyword and an optional colour suggestion.
The plugin must:

- re-check protection immediately before applying;
- add the keyword without replacing user keywords;
- set Red only when the current label is empty;
- leave ratings, flags, develop settings, captions, titles, and user labels
  untouched;
- record the result ID once applied;
- remove AI metadata on undo only if it still exactly matches what was applied.

The plugin supplies any prior application receipt from the trusted ledger and
an opaque Lightroom metadata revision that changes whenever relevant catalogue
metadata changes. A receipt is valid only when its asset ID, active result ID,
metadata revision, and exact applied keyword/colour match the current catalogue
state. Here, active result means the result currently applied in Lightroom, not
the new result being evaluated. A new result may supersede unchanged AI-owned
metadata. A stale, missing, or mismatched receipt never claims ownership of a
user's Red label or keyword.

When a new result supersedes an applied AI result, the proposal carries the old
receipt's exact asset, applied result, metadata revision, keyword, and optional
Red ownership. The plugin must apply one transaction: revalidate that evidence,
remove only the old AI keyword, retain or clear only a receipt-owned Red label,
apply the new AI keyword/colour if any, and record the new receipt/revision. A
Focus-to-Uncertain transition therefore leaves exactly one AI review keyword.

## NAS placement

The persistent deployment belongs under `nas-config/k8s/services/<service>` and
Argo CD. Photo mounts are read-only. The result store and bounded cache use a
separate persistent volume. No HTTP interface is required for the first worker;
if one is later introduced, it must use a reviewed Traefik hostname route.
