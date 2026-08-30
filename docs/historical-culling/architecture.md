# Historical Culling Architecture

## Trust boundaries

```text
Lightroom catalogue                 NAS headless worker
-------------------                 -------------------
exports protected-state manifest -> reads photo roots (read-only)
                                    runs categorisation/focus analysis
                                    stores versioned results in app state
polls result manifest            <- emits proposed metadata deltas
applies additive metadata
Lightroom remains XMP authority
```

The Lightroom catalogue is authoritative for human intent. The NAS is
authoritative only for reproducible analysis results. Original images and their
sidecars are never an output surface for the worker.

## Components

1. **Library scanner** recursively discovers all supported RAW and rendered
   formats and creates a source fingerprint. It does not use bare filename as
   identity.
2. **Analysis pipeline** reuses Kingfisher's bird segmentation and focus
   scoring, then adds broad categorisation and BioCLIP taxonomy in later slices.
3. **Review policy** combines analysis evidence with Lightroom protection
   signals. It emits a narrow proposal or no action.
4. **Result store** keeps immutable runs, application-ledger receipts, and
   idempotent current proposals in a dedicated writable state root outside the
   photo library. Receipts are trusted state and are never inferred from colour
   or keyword values in Lightroom. Ownership requires an exact asset ID, active
   applied result ID, metadata revision, keyword, and colour match. A new
   analysis result has a separate identity and may supersede that applied
   result while the receipt still proves ownership of the old metadata.
5. **Lightroom plugin** exports catalogue signals and applies deltas while
   Lightroom is open. It records applied result IDs for idempotency and undo.
6. **NAS scheduler** runs backfill and watch queues at bounded CPU priority. Its
   Kubernetes workload receives read-only photo mounts and a separate writable
   state volume.

## Existing Kingfisher paths excluded from historical mode

- `move_rejects_to_folder` must not be reachable.
- `metadata_writer.write_xmp_metadata` must not be called.
- AI quality must not be converted into a Lightroom star rating.
- Per-folder `.kingfisher` state is not the durable historical ledger because
  it resides inside the photo root and can be cleared by the desktop app.

## Failure behaviour

- Missing Lightroom state: analyse for search, but emit no actionable review
  metadata. External XMP additionally records the asset as protected. With
  catalogue state present, that state is authoritative and XMP presence alone
  is not protection.
- Missing or low-confidence subject/focus evidence: propose uncertainty only
  when a wildlife subject was detected.
- Interrupted scan: resume from the last committed item.
- Changed file/model/policy: create a new result and supersede the previous
  proposal without mutating its audit record. The application transaction
  revalidates the old receipt/revision, removes only its exact AI metadata,
  applies the new state, and records the new receipt atomically.
- Plugin unavailable: proposals remain pending; no sidecar fallback occurs.
