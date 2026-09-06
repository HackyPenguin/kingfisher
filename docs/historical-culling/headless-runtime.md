# Historical Headless Runtime

The historical runtime is an analysis-only NAS process. It indexes a photo
library through `HistoricalIndexer`, stores durable state through
`HistoricalStore`, and analyzes indexed versions through
`HistoricalAnalysisRunner`. It does not import the legacy desktop pipeline and
cannot write photos, sidecars, Lightroom metadata, ratings, labels, review
proposals, or XMP.

## Storage boundary

Use three separate paths:

- source root: the photo library, mounted read-only;
- state root: a dedicated writable directory outside the source root;
- artifact root: a provisioned model tree, mounted read-only during inference.

The CLI rejects a state root beneath the source root. Source discovery skips
symlinks and `.kingfisher` directories. Analysis opens sources through
no-follow descriptors anchored at the source root and checks the indexed size
and SHA-256 before decoding. The default 512 MiB `--max-source-bytes` cap is
part of the analysis-run identity and is checked before allocating source
buffers. Larger assets receive an immutable `source_too_large` skip for that
version/run pair, so they cannot exhaust memory or block later work.

The state directory must be a real directory owned
by the runtime UID; SQLite database, WAL, SHM, and journal paths reject
symlinks, hardlinks, foreign ownership, and identity changes before use.

## Immutable model artifacts

Provisioning is the only command that uses the network. It downloads immutable
repository revisions into a sibling staging directory, verifies every byte,
fsyncs the staged tree, and atomically renames the complete tree into place.
An existing valid tree is reused; any existing invalid or non-pristine tree is
left untouched and causes the command to fail. The final publish uses an atomic
no-replace rename, so a concurrent destination is never overwritten.

```bash
python -m analyzer.kestrel_analyzer.historical_cli artifacts provision \
  --artifact-root /srv/kingfisher-artifacts

python -m analyzer.kestrel_analyzer.historical_cli artifacts verify \
  --artifact-root /srv/kingfisher-artifacts
```

The exact defaults are part of `ModelSpec`:

| Artifact | Revision | SHA-256 |
| --- | --- | --- |
| BioCLIP 2 config | `2957b322090f9cb17ae72c71981c7218a28d81e0` | `1bf947e96e943fe50efd5c3e26c37f843a2fa3c358967719a68c8a6d17ce68c8` |
| BioCLIP 2 weights | `2957b322090f9cb17ae72c71981c7218a28d81e0` | `b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef` |
| TreeOfLife species embeddings | `5f2dc493b3dc0e544438a04038ab15faa646b749` | `c72442de7b0cb7fcb55ab7ca08099d0f42fbd6769efe16ca64c1daa7a8b87db2` |
| TreeOfLife species labels | `5f2dc493b3dc0e544438a04038ab15faa646b749` | `4648928b006f85d83d28e5a27074ca9363465d82e778d708b369c5eaf54b8ef5` |

Inference verifies the staged manifest and all four files before constructing
the provider. The provider verifies them again before and after classifier
construction, uses OpenCLIP's `local-dir:` loader, and overrides taxonomy file
resolution with those local paths. There is no download fallback.

## Bounded commands and output

Every operational command requires an explicit source root, state root, and
library ID. Indexing requires `--max-items`; analysis requires `--limit`.
Values are restricted to `0..1000000`, and retries to `0..10` additional
attempts per asset. `--max-source-bytes` is positive and capped at 4 GiB.

```bash
# Index at most 5,000 source files.
python -m analyzer.kestrel_analyzer.historical_cli index \
  --source-root /photos --state-root /state --library-id family-library \
  --max-items 5000

# Index and analyze bounded work in one invocation.
python -m analyzer.kestrel_analyzer.historical_cli run \
  --source-root /photos --state-root /state --library-id family-library \
  --artifact-root /models --max-items 5000 --limit 100 --max-retries 2 \
  --max-source-bytes 536870912

# Inspect deterministic counters without loading a model.
python -m analyzer.kestrel_analyzer.historical_cli status \
  --source-root /photos --state-root /state --library-id family-library

# Opt-in real-model smoke against one already indexed path.
python -m analyzer.kestrel_analyzer.historical_cli smoke \
  --source-root /photos --state-root /state --library-id family-library \
  --artifact-root /models --relative-path 2026/example.jpg
```

The smoke command always performs decode and real-model prediction. If the
asset already has an immutable result for the active run, the stored result is
returned with `cached: true` only after the model path has been exercised.

Stdout contains exactly one canonical, key-sorted JSON document. Summaries have
stable schemas, sorted paths and error codes, explicit selected/deferred counts,
result IDs, cache counts, retry counts, and failure-attempt counts. They do not
contain exception text. A bounded incomplete pass reports `bounded`; exhausted
asset retries report `completed_with_failures` and exit 1.
Terminal exclusions are reported separately under `skips` and are not selected
again unless the source version or analysis configuration changes.

Pass the reported `--scan-id` to a later `index` or `run` invocation to resume
the same scan. Durable observations are retained, so repeated fixed-size calls
advance through pending paths while a final complete traversal reconciles paths
added or removed during the bounded scan.

SIGTERM and SIGINT request a cooperative stop. The current hash or inference is
allowed to finish, SQLite is closed normally, and an `interrupted` summary is
emitted with exit code 143. Index scans remain `running` and can be restarted
with their reported `scan_id`; interrupted analysis remains stale and is picked
up by a later run. Artifact hashing/downloads check termination between chunks,
discard staging, and never publish a partial tree. A shutdown never creates a
synthetic retry failure.

## Container

`Dockerfile.headless` uses a digest-pinned Python base, builds only for the
historical path, and installs the exact direct requirements in
`requirements-headless.txt`. The final image runs as UID/GID 65532 and does not
contain the legacy pipeline, metadata writer, or ratings module. Offline flags
for Hugging Face and Transformers are set in addition to the local-only loader.

```bash
docker build --platform linux/amd64 --target test \
  -f Dockerfile.headless -t kingfisher-headless:test .
docker build --platform linux/amd64 --target runtime \
  -f Dockerfile.headless -t kingfisher-headless:local .

docker run --rm --platform linux/amd64 --network none --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --mount type=bind,src=/srv/photos,dst=/photos,readonly \
  --mount type=bind,src=/srv/kingfisher-state,dst=/state \
  --mount type=bind,src=/srv/kingfisher-artifacts,dst=/models,readonly \
  kingfisher-headless:local run \
  --source-root /photos --state-root /state --library-id family-library \
  --artifact-root /models --max-items 5000 --limit 100 --max-retries 2
```

The writable state directory must be owned by UID/GID 65532. Cache state is
under `/state/cache`; the root filesystem and `/models` can remain read-only.
The GitHub workflow tests the `linux/amd64` image on pull requests and main,
then publishes only `sha-<40-character commit>` to GHCR on main. Deploy the
content-addressed digest recorded by the workflow, not a mutable convenience
tag.
