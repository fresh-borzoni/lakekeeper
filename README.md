# pylakekeeper

Python client for Lakekeeper datasets — a versioned collection of files with
branchable, immutable snapshots.

```python
from pylakekeeper import LakekeeperClient

client = LakekeeperClient("https://lakekeeper.example", warehouse="<warehouse-id>", token=tok)

ds = client.create_dataset("ml", "images")

snapshot = ds.commit(added=[
    {"logical-key": "train/0001.jpg", "physical-path": "s3://bucket/raw/0001.jpg"},
])
ds.tag("v1")
```

`v1` now resolves to exactly those files, permanently, however far `main` moves on.

## Reading a version

```python
for f in ds.files("v1"):
    print(f.logical_key, f.size)

ds.download("./checkout", ref="v1")     # local copy, laid out by logical key
```

Both go through credentials the server vends scoped to the dataset's own prefix.

## Committing concurrently

A commit names the snapshot the caller believes the branch is on. If another
writer got there first the server refuses, and the conflict carries the head to
rebase onto — nothing physical was written, so retrying is cheap.

```python
from pylakekeeper import CommitConflict

while True:
    head = ds.refs()[0]["snapshot-id"]
    try:
        ds.commit(added=new_files, parent_snapshot_id=head)
        break
    except CommitConflict as conflict:
        head = conflict.current_snapshot_id   # re-read and retry
```

## Pagination

`files()` is a generator that follows the server's page token. A page may come
back **short or empty while files remain** — the server bounds each scan by key
range and applies removals and filters afterwards — so an empty page is not the
end. Only the absence of a token ends the walk; the generator handles this, but
code that pages by hand must too.

## Importing an existing prefix

The server lists the bucket and registers what it finds; nothing is copied and
the caller does not enumerate the prefix itself.

```python
ds.import_objects()                                  # register new objects
ds.import_objects(mode="sync")                       # also record changes and deletions
ds.import_objects(queued=True)                       # run the scan on the task queue
```

## Status

Covers create/list/load, commit, refs and tags, file listing, credential-backed
reads, local checkout, and imports. Not yet wired: diff between refs and per-file
annotations. `filesystem()` maps S3 credentials only.
