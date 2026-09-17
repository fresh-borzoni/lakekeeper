"""End-to-end dataset tests against a running Lakekeeper and real object storage.

Driven through `pylakekeeper`, so these cover the client and the server together:
a wire-shape mismatch in the SDK fails here even though both sides pass their own
unit tests.

Import is the reason these need real storage. It registers what it finds by
*listing the prefix*, so proving it works means putting objects there and asking
the server to discover them -- something no in-memory backend can stand in for.
"""

import uuid

import conftest
import pytest

pylakekeeper = pytest.importorskip("pylakekeeper")

from pylakekeeper import CommitConflict, LakekeeperClient  # noqa: E402


@pytest.fixture
def client(warehouse: conftest.Warehouse) -> LakekeeperClient:
    # The catalog URL ends in /catalog/v1/; datasets live beside it under
    # /lakekeeper/v1/, so the client is given the server root.
    base = warehouse.server.catalog_url.split("/catalog/")[0]
    return LakekeeperClient(
        base_url=base,
        warehouse=str(warehouse.warehouse_id),
        token=warehouse.access_token,
    )


@pytest.fixture
def dataset(client: LakekeeperClient, namespace: conftest.Namespace):
    return client.create_dataset(namespace.name[0], f"ds-{uuid.uuid4().hex[:8]}")


def _file(key: str, location: str) -> dict:
    return {"logical-key": key, "physical-path": f"{location}/{key}"}


def test_commit_tag_and_read_back(dataset):
    location = dataset.load()["dataset"]["location"]

    first = dataset.commit(added=[_file("a.parquet", location), _file("b.parquet", location)])
    dataset.tag("v1")

    second = dataset.commit(
        added=[_file("c.parquet", location)],
        removed=["a.parquet"],
        parent_snapshot_id=first.snapshot_id,
    )
    assert second.parent_snapshot_id == first.snapshot_id

    # The tag keeps naming what it named, however far main moves on. This is the
    # whole reproducibility claim, so it is asserted against the server rather
    # than trusted.
    assert sorted(f.logical_key for f in dataset.files("v1")) == ["a.parquet", "b.parquet"]
    assert sorted(f.logical_key for f in dataset.files("main")) == ["b.parquet", "c.parquet"]


def test_stale_commit_conflicts_and_reports_the_head(dataset):
    location = dataset.load()["dataset"]["location"]
    first = dataset.commit(added=[_file("a.parquet", location)])
    dataset.commit(added=[_file("b.parquet", location)], parent_snapshot_id=first.snapshot_id)

    with pytest.raises(CommitConflict) as excinfo:
        # Still believes the branch is on the first snapshot.
        dataset.commit(added=[_file("c.parquet", location)], parent_snapshot_id=first.snapshot_id)

    # The conflict has to carry the head, or a writer cannot rebase without a
    # separate round trip.
    assert excinfo.value.current_snapshot_id is not None
    assert excinfo.value.current_snapshot_id != first.snapshot_id


def test_branch_diverges_without_touching_main(dataset):
    location = dataset.load()["dataset"]["location"]
    base = dataset.commit(added=[_file("shared.parquet", location)])

    dataset.create_ref("experiment", typ="branch", snapshot_id=base.snapshot_id)
    dataset.commit(
        added=[_file("only-on-experiment.parquet", location)],
        branch="experiment",
        parent_snapshot_id=base.snapshot_id,
    )

    assert [f.logical_key for f in dataset.files("main")] == ["shared.parquet"]
    assert sorted(f.logical_key for f in dataset.files("experiment")) == [
        "only-on-experiment.parquet",
        "shared.parquet",
    ]


def test_import_registers_objects_already_in_storage(dataset):
    """The M2 path: put objects in the bucket, let the server discover them.

    Nothing is copied -- import writes manifest rows for what it lists, so the
    physical paths it records are the objects that were already there.
    """
    location = dataset.load()["dataset"]["location"]
    fs = dataset.filesystem()

    keys = ["shard=00/a.parquet", "shard=00/b.parquet", "shard=01/c.parquet", "notes.txt"]
    for key in keys:
        with fs.open(f"{location}/{key}", "wb") as handle:
            handle.write(b"x" * 16)

    result = dataset.import_objects()
    assert result.imported == len(keys), result
    assert result.truncated is False

    listed = {f.logical_key: f for f in dataset.files("main")}
    assert sorted(listed) == sorted(keys)
    # Sizes come from the listing, and content type is inferred from the
    # extension -- neither was supplied by the caller.
    assert listed["notes.txt"].size == 16
    assert listed["shard=00/a.parquet"].content_type == "application/vnd.apache.parquet"
    assert listed["notes.txt"].content_type == "text/plain"
    # The physical path must be the object that was already there, not a copy.
    assert listed["notes.txt"].physical_path.endswith("notes.txt")


def test_import_honours_suffix_and_sub_prefix(dataset):
    location = dataset.load()["dataset"]["location"]
    fs = dataset.filesystem()
    for key in ["shard=00/a.parquet", "shard=00/b.txt", "shard=01/c.parquet"]:
        with fs.open(f"{location}/{key}", "wb") as handle:
            handle.write(b"x")

    result = dataset.import_objects(sub_prefix="shard=00", suffix=".parquet")
    assert result.imported == 1
    assert [f.logical_key for f in dataset.files("main")] == ["shard=00/a.parquet"]


def test_import_refuses_a_sub_prefix_that_escapes_the_dataset(dataset):
    from pylakekeeper import LakekeeperError

    with pytest.raises(LakekeeperError) as excinfo:
        dataset.import_objects(sub_prefix="../elsewhere")
    assert excinfo.value.status == 400


def test_download_reproduces_a_tagged_version(dataset, tmp_path):
    location = dataset.load()["dataset"]["location"]
    fs = dataset.filesystem()
    for key in ["train/0001.txt", "train/0002.txt"]:
        with fs.open(f"{location}/{key}", "wb") as handle:
            handle.write(key.encode())

    dataset.import_objects()
    dataset.tag("v1")

    written = dataset.download(str(tmp_path), ref="v1")
    assert written == 2
    assert (tmp_path / "train" / "0001.txt").read_bytes() == b"train/0001.txt"


def test_rescan_sync_detects_modifications_and_deletions(dataset):
    """The M2 re-scan: make the snapshot mirror the bucket, not just grow."""
    location = dataset.load()["dataset"]["location"]
    fs = dataset.filesystem()
    for key in ["keep.txt", "changed.txt", "gone.txt"]:
        with fs.open(f"{location}/{key}", "wb") as handle:
            handle.write(b"original")

    assert dataset.import_objects().imported == 3

    # Re-importing an untouched prefix must be a no-op, or a nightly sync would
    # rewrite the whole manifest every night.
    noop = dataset.import_objects(mode="sync")
    assert (noop.imported, noop.modified, noop.removed) == (0, 0, 0)

    with fs.open(f"{location}/changed.txt", "wb") as handle:
        handle.write(b"a different length entirely")
    fs.rm(f"{location}/gone.txt")
    with fs.open(f"{location}/brand-new.txt", "wb") as handle:
        handle.write(b"new")

    result = dataset.import_objects(mode="sync")
    assert (result.imported, result.modified, result.removed) == (1, 1, 1)
    assert sorted(f.logical_key for f in dataset.files("main")) == [
        "brand-new.txt",
        "changed.txt",
        "keep.txt",
    ]


def test_add_only_mode_never_removes(dataset):
    """The default must not drop entries just because an object vanished."""
    location = dataset.load()["dataset"]["location"]
    fs = dataset.filesystem()
    for key in ["a.txt", "b.txt"]:
        with fs.open(f"{location}/{key}", "wb") as handle:
            handle.write(b"x")
    dataset.import_objects()

    fs.rm(f"{location}/b.txt")
    assert dataset.import_objects().removed == 0
    assert sorted(f.logical_key for f in dataset.files("main")) == ["a.txt", "b.txt"]


def test_sync_refuses_a_truncated_scan(dataset):
    """Absence is only evidence of deletion if the whole prefix was seen."""
    from pylakekeeper import LakekeeperError

    location = dataset.load()["dataset"]["location"]
    fs = dataset.filesystem()
    for key in ["a.txt", "b.txt", "c.txt"]:
        with fs.open(f"{location}/{key}", "wb") as handle:
            handle.write(b"x")

    with pytest.raises(LakekeeperError) as excinfo:
        dataset.import_objects(mode="sync", max_files=2)
    assert excinfo.value.status == 400


def test_sync_on_a_fresh_dataset_is_the_first_import(dataset):
    """Sync must be usable as the first thing done to a dataset.

    A branch with no commits resolves to no files; reading that as an error would
    make sync unusable for the case it is most obviously wanted for.
    """
    location = dataset.load()["dataset"]["location"]
    fs = dataset.filesystem()
    for key in ["a.txt", "b.txt"]:
        with fs.open(f"{location}/{key}", "wb") as handle:
            handle.write(b"x")

    result = dataset.import_objects(mode="sync")
    assert (result.imported, result.modified, result.removed) == (2, 0, 0)
    assert sorted(f.logical_key for f in dataset.files("main")) == ["a.txt", "b.txt"]


def test_queued_import_runs_on_the_task_queue(dataset):
    """The scan moves off the request path; the snapshot appears when it lands."""
    import time

    location = dataset.load()["dataset"]["location"]
    fs = dataset.filesystem()
    for key in ["q/a.txt", "q/b.txt", "q/c.txt"]:
        with fs.open(f"{location}/{key}", "wb") as handle:
            handle.write(b"x")

    result = dataset.import_objects(queued=True)
    assert result.queued
    assert result.snapshot_id is None

    deadline = time.time() + 60
    while time.time() < deadline:
        keys = sorted(f.logical_key for f in dataset.files("main"))
        if keys:
            assert keys == ["q/a.txt", "q/b.txt", "q/c.txt"]
            return
        time.sleep(1)
    raise AssertionError("queued import did not land within 60s")
