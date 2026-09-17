"""Unit tests for the client's request shaping and pagination contract.

These stub the transport rather than talking to a server: what is being checked
is that the client follows the protocol correctly, above all that it does not
stop early on a short page.
"""

from __future__ import annotations

import pytest

from pylakekeeper import CommitConflict, LakekeeperClient, NotFound
from pylakekeeper.client import encode_namespace


class StubClient(LakekeeperClient):
    """Records requests and replays canned responses."""

    def __init__(self, responses):
        super().__init__("http://stub", "wh-1")
        self._responses = list(responses)
        self.calls = []

    def request(self, method, path, *, json=None, params=None, headers=None):
        self.calls.append({"method": method, "path": path, "json": json, "params": params})
        return self._responses.pop(0)


def test_multi_level_namespace_is_one_encoded_segment():
    # "." separates levels but is itself legal in a name, so the wire form uses
    # the unit separator.
    assert encode_namespace("a.b") == "a%1Fb"
    assert encode_namespace("plain") == "plain"


def test_files_follows_pages_until_the_token_is_absent():
    client = StubClient(
        [
            {"files": [_file("a")], "next-page-token": "t1"},
            {"files": [_file("b")], "next-page-token": "t2"},
            {"files": [_file("c")]},
        ]
    )
    keys = [f.logical_key for f in client.dataset("ns", "ds").files()]
    assert keys == ["a", "b", "c"]
    assert [c["params"].get("pageToken") if c["params"] else None for c in client.calls] == [
        None,
        "t1",
        "t2",
    ]


def test_files_does_not_stop_on_an_empty_page_that_carries_a_token():
    # The server bounds each scan by key range and applies removals afterwards, so
    # a page can be empty while files remain. Treating empty as the end would
    # silently truncate the file list.
    client = StubClient(
        [
            {"files": [], "next-page-token": "t1"},
            {"files": [], "next-page-token": "t2"},
            {"files": [_file("survivor")]},
        ]
    )
    keys = [f.logical_key for f in client.dataset("ns", "ds").files()]
    assert keys == ["survivor"]
    assert len(client.calls) == 3


def test_commit_sends_the_expected_parent_and_omits_it_when_absent():
    client = StubClient([{"snapshot-id": "s2", "parent-snapshot-id": "s1", "location": "loc"}])
    snapshot = client.dataset("ns", "ds").commit(
        added=[{"logical-key": "a", "physical-path": "s3://b/a"}],
        parent_snapshot_id="s1",
    )
    assert snapshot.snapshot_id == "s2"
    assert client.calls[0]["json"]["parent-snapshot-id"] == "s1"

    client = StubClient([{"snapshot-id": "s1", "location": "loc"}])
    client.dataset("ns", "ds").commit(added=[])
    # A fresh branch has no parent; sending null would not mean the same thing.
    assert "parent-snapshot-id" not in client.calls[0]["json"]


def test_create_ref_requires_exactly_one_source():
    dataset = StubClient([]).dataset("ns", "ds")
    with pytest.raises(ValueError):
        dataset.create_ref("v1")
    with pytest.raises(ValueError):
        dataset.create_ref("v1", snapshot_id="s1", from_ref="main")


def test_conflict_carries_the_head_to_rebase_onto():
    from pylakekeeper.errors import raise_for_status

    with pytest.raises(CommitConflict) as excinfo:
        raise_for_status(
            409,
            {
                "error": {
                    "message": "branch moved",
                    "type": "DatasetCommitConflict",
                    "stack": ["Branch moved", "current-snapshot-id: s7"],
                }
            },
        )
    assert excinfo.value.current_snapshot_id == "s7"


def test_missing_dataset_raises_not_found():
    from pylakekeeper.errors import raise_for_status

    with pytest.raises(NotFound):
        raise_for_status(404, {"error": {"message": "gone", "type": "DatasetNotFound"}})


def test_ref_source_matches_the_server_discriminator():
    # The server's source union keys the ref variant on "name", not "ref"; a
    # mismatch is a 400 that no unit test on our side would otherwise catch.
    client = StubClient([{}, {}])
    dataset = client.dataset("ns", "ds")
    dataset.tag("v1", ref="main")
    assert client.calls[0]["json"]["source"] == {"type": "ref", "name": "main"}
    dataset.create_ref("b1", typ="branch", snapshot_id="s9")
    assert client.calls[1]["json"]["source"] == {"type": "snapshot", "snapshot-id": "s9"}


def _file(key):
    return {"logical-key": key, "physical-path": f"s3://bucket/{key}"}
