"""Dataset handle: commits, refs, and reading a ref's files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence
from urllib.parse import quote

from .client import LakekeeperClient, encode_namespace


@dataclass(frozen=True)
class DatasetFile:
    """One file as a snapshot's manifest records it."""

    logical_key: str
    physical_path: str
    size: Optional[int] = None
    etag: Optional[str] = None
    content_type: Optional[str] = None
    checksum: Optional[str] = None
    version_id: Optional[str] = None

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "DatasetFile":
        return cls(
            logical_key=payload["logical-key"],
            physical_path=payload["physical-path"],
            size=payload.get("size"),
            etag=payload.get("etag"),
            content_type=payload.get("content-type"),
            checksum=payload.get("checksum"),
            version_id=payload.get("version-id"),
        )


@dataclass(frozen=True)
class ImportResult:
    """What an import registered.

    ``truncated`` means the scan stopped at ``max_files`` and more objects remain
    under the prefix -- the snapshot is still valid, just partial.
    """

    snapshot_id: Optional[str]
    #: Set instead of ``snapshot_id`` when the scan was queued.
    task_id: Optional[str]
    imported: int
    modified: int
    removed: int
    truncated: bool

    @property
    def queued(self) -> bool:
        return self.task_id is not None


@dataclass(frozen=True)
class Snapshot:
    """The result of a commit."""

    snapshot_id: str
    parent_snapshot_id: Optional[str]
    location: str

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "Snapshot":
        return cls(
            snapshot_id=payload["snapshot-id"],
            parent_snapshot_id=payload.get("parent-snapshot-id"),
            location=payload.get("location", ""),
        )


class Dataset:
    """A versioned collection of files.

    Obtained from :meth:`LakekeeperClient.dataset`; holds no server state of its
    own, so it stays valid across commits.
    """

    def __init__(self, client: LakekeeperClient, namespace: str, name: str) -> None:
        self._client = client
        self.namespace = namespace
        self.name = name

    def __repr__(self) -> str:
        return f"Dataset({self.namespace}.{self.name})"

    # -- paths -------------------------------------------------------------

    def _path(self, suffix: str = "") -> str:
        return self._client.catalog_path(
            f"/namespaces/{encode_namespace(self.namespace)}"
            f"/datasets/{quote(self.name, safe='')}{suffix}"
        )

    def _ref_path(self, ref: str, suffix: str = "") -> str:
        return self._path(f"/refs/{quote(ref, safe='')}{suffix}")

    # -- metadata ----------------------------------------------------------

    def load(self) -> Dict[str, Any]:
        return self._client.request("GET", self._path())

    def refs(self) -> List[Dict[str, Any]]:
        return self._client.request("GET", self._path("/refs")).get("refs", [])

    # -- reading -----------------------------------------------------------

    def files(
        self, ref: str = "main", content_type: Optional[str] = None
    ) -> Iterator[DatasetFile]:
        """Yield every file the ref resolves to.

        Pages are followed until the server stops handing back a token. A page
        may legitimately come back **short or empty** while files remain -- the
        server bounds each scan by key range and applies removals and the
        content-type filter afterwards -- so emptiness must never be read as the
        end. Only the absence of a token ends the walk.
        """
        token = None
        while True:
            params: Dict[str, Any] = {}
            if content_type:
                params["contentType"] = content_type
            if token:
                params["pageToken"] = token
            page = self._client.request(
                "GET", self._ref_path(ref, "/files"), params=params or None
            )
            for entry in page.get("files", []):
                yield DatasetFile.from_json(entry)
            token = page.get("next-page-token")
            if not token:
                return

    def credentials(self) -> List[Dict[str, Any]]:
        """Storage credentials scoped to this dataset's prefix."""
        payload = self._client.request(
            "GET",
            self._path("/credentials"),
            headers={"x-iceberg-access-delegation": "vended-credentials"},
        )
        return payload.get("storage-credentials", [])

    # -- writing -----------------------------------------------------------

    def commit(
        self,
        added: Sequence[Dict[str, Any]] = (),
        removed: Sequence[str] = (),
        branch: str = "main",
        parent_snapshot_id: Optional[str] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> Snapshot:
        """Append a snapshot and move ``branch`` to it.

        ``parent_snapshot_id`` is the snapshot the caller believes the branch is
        on; the server refuses the commit if it has moved, raising
        :class:`CommitConflict` with the current head. Pass ``None`` only for the
        first commit on a fresh branch.
        """
        body: Dict[str, Any] = {
            "added": list(added),
            "removed": list(removed),
        }
        if parent_snapshot_id is not None:
            body["parent-snapshot-id"] = parent_snapshot_id
        if summary is not None:
            body["summary"] = summary
        payload = self._client.request(
            "POST", self._path(f"/branches/{quote(branch, safe='')}/commits"), json=body
        )
        return Snapshot.from_json(payload)

    def import_objects(
        self,
        branch: str = "main",
        sub_prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        max_files: Optional[int] = None,
        summary: Optional[Dict[str, Any]] = None,
        mode: str = "add-only",
        queued: bool = False,
    ) -> "ImportResult":
        """Register the objects already under the dataset's prefix.

        The server lists the prefix and writes manifest rows for what it finds;
        nothing is copied, and the caller does not enumerate the prefix itself.
        That is what makes this usable on prefixes too large to put in a request
        body.

        ``mode="sync"`` makes the snapshot mirror the prefix: files whose bytes
        changed are recorded as modified, and files that have disappeared are
        removed. The default only ever adds, because dropping manifest entries is
        not something to do by accident.

        ``queued=True`` hands the scan to the server's task queue and returns a
        task id instead of a snapshot: a prefix with millions of objects takes
        longer than a request should live. At most one import per dataset is
        active at a time, so firing this twice enqueues once.
        """
        body: Dict[str, Any] = {}
        if queued:
            body["queued"] = True
        if mode != "add-only":
            body["mode"] = mode
        if branch != "main":
            body["branch"] = branch
        if sub_prefix is not None:
            body["sub-prefix"] = sub_prefix
        if suffix is not None:
            body["suffix"] = suffix
        if max_files is not None:
            body["max-files"] = max_files
        if summary is not None:
            body["summary"] = summary
        payload = self._client.request("POST", self._path("/import"), json=body)
        return ImportResult(
            snapshot_id=payload.get("snapshot-id"),
            task_id=payload.get("task-id"),
            imported=payload["imported"],
            modified=payload["modified"],
            removed=payload["removed"],
            truncated=payload["truncated"],
        )

    def create_ref(
        self,
        name: str,
        typ: str = "tag",
        snapshot_id: Optional[str] = None,
        from_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a branch or tag, from a snapshot id or another ref."""
        if (snapshot_id is None) == (from_ref is None):
            raise ValueError("pass exactly one of snapshot_id or from_ref")
        source = (
            {"type": "snapshot", "snapshot-id": snapshot_id}
            if snapshot_id is not None
            else {"type": "ref", "name": from_ref}
        )
        return self._client.request(
            "POST", self._path("/refs"), json={"name": name, "typ": typ, "source": source}
        )

    def tag(self, name: str, ref: str = "main") -> Dict[str, Any]:
        """Pin the current head of ``ref`` under an immutable tag."""
        return self.create_ref(name, typ="tag", from_ref=ref)

    def delete_ref(self, name: str) -> None:
        self._client.request("DELETE", self._ref_path(name))

    # -- data access -------------------------------------------------------

    def download(self, dest: str, ref: str = "main", content_type: Optional[str] = None) -> int:
        """Copy every file the ref resolves to into ``dest``.

        Returns the number of files written. Layout under ``dest`` mirrors the
        logical keys, so a checkout is reproducible from the tag alone.
        """
        fs = self.filesystem()
        written = 0
        for entry in self.files(ref, content_type=content_type):
            target = os.path.join(dest, entry.logical_key)
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            fs.get_file(entry.physical_path, target)
            written += 1
        return written

    def filesystem(self):
        """An fsspec filesystem authenticated with this dataset's credentials.

        Only S3 is wired today; the credential shape the server returns is
        per-backend, so each one needs its own mapping rather than a generic
        passthrough.
        """
        creds = self.credentials()
        if not creds:
            raise RuntimeError(
                "server vended no credentials for this dataset -- the warehouse's "
                "storage profile may not support delegation"
            )
        config = creds[0].get("config", {})
        prefix = creds[0].get("prefix", "")

        if prefix.startswith("s3"):
            import s3fs

            return s3fs.S3FileSystem(
                key=config.get("s3.access-key-id"),
                secret=config.get("s3.secret-access-key"),
                token=config.get("s3.session-token"),
                client_kwargs={"endpoint_url": config.get("s3.endpoint")},
            )
        raise NotImplementedError(f"no filesystem mapping for credential prefix {prefix!r}")
