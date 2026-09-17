"""Errors raised by the client.

Lakekeeper returns a JSON body shaped `{"error": {"message", "type", "code"}}`;
the type is what callers branch on, so it is preserved rather than flattened
into the message.
"""

from __future__ import annotations

from typing import Any, Optional


class LakekeeperError(Exception):
    """A request was refused by the server."""

    def __init__(self, status: int, payload: Any) -> None:
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        self.status = status
        self.type: Optional[str] = error.get("type")
        self.message: str = error.get("message") or str(payload)
        super().__init__(f"{status} {self.type or 'Error'}: {self.message}")


class NotFound(LakekeeperError):
    """The warehouse, namespace, dataset or ref does not exist."""


class CommitConflict(LakekeeperError):
    """Another writer moved the branch first.

    Carries the branch head the commit must be rebased onto. A conflict costs
    nothing physical -- no bytes were written -- so the caller can re-read, adjust
    its delta and retry.
    """

    #: Prefix the server uses to carry the head in the error stack. `ErrorModel`
    #: has no structured payload slot, so this is the agreed form.
    _HEAD_PREFIX = "current-snapshot-id: "

    def __init__(self, status: int, payload: Any) -> None:
        super().__init__(status, payload)
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        self.current_snapshot_id: Optional[str] = None
        for entry in error.get("stack") or []:
            if isinstance(entry, str) and entry.startswith(self._HEAD_PREFIX):
                self.current_snapshot_id = entry[len(self._HEAD_PREFIX) :].strip()
                break


def raise_for_status(status: int, payload: Any) -> None:
    if status < 400:
        return
    if status == 404:
        raise NotFound(status, payload)
    if status == 409:
        raise CommitConflict(status, payload)
    raise LakekeeperError(status, payload)
