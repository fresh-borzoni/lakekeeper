"""HTTP client and entry point."""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional
from urllib.parse import quote

import requests

from .errors import raise_for_status

#: Multi-level namespaces travel as one path segment, separated by this unit
#: separator. That is the Iceberg REST convention Lakekeeper follows; a literal
#: "." is a legal namespace character, so it cannot be the separator.
NAMESPACE_SEPARATOR = "\x1f"


def encode_namespace(namespace: str) -> str:
    """Encode a dotted namespace for use as a single URL path segment."""
    return quote(namespace.replace(".", NAMESPACE_SEPARATOR), safe="")


class LakekeeperClient:
    """Connection to a Lakekeeper server.

    ``warehouse`` is the warehouse id used as the URL prefix for catalog routes.
    """

    def __init__(
        self,
        base_url: str,
        warehouse: str,
        token: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.warehouse = warehouse
        self.timeout = timeout
        self._session = session or requests.Session()
        if token:
            self._session.headers["authorization"] = f"Bearer {token}"

    # -- low-level ---------------------------------------------------------

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        response = self._session.request(
            method,
            f"{self.base_url}{path}",
            json=json,
            params=params,
            headers=headers,
            timeout=self.timeout,
        )
        if response.status_code == 204 or not response.content:
            raise_for_status(response.status_code, {})
            return None
        try:
            payload = response.json()
        except ValueError:
            payload = {"error": {"message": response.text}}
        raise_for_status(response.status_code, payload)
        return payload

    def catalog_path(self, path: str) -> str:
        return f"/lakekeeper/v1/{quote(self.warehouse, safe='')}{path}"

    # -- datasets ----------------------------------------------------------

    def create_dataset(
        self,
        namespace: str,
        name: str,
        location: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> "Dataset":
        body: Dict[str, Any] = {"name": name}
        if location is not None:
            body["location"] = location
        if constraints is not None:
            body["constraints"] = constraints
        self.request(
            "POST",
            self.catalog_path(f"/namespaces/{encode_namespace(namespace)}/datasets"),
            json=body,
        )
        return self.dataset(namespace, name)

    def dataset(self, namespace: str, name: str) -> "Dataset":
        """A handle to a dataset. Does not contact the server."""
        from .dataset import Dataset

        return Dataset(self, namespace, name)

    def list_datasets(self, namespace: str) -> Iterator[Dict[str, Any]]:
        """Yield every dataset in a namespace, following pagination."""
        path = self.catalog_path(f"/namespaces/{encode_namespace(namespace)}/datasets")
        token = None
        while True:
            params = {"pageToken": token} if token else None
            page = self.request("GET", path, params=params)
            for identifier in page.get("identifiers", []):
                yield identifier
            token = page.get("next-page-token")
            if not token:
                return
