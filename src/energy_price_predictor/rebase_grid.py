"""Client for the Rebase Grid API (European power system data).

Docs: https://grid.rebase.energy/docs/v1/  (OpenAPI: .../docs/v1/openapi.json)

Typical use::

    from energy_price_predictor import RebaseGridClient

    grid = RebaseGridClient()  # reads REBASE_GRID_API_KEY from the environment / .env
    prices = grid.values(
        "DK-DK1",
        "price.day_ahead.day_ahead_auction_price",
        provider="entsoe",
        start="2024-01-01",
        end="2024-12-31T23:00Z",
    )

Behaviour worth knowing:

* All times are UTC. ``start`` and ``end`` are inclusive.
* A values call may span at most 100,000 points, so :meth:`values` splits long ranges into
  windows sized from the series' resolution (read once from the metadata and cached).
* 429 and 503 responses are retried after ``Retry-After``; other errors raise
  :class:`RebaseGridError`.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://grid.rebase.energy/api/v1"
API_KEY_ENV_VAR = "REBASE_GRID_API_KEY"

MAX_POINTS_PER_CALL = 100_000
MAX_SPAN_IRREGULAR = pd.Timedelta(days=366)
MAX_SERIES_PER_MANY_CALL = 2_000
MAX_PAGE_LIMIT = 1_000
RETRY_STATUSES = {429, 503}

TimeLike = str | pd.Timestamp


class RebaseGridError(RuntimeError):
    """An error response from the API, with the fields it carries."""

    def __init__(self, status: int, body: dict[str, Any]):
        self.status = status
        self.error = body.get("error")
        self.param = body.get("param")
        self.closest = body.get("closest")
        self.providers = body.get("providers")
        self.body = body
        message = f"{status} {self.error}: {body.get('message', '')}"
        if self.param:
            message += f" (param: {self.param})"
        if self.closest:
            message += f" (closest: {self.closest})"
        if self.providers:
            message += f" (providers: {self.providers})"
        super().__init__(message)


@dataclass(frozen=True)
class SeriesRef:
    """One timeseries from one provider: what a values call names."""

    entity: str
    name: str
    provider: str

    def as_dict(self) -> dict[str, str]:
        return {"entity": self.entity, "name": self.name, "provider": self.provider}


class RebaseGridClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 5,
        session: requests.Session | None = None,
    ):
        if api_key is None:
            load_dotenv()
            api_key = os.environ.get(API_KEY_ENV_VAR)
        if not api_key:
            raise ValueError(f"No API key: pass api_key or set {API_KEY_ENV_VAR} (e.g. in .env).")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.session.headers["X-API-Key"] = api_key
        self._metadata_cache: dict[tuple[str, str], dict[str, Any]] = {}

    # ------------------------------------------------------------------ HTTP

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        for attempt in range(self.max_retries + 1):
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            if response.status_code in RETRY_STATUSES and attempt < self.max_retries:
                time.sleep(_retry_after_seconds(response, attempt))
                continue
            if not response.ok:
                try:
                    body = response.json()
                except ValueError:
                    body = {"error": "http_error", "message": response.text[:500]}
                raise RebaseGridError(response.status_code, body)
            return response.json()
        raise AssertionError("unreachable")

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("GET", path, params=_drop_none(params or {}))

    def _get_paged(self, path: str, params: dict[str, Any], items_key: str) -> list[dict[str, Any]]:
        """Follow ``next`` (sent back as ``cursor``) until it is null."""
        params = dict(params)
        items: list[dict[str, Any]] = []
        while True:
            page = self._get(path, params)
            items.extend(page.get(items_key, []))
            if not page.get("next"):
                return items
            params["cursor"] = page["next"]

    # --------------------------------------------------------------- Account

    def me(self) -> dict[str, Any]:
        """The owner of the key. Use to check the key works."""
        return self._get("/me")

    def version(self) -> dict[str, Any]:
        return self._get("/version")

    # -------------------------------------------------------------- Discover

    def entity_tree(self) -> list[dict[str, Any]]:
        """The whole entity tree (countries, price zones, areas, ...), nested via ``children``."""
        return self._get("/entities")["entities"]

    def entities(self) -> pd.DataFrame:
        """The entity tree flattened to one row per entity: id, kind, name, parent, timeseries."""
        rows: list[dict[str, Any]] = []
        for node, parent in _walk_tree(self.entity_tree()):
            rows.append(
                {
                    "id": node["id"],
                    "kind": node.get("kind"),
                    "name": node.get("name"),
                    "parent": parent,
                    "timeseries": node.get("timeseries"),
                }
            )
        return pd.DataFrame(rows)

    def entity(self, entity: str) -> dict[str, Any]:
        """One entity with its parent, children and relations."""
        return self._get(f"/entities/{entity}")

    def timeseries_names(
        self,
        entity: str,
        *,
        provider: str | None = None,
        category: str | None = None,
        q: str | None = None,
        data_type: str | None = None,
    ) -> pd.DataFrame:
        """The timeseries names on one entity, with their providers."""
        body = self._get(
            f"/entities/{entity}/timeseries",
            {"provider": provider, "category": category, "q": q, "data_type": data_type},
        )
        return pd.DataFrame(_first_list(body))

    def search(
        self,
        *,
        entity: str | None = None,
        q: str | None = None,
        category: str | None = None,
        subcategory: str | None = None,
        provider: str | None = None,
        data_type: str | None = None,
    ) -> pd.DataFrame:
        """Find timeseries across entities, following every page."""
        params = {
            "entity": entity,
            "q": q,
            "category": category,
            "subcategory": subcategory,
            "provider": provider,
            "data_type": data_type,
            "limit": MAX_PAGE_LIMIT,
        }
        return pd.DataFrame(self._get_paged("/timeseries", _drop_none(params), "items"))

    def plants(self, entity: str, **filters: Any) -> pd.DataFrame:
        """Plants in an entity. Filters: technology, min_mw, has_timeseries, bbox, provider, portfolio."""
        return self._list_assets(entity, "plants", filters)

    def units(self, entity: str, **filters: Any) -> pd.DataFrame:
        """Units in an entity. Filters: type, production_type, fuel, min_kw, has_timeseries, bbox, provider."""
        return self._list_assets(entity, "units", filters)

    def portfolios(self, entity: str, **filters: Any) -> pd.DataFrame:
        """Portfolios in an entity. Filters: portfolio_kind, min_mw, has_timeseries, provider."""
        return self._list_assets(entity, "portfolios", filters)

    def _list_assets(self, entity: str, kind: str, filters: dict[str, Any]) -> pd.DataFrame:
        params = _drop_none({**filters, "limit": MAX_PAGE_LIMIT})
        return pd.DataFrame(self._get_paged(f"/entities/{entity}/{kind}", params, "items"))

    # -------------------------------------------------------------- Describe

    def metadata(self, entity: str, name: str, *, refresh: bool = False) -> dict[str, Any]:
        """What a timeseries measures, its unit, and per provider its resolution, extent and latency.

        Cached per client: the API recomputes it at most once a day.
        """
        key = (entity, name)
        if refresh or key not in self._metadata_cache:
            self._metadata_cache[key] = self._get(f"/entities/{entity}/timeseries/{name}")
        return self._metadata_cache[key]

    def provider_metadata(self, entity: str, name: str, provider: str) -> dict[str, Any]:
        meta = self.metadata(entity, name)
        for entry in meta.get("providers", []):
            if entry.get("provider") == provider:
                return entry
        available = [entry.get("provider") for entry in meta.get("providers", [])]
        raise ValueError(f"{provider!r} does not publish {entity}/{name}; providers: {available}")

    # ------------------------------------------------------------------ Read

    def values(
        self,
        entity: str,
        name: str,
        *,
        provider: str,
        start: TimeLike,
        end: TimeLike,
        as_of: TimeLike | None = None,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Values of one timeseries over ``[start, end]`` (inclusive, UTC).

        ``as_of`` omitted gives the latest version; an instant gives the values as known then.
        For one snapshot per period use :meth:`snapshots`.

        Returns a DataFrame indexed by ``valid_time`` (UTC) with one column per value column
        (usually just ``value``) plus ``knowledge_time`` and ``annotation``. Long ranges are
        split into windows of at most 100,000 points.
        """
        if as_of is not None and _is_snapshot_as_of(as_of):
            raise ValueError("as_of is a period or 'rolling': use snapshots() instead.")
        start_ts, end_ts = _utc(start), _utc(end)
        path = f"/entities/{entity}/timeseries/{name}/values"
        base_params = {
            "provider": provider,
            "as_of": _iso(as_of) if as_of is not None else None,
            "columns": ",".join(columns) if columns else None,
        }
        frames = []
        for window_start, window_end in self._windows(entity, name, provider, start_ts, end_ts):
            body = self._get(path, {**base_params, "start": _iso(window_start), "end": _iso(window_end)})
            frames.append(_points_to_frame(body.get("points", [])))
        frame = pd.concat(frames) if frames else _points_to_frame([])
        frame = frame[~frame.index.duplicated(keep="last")]
        frame.attrs.update(entity=entity, name=name, provider=provider, unit=body.get("unit") if frames else None)
        return frame

    def snapshots(
        self,
        entity: str,
        name: str,
        *,
        provider: str,
        start: TimeLike,
        end: TimeLike,
        as_of: str,
        at: str | None = None,
        columns: Iterable[str] | None = None,
        max_snapshots: int = 1_000,
    ) -> pd.DataFrame:
        """The window as it was known at many instants (forecast vintages, revisions).

        ``as_of`` is an ISO 8601 period (``"PT1H"``, ``"P1D"``) or ``"rolling"``; ``at`` anchors
        the period to a time of day (``"10:00"``). Keep the window short or the period long:
        one request may produce at most 10,000 snapshots.

        Returns a long DataFrame with columns ``as_of``, ``valid_time``, the value column(s),
        ``knowledge_time`` and ``annotation``.
        """
        path = f"/entities/{entity}/timeseries/{name}/values"
        params = {
            "provider": provider,
            "start": _iso(_utc(start)),
            "end": _iso(_utc(end)),
            "as_of": as_of,
            "at": at,
            "columns": ",".join(columns) if columns else None,
            "max_snapshots": max_snapshots,
        }
        frames = []
        for snapshot in self._get_paged(path, _drop_none(params), "snapshots"):
            frame = _points_to_frame(snapshot.get("points", [])).reset_index()
            frame.insert(0, "as_of", pd.Timestamp(snapshot["as_of"]))
            frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=["as_of", "valid_time", "value", "knowledge_time", "annotation"])
        return pd.concat(frames, ignore_index=True)

    def many_values(
        self,
        series: Iterable[SeriesRef | tuple[str, str, str] | dict[str, str]],
        *,
        start: TimeLike,
        end: TimeLike,
        raise_on_error: bool = False,
    ) -> pd.DataFrame:
        """Latest values of many timeseries over one shared window, in as few calls as possible.

        ``series`` items are :class:`SeriesRef`, ``(entity, name, provider)`` tuples or dicts.
        Returns a long DataFrame: ``entity``, ``name``, ``provider``, ``valid_time``, value
        column(s), ``knowledge_time``, ``annotation``. Series that fail are skipped (or raise if
        ``raise_on_error``); their errors are in ``frame.attrs["errors"]``.
        """
        refs = [_as_series_ref(item) for item in series]
        frames: list[pd.DataFrame] = []
        errors: list[dict[str, Any]] = []
        for batch_start in range(0, len(refs), MAX_SERIES_PER_MANY_CALL):
            batch = refs[batch_start : batch_start + MAX_SERIES_PER_MANY_CALL]
            body: dict[str, Any] = {
                "start": _iso(_utc(start)),
                "end": _iso(_utc(end)),
                "series": [ref.as_dict() for ref in batch],
            }
            while True:
                page = self._request("POST", "/timeseries/values", json=body)
                for block in page.get("series", []):
                    if "error" in block:
                        errors.append(block)
                        continue
                    frame = _points_to_frame(block.get("points", [])).reset_index()
                    frame.insert(0, "provider", block["provider"])
                    frame.insert(0, "name", block["name"])
                    frame.insert(0, "entity", block["entity"])
                    frames.append(frame)
                if not page.get("next"):
                    break
                body["cursor"] = page["next"]
        if errors and raise_on_error:
            raise RebaseGridError(400, {"error": "series_errors", "message": str(errors)})
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        result.attrs["errors"] = errors
        return result

    # --------------------------------------------------------------- helpers

    def _windows(
        self, entity: str, name: str, provider: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
        """Consecutive inclusive windows that each stay under the per-call point limit."""
        if end < start:
            raise ValueError(f"end ({end}) is before start ({start})")
        step, max_span = self._max_span(entity, name, provider)
        window_start = start
        while window_start <= end:
            window_end = min(window_start + max_span, end)
            yield window_start, window_end
            window_start = window_end + step

    def _max_span(self, entity: str, name: str, provider: str) -> tuple[pd.Timedelta, pd.Timedelta]:
        """(step, span): the finest resolution the series has had, and the widest window it allows."""
        entry = self.provider_metadata(entity, name, provider)
        resolutions = [entry.get("resolution")]
        resolutions += [item.get("resolution") for item in entry.get("resolution_history") or []]
        steps = [step for step in (_parse_resolution(res) for res in resolutions) if step is not None]
        if not steps:
            return pd.Timedelta(seconds=1), MAX_SPAN_IRREGULAR
        step = min(steps)
        # Inclusive end: a span of k steps holds k + 1 points.
        return step, step * (MAX_POINTS_PER_CALL - 1)


# ---------------------------------------------------------------- module helpers


def _retry_after_seconds(response: requests.Response, attempt: int) -> float:
    header = response.headers.get("Retry-After")
    try:
        return max(float(header), 0.0) if header is not None else 2.0**attempt
    except ValueError:
        return 2.0**attempt


def _drop_none(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value is not None}


def _utc(value: TimeLike) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _iso(value: TimeLike) -> str:
    return _utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_snapshot_as_of(value: TimeLike) -> bool:
    return isinstance(value, str) and (value == "rolling" or value.upper().startswith("P"))


def _parse_resolution(resolution: str | None) -> pd.Timedelta | None:
    """ISO 8601 duration -> Timedelta; None for irregular or calendar (month/year) resolutions."""
    if not resolution:
        return None
    try:
        return pd.Timedelta(resolution)
    except ValueError:
        return None


def _points_to_frame(points: list[dict[str, Any]]) -> pd.DataFrame:
    """API points -> DataFrame indexed by valid_time, flattening multi-column ``values``."""
    if not points:
        frame = pd.DataFrame(columns=["value", "knowledge_time", "annotation"])
        frame.index = pd.DatetimeIndex([], tz="UTC", name="valid_time")
        return frame
    frame = pd.DataFrame(points)
    if "values" in frame.columns:
        value_columns = pd.DataFrame(frame.pop("values").tolist(), index=frame.index)
        frame = pd.concat([frame, value_columns], axis=1)
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True, format="ISO8601")
    frame["knowledge_time"] = pd.to_datetime(frame["knowledge_time"], utc=True, format="ISO8601")
    ordered = [col for col in frame.columns if col not in ("valid_time", "knowledge_time", "annotation")]
    return frame.set_index("valid_time")[ordered + ["knowledge_time", "annotation"]].sort_index()


def _walk_tree(nodes: list[dict[str, Any]], parent: str | None = None) -> Iterator[tuple[dict[str, Any], str | None]]:
    for node in nodes:
        yield node, parent
        yield from _walk_tree(node.get("children", []), node["id"])


def _first_list(body: dict[str, Any]) -> list[dict[str, Any]]:
    """The list in a response whose list key is not fixed across endpoints (e.g. items/timeseries)."""
    for key in ("items", "timeseries"):
        if isinstance(body.get(key), list):
            return body[key]
    for value in body.values():
        if isinstance(value, list):
            return value
    return []


def _as_series_ref(item: SeriesRef | tuple[str, str, str] | dict[str, str]) -> SeriesRef:
    if isinstance(item, SeriesRef):
        return item
    if isinstance(item, dict):
        return SeriesRef(item["entity"], item["name"], item["provider"])
    return SeriesRef(*item)
