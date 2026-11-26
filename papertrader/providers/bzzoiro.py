"""Bzzoiro adapter: fixtures, results, and consensus 1X2 odds.

Bzzoiro's free tier provides fixtures and a consensus price updated on its own
schedule (often every four hours more than a day out, every 30 minutes near
kickoff).  The client keeps retries small and never logs the API key.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

import requests

from ..domain import (
    Fixture,
    FixtureResult,
    OddsSnapshot,
    SnapshotAttempt,
    parse_utc,
)
from .base import ProviderError

DEFAULT_BASE_URL = "https://sports.bzzoiro.com"

logger = logging.getLogger(__name__)


def _team_name(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("name", "display_name", "title", "short_name"):
            if value.get(key):
                return str(value[key])
        return ""
    return "" if value is None else str(value)


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def season_year(kickoff: datetime) -> int:
    """Season label year for a UTC kickoff (July-June football calendar)."""
    return kickoff.year if kickoff.month >= 7 else kickoff.year - 1


class BzzoiroClient:
    """Minimal JSON client with retry/backoff for Bzzoiro API v2."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 3,
    ):
        if not api_key:
            raise ValueError("Bzzoiro API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = float(timeout)
        self.sleep = sleep
        self.max_retries = max(1, int(max_retries))

    def _url(self, path: str) -> str:
        return path if path.startswith("http") else f"{self.base_url}{path}"

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Accept": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    self._url(path),
                    headers=headers,
                    params=params,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                self.sleep(min(2 ** attempt, 5))
                continue

            if response.status_code == 429 or 500 <= response.status_code < 600:
                retry_after = response.headers.get("Retry-After", "")
                try:
                    delay = float(retry_after)
                except (TypeError, ValueError):
                    delay = float(2 ** attempt)
                self.sleep(min(max(delay, 0.0), 30.0))
                last_error = ProviderError(
                    f"Bzzoiro HTTP {response.status_code}"
                )
                continue
            if response.status_code >= 400:
                raise ProviderError(
                    f"Bzzoiro HTTP {response.status_code}"
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderError("Bzzoiro returned invalid JSON") from exc
            if not isinstance(payload, dict):
                raise ProviderError("Bzzoiro returned a non-object payload")
            return payload
        raise ProviderError(f"Bzzoiro request failed: {last_error}")

    def events(
        self,
        league_id: int,
        date_from: date,
        date_to: date,
        status: str | None = None,
        season_id: str | int | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Fetch all pages of an events query."""
        params: dict[str, Any] = {
            "league_id": int(league_id),
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
            "limit": int(limit),
        }
        if status:
            params["status"] = status
        if season_id is not None:
            params["season_id"] = season_id

        results: list[dict[str, Any]] = []
        next_path: str | None = "/api/v2/events/"
        first = True
        for _ in range(50):
            if next_path is None:
                break
            payload = self._get(next_path, params if first else None)
            first = False
            chunk = payload.get("results")
            if chunk is None:
                # A few endpoints return a bare list-like object.
                if isinstance(payload.get("data"), list):
                    chunk = payload["data"]
                else:
                    chunk = []
            results.extend(item for item in chunk if isinstance(item, dict))
            next_path = payload.get("next")
            if next_path is not None and not isinstance(next_path, str):
                raise ProviderError("Bzzoiro pagination returned a bad URL")
        return results

    def consensus(self, event_id: str | int) -> dict[str, Any]:
        return self._get(f"/api/v2/events/{event_id}/odds/")

    def season_start(
        self,
        league_id: int,
        season_id: str | int | None,
        label_year: int,
        timezone_name: str = "Europe/Madrid",
    ) -> date | None:
        """Earliest local date in the July-September portion of a season."""
        events = self.events(
            league_id=league_id,
            date_from=date(label_year, 7, 1),
            date_to=date(label_year, 9, 30),
            season_id=season_id,
        )
        dates = []
        for event in events:
            kickoff = parse_utc(event.get("event_date"))
            if kickoff is not None:
                dates.append(
                    kickoff.astimezone(ZoneInfo(timezone_name)).date()
                )
        return min(dates) if dates else None


def parse_consensus(
    payload: dict[str, Any],
    fetched_at: datetime | None = None,
) -> SnapshotAttempt:
    """Turn a Bzzoiro odds payload into a complete snapshot attempt."""
    odds = payload.get("odds") if isinstance(payload, dict) else None
    if not isinstance(odds, dict):
        return SnapshotAttempt(None, "no_odds_block", raw=payload)
    try:
        home = float(odds["home_win"])
        draw = float(odds["draw"])
        away = float(odds["away_win"])
    except (KeyError, TypeError, ValueError):
        return SnapshotAttempt(None, "incomplete_outcomes", raw=payload)
    if not all(value == value and value > 1.0 for value in (home, draw, away)):
        return SnapshotAttempt(None, "incomplete_outcomes", raw=payload)

    observed = parse_utc(payload.get("last_update_at"))
    snapshot = OddsSnapshot(
        source="consensus",
        home=home,
        draw=draw,
        away=away,
        observed_at=observed,
        fetched_at=fetched_at or datetime.now(timezone.utc),
        raw=payload,
    )
    if observed is None:
        return SnapshotAttempt(snapshot, "missing_update_time", raw=payload)
    return SnapshotAttempt(snapshot)


def parse_event(
    payload: dict[str, Any],
    league: str,
    season_start: date | None = None,
) -> Fixture:
    """Parse one Bzzoiro event into a :class:`Fixture`."""
    kickoff = parse_utc(payload.get("event_date"))
    if kickoff is None:
        raise ProviderError(
            f"Bzzoiro event {payload.get('id')} has invalid event_date"
        )
    event_id = payload.get("id")
    if event_id is None:
        raise ProviderError("Bzzoiro event is missing id")
    status = str(payload.get("status", "notstarted")).strip().lower()
    if status in {"scheduled", "ns", "not_started", "upcoming"}:
        status = "notstarted"
    home_team = _team_name(payload.get("home_team"))
    away_team = _team_name(payload.get("away_team"))
    if not home_team or not away_team:
        raise ProviderError(
            f"Bzzoiro event {event_id} is missing a team name"
        )
    return Fixture(
        event_id=str(event_id),
        league=league,
        home_team=home_team,
        away_team=away_team,
        kickoff_utc=kickoff,
        status=status,
        season_id=str(payload["season_id"])
        if payload.get("season_id") is not None else None,
        season_start=season_start,
        round_number=_int_or_none(payload.get("round_number")),
        home_score=_int_or_none(payload.get("home_score")),
        away_score=_int_or_none(payload.get("away_score")),
    )


def parse_result(payload: dict[str, Any]) -> FixtureResult:
    event_id = payload.get("id")
    if event_id is None:
        raise ProviderError("Bzzoiro result is missing id")
    status = str(payload.get("status", "")).strip().lower()
    return FixtureResult(
        event_id=str(event_id),
        status=status or "unknown",
        home_score=_int_or_none(payload.get("home_score")),
        away_score=_int_or_none(payload.get("away_score")),
    )


class BzzoiroFixtureProvider:
    """Fixtures, results, and consensus from one Bzzoiro client."""

    def __init__(
        self,
        client: BzzoiroClient,
        league: str = "La Liga",
        league_id: int = 3,
        timezone: str = "Europe/Madrid",
    ):
        self.client = client
        self.league = league
        self.league_id = int(league_id)
        self.timezone = timezone
        self._season_cache: dict[tuple[str, int], date | None] = {}

    def _season_start(
        self, season_id: str | None, kickoff: datetime
    ) -> date | None:
        if season_id is None:
            return None
        label_year = season_year(kickoff)
        key = (str(season_id), label_year)
        if key not in self._season_cache:
            self._season_cache[key] = self.client.season_start(
                self.league_id, season_id, label_year, self.timezone
            )
        return self._season_cache[key]

    def fetch_fixtures(
        self,
        start_date: date,
        end_date: date,
        include_all: bool = False,
    ) -> list[Fixture]:
        """Fetch fixtures for a date range.

        ``include_all=True`` also returns finished/postponed events so a
        previously scheduled fixture can be updated to its terminal status;
        the scheduler only acts on ``notstarted`` rows.
        """
        payloads = self.client.events(
            league_id=self.league_id,
            date_from=start_date,
            date_to=end_date,
            status=None if include_all else "notstarted",
        )
        fixtures = []
        for payload in payloads:
            try:
                kickoff = parse_utc(payload.get("event_date"))
                if kickoff is None:
                    raise ProviderError("missing event_date")
                season_id = (
                    str(payload["season_id"])
                    if payload.get("season_id") is not None else None
                )
                season_start = self._season_start(season_id, kickoff)
                fixtures.append(
                    parse_event(
                        payload, self.league, season_start=season_start
                    )
                )
            except (ProviderError, KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "skipping malformed Bzzoiro fixture %r: %s",
                    payload.get("id"), exc,
                )
        fixtures.sort(key=lambda f: (f.kickoff_utc, f.event_id))
        return fixtures

    def fetch_results(
        self, start_date: date, end_date: date
    ) -> list[FixtureResult]:
        # Do not filter to ``finished``: cancelled/postponed events must be
        # returned so pending bets can be voided.
        payloads = self.client.events(
            league_id=self.league_id,
            date_from=start_date,
            date_to=end_date,
        )
        results = []
        for payload in payloads:
            try:
                results.append(parse_result(payload))
            except (ProviderError, KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "skipping malformed Bzzoiro result %r: %s",
                    payload.get("id"), exc,
                )
        return results

    def fetch_consensus(self, event_id: str) -> SnapshotAttempt:
        payload = self.client.consensus(event_id)
        return parse_consensus(payload)
