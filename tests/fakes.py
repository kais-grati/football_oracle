"""Test doubles and shared fixtures for the paper-trader suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

from papertrader.domain import (
    Fixture,
    FixtureResult,
    ModelPrediction,
    OddsSnapshot,
    SnapshotAttempt,
)

MADRID = ZoneInfo("Europe/Madrid")


def local_kickoff(
    hour: int,
    minute: int = 0,
    day: int = 28,
    month: int = 11,
    year: int = 2026,
    second: int = 0,
) -> datetime:
    """A Europe/Madrid local kickoff as UTC."""
    local = datetime(year, month, day, hour, minute, second, tzinfo=MADRID)
    return local.astimezone(timezone.utc)


def make_fixture(
    event_id: str = "1",
    *,
    hour: int = 14,
    minute: int = 0,
    home: str = "Real Madrid",
    away: str = "Deportivo Alavés",
    season_start: date | None = date(2026, 8, 15),
    league: str = "La Liga",
) -> Fixture:
    return Fixture(
        event_id=event_id,
        league=league,
        home_team=home,
        away_team=away,
        kickoff_utc=local_kickoff(hour, minute),
        status="notstarted",
        season_id="1307",
        season_start=season_start,
        round_number=14,
    )


def make_snapshot(
    source: str = "consensus",
    *,
    home: float = 1.60,
    draw: float = 4.20,
    away: float = 5.50,
    observed_at: datetime | None = None,
) -> OddsSnapshot:
    return OddsSnapshot(
        source=source,
        home=home,
        draw=draw,
        away=away,
        observed_at=observed_at,
        fetched_at=observed_at,
        raw={"source": source},
    )


class FakeClock:
    def __init__(self, now: datetime):
        self.current = now

    def now(self) -> datetime:
        return self.current

    def set(self, now: datetime) -> None:
        self.current = now

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


@dataclass
class FakeConsensus:
    attempts: dict[str, SnapshotAttempt] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def fetch_consensus(self, event_id: str) -> SnapshotAttempt:
        self.calls.append(str(event_id))
        value = self.attempts.get(str(event_id), SnapshotAttempt(None, "missing"))
        if isinstance(value, SnapshotAttempt):
            return value
        if isinstance(value, OddsSnapshot):
            return SnapshotAttempt(value)
        if isinstance(value, Exception):
            raise value
        return SnapshotAttempt(None, "missing")


@dataclass
class FakeBet365:
    snapshots: dict[str, OddsSnapshot] = field(default_factory=dict)
    calls: list[list[str]] = field(default_factory=list)

    def fetch_board(self, fixtures: list[Fixture]) -> dict[str, OddsSnapshot]:
        self.calls.append([fixture.event_id for fixture in fixtures])
        return {
            fixture.event_id: self.snapshots[fixture.event_id]
            for fixture in fixtures
            if fixture.event_id in self.snapshots
        }


@dataclass
class FakeFixtureProvider:
    fixtures: list[Fixture] = field(default_factory=list)
    results: list[FixtureResult] = field(default_factory=list)
    fixture_calls: list[tuple[date, date]] = field(default_factory=list)
    result_calls: list[tuple[date, date]] = field(default_factory=list)

    def fetch_fixtures(
        self, start_date: date, end_date: date, include_all: bool = False
    ) -> list[Fixture]:
        self.fixture_calls.append((start_date, end_date))
        return [
            fixture for fixture in self.fixtures
            if start_date <= fixture.kickoff_utc.date() <= end_date
        ]

    def fetch_results(
        self, start_date: date, end_date: date
    ) -> list[FixtureResult]:
        self.result_calls.append((start_date, end_date))
        return list(self.results)


@dataclass
class FakeResults:
    results: list[FixtureResult] = field(default_factory=list)
    calls: list[tuple[date, date]] = field(default_factory=list)

    def fetch_results(
        self, start_date: date, end_date: date
    ) -> list[FixtureResult]:
        self.calls.append((start_date, end_date))
        return list(self.results)


@dataclass
class FakeModel:
    logreg_prob: float = 0.70
    tree_prob: float = 0.72
    logreg_pick: bool = True
    tree_pick: bool = True
    version: str = "fake-model-v1"
    calls: list[dict] = field(default_factory=list)

    def predict(self, features: dict) -> ModelPrediction:
        self.calls.append(dict(features))
        return ModelPrediction(
            logreg_prob=self.logreg_prob,
            tree_prob=self.tree_prob,
            logreg_pick=self.logreg_pick,
            tree_pick=self.tree_pick,
            version=self.version,
        )


@dataclass
class FakeNotifier:
    enabled: bool = True
    result: bool = True
    calls: list[tuple[dict, list[dict]]] = field(default_factory=list)

    def send_bets(self, fixture: dict, wallet_bets: list[dict]) -> bool:
        self.calls.append((dict(fixture), [dict(item) for item in wallet_bets]))
        return self.result


@dataclass
class FakeResponse:
    status_code: int = 200
    payload: object = None
    headers: dict[str, str] = field(default_factory=dict)

    def json(self):
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses: list[FakeResponse] | None = None):
        self.responses = list(responses or [])
        self.calls: list[tuple[str, str, dict]] = []

    def _call(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        if not self.responses:
            return FakeResponse(200, {})
        return self.responses.pop(0)

    def get(self, url, **kwargs):
        return self._call("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._call("POST", url, **kwargs)


def no_sleep(seconds: float) -> None:
    return None
