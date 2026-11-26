"""Small value objects shared by the scheduler, providers, and ledger.

All timestamps are stored and compared in UTC.  Local-time conversion happens
only in the feature builder, which is the one place where the model's
calendar features need Europe/Madrid wall time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo

Selection = Literal["home", "draw", "away"]
SELECTIONS: tuple[str, ...] = ("home", "draw", "away")


def ensure_utc(value: datetime) -> datetime:
    """Return ``value`` as an aware UTC datetime."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_utc(value: Any) -> datetime | None:
    """Parse an API timestamp (ISO-8601, optional ``Z``) into UTC.

    Returns ``None`` for null/blank input.  Naive values are assumed UTC;
    this matches both providers, which return UTC.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ensure_utc(parsed)


def iso_utc(value: datetime | None) -> str | None:
    """Serialize a datetime as a stable UTC ISO-8601 string."""
    if value is None:
        return None
    return ensure_utc(value).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Fixture:
    """A scheduled match from the fixture provider."""

    event_id: str
    league: str
    home_team: str
    away_team: str
    kickoff_utc: datetime
    status: str = "notstarted"
    season_id: str | None = None
    season_start: date | None = None
    round_number: int | None = None
    home_score: int | None = None
    away_score: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", str(self.event_id))
        object.__setattr__(self, "kickoff_utc", ensure_utc(self.kickoff_utc))

    def decision_at(self, lead_minutes: int) -> datetime:
        return self.kickoff_utc - timedelta(minutes=lead_minutes)

    def local_kickoff(self, timezone_name: str = "Europe/Madrid") -> datetime:
        return self.kickoff_utc.astimezone(ZoneInfo(timezone_name))

    @property
    def label(self) -> str:
        return f"{self.home_team} vs {self.away_team}"


@dataclass(frozen=True)
class OddsSnapshot:
    """A complete 1X2 price snapshot from one provider.

    ``observed_at`` is the provider's own update timestamp; ``fetched_at`` is
    when this process retrieved it.  Freshness checks use ``observed_at``.
    """

    source: str
    home: float
    draw: float
    away: float
    observed_at: datetime | None = None
    fetched_at: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "home", float(self.home))
        object.__setattr__(self, "draw", float(self.draw))
        object.__setattr__(self, "away", float(self.away))
        if self.observed_at is not None:
            object.__setattr__(self, "observed_at", ensure_utc(self.observed_at))
        if self.fetched_at is not None:
            object.__setattr__(self, "fetched_at", ensure_utc(self.fetched_at))

    @property
    def odds(self) -> dict[str, float]:
        return {"home": self.home, "draw": self.draw, "away": self.away}

    def price(self, selection: str) -> float | None:
        return self.odds.get(selection)

    def is_complete(self) -> bool:
        return all(
            value == value and value > 1.0  # NaN-safe and positive
            for value in (self.home, self.draw, self.away)
        )


@dataclass(frozen=True)
class SnapshotAttempt:
    """The result of asking a provider for a snapshot.

    ``reason`` explains why ``snapshot`` is missing or why the caller should
    not trust it.  A snapshot can be present with a reason; the freshness
    policy in the engine makes the final call.
    """

    snapshot: OddsSnapshot | None = None
    reason: str | None = None
    raw: Any = field(default=None, repr=False)


@dataclass(frozen=True)
class FixtureResult:
    """A final (or void) result for one fixture."""

    event_id: str
    status: str = "finished"
    home_score: int | None = None
    away_score: int | None = None

    @property
    def is_final(self) -> bool:
        return str(self.status).lower() in {
            "finished", "ft", "full_time", "ended", "awarded"
        }

    @property
    def is_void(self) -> bool:
        return str(self.status).lower() in {
            "cancelled", "canceled", "postponed", "abandoned", "void"
        }

    def winner(self) -> str | None:
        if self.is_void or not self.is_final:
            return None
        if self.home_score is None or self.away_score is None:
            return None
        if self.home_score > self.away_score:
            return "home"
        if self.home_score < self.away_score:
            return "away"
        return "draw"


@dataclass(frozen=True)
class ModelPrediction:
    """Both models' binary calls and probabilities for one fixture."""

    logreg_prob: float
    tree_prob: float
    logreg_pick: bool
    tree_pick: bool
    version: str = "unknown"

    @property
    def agrees(self) -> bool:
        return bool(self.logreg_pick and self.tree_pick)


@dataclass
class DecisionOutcome:
    """Outcome of processing one fixture; useful for logs and tests."""

    fixture_id: str
    status: str
    decision_id: int | None = None
    odds_source: str = "none"
    selection: str | None = None
    reason: str | None = None
    fallback_reason: str | None = None
    bets: list[dict[str, Any]] = field(default_factory=list)
    recorded: bool = False
    retry: bool = False

    @property
    def placed(self) -> bool:
        return self.status == "placed" and bool(self.bets)
