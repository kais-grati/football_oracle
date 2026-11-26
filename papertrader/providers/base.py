"""Provider interfaces.

The scheduler depends on these protocols rather than on HTTP clients.  Tests
provide in-memory fakes; production wires the Bzzoiro and MoviOdds adapters.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from ..domain import Fixture, FixtureResult, SnapshotAttempt


class ProviderError(RuntimeError):
    """Raised when a provider cannot return a usable response."""


class FixtureProvider(Protocol):
    def fetch_fixtures(
        self, start_date: date, end_date: date, include_all: bool = False
    ) -> list[Fixture]:
        ...


class ConsensusProvider(Protocol):
    def fetch_consensus(self, event_id: str) -> SnapshotAttempt:
        ...


class Bet365Provider(Protocol):
    def fetch_board(self, fixtures: list[Fixture]) -> dict[str, object]:
        """Return ``event_id -> OddsSnapshot`` for fixtures on the board."""
        ...


class ResultsProvider(Protocol):
    def fetch_results(
        self, start_date: date, end_date: date
    ) -> list[FixtureResult]:
        ...
