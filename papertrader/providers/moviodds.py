"""MoviOdds adapter for Bet365 execution prices.

MoviOdds exposes a Bet365 board per league.  The board is matched to Bzzoiro
fixtures by kickoff (UTC) plus normalized team names; never by name alone.
Only the full-time 1X2 group (group 40) is used.
"""

from __future__ import annotations

import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Callable

import requests

from ..domain import Fixture, OddsSnapshot, parse_utc
from .base import ProviderError

DEFAULT_BASE_URL = "https://moviodds.com"
FULL_TIME_GROUP = 40
KICKOFF_TOLERANCE = timedelta(minutes=2)
NAME_SIMILARITY = 0.82

# Generic club tokens do not identify a team.  "athletic" is reduced too so
# "Athletic Bilbao" matches a provider's "Ath Bilbao"; the city remains.
_STOPWORDS = {
    "fc", "cf", "cd", "ud", "sd", "club", "de", "deportivo", "ath",
    "athletic",
}

# Provider naming differs for a few La Liga clubs even after generic tokens
# are stripped.  Keep this small and explicit; fuzzy matching is a fallback,
# not a licence to merge unrelated teams.
_ALIASES = {
    "real racing": "racing santander",
}


def normalize_team(value: Any) -> str:
    """Accent-insensitive, punctuation-free team key."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [token for token in text.split() if token not in _STOPWORDS]
    normalized = " ".join(tokens)
    normalized = normalized or text.strip()
    return _ALIASES.get(normalized, normalized)


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def teams_match(
    home_a: str, away_a: str, home_b: str, away_b: str
) -> bool:
    """Whether two home/away pairs identify the same fixture."""
    home = _similarity(normalize_team(home_a), normalize_team(home_b))
    away = _similarity(normalize_team(away_a), normalize_team(away_b))
    return home >= NAME_SIMILARITY and away >= NAME_SIMILARITY


class MoviOddsClient:
    """Small JSON client with retry/backoff for the MoviOdds board."""

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
            raise ValueError("MoviOdds API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = float(timeout)
        self.sleep = sleep
        self.max_retries = max(1, int(max_retries))

    def board(
        self, league: str = "Spain La Liga"
    ) -> list[dict[str, Any]]:
        """Fetch the Bet365 board for one league."""
        headers = {
            "X-API-Key": self.api_key,
            "Accept": "application/json",
        }
        params = {
            "sport": "soccer",
            "groups": FULL_TIME_GROUP,
            "site": "com",
            "league": league,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    f"{self.base_url}/odds",
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
                    f"MoviOdds HTTP {response.status_code}"
                )
                continue
            if response.status_code >= 400:
                raise ProviderError(
                    f"MoviOdds HTTP {response.status_code}"
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderError("MoviOdds returned invalid JSON") from exc
            return self._matches(payload)
        raise ProviderError(f"MoviOdds request failed: {last_error}")

    @staticmethod
    def _matches(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            raise ProviderError("MoviOdds returned an unexpected payload")
        matches = payload.get("matches")
        if isinstance(matches, list):
            return [item for item in matches if isinstance(item, dict)]
        # The single-match shape is useful in tests and is accepted here.
        if {"home", "away", "kickoff_utc"} <= set(payload):
            return [payload]
        return []


class MoviOddsBet365Provider:
    """Map a MoviOdds board onto Bzzoiro fixture event IDs."""

    def __init__(
        self,
        client: MoviOddsClient,
        board_league: str = "Spain La Liga",
    ):
        self.client = client
        self.board_league = board_league

    def fetch_board(
        self, fixtures: list[Fixture]
    ) -> dict[str, OddsSnapshot]:
        if not fixtures:
            return {}
        matches = self.client.board(self.board_league)
        fetched_at = datetime.now(timezone.utc)
        result: dict[str, OddsSnapshot] = {}
        for fixture in fixtures:
            best = self._best_match(fixture, matches)
            if best is None:
                continue
            snapshot = self._snapshot(best, fixture, fetched_at)
            if snapshot is not None:
                result[fixture.event_id] = snapshot
        return result

    @staticmethod
    def _best_match(
        fixture: Fixture, matches: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_score = -1.0
        for match in matches:
            if bool(match.get("in_play")):
                continue
            kickoff = parse_utc(match.get("kickoff_utc"))
            if kickoff is None:
                continue
            if abs((kickoff - fixture.kickoff_utc).total_seconds()) > (
                KICKOFF_TOLERANCE.total_seconds()
            ):
                continue
            home = normalize_team(match.get("home"))
            away = normalize_team(match.get("away"))
            our_home = normalize_team(fixture.home_team)
            our_away = normalize_team(fixture.away_team)
            score = min(
                _similarity(home, our_home),
                _similarity(away, our_away),
            )
            if score >= NAME_SIMILARITY and score > best_score:
                best, best_score = match, score
        return best

    def _snapshot(
        self,
        match: dict[str, Any],
        fixture: Fixture,
        fetched_at: datetime,
    ) -> OddsSnapshot | None:
        odds: dict[str, float | None] = {"home": None, "draw": None, "away": None}
        quote_times: list[datetime] = []
        for quote in match.get("quotes", []) or []:
            if not isinstance(quote, dict):
                continue
            try:
                group = int(quote.get("group_id", -1))
            except (TypeError, ValueError):
                continue
            if group != FULL_TIME_GROUP:
                continue
            label = str(quote.get("n2", "")).strip().upper()
            if label in {"1", "HOME"}:
                side = "home"
            elif label in {"X", "DRAW"}:
                side = "draw"
            elif label in {"2", "AWAY"}:
                side = "away"
            else:
                normalized = normalize_team(quote.get("label"))
                if normalized == normalize_team(fixture.home_team):
                    side = "home"
                elif normalized == normalize_team(fixture.away_team):
                    side = "away"
                elif normalized in {"draw", "tie"}:
                    side = "draw"
                else:
                    continue
            try:
                price = float(quote.get("odds_dec"))
            except (TypeError, ValueError):
                continue
            if price > 1.0:
                odds[side] = price
            observed = parse_utc(quote.get("ts"))
            if observed is not None:
                quote_times.append(observed)

        if not all(value is not None for value in odds.values()):
            return None

        observed = parse_utc(match.get("polled_at"))
        if observed is None:
            group_status = match.get("group_status") or {}
            if isinstance(group_status, dict):
                group = group_status.get(str(FULL_TIME_GROUP)) or {}
                if isinstance(group, dict):
                    observed = parse_utc(group.get("polled_at"))
        if observed is None and quote_times:
            observed = max(quote_times)
        if observed is None:
            observed = parse_utc(match.get("last_seen"))

        return OddsSnapshot(
            source="bet365",
            home=float(odds["home"]),
            draw=float(odds["draw"]),
            away=float(odds["away"]),
            observed_at=observed,
            fetched_at=fetched_at,
            raw=match,
        )
