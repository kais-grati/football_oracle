from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from papertrader.domain import Fixture, FixtureResult
from papertrader.providers.base import ProviderError
from papertrader.providers.bzzoiro import (
    BzzoiroClient,
    BzzoiroFixtureProvider,
    parse_consensus,
    parse_event,
    parse_result,
)
from papertrader.providers.moviodds import (
    MoviOddsBet365Provider,
    MoviOddsClient,
    normalize_team,
    teams_match,
)

from .fakes import FakeResponse, FakeSession, no_sleep

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str):
    return json.loads((FIXTURES / name).read_text())


def test_parse_populated_consensus():
    attempt = parse_consensus(load("bzzoiro_odds_populated.json"))
    assert attempt.reason is None
    snapshot = attempt.snapshot
    assert snapshot is not None
    assert (snapshot.home, snapshot.draw, snapshot.away) == (1.15, 8.61, 14.86)
    assert snapshot.source == "consensus"
    assert snapshot.observed_at is not None
    assert snapshot.observed_at.tzinfo is not None


def test_parse_null_consensus_is_incomplete():
    attempt = parse_consensus(load("bzzoiro_odds_null.json"))
    assert attempt.snapshot is None
    assert attempt.reason == "incomplete_outcomes"


def test_parse_consensus_missing_block():
    attempt = parse_consensus({"event_id": 1})
    assert attempt.snapshot is None
    assert attempt.reason == "no_odds_block"


def test_parse_event_and_result():
    fixture = parse_event(
        load("bzzoiro_event.json"), "La Liga", season_start=date(2026, 8, 15)
    )
    assert fixture.event_id == "213657"
    assert fixture.home_team == "Real Madrid"
    assert fixture.away_team == "Deportivo Alavés"
    assert fixture.kickoff_utc == datetime(
        2026, 11, 28, 20, tzinfo=timezone.utc
    )
    assert fixture.season_id == "1307"
    assert fixture.season_start == date(2026, 8, 15)
    assert fixture.round_number == 14

    result = parse_result({
        "id": 213657, "status": "finished",
        "home_score": 2, "away_score": 1,
    })
    assert result == FixtureResult(
        event_id="213657", status="finished", home_score=2, away_score=1
    )
    assert result.winner() == "home"


class StubBzzoiroClient:
    def __init__(self):
        self.event_calls = []
        self.season_calls = []

    def events(self, **kwargs):
        self.event_calls.append(kwargs)
        if kwargs.get("status") in (None, "finished"):
            return [{
                "id": 99, "status": "finished",
                "home_score": 0, "away_score": 0,
            }]
        return [load("bzzoiro_event.json")]

    def season_start(self, league_id, season_id, label_year, timezone_name):
        self.season_calls.append(
            (league_id, season_id, label_year, timezone_name)
        )
        return date(2026, 8, 15)

    def consensus(self, event_id):
        return load("bzzoiro_odds_populated.json")


def test_bzzoiro_fixture_provider_wires_season_and_results():
    client = StubBzzoiroClient()
    provider = BzzoiroFixtureProvider(client)
    fixtures = provider.fetch_fixtures(date(2026, 11, 28), date(2026, 11, 28))
    assert len(fixtures) == 1
    assert fixtures[0].season_start == date(2026, 8, 15)
    assert client.season_calls == [
        (3, "1307", 2026, "Europe/Madrid")
    ]

    results = provider.fetch_results(date(2026, 11, 28), date(2026, 11, 28))
    assert results[0].event_id == "99"
    attempt = provider.fetch_consensus("213657")
    assert attempt.snapshot is not None


def test_bzzoiro_provider_skips_malformed_rows():
    class MalformedClient(StubBzzoiroClient):
        def events(self, **kwargs):
            if kwargs.get("status") is None:
                return [{}, {
                    "id": 99, "status": "finished",
                    "home_score": 1, "away_score": 0,
                }]
            return [{"id": "bad", "home_team": "A", "away_team": "B"},
                    load("bzzoiro_event.json")]

    provider = BzzoiroFixtureProvider(MalformedClient())
    fixtures = provider.fetch_fixtures(date(2026, 11, 28), date(2026, 11, 28))
    assert [fixture.event_id for fixture in fixtures] == ["213657"]
    results = provider.fetch_results(date(2026, 11, 1), date(2026, 11, 2))
    assert [result.event_id for result in results] == ["99"]


def test_client_retries_429():
    session = FakeSession([
        FakeResponse(429, {}, {"Retry-After": "0"}),
        FakeResponse(200, {"results": []}),
    ])
    client = BzzoiroClient(
        "key", session=session, sleep=no_sleep, max_retries=3
    )
    assert client.events(3, date(2026, 11, 28), date(2026, 11, 28)) == []
    assert len(session.calls) == 2


def test_client_rejects_http_error():
    session = FakeSession([FakeResponse(400, {"detail": "bad"})])
    client = BzzoiroClient("key", session=session, sleep=no_sleep)
    with pytest.raises(ProviderError):
        client.events(3, date(2026, 11, 28), date(2026, 11, 28))


def test_client_follows_pagination():
    session = FakeSession([
        FakeResponse(200, {
            "results": [{"id": 1}],
            "next": "https://example.test/page2",
        }),
        FakeResponse(200, {"results": [{"id": 2}], "next": None}),
    ])
    client = BzzoiroClient("key", session=session, sleep=no_sleep)
    rows = client.events(3, date(2026, 11, 28), date(2026, 11, 28))
    assert [row["id"] for row in rows] == [1, 2]
    assert session.calls[1][1] == "https://example.test/page2"


def test_team_normalization_handles_accents_and_provider_variants():
    assert normalize_team("Deportivo Alavés") == "alaves"
    assert normalize_team("Alaves") == "alaves"
    assert normalize_team("Athletic Bilbao") == "bilbao"
    assert normalize_team("Ath Bilbao") == "bilbao"
    assert normalize_team("Real Racing Club") == "racing santander"
    assert normalize_team("Racing Santander") == "racing santander"
    assert teams_match(
        "Athletic Bilbao", "Real Sociedad", "Ath Bilbao", "Real Sociedad"
    )
    assert not teams_match("Levante", "Racing Santander", "Levante", "Getafe")


class StubMoviClient:
    def __init__(self, matches):
        self.matches = matches
        self.calls = []

    def board(self, league="Spain La Liga"):
        self.calls.append(league)
        return self.matches


def fixture_from_sample(**overrides) -> Fixture:
    data = {
        "event_id": "movi-1",
        "league": "La Liga",
        "home_team": "Levante",
        "away_team": "Racing Santander",
        "kickoff_utc": datetime(2026, 11, 27, 20, tzinfo=timezone.utc),
        "season_start": date(2026, 8, 15),
    }
    data.update(overrides)
    return Fixture(**data)


def test_moviodds_board_maps_bet365_quotes():
    sample = load("moviodds_match.json")
    provider = MoviOddsBet365Provider(StubMoviClient([sample]))
    fixture = fixture_from_sample()
    board = provider.fetch_board([fixture])
    assert set(board) == {"movi-1"}
    snapshot = board["movi-1"]
    assert (snapshot.home, snapshot.draw, snapshot.away) == (2.05, 3.6, 3.2)
    assert snapshot.source == "bet365"
    assert snapshot.observed_at is not None


def test_moviodds_requires_matching_kickoff_and_orientation():
    sample = load("moviodds_match.json")
    provider = MoviOddsBet365Provider(StubMoviClient([sample]))
    wrong_time = fixture_from_sample(
        kickoff_utc=datetime(2026, 11, 27, 21, tzinfo=timezone.utc)
    )
    reversed_teams = fixture_from_sample(
        home_team="Racing Santander", away_team="Levante"
    )
    assert provider.fetch_board([wrong_time]) == {}
    assert provider.fetch_board([reversed_teams]) == {}


def test_moviodds_skips_in_play_and_incomplete_quotes():
    sample = load("moviodds_match.json")
    in_play = json.loads(json.dumps(sample))
    in_play["in_play"] = True
    assert MoviOddsBet365Provider(
        StubMoviClient([in_play])
    ).fetch_board([fixture_from_sample()]) == {}

    incomplete = json.loads(json.dumps(sample))
    incomplete["quotes"] = incomplete["quotes"][:2]
    assert MoviOddsBet365Provider(
        StubMoviClient([incomplete])
    ).fetch_board([fixture_from_sample()]) == {}


def test_moviodds_client_accepts_board_and_single_match_shapes():
    sample = load("moviodds_match.json")
    assert MoviOddsClient._matches({"matches": [sample]}) == [sample]
    assert MoviOddsClient._matches(sample) == [sample]
    assert MoviOddsClient._matches([sample]) == [sample]
    assert MoviOddsClient._matches({"matches": []}) == []
