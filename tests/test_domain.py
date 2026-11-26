from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from papertrader.domain import (
    Fixture,
    FixtureResult,
    OddsSnapshot,
    ensure_utc,
    iso_utc,
    parse_utc,
)


def test_parse_and_serialize_utc():
    assert parse_utc("2026-11-28T20:00:00Z") == datetime(
        2026, 11, 28, 20, tzinfo=timezone.utc
    )
    assert parse_utc("2026-11-28T20:00:00+01:00") == datetime(
        2026, 11, 28, 19, tzinfo=timezone.utc
    )
    naive = parse_utc("2026-11-28T20:00:00")
    assert naive is not None and naive.tzinfo == timezone.utc
    assert parse_utc(None) is None
    assert parse_utc("not a date") is None
    assert iso_utc(naive) == "2026-11-28T20:00:00+00:00"


def test_ensure_utc_accepts_naive_and_aware():
    naive = ensure_utc(datetime(2026, 1, 1, 12))
    assert naive == datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    aware = ensure_utc(datetime(2026, 1, 1, 13, tzinfo=timezone.utc))
    assert aware == datetime(2026, 1, 1, 13, tzinfo=timezone.utc)


def test_fixture_decision_time_and_local_kickoff():
    fixture = Fixture(
        event_id=1,
        league="La Liga",
        home_team="H",
        away_team="A",
        kickoff_utc=datetime(2026, 11, 28, 20, tzinfo=timezone.utc),
        season_start=date(2026, 8, 15),
    )
    assert fixture.event_id == "1"
    assert fixture.decision_at(30) == datetime(
        2026, 11, 28, 19, 30, tzinfo=timezone.utc
    )
    assert fixture.local_kickoff("Europe/Madrid").hour == 21


def test_odds_snapshot_completeness_and_price():
    snapshot = OddsSnapshot("consensus", 1.5, 4.0, 7.0)
    assert snapshot.is_complete()
    assert snapshot.price("home") == 1.5
    assert snapshot.price("draw") == 4.0
    assert snapshot.price("unknown") is None
    assert not OddsSnapshot("consensus", 1.5, float("nan"), 7.0).is_complete()
    assert not OddsSnapshot("consensus", 1.5, 0.0, 7.0).is_complete()


@pytest.mark.parametrize(
    "status,home,away,expected",
    [
        ("finished", 2, 1, "home"),
        ("finished", 1, 1, "draw"),
        ("finished", 0, 3, "away"),
        ("postponed", 2, 1, None),
        ("notstarted", 2, 1, None),
        ("finished", None, 1, None),
    ],
)
def test_fixture_result_winner(status, home, away, expected):
    result = FixtureResult("1", status, home, away)
    assert result.winner() == expected


def test_fixture_result_void_flag():
    assert FixtureResult("1", "postponed").is_void
    assert FixtureResult("1", "cancelled").is_void
    assert not FixtureResult("1", "finished", 1, 0).is_void
