from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from papertrader import db
from papertrader.config import Settings
from papertrader.domain import FixtureResult, SnapshotAttempt
from papertrader.engine import DecisionEngine
from papertrader.model import train_market_model
from papertrader.providers.base import ProviderError
from papertrader.providers.bzzoiro import BzzoiroFixtureProvider
from papertrader.providers.moviodds import MoviOddsClient
from papertrader.scheduler import Scheduler
from papertrader.settlement import SettlementService

from .fakes import (
    FakeBet365,
    FakeClock,
    FakeConsensus,
    FakeFixtureProvider,
    FakeModel,
    FakeNotifier,
    FakeResponse,
    FakeResults,
    FakeSession,
    local_kickoff,
    make_fixture,
    make_snapshot,
    no_sleep,
)


def engine_for(tmp_path, consensus_odds, board_odds, kickoff_hour=14):
    settings = Settings(
        db_path=tmp_path / "ledger.sqlite",
        model_path=tmp_path / "model.joblib",
    )
    conn = db.connect(settings.db_path)
    fixture = make_fixture(hour=kickoff_hour)
    db.upsert_fixtures(conn, [fixture])
    now = local_kickoff(kickoff_hour - 1, 30)
    consensus = FakeConsensus({
        "1": SnapshotAttempt(make_snapshot(
            home=consensus_odds[0], draw=consensus_odds[1],
            away=consensus_odds[2], observed_at=now - timedelta(minutes=5),
        ))
    })
    board = FakeBet365({
        "1": make_snapshot(
            "bet365", home=board_odds[0], draw=board_odds[1],
            away=board_odds[2], observed_at=now - timedelta(minutes=1),
        )
    })
    engine = DecisionEngine(
        conn, FakeModel(), consensus, settings, FakeNotifier()
    )
    return engine, conn, fixture, board, now


@pytest.mark.parametrize(
    "odds,board_odds,selection,execution",
    [
        ((3.50, 2.80, 3.20), (3.50, 2.90, 3.20), "draw", 2.90),
        ((5.00, 4.00, 1.50), (5.00, 4.00, 1.55), "away", 1.55),
    ],
)
def test_engine_handles_draw_and_away_favorites(
    tmp_path, odds, board_odds, selection, execution
):
    engine, conn, fixture, board, now = engine_for(
        tmp_path, odds, board_odds
    )
    outcome = engine.process(fixture, board.fetch_board([fixture]), now)
    assert outcome.status == "placed"
    assert outcome.selection == selection
    decision = db.get_decision(conn, "1")
    assert decision["selection"] == selection
    assert decision["execution_odds"] == execution
    assert conn.execute(
        "SELECT COUNT(*) FROM bets WHERE selection = ?", (selection,)
    ).fetchone()[0] == 6
    conn.close()


def test_two_fixtures_at_same_decision_time_share_one_board_request(tmp_path):
    settings = Settings(db_path=tmp_path / "ledger.sqlite")
    conn = db.connect(settings.db_path)
    fixtures = [make_fixture("1", hour=14), make_fixture("2", hour=14)]
    db.upsert_fixtures(conn, fixtures)
    now = local_kickoff(13, 30)
    consensus = FakeConsensus({
        fixture.event_id: SnapshotAttempt(make_snapshot(
            observed_at=local_kickoff(13, 25)
        ))
        for fixture in fixtures
    })
    board = FakeBet365({
        fixture.event_id: make_snapshot(
            "bet365", observed_at=local_kickoff(13, 29)
        )
        for fixture in fixtures
    })
    provider = FakeFixtureProvider(fixtures=fixtures)
    notifier = FakeNotifier()
    engine = DecisionEngine(
        conn, FakeModel(), consensus, settings, notifier
    )
    scheduler = Scheduler(
        settings, conn, engine, provider, board,
        SettlementService(conn, provider), FakeClock(now),
    )

    result = scheduler.tick(now)
    assert result.due_fixtures == 2
    assert result.processed == 2
    assert len(board.calls) == 1
    assert sorted(board.calls[0]) == ["1", "2"]
    assert len(notifier.calls) == 2
    assert conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0] == 12
    for wallet in db.wallet_rows(conn):
        assert wallet["available"] >= 0
        assert wallet["pending_exposure"] <= wallet["bankroll"]
    conn.close()


def test_lost_bets_reduce_bankroll(tmp_path):
    engine, conn, fixture, board, now = engine_for(
        tmp_path, (1.6, 4.2, 5.5), (1.6, 4.2, 5.5)
    )
    engine.process(fixture, board.fetch_board([fixture]), now)
    before = db.get_wallet(conn, "Conservative")
    service = SettlementService(
        conn, FakeResults([FixtureResult("1", "finished", 0, 2)])
    )
    summary = service.settle_pending()
    assert summary["profit"] < 0
    after = db.get_wallet(conn, "Conservative")
    assert after["bankroll"] < before["bankroll"]
    statuses = {
        row["status"] for row in conn.execute(
            "SELECT status FROM bets"
        )
    }
    assert statuses == {"lost"}
    conn.close()


def test_postponed_fixture_status_is_synced_and_not_decided(tmp_path):
    settings = Settings(db_path=tmp_path / "ledger.sqlite")
    conn = db.connect(settings.db_path)
    postponed = make_fixture("1", hour=14)
    postponed = type(postponed)(
        **{**postponed.__dict__, "status": "postponed"}
    )
    provider = FakeFixtureProvider(fixtures=[postponed])
    consensus = FakeConsensus()
    board = FakeBet365()
    engine = DecisionEngine(
        conn, FakeModel(), consensus, settings, FakeNotifier()
    )
    scheduler = Scheduler(
        settings, conn, engine, provider, board,
        SettlementService(conn, provider),
        FakeClock(local_kickoff(13, 30)),
    )
    result = scheduler.tick(local_kickoff(13, 30))
    assert result.due_fixtures == 0
    assert db.get_fixture(conn, "1").status == "postponed"
    assert db.get_decision(conn, "1") is None
    conn.close()


def test_exact_safety_cutoff_is_closed(tmp_path):
    conn = db.connect(tmp_path / "ledger.sqlite")
    fixture = make_fixture("1", hour=14)
    db.upsert_fixtures(conn, [fixture])
    assert db.due_fixtures(
        conn, local_kickoff(13, 49, second=59), 30, 10
    )
    assert db.due_fixtures(conn, local_kickoff(13, 50), 30, 10) == []
    assert db.mark_missed(conn, local_kickoff(13, 50), 30, 10) == 1
    conn.close()


def test_model_rejects_single_class_target(tmp_path):
    from .test_model_and_sizing import synthetic_history

    data = synthetic_history(30)
    data["winner"] = "home"
    with pytest.raises(ValueError):
        train_market_model(tmp_path / "one.joblib", data)


class RecordingBzzoiro:
    def __init__(self):
        self.event_calls = []
        self.season_calls = []

    def events(self, **kwargs):
        self.event_calls.append(kwargs)
        return [{
            "id": 1,
            "event_date": "2026-11-27T20:00:00+00:00",
            "status": "notstarted",
            "season_id": 1307,
            "home_team": "A",
            "away_team": "B",
        }]

    def season_start(self, *args, **kwargs):
        self.season_calls.append((args, kwargs))
        return date(2026, 8, 15)


def test_include_all_queries_unfiltered_status():
    client = RecordingBzzoiro()
    provider = BzzoiroFixtureProvider(client)
    provider.fetch_fixtures(date(2026, 11, 27), date(2026, 11, 27),
                            include_all=True)
    assert client.event_calls[0]["status"] is None
    assert len(client.season_calls) == 1

    client2 = RecordingBzzoiro()
    BzzoiroFixtureProvider(client2).fetch_fixtures(
        date(2026, 11, 27), date(2026, 11, 27)
    )
    assert client2.event_calls[0]["status"] == "notstarted"


def test_moviodds_retries_429_and_rejects_http_error():
    session = FakeSession([
        FakeResponse(429, {}, {"Retry-After": "0"}),
        FakeResponse(200, {"matches": []}),
    ])
    client = MoviOddsClient("key", session=session, sleep=no_sleep)
    assert client.board("Spain La Liga") == []
    assert len(session.calls) == 2

    session = FakeSession([FakeResponse(400, {"detail": "bad"})])
    client = MoviOddsClient("key", session=session, sleep=no_sleep)
    with pytest.raises(ProviderError):
        client.board("Spain La Liga")
