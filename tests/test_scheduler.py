from __future__ import annotations

from datetime import timedelta

import pytest

from papertrader import db
from papertrader.config import Settings
from papertrader.domain import FixtureResult, SnapshotAttempt
from papertrader.engine import DecisionEngine
from papertrader.scheduler import Scheduler
from papertrader.settlement import SettlementService

from .fakes import (
    FakeBet365,
    FakeClock,
    FakeConsensus,
    FakeFixtureProvider,
    FakeModel,
    FakeNotifier,
    FakeResults,
    local_kickoff,
    make_fixture,
    make_snapshot,
)


def build(tmp_path, fixtures, consensus, board, results=None, now=None):
    settings = Settings(
        db_path=tmp_path / "ledger.sqlite",
        model_path=tmp_path / "model.joblib",
        decision_lead_minutes=30,
        safety_margin_minutes=10,
        max_staleness_minutes=45,
        fallback_stake_multiplier=0.5,
    )
    conn = db.connect(settings.db_path)
    provider = FakeFixtureProvider(
        fixtures=fixtures, results=results or []
    )
    model = FakeModel()
    notifier = FakeNotifier()
    engine = DecisionEngine(conn, model, consensus, settings, notifier)
    settlement = SettlementService(conn, provider)
    clock = FakeClock(now or local_kickoff(13, 30))
    scheduler = Scheduler(
        settings, conn, engine, provider, board, settlement, clock
    )
    return scheduler, conn, provider, notifier


def test_matchday_decides_each_fixture_at_its_own_t_minus_30(tmp_path):
    first = make_fixture("1", hour=14)
    second = make_fixture("2", hour=16, minute=15)
    consensus = FakeConsensus({
        "1": SnapshotAttempt(make_snapshot(observed_at=local_kickoff(13, 25))),
        "2": SnapshotAttempt(make_snapshot(observed_at=local_kickoff(15, 40))),
    })
    board = FakeBet365({
        "1": make_snapshot("bet365", observed_at=local_kickoff(13, 29)),
        "2": make_snapshot("bet365", observed_at=local_kickoff(15, 44)),
    })
    scheduler, conn, _, notifier = build(
        tmp_path, [first, second], consensus, board
    )

    result = scheduler.tick(local_kickoff(13, 30))
    assert result.due_fixtures == 1
    assert result.processed == 1
    assert db.get_decision(conn, "1") is not None
    assert db.get_decision(conn, "2") is None
    assert len(notifier.calls) == 1

    # The second fixture is not due yet.
    result = scheduler.tick(local_kickoff(13, 31))
    assert result.due_fixtures == 0
    assert db.get_decision(conn, "2") is None

    result = scheduler.tick(local_kickoff(15, 45))
    assert result.due_fixtures == 1
    assert result.processed == 1
    assert db.get_decision(conn, "2") is not None
    assert len(notifier.calls) == 2
    conn.close()


def test_settlement_runs_before_the_next_decision(tmp_path):
    first = make_fixture("1", hour=14)
    second = make_fixture("2", hour=16, minute=15)
    consensus = FakeConsensus({
        "1": SnapshotAttempt(make_snapshot(observed_at=local_kickoff(13, 25))),
        "2": SnapshotAttempt(make_snapshot(observed_at=local_kickoff(15, 40))),
    })
    board = FakeBet365({
        "1": make_snapshot("bet365", observed_at=local_kickoff(13, 29)),
        "2": make_snapshot("bet365", observed_at=local_kickoff(15, 44)),
    })
    results = [
        FixtureResult("1", "finished", home_score=3, away_score=0),
    ]
    scheduler, conn, _, _ = build(
        tmp_path, [first, second], consensus, board, results=results
    )

    scheduler.tick(local_kickoff(13, 30))
    first_wallet = db.get_wallet(conn, "Conservative")
    assert first_wallet["pending_exposure"] > 0

    # At 15:45 the first result has arrived; settle it, then decide the 16:15.
    result = scheduler.tick(local_kickoff(15, 45))
    assert result.settled["settled"] == 6
    assert result.settled["profit"] > 0
    second_wallet = db.get_wallet(conn, "Conservative")
    assert second_wallet["bankroll"] > first_wallet["bankroll"]
    assert db.get_decision(conn, "2") is not None
    # The first bets are settled; only the second fixture is pending.
    pending = db.pending_bets(conn)
    assert {row["match_id"] for row in pending} == {"2"}
    conn.close()


def test_missed_window_is_recorded_without_a_decision(tmp_path):
    fixture = make_fixture("1", hour=14)
    scheduler, conn, _, _ = build(
        tmp_path,
        [fixture],
        FakeConsensus(),
        FakeBet365(),
    )
    result = scheduler.tick(local_kickoff(13, 55))
    assert result.due_fixtures == 0
    assert result.missed == 1
    decision = db.get_decision(conn, "1")
    assert decision["status"] == "skipped"
    assert decision["reason"] == "missed_window"
    conn.close()


def test_restart_does_not_duplicate_or_redecide(tmp_path):
    fixture = make_fixture("1", hour=14)
    consensus = FakeConsensus({
        "1": SnapshotAttempt(make_snapshot(observed_at=local_kickoff(13, 25))),
    })
    board = FakeBet365({
        "1": make_snapshot("bet365", observed_at=local_kickoff(13, 29)),
    })
    scheduler, conn, _, _ = build(tmp_path, [fixture], consensus, board)
    scheduler.tick(local_kickoff(13, 30))
    assert conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0] == 6

    # A fresh scheduler over the same ledger is a restart.  It must see the
    # decision and place nothing.
    scheduler2 = Scheduler(
        scheduler.settings,
        conn,
        scheduler.engine,
        scheduler.fixtures,
        scheduler.bet365,
        scheduler.settlement,
        FakeClock(local_kickoff(13, 31)),
    )
    result = scheduler2.tick(local_kickoff(13, 31))
    assert result.due_fixtures == 0
    assert conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0] == 6
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    conn.close()


def test_fixture_sync_is_throttled_then_refreshed_by_day(tmp_path):
    fixture = make_fixture("1", hour=14)
    scheduler, conn, provider, _ = build(
        tmp_path,
        [fixture],
        FakeConsensus(),
        FakeBet365(),
    )
    scheduler.tick(local_kickoff(13, 30))
    assert len(provider.fixture_calls) == 1

    scheduler.tick(local_kickoff(13, 31))
    assert len(provider.fixture_calls) == 1

    # Six hours later the refresh interval has elapsed.
    scheduler.tick(local_kickoff(19, 31))
    assert len(provider.fixture_calls) == 2

    # A new local day forces a refresh even inside the interval.
    scheduler.tick(local_kickoff(0, 1, day=29))
    assert len(provider.fixture_calls) == 3
    conn.close()


def test_retries_are_backed_off_between_ticks(tmp_path):
    fixture = make_fixture("1", hour=14)
    consensus = FakeConsensus()
    scheduler, conn, _, _ = build(
        tmp_path, [fixture], consensus, FakeBet365()
    )
    first = scheduler.tick(local_kickoff(13, 30))
    assert first.retries == 1
    assert consensus.calls == ["1"]

    second = scheduler.tick(local_kickoff(13, 31))
    assert second.due_fixtures == 1
    assert second.deferred == 1
    assert consensus.calls == ["1"]

    third = scheduler.tick(local_kickoff(13, 32, second=1))
    assert third.retries == 1
    assert consensus.calls == ["1", "1"]
    conn.close()


def test_bet365_board_failure_does_not_crash_other_work(tmp_path):
    class BrokenBoard:
        def fetch_board(self, fixtures):
            raise RuntimeError("board down")

    fixture = make_fixture("1", hour=14)
    consensus = FakeConsensus({
        "1": SnapshotAttempt(make_snapshot(observed_at=local_kickoff(13, 25))),
    })
    scheduler, conn, _, _ = build(
        tmp_path, [fixture], consensus, BrokenBoard()
    )
    result = scheduler.tick(local_kickoff(13, 30))
    assert result.errors
    assert any("bet365_board" in error for error in result.errors)
    assert db.get_decision(conn, "1") is None  # retry, not a fake bet
    conn.close()


def test_settlement_is_throttled_between_ticks(tmp_path):
    fixture = make_fixture("1", hour=14)
    consensus = FakeConsensus({
        "1": SnapshotAttempt(make_snapshot(observed_at=local_kickoff(13, 25))),
    })
    board = FakeBet365({
        "1": make_snapshot("bet365", observed_at=local_kickoff(13, 29)),
    })
    results = [FixtureResult("1", "finished", 2, 0)]
    scheduler, conn, provider, _ = build(
        tmp_path, [fixture], consensus, board, results=results
    )
    scheduler.tick(local_kickoff(13, 30))
    scheduler.tick(local_kickoff(13, 31))
    assert provider.result_calls == []  # deferred, and no result call yet
    scheduler.tick(local_kickoff(13, 35))
    assert len(provider.result_calls) == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM bets WHERE status = 'pending'"
    ).fetchone()[0] == 0
    conn.close()


def test_slow_board_call_cannot_place_after_cutoff(tmp_path):
    class SlowBoard(FakeBet365):
        def __init__(self, clock, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.clock = clock

        def fetch_board(self, fixtures):
            self.clock.advance(20 * 60)
            return super().fetch_board(fixtures)

    fixture = make_fixture("1", hour=14)
    consensus = FakeConsensus({
        "1": SnapshotAttempt(make_snapshot(observed_at=local_kickoff(13, 25))),
    })
    clock = FakeClock(local_kickoff(13, 30))
    board = SlowBoard(
        clock,
        snapshots={
            "1": make_snapshot("bet365", observed_at=local_kickoff(13, 29))
        },
    )
    settings = Settings(db_path=tmp_path / "ledger.sqlite")
    conn = db.connect(settings.db_path)
    provider = FakeFixtureProvider(fixtures=[fixture])
    engine = DecisionEngine(
        conn, FakeModel(), consensus, settings, FakeNotifier()
    )
    scheduler = Scheduler(
        settings, conn, engine, provider, board,
        SettlementService(conn, provider), clock,
    )
    result = scheduler.tick()
    assert result.processed == 0
    assert result.skipped == 1
    assert conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0] == 0
    assert db.get_decision(conn, "1")["status"] == "skipped"
    conn.close()


def test_settlement_void_and_closed_bet(tmp_path):
    fixture = make_fixture("1", hour=14)
    consensus = FakeConsensus({
        "1": SnapshotAttempt(make_snapshot(observed_at=local_kickoff(13, 25))),
    })
    board = FakeBet365({
        "1": make_snapshot("bet365", observed_at=local_kickoff(13, 29)),
    })
    results = [FixtureResult("1", "postponed")]
    scheduler, conn, _, _ = build(
        tmp_path, [fixture], consensus, board, results=results
    )
    scheduler.tick(local_kickoff(13, 30))
    result = scheduler.tick(local_kickoff(15, 0))
    assert result.settled["void"] == 6
    assert all(
        row["status"] == "void"
        for row in conn.execute("SELECT status FROM bets")
    )
    conn.close()
