from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from papertrader import db
from papertrader.config import Settings
from papertrader.domain import SnapshotAttempt
from papertrader.engine import DecisionEngine

from .fakes import (
    FakeBet365,
    FakeConsensus,
    FakeModel,
    FakeNotifier,
    local_kickoff,
    make_fixture,
    make_snapshot,
)


def setup_engine(tmp_path, consensus, board, model=None, notifier=None):
    settings = Settings(
        db_path=tmp_path / "ledger.sqlite",
        model_path=tmp_path / "model.joblib",
        decision_lead_minutes=30,
        safety_margin_minutes=10,
        max_staleness_minutes=45,
        fallback_stake_multiplier=0.5,
    )
    conn = db.connect(settings.db_path)
    fixture = make_fixture()
    db.upsert_fixtures(conn, [fixture])
    engine = DecisionEngine(
        conn,
        model or FakeModel(),
        consensus,
        settings,
        notifier or FakeNotifier(),
    )
    return engine, conn, fixture, settings


def test_consensus_decision_places_six_wallet_bets(tmp_path):
    now = local_kickoff(13, 30)
    consensus = FakeConsensus({
        "1": SnapshotAttempt(
            make_snapshot(observed_at=local_kickoff(13, 25)),
        )
    })
    board = FakeBet365({
        "1": make_snapshot("bet365", home=1.70, draw=4.20, away=5.80,
                            observed_at=local_kickoff(13, 29)),
    })
    notifier = FakeNotifier()
    engine, conn, fixture, _ = setup_engine(
        tmp_path, consensus, board, notifier=notifier
    )

    outcome = engine.process(fixture, board.fetch_board([fixture]), now)

    assert outcome.status == "placed"
    assert outcome.recorded is True
    assert outcome.odds_source == "consensus"
    assert outcome.selection == "home"
    assert len(outcome.bets) == 6
    decision = db.get_decision(conn, "1")
    assert decision["execution_odds"] == 1.70
    assert decision["home_odds"] == 1.60  # consensus feature price
    assert decision["actual_lead_seconds"] == 1800
    assert decision["staleness_seconds"] == 300
    assert conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0] == 6
    assert conn.execute(
        "SELECT COUNT(*) FROM bets WHERE discord_sent = 1"
    ).fetchone()[0] == 6
    assert len(notifier.calls) == 1
    assert notifier.calls[0][0]["odds_source"] == "consensus"
    assert len(notifier.calls[0][1]) == 6
    sources = {
        row["source"]: row["n"] for row in conn.execute(
            "SELECT source, COUNT(*) AS n FROM snapshots GROUP BY source"
        )
    }
    assert sources == {"consensus": 1, "bet365": 1}
    conn.close()


def test_incomplete_consensus_falls_back(tmp_path):
    now = local_kickoff(13, 30)
    consensus = FakeConsensus({
        "1": SnapshotAttempt(
            make_snapshot(observed_at=local_kickoff(13, 25), draw=0.0),
            "incomplete_outcomes",
        )
    })
    board = FakeBet365({
        "1": make_snapshot("bet365", observed_at=local_kickoff(13, 29))
    })
    engine, conn, fixture, _ = setup_engine(tmp_path, consensus, board)
    outcome = engine.process(fixture, board.fetch_board([fixture]), now)
    assert outcome.status == "placed"
    assert outcome.odds_source == "bet365_fallback"
    assert db.get_decision(conn, "1")["fallback_reason"] == (
        "incomplete_outcomes"
    )
    snapshots = {
        row["source"]: row["n"] for row in conn.execute(
            "SELECT source, COUNT(*) AS n FROM snapshots GROUP BY source"
        )
    }
    # The invalid consensus attempt is archived too; the decision still uses
    # the fresh Bet365 fallback.
    assert snapshots == {"consensus": 1, "bet365": 1}
    conn.close()


def test_fallback_uses_bet365_features_and_half_stake(tmp_path):
    now = local_kickoff(13, 30)
    consensus = FakeConsensus({"1": SnapshotAttempt(None, "stale")})
    bet365 = make_snapshot(
        "bet365", home=1.60, draw=4.20, away=5.50,
        observed_at=local_kickoff(13, 29),
    )
    board = FakeBet365({"1": bet365})
    notifier = FakeNotifier()
    engine, conn, fixture, _ = setup_engine(
        tmp_path, consensus, board, notifier=notifier
    )

    outcome = engine.process(fixture, board.fetch_board([fixture]), now)

    assert outcome.status == "placed"
    assert outcome.odds_source == "bet365_fallback"
    decision = db.get_decision(conn, "1")
    assert decision["fallback_reason"] == "stale"
    assert decision["fallback_snapshot_at"] is not None
    assert decision["selection"] == "home"

    # The conservative wallet pays 1% below the 0.65 crossing and 2% above,
    # so confidence 0.598 -> 0.92% of 100; a 50% fallback haircut -> 0.46.
    conservative = conn.execute(
        "SELECT stake FROM bets b JOIN wallets w ON w.id = b.wallet_id "
        "WHERE w.name = 'Conservative'"
    ).fetchone()["stake"]
    assert conservative == pytest.approx(0.46, abs=0.01)
    assert notifier.calls[0][0]["odds_source"] == "bet365_fallback"
    assert "stale" in notifier.calls[0][0]["fallback_reason"]
    conn.close()


def test_stale_consensus_falls_back_to_fresh_bet365(tmp_path):
    now = local_kickoff(13, 30)
    stale = make_snapshot(observed_at=now - timedelta(minutes=60))
    consensus = FakeConsensus({"1": SnapshotAttempt(stale)})
    board = FakeBet365({
        "1": make_snapshot("bet365", observed_at=local_kickoff(13, 29))
    })
    engine, conn, fixture, _ = setup_engine(tmp_path, consensus, board)

    outcome = engine.process(fixture, board.fetch_board([fixture]), now)
    assert outcome.odds_source == "bet365_fallback"
    decision = db.get_decision(conn, "1")
    assert decision["fallback_reason"].startswith("stale_snapshot")
    conn.close()


def test_missing_everything_retries_then_skips_at_cutoff(tmp_path):
    now = local_kickoff(13, 30)
    engine, conn, fixture, _ = setup_engine(
        tmp_path, FakeConsensus(), FakeBet365()
    )
    outcome = engine.process(fixture, {}, now)
    assert outcome.status == "retry"
    assert outcome.recorded is False
    assert db.get_decision(conn, "1") is None

    late = local_kickoff(13, 51)
    outcome = engine.process(fixture, {}, late)
    assert outcome.status == "skipped"
    assert outcome.recorded is True
    decision = db.get_decision(conn, "1")
    assert decision["status"] == "skipped"
    assert decision["odds_source"] == "none"
    assert conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0] == 0
    conn.close()


def test_models_disagree_records_no_bet_and_does_not_notify(tmp_path):
    now = local_kickoff(13, 30)
    consensus = FakeConsensus({
        "1": SnapshotAttempt(make_snapshot(observed_at=local_kickoff(13, 25)))
    })
    board = FakeBet365({
        "1": make_snapshot("bet365", observed_at=local_kickoff(13, 29))
    })
    notifier = FakeNotifier()
    engine, conn, fixture, _ = setup_engine(
        tmp_path, consensus, board,
        model=FakeModel(logreg_pick=True, tree_pick=False),
        notifier=notifier,
    )
    outcome = engine.process(fixture, board.fetch_board([fixture]), now)
    assert outcome.status == "no_bet"
    assert outcome.recorded is True
    assert outcome.reason == "models_disagree"
    assert conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0] == 0
    assert notifier.calls == []
    assert db.get_decision(conn, "1")["p_logreg"] is not None
    conn.close()


def test_execution_missing_is_retried_even_with_good_consensus(tmp_path):
    now = local_kickoff(13, 30)
    consensus = FakeConsensus({
        "1": SnapshotAttempt(make_snapshot(observed_at=local_kickoff(13, 25)))
    })
    engine, conn, fixture, _ = setup_engine(
        tmp_path, consensus, FakeBet365()
    )
    outcome = engine.process(fixture, {}, now)
    assert outcome.status == "retry"
    assert "execution" in (outcome.reason or "")
    assert db.get_decision(conn, "1") is None
    conn.close()


def test_missing_season_start_retries_then_skips(tmp_path):
    now = local_kickoff(13, 30)
    consensus = FakeConsensus({
        "1": SnapshotAttempt(make_snapshot(observed_at=local_kickoff(13, 25)))
    })
    board = FakeBet365({
        "1": make_snapshot("bet365", observed_at=local_kickoff(13, 29))
    })
    engine, conn, fixture, _ = setup_engine(tmp_path, consensus, board)
    fixture = replace(fixture, season_start=None)
    outcome = engine.process(fixture, board.fetch_board([fixture]), now)
    assert outcome.status == "retry"
    assert "season_start" in (outcome.reason or "")

    outcome = engine.process(
        fixture, board.fetch_board([fixture]), local_kickoff(13, 51)
    )
    assert outcome.status == "skipped"
    assert db.get_decision(conn, "1")["status"] == "skipped"
    conn.close()


def test_after_cutoff_never_places_a_bet(tmp_path):
    now = local_kickoff(13, 55)
    consensus = FakeConsensus({
        "1": SnapshotAttempt(make_snapshot(observed_at=now))
    })
    board = FakeBet365({
        "1": make_snapshot("bet365", observed_at=now)
    })
    notifier = FakeNotifier()
    engine, conn, fixture, _ = setup_engine(
        tmp_path, consensus, board, notifier=notifier
    )
    outcome = engine.process(fixture, board.fetch_board([fixture]), now)
    assert outcome.status == "skipped"
    assert "outside_cutoff" in (db.get_decision(conn, "1")["reason"] or "")
    assert conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0] == 0
    assert notifier.calls == []
    conn.close()


def test_stale_execution_is_not_used(tmp_path):
    now = local_kickoff(13, 30)
    consensus = FakeConsensus({
        "1": SnapshotAttempt(make_snapshot(observed_at=local_kickoff(13, 25)))
    })
    board = FakeBet365({
        "1": make_snapshot(
            "bet365", observed_at=now - timedelta(minutes=60)
        )
    })
    engine, conn, fixture, _ = setup_engine(tmp_path, consensus, board)
    outcome = engine.process(fixture, board.fetch_board([fixture]), now)
    assert outcome.status == "retry"
    assert "execution_stale_snapshot" in (outcome.reason or "")
    assert db.get_decision(conn, "1") is None
    conn.close()


def test_processing_twice_is_idempotent(tmp_path):
    now = local_kickoff(13, 30)
    consensus = FakeConsensus({
        "1": SnapshotAttempt(make_snapshot(observed_at=local_kickoff(13, 25)))
    })
    board = FakeBet365({
        "1": make_snapshot("bet365", observed_at=local_kickoff(13, 29))
    })
    engine, conn, fixture, _ = setup_engine(tmp_path, consensus, board)
    snapshots = board.fetch_board([fixture])
    first = engine.process(fixture, snapshots, now)
    second = engine.process(fixture, snapshots, now)
    assert first.status == "placed"
    assert second.status == "already_decided"
    assert conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0] == 6
    conn.close()


def test_consensus_provider_exception_is_isolated(tmp_path):
    now = local_kickoff(13, 30)
    consensus = FakeConsensus({"1": RuntimeError("network down")})
    board = FakeBet365({
        "1": make_snapshot("bet365", observed_at=local_kickoff(13, 29))
    })
    engine, conn, fixture, _ = setup_engine(tmp_path, consensus, board)
    outcome = engine.process(fixture, board.fetch_board([fixture]), now)
    assert outcome.status == "placed"
    assert outcome.odds_source == "bet365_fallback"
    assert "provider_error" in (
        db.get_decision(conn, "1")["fallback_reason"] or ""
    )
    conn.close()


def test_notifier_failure_still_records_bets(tmp_path):
    now = local_kickoff(13, 30)
    consensus = FakeConsensus({
        "1": SnapshotAttempt(make_snapshot(observed_at=local_kickoff(13, 25)))
    })
    board = FakeBet365({
        "1": make_snapshot("bet365", observed_at=local_kickoff(13, 29))
    })
    notifier = FakeNotifier(result=False)
    engine, conn, fixture, _ = setup_engine(
        tmp_path, consensus, board, notifier=notifier
    )
    outcome = engine.process(fixture, board.fetch_board([fixture]), now)
    assert outcome.status == "placed"
    assert conn.execute(
        "SELECT COUNT(*) FROM bets WHERE discord_sent = 1"
    ).fetchone()[0] == 0
    assert len(notifier.calls) == 1
    conn.close()
