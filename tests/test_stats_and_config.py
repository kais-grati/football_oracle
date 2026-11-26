from __future__ import annotations

import json
from datetime import timedelta

import pytest

from papertrader import db
from papertrader.config import Settings, load_env_file
from papertrader.domain import FixtureResult, SnapshotAttempt
from papertrader.engine import DecisionEngine
from papertrader.settlement import SettlementService
from papertrader.stats import (
    all_stats,
    pending_table,
    source_stats,
    status_report,
    wallet_stats,
)

from .fakes import (
    FakeBet365,
    FakeConsensus,
    FakeModel,
    FakeNotifier,
    FakeResults,
    local_kickoff,
    make_fixture,
    make_snapshot,
)


def make_engine(tmp_path, *, fallback=False, model=None, expect="placed"):
    settings = Settings(
        db_path=tmp_path / "ledger.sqlite",
        model_path=tmp_path / "model.joblib",
    )
    conn = db.connect(settings.db_path)
    fixture = make_fixture(hour=14)
    db.upsert_fixtures(conn, [fixture])
    now = local_kickoff(13, 30)
    consensus_attempt = (
        SnapshotAttempt(None, "missing")
        if fallback
        else SnapshotAttempt(make_snapshot(observed_at=local_kickoff(13, 25)))
    )
    consensus = FakeConsensus({"1": consensus_attempt})
    board = FakeBet365({
        "1": make_snapshot("bet365", observed_at=local_kickoff(13, 29))
    })
    engine = DecisionEngine(
        conn, model or FakeModel(), consensus, settings, FakeNotifier()
    )
    outcome = engine.process(fixture, board.fetch_board([fixture]), now)
    if expect is not None:
        assert outcome.status == expect
    return conn, fixture


def test_wallet_stats_and_source_stats_after_settlement(tmp_path):
    conn, fixture = make_engine(tmp_path)
    service = SettlementService(
        conn,
        FakeResults([FixtureResult("1", "finished", 2, 0)]),
    )
    summary = service.settle_pending()
    assert summary["settled"] == 6

    table = all_stats(conn)
    conservative = table.set_index("Wallet").loc["Conservative"]
    assert conservative["Settled"] == 1
    assert conservative["W/L/V"] == "1/0/0"
    assert conservative["Hit %"] == 100.0
    assert conservative["Yield %"] > 0
    assert conservative["Pending"] == 0

    sources = source_stats(conn).set_index("Source")
    assert sources.loc["consensus", "Bets"] == 6
    assert sources.loc["consensus", "Decisions"] == 1
    assert sources.loc["consensus", "Placed decisions"] == 1
    assert sources.loc["consensus", "Won"] == 6

    assert pending_table(conn).empty
    report = status_report(conn, now=local_kickoff(13, 31))
    assert report["pending_bets"] == 0
    # The fixture has a decision, so it is no longer a next decision.
    assert all(
        item["event_id"] != "1" for item in report["next_decisions"]
    )
    conn.close()


def test_fallback_is_visible_in_source_stats(tmp_path):
    conn, _ = make_engine(tmp_path, fallback=True)
    sources = source_stats(conn).set_index("Source")
    assert "bet365_fallback" in sources.index
    assert sources.loc["bet365_fallback", "Bets"] == 6
    assert sources.loc["bet365_fallback", "Decisions"] == 1
    conn.close()


def test_no_bet_and_skipped_decisions_are_counted_by_source(tmp_path):
    conn, _ = make_engine(
        tmp_path, model=FakeModel(tree_pick=False), expect="no_bet"
    )
    sources = source_stats(conn).set_index("Source")
    assert sources.loc["consensus", "No bet"] == 1
    assert sources.loc["consensus", "Bets"] == 0

    # A fixture that misses its window has source none and status skipped.
    late = make_fixture("2", hour=16)
    db.upsert_fixtures(conn, [late])
    db.mark_missed(conn, local_kickoff(16, 0), 30, 10)
    sources = source_stats(conn).set_index("Source")
    assert sources.loc["none", "Skipped"] == 1
    conn.close()


def test_status_report_shows_next_decision_time(tmp_path):
    conn = db.connect(tmp_path / "ledger.sqlite")
    fixture = make_fixture("1", hour=14)
    db.upsert_fixtures(conn, [fixture])
    report = status_report(
        conn, now=local_kickoff(13, 0), lead_minutes=30
    )
    item = report["next_decisions"][0]
    assert item["event_id"] == "1"
    assert item["decision_at"].startswith("2026-11-28 12:30")
    conn.close()


def test_settings_from_env_and_repr_hides_secrets():
    settings = Settings.from_env({
        "BZZOIRO-API-KEY": "secret-bz",
        "ODDS-API-KEY": "secret-mo",
        "DISCORD_WEBHOOK_URL": "secret-hook",
        "PAPER_DECISION_LEAD_MINUTES": "25",
        "PAPER_FALLBACK_STAKE_MULTIPLIER": "0.4",
        "PAPER_RETRY_INTERVAL_SECONDS": "90",
        "PAPER_SETTLEMENT_INTERVAL_SECONDS": "60",
        "TZ": "Europe/Madrid",
    })
    assert settings.bzzoiro_api_key == "secret-bz"
    assert settings.moviodds_api_key == "secret-mo"
    assert settings.webhook_url == "secret-hook"
    assert settings.decision_lead_minutes == 25
    assert settings.fallback_stake_multiplier == 0.4
    assert settings.retry_interval_seconds == 90
    assert settings.settlement_interval_seconds == 60
    assert settings.decision_lead == timedelta(minutes=25)
    rendered = repr(settings)
    assert "secret-bz" not in rendered
    assert "secret-mo" not in rendered
    assert "secret-hook" not in rendered


def test_load_env_file_does_not_override(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "# comment\nA=one\nexport B=two\nC='three'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("A", raising=False)
    monkeypatch.delenv("B", raising=False)
    monkeypatch.delenv("C", raising=False)
    monkeypatch.setenv("A", "existing")
    load_env_file(path)
    import os

    assert os.environ["A"] == "existing"
    assert os.environ["B"] == "two"
    assert os.environ["C"] == "three"
