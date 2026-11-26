from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from papertrader import db
from papertrader.domain import FixtureResult, OddsSnapshot

from .fakes import local_kickoff, make_fixture, make_snapshot


def test_connect_creates_six_wallets(tmp_path):
    conn = db.connect(tmp_path / "ledger.sqlite")
    wallets = db.wallet_rows(conn)
    assert [wallet["name"] for wallet in wallets] == [
        "Conservative", "Moderate", "Medium", "Medium-High", "High",
        "Aggressive",
    ]
    assert all(wallet["available"] == 100.0 for wallet in wallets)
    conn.close()


def test_due_window_and_missed_window(tmp_path):
    conn = db.connect(tmp_path / "ledger.sqlite")
    fixture = make_fixture(hour=14)
    db.upsert_fixtures(conn, [fixture])

    before = local_kickoff(13, 29)
    at_decision = local_kickoff(13, 30)
    after_cutoff = local_kickoff(13, 51)

    assert db.due_fixtures(conn, before, 30, 10) == []
    assert [f.event_id for f in db.due_fixtures(conn, at_decision, 30, 10)] == [
        "1"
    ]
    assert db.due_fixtures(conn, after_cutoff, 30, 10) == []
    assert db.past_cutoff_fixtures(conn, after_cutoff, 30, 10)[0].event_id == "1"

    assert db.mark_missed(conn, after_cutoff, 30, 10) == 1
    decision = db.get_decision(conn, "1")
    assert decision["status"] == "skipped"
    assert decision["reason"] == "missed_window"
    # Repeating the tick never duplicates the skipped decision.
    assert db.mark_missed(conn, after_cutoff, 30, 10) == 0
    conn.close()


def test_decision_and_bets_are_idempotent(tmp_path):
    conn = db.connect(tmp_path / "ledger.sqlite")
    fixture = make_fixture()
    db.upsert_fixtures(conn, [fixture])
    wallet = db.wallet_rows(conn)[0]
    decision = {
        "fixture_id": fixture.event_id,
        "created_at": "2026-11-25T12:30:00+00:00",
        "decision_at_utc": "2026-11-25T12:30:00+00:00",
        "odds_source": "consensus",
        "status": "placed",
    }
    bet = {
        "wallet_id": wallet["id"],
        "created_at": "2026-11-25T12:30:00+00:00",
        "match_id": fixture.event_id,
        "league": fixture.league,
        "kickoff": "2026-11-28T14:00+01:00",
        "home_team": fixture.home_team,
        "away_team": fixture.away_team,
        "selection": "home",
        "odds_taken": 1.8,
        "stake": 2.0,
        "status": "pending",
    }
    first = db.insert_decision_and_bets(conn, decision, [bet])
    second = db.insert_decision_and_bets(conn, decision, [bet])
    assert first is not None
    assert second is None
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0] == 1

    wallet_after = db.get_wallet(conn, wallet["name"])
    assert wallet_after["pending_exposure"] == 2.0
    assert wallet_after["available"] == 98.0
    conn.close()


def test_upsert_updates_scores_and_status(tmp_path):
    conn = db.connect(tmp_path / "ledger.sqlite")
    fixture = make_fixture()
    db.upsert_fixtures(conn, [fixture])
    updated = FixtureResult(event_id="1", status="finished",
                            home_score=2, away_score=1)
    from dataclasses import replace

    db.upsert_fixtures(
        conn,
        [replace(fixture, status="finished", home_score=2, away_score=1)],
    )
    stored = db.get_fixture(conn, "1")
    assert stored.status == "finished"
    assert stored.home_score == 2
    assert stored.away_score == 1
    assert updated.winner() == "home"
    conn.close()


def test_snapshot_archive(tmp_path):
    conn = db.connect(tmp_path / "ledger.sqlite")
    fixture = make_fixture()
    db.upsert_fixtures(conn, [fixture])
    snapshot = make_snapshot(observed_at=local_kickoff(13, 25))
    db.insert_snapshot(conn, fixture.event_id, snapshot)
    row = conn.execute("SELECT * FROM snapshots").fetchone()
    assert row["fixture_id"] == "1"
    assert row["source"] == "consensus"
    assert row["home_odds"] == snapshot.home
    assert row["raw"]
    conn.close()


def test_single_wallet_ledger_migration(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE state (
            id INTEGER PRIMARY KEY,
            starting_capital REAL NOT NULL,
            bankroll REAL NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            match_id TEXT NOT NULL,
            league TEXT,
            kickoff TEXT,
            home_team TEXT,
            away_team TEXT,
            selection TEXT NOT NULL,
            odds_taken REAL NOT NULL,
            bookmaker TEXT,
            confidence REAL,
            p_logreg REAL,
            p_tree REAL,
            stake REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            closing_odds REAL,
            clv REAL,
            result TEXT,
            profit REAL,
            settled_at TEXT,
            model_version TEXT,
            discord_sent INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO state VALUES (1, 50, 55, '2026-01-01T00:00:00+00:00');
        INSERT INTO bets (
            created_at, match_id, selection, odds_taken, stake, status
        ) VALUES ('2026-01-02T00:00:00+00:00', 'old-1', 'home', 2.0, 5,
                  'pending');
        """
    )
    conn.commit()
    conn.close()

    migrated = db.connect(path)
    wallet = db.get_wallet(migrated, "Conservative")
    assert wallet["starting_capital"] == 50.0
    assert wallet["bankroll"] == 55.0
    bet = migrated.execute("SELECT * FROM bets").fetchone()
    assert bet["match_id"] == "old-1"
    assert bet["wallet_id"] == wallet["id"]
    assert "decision_id" in {
        row["name"] for row in migrated.execute("PRAGMA table_info(bets)")
    }
    migrated.close()


def test_multi_wallet_ledger_gains_decision_id(tmp_path):
    path = tmp_path / "multi.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE wallets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            profile TEXT NOT NULL,
            starting_capital REAL NOT NULL,
            bankroll REAL NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            match_id TEXT NOT NULL,
            selection TEXT NOT NULL,
            odds_taken REAL NOT NULL,
            stake REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            UNIQUE(wallet_id, match_id)
        );
        INSERT INTO wallets (name, profile, starting_capital, bankroll,
            created_at) VALUES ('Conservative', 'Conservative', 100, 100,
            '2026-01-01T00:00:00+00:00');
        """
    )
    conn.commit()
    conn.close()

    migrated = db.connect(path)
    columns = {
        row["name"] for row in migrated.execute("PRAGMA table_info(bets)")
    }
    assert "decision_id" in columns
    assert len(db.wallet_rows(migrated)) == 6
    migrated.close()
