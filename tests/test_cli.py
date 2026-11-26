from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from papertrader import db
from papertrader.cli import main
from papertrader.domain import SnapshotAttempt
from papertrader.engine import DecisionEngine
from papertrader.config import Settings

from .fakes import (
    FakeBet365,
    FakeConsensus,
    FakeModel,
    FakeNotifier,
    local_kickoff,
    make_fixture,
    make_snapshot,
)


def seed_db(tmp_path):
    settings = Settings(db_path=tmp_path / "ledger.sqlite")
    conn = db.connect(settings.db_path)
    fixture = make_fixture(hour=14)
    db.upsert_fixtures(conn, [fixture])
    consensus = FakeConsensus({
        "1": SnapshotAttempt(make_snapshot(observed_at=local_kickoff(13, 25)))
    })
    board = FakeBet365({
        "1": make_snapshot("bet365", observed_at=local_kickoff(13, 29))
    })
    engine = DecisionEngine(
        conn, FakeModel(), consensus, settings, FakeNotifier()
    )
    engine.process(fixture, board.fetch_board([fixture]), local_kickoff(13, 30))
    conn.close()
    return settings.db_path


def test_cli_stats_json(tmp_path, capsys):
    path = seed_db(tmp_path)
    main(["--env", "/nonexistent", "stats", "--db", str(path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["wallets"]) == 6
    assert payload["wallets"][0]["Wallet"] == "Conservative"
    assert payload["wallets"][0]["Pending"] == 1
    assert payload["sources"][0]["Source"] == "consensus"
    assert len(payload["pending"]) == 6


def test_cli_stats_wallet_detail(tmp_path, capsys):
    path = seed_db(tmp_path)
    main([
        "--env", "/nonexistent", "stats", "--db", str(path),
        "--wallet", "Conservative", "--json",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert payload["wallet"] == "Conservative"
    assert payload["pending_bets"] == 1
    assert payload["bankroll"] == 100.0


def test_cli_stats_unknown_wallet_exits(tmp_path):
    path = seed_db(tmp_path)
    with pytest.raises(SystemExit):
        main([
            "--env", "/nonexistent", "stats", "--db", str(path),
            "--wallet", "Nope",
        ])


def test_cli_status_json(tmp_path, capsys):
    path = seed_db(tmp_path)
    main(["--env", "/nonexistent", "status", "--db", str(path)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["pending_bets"] == 6
    assert payload["open_exposure"] > 0
    assert "next_decisions" in payload


def test_cli_status_shows_next_decision(tmp_path, capsys):
    path = tmp_path / "ledger.sqlite"
    conn = db.connect(path)
    fixture = make_fixture(hour=14)
    fixture = fixture.__class__(
        **{**fixture.__dict__,
           "kickoff_utc": datetime(2030, 11, 28, 13, tzinfo=timezone.utc)}
    )
    db.upsert_fixtures(conn, [fixture])
    conn.close()
    main(["--env", "/nonexistent", "status", "--db", str(path)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["next_decisions"][0]["event_id"] == "1"
    assert payload["next_decisions"][0]["decision_at"].startswith(
        "2030-11-28 12:30"
    )


def test_cli_help_and_wrapper_entrypoint():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "paper_trader.py", "--help"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0
    assert "T-30m" in completed.stdout
    assert "tick" in completed.stdout
    assert "stats" in completed.stdout
