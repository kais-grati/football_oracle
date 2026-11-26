"""SQLite ledger for fixtures, decisions, snapshots, and wallet bets.

The schema is intentionally additive over the existing paper-trader database:
old single-wallet ledgers and the current multi-wallet ledgers both migrate in
place.  The two important uniqueness constraints are:

* one ``decisions`` row per fixture (idempotent processing), and
* one ``bets`` row per ``(wallet, match)`` (idempotent wallet writes).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .domain import Fixture, OddsSnapshot, iso_utc, parse_utc
from .sizing import WALLET_PROFILES

SCHEMA = """
CREATE TABLE IF NOT EXISTS wallets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    profile TEXT NOT NULL,
    starting_capital REAL NOT NULL,
    bankroll REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fixtures (
    event_id TEXT PRIMARY KEY,
    league TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    kickoff_utc TEXT NOT NULL,
    status TEXT NOT NULL,
    season_id TEXT,
    season_start TEXT,
    round_number INTEGER,
    home_score INTEGER,
    away_score INTEGER,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id TEXT NOT NULL UNIQUE REFERENCES fixtures(event_id),
    created_at TEXT NOT NULL,
    decision_at_utc TEXT NOT NULL,
    actual_lead_seconds INTEGER,
    odds_source TEXT NOT NULL,
    consensus_attempted_at TEXT,
    consensus_snapshot_at TEXT,
    fallback_snapshot_at TEXT,
    fallback_reason TEXT,
    snapshot_at TEXT,
    staleness_seconds INTEGER,
    model_version TEXT,
    selection TEXT,
    home_odds REAL,
    draw_odds REAL,
    away_odds REAL,
    execution_odds REAL,
    confidence REAL,
    p_logreg REAL,
    p_tree REAL,
    status TEXT NOT NULL,
    reason TEXT,
    raw_consensus TEXT,
    raw_bet365 TEXT
);

CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet_id INTEGER NOT NULL REFERENCES wallets(id),
    decision_id INTEGER REFERENCES decisions(id),
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
    discord_sent INTEGER NOT NULL DEFAULT 0,
    UNIQUE(wallet_id, match_id)
);

CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id TEXT NOT NULL,
    source TEXT NOT NULL,
    observed_at TEXT,
    fetched_at TEXT NOT NULL,
    home_odds REAL NOT NULL,
    draw_odds REAL NOT NULL,
    away_odds REAL NOT NULL,
    raw TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bets_wallet_status
ON bets(wallet_id, status);

CREATE INDEX IF NOT EXISTS idx_fixtures_kickoff
ON fixtures(kickoff_utc);

CREATE INDEX IF NOT EXISTS idx_bets_match
ON bets(match_id);

CREATE INDEX IF NOT EXISTS idx_snapshots_fixture
ON snapshots(fixture_id);
"""

BET_COLUMNS = [
    "created_at", "match_id", "league", "kickoff", "home_team", "away_team",
    "selection", "odds_taken", "bookmaker", "confidence", "p_logreg",
    "p_tree", "stake", "status", "closing_odds", "clv", "result", "profit",
    "settled_at", "model_version", "discord_sent",
]

DECISION_FIELDS = [
    "fixture_id", "created_at", "decision_at_utc", "actual_lead_seconds",
    "odds_source", "consensus_attempted_at", "consensus_snapshot_at",
    "fallback_snapshot_at", "fallback_reason", "snapshot_at",
    "staleness_seconds", "model_version", "selection", "home_odds",
    "draw_odds", "away_odds", "execution_odds", "confidence", "p_logreg",
    "p_tree", "status", "reason", "raw_consensus", "raw_bet365",
]

BET_INSERT_FIELDS = [
    "wallet_id", "decision_id", "created_at", "match_id", "league",
    "kickoff", "home_team", "away_team", "selection", "odds_taken",
    "bookmaker", "confidence", "p_logreg", "p_tree", "stake", "status",
    "model_version", "discord_sent",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def _migrate_single_wallet(
    conn: sqlite3.Connection, starting_capital: float
) -> None:
    """Upgrade a pre-wallet ledger, if one is present."""
    tables = _tables(conn)
    if "bets" not in tables or "wallet_id" in _table_columns(conn, "bets"):
        return

    old_bets = [dict(row) for row in conn.execute("SELECT * FROM bets")]
    old_state = None
    if "state" in tables:
        old_state = conn.execute("SELECT * FROM state WHERE id = 1").fetchone()

    starting = (
        float(old_state["starting_capital"]) if old_state else starting_capital
    )
    bankroll = float(old_state["bankroll"]) if old_state else starting
    created = old_state["created_at"] if old_state else now_iso()

    conn.execute("ALTER TABLE bets RENAME TO bets_single_wallet")
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO wallets "
        "(name, profile, starting_capital, bankroll, created_at) "
        "VALUES ('Conservative', 'Conservative', ?, ?, ?)",
        (starting, bankroll, created),
    )
    wallet_id = conn.execute(
        "SELECT id FROM wallets WHERE name = 'Conservative'"
    ).fetchone()["id"]
    for bet in old_bets:
        values = [bet.get(column) for column in BET_COLUMNS]
        conn.execute(
            f"""
            INSERT OR IGNORE INTO bets (wallet_id, {', '.join(BET_COLUMNS)})
            VALUES ({', '.join(['?'] * (len(BET_COLUMNS) + 1))})
            """,
            (wallet_id, *values),
        )
    conn.execute("DROP TABLE bets_single_wallet")
    if old_state is not None:
        conn.execute("DROP TABLE state")
    conn.commit()


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a ledger was first created."""
    bet_columns = _table_columns(conn, "bets")
    if "decision_id" not in bet_columns:
        conn.execute("ALTER TABLE bets ADD COLUMN decision_id INTEGER")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_bets_decision ON bets(decision_id)"
    )


def _ensure_wallets(conn: sqlite3.Connection, starting_capital: float) -> None:
    for name in WALLET_PROFILES:
        conn.execute(
            "INSERT OR IGNORE INTO wallets "
            "(name, profile, starting_capital, bankroll, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, name, float(starting_capital), float(starting_capital),
             now_iso()),
        )


def connect(
    db_path: str | Path, starting_capital: float = 100.0
) -> sqlite3.Connection:
    """Open the ledger, migrate it, and ensure all six wallets exist."""
    path_text = str(db_path)
    if path_text != ":memory:":
        Path(path_text).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path_text)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if path_text != ":memory:":
        try:
            conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.DatabaseError:
            pass
    _migrate_single_wallet(conn, starting_capital)
    conn.executescript(SCHEMA)
    _add_missing_columns(conn)
    _ensure_wallets(conn, starting_capital)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def fixture_from_row(row: sqlite3.Row | dict[str, Any]) -> Fixture:
    kickoff = parse_utc(row["kickoff_utc"])
    if kickoff is None:
        raise ValueError(f"fixture {row['event_id']} has invalid kickoff")
    season_start = row["season_start"]
    if season_start:
        season_start = date.fromisoformat(str(season_start))
    return Fixture(
        event_id=str(row["event_id"]),
        league=str(row["league"]),
        home_team=str(row["home_team"]),
        away_team=str(row["away_team"]),
        kickoff_utc=kickoff,
        status=str(row["status"]),
        season_id=str(row["season_id"]) if row["season_id"] else None,
        season_start=season_start,
        round_number=row["round_number"],
        home_score=row["home_score"],
        away_score=row["away_score"],
    )


def upsert_fixtures(
    conn: sqlite3.Connection,
    fixtures: Iterable[Fixture],
    updated_at: datetime | None = None,
) -> int:
    """Insert or refresh fixtures, keyed by provider event ID."""
    stamp = iso_utc(updated_at or datetime.now(timezone.utc))
    count = 0
    with conn:
        for fixture in fixtures:
            conn.execute(
                """
                INSERT INTO fixtures (
                    event_id, league, home_team, away_team, kickoff_utc,
                    status, season_id, season_start, round_number,
                    home_score, away_score, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    league = excluded.league,
                    home_team = excluded.home_team,
                    away_team = excluded.away_team,
                    kickoff_utc = excluded.kickoff_utc,
                    status = excluded.status,
                    season_id = excluded.season_id,
                    season_start = excluded.season_start,
                    round_number = excluded.round_number,
                    home_score = COALESCE(excluded.home_score,
                                          fixtures.home_score),
                    away_score = COALESCE(excluded.away_score,
                                          fixtures.away_score),
                    updated_at = excluded.updated_at
                """,
                (
                    fixture.event_id,
                    fixture.league,
                    fixture.home_team,
                    fixture.away_team,
                    iso_utc(fixture.kickoff_utc),
                    fixture.status,
                    fixture.season_id,
                    fixture.season_start.isoformat()
                    if fixture.season_start else None,
                    fixture.round_number,
                    fixture.home_score,
                    fixture.away_score,
                    stamp,
                ),
            )
            count += 1
    return count


def get_fixture(
    conn: sqlite3.Connection, event_id: str
) -> Fixture | None:
    row = conn.execute(
        "SELECT * FROM fixtures WHERE event_id = ?", (str(event_id),)
    ).fetchone()
    return fixture_from_row(row) if row else None


def all_fixtures(conn: sqlite3.Connection) -> list[Fixture]:
    return [
        fixture_from_row(row)
        for row in conn.execute("SELECT * FROM fixtures ORDER BY kickoff_utc")
    ]


def due_fixtures(
    conn: sqlite3.Connection,
    now: datetime,
    lead_minutes: int,
    safety_minutes: int,
) -> list[Fixture]:
    """Fixtures whose T-lead passed but whose cutoff has not."""
    rows = conn.execute(
        """
        SELECT f.*
        FROM fixtures f
        LEFT JOIN decisions d ON d.fixture_id = f.event_id
        WHERE d.id IS NULL AND f.status = 'notstarted'
        ORDER BY f.kickoff_utc, f.event_id
        """
    ).fetchall()
    safety = timedelta(minutes=safety_minutes)
    due: list[Fixture] = []
    for row in rows:
        fixture = fixture_from_row(row)
        if fixture.decision_at(lead_minutes) <= now:
            if fixture.kickoff_utc > now + safety:
                due.append(fixture)
    due.sort(key=lambda f: (f.decision_at(lead_minutes), f.kickoff_utc,
                            f.event_id))
    return due


def past_cutoff_fixtures(
    conn: sqlite3.Connection,
    now: datetime,
    lead_minutes: int,
    safety_minutes: int,
) -> list[Fixture]:
    """Undecided fixtures that can no longer be bet safely."""
    rows = conn.execute(
        """
        SELECT f.*
        FROM fixtures f
        LEFT JOIN decisions d ON d.fixture_id = f.event_id
        WHERE d.id IS NULL AND f.status = 'notstarted'
        ORDER BY f.kickoff_utc, f.event_id
        """
    ).fetchall()
    safety = timedelta(minutes=safety_minutes)
    missed: list[Fixture] = []
    for row in rows:
        fixture = fixture_from_row(row)
        if (
            fixture.decision_at(lead_minutes) <= now
            and fixture.kickoff_utc <= now + safety
        ):
            missed.append(fixture)
    return missed


def mark_missed(
    conn: sqlite3.Connection,
    now: datetime,
    lead_minutes: int,
    safety_minutes: int,
    reason: str = "missed_window",
) -> int:
    """Record a skipped decision for fixtures whose window has closed."""
    missed = past_cutoff_fixtures(conn, now, lead_minutes, safety_minutes)
    count = 0
    with conn:
        for fixture in missed:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO decisions (
                    fixture_id, created_at, decision_at_utc,
                    actual_lead_seconds, odds_source, status, reason
                ) VALUES (?, ?, ?, ?, 'none', 'skipped', ?)
                """,
                (
                    fixture.event_id,
                    iso_utc(now),
                    iso_utc(fixture.decision_at(lead_minutes)),
                    int((fixture.kickoff_utc - now).total_seconds()),
                    reason,
                ),
            )
            count += cursor.rowcount
    return count


# ---------------------------------------------------------------------------
# Decisions and bets
# ---------------------------------------------------------------------------

def insert_decision_and_bets(
    conn: sqlite3.Connection,
    decision: dict[str, Any],
    bets: Iterable[dict[str, Any]],
) -> int | None:
    """Atomically insert one decision and its wallet bets.

    Returns ``None`` if the fixture was already decided (a concurrent or
    previous process won the race); otherwise returns the decision id.
    """
    values = []
    for field in DECISION_FIELDS:
        value = decision.get(field)
        if field in {"raw_consensus", "raw_bet365"} and not isinstance(
            value, (str, type(None))
        ):
            value = json.dumps(value, default=str)
        values.append(value)

    with conn:
        cursor = conn.execute(
            f"""
            INSERT OR IGNORE INTO decisions ({', '.join(DECISION_FIELDS)})
            VALUES ({', '.join(['?'] * len(DECISION_FIELDS))})
            """,
            values,
        )
        if cursor.rowcount == 0:
            return None
        decision_id = int(cursor.lastrowid)
        for bet in bets:
            payload = dict(bet)
            payload["decision_id"] = decision_id
            payload.setdefault("status", "pending")
            payload.setdefault("created_at", now_iso())
            payload.setdefault("discord_sent", 0)
            columns = [field for field in BET_INSERT_FIELDS if field in payload]
            conn.execute(
                f"""
                INSERT OR IGNORE INTO bets ({', '.join(columns)})
                VALUES ({', '.join(['?'] * len(columns))})
                """,
                [payload[field] for field in columns],
            )
    return decision_id


def insert_snapshot(
    conn: sqlite3.Connection, fixture_id: str, snapshot: OddsSnapshot
) -> None:
    """Archive the exact snapshot used or attempted for a decision."""
    raw = snapshot.raw
    if not isinstance(raw, (str, type(None))):
        raw = json.dumps(raw, default=str)
    with conn:
        conn.execute(
            """
            INSERT INTO snapshots (
                fixture_id, source, observed_at, fetched_at,
                home_odds, draw_odds, away_odds, raw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(fixture_id),
                snapshot.source,
                iso_utc(snapshot.observed_at),
                iso_utc(snapshot.fetched_at or datetime.now(timezone.utc)),
                snapshot.home,
                snapshot.draw,
                snapshot.away,
                raw,
            ),
        )


def get_decision(
    conn: sqlite3.Connection, fixture_id: str
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM decisions WHERE fixture_id = ?", (str(fixture_id),)
    ).fetchone()


# ---------------------------------------------------------------------------
# Wallets and settlement
# ---------------------------------------------------------------------------

def wallet_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """All wallets with pending exposure and available bankroll."""
    rows = conn.execute(
        """
        SELECT w.*,
               COALESCE(SUM(CASE WHEN b.status = 'pending'
                                 THEN b.stake ELSE 0 END), 0)
                   AS pending_exposure
        FROM wallets w
        LEFT JOIN bets b ON b.wallet_id = w.id
        GROUP BY w.id
        ORDER BY w.id
        """
    ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["pending_exposure"] = float(item["pending_exposure"] or 0.0)
        item["available"] = float(item["bankroll"]) - item["pending_exposure"]
        result.append(item)
    return result


def get_wallet(
    conn: sqlite3.Connection, name: str
) -> dict[str, Any] | None:
    for wallet in wallet_rows(conn):
        if wallet["name"] == name:
            return wallet
    return None


def pending_bets(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT b.*, w.name AS wallet
        FROM bets b
        JOIN wallets w ON w.id = b.wallet_id
        WHERE b.status = 'pending'
        ORDER BY b.kickoff, b.id
        """
    ).fetchall()


def settle_bet(
    conn: sqlite3.Connection,
    bet_id: int,
    status: str,
    result: str | None,
    profit: float,
    closing_odds: float | None = None,
    clv: float | None = None,
) -> None:
    conn.execute(
        """
        UPDATE bets
        SET status = ?, result = ?, profit = ?, settled_at = ?,
            closing_odds = ?, clv = ?
        WHERE id = ? AND status = 'pending'
        """,
        (
            status,
            result,
            float(profit),
            now_iso(),
            closing_odds,
            clv,
            int(bet_id),
        ),
    )


def update_wallet_bankroll(
    conn: sqlite3.Connection, wallet_id: int, profit: float
) -> None:
    conn.execute(
        "UPDATE wallets SET bankroll = bankroll + ? WHERE id = ?",
        (float(profit), int(wallet_id)),
    )


def update_discord_sent(
    conn: sqlite3.Connection, bet_ids: Iterable[int], sent: bool
) -> None:
    with conn:
        for bet_id in bet_ids:
            conn.execute(
                "UPDATE bets SET discord_sent = ? WHERE id = ?",
                (int(sent), int(bet_id)),
            )


# ---------------------------------------------------------------------------
# Small metadata store
# ---------------------------------------------------------------------------

def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM meta WHERE key = ?", (str(key),)
    ).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(key), str(value)),
        )
