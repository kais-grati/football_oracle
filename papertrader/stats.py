"""Ledger reporting: per-wallet performance and source-separated monitoring."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from . import db


def _numbers(bets: pd.DataFrame) -> dict[str, Any]:
    settled = bets[bets["status"].isin(["won", "lost", "void"])].copy()
    pending = bets[bets["status"] == "pending"].copy()
    decided = settled[settled["status"].isin(["won", "lost"])]
    total_staked = float(decided["stake"].sum()) if not decided.empty else 0.0
    net_profit = float(settled["profit"].sum()) if not settled.empty else 0.0
    wins = int((settled["status"] == "won").sum())
    losses = int((settled["status"] == "lost").sum())
    voids = int((settled["status"] == "void").sum())
    hit_rate = wins / len(decided) * 100 if len(decided) else None
    avg_odds = float(decided["odds_taken"].mean()) if len(decided) else None
    yield_pct = net_profit / total_staked * 100 if total_staked else None
    clv_values = settled["clv"].dropna() if "clv" in settled else pd.Series(
        dtype=float
    )
    avg_clv = float(clv_values.mean() * 100) if len(clv_values) else None
    return {
        "settled_bets": len(settled),
        "won": wins,
        "lost": losses,
        "void": voids,
        "hit_rate_pct": hit_rate,
        "total_staked": total_staked,
        "net_profit": net_profit,
        "yield_pct": yield_pct,
        "avg_odds": avg_odds,
        "avg_clv_pct": avg_clv,
        "pending_bets": len(pending),
        "open_exposure": float(pending["stake"].sum()) if len(pending) else 0.0,
    }


def wallet_stats(
    conn: sqlite3.Connection, wallet: sqlite3.Row | dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    row = dict(wallet)
    bets = pd.read_sql_query(
        "SELECT * FROM bets WHERE wallet_id = ? ORDER BY id",
        conn,
        params=(row["id"],),
    )
    numbers = _numbers(bets)
    settled = bets[bets["status"].isin(["won", "lost", "void"])].copy()
    pending = bets[bets["status"] == "pending"].copy()

    starting = float(row["starting_capital"])
    bankroll = float(row["bankroll"])
    if settled.empty:
        max_drawdown = 0.0
    else:
        ordered = settled.sort_values(["settled_at", "id"])
        equity = starting + ordered["profit"].cumsum()
        path = pd.concat([pd.Series([starting]), equity], ignore_index=True)
        peak = path.cummax()
        max_drawdown = float(((path - peak) / peak).min() * 100)

    stats = {
        "wallet": row["name"],
        "starting_capital": starting,
        "bankroll": bankroll,
        "open_exposure": numbers["open_exposure"],
        "available": bankroll - numbers["open_exposure"],
        **numbers,
        "roi_pct": (
            numbers["net_profit"] / starting * 100 if starting else None
        ),
        "max_drawdown_pct": max_drawdown,
    }
    return stats, pending, settled


def all_stats(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = []
    for wallet in db.wallet_rows(conn):
        stats, _, _ = wallet_stats(conn, wallet)
        rows.append({
            "Wallet": stats["wallet"],
            "Starting": stats["starting_capital"],
            "Bankroll": stats["bankroll"],
            "Open": stats["open_exposure"],
            "Available": stats["available"],
            "Settled": stats["settled_bets"],
            "W/L/V": f"{stats['won']}/{stats['lost']}/{stats['void']}",
            "Hit %": stats["hit_rate_pct"],
            "Staked": stats["total_staked"],
            "Net": stats["net_profit"],
            "Yield %": stats["yield_pct"],
            "ROI %": stats["roi_pct"],
            "Max DD %": stats["max_drawdown_pct"],
            "Pending": stats["pending_bets"],
        })
    return pd.DataFrame(rows)


def source_stats(conn: sqlite3.Connection) -> pd.DataFrame:
    """Bet and decision metrics split by ``consensus`` vs fallback.

    Fallback performance must not be hidden inside the consensus aggregate;
    this table is the first check when fallback volume or yield drifts.
    Sources with decisions but no bets (all no-bet or skipped) are included so
    the operational mix is visible.
    """
    bets = pd.read_sql_query(
        """
        SELECT b.*, COALESCE(d.odds_source, 'none') AS odds_source
        FROM bets b
        LEFT JOIN decisions d ON d.id = b.decision_id
        """,
        conn,
    )
    by_source: dict[str, dict[str, Any]] = {}
    if not bets.empty:
        for source, group in bets.groupby("odds_source", dropna=False):
            numbers = _numbers(group)
            by_source[str(source)] = {
                "Source": str(source),
                "Bets": len(group),
                "Settled": numbers["settled_bets"],
                "Won": numbers["won"],
                "Lost": numbers["lost"],
                "Hit %": numbers["hit_rate_pct"],
                "Staked": numbers["total_staked"],
                "Net": numbers["net_profit"],
                "Yield %": numbers["yield_pct"],
                "Pending": numbers["pending_bets"],
                "Decisions": 0,
                "Placed decisions": 0,
                "No bet": 0,
                "Skipped": 0,
            }

    decisions = pd.read_sql_query(
        """
        SELECT odds_source, status, COUNT(*) AS decisions
        FROM decisions
        GROUP BY odds_source, status
        """,
        conn,
    )
    if not decisions.empty:
        for _, row in decisions.iterrows():
            source = str(row["odds_source"])
            entry = by_source.setdefault(source, {
                "Source": source, "Bets": 0, "Settled": 0, "Won": 0,
                "Lost": 0, "Hit %": None, "Staked": 0.0, "Net": 0.0,
                "Yield %": None, "Pending": 0, "Decisions": 0,
                "Placed decisions": 0, "No bet": 0, "Skipped": 0,
            })
            count = int(row["decisions"])
            entry["Decisions"] += count
            if row["status"] == "placed":
                entry["Placed decisions"] += count
            elif row["status"] == "no_bet":
                entry["No bet"] += count
            elif row["status"] == "skipped":
                entry["Skipped"] += count
    return pd.DataFrame(list(by_source.values()))


def pending_table(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = db.pending_bets(conn)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(row) for row in rows])


def status_report(
    conn: sqlite3.Connection,
    now: datetime | None = None,
    lead_minutes: int = 30,
) -> dict[str, Any]:
    """Operational snapshot for the CLI ``status`` command."""
    current = now or datetime.now(timezone.utc)
    next_decision = conn.execute(
        """
        SELECT f.event_id, f.home_team, f.away_team, f.kickoff_utc,
               f.season_start,
               datetime(f.kickoff_utc, ?) AS decision_at
        FROM fixtures f
        LEFT JOIN decisions d ON d.fixture_id = f.event_id
        WHERE d.id IS NULL AND f.status = 'notstarted'
          AND f.kickoff_utc > ?
        ORDER BY f.kickoff_utc
        LIMIT 10
        """,
        (f"-{int(lead_minutes)} minutes",
         current.isoformat(timespec="seconds")),
    ).fetchall()
    pending = db.pending_bets(conn)
    last_sync = db.get_meta(conn, "fixtures_last_sync")
    return {
        "now": current.isoformat(timespec="seconds"),
        "next_decisions": [dict(row) for row in next_decision],
        "pending_bets": len(pending),
        "open_exposure": sum(float(row["stake"]) for row in pending),
        "fixtures_last_sync": last_sync,
    }
