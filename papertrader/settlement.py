"""Settlement of pending wallet bets against provider results."""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Any

from . import db
from .domain import FixtureResult, parse_utc

logger = logging.getLogger(__name__)


class SettlementService:
    """Fetch results for pending bets and update bets + wallet bankrolls."""

    def __init__(self, conn: sqlite3.Connection, results_provider):
        self.conn = conn
        self.results_provider = results_provider

    def settle_pending(self, now: datetime | None = None) -> dict[str, Any]:
        pending = db.pending_bets(self.conn)
        if not pending:
            return {"settled": 0, "void": 0, "profit": 0.0, "by_wallet": {}}

        dates = []
        for bet in pending:
            kickoff = parse_utc(bet["kickoff"])
            if kickoff is not None:
                dates.append(kickoff.date())
        if not dates:
            return {"settled": 0, "void": 0, "profit": 0.0, "by_wallet": {}}

        # Include a day on each side: a local kickoff date and the provider's
        # UTC event date can differ by one day.
        start = min(dates) - timedelta(days=1)
        end = max(dates) + timedelta(days=1)
        results = self.results_provider.fetch_results(start, end)
        lookup = {result.event_id: result for result in results}

        profits: Counter[int] = Counter()
        names: dict[int, str] = {}
        settled = 0
        void = 0
        with self.conn:
            for bet in pending:
                result: FixtureResult | None = lookup.get(str(bet["match_id"]))
                if result is None:
                    continue
                names[int(bet["wallet_id"])] = str(bet["wallet"])
                if result.is_void:
                    outcome = "void"
                    profit = 0.0
                    winner = None
                    void += 1
                else:
                    winner = result.winner()
                    if winner is None:
                        continue
                    outcome = "won" if bet["selection"] == winner else "lost"
                    profit = (
                        float(bet["stake"]) * (float(bet["odds_taken"]) - 1.0)
                        if outcome == "won"
                        else -float(bet["stake"])
                    )
                db.settle_bet(
                    self.conn,
                    int(bet["id"]),
                    outcome,
                    winner,
                    profit,
                )
                profits[int(bet["wallet_id"])] += profit
                settled += 1
            for wallet_id, profit in profits.items():
                db.update_wallet_bankroll(
                    self.conn, wallet_id, float(profit)
                )

        summary = {
            "settled": settled,
            "void": void,
            "profit": float(sum(profits.values())),
            "by_wallet": {
                names.get(wallet_id, str(wallet_id)): float(profit)
                for wallet_id, profit in profits.items()
            },
        }
        logger.info(
            "settled %d bets (%d void), profit %+.2f",
            settled, void, summary["profit"],
        )
        return summary
